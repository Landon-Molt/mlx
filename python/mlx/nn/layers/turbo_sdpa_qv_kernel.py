"""
Quantized-V SDPA vector kernel for mx.fast.metal_kernel().

This is a variant of MLX's sdpa_vector that reads V from 4-bit quantized
storage (matching mx.quantize with group_size=32, bits=4) instead of FP16.
K remains FP16 for scoring. Supports GQA via gqa_factor.

Dequant format (mx.quantize compatible):
  - qv_data:   uint32[], 8 x 4-bit values packed per word
  - qv_scales: float[],  1 scale per group of 32 elements
  - qv_biases: float[],  1 bias  per group of 32 elements
  - value = ((qv_data[i/8] >> ((i%8)*4)) & 0xF) * scale + bias

Usage:
    kernel = get_sdpa_vector_qv_kernel()
    outputs = kernel(
        inputs=[queries, keys, qv_data, qv_scales, qv_biases, params],
        output_shapes=[(B_nq, D)],
        output_dtypes=[mx.float16],
        grid=(B_nq, n_q_seq, 1),
        threadgroup=(32, 32, 1),  # BN=32 simdgroups x BD=32 threads/simdgroup
        stream=mx.gpu,
    )
"""

from typing import Dict

import mlx.core as mx

# ---------------------------------------------------------------------------
# Metal kernel source — sdpa_vector with quantized V (4-bit, group_size=32)
# ---------------------------------------------------------------------------
# This follows the exact same online softmax algorithm as sdpa_vector.h but
# replaces the FP16 V read with an inline 4-bit dequantization.
#
# Template constants (set via params buffer):
#   D        — head dimension (e.g. 128)
#   N        — number of KV sequence positions
#   gqa_factor — number of Q heads per KV head
#   scale    — attention scale (typically 1/sqrt(D))
#
# Threading model (matches sdpa_vector):
#   threadgroup = (32, 32, 1)  →  BN=32 simdgroups, BD=32 threads per simdgroup
#   grid.x = B * n_q_heads     (one threadgroup per batch*q_head)
#   grid.y = n_q_seq            (one threadgroup per query position)
#
# Buffer layout:
#   queries:   [B*n_q_heads, n_q_seq, D] float16
#   keys:      [B*n_kv_heads, N, D] float16
#   qv_data:   [B*n_kv_heads, N, D/8] uint32  (packed 4-bit)
#   qv_scales: [B*n_kv_heads, N, n_groups] float32  (n_groups = D/32)
#   qv_biases: [B*n_kv_heads, N, n_groups] float32
#   params:    [8] uint32 — {D, N, gqa_factor, scale_as_uint, k_head_stride,
#                            k_seq_stride, v_head_stride, v_seq_stride}
# ---------------------------------------------------------------------------

_SDPA_VECTOR_QV_HEADER = """
// Limits helper for finite_min
template <typename T> struct QVLimits;
template <> struct QVLimits<float> {
    static constexpr constant float finite_min = -1e38f;
};
"""

_SDPA_VECTOR_QV_SOURCE = """
    // ---------------------------------------------------------------
    // sdpa_vector_qv — Quantized-V variant of sdpa_vector
    //
    // K is read as float16 for scoring.
    // V is dequantized inline from 4-bit packed uint32 storage.
    // Uses online softmax (same numerics as sdpa_vector.h).
    // ---------------------------------------------------------------

    // Compile-time constants
    // D is passed via params[0], but we treat it as runtime to keep
    // the kernel generic. For perf-critical paths you'd template on D.
    const uint D         = params[0];
    const int  N_kv      = int(params[1]);
    const int  gqa       = int(params[2]);
    // scale is passed as uint32 bit-pattern of float
    const float attn_scale = as_type<float>(params[3]);
    const int  k_head_stride = int(params[4]);
    const int  k_seq_stride  = int(params[5]);
    // V strides — in units of uint32 for data, float for scales/biases
    const int  v_head_stride_data = int(params[6]);  // per-head stride in uint32 words
    const int  v_seq_stride_data  = int(params[7]);  // per-seq stride in uint32 words
    // Derived
    const uint packed_dim = D / 8;       // uint32 words per V row
    const uint n_groups   = D / 32;      // number of quantization groups per row
    const int  v_head_stride_group = int(N_kv * n_groups);  // scale/bias head stride
    const int  v_seq_stride_group  = int(n_groups);         // scale/bias seq stride

    // Threading layout — matches sdpa_vector exactly
    // BN = 32 simdgroups process 32 KV positions in parallel
    // BD = 32 threads per simdgroup split the D dimension
    const int BN = 32;
    const int BD = 32;
    const int qk_per_thread = int(D) / BD;  // elements of Q/K per thread
    const int v_per_thread  = int(D) / BD;  // elements of V per thread

    // Thread / simdgroup indices
    uint simd_gid = thread_position_in_threadgroup.y;  // which simdgroup [0..BN)
    uint simd_lid = thread_position_in_threadgroup.x;  // lane within simdgroup [0..BD)

    // Batch/head indices
    int q_batch_head_idx = threadgroup_position_in_grid.x;
    int q_seq_idx        = threadgroup_position_in_grid.y;
    int kv_head_idx      = q_batch_head_idx / gqa;
    int n_q_seq          = threadgroups_per_grid.y;

    // Pointer offsets into Q (FP16)
    int o_offset = q_batch_head_idx * n_q_seq + q_seq_idx;
    int q_base   = o_offset * int(D) + int(simd_lid) * qk_per_thread;

    // Pointer offsets into K (FP16)
    int k_base = kv_head_idx * k_head_stride
               + int(simd_gid) * k_seq_stride
               + int(simd_lid) * qk_per_thread;

    // Pointer offsets into V data (uint32 packed)
    // Each thread reads v_per_thread consecutive elements from the V dimension.
    // Thread simd_lid handles elements [simd_lid * v_per_thread .. (simd_lid+1) * v_per_thread).
    int v_data_base = kv_head_idx * v_head_stride_data
                    + int(simd_gid) * v_seq_stride_data;
    // For scales/biases: offset to the correct head+initial seq position
    int v_group_base = kv_head_idx * v_head_stride_group
                     + int(simd_gid) * v_seq_stride_group;

    // Output pointer
    int out_base = o_offset * int(D) + int(simd_gid) * v_per_thread;

    // Load query into registers (pre-scaled)
    float q_reg[4];   // max qk_per_thread = D/32; for D=128 this is 4
    for (int j = 0; j < qk_per_thread; j++) {
        q_reg[j] = attn_scale * float(queries[q_base + j]);
    }

    // Zero the output accumulator
    float o_reg[4];   // max v_per_thread = D/32 = 4 for D=128
    for (int j = 0; j < v_per_thread; j++) {
        o_reg[j] = 0.0f;
    }

    // Shared memory for the reduction phase
    threadgroup float tg_outputs[32 * 32];   // BN * BD
    threadgroup float tg_max_scores[32];     // BN
    threadgroup float tg_sum_exp[32];        // BN

    float max_score = -1e38f;
    float sum_exp_score = 0.0f;

    // Inner loop strides (advance by BN sequence positions per iteration)
    int inner_k_stride = BN * k_seq_stride;
    int inner_v_data_stride = BN * v_seq_stride_data;
    int inner_v_group_stride = BN * v_seq_stride_group;

    // Main loop over KV sequence positions
    int cur_k = k_base;
    int cur_v_data  = v_data_base;
    int cur_v_group = v_group_base;

    for (int i = int(simd_gid); i < N_kv; i += BN) {

        // --- Score: dot(q, k) via simd_sum ---
        float score = 0.0f;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q_reg[j] * float(keys[cur_k + j]);
        }
        score = simd_sum(score);

        // --- Online softmax update ---
        float new_max   = max(max_score, score);
        float factor    = exp(max_score - new_max);
        float exp_score = exp(score - new_max);
        max_score     = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        // --- Dequantize V and accumulate ---
        // This thread handles V elements [simd_lid * v_per_thread .. +v_per_thread)
        // for the current sequence position.
        //
        // mx.quantize 4-bit layout:
        //   word index   = elem / 8
        //   bit position = (elem % 8) * 4
        //   raw 4-bit    = (word >> bit_pos) & 0xF
        //   dequant      = raw * scale[group] + bias[group]
        //   group index  = elem / 32
        for (int j = 0; j < v_per_thread; j++) {
            int elem = int(simd_lid) * v_per_thread + j;
            int word_idx = elem / 8;
            int bit_pos  = (elem % 8) * 4;
            uint raw = (qv_data[cur_v_data + word_idx] >> bit_pos) & 0xFu;

            int grp = elem / 32;
            float scale = qv_scales[cur_v_group + grp];
            float bias  = qv_biases[cur_v_group + grp];

            float v_val = float(raw) * scale + bias;

            o_reg[j] = o_reg[j] * factor + exp_score * v_val;
        }

        // Advance pointers to next KV position block
        cur_k       += inner_k_stride;
        cur_v_data  += inner_v_data_stride;
        cur_v_group += inner_v_group_stride;
    }

    // ---------------------------------------------------------------
    // Reduction across simdgroups (same as sdpa_vector.h)
    // ---------------------------------------------------------------

    // Communicate max and sum_exp across simdgroups
    if (simd_lid == 0) {
        tg_max_scores[simd_gid] = max_score;
        tg_sum_exp[simd_gid]    = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each lane reads one simdgroup's max/sum
    float sg_max = tg_max_scores[simd_lid];
    float new_global_max = simd_max(sg_max);
    float sg_factor = exp(sg_max - new_global_max);
    float total_sum_exp = simd_sum(tg_sum_exp[simd_lid] * sg_factor);

    // Aggregate partial outputs across simdgroups via transpose trick
    // factor for THIS simdgroup's contribution
    float my_factor = exp(max_score - new_global_max);
    for (int j = 0; j < v_per_thread; j++) {
        // Write this simdgroup's partial into shared memory (transposed)
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[j] * my_factor;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Read back in transposed order and sum across lanes
        float val = tg_outputs[simd_gid * BD + simd_lid];
        // Reuse sg_factor per-lane: each lane holds a different simdgroup's factor
        // But we already pre-multiplied by my_factor above, so just simd_sum
        o_reg[j] = simd_sum(val);
        o_reg[j] = (total_sum_exp == 0.0f) ? o_reg[j] : (o_reg[j] / total_sum_exp);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output — only lane 0 of each simdgroup writes
    if (simd_lid == 0) {
        for (int j = 0; j < v_per_thread; j++) {
            output[out_base + j] = half(o_reg[j]);
        }
    }
"""

# ---------------------------------------------------------------------------
# Kernel cache and factory
# ---------------------------------------------------------------------------
_sdpa_qv_kernel_cache: Dict[str, object] = {}


def get_sdpa_vector_qv_kernel():
    """Get or compile the sdpa_vector_qv Metal kernel.

    Returns a callable kernel object. Call it as:

        kernel(
            inputs=[queries, keys, qv_data, qv_scales, qv_biases, params],
            output_shapes=[(B_nq * n_q_seq, D)],
            output_dtypes=[mx.float16],
            grid=(B_nq, n_q_seq, 1),
            threadgroup=(32, 32, 1),
            stream=mx.gpu,
        )

    Where:
        queries:   [B*n_q_heads, n_q_seq, D] float16
        keys:      [B*n_kv_heads, N, D] float16
        qv_data:   [B*n_kv_heads, N, D/8] uint32
        qv_scales: [B*n_kv_heads, N, D/32] float32
        qv_biases: [B*n_kv_heads, N, D/32] float32
        params:    [8] uint32 — see source comments for layout

    Build params with:
        import struct
        scale_uint = struct.unpack('I', struct.pack('f', scale))[0]
        params = mx.array([D, N, gqa_factor, scale_uint,
                           k_head_stride, k_seq_stride,
                           v_head_stride_data, v_seq_stride_data],
                          dtype=mx.uint32)
    """
    cache_key = "sdpa_vector_qv_4bit"
    if cache_key in _sdpa_qv_kernel_cache:
        return _sdpa_qv_kernel_cache[cache_key]

    kernel = mx.fast.metal_kernel(
        name="sdpa_vector_qv_4bit",
        input_names=[
            "queries",    # [B*nq, n_q_seq, D] float16
            "keys",       # [B*nkv, N, D] float16
            "qv_data",    # [B*nkv, N, D/8] uint32 — packed 4-bit V
            "qv_scales",  # [B*nkv, N, D/32] float32 — per-group scale
            "qv_biases",  # [B*nkv, N, D/32] float32 — per-group bias
            "params",     # [8] uint32
        ],
        output_names=[
            "output",     # [B*nq * n_q_seq, D] float16
        ],
        header=_SDPA_VECTOR_QV_HEADER,
        source=_SDPA_VECTOR_QV_SOURCE,
        ensure_row_contiguous=True,
        atomic_outputs=False,
    )

    _sdpa_qv_kernel_cache[cache_key] = kernel
    return kernel
