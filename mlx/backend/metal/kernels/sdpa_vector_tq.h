// Copyright © 2024-2026 Apple Inc.
// TurboQuant variant of sdpa_vector.
//
// Reads keys from TurboQuant ProdCodec format (MSE codebook + QJL signs)
// and values from TurboQuant MSECodec format (codebook + norms).
// Queries are pre-transformed (q_rot, q_proj) by the host.
//
// Same online softmax algorithm and threading model as sdpa_vector.h.

#include <metal_simdgroup>

using namespace metal;

constant bool has_mask [[function_constant(20)]];
constant bool do_causal [[function_constant(22)]];
constant bool bool_mask [[function_constant(23)]];
constant bool float_mask [[function_constant(24)]];

// TQ decode kernel: one query position per threadgroup.
// Grid: (B*n_q_heads, q_seq_len, 1)
// Threadgroup: (32, 32, 1) — BN=32 simdgroups × BD=32 lanes
//
// Template params:
//   T       — output dtype (float16)
//   D       — head dimension (key and value)
//   K_BITS  — key MSE quantization bits (e.g. 3 for 4-bit ProdCodec)
//   V_BITS  — value quantization bits (e.g. 4)
template <typename T, int D, int K_BITS, int V_BITS>
[[kernel]] void sdpa_vector_tq(
    // Queries (pre-transformed by ProdCodec)
    const device T* q_rot [[buffer(0)]],
    const device T* q_proj [[buffer(1)]],
    // Key state (ProdCodec: MSE + QJL)
    const device T* key_norms [[buffer(2)]],
    const device uint32_t* key_mse_indices [[buffer(3)]],
    const device T* key_res_norms [[buffer(4)]],
    const device uint32_t* key_signs [[buffer(5)]],
    // Value state (MSECodec)
    const device T* val_norms [[buffer(6)]],
    const device uint32_t* val_indices [[buffer(7)]],
    // Codebooks
    const device float* key_codebook [[buffer(8)]],
    const device float* key_scale [[buffer(9)]],
    const device float* val_codebook [[buffer(10)]],
    // Output
    device T* out [[buffer(11)]],
    // Params
    const constant int& gqa_factor [[buffer(12)]],
    const constant int& N [[buffer(13)]],
    const constant size_t& k_head_stride [[buffer(14)]],
    const constant size_t& v_head_stride [[buffer(15)]],
    // Mask
    const device bool* bmask [[buffer(16), function_constant(bool_mask)]],
    const device T* fmask [[buffer(17), function_constant(float_mask)]],
    const constant int& mask_kv_seq_stride
        [[buffer(18), function_constant(has_mask)]],
    const constant int& mask_q_seq_stride
        [[buffer(19), function_constant(has_mask)]],
    const constant int& mask_head_stride
        [[buffer(20), function_constant(has_mask)]],
    // Thread indices
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int elem_per_thread = D / BD;

  // Packed widths
  constexpr uint K_MSE_MASK = (1u << K_BITS) - 1u;
  constexpr uint V_MASK = (1u << V_BITS) - 1u;
  constexpr int k_packed_width = (D * K_BITS + 31) / 32;
  constexpr int k_sign_width = (D + 31) / 32;
  constexpr int v_packed_width = (D * V_BITS + 31) / 32;

  typedef float U;

  // Per-thread registers
  thread U qr[elem_per_thread];
  thread U qp[elem_per_thread];
  thread U o[elem_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Batch/head indices
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;

  // Query pointers (pre-transformed: q_rot and q_proj)
  const int q_base = o_offset * D + simd_lid * elem_per_thread;

  // Key/Value head offsets
  const int kv_base = kv_head_idx * int(k_head_stride);
  const int vv_base = kv_head_idx * int(v_head_stride);

  // Load QJL scale
  const U kscale = key_scale[0];

  // Load query into registers
  for (int i = 0; i < elem_per_thread; i++) {
    qr[i] = static_cast<U>(q_rot[q_base + i]);
    qp[i] = static_cast<U>(q_proj[q_base + i]);
  }
  for (int i = 0; i < elem_per_thread; i++) {
    o[i] = 0;
  }

  // Precompute value bit offsets for this thread's dims
  int v_words[elem_per_thread], v_offs[elem_per_thread];
  bool v_spills[elem_per_thread];
  for (int i = 0; i < elem_per_thread; i++) {
    int d = simd_lid * elem_per_thread + i;
    int bo = d * V_BITS;
    v_words[i] = bo / 32;
    v_offs[i] = bo % 32;
    v_spills[i] = (bo % 32 + V_BITS) > 32;
  }

  // Mask setup
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * D + simd_gid * elem_per_thread;

  U max_score = -1e38f;
  U sum_exp_score = 0;

  // Main loop: each simdgroup processes one KV token, stride by BN
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= -1e37f);
    }

    if (use_key) {
      // --- Score: ProdCodec key scoring ---
      // Pointers for this token
      const device uint32_t* mse_ptr = key_mse_indices + kv_base * k_packed_width + i * k_packed_width;
      const device uint32_t* sign_ptr = key_signs + kv_base * k_sign_width + i * k_sign_width;
      U kn = static_cast<U>(key_norms[kv_base + i]);
      U ksr = kn * kscale * static_cast<U>(key_res_norms[kv_base + i]);

      U score = 0;
      for (int j = 0; j < elem_per_thread; j++) {
        int d = simd_lid * elem_per_thread + j;
        // Unpack MSE key index
        int bo = d * K_BITS;
        uint idx = (mse_ptr[bo >> 5] >> (bo & 31));
        if ((bo & 31) + K_BITS > 32) {
          idx |= mse_ptr[(bo >> 5) + 1] << (K_BITS - ((bo & 31) + K_BITS - 32));
        }
        idx &= K_MSE_MASK;
        U code = key_codebook[idx];

        // QJL sign bit
        uint sb = (sign_ptr[d >> 5] >> (d & 31)) & 1u;
        score += kn * qr[j] * code + ksr * (sb ? qp[j] : -qp[j]);
      }
      score = simd_sum(score);

      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // --- Online softmax update ---
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // --- Value accumulation: MSECodec ---
      const device uint32_t* vt = val_indices + vv_base * v_packed_width + i * v_packed_width;
      U vnorm = static_cast<U>(val_norms[vv_base + i]);

      for (int j = 0; j < elem_per_thread; j++) {
        uint vv = (vt[v_words[j]] >> v_offs[j]);
        if (v_spills[j]) {
          vv |= vt[v_words[j] + 1] << (V_BITS - (v_offs[j] + V_BITS - 32));
        }
        vv &= V_MASK;
        U v_code = val_codebook[vv] * vnorm;
        o[j] = o[j] * factor + exp_score * v_code;
      }
    }

    // Advance mask pointers
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // --- Cross-simdgroup reduction (same as sdpa_vector.h) ---

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < elem_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write output — only lane 0 of each simdgroup writes
  if (simd_lid == 0) {
    for (int i = 0; i < elem_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
