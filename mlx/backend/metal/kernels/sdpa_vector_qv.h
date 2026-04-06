// Copyright © 2025 Apple Inc. / TurboQuant+ (Tom Turney)
//
// Quantized-V variant of sdpa_vector. K stays FP16 for scoring.
// V is dequantized inline from 4-bit packed uint32 storage matching
// mx.quantize(group_size=32, bits=4) format.
//
// Dequant: value[i] = ((word[i/8] >> ((i%8)*4)) & 0xF) * scale[i/32] + bias[i/32]

#include <metal_simdgroup>

using namespace metal;

// Reuse function constants from sdpa_vector.h
// constant bool has_mask [[function_constant(20)]];
// constant bool query_transposed [[function_constant(21)]];
// constant bool do_causal [[function_constant(22)]];
// constant bool bool_mask [[function_constant(23)]];
// constant bool float_mask [[function_constant(24)]];
// constant bool has_sinks [[function_constant(25)]];

template <typename T, int D>
[[kernel]] void sdpa_vector_qv(
    const device T* queries [[buffer(0)]],
    const device T* keys [[buffer(1)]],
    // Quantized V: packed 4-bit data + per-group scales + biases
    const device uint* qv_data [[buffer(2)]],
    const device float* qv_scales [[buffer(3)]],
    const device float* qv_biases [[buffer(4)]],
    device T* out [[buffer(5)]],
    const constant int& gqa_factor [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant size_t& k_head_stride [[buffer(8)]],
    const constant size_t& k_seq_stride [[buffer(9)]],
    // V strides in units of their respective types
    const constant size_t& qv_data_head_stride [[buffer(10)]],   // per-head in uint words
    const constant size_t& qv_data_seq_stride [[buffer(11)]],    // per-seq in uint words
    const constant size_t& qv_group_head_stride [[buffer(12)]],  // per-head in floats
    const constant size_t& qv_group_seq_stride [[buffer(13)]],   // per-seq in floats
    const constant float& scale [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {

  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  constexpr int packed_dim = D / 8;      // uint32 words per V row
  constexpr int n_groups = D / 32;       // quantization groups per row

  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_data_stride = BN * int(qv_data_seq_stride);
  int inner_v_group_stride = BN * int(qv_group_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;

  // K pointer (FP16)
  const device T* k_ptr = keys
      + kv_head_idx * k_head_stride
      + simd_gid * k_seq_stride
      + simd_lid * qk_per_thread;

  // Q pointer (FP16)
  const device T* q_ptr = queries + q_offset * D + simd_lid * qk_per_thread;

  // V quantized data pointer (uint32 packed)
  const device uint* vd_ptr = qv_data
      + kv_head_idx * qv_data_head_stride
      + simd_gid * qv_data_seq_stride;

  // V scales/biases pointer (float)
  const device float* vs_ptr = qv_scales
      + kv_head_idx * qv_group_head_stride
      + simd_gid * qv_group_seq_stride;
  const device float* vb_ptr = qv_biases
      + kv_head_idx * qv_group_head_stride
      + simd_gid * qv_group_seq_stride;

  out += o_offset * D + simd_gid * v_per_thread;

  // Read the query and zero the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * q_ptr[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;

  // For each key-value pair
  for (int i = simd_gid; i < N; i += BN) {

    // Read the key (FP16)
    for (int j = 0; j < qk_per_thread; j++) {
      k[j] = k_ptr[j];
    }

    // Compute Q·K score
    U score = 0;
    for (int j = 0; j < qk_per_thread; j++) {
      score += q[j] * k[j];
    }
    score = simd_sum(score);

    // Online softmax update
    U new_max = max(max_score, score);
    U factor = fast::exp(max_score - new_max);
    U exp_score = fast::exp(score - new_max);

    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;

    // Dequantize V and accumulate (weighted by attention score)
    // This thread handles V elements [simd_lid * v_per_thread .. +v_per_thread)
    for (int j = 0; j < v_per_thread; j++) {
      int elem = int(simd_lid) * v_per_thread + j;
      int word_idx = elem / 8;
      int bit_pos = (elem % 8) * 4;
      uint raw = (vd_ptr[word_idx] >> bit_pos) & 0xFu;

      int grp = elem / 32;
      U v_val = U(raw) * vs_ptr[grp] + vb_ptr[grp];

      o[j] = o[j] * factor + exp_score * v_val;
    }

    // Advance pointers
    k_ptr += inner_k_stride;
    vd_ptr += inner_v_data_stride;
    vs_ptr += inner_v_group_stride;
    vb_ptr += inner_v_group_stride;
  }

  // Reduction across simdgroups (identical to sdpa_vector)
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}
