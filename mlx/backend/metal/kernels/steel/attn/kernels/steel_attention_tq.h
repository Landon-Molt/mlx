// Copyright © 2024-2026 Apple Inc.
// TurboQuant variant of steel_attention.
//
// Loads K and V from TurboQuant MSE codebook format instead of fp16.
// Dequantizes into threadgroup memory as fp16, then uses standard
// steel MMA for Q@K^T scoring and softmax(S)@V accumulation.
//
// MSE-only scoring (no QJL correction) for simplicity.
// Keys and values both use MSECodec format: codebook[index] * norm.

#include "mlx/backend/metal/kernels/steel/attn/attn.h"

using namespace mlx::steel;

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];
constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// TQ-specific params passed alongside AttnParams
struct TQParams {
  int k_bits;
  int v_bits;
  int k_packed_width;   // (D * k_bits + 31) / 32
  int v_packed_width;   // (D * v_bits + 31) / 32
};

// Inline dequant: unpack codebook index from packed uint32 array
template <int BITS>
METAL_FUNC float tq_dequant_elem(
    const device uint32_t* packed,
    int d,
    const device float* codebook,
    float norm) {
  constexpr uint MASK = (1u << BITS) - 1u;
  int bo = d * BITS;
  uint idx = (packed[bo >> 5] >> (bo & 31));
  if ((bo & 31) + BITS > 32) {
    idx |= packed[(bo >> 5) + 1] << (BITS - ((bo & 31) + BITS - 32));
  }
  idx &= MASK;
  return codebook[idx] * norm;
}

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    int K_BITS,
    int V_BITS,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_tq(
    // Queries (pre-rotated + projected, fp16)
    const device T* Q [[buffer(0)]],
    const device T* Q_proj [[buffer(1)]],
    // Key TQ state (ProdCodec: MSE + QJL)
    const device half* key_norms [[buffer(2)]],
    const device uint32_t* key_indices [[buffer(3)]],
    const device half* key_res_norms [[buffer(4)]],
    const device uint32_t* key_signs [[buffer(5)]],
    // Value TQ state (MSECodec)
    const device half* val_norms [[buffer(6)]],
    const device uint32_t* val_indices [[buffer(7)]],
    // Codebooks + scale
    const device float* key_codebook [[buffer(8)]],
    const device float* key_scale [[buffer(9)]],
    const device float* val_codebook [[buffer(10)]],
    // Output
    device T* O [[buffer(11)]],
    // Params
    const constant AttnParams* params [[buffer(12)]],
    const constant TQParams* tq_params [[buffer(13)]],
    // Thread indices
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  (void)lid;

  const int kv_packed_stride_k = tq_params->k_packed_width;
  const int kv_packed_stride_v = tq_params->v_packed_width;
  const int kv_sign_stride = (BD + 31) / 32;
  const float kscale = key_scale[0];

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] +
      tidl.y * params->Q_strides[1] +
      tidl.x * BQ * params->Q_strides[2];

  Q_proj += tidl.z * params->Q_strides[0] +
      tidl.y * params->Q_strides[1] +
      tidl.x * BQ * params->Q_strides[2];

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  ulong kv_offset = tidl.z * params->K_strides[0] + kv_head_idx * params->K_strides[1];

  O += tidl.z * params->O_strides[0] +
      tidl.y * params->O_strides[1] +
      tidl.x * BQ * params->O_strides[2];

  // Threadgroup memory — need space for Q, Q_proj, K, K_sign, V
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ_tgp = BD + padQ;
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;

  constexpr short tgp_mem_0 = (BK + padK) * (BD);
  constexpr short tgp_mem_1 = BK * (BD + padV);
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1;

  // Single Q buffer — Q_rot and Q_proj loaded sequentially to halve
  // threadgroup memory. This allows D up to 512 within 32KB limit.
  threadgroup T Q_smem[BQ * (BD + padQ)];
  threadgroup T KV_smem[tgp_mem_s];

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem;
  threadgroup T* Vs = KV_smem;

  // Q loader — reused for both Q_rot and Q_proj
  using QBlockLoader = BlockLoaderT<T, BQ, BD, LDQ_tgp, 1, 1, WM * WN * 32>;

  QBlockLoader loader_q(Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  QBlockLoader loader_qp(Q_proj, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);

  const AccumType scale = params->scale * M_LOG2E_F;

  // MMA tiles
  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  constexpr int TQ_tiles = BQ / (kNWarps * kFragSize);
  constexpr int TK_tiles = BK / kFragSize;
  constexpr int TD_tiles = BD / kFragSize;

  MMATile<AccumType, TQ_tiles, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK_tiles, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ_tiles, TK_tiles, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ_tiles, TD_tiles, MMAFrag_acc_t> Otile;

  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ_tiles * simd_group_id;

  const short Qs_offset = (tm + sm) * LDQ_tgp + sn;
  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;

  constexpr short Qs_tile_stride = kFragSize;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q_rot into shared memory (Q_proj loaded later per KV block)
  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }

  // Softmax accumulators
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;
  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  // Thread index for TQ loading
  const int thread_idx = simd_group_id * 32 + simd_lane_id;
  const int total_threads = WM * WN * 32;

  // KV block loop
  for (int kb = 0; kb < params->NK; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Load K_MSE tile: dequant codebook → fp16 transposed ===
    {
      const int kv_seq_start = kb * BK;
      const int kv_seq_end = min(kv_seq_start + BK, params->kL);
      const int valid_tokens = kv_seq_end - kv_seq_start;
      const int total_elems = BK * BD;
      const int elems_per_thread = (total_elems + total_threads - 1) / total_threads;

      for (int e = 0; e < elems_per_thread; e++) {
        int flat_idx = thread_idx * elems_per_thread + e;
        if (flat_idx >= total_elems) break;

        int t_local = flat_idx / BD;
        int d = flat_idx % BD;
        int t_global = kv_seq_start + t_local;

        T val = T(0);
        if (t_local < valid_tokens && t_global < params->kL) {
          float kn = float(key_norms[kv_offset + t_global]);
          const device uint32_t* kp = key_indices + (kv_offset + t_global) * kv_packed_stride_k;
          val = T(tq_dequant_elem<K_BITS>(kp, d, key_codebook, kn));
        }
        // Transposed: Ks[d][t]
        Ks[d * LDK_tgp + t_local] = val;
      }
    }

    // S_mse = Q_rot @ K_mse^T
    Stile.clear();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD_tiles; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(&Ks[Ks_offset + dd * Ks_tile_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // === QJL correction: reload Q_proj into Q_smem, load K_sign, second matmul ===

    // Step 1: Load K_sign tile (reuse KV_smem)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const int kv_seq_start = kb * BK;
      const int kv_seq_end = min(kv_seq_start + BK, params->kL);
      const int valid_tokens = kv_seq_end - kv_seq_start;
      const int total_elems = BK * BD;
      const int elems_per_thread = (total_elems + total_threads - 1) / total_threads;

      for (int e = 0; e < elems_per_thread; e++) {
        int flat_idx = thread_idx * elems_per_thread + e;
        if (flat_idx >= total_elems) break;

        int t_local = flat_idx / BD;
        int d = flat_idx % BD;
        int t_global = kv_seq_start + t_local;

        T val = T(0);
        if (t_local < valid_tokens && t_global < params->kL) {
          float kn = float(key_norms[kv_offset + t_global]);
          float krn = float(key_res_norms[kv_offset + t_global]);
          float ksr = kn * kscale * krn;
          const device uint32_t* sp = key_signs + (kv_offset + t_global) * kv_sign_stride;
          uint sb = (sp[d >> 5] >> (d & 31)) & 1u;
          val = T(sb ? ksr : -ksr);
        }
        Ks[d * LDK_tgp + t_local] = val;
      }
    }

    // Step 2: Load Q_proj into Q_smem (overwriting Q_rot — will reload later)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
      loader_qp.load_safe(short2(BD, params->qL_rem));
    } else {
      loader_qp.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: S_qjl = Q_proj @ K_sign^T  (add to S_mse already in Stile)
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD_tiles; dd++) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(&Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(&Ks[Ks_offset + dd * Ks_tile_stride]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }

    // Step 4: Reload Q_rot for next iteration's MSE scoring
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
      loader_q.load_safe(short2(BD, params->qL_rem));
    } else {
      loader_q.load_unsafe();
    }

    // Apply scale (log2e for fast::exp2 in softmax)
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    // Mask out invalid tokens
    if (!align_K && kb == (params->NK_aligned)) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Causal mask
    if (do_causal) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Load V tile: dequant TQ → fp16 in shared memory ===
    {
      const int kv_seq_start = kb * BK;
      const int kv_seq_end = min(kv_seq_start + BK, params->kL);
      const int valid_tokens = kv_seq_end - kv_seq_start;

      const int total_elems = BK * BD;
      const int elems_per_thread = (total_elems + total_threads - 1) / total_threads;

      for (int e = 0; e < elems_per_thread; e++) {
        int flat_idx = thread_idx * elems_per_thread + e;
        if (flat_idx >= total_elems) break;

        int t_local = flat_idx / BD;
        int d = flat_idx % BD;
        int t_global = kv_seq_start + t_local;

        T val = T(0);
        if (t_local < valid_tokens && t_global < params->kL) {
          float vn = float(val_norms[kv_offset + t_global]);
          const device uint32_t* vp = val_indices + (kv_offset + t_global) * kv_packed_stride_v;
          val = T(tq_dequant_elem<V_BITS>(vp, d, val_codebook, vn));
        }
        // Store non-transposed: Vs[t][d]
        Vs[t_local * LDV_tgp + d] = val;
      }
    }

    // Online softmax
    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }
    Otile.template row_bin_op<MulOp>(factor);

    // O += S @ V
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ_tiles; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TD_tiles; id++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK_tiles; ik++) {
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }
          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;
          Vtile.template load<T, 1, 1, LDV_tgp, 1>(&Vs[Vs_offset + kk * LDV_tgp + dd]);
          if constexpr (BD == 128) {
            simdgroup_barrier(mem_flags::mem_none);
          }
          MMAFrag_acc_t::mma(
              Otile.frag_at(iq, id),
              Stile.frag_at(iq, ik),
              Vtile.frag_at(0, 0),
              Otile.frag_at(iq, id));
        }
      }
    }
  }

  // Normalize and store
  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  O += (tm + sm) * params->O_strides[2] + sn;
  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));
    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;
    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
}
