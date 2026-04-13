#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_tq.h"

using namespace metal;

// Instantiate TQ attention for common configurations
// BQ=32, BK=32, BD=128, WM=4, WN=1 — matches steel_attention.h defaults
#define instantiate_attention_tq(type, bq, bk, bd, wm, wn, k_bits, v_bits) \
  instantiate_kernel(                                                       \
      "attention_tq_" #type "_bq" #bq "_bk" #bk "_bd" #bd                  \
      "_wm" #wm "_wn" #wn "_k" #k_bits "_v" #v_bits,                       \
      attention_tq,                                                         \
      type, bq, bk, bd, wm, wn, k_bits, v_bits, float)

// 4-bit TQ (MSE key bits = 3, value bits = 4) for common head dims
// BK=16 for BD>=128 to reduce threadgroup memory pressure
#define instantiate_attention_tq_heads(type)                    \
  instantiate_attention_tq(type, 32, 32, 64, 4, 1, 3, 4)      \
  instantiate_attention_tq(type, 32, 32, 128, 4, 1, 3, 4)     \
  instantiate_attention_tq(type, 32, 16, 256, 4, 1, 3, 4)     \
  instantiate_attention_tq(type, 16, 16, 512, 2, 1, 3, 4)

instantiate_attention_tq_heads(float16_t)
// clang-format on
