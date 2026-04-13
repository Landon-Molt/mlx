// Copyright © 2025 Shiyang "Landon" Yue
// Steel flash attention with quantized V (prefill L>1)
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_qv.h"

using namespace metal;

#define instantiate_attention_qv(type, bq, bk, bd, wm, wn) \
  instantiate_kernel(                                        \
      "attention_qv_" #type "_bq" #bq "_bk" #bk "_bd" #bd  \
      "_wm" #wm "_wn" #wn,                                  \
      attention_qv, type, bq, bk, bd, wm, wn, float)

#define instantiate_attention_qv_heads(type)         \
  instantiate_attention_qv(type, 32, 32, 64, 4, 1)  \
  instantiate_attention_qv(type, 32, 32, 128, 4, 1) \
  instantiate_attention_qv(type, 32, 16, 256, 4, 1)

instantiate_attention_qv_heads(float16_t)
// clang-format on
