#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"

using namespace metal;

// SDPA vector instantiations
#define instantiate_sdpa_vector_aggregation(type, value_dim) \
  instantiate_kernel(                                        \
      "sdpa_vector_2pass_2_" #type "_" #value_dim,           \
      sdpa_vector_2pass_2,                                   \
      type,                                                  \
      value_dim)

#define instantiate_sdpa_vector(type, qk_dim, value_dim)       \
  instantiate_kernel(                                          \
      "sdpa_vector_" #type "_" #qk_dim "_" #value_dim,         \
      sdpa_vector,                                             \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)                                               \
  instantiate_kernel(                                          \
      "sdpa_vector_2pass_1_" #type "_" #qk_dim "_" #value_dim, \
      sdpa_vector_2pass_1,                                     \
      type,                                                    \
      qk_dim,                                                  \
      value_dim)

#define instantiate_sdpa_vector_heads(type)      \
  instantiate_sdpa_vector(type, 64, 64)          \
  instantiate_sdpa_vector(type, 96, 96)          \
  instantiate_sdpa_vector(type, 128, 128)        \
  instantiate_sdpa_vector(type, 256, 256)        \
  instantiate_sdpa_vector_aggregation(type, 64)  \
  instantiate_sdpa_vector_aggregation(type, 96)  \
  instantiate_sdpa_vector_aggregation(type, 128) \
  instantiate_sdpa_vector_aggregation(type, 256)

instantiate_sdpa_vector_heads(float)
instantiate_sdpa_vector_heads(bfloat16_t)
instantiate_sdpa_vector_heads(float16_t)
    // clang-format on

// Quantized-V SDPA vector instantiations (K=fp16, V=4bit/8bit scalar)
#include "mlx/backend/metal/kernels/sdpa_vector_qv.h"

#define instantiate_sdpa_vector_qv(type, dim) \
  instantiate_kernel("sdpa_vector_qv_" #type "_" #dim, sdpa_vector_qv, type, dim)
#define instantiate_sdpa_vector_qv8(type, dim) \
  instantiate_kernel("sdpa_vector_qv8_" #type "_" #dim, sdpa_vector_qv8, type, dim)

#define instantiate_sdpa_vector_qv_heads(type) \
  instantiate_sdpa_vector_qv(type, 64)  \
  instantiate_sdpa_vector_qv(type, 96)  \
  instantiate_sdpa_vector_qv(type, 128) \
  instantiate_sdpa_vector_qv(type, 256) \
  instantiate_sdpa_vector_qv8(type, 64)  \
  instantiate_sdpa_vector_qv8(type, 96)  \
  instantiate_sdpa_vector_qv8(type, 128) \
  instantiate_sdpa_vector_qv8(type, 256)

instantiate_sdpa_vector_qv_heads(float16_t)
instantiate_sdpa_vector_qv_heads(bfloat16_t)
