#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector_tq.h"

using namespace metal;

// TQ SDPA vector instantiations
// For now: float16 output, common head dims, 3-bit keys (4-bit ProdCodec), 4-bit values
#define instantiate_sdpa_vector_tq(type, dim, k_bits, v_bits) \
  instantiate_kernel(                                          \
      "sdpa_vector_tq_" #type "_" #dim "_k" #k_bits "_v" #v_bits, \
      sdpa_vector_tq,                                          \
      type,                                                    \
      dim,                                                     \
      k_bits,                                                  \
      v_bits)

// 2-pass variant for long sequences
#define instantiate_sdpa_vector_tq_2pass(type, dim, k_bits, v_bits) \
  instantiate_kernel(                                                \
      "sdpa_vector_tq_2pass_1_" #type "_" #dim "_k" #k_bits "_v" #v_bits, \
      sdpa_vector_tq_2pass_1,                                        \
      type,                                                          \
      dim,                                                           \
      k_bits,                                                        \
      v_bits)

// 4-bit TurboQuant: key MSE bits = 3 (ProdCodec uses bits-1), value bits = 4
#define instantiate_sdpa_vector_tq_heads(type)           \
  instantiate_sdpa_vector_tq(type, 64, 3, 4)            \
  instantiate_sdpa_vector_tq(type, 128, 3, 4)           \
  instantiate_sdpa_vector_tq(type, 256, 3, 4)           \
  instantiate_sdpa_vector_tq(type, 512, 3, 4)           \
  instantiate_sdpa_vector_tq_2pass(type, 64, 3, 4)      \
  instantiate_sdpa_vector_tq_2pass(type, 128, 3, 4)     \
  instantiate_sdpa_vector_tq_2pass(type, 256, 3, 4)     \
  instantiate_sdpa_vector_tq_2pass(type, 512, 3, 4)

instantiate_sdpa_vector_tq_heads(float16_t)
// clang-format on
