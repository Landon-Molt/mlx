#!/usr/bin/env python3
"""Debug correctness of fused single-dispatch kernel vs legacy 4-dispatch."""

import math
import mlx.core as mx
from mlx.nn.layers.turbo_kv_cache import (
    turbo_asymmetric_attention,
    turbo_fused_asymmetric_attention_single_dispatch,
    turbo_encode,
    turbo_weighted_value_sum,
    _sign_flip_vector,
    _sign_flip_vector2,
)

B = 1
n_q_heads = 4  # small for debug
n_kv_heads = 4  # no GQA to simplify
dim = 128
T_kv = 8  # tiny for manual inspection
bits = 4
seed = 42
scale = 1.0 / math.sqrt(dim)

# Generate test data
queries = mx.random.normal((B, n_q_heads, 1, dim))
keys = mx.random.normal((B, n_kv_heads, T_kv, dim))
values = mx.random.normal((B, n_kv_heads, T_kv, dim))

# Encode V
pv, vn = turbo_encode(values.reshape(B * n_kv_heads, T_kv, dim), bits=bits, seed=seed)
pv = pv.reshape(B, n_kv_heads, T_kv, -1)
vn = vn.reshape(B, n_kv_heads, T_kv, 1)
mx.eval(queries, keys, values, pv, vn)

# Legacy path step by step
scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale
weights = mx.softmax(scores, axis=-1)
mx.eval(scores, weights)

print("Scores[0,0,0,:8]:", scores[0, 0, 0, :].tolist())
print("Weights[0,0,0,:8]:", weights[0, 0, 0, :].tolist())

# V weighted sum (WHT domain)
v_wht = turbo_weighted_value_sum(weights, pv, vn, dim, bits=bits, seed=seed)
mx.eval(v_wht)

# Full legacy
out_legacy = turbo_asymmetric_attention(queries, keys, pv, vn, dim=dim, bits=bits, seed=seed, scale=scale)
mx.eval(out_legacy)

# Fused single dispatch
out_fused = turbo_fused_asymmetric_attention_single_dispatch(
    queries, keys, pv, vn, dim=dim, bits=bits, seed=seed, scale=scale
)
mx.eval(out_fused)

print("\nLegacy output[0,0,0,:8]:", out_legacy[0, 0, 0, :8].tolist())
print("Fused  output[0,0,0,:8]:", out_fused[0, 0, 0, :8].tolist())

diff = mx.abs(out_legacy - out_fused)
print(f"\nMax diff: {mx.max(diff).item():.6f}")
print(f"Mean diff: {mx.mean(diff).item():.6f}")

# Check if inverse WHT is the issue — compare pre-WHT results
# The fused kernel stores WHT-domain V sum then does inverse WHT in-kernel.
# Legacy does inverse WHT in Python via mx.hadamard_transform.
# Let's check: compute WHT-domain V sum manually and compare
signs1 = _sign_flip_vector(dim, seed)
signs2 = _sign_flip_vector2(dim, seed)
mx.eval(signs1, signs2)

# The v_wht from turbo_weighted_value_sum already has inverse WHT applied.
# Let's compute it without inverse WHT:
# In legacy: output = signs1 * WHT(signs2 * v_rot) where v_rot is WHT-domain sum
# So v_rot = inverse(signs2 * WHT^-1(signs1 * output))
# Since WHT is self-inverse: v_rot = signs2 * WHT(signs1 * output) / ... no this gets messy.

# Instead let's just check shapes and values more carefully
print(f"\nQuery shape: {queries.shape}")
print(f"Key shape: {keys.shape}")
print(f"PV shape: {pv.shape}")
print(f"VN shape: {vn.shape}")

# --- GQA test ---
print("\n=== GQA Test (n_q=32, n_kv=8) ===")
n_q_heads_gqa = 32
n_kv_heads_gqa = 8
T_kv_gqa = 64

queries_gqa = mx.random.normal((B, n_q_heads_gqa, 1, dim))
keys_gqa = mx.random.normal((B, n_kv_heads_gqa, T_kv_gqa, dim))
values_gqa = mx.random.normal((B, n_kv_heads_gqa, T_kv_gqa, dim))

pv_gqa, vn_gqa = turbo_encode(
    values_gqa.reshape(B * n_kv_heads_gqa, T_kv_gqa, dim), bits=bits, seed=seed
)
pv_gqa = pv_gqa.reshape(B, n_kv_heads_gqa, T_kv_gqa, -1)
vn_gqa = vn_gqa.reshape(B, n_kv_heads_gqa, T_kv_gqa, 1)
mx.eval(queries_gqa, keys_gqa, values_gqa, pv_gqa, vn_gqa)

out_legacy_gqa = turbo_asymmetric_attention(
    queries_gqa, keys_gqa, pv_gqa, vn_gqa, dim=dim, bits=bits, seed=seed, scale=scale
)
out_fused_gqa = turbo_fused_asymmetric_attention_single_dispatch(
    queries_gqa, keys_gqa, pv_gqa, vn_gqa, dim=dim, bits=bits, seed=seed, scale=scale
)
mx.eval(out_legacy_gqa, out_fused_gqa)

diff_gqa = mx.abs(out_legacy_gqa - out_fused_gqa)
print(f"GQA Max diff: {mx.max(diff_gqa).item():.6f}")
print(f"GQA Mean diff: {mx.mean(diff_gqa).item():.6f}")

# --- Larger T_kv test ---
print("\n=== Large T_kv Test (T_kv=1024) ===")
T_kv_large = 1024
keys_large = mx.random.normal((B, n_kv_heads, T_kv_large, dim))
values_large = mx.random.normal((B, n_kv_heads, T_kv_large, dim))
pv_large, vn_large = turbo_encode(
    values_large.reshape(B * n_kv_heads, T_kv_large, dim), bits=bits, seed=seed
)
pv_large = pv_large.reshape(B, n_kv_heads, T_kv_large, -1)
vn_large = vn_large.reshape(B, n_kv_heads, T_kv_large, 1)
mx.eval(keys_large, values_large, pv_large, vn_large)

out_legacy_large = turbo_asymmetric_attention(
    queries, keys_large, pv_large, vn_large, dim=dim, bits=bits, seed=seed, scale=scale
)
out_fused_large = turbo_fused_asymmetric_attention_single_dispatch(
    queries, keys_large, pv_large, vn_large, dim=dim, bits=bits, seed=seed, scale=scale
)
mx.eval(out_legacy_large, out_fused_large)

diff_large = mx.abs(out_legacy_large - out_fused_large)
print(f"Large T_kv Max diff: {mx.max(diff_large).item():.6f}")
print(f"Large T_kv Mean diff: {mx.mean(diff_large).item():.6f}")
