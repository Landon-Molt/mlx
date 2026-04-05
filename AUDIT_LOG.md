# TurboQuant+ Parity Audit Log
**Date:** 2026-04-05
**Branch:** feature/turboquant-plus
**Auditor:** Claude Opus 4.6 + Tom Turney

---

## AUDIT 1: Attention Numerics Parity (MLX vs llama.cpp)

| Check | MLX | llama.cpp | Verdict |
|-------|-----|-----------|---------|
| **Accumulation precision** | FP32 (`typedef float U` in sdpa_vector.h:50; `AccumHelper<T>::accum_type = float` in transforms.h:57) | FP32 (Metal SDPA accumulates in float) | **MATCH** |
| **Softmax precision** | FP32 online softmax with `fast::exp()` in sdpa_vector.h:126-127; `mx.softmax(precise=True)` in base.py:97 | FP32 online softmax with `fast::exp()` in Metal flash-attention kernel | **MATCH** |
| **GQA head replication** | `kv_head_idx = q_batch_head_idx / gqa_factor` (sdpa_vector.h:63); `kv_head_idx = tid.y / params->gqa_factor` (steel_attention.h:94) — index division, no data copy | Index division `nkvh = n_heads / gqa_factor` — same approach | **MATCH** |
| **RoPE application point** | Before cache update: `keys = self.rope(keys, offset=cache.offset)` then `cache.update_and_fetch(keys, values)` (qwen2.py:73-75) | Before cache write: RoPE applied to K before storing in KV cache | **MATCH** |
| **Mask semantics** | Additive float mask or bool mask; causal = string "causal" short-circuiting to `do_causal` kernel constant (sdpa_vector.h:101-102) | Additive float mask for general; causal via `do_causal` constant | **MATCH** |

**Result: 5/5 MATCH. No issues found.**

---

## AUDIT 2: SRHT/WHT Implementation Parity

| Check | MLX | llama.cpp | Verdict |
|-------|-----|-----------|---------|
| **Normalization factor** | `mx.hadamard_transform` defaults to `1/sqrt(N)` (mlx/ops.cpp:510) | `inv_sqrt_128 = 0.08838834764831845` = `1/sqrt(128)` (turbo-wht.h:31) | **MATCH** (both = 1/sqrt(N)) |
| **Sign application order** | `signs2 * WHT(signs1 * x)` (turbo_kv_cache.py:297) | `signs1 → FWHT → signs2` (turbo-wht.h:38-41) | **MATCH** (same S2·H·S1 order) |
| **Sign values** | PRNG-generated via `mx.random.uniform(key=[seed,0])` | Hardcoded arrays in turbo-wht.h, generated from same seed=42 but different PRNG | **INTENTIONAL MISMATCH** — different PRNGs produce different ±1 patterns. Self-consistent within each impl. |
| **Centroid values** | Beta distribution centroids: `[-0.2364, ..., 0.2364]` for (4,128) | PolarQuant centroids: `[-0.1739, ..., 0.1739]` for 4-bit | **INTENTIONAL MISMATCH** — MLX uses Beta dist (proven +47% PPL), llama.cpp uses PolarQuant. ~1.36-1.60x ratio. Self-consistent. |

**Result: 2/4 exact match, 2/4 intentional design divergence. No bugs.**

---

## AUDIT 3: Weight Format Parity

| Check | MLX | llama.cpp Q8_0 | Verdict |
|-------|-----|----------------|---------|
| **Quantization type** | Affine: `x_hat = scale * q + bias` | Symmetric: `x_hat = q * d` (no bias) | **MISMATCH** |
| **Block/group size** | Configurable `group_size` (default 64) | Fixed `QK8_0 = 32` | **MISMATCH** |
| **Storage** | `(uint32 packed, float32 scales, float32 biases)` | `(int8[32], fp16 d)` per block | **MISMATCH** |

**Result: MLX's `mx.quantize` 8-bit != GGUF Q8_0. These are fundamentally different representations.** MLX uses affine quantization with bias; Q8_0 is symmetric with zero-point=0. This is expected — they serve different ecosystems. The TurboQuant KV cache doesn't use `mx.quantize` at all (it uses its own SRHT+Lloyd-Max pipeline), so this mismatch has **no impact on TurboQuant correctness**.

---

## AUDIT 4: Compressed-Domain Kernel Equivalence

| Metric | Value |
|--------|-------|
| Max absolute diff (fused vs reference) | **2.98e-07** |
| Mean absolute diff | **5.40e-08** |
| Relative error | **4.77e-07** |
| Threshold | 1e-4 |

**Result: PASS. Fused kernel output matches reference within FP32 epsilon.**

Tested configs:
- B=1, heads=4, T_kv=32, dim=128: rel_err=1.18e-07
- B=2, heads=8, T_kv=256, dim=128: rel_err=5.07e-08
- B=1, heads=4, T_kv=64, dim=64: rel_err=1.10e-07

---

## AUDIT 5: RoPE/Cache Position Tracking

| Check | TurboKVCache | KVCache (baseline) | Verdict |
|-------|-------------|-------------------|---------|
| **After prefill (5 tokens)** | offset=5 | offset=5 | **MATCH** |
| **After 15 decode steps** | offset=20 | offset=20 | **MATCH** |
| **K seq_len returned** | 20 | 20 | **MATCH** |
| **make_mask at offset=20** | None (causal, N=1) | None (causal, N=1) | **MATCH** |
| **RoPE uses cache.offset** | Yes (qwen2.py:73) | Yes (same code path) | **MATCH** |

**Result: 5/5 MATCH. Offset increments identically. RoPE positions are correct.**

---

## BUG FOUND AND FIXED

**File:** `tests/test_turbo_fused_attn.py`
**Test:** `test_wht_domain_math()`
**Bug:** The test computed `q_rot = WHT(signs1 * q)` but the correct SRHT is `q_rot = signs2 * WHT(signs1 * q)`. Missing the post-WHT sign flip (`signs2`).
**Impact:** Test-only bug. The actual fused kernel and encode/decode pipeline correctly apply both sign flips. This test was validating the mathematical identity incorrectly.
**Fix:** Added `signs2` import and applied it: `q_rot = (WHT(signs1 * q)) * signs2`.
**Verification:** rel_err dropped from 8.9e-02 to 8.3e-08. All 7 tests now pass.

---

## Summary

| Audit | Result |
|-------|--------|
| 1. Attention Numerics | **ALL MATCH** |
| 2. SRHT/WHT | **2 match, 2 intentional divergence** |
| 3. Weight Format | **Mismatch (irrelevant to TurboQuant)** |
| 4. Fused Kernel | **PASS (rel_err < 5e-07)** |
| 5. RoPE/Cache Position | **ALL MATCH** |
| Test fix | **1 bug fixed (test-only)** |
