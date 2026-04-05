# TurboQuant KV Cache for MLX — Changelog

## feature/turboquant-plus (28 commits)

### Core Implementation
- `turbo_encode()` / `turbo_decode()` — SRHT + Lloyd-Max quantization
- `TurboKVCache` — two-phase (raw prefill → compressed decode)
- `turbo_fused_attention()` — compressed-domain Metal kernel (symmetric)
- `turbo_asymmetric_attention()` — K=FP16 scoring + packed V sum (asymmetric)
- `turbo_weighted_value_sum()` — tiled Metal kernel for V weighted sum
- `patch_mlx_lm()` — runtime SDPA patching for zero-copy fused attention
- `recommend_config()` — model-aware config recommendations

### Metal Kernels
- Fused encode: norm → sign → WHT → quantize → pack (single dispatch)
- Fused decode: unpack → centroid → WHT → sign → scale (single dispatch)
- Fused attention: Q×packed_K scoring + softmax + packed_V sum (single dispatch)
- **Two-pass TurboFlash (B=64)**: pass 1 = T/64 parallel threadgroups for block scoring + partial V sum, pass 2 = merge with online softmax correction + inverse WHT. Inspired by Eric Kryski's architecture.
- NR0=2 multi-row: share KV dequant across 2 query rows
- Tiled weighted V sum: thread-per-dim + T_kv tiling (TILE_T=256)

### Quality Features
- Beta distribution centroids (empirically proven better than N(0,1))
- Dual SRHT sign arrays (signs1 → WHT → signs2)
- Boundary layer protection (first/last 2 KV layers at FP16)
- Asymmetric K/V (K=FP16, V=compressed — mandatory for dense models)
- Deferred compression threshold (raw FP16 below min_compress_tokens)
- Compact mode (drop decoded FP16 above threshold for memory)
- Non-power-of-2 dim support (zero-padding to next power of 2)
- Block size=32 WHT option (better PPL + faster encode)
- Sparse attention mask (skip near-zero V contributions)

### Configurations
- `TURBO_BOUNDARY_LAYERS=N` — override boundary layer count
- `TURBO_USE_N01_CENTROIDS=1` — A/B test N(0,1) vs Beta centroids
- `TURBO_DISABLE_FUSED_KERNEL=1` — force Python encode/decode path

### Bug Fixes
- GQA head derivation (queries vs packed_keys)
- Asymmetric patched path skipping decoded V seeding
- Python `from X import Y` reference copy in module patching
- Boundary layer NaN from extreme V norms (184x on last layer)
- Fused V sum kernel: serial loop → tiled parallel (128 barriers → 1/tile)

### Validated On
- Qwen3.5-2B-8bit (hybrid, 6/24 KV layers)
- Qwen2.5-7B-Instruct-8bit (dense, 28/28 KV layers)
- phi-4-8bit (dense, 40/40 KV layers)
- Qwen3.5-27B-8bit (hybrid, 16/64 KV layers)
- Qwen3.5-35B-A3B-8bit (MoE, 10/40 KV layers)
- M5 Max 128GB + M2 Pro 16GB

### Best Result
7B dense (Qwen2.5-7B-Instruct-8bit, M5 Max):
- Decode: **100% baseline** (63.7 vs 63.7 tok/s) — two-pass TurboFlash kernel
- PPL: +0.04%
- KLD: 0.000305, Top-1: 100%
- NIAH: 30/30 PASS
- Memory: zero extra (fused path)
- Prefill: 2.2x faster
