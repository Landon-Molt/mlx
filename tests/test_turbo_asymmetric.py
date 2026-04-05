#!/usr/bin/env python3
"""Test: Asymmetric fused attention (K=FP16, V=turbo4).

Validates that the asymmetric attention path (turbo_asymmetric_attention and
turbo_weighted_value_sum) produces correct output by comparing against the
decode-then-matmul reference.

The asymmetric path is the recommended config for TurboQuant — K stays at FP16
(errors amplified by softmax exponential), V is turbo4 (errors merely averaged).
This test ensures the Metal kernel weighted V sum matches decoding V to FP16
and doing standard matmul.

Usage:
    python3 tests/test_turbo_asymmetric.py
    python3 tests/test_turbo_asymmetric.py -v          # verbose
    python3 tests/test_turbo_asymmetric.py --benchmark  # include perf test
"""

import argparse
import math
import sys
import time

import mlx.core as mx

from mlx.nn.layers.turbo_kv_cache import (
    TurboKVCache,
    turbo_asymmetric_attention,
    turbo_decode,
    turbo_encode,
    turbo_weighted_value_sum,
    _get_codebook,
    _sign_flip_vector,
    _sign_flip_vector2,
)


def test_weighted_value_sum_basic(verbose=False):
    """Basic correctness: weighted V sum kernel vs decode-then-matmul."""
    print("=== Test: Weighted V sum basic (B=1, heads=4, T_kv=32, dim=128) ===")

    B, n_heads, T_kv, dim = 1, 4, 32, 128
    bits = 4
    seed = 42

    mx.random.seed(0)
    v = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)
    # Random attention weights (post-softmax)
    raw_scores = mx.random.normal((B, n_heads, 1, T_kv)).astype(mx.float32)
    weights = mx.softmax(raw_scores, axis=-1)

    # Encode V
    pv, vn = turbo_encode(v, bits=bits, seed=seed)

    if verbose:
        print(f"  packed_values shape: {pv.shape}, dtype: {pv.dtype}")
        print(f"  value_norms shape: {vn.shape}, dtype: {vn.dtype}")
        print(f"  weights shape: {weights.shape}")

    # Reference: decode V to FP16, then matmul
    v_decoded = turbo_decode(pv, vn, dim, bits=bits, seed=seed)
    ref_out = weights @ v_decoded
    mx.eval(ref_out)

    # Test: weighted V sum kernel
    test_out = turbo_weighted_value_sum(weights, pv, vn, dim, bits=bits, seed=seed)
    mx.eval(test_out)

    if verbose:
        print(f"  ref_out shape: {ref_out.shape}")
        print(f"  test_out shape: {test_out.shape}")

    # Compare
    diff = mx.abs(ref_out - test_out)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()

    print(f"  max diff: {max_diff:.6f}")
    print(f"  mean diff: {mean_diff:.6f}")

    # Tolerance: both paths go through WHT inverse, but the kernel accumulates
    # in float32 vs the decode path which may have different FP rounding
    assert max_diff < 0.01, f"max diff {max_diff} exceeds tolerance 0.01"
    assert mean_diff < 0.001, f"mean diff {mean_diff} exceeds tolerance 0.001"
    print("  PASSED\n")


def test_weighted_value_sum_gqa(verbose=False):
    """GQA correctness: 28 query heads, 4 KV heads."""
    print("=== Test: Weighted V sum GQA (B=1, nq=28, nkv=4, T_kv=64, dim=128) ===")

    B, n_q_heads, n_kv_heads, T_kv, dim = 1, 28, 4, 64, 128
    bits = 4
    seed = 42

    mx.random.seed(1)
    v = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float32)
    raw_scores = mx.random.normal((B, n_q_heads, 1, T_kv)).astype(mx.float32)
    weights = mx.softmax(raw_scores, axis=-1)

    pv, vn = turbo_encode(v, bits=bits, seed=seed)

    # Reference: expand KV heads, decode, matmul
    gqa_factor = n_q_heads // n_kv_heads
    v_expanded = mx.repeat(v, gqa_factor, axis=1)  # Use original v as proxy
    v_decoded = turbo_decode(
        mx.repeat(pv, gqa_factor, axis=1),
        mx.repeat(vn, gqa_factor, axis=1),
        dim, bits=bits, seed=seed,
    )
    ref_out = weights @ v_decoded
    mx.eval(ref_out)

    # Test: kernel handles GQA internally
    test_out = turbo_weighted_value_sum(weights, pv, vn, dim, bits=bits, seed=seed)
    mx.eval(test_out)

    diff = mx.abs(ref_out - test_out)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()

    print(f"  max diff: {max_diff:.6f}")
    print(f"  mean diff: {mean_diff:.6f}")

    assert max_diff < 0.01, f"max diff {max_diff} exceeds tolerance 0.01"
    assert mean_diff < 0.001, f"mean diff {mean_diff} exceeds tolerance 0.001"
    print("  PASSED\n")


def test_asymmetric_attention_basic(verbose=False):
    """Full asymmetric attention: FP16 K scoring + turbo V weighted sum."""
    print("=== Test: Asymmetric attention (B=1, heads=4, T_kv=32, dim=128) ===")

    B, n_heads, T_kv, dim = 1, 4, 32, 128
    bits = 4
    seed = 42
    scale = 1.0 / math.sqrt(dim)

    mx.random.seed(2)
    q = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
    k = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)
    v = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)

    # Encode V only (K stays FP16)
    pv, vn = turbo_encode(v, bits=bits, seed=seed)

    # Reference: standard attention with decoded V
    v_decoded = turbo_decode(pv, vn, dim, bits=bits, seed=seed)
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    weights = mx.softmax(scores, axis=-1)
    ref_out = weights @ v_decoded
    mx.eval(ref_out)

    # Test: asymmetric attention
    test_out = turbo_asymmetric_attention(
        q, k, pv, vn, dim, bits=bits, seed=seed, scale=scale,
    )
    mx.eval(test_out)

    if verbose:
        print(f"  ref_out shape: {ref_out.shape}")
        print(f"  test_out shape: {test_out.shape}")

    diff = mx.abs(ref_out - test_out)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()

    print(f"  max diff: {max_diff:.6f}")
    print(f"  mean diff: {mean_diff:.6f}")

    assert max_diff < 0.01, f"max diff {max_diff} exceeds tolerance 0.01"
    assert mean_diff < 0.001, f"mean diff {mean_diff} exceeds tolerance 0.001"
    print("  PASSED\n")


def test_asymmetric_attention_gqa(verbose=False):
    """GQA asymmetric: 28 query heads, 4 KV heads — the 7B dense config."""
    print("=== Test: Asymmetric GQA (B=1, nq=28, nkv=4, T_kv=128, dim=128) ===")

    B, n_q_heads, n_kv_heads, T_kv, dim = 1, 28, 4, 128, 128
    bits = 4
    seed = 42
    scale = 1.0 / math.sqrt(dim)

    mx.random.seed(3)
    q = mx.random.normal((B, n_q_heads, 1, dim)).astype(mx.float32)
    k = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float32)
    v = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float32)

    pv, vn = turbo_encode(v, bits=bits, seed=seed)

    # Reference: expand K for GQA, decode V, standard attention
    gqa_factor = n_q_heads // n_kv_heads
    k_expanded = mx.repeat(k, gqa_factor, axis=1)
    v_decoded = turbo_decode(
        mx.repeat(pv, gqa_factor, axis=1),
        mx.repeat(vn, gqa_factor, axis=1),
        dim, bits=bits, seed=seed,
    )
    scores = (q @ k_expanded.transpose(0, 1, 3, 2)) * scale
    weights = mx.softmax(scores, axis=-1)
    ref_out = weights @ v_decoded
    mx.eval(ref_out)

    # Test: asymmetric handles GQA internally
    test_out = turbo_asymmetric_attention(
        q, k, pv, vn, dim, bits=bits, seed=seed, scale=scale,
    )
    mx.eval(test_out)

    diff = mx.abs(ref_out - test_out)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()

    print(f"  max diff: {max_diff:.6f}")
    print(f"  mean diff: {mean_diff:.6f}")

    assert max_diff < 0.01, f"max diff {max_diff} exceeds tolerance 0.01"
    assert mean_diff < 0.001, f"mean diff {mean_diff} exceeds tolerance 0.001"
    print("  PASSED\n")


def test_asymmetric_dims(verbose=False):
    """Test various dim sizes: 64, 128, 256."""
    print("=== Test: Asymmetric attention across dims (64, 128, 256) ===")

    for dim in [64, 128, 256]:
        B, n_heads, T_kv = 1, 4, 32
        bits = 4
        seed = 42
        scale = 1.0 / math.sqrt(dim)

        mx.random.seed(dim)
        q = mx.random.normal((B, n_heads, 1, dim)).astype(mx.float32)
        k = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)
        v = mx.random.normal((B, n_heads, T_kv, dim)).astype(mx.float32)

        pv, vn = turbo_encode(v, bits=bits, seed=seed)

        # Reference
        v_decoded = turbo_decode(pv, vn, dim, bits=bits, seed=seed)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        weights = mx.softmax(scores, axis=-1)
        ref_out = weights @ v_decoded
        mx.eval(ref_out)

        # Test
        test_out = turbo_asymmetric_attention(
            q, k, pv, vn, dim, bits=bits, seed=seed, scale=scale,
        )
        mx.eval(test_out)

        diff = mx.abs(ref_out - test_out)
        max_diff = mx.max(diff).item()
        mean_diff = mx.mean(diff).item()

        status = "PASS" if max_diff < 0.01 else "FAIL"
        print(f"  dim={dim}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]")
        assert max_diff < 0.01, f"dim={dim}: max diff {max_diff} exceeds tolerance"

    print("  PASSED\n")


def test_turbo_kv_cache_asymmetric(verbose=False):
    """Test TurboKVCache in asymmetric mode (key_bits=0, bits=4)."""
    print("=== Test: TurboKVCache asymmetric (key_bits=0, bits=4) ===")

    B, n_q_heads, n_kv_heads, T_kv, dim = 1, 8, 8, 16, 128
    bits = 4
    seed = 42

    mx.random.seed(4)

    # Create asymmetric cache
    cache = TurboKVCache(bits=4, key_bits=0, seed=seed, min_compress_tokens=0)
    assert not cache.compress_keys, "K should not be compressed with key_bits=0"
    assert cache.compress_values, "V should be compressed with bits=4"

    # Simulate prefill
    k_prefill = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float16)
    v_prefill = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float16)
    all_k, all_v = cache.update_and_fetch(k_prefill, v_prefill)
    mx.eval(all_k, all_v)

    # Check that raw prefill was stored
    assert cache._raw_keys is not None or cache._fp_keys is not None or cache._is_compressed

    # Simulate decode steps to trigger compression
    for step in range(3):
        k_new = mx.random.normal((B, n_kv_heads, 1, dim)).astype(mx.float16)
        v_new = mx.random.normal((B, n_kv_heads, 1, dim)).astype(mx.float16)
        all_k, all_v = cache.update_and_fetch(k_new, v_new)
        mx.eval(all_k, all_v)

    assert cache._is_compressed, "Cache should be compressed after decode steps"
    assert cache._fp_keys is not None, "FP keys should be stored (K not compressed)"
    assert cache._packed_values is not None, "Packed values should exist (V compressed)"
    assert cache._packed_keys is None, "Packed keys should NOT exist (K not compressed)"

    print(f"  offset: {cache.offset}")
    print(f"  fp_keys shape: {cache._fp_keys.shape}")
    print(f"  packed_values shape: {cache._packed_values.shape}")
    print(f"  value_norms shape: {cache._value_norms.shape}")
    print("  PASSED\n")


def test_turbo_kv_cache_asymmetric_patched(verbose=False):
    """Test patched asymmetric mode: no V decode buffers."""
    print("=== Test: TurboKVCache asymmetric patched (no V decode) ===")

    B, n_kv_heads, dim = 1, 4, 128
    T_prefill = 16
    bits = 4
    seed = 42

    mx.random.seed(5)

    # Create asymmetric cache with patched flag
    cache = TurboKVCache(bits=4, key_bits=0, seed=seed, min_compress_tokens=0)
    cache._patched = True

    # Prefill
    k_prefill = mx.random.normal((B, n_kv_heads, T_prefill, dim)).astype(mx.float16)
    v_prefill = mx.random.normal((B, n_kv_heads, T_prefill, dim)).astype(mx.float16)
    all_k, all_v = cache.update_and_fetch(k_prefill, v_prefill)
    mx.eval(all_k, all_v)

    # First decode step triggers compression
    k_new = mx.random.normal((B, n_kv_heads, 1, dim)).astype(mx.float16)
    v_new = mx.random.normal((B, n_kv_heads, 1, dim)).astype(mx.float16)
    all_k, all_v = cache.update_and_fetch(k_new, v_new)
    mx.eval(all_k, all_v)

    assert cache._is_compressed, "Cache should be compressed"

    # Second decode step — should use patched asymmetric path
    k_new2 = mx.random.normal((B, n_kv_heads, 1, dim)).astype(mx.float16)
    v_new2 = mx.random.normal((B, n_kv_heads, 1, dim)).astype(mx.float16)
    all_k, all_v = cache.update_and_fetch(k_new2, v_new2)
    mx.eval(all_k, all_v)

    # Verify: no decoded V buffer (the whole point of patched asymmetric)
    assert cache._decoded_values is None, \
        "Decoded values should be None in patched asymmetric mode"
    assert cache._fp_keys is not None, \
        "FP keys should exist (K not compressed)"
    assert cache._packed_values is not None, \
        "Packed values should exist (V compressed)"
    assert cache._fp_keys.shape[2] == T_prefill + 2, \
        f"Expected {T_prefill + 2} tokens in FP keys, got {cache._fp_keys.shape[2]}"
    assert cache._packed_values.shape[2] == T_prefill + 2, \
        f"Expected {T_prefill + 2} tokens in packed values, got {cache._packed_values.shape[2]}"

    print(f"  offset: {cache.offset}")
    print(f"  fp_keys shape: {cache._fp_keys.shape}")
    print(f"  packed_values shape: {cache._packed_values.shape}")
    print(f"  decoded_values: {cache._decoded_values}")
    print("  PASSED\n")


def benchmark_asymmetric(verbose=False):
    """Benchmark: asymmetric attention vs decode-then-matmul on 7B-scale config."""
    print("=== Benchmark: Asymmetric vs decode-then-matmul ===")
    print("  Config: B=1, nq=28, nkv=4, dim=128 (7B dense GQA)")

    B, n_q_heads, n_kv_heads, dim = 1, 28, 4, 128
    bits = 4
    seed = 42
    scale = 1.0 / math.sqrt(dim)
    n_warmup = 5
    n_iters = 50

    for T_kv in [128, 512, 1024, 2048, 4096]:
        mx.random.seed(42)
        q = mx.random.normal((B, n_q_heads, 1, dim)).astype(mx.float32)
        k = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float32)
        v = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float32)
        pv, vn = turbo_encode(v, bits=bits, seed=seed)
        mx.eval(pv, vn)

        # Warmup + benchmark: decode-then-matmul
        gqa_factor = n_q_heads // n_kv_heads
        k_expanded = mx.repeat(k, gqa_factor, axis=1)
        pv_expanded = mx.repeat(pv, gqa_factor, axis=1)
        vn_expanded = mx.repeat(vn, gqa_factor, axis=1)
        for _ in range(n_warmup):
            v_dec = turbo_decode(pv_expanded, vn_expanded, dim, bits=bits, seed=seed)
            scores = (q @ k_expanded.transpose(0, 1, 3, 2)) * scale
            weights = mx.softmax(scores, axis=-1)
            ref_out = weights @ v_dec
            mx.eval(ref_out)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            v_dec = turbo_decode(pv_expanded, vn_expanded, dim, bits=bits, seed=seed)
            scores = (q @ k_expanded.transpose(0, 1, 3, 2)) * scale
            weights = mx.softmax(scores, axis=-1)
            ref_out = weights @ v_dec
            mx.eval(ref_out)
        t_ref = (time.perf_counter() - t0) / n_iters * 1000

        # Warmup + benchmark: asymmetric fused
        for _ in range(n_warmup):
            test_out = turbo_asymmetric_attention(
                q, k, pv, vn, dim, bits=bits, seed=seed, scale=scale,
            )
            mx.eval(test_out)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            test_out = turbo_asymmetric_attention(
                q, k, pv, vn, dim, bits=bits, seed=seed, scale=scale,
            )
            mx.eval(test_out)
        t_asym = (time.perf_counter() - t0) / n_iters * 1000

        speedup = t_ref / t_asym if t_asym > 0 else float('inf')
        print(f"  T_kv={T_kv:5d}: decode+matmul={t_ref:.2f}ms, asymmetric={t_asym:.2f}ms, speedup={speedup:.2f}x")

    print()


def main():
    parser = argparse.ArgumentParser(description="Test asymmetric fused attention")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Include perf benchmark")
    args = parser.parse_args()

    if not mx.metal.is_available():
        print("ERROR: Metal GPU not available. These tests require Apple Silicon.")
        sys.exit(1)

    print(f"Metal device: {mx.metal.device_info()['device_name']}\n")

    # Correctness tests
    test_weighted_value_sum_basic(verbose=args.verbose)
    test_weighted_value_sum_gqa(verbose=args.verbose)
    test_asymmetric_attention_basic(verbose=args.verbose)
    test_asymmetric_attention_gqa(verbose=args.verbose)
    test_asymmetric_dims(verbose=args.verbose)
    test_turbo_kv_cache_asymmetric(verbose=args.verbose)
    test_turbo_kv_cache_asymmetric_patched(verbose=args.verbose)

    if args.benchmark:
        benchmark_asymmetric(verbose=args.verbose)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
