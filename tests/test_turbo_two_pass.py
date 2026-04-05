#!/usr/bin/env python3
"""Test & benchmark: Two-pass fused asymmetric attention (TurboFlash, B=64).

Validates correctness against single-pass kernel and benchmarks both approaches
plus the 4-dispatch legacy path.

Usage:
    python3 tests/test_turbo_two_pass.py              # correctness only
    python3 tests/test_turbo_two_pass.py -v            # verbose
    python3 tests/test_turbo_two_pass.py --benchmark   # include perf comparison
"""

import argparse
import math
import sys
import time

import mlx.core as mx

from mlx.nn.layers.turbo_kv_cache import (
    turbo_asymmetric_attention,
    turbo_encode,
    turbo_fused_asymmetric_attention_single_dispatch,
    turbo_two_pass_asymmetric_attention,
    _get_codebook,
    _sign_flip_vector,
    _sign_flip_vector2,
)


def _make_test_data(B=1, n_q_heads=32, n_kv_heads=8, T_kv=512, dim=128, bits=4, seed=42):
    """Generate synthetic test data for asymmetric attention."""
    mx.random.seed(0)
    queries = mx.random.normal((B, n_q_heads, 1, dim)).astype(mx.float32)
    fp_keys = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float32)
    values = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float32)
    packed_v, v_norms = turbo_encode(values, bits=bits, seed=seed)
    mx.eval(queries, fp_keys, packed_v, v_norms)
    return queries, fp_keys, packed_v, v_norms


def test_correctness_vs_single_pass(verbose=False):
    """Verify two-pass output matches single-pass within tolerance."""
    print("=== Test: Two-pass vs single-pass correctness ===")

    configs = [
        # (B, n_q_heads, n_kv_heads, T_kv, dim)
        (1, 4, 4, 64, 128),     # exactly 1 block
        (1, 4, 4, 128, 128),    # exactly 2 blocks
        (1, 4, 4, 200, 128),    # partial last block (200 / 64 = 3.125 blocks)
        (1, 32, 8, 512, 128),   # GQA, 8 blocks — typical decode
        (1, 32, 8, 1024, 128),  # 16 blocks — longer context
        (1, 8, 8, 63, 128),     # edge: less than 1 full block
        (1, 4, 4, 1, 128),      # edge: single token
    ]

    all_passed = True
    for B, nq, nkv, T, dim in configs:
        label = f"B={B}, nq={nq}, nkv={nkv}, T={T}, dim={dim}"
        queries, fp_keys, packed_v, v_norms = _make_test_data(
            B=B, n_q_heads=nq, n_kv_heads=nkv, T_kv=T, dim=dim
        )

        # Single-pass reference
        ref = turbo_fused_asymmetric_attention_single_dispatch(
            queries, fp_keys, packed_v, v_norms, dim=dim
        )
        mx.eval(ref)

        # Two-pass
        test = turbo_two_pass_asymmetric_attention(
            queries, fp_keys, packed_v, v_norms, dim=dim
        )
        mx.eval(test)

        diff = mx.abs(ref - test)
        max_diff = mx.max(diff).item()
        mean_diff = mx.mean(diff).item()

        # Tolerance: turbo4 quantization means ~1e-3 is expected noise floor.
        # Two-pass vs single-pass should match closer since they use the same
        # V data — only the softmax ordering differs (online vs block-merge).
        passed = max_diff < 0.01
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] {label}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        if verbose:
            print(f"         ref[:4]  = {ref[0, 0, 0, :4].tolist()}")
            print(f"         test[:4] = {test[0, 0, 0, :4].tolist()}")

    return all_passed


def test_correctness_vs_4dispatch(verbose=False):
    """Verify two-pass matches the legacy 4-dispatch path."""
    print("=== Test: Two-pass vs 4-dispatch correctness ===")

    B, nq, nkv, T, dim = 1, 32, 8, 512, 128
    queries, fp_keys, packed_v, v_norms = _make_test_data(
        B=B, n_q_heads=nq, n_kv_heads=nkv, T_kv=T, dim=dim
    )

    # 4-dispatch reference
    ref = turbo_asymmetric_attention(
        queries, fp_keys, packed_v, v_norms, dim=dim
    )
    mx.eval(ref)

    # Two-pass
    test = turbo_two_pass_asymmetric_attention(
        queries, fp_keys, packed_v, v_norms, dim=dim
    )
    mx.eval(test)

    diff = mx.abs(ref - test)
    max_diff = mx.max(diff).item()
    mean_diff = mx.mean(diff).item()

    passed = max_diff < 0.01
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if verbose:
        print(f"         ref[:4]  = {ref[0, 0, 0, :4].tolist()}")
        print(f"         test[:4] = {test[0, 0, 0, :4].tolist()}")

    return passed


def benchmark_all(n_warmup=5, n_iter=50):
    """Benchmark two-pass vs single-pass vs 4-dispatch at various T_kv."""
    print("\n=== Benchmark: Two-pass vs Single-pass vs 4-dispatch ===")

    B, nq, nkv, dim = 1, 32, 8, 128
    t_kv_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    print(f"{'T_kv':>8} | {'4-dispatch':>12} | {'single-pass':>12} | {'two-pass':>12} | {'speedup':>8}")
    print("-" * 72)

    for T in t_kv_sizes:
        queries, fp_keys, packed_v, v_norms = _make_test_data(
            B=B, n_q_heads=nq, n_kv_heads=nkv, T_kv=T, dim=dim
        )

        timings = {}

        # --- 4-dispatch ---
        for _ in range(n_warmup):
            out = turbo_asymmetric_attention(queries, fp_keys, packed_v, v_norms, dim=dim)
            mx.eval(out)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            out = turbo_asymmetric_attention(queries, fp_keys, packed_v, v_norms, dim=dim)
            mx.eval(out)
        timings['4-dispatch'] = (time.perf_counter() - t0) / n_iter * 1000

        # --- Single-pass ---
        for _ in range(n_warmup):
            out = turbo_fused_asymmetric_attention_single_dispatch(queries, fp_keys, packed_v, v_norms, dim=dim)
            mx.eval(out)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            out = turbo_fused_asymmetric_attention_single_dispatch(queries, fp_keys, packed_v, v_norms, dim=dim)
            mx.eval(out)
        timings['single-pass'] = (time.perf_counter() - t0) / n_iter * 1000

        # --- Two-pass ---
        for _ in range(n_warmup):
            out = turbo_two_pass_asymmetric_attention(queries, fp_keys, packed_v, v_norms, dim=dim)
            mx.eval(out)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            out = turbo_two_pass_asymmetric_attention(queries, fp_keys, packed_v, v_norms, dim=dim)
            mx.eval(out)
        timings['two-pass'] = (time.perf_counter() - t0) / n_iter * 1000

        speedup = timings['single-pass'] / timings['two-pass']

        print(
            f"{T:>8} | {timings['4-dispatch']:>10.3f}ms | {timings['single-pass']:>10.3f}ms | "
            f"{timings['two-pass']:>10.3f}ms | {speedup:>6.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(description="Test two-pass TurboFlash attention")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    args = parser.parse_args()

    passed = True
    passed &= test_correctness_vs_single_pass(verbose=args.verbose)
    passed &= test_correctness_vs_4dispatch(verbose=args.verbose)

    if args.benchmark:
        benchmark_all()

    if passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
