#!/usr/bin/env python3
"""Benchmark: thread-per-dim weighted V sum kernel.

Compares the fused kernel against decode-then-matmul reference at various
context lengths. Also benchmarks the kernel in isolation.

Usage:
    python3 tests/bench_weighted_v_sum.py
"""

import math
import time

import mlx.core as mx

from mlx.nn.layers.turbo_kv_cache import (
    turbo_asymmetric_attention,
    turbo_decode,
    turbo_encode,
    turbo_weighted_value_sum,
)


def bench_kernel_isolation():
    """Benchmark just the weighted V sum kernel at various T_kv."""
    print("=== Kernel isolation: turbo_weighted_value_sum ===")
    print("  Config: B=1, n_heads=28 (nkv=4, GQA), dim=128, bits=4\n")

    B, n_q_heads, n_kv_heads, dim = 1, 28, 4, 128
    bits = 4
    seed = 42
    n_warmup = 10
    n_iters = 100

    for T_kv in [64, 256, 1024, 4096, 8192, 16384]:
        mx.random.seed(42)
        v = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float32)
        raw_scores = mx.random.normal((B, n_q_heads, 1, T_kv)).astype(mx.float32)
        weights = mx.softmax(raw_scores, axis=-1)
        pv, vn = turbo_encode(v, bits=bits, seed=seed)
        mx.eval(pv, vn, weights)

        # Warmup
        for _ in range(n_warmup):
            out = turbo_weighted_value_sum(weights, pv, vn, dim, bits=bits, seed=seed)
            mx.eval(out)

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(n_iters):
            out = turbo_weighted_value_sum(weights, pv, vn, dim, bits=bits, seed=seed)
            mx.eval(out)
        t_ms = (time.perf_counter() - t0) / n_iters * 1000

        print(f"  T_kv={T_kv:6d}: {t_ms:.3f} ms")

    print()


def bench_vs_decode_matmul():
    """Benchmark asymmetric attention vs decode-then-matmul."""
    print("=== Asymmetric attention vs decode-then-matmul ===")
    print("  Config: B=1, nq=28, nkv=4, dim=128 (7B dense GQA)\n")

    B, n_q_heads, n_kv_heads, dim = 1, 28, 4, 128
    bits = 4
    seed = 42
    scale = 1.0 / math.sqrt(dim)
    n_warmup = 10
    n_iters = 100

    for T_kv in [1024, 4096, 8192, 16384]:
        mx.random.seed(42)
        q = mx.random.normal((B, n_q_heads, 1, dim)).astype(mx.float32)
        k = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float32)
        v = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float32)
        pv, vn = turbo_encode(v, bits=bits, seed=seed)
        mx.eval(q, k, pv, vn)

        # --- Decode-then-matmul baseline ---
        gqa_factor = n_q_heads // n_kv_heads
        k_expanded = mx.repeat(k, gqa_factor, axis=1)
        pv_expanded = mx.repeat(pv, gqa_factor, axis=1)
        vn_expanded = mx.repeat(vn, gqa_factor, axis=1)
        mx.eval(k_expanded, pv_expanded, vn_expanded)

        for _ in range(n_warmup):
            v_dec = turbo_decode(pv_expanded, vn_expanded, dim, bits=bits, seed=seed)
            scores = (q @ k_expanded.transpose(0, 1, 3, 2)) * scale
            w = mx.softmax(scores, axis=-1)
            ref_out = w @ v_dec
            mx.eval(ref_out)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            v_dec = turbo_decode(pv_expanded, vn_expanded, dim, bits=bits, seed=seed)
            scores = (q @ k_expanded.transpose(0, 1, 3, 2)) * scale
            w = mx.softmax(scores, axis=-1)
            ref_out = w @ v_dec
            mx.eval(ref_out)
        t_ref = (time.perf_counter() - t0) / n_iters * 1000

        # --- Asymmetric fused ---
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
        print(f"  T_kv={T_kv:6d}: decode+matmul={t_ref:.2f}ms  asymmetric={t_asym:.2f}ms  speedup={speedup:.2f}x")

    print()


def bench_correctness_at_scale():
    """Verify correctness at large T_kv values."""
    print("=== Correctness at scale ===\n")

    B, n_q_heads, n_kv_heads, dim = 1, 28, 4, 128
    bits = 4
    seed = 42
    scale = 1.0 / math.sqrt(dim)

    for T_kv in [1024, 4096, 8192, 16384]:
        mx.random.seed(T_kv)
        q = mx.random.normal((B, n_q_heads, 1, dim)).astype(mx.float32)
        k = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float32)
        v = mx.random.normal((B, n_kv_heads, T_kv, dim)).astype(mx.float32)
        pv, vn = turbo_encode(v, bits=bits, seed=seed)

        # Reference
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

        # Fused
        test_out = turbo_asymmetric_attention(
            q, k, pv, vn, dim, bits=bits, seed=seed, scale=scale,
        )
        mx.eval(test_out)

        diff = mx.abs(ref_out - test_out)
        max_diff = mx.max(diff).item()
        mean_diff = mx.mean(diff).item()

        status = "PASS" if max_diff < 0.01 else "FAIL"
        print(f"  T_kv={T_kv:6d}: max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  [{status}]")

    print()


if __name__ == "__main__":
    print(f"Device: {mx.device_info()['device_name']}\n")

    bench_correctness_at_scale()
    bench_kernel_isolation()
    bench_vs_decode_matmul()
