#!/usr/bin/env python3
"""Benchmark: fused single-dispatch vs 4-dispatch asymmetric attention.

Compares:
  1. turbo_asymmetric_attention (4 dispatches: GQA expand, Q×K, softmax, V sum)
  2. turbo_fused_asymmetric_attention_single_dispatch (1 dispatch: everything fused)
  3. Native mx.fast.scaled_dot_product_attention (baseline reference)

Usage:
    python3 benchmarks/bench_fused_asymmetric.py
"""

import time
import mlx.core as mx
import mlx.nn as nn

# Import from turbo_kv_cache
from mlx.nn.layers.turbo_kv_cache import (
    turbo_asymmetric_attention,
    turbo_fused_asymmetric_attention_single_dispatch,
    turbo_encode,
)


def bench(fn, name, warmup=10, iters=100):
    """Benchmark a function, returning median time in ms."""
    # Warmup
    for _ in range(warmup):
        out = fn()
        mx.eval(out)

    # Timed runs
    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[9 * len(times) // 10]
    print(f"  {name:50s}  median={median:.3f}ms  p10={p10:.3f}ms  p90={p90:.3f}ms")
    return median


def main():
    # Typical decode params: Llama-3.1-8B / Qwen2.5-7B style
    B = 1
    n_q_heads = 32
    n_kv_heads = 8
    dim = 128
    bits = 4
    seed = 42

    for T_kv in [128, 256, 512, 1024, 2048]:
        print(f"\n{'='*70}")
        print(f"B={B}, n_q={n_q_heads}, n_kv={n_kv_heads}, dim={dim}, T_kv={T_kv}")
        print(f"{'='*70}")

        # Generate test data
        queries = mx.random.normal((B, n_q_heads, 1, dim))
        keys = mx.random.normal((B, n_kv_heads, T_kv, dim))
        values = mx.random.normal((B, n_kv_heads, T_kv, dim))

        # Encode V with turbo4
        pv, vn = turbo_encode(values.reshape(B * n_kv_heads, T_kv, dim), bits=bits, seed=seed)
        pv = pv.reshape(B, n_kv_heads, T_kv, -1)
        vn = vn.reshape(B, n_kv_heads, T_kv, 1)

        # Materialize everything
        mx.eval(queries, keys, values, pv, vn)

        # 1. Native SDPA (FP16 K and V — the speed target)
        # Expand KV for GQA to match query heads
        gqa = n_q_heads // n_kv_heads
        keys_expanded = mx.repeat(keys, gqa, axis=1)
        values_expanded = mx.repeat(values, gqa, axis=1)
        mx.eval(keys_expanded, values_expanded)

        def native_sdpa():
            return mx.fast.scaled_dot_product_attention(
                queries, keys_expanded, values_expanded, scale=1.0 / (dim ** 0.5),
            )

        t_native = bench(native_sdpa, "Native SDPA (FP16 K+V)")

        # 2. Legacy 4-dispatch asymmetric attention
        def legacy_asymmetric():
            return turbo_asymmetric_attention(
                queries, keys, pv, vn, dim=dim, bits=bits, seed=seed,
            )

        t_legacy = bench(legacy_asymmetric, "turbo_asymmetric_attention (4 dispatches)")

        # 3. Fused single-dispatch asymmetric attention
        def fused_asymmetric():
            return turbo_fused_asymmetric_attention_single_dispatch(
                queries, keys, pv, vn, dim=dim, bits=bits, seed=seed,
            )

        t_fused = bench(fused_asymmetric, "turbo_fused_single_dispatch (1 dispatch)")

        # Correctness check: compare outputs
        out_legacy = legacy_asymmetric()
        out_fused = fused_asymmetric()
        mx.eval(out_legacy, out_fused)
        max_diff = mx.max(mx.abs(out_legacy - out_fused)).item()
        mean_diff = mx.mean(mx.abs(out_legacy - out_fused)).item()
        print(f"\n  Correctness: max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
        print(f"  Speedup vs legacy: {t_legacy / t_fused:.2f}x")
        print(f"  Speedup vs native: {t_native / t_fused:.2f}x  (target: >=1.0x)")


if __name__ == "__main__":
    main()
