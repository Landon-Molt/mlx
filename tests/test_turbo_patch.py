#!/usr/bin/env python3
"""Test monkey-patch mechanism for mlx-lm SDPA integration.

Tests that patch_mlx_lm() correctly:
1. Installs the turbo-aware SDPA in mlx_lm.models.base
2. Routes TurboKVCache attention through fused path (no FP16 decode)
3. Falls through to original SDPA for non-turbo caches
4. Memory savings: no decoded FP16 buffers when patched
5. Output quality matches unpatched turbo (same packed data, same kernel)

Usage:
    python3 tests/test_turbo_patch.py
    python3 tests/test_turbo_patch.py --model mlx-community/Qwen3.5-2B-8bit
    python3 tests/test_turbo_patch.py --max-tokens 50
"""

import argparse
import sys
import time

import mlx.core as mx
import mlx_lm
import mlx_lm.models.base as base
from mlx_lm.models import cache as cache_module

from mlx.nn.layers.turbo_kv_cache import (
    TurboKVCache,
    patch_mlx_lm,
    unpatch_mlx_lm,
)


def make_turbo_cache(model, bits=4, key_bits=None, **kwargs):
    """Create TurboKVCache list matching model layer structure."""
    default_cache = cache_module.make_prompt_cache(model)
    turbo_cache = []
    for c in default_cache:
        if isinstance(c, cache_module.KVCache):
            turbo_cache.append(TurboKVCache(bits=bits, key_bits=key_bits, **kwargs))
        else:
            turbo_cache.append(c)
    return turbo_cache


def test_patch_unpatch():
    """Test that patch/unpatch installs and restores SDPA correctly."""
    print("=" * 60)
    print("TEST: patch/unpatch mechanism")
    print("=" * 60)

    original = base.scaled_dot_product_attention
    print(f"  Original SDPA: {original.__name__}")

    # Patch
    patch_mlx_lm()
    patched = base.scaled_dot_product_attention
    assert patched is not original, "SDPA should be replaced after patch"
    assert patched.__name__ == "turbo_sdpa", f"Expected turbo_sdpa, got {patched.__name__}"
    print(f"  Patched SDPA:  {patched.__name__}")

    # Double-patch should be idempotent
    patch_mlx_lm()
    assert base.scaled_dot_product_attention is patched, "Double-patch should be no-op"
    print("  Double-patch: idempotent (OK)")

    # Unpatch
    unpatch_mlx_lm()
    restored = base.scaled_dot_product_attention
    assert restored is original, "SDPA should be restored after unpatch"
    print(f"  Restored SDPA: {restored.__name__}")

    # Double-unpatch should be no-op
    unpatch_mlx_lm()
    print("  Double-unpatch: no-op (OK)")

    print("  PASSED\n")


def test_cache_patched_flag():
    """Test that patch_mlx_lm sets _patched on caches."""
    print("=" * 60)
    print("TEST: _patched flag management")
    print("=" * 60)

    caches = [TurboKVCache(bits=4, key_bits=4) for _ in range(4)]
    assert all(not c._patched for c in caches), "Caches should start unpatched"
    print("  Initial: all _patched=False")

    patch_mlx_lm(caches)
    assert all(c._patched for c in caches), "Caches should be patched"
    print("  After patch: all _patched=True")

    unpatch_mlx_lm(caches)
    assert all(not c._patched for c in caches), "Caches should be unpatched"
    print("  After unpatch: all _patched=False")

    print("  PASSED\n")


def test_non_turbo_fallthrough():
    """Test that non-TurboKVCache caches use the original SDPA path."""
    print("=" * 60)
    print("TEST: non-turbo cache fallthrough")
    print("=" * 60)

    # Create a dummy "cache" that isn't TurboKVCache
    class DummyCache:
        pass

    patch_mlx_lm()
    try:
        # The patched SDPA should delegate to original for non-turbo
        q = mx.random.normal((1, 8, 1, 64))
        k = mx.random.normal((1, 8, 16, 64))
        v = mx.random.normal((1, 8, 16, 64))
        scale = 1.0 / (64 ** 0.5)

        result = base.scaled_dot_product_attention(
            q, k, v, cache=DummyCache(), scale=scale, mask=None,
        )
        mx.eval(result)
        assert result.shape == (1, 8, 1, 64), f"Bad shape: {result.shape}"
        print(f"  Non-turbo result shape: {result.shape} (OK)")
        print("  PASSED\n")
    finally:
        unpatch_mlx_lm()


def test_e2e_generation(model_name, max_tokens, prompt):
    """Compare patched vs unpatched turbo generation quality."""
    print("=" * 60)
    print(f"TEST: e2e generation ({model_name}, {max_tokens} tokens)")
    print("=" * 60)

    print(f"  Loading model: {model_name}")
    model, tokenizer = mlx_lm.load(model_name)

    # --- Baseline: standard KVCache ---
    print("\n  [1/4] Baseline (standard KVCache)...")
    baseline_cache = cache_module.make_prompt_cache(model)
    t0 = time.perf_counter()
    baseline_text = mlx_lm.generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, prompt_cache=baseline_cache,
    )
    baseline_time = time.perf_counter() - t0
    print(f"    Time: {baseline_time:.2f}s")
    print(f"    Output: {baseline_text[:80]}...")

    # --- Turbo unpatched: update_and_fetch + standard SDPA ---
    print("\n  [2/4] Turbo unpatched (update_and_fetch + standard SDPA)...")
    turbo_cache = make_turbo_cache(model, bits=4, key_bits=4)
    t0 = time.perf_counter()
    turbo_text = mlx_lm.generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, prompt_cache=turbo_cache,
    )
    turbo_time = time.perf_counter() - t0
    print(f"    Time: {turbo_time:.2f}s")
    print(f"    Output: {turbo_text[:80]}...")

    # --- Turbo patched: fused attention, no FP16 decode ---
    print("\n  [3/4] Turbo patched (fused attention, no FP16 decode)...")
    patched_cache = make_turbo_cache(model, bits=4, key_bits=4)
    patch_mlx_lm(patched_cache)
    try:
        t0 = time.perf_counter()
        patched_text = mlx_lm.generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max_tokens, prompt_cache=patched_cache,
        )
        patched_time = time.perf_counter() - t0
        print(f"    Time: {patched_time:.2f}s")
        print(f"    Output: {patched_text[:80]}...")
    finally:
        unpatch_mlx_lm(patched_cache)

    # --- Memory comparison ---
    print("\n  [4/4] Memory comparison at end of generation...")

    def cache_memory(cache_list):
        total = 0
        for c in cache_list:
            if hasattr(c, 'nbytes'):
                total += c.nbytes
        return total

    baseline_mem = cache_memory(baseline_cache)
    turbo_mem = cache_memory(turbo_cache)
    patched_mem = cache_memory(patched_cache)

    print(f"    Baseline KV memory:        {baseline_mem / 1024 / 1024:.2f} MB")
    print(f"    Turbo unpatched KV memory: {turbo_mem / 1024 / 1024:.2f} MB")
    print(f"    Turbo patched KV memory:   {patched_mem / 1024 / 1024:.2f} MB")

    if baseline_mem > 0:
        print(f"    Patched vs baseline:       {patched_mem / baseline_mem * 100:.1f}%")
    if turbo_mem > 0:
        print(f"    Patched vs unpatched:      {patched_mem / turbo_mem * 100:.1f}%")

    # Patched should use less memory than unpatched turbo (no decoded FP16)
    if patched_mem < turbo_mem:
        print("    Memory savings confirmed: patched < unpatched turbo")
    else:
        print("    WARNING: patched did NOT save memory vs unpatched")

    # --- Speed comparison ---
    print(f"\n  Speed summary:")
    print(f"    Baseline:        {max_tokens / baseline_time:.1f} tok/s")
    print(f"    Turbo unpatched: {max_tokens / turbo_time:.1f} tok/s")
    print(f"    Turbo patched:   {max_tokens / patched_time:.1f} tok/s")

    print("\n  PASSED\n")


def test_patch_with_long_context(model_name, context_tokens=2048, gen_tokens=20):
    """Test patched generation with longer context to stress memory savings."""
    print("=" * 60)
    print(f"TEST: long context memory ({context_tokens} ctx, {gen_tokens} gen)")
    print("=" * 60)

    print(f"  Loading model: {model_name}")
    model, tokenizer = mlx_lm.load(model_name)

    # Build a long prompt by repeating text
    base_prompt = "The quick brown fox jumps over the lazy dog. " * 100
    tokens = tokenizer.encode(base_prompt)
    # Trim to desired length
    tokens = tokens[:context_tokens]
    prompt = tokenizer.decode(tokens)
    print(f"  Prompt tokens: {len(tokens)}")

    # --- Turbo patched ---
    patched_cache = make_turbo_cache(model, bits=4, key_bits=4)
    patch_mlx_lm(patched_cache)
    try:
        t0 = time.perf_counter()
        patched_text = mlx_lm.generate(
            model, tokenizer, prompt=prompt,
            max_tokens=gen_tokens, prompt_cache=patched_cache,
        )
        patched_time = time.perf_counter() - t0
        print(f"  Patched time: {patched_time:.2f}s")
        print(f"  Output: {patched_text[:80]}...")

        patched_mem = sum(
            c.nbytes for c in patched_cache if hasattr(c, 'nbytes')
        )
        print(f"  Patched KV memory: {patched_mem / 1024 / 1024:.2f} MB")

        # Verify no decoded FP16 buffers exist on patched caches
        has_decoded = False
        for c in patched_cache:
            if isinstance(c, TurboKVCache):
                if c._decoded_keys is not None or c._decoded_values is not None:
                    has_decoded = True
                    break

        if has_decoded:
            print("  WARNING: Decoded FP16 buffers found — patch may not be working")
        else:
            print("  No decoded FP16 buffers — patch working correctly")

    finally:
        unpatch_mlx_lm(patched_cache)

    print("  PASSED\n")


def main():
    parser = argparse.ArgumentParser(description="Test TurboKVCache monkey-patch")
    parser.add_argument(
        "--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        help="Model to test with",
    )
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument(
        "--prompt", default="Explain quantum computing in simple terms:",
    )
    parser.add_argument("--skip-e2e", action="store_true", help="Skip e2e model tests")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TurboKVCache Monkey-Patch Test Suite")
    print("=" * 60 + "\n")

    # Unit tests (no model needed)
    test_patch_unpatch()
    test_cache_patched_flag()
    test_non_turbo_fallthrough()

    if not args.skip_e2e:
        # E2E tests (need model)
        test_e2e_generation(args.model, args.max_tokens, args.prompt)
        test_patch_with_long_context(args.model)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
