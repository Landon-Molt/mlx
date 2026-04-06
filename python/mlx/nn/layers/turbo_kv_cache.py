# Copyright © 2025 Apple Inc.
# TurboQuant KV Cache Quantization for MLX
#
# Implements the TurboQuant algorithm (Google Research, 2024) for KV cache
# compression in transformer models. Uses Subsampled Randomized Hadamard
# Transform (SRHT) + Lloyd-Max quantization on the unit sphere.
#
# Reference: "TurboQuant: Online Vector Quantization for KV Cache Compression"
# Implementation: Tom Turney (TurboQuant+)

import math
import os
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
from mlx.nn.layers.base import Module


# ---------------------------------------------------------------------------
# Beta distribution centroids for unit-sphere-normalized coordinates
# Pre-computed for common (bits, dim) combos. These are the correct
# distribution for coordinates after WHT rotation on the unit sphere
# (Eric's derivation). Proven +47% PPL vs N(0,1) at short context (128 tok)
# on Qwen3.5-2B A/B test.
#
# Fallback: N(0,1) Lloyd-Max centroids scaled by 1/sqrt(d) for unknown dims.
# Toggle: set TURBO_USE_N01_CENTROIDS=1 to force N(0,1) for A/B testing.
# ---------------------------------------------------------------------------

# Beta distribution centroids keyed by (bits, dim) — already scaled for dim.
# Do NOT multiply by 1/sqrt(d) again.
_BETA_CENTROIDS: Dict[Tuple[int, int], List[float]] = {
    (4, 64): [
        -0.32913971, -0.25096416, -0.19681059, -0.15295772,
        -0.11478586, -0.08000945, -0.04726735, -0.01563822,
        0.01563822, 0.04723797, 0.07994876, 0.11472529,
        0.15289739, 0.19675052, 0.25090477, 0.32908401,
    ],
    (4, 128): [
        -0.23639172, -0.17934021, -0.14023653, -0.10881814,
        -0.08157559, -0.05678632, -0.03350975, -0.01108178,
        0.01108178, 0.03350975, 0.05678631, 0.08157560,
        0.10881804, 0.14023650, 0.17934017, 0.23639278,
    ],
    (4, 32): [
        -0.45436703, -0.35035647, -0.27666714, -0.21609482,
        -0.16273504, -0.11373488, -0.06734953, -0.02223680,
        0.02223680, 0.06734953, 0.11373488, 0.16273504,
        0.21609482, 0.27666714, 0.35035647, 0.45436703,
    ],
    (4, 256): [
        -0.16852295, -0.12754069, -0.09961203, -0.07719406,
        -0.05781249, -0.04021866, -0.02370371, -0.00783269,
        0.00783269, 0.02370371, 0.04021868, 0.05781246,
        0.07719407, 0.09961203, 0.12754090, 0.16852276,
    ],
    (3, 128): [
        -0.18828832, -0.11801215, -0.06648001, -0.02156330,
        0.02156329, 0.06648005, 0.11801218, 0.18828897,
    ],
    (2, 128): [
        -0.13302007, -0.03998107, 0.03998102, 0.13302033,
    ],
}

# N(0,1) Lloyd-Max centroids (fallback for unknown dims, scaled by 1/sqrt(d))
_LLOYD_MAX_CENTROIDS = {
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [
        -2.1520, -1.3440, -0.7560, -0.2451,
        0.2451, 0.7560, 1.3440, 2.1520,
    ],
    4: [
        -2.7326, -2.0690, -1.6180, -1.2562,
        -0.9423, -0.6568, -0.3881, -0.1284,
        0.1284, 0.3881, 0.6568, 0.9423,
        1.2562, 1.6180, 2.0690, 2.7326,
    ],
}


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    if n & (n - 1) == 0:
        return n
    return 1 << (n - 1).bit_length()


class TurboQuantCodebook:
    """Pre-computed codebook for TurboQuant.

    Uses Beta distribution centroids (proven better) when available for the
    given (bits, dim) pair. Falls back to N(0,1) Lloyd-Max centroids scaled
    by 1/sqrt(dim) for unknown dims.

    Beta centroids are already scaled for their dim — no 1/sqrt(d) applied.
    Set TURBO_USE_N01_CENTROIDS=1 to force N(0,1) fallback for A/B testing.

    Boundaries are midpoints between adjacent centroids.

    **Design choice — pure centroid quantization, no residual correction:**
    QJL (random Gaussian projection for residual) was tested and found actively
    harmful for autoregressive generation (turbo4-resurrection.md). Variance from
    the random projection compounds across decode steps. 16 centroids (4-bit)
    without correction outperform 8 centroids (3-bit) with correction.

    For non-power-of-2 dims (e.g. Qwen3-4B d=80), the codebook is built for
    the padded_dim (next power of 2). The encode/decode functions handle
    zero-padding input to padded_dim and stripping back to original dim.

    Args:
        bits (int): Quantization bit-width (2, 3, or 4).
        dim (int): Head dimension (e.g. 64, 80, 128, 256). Non-power-of-2
            dims are supported via zero-padding to next power of 2.

    Example:
        >>> cb = TurboQuantCodebook(bits=4, dim=128)
        >>> cb.centroids.shape  # (16,)
        >>> cb = TurboQuantCodebook(bits=4, dim=80)  # Qwen3-4B
        >>> cb.padded_dim  # 128
    """

    def __init__(self, bits: int, dim: int):
        if bits not in _LLOYD_MAX_CENTROIDS:
            raise ValueError(f"Unsupported bits={bits}. Must be 2, 3, or 4.")

        self.bits = bits
        self.dim = dim
        self.padded_dim = _next_power_of_2(dim)
        self.n_levels = 1 << bits

        # Centroids are computed for padded_dim since WHT operates on that size
        effective_dim = self.padded_dim

        force_n01 = os.environ.get("TURBO_USE_N01_CENTROIDS", "0") == "1"
        beta_key = (bits, effective_dim)

        if not force_n01 and beta_key in _BETA_CENTROIDS:
            # Beta distribution centroids — already scaled for this dim
            raw = _BETA_CENTROIDS[beta_key]
            self.centroids = mx.array(raw, dtype=mx.float32)
            self._centroid_source = "beta"
        else:
            # Fallback: N(0,1) Lloyd-Max scaled by 1/sqrt(effective_dim)
            scale = 1.0 / math.sqrt(effective_dim)
            raw = _LLOYD_MAX_CENTROIDS[bits]
            self.centroids = mx.array([c * scale for c in raw], dtype=mx.float32)
            self._centroid_source = "n01"

        # Boundaries = midpoints between adjacent centroids
        c = self.centroids
        self.boundaries = (c[:-1] + c[1:]) / 2.0


def _get_codebook(bits: int, dim: int) -> TurboQuantCodebook:
    """Get or create a codebook. Cached per (bits, dim) pair."""
    # TODO: Add proper LRU cache if this becomes a bottleneck
    return TurboQuantCodebook(bits, dim)


# ---------------------------------------------------------------------------
# TODO: InnerQ per-channel scale precision correction (Item 7)
# llama.cpp's CUDA backend has turbo-innerq.cuh: per-channel equalization
# that calibrates K² statistics over tokens and applies scale correction
# before quantization. Initialized to identity (all 1.0), activated via
# TURBO_INNERQ_TOKENS env. This is CUDA-only in llama.cpp (not in their
# Metal path either), so not applicable to MLX currently. If MLX ever gets
# a CUDA backend, this could be ported. For Metal, the Beta-distribution
# centroids already handle the distribution well.
# ---------------------------------------------------------------------------
# Sign-flip PRNG — deterministic random signs from seed
# ---------------------------------------------------------------------------


def _sign_flip_vector(dim: int, seed: int) -> mx.array:
    """Generate a deterministic {-1, +1} sign vector from seed.

    Uses mx.random with a fixed key so the same seed always produces
    the same sign pattern. This is the 'S1' (pre-WHT) in the full SRHT = S2·H·S1.

    Args:
        dim: Length of the sign vector.
        seed: Random seed for reproducibility.

    Returns:
        mx.array of shape (dim,) with values in {-1, +1}.
    """
    key = mx.array([seed, 0], dtype=mx.uint32)
    # Uniform [0,1) → threshold at 0.5 → {-1, +1}
    r = mx.random.uniform(shape=(dim,), key=key)
    signs = mx.where(r < 0.5, mx.array(-1.0), mx.array(1.0))
    return signs


def _sign_flip_vector2(dim: int, seed: int) -> mx.array:
    """Generate a SECOND independent {-1, +1} sign vector for post-WHT flip.

    llama.cpp SRHT uses TWO sign arrays: x_rot = signs2 * WHT(signs1 * x) / sqrt(n).
    The post-WHT sign flip (signs2) breaks any remaining structure after the
    Hadamard transform, providing better decorrelation for quantization.

    Uses seed + 1000 to generate an independent sign pattern from _sign_flip_vector.

    Args:
        dim: Length of the sign vector.
        seed: Random seed (will be offset by +1000 for independence).

    Returns:
        mx.array of shape (dim,) with values in {-1, +1}.
    """
    key = mx.array([seed + 1000, 0], dtype=mx.uint32)
    r = mx.random.uniform(shape=(dim,), key=key)
    signs = mx.where(r < 0.5, mx.array(-1.0), mx.array(1.0))
    return signs


# ---------------------------------------------------------------------------
# Core encode / decode
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Compiled encode pipeline cache — keyed by (bits, dim, seed) to avoid
# recompilation. mx.compile fuses the MLX graph into a single Metal dispatch,
# eliminating kernel launch overhead between normalize → sign_flip → WHT →
# quantize → pack. The sign_flip multiply is absorbed into the WHT launch.
# ---------------------------------------------------------------------------
_compiled_encode_cache: Dict[Tuple[int, int, int], object] = {}


def _make_compiled_encode(bits: int, dim: int, seed: int, block_size: int = 0):
    """Create a compiled (fused) encode function for given (bits, dim, seed, block_size).

    The compiled function fuses: norm → normalize → sign_flip → hadamard →
    boundary_quantize → pack into a single MLX graph evaluation. This
    eliminates per-op kernel launch overhead (5-7 Metal dispatches → 1).

    Falls back gracefully if mx.compile is not available or fails.

    Args:
        bits: Quantization bit-width.
        dim: Head dimension.
        seed: SRHT random seed.
        block_size: WHT block size. 0 or dim = full WHT. 32 = blocked WHT
            (reshape to blocks of 32, WHT per block). Must divide dim evenly.

    Returns:
        Compiled callable that takes x and returns (packed, norms).
    """
    # Resolve block_size: 0 means full dim
    bs = dim if (block_size == 0 or block_size >= dim) else block_size

    cb = _get_codebook(bits, bs)
    signs1 = _sign_flip_vector(bs, seed)
    signs2 = _sign_flip_vector2(bs, seed)
    boundaries = cb.boundaries
    n_blocks = dim // bs

    if bs < dim:
        # Blocked WHT path
        def _encode_inner(x):
            # 1. Norms + normalize
            norms = mx.linalg.norm(x, axis=-1, keepdims=True)
            safe_norms = mx.maximum(norms, mx.array(1e-10))
            x_unit = x / safe_norms

            # 2. Reshape (..., dim) → (..., n_blocks, block_size)
            x_blocked = x_unit.reshape(*x_unit.shape[:-1], n_blocks, bs)

            # 3. Dual sign flip + WHT per block
            x_wht = mx.hadamard_transform(x_blocked * signs1) * signs2

            # 4. Reshape back to (..., dim)
            x_rotated = x_wht.reshape(*x_unit.shape)

            # 5. Boundary quantize
            indices = mx.sum(
                mx.expand_dims(x_rotated, axis=-1) > mx.expand_dims(boundaries, axis=0),
                axis=-1,
            ).astype(mx.uint32)

            # 6. Pack into uint32
            packed = _pack_indices(indices, bits)

            return packed, norms
    else:
        # Full WHT path (original)
        def _encode_inner(x):
            # 1. Norms + normalize
            norms = mx.linalg.norm(x, axis=-1, keepdims=True)
            safe_norms = mx.maximum(norms, mx.array(1e-10))
            x_unit = x / safe_norms

            # 2. Dual sign flip + WHT: x_rot = signs2 * WHT(signs1 * x) / sqrt(n)
            x_rotated = mx.hadamard_transform(x_unit * signs1) * signs2

            # 3. Boundary quantize
            indices = mx.sum(
                mx.expand_dims(x_rotated, axis=-1) > mx.expand_dims(boundaries, axis=0),
                axis=-1,
            ).astype(mx.uint32)

            # 4. Pack into uint32
            packed = _pack_indices(indices, bits)

            return packed, norms

    try:
        compiled_fn = mx.compile(_encode_inner)
        return compiled_fn
    except Exception:
        # mx.compile may not be available in all MLX versions
        # Fall back to uncompiled
        return _encode_inner


def turbo_encode(
    x: mx.array,
    bits: int = 4,
    seed: int = 42,
    block_size: int = 0,
) -> Tuple[mx.array, mx.array]:
    """Encode vectors using TurboQuant (SRHT + Lloyd-Max quantization).

    Applies: normalize → sign_flip → hadamard_transform �� boundary quantize → pack.

    Uses ``mx.compile()`` to fuse the full pipeline into a single Metal dispatch,
    eliminating per-op kernel launch overhead. The compiled function is cached
    per (bits, dim, seed) triple.

    Args:
        x: Input tensor of shape (..., dim). dim must be power of 2.
        bits: Quantization bit-width (2, 3, or 4). Default: 4.
        seed: Random seed for the sign-flip diagonal. Default: 42.

    Returns:
        Tuple of:
            - packed_indices: uint32 tensor with packed quantization indices.
              Shape (..., dim * bits / 32).
            - norms: float32 tensor of per-vector L2 norms. Shape (..., 1).

    Example:
        >>> x = mx.random.normal((4, 8, 128))  # (batch, seq, dim)
        >>> packed, norms = turbo_encode(x, bits=4, seed=42)
        >>> packed.shape  # (4, 8, 16) — 128 * 4 / 32 = 16 uint32s
        >>> norms.shape   # (4, 8, 1)
    """
    dim = x.shape[-1]
    padded_dim = _next_power_of_2(dim)

    # Pad non-power-of-2 dims with zeros before WHT (e.g. Qwen3-4B d=80 → 128)
    if padded_dim != dim:
        pad_width = padded_dim - dim
        padding = mx.zeros((*x.shape[:-1], pad_width), dtype=x.dtype)
        x = mx.concatenate([x, padding], axis=-1)

    # block_size operates on the padded dim
    bs = padded_dim if (block_size == 0 or block_size >= padded_dim) else block_size

    # Try fused Metal kernel first (fastest path — single Metal dispatch)
    # Toggle: set TURBO_DISABLE_FUSED_KERNEL=1 to force mx.compile path
    # Fused kernel supports block_size via params[3]
    use_fused = (
        bits == 4
        and padded_dim <= 256
        and (bs & (bs - 1)) == 0
        and bs == padded_dim  # fused kernel only supports full-dim WHT
        and os.environ.get("TURBO_DISABLE_FUSED_KERNEL", "0") != "1"
    )
    if use_fused:
        return turbo_encode_fused(x, bits=bits, seed=seed, block_size=block_size)

    # Fallback: mx.compile path
    cache_key = (bits, padded_dim, seed, bs)
    if cache_key not in _compiled_encode_cache:
        _compiled_encode_cache[cache_key] = _make_compiled_encode(bits, padded_dim, seed, block_size=block_size)

    return _compiled_encode_cache[cache_key](x)


def turbo_encode_uncompiled(
    x: mx.array,
    bits: int = 4,
    seed: int = 42,
    block_size: int = 0,
) -> Tuple[mx.array, mx.array]:
    """Uncompiled encode path — for benchmarking against compiled version.

    Identical to the original turbo_encode without mx.compile fusion.
    Used as baseline to measure the compile optimization speedup.

    Args:
        x: Input tensor of shape (..., dim). dim must be power of 2.
        bits: Quantization bit-width (2, 3, or 4). Default: 4.
        seed: Random seed for the sign-flip diagonal. Default: 42.
        block_size: WHT block size. 0 or dim = full WHT, 32 = blocked WHT.

    Returns:
        Same as turbo_encode: (packed_indices, norms).
    """
    dim = x.shape[-1]
    padded_dim = _next_power_of_2(dim)

    # Pad non-power-of-2 dims with zeros
    if padded_dim != dim:
        pad_width = padded_dim - dim
        padding = mx.zeros((*x.shape[:-1], pad_width), dtype=x.dtype)
        x = mx.concatenate([x, padding], axis=-1)

    bs = padded_dim if (block_size == 0 or block_size >= padded_dim) else block_size
    cb = _get_codebook(bits, bs)

    # 1. Extract norms and normalize to unit sphere
    norms = mx.linalg.norm(x, axis=-1, keepdims=True)
    safe_norms = mx.maximum(norms, mx.array(1e-10))
    x_unit = x / safe_norms

    # 2. Apply dual sign flip + WHT
    signs1 = _sign_flip_vector(bs, seed)
    signs2 = _sign_flip_vector2(bs, seed)

    if bs < padded_dim:
        # Blocked WHT: reshape → signs → WHT per block → signs → reshape back
        n_blocks = padded_dim // bs
        x_blocked = x_unit.reshape(*x_unit.shape[:-1], n_blocks, bs)
        x_wht = mx.hadamard_transform(x_blocked * signs1) * signs2
        x_rotated = x_wht.reshape(*x_unit.shape)
    else:
        # Full WHT path
        x_flipped = x_unit * signs1
        x_rotated = mx.hadamard_transform(x_flipped) * signs2

    # 3. Boundary quantize → indices (pure centroid, NO residual correction)
    #
    # CONFIRMED: No QJL (random Gaussian projection) residual correction.
    # Pure centroid quantization without correction is strictly better for inference.
    boundaries = cb.boundaries  # (n_levels - 1,)
    indices = mx.sum(
        mx.expand_dims(x_rotated, axis=-1) > mx.expand_dims(boundaries, axis=0),
        axis=-1,
    ).astype(mx.uint32)

    # 4. Pack indices into uint32
    packed = _pack_indices(indices, bits)

    return packed, norms


def turbo_decode(
    packed_indices: mx.array,
    norms: mx.array,
    dim: int,
    bits: int = 4,
    seed: int = 42,
    block_size: int = 0,
) -> mx.array:
    """Decode TurboQuant-compressed vectors back to full precision.

    Applies: unpack → codebook lookup → inverse_hadamard → inverse_sign_flip → scale.

    Args:
        packed_indices: uint32 tensor from turbo_encode. Shape (..., dim * bits / 32).
        norms: float32 norms from turbo_encode. Shape (..., 1).
        dim: Original head dimension.
        bits: Quantization bit-width (2, 3, or 4). Default: 4.
        seed: Must match the seed used in turbo_encode. Default: 42.

    Returns:
        Reconstructed tensor of shape (..., dim).

    Example:
        >>> packed, norms = turbo_encode(x, bits=4, seed=42)
        >>> x_hat = turbo_decode(packed, norms, dim=128, bits=4, seed=42)
        >>> x_hat.shape  # same as original x
    """
    padded_dim = _next_power_of_2(dim)
    bs = padded_dim if (block_size == 0 or block_size >= padded_dim) else block_size

    # Try fused Metal kernel first (fastest path — single Metal dispatch)
    # Fused kernel only supports full-dim WHT on power-of-2 dims
    use_fused = (
        bits == 4
        and padded_dim <= 256
        and (bs & (bs - 1)) == 0
        and bs == padded_dim  # fused kernel only supports full-dim WHT
        and os.environ.get("TURBO_DISABLE_FUSED_KERNEL", "0") != "1"
    )
    if use_fused:
        result = turbo_decode_fused(packed_indices, norms, padded_dim, bits=bits, seed=seed, block_size=block_size)
        # Strip padding if needed
        if padded_dim != dim:
            result = result[..., :dim]
        return result

    # Fallback: Python graph path
    cb = _get_codebook(bits, bs)

    # 1. Unpack indices — packed storage uses padded_dim
    indices = _unpack_indices(packed_indices, bits, padded_dim)

    # 2. Codebook lookup
    centroids = cb.centroids  # (n_levels,)
    x_rotated = centroids[indices]  # (..., padded_dim)

    # 3. Inverse dual sign flip + WHT
    # Encode was: x_rot = signs2 * WHT(signs1 * x)
    # Decode is:  x = signs1 * WHT(signs2 * x_rot)  (signs are self-inverse)
    signs1 = _sign_flip_vector(bs, seed)
    signs2 = _sign_flip_vector2(bs, seed)

    if bs < padded_dim:
        # Blocked inverse WHT: reshape → signs → WHT → signs → reshape back
        n_blocks = padded_dim // bs
        x_blocked = x_rotated.reshape(*x_rotated.shape[:-1], n_blocks, bs)
        x_flipped = x_blocked * signs2
        x_wht = mx.hadamard_transform(x_flipped)
        x_unit = (x_wht * signs1).reshape(*x_rotated.shape)
    else:
        # Full WHT path
        x_flipped = mx.hadamard_transform(x_rotated * signs2)
        x_unit = x_flipped * signs1

    # 4. Scale by norms
    x_reconstructed = x_unit * norms

    # 5. Strip padding if needed (non-power-of-2 original dim)
    if padded_dim != dim:
        x_reconstructed = x_reconstructed[..., :dim]

    return x_reconstructed


# ---------------------------------------------------------------------------
# Fused Metal kernels for turbo encode / decode
# ---------------------------------------------------------------------------
# These eliminate Python graph construction overhead by running the entire
# encode (norm→normalize→sign_flip→hadamard→quantize→pack) or decode
# (unpack→lookup→hadamard→sign_flip→scale) pipeline as a single Metal
# dispatch via mx.fast.metal_kernel.
# ---------------------------------------------------------------------------

_TURBO_ENCODE_HEADER = """
// Fused turbo_encode Metal kernel
// One threadgroup per vector. Each thread handles one element of the dim.
//
// Pipeline: norm → normalize → sign_flip → WHT butterfly → boundary quantize → pack
//
// Uses threadgroup shared memory for:
//   1. WHT butterfly stages (in-place)
//   2. Norm reduction
//   3. Pack reduction (gather indices → uint32)

// Inline boundary quantize: count how many boundaries the value exceeds
inline uint boundary_quantize(float val, const device float* boundaries, uint n_boundaries) {
    uint idx = 0;
    for (uint i = 0; i < n_boundaries; i++) {
        idx += (val > boundaries[i]) ? 1 : 0;
    }
    return idx;
}
"""

_TURBO_ENCODE_SOURCE_4BIT = """
    // Grid: (num_vectors * dim, 1, 1) — dim threads per vector
    // Threadgroup: (dim, 1, 1) — one threadgroup = one vector
    //
    // Inputs:
    //   x:          [num_vectors, dim]     — input vectors (float32)
    //   signs1:     [dim]                  — pre-WHT sign flip array {-1, +1}
    //   signs2:     [dim]                  — post-WHT sign flip array {-1, +1}
    //   boundaries: [15]                   — quantization boundaries (4-bit: 15)
    //   params:     [3]                    — {dim, num_vectors, packed_dim}
    //
    // Outputs:
    //   packed_out:  [num_vectors, packed_dim] — packed 4-bit indices (uint32)
    //   norms_out:   [num_vectors]             — L2 norms

    uint tid = thread_position_in_threadgroup.x;
    uint vec_idx = threadgroup_position_in_grid.x;
    uint dim = params[0];
    uint num_vectors = params[1];
    uint packed_dim = params[2];
    uint wht_block_size = params[3];  // block_size for WHT (32 or dim)

    if (vec_idx >= num_vectors || tid >= dim) return;

    // Shared memory for WHT butterfly + norm reduction
    threadgroup float shared_data[256];  // max dim=256
    threadgroup float shared_norm[1];

    // 1. Load input element
    float val = x[vec_idx * dim + tid];

    // 2. Compute L2 norm via threadgroup reduction
    //    Each thread contributes val*val, then we do a tree reduction
    shared_data[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for sum of squares
    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float norm = sqrt(shared_data[0]);
        shared_norm[0] = max(norm, 1e-10f);
        // Write norm output
        norms_out[vec_idx] = norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float norm = shared_norm[0];

    // 3. Normalize to unit sphere
    val = val / norm;

    // 4. Pre-WHT sign flip (signs1) — signs are block_size-length, tiled via modulo
    uint sign_idx = tid % wht_block_size;
    val = val * signs1[sign_idx];

    // 5. WHT butterfly (in-place via shared memory, double-buffered reads)
    //    Hadamard with 1/sqrt(block_size) normalization (orthonormal)
    //    log2(block_size) stages of butterfly operations within each block
    //    Each stage: read pair into registers, barrier, write result
    shared_data[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint log2_bs = 0;
    for (uint d = wht_block_size; d > 1; d >>= 1) log2_bs++;

    // WHT butterfly within each block of wht_block_size
    uint block_start = (tid / wht_block_size) * wht_block_size;

    for (uint stage = 0; stage < log2_bs; stage++) {
        uint half_block = 1u << stage;
        uint bfly_size = half_block << 1;
        uint local_tid = tid - block_start;  // position within WHT block
        uint bfly_idx = local_tid / bfly_size;
        uint local_idx = local_tid % bfly_size;
        uint base = block_start + bfly_idx * bfly_size;

        // Read both operands into registers BEFORE any thread writes
        float a = shared_data[base + (local_idx % half_block)];
        float b = shared_data[base + (local_idx % half_block) + half_block];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Write: top half gets a+b, bottom half gets a-b
        shared_data[tid] = (local_idx < half_block) ? (a + b) : (a - b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Apply 1/sqrt(block_size) normalization + post-WHT sign flip (signs2)
    float inv_sqrt_bs = rsqrt((float)wht_block_size);
    float rotated = shared_data[tid] * inv_sqrt_bs * signs2[sign_idx];

    // 6. Boundary quantize — 4-bit has 15 boundaries
    uint idx = boundary_quantize(rotated, boundaries, 15);

    // 7. Pack 4-bit indices into uint32 (8 indices per word)
    //    Thread tid maps to word (tid / 8), position (tid % 8)
    uint word_idx = tid / 8;
    uint pos_in_word = tid % 8;

    // Use shared memory to gather indices for packing
    // Each thread atomically ORs its index into the correct word
    threadgroup uint shared_packed[32];  // max packed_dim=32 (dim=256)
    if (tid < packed_dim) {
        shared_packed[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Atomic OR each thread's contribution into the packed word
    // For 4-bit: shift index by (pos_in_word * 4) bits
    uint shifted = idx << (pos_in_word * 4);

    // Use atomic_fetch_or for thread-safe packing
    // metal::atomic_fetch_or is not available on threadgroup memory in all cases,
    // so we use a different approach: one thread per word gathers all 8 indices.

    // Store index in shared memory at tid position
    threadgroup uint shared_indices[256];
    shared_indices[tid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One thread per packed word gathers 8 indices and packs them
    if (tid < packed_dim) {
        uint packed_word = 0;
        uint base = tid * 8;
        for (uint i = 0; i < 8 && (base + i) < dim; i++) {
            packed_word |= (shared_indices[base + i] & 0xF) << (i * 4);
        }
        packed_out[vec_idx * packed_dim + tid] = packed_word;
    }
"""

_TURBO_DECODE_HEADER = """
// Fused turbo_decode Metal kernel
// One threadgroup per vector. Each thread handles one element of the dim.
//
// Pipeline: unpack → codebook lookup → WHT butterfly → sign_flip → scale by norm
"""

_TURBO_DECODE_SOURCE_4BIT = """
    // Grid: (num_vectors * dim, 1, 1) — dim threads per vector
    // Threadgroup: (dim, 1, 1) — one threadgroup = one vector
    //
    // Inputs:
    //   packed_in:   [num_vectors, packed_dim] — packed 4-bit indices (uint32)
    //   norms_in:    [num_vectors]             — L2 norms
    //   centroids:   [16]                      — centroid lookup table
    //   signs1:      [dim]                     — pre-WHT sign flip array {-1, +1}
    //   signs2:      [dim]                     — post-WHT sign flip array {-1, +1}
    //   params:      [3]                       — {dim, num_vectors, packed_dim}
    //
    // Outputs:
    //   x_out:       [num_vectors, dim]        — reconstructed vectors (float32)

    uint tid = thread_position_in_threadgroup.x;
    uint vec_idx = threadgroup_position_in_grid.x;
    uint dim = params[0];
    uint num_vectors = params[1];
    uint packed_dim = params[2];
    uint wht_block_size = params[3];  // block_size for WHT (32 or dim)

    if (vec_idx >= num_vectors || tid >= dim) return;

    // Shared memory for WHT butterfly
    threadgroup float shared_data[256];  // max dim=256

    // 1. Unpack: extract 4-bit index for this thread's position
    uint word_idx = tid / 8;
    uint pos_in_word = tid % 8;
    uint packed_word = packed_in[vec_idx * packed_dim + word_idx];
    uint idx = (packed_word >> (pos_in_word * 4)) & 0xF;

    // 2. Codebook lookup + pre-inverse-WHT sign flip (signs2)
    //    Signs are block_size-length, tiled via modulo
    uint sign_idx = tid % wht_block_size;
    float val = centroids[idx] * signs2[sign_idx];

    // 3. Inverse WHT butterfly within each block (Hadamard is self-inverse up to scaling)
    shared_data[tid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint log2_bs = 0;
    for (uint d = wht_block_size; d > 1; d >>= 1) log2_bs++;

    uint block_start = (tid / wht_block_size) * wht_block_size;

    for (uint stage = 0; stage < log2_bs; stage++) {
        uint half_block = 1u << stage;
        uint bfly_size = half_block << 1;
        uint local_tid = tid - block_start;
        uint bfly_idx = local_tid / bfly_size;
        uint local_idx = local_tid % bfly_size;
        uint base = block_start + bfly_idx * bfly_size;

        float a = shared_data[base + (local_idx % half_block)];
        float b = shared_data[base + (local_idx % half_block) + half_block];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        shared_data[tid] = (local_idx < half_block) ? (a + b) : (a - b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sqrt_bs = rsqrt((float)wht_block_size);
    float rotated = shared_data[tid] * inv_sqrt_bs;

    // 4. Inverse sign flip — signs1 (post-inverse-WHT, undoes the pre-WHT flip)
    rotated = rotated * signs1[sign_idx];

    // 5. Scale by norm
    float norm = norms_in[vec_idx];
    x_out[vec_idx * dim + tid] = rotated * norm;
"""


def _get_turbo_encode_kernel(bits: int):
    """Get or create the fused encode Metal kernel.

    Args:
        bits: Quantization bit-width (currently only 4-bit supported).

    Returns:
        Compiled Metal kernel callable.
    """
    cache_key = f"turbo_encode_{bits}bit"
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    if bits != 4:
        raise NotImplementedError(
            f"Fused encode kernel not yet implemented for {bits}-bit. "
            "Use turbo_encode() (Python graph) instead."
        )

    kernel = mx.fast.metal_kernel(
        name=f"turbo_encode_{bits}bit",
        input_names=["x", "signs1", "signs2", "boundaries", "params"],
        output_names=["packed_out", "norms_out"],
        header=_TURBO_ENCODE_HEADER,
        source=_TURBO_ENCODE_SOURCE_4BIT,
        ensure_row_contiguous=True,
        atomic_outputs=False,
    )

    _kernel_cache[cache_key] = kernel
    return kernel


def _get_turbo_decode_kernel(bits: int):
    """Get or create the fused decode Metal kernel.

    Args:
        bits: Quantization bit-width (currently only 4-bit supported).

    Returns:
        Compiled Metal kernel callable.
    """
    cache_key = f"turbo_decode_{bits}bit"
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    if bits != 4:
        raise NotImplementedError(
            f"Fused decode kernel not yet implemented for {bits}-bit. "
            "Use turbo_decode() (Python graph) instead."
        )

    kernel = mx.fast.metal_kernel(
        name=f"turbo_decode_{bits}bit",
        input_names=["packed_in", "norms_in", "centroids", "signs1", "signs2", "params"],
        output_names=["x_out"],
        header=_TURBO_DECODE_HEADER,
        source=_TURBO_DECODE_SOURCE_4BIT,
        ensure_row_contiguous=True,
        atomic_outputs=False,
    )

    _kernel_cache[cache_key] = kernel
    return kernel


# Cache for pre-computed sign flip vectors (avoid regenerating each call)
_sign_cache: Dict[Tuple[int, int], mx.array] = {}


def _get_signs(dim: int, seed: int) -> Tuple[mx.array, mx.array]:
    """Get cached dual sign flip vectors (signs1, signs2)."""
    key = (dim, seed)
    if key not in _sign_cache:
        _sign_cache[key] = (
            _sign_flip_vector(dim, seed),
            _sign_flip_vector2(dim, seed),
        )
    return _sign_cache[key]


def turbo_encode_fused(
    x: mx.array,
    bits: int = 4,
    seed: int = 42,
    block_size: int = 0,
) -> Tuple[mx.array, mx.array]:
    """Fused Metal kernel encode — single dispatch replaces ~8 graph nodes.

    Runs the full encode pipeline (norm → normalize → sign_flip → WHT →
    boundary quantize → pack) as one Metal kernel, eliminating Python graph
    construction overhead that dominates small-batch decode latency.

    Falls back to turbo_encode() for non-4-bit or non-power-of-2 dims.

    Args:
        x: Input tensor of shape (..., dim). dim must be power of 2, max 256.
        bits: Quantization bit-width. Default: 4 (only 4-bit fused kernel).
        seed: Random seed for the sign-flip diagonal. Default: 42.

    Returns:
        Tuple of (packed_indices, norms) — same format as turbo_encode().

    Example:
        >>> x = mx.random.normal((1, 8, 1, 128))
        >>> packed, norms = turbo_encode_fused(x, bits=4)
        >>> packed.shape  # (1, 8, 1, 16)
        >>> norms.shape   # (1, 8, 1)
    """
    dim = x.shape[-1]
    bs = dim if (block_size == 0 or block_size >= dim) else block_size

    # Fallback for unsupported configs
    if bits != 4 or dim > 256 or (dim & (dim - 1)) != 0 or (bs & (bs - 1)) != 0:
        return turbo_encode(x, bits=bits, seed=seed, block_size=block_size)

    kernel = _get_turbo_encode_kernel(bits)
    # Codebook uses block_size, not dim — centroids are scaled for the WHT size
    cb = _get_codebook(bits, bs)
    # Signs are block_size-length — the Metal kernel tiles via modulo
    signs1, signs2 = _get_signs(bs, seed)

    # Flatten to (num_vectors, dim)
    leading_shape = x.shape[:-1]
    num_vectors = 1
    for s in leading_shape:
        num_vectors *= s

    x_flat = x.reshape(num_vectors, dim).astype(mx.float32)
    packed_dim = dim // 8  # 4-bit: 8 indices per uint32

    # params[3] = block_size for the Metal kernel's WHT butterfly
    params = mx.array([dim, num_vectors, packed_dim, bs], dtype=mx.uint32)

    outputs = kernel(
        inputs=[x_flat, signs1, signs2, cb.boundaries, params],
        output_shapes=[
            (num_vectors, packed_dim),  # packed_out
            (num_vectors,),             # norms_out
        ],
        output_dtypes=[mx.uint32, mx.float32],
        grid=(num_vectors * dim, 1, 1),  # total threads = num_vectors * dim
        threadgroup=(dim, 1, 1),          # one threadgroup per vector
        init_value=0,
        stream=mx.gpu,
    )

    packed = outputs[0].reshape(*leading_shape, packed_dim)
    norms = outputs[1].reshape(*leading_shape, 1)

    return packed, norms


def turbo_decode_fused(
    packed_indices: mx.array,
    norms: mx.array,
    dim: int,
    bits: int = 4,
    seed: int = 42,
    block_size: int = 0,
) -> mx.array:
    """Fused Metal kernel decode — single dispatch replaces ~6 graph nodes.

    Runs the full decode pipeline (unpack → codebook lookup → WHT → sign_flip
    → scale) as one Metal kernel.

    Falls back to turbo_decode() for non-4-bit or non-power-of-2 dims.

    Args:
        packed_indices: uint32 tensor from turbo_encode. Shape (..., packed_dim).
        norms: float32 norms from turbo_encode. Shape (..., 1).
        dim: Original head dimension.
        bits: Quantization bit-width. Default: 4.
        seed: Must match the seed used in turbo_encode. Default: 42.

    Returns:
        Reconstructed tensor of shape (..., dim).

    Example:
        >>> packed, norms = turbo_encode_fused(x, bits=4)
        >>> x_hat = turbo_decode_fused(packed, norms, dim=128)
    """
    bs = dim if (block_size == 0 or block_size >= dim) else block_size

    # Fallback for unsupported configs
    if bits != 4 or dim > 256 or (dim & (dim - 1)) != 0 or (bs & (bs - 1)) != 0:
        return turbo_decode(packed_indices, norms, dim, bits=bits, seed=seed, block_size=block_size)

    kernel = _get_turbo_decode_kernel(bits)
    # Codebook uses block_size, not dim — centroids are scaled for the WHT size
    cb = _get_codebook(bits, bs)
    # Signs are block_size-length — the Metal kernel tiles via modulo
    signs1, signs2 = _get_signs(bs, seed)

    packed_dim = dim // 8
    leading_shape = packed_indices.shape[:-1]
    num_vectors = 1
    for s in leading_shape:
        num_vectors *= s

    packed_flat = packed_indices.reshape(num_vectors, packed_dim).astype(mx.uint32)
    # Flatten norms — handle both (..., 1) and (...,) shapes
    norms_flat = norms.reshape(num_vectors).astype(mx.float32)

    # params[3] = block_size for the Metal kernel's WHT butterfly
    params = mx.array([dim, num_vectors, packed_dim, bs], dtype=mx.uint32)

    outputs = kernel(
        inputs=[packed_flat, norms_flat, cb.centroids, signs1, signs2, params],
        output_shapes=[
            (num_vectors, dim),  # x_out
        ],
        output_dtypes=[mx.float32],
        grid=(num_vectors * dim, 1, 1),  # total threads = num_vectors * dim
        threadgroup=(dim, 1, 1),          # one threadgroup per vector
        init_value=0,
        stream=mx.gpu,
    )

    return outputs[0].reshape(*leading_shape, dim)


# ---------------------------------------------------------------------------
# Packing / unpacking bit indices into uint32
# ---------------------------------------------------------------------------


def _pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack N-bit indices into uint32 words.

    Args:
        indices: uint32 tensor of shape (..., dim) with values in [0, 2^bits).
        bits: Bit-width per index (2, 3, or 4).

    Returns:
        uint32 tensor of shape (..., packed_dim) where packed_dim = ceil(dim * bits / 32).
    """
    dim = indices.shape[-1]
    leading_shape = indices.shape[:-1]
    indices_per_word = 32 // bits

    # For bits that evenly divide 32 (2, 4), this is clean.
    # For bits=3, we pad dim to next multiple of indices_per_word (10 per uint32).
    if 32 % bits != 0:
        # bits=3: 10 indices per uint32, with 2 wasted bits
        pad_to = math.ceil(dim / indices_per_word) * indices_per_word
        if pad_to > dim:
            padding = mx.zeros((*leading_shape, pad_to - dim), dtype=mx.uint32)
            indices = mx.concatenate([indices, padding], axis=-1)
            dim = pad_to

    # Reshape to (..., n_words, indices_per_word)
    n_words = dim // indices_per_word
    indices = indices.reshape(*leading_shape, n_words, indices_per_word)

    # Shift each index to its bit position and OR together
    shifts = mx.array([i * bits for i in range(indices_per_word)], dtype=mx.uint32)
    packed = mx.sum(indices << shifts, axis=-1).astype(mx.uint32)

    return packed


def _unpack_indices(packed: mx.array, bits: int, dim: int) -> mx.array:
    """Unpack uint32 words into N-bit indices.

    Args:
        packed: uint32 tensor of shape (..., packed_dim).
        bits: Bit-width per index (2, 3, or 4).
        dim: Original dimension (number of indices to unpack).

    Returns:
        uint32 tensor of shape (..., dim) with values in [0, 2^bits).
    """
    leading_shape = packed.shape[:-1]
    indices_per_word = 32 // bits
    mask = mx.array((1 << bits) - 1, dtype=mx.uint32)

    # Expand each uint32 into its constituent indices
    # packed shape: (..., n_words) → (..., n_words, 1)
    packed_expanded = mx.expand_dims(packed, axis=-1)

    # Shift amounts: [0, bits, 2*bits, ...]
    shifts = mx.array([i * bits for i in range(indices_per_word)], dtype=mx.uint32)

    # Extract indices: shift right then mask
    indices = (packed_expanded >> shifts) & mask  # (..., n_words, indices_per_word)

    # Reshape back to flat
    indices = indices.reshape(*leading_shape, -1)

    # Trim padding if needed (for bits=3)
    if indices.shape[-1] > dim:
        indices = indices[..., :dim]

    return indices


# ---------------------------------------------------------------------------
# Sparse attention masking (sparse-v-dequant.md)
# ---------------------------------------------------------------------------


def sparse_attention_mask(
    weights: mx.array,
    threshold: float = 1e-6,
) -> mx.array:
    """Create a boolean mask that zeros out near-zero attention weights.

    After softmax, many attention positions have negligible weight (< 1e-6).
    Zeroing these before the V matmul lets MLX potentially skip those lanes,
    saving compute proportional to sparsity.

    From sparse-v-dequant.md: on Apple Silicon, skipping dequant for near-zero
    weights saved meaningful compute. This mask is the first step — it prevents
    near-zero weights from contributing to the V weighted sum. A full sparse
    implementation would also skip the V dequant itself (requires fused kernel).

    Args:
        weights: Post-softmax attention weights, shape (..., q_len, kv_len).
        threshold: Weights below this value are masked out. Default: 1e-6.

    Returns:
        Boolean mask of same shape as weights (1.0 where weight >= threshold,
        0.0 where weight < threshold). Multiply with weights before V matmul.

    Example:
        >>> weights = mx.softmax(scores, axis=-1)
        >>> mask = sparse_attention_mask(weights, threshold=1e-6)
        >>> weights = weights * mask  # zero out negligible positions
        >>> output = weights @ values  # MLX can skip zeroed lanes
    """
    return (weights >= threshold).astype(weights.dtype)


# ---------------------------------------------------------------------------
# TurboQuant attention (decode-then-matmul, not fused)
# ---------------------------------------------------------------------------


def turbo_attention(
    queries: mx.array,
    packed_keys: mx.array,
    key_norms: mx.array,
    packed_values: mx.array,
    value_norms: mx.array,
    dim: int,
    bits: int = 4,
    seed: int = 42,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Compute attention with TurboQuant-compressed KV cache.

    This is the decode-then-matmul approach — we decompress K and V,
    then do standard scaled dot-product attention. A fused Metal kernel
    version will come later for better performance.

    **Asymmetric K/V quantization** (recommended):

    K precision dominates output quality because errors in K are amplified
    through the softmax exponential — small errors in dot products become
    large errors in attention weights. V errors are merely averaged.

    Recommended config: K stays at FP16, V at turbo4 (``key_bits=0, bits=4``
    in TurboQuantKVCache, aka "turbo0v4"). Symmetric turbo3/turbo3 works
    but asymmetric gives strictly better quality for the same memory budget.

    Args:
        queries: Query tensor, shape (batch, heads, q_len, dim).
        packed_keys: Packed key indices from turbo_encode, shape (batch, heads, kv_len, packed_dim).
        key_norms: Key norms, shape (batch, heads, kv_len, 1).
        packed_values: Packed value indices, shape (batch, heads, kv_len, packed_dim).
        value_norms: Value norms, shape (batch, heads, kv_len, 1).
        dim: Head dimension.
        bits: Quantization bit-width. Default: 4.
        seed: SRHT seed. Default: 42.
        scale: Attention scale factor. Default: 1/sqrt(dim).
        mask: Optional attention mask, shape broadcastable to (batch, heads, q_len, kv_len).

    Returns:
        Attention output, shape (batch, heads, q_len, dim).

    Example:
        >>> q = mx.random.normal((1, 8, 1, 128))
        >>> k = mx.random.normal((1, 8, 32, 128))
        >>> v = mx.random.normal((1, 8, 32, 128))
        >>> pk, kn = turbo_encode(k, bits=4)
        >>> pv, vn = turbo_encode(v, bits=4)
        >>> out = turbo_attention(q, pk, kn, pv, vn, dim=128, bits=4)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Decode keys and values
    # TODO: Fused Metal kernel — decode + matmul in one pass to avoid materializing full K/V
    keys = turbo_decode(packed_keys, key_norms, dim, bits, seed)
    values = turbo_decode(packed_values, value_norms, dim, bits, seed)

    # Standard scaled dot-product attention
    # Q·K^T
    scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale

    if mask is not None:
        scores = scores + mask

    weights = mx.softmax(scores, axis=-1)

    # Apply sparse attention mask to skip near-zero V contributions
    # See: sparse-v-dequant.md — many post-softmax weights are near-zero,
    # making those V dequant+matmul ops wasted compute. By zeroing them
    # out before the matmul, MLX can potentially skip those lanes entirely.
    #
    # TODO: Sparse V integration with TurboKVCache (Item 9)
    # This sparse mask works in turbo_attention() (decode-then-matmul path),
    # but TurboKVCache uses standard SDPA via mlx-lm's base.py — it returns
    # plain mx.array K/V tensors and the model calls mx.fast.scaled_dot_product_attention.
    # To integrate sparse V at the model level:
    #   1. Add a post-SDPA hook in TurboKVCache, or
    #   2. Modify mlx-lm's base.py to accept a sparse_mask callback, or
    #   3. Implement a fused Metal kernel that does SDPA + sparse skip in one pass
    # Option 3 is the real win — skip V dequant entirely for near-zero attention
    # positions, saving both compute and memory bandwidth. Options 1-2 still
    # materialize all V tokens to FP16 before the matmul.
    # For now, sparse_attention_mask() is available as a standalone utility that
    # users can apply manually if they write custom attention loops.
    sparse_mask = sparse_attention_mask(weights)
    weights = weights * sparse_mask

    # Weighted sum of values
    # TODO: Full sparse V optimization — skip dequant entirely for masked
    # positions. The current decode-then-matmul approach materializes all V
    # tokens to FP16 before the matmul. True sparse dequant would only decode
    # the V tokens with significant attention weight, saving both compute and
    # memory bandwidth. This requires a fused kernel that checks attention
    # weights BEFORE dequanting each V token. Expected benefit on Apple Silicon:
    # 15-30% decode speedup at long context (>4K tokens) where most attention
    # mass concentrates on a few positions. (sparse-v-dequant.md)
    output = weights @ values

    return output


# ---------------------------------------------------------------------------
# Asymmetric fused attention: FP16 K scoring + turbo V weighted sum
# ---------------------------------------------------------------------------
# For asymmetric config (K=FP16, V=turbo4), the scoring phase (Q × K^T) uses
# raw FP16 keys — standard matmul, already fast. Only the V weighted sum
# needs to touch packed data. This kernel takes pre-computed attention weights
# and does weighted centroid lookup directly on packed V, avoiding full V decode.
#
# This is simpler than the full fused kernel because scores are already computed.
# It's just: for each dim, sum over t of: weight[t] * centroid[packed_v[t][d]] * norm[t]
# Then inverse-rotate once in Python.
# ---------------------------------------------------------------------------

_TURBO_WEIGHTED_V_SUM_HEADER = """
// Inline unpack: extract a 4-bit index from a uint32 word
inline uint unpack4(uint word, uint pos) {
    return (word >> (pos * 4)) & 0xF;
}
"""

_TURBO_WEIGHTED_V_SUM_SOURCE = """
    // Thread-per-dim kernel with T_kv tiling across threadgroups
    //
    // Grid: (n_bh * n_tiles, dim, 1) — threadgroups tile both (bh, T_kv)
    // Threadgroup: (1, dim_tg, 1) — one thread per dimension within a tile
    //
    // Each threadgroup processes a TILE_T chunk of tokens for one (batch, head).
    // Cooperative wn load into shared memory, then per-dim accumulation.
    // Partial results written to out_accum[tile_idx, bh_idx, dim] then summed
    // in Python.
    //
    // Inputs:
    //   weights:      [n_bh, T_kv]              — post-softmax attention weights
    //   packed_v:     [n_bh, T_kv, packed_dim]   — packed 4-bit V indices
    //   v_norms:      [n_bh, T_kv]              — V L2 norms
    //   centroids:    [n_levels]                  — centroid lookup table
    //   params:       [4]                         — {dim, T_kv, packed_dim, n_tiles}
    //
    // Outputs:
    //   out_accum:    [n_bh * n_tiles, dim]      — partial tile results

    constexpr int TILE_T = 256;

    uint flat_idx = threadgroup_position_in_grid.x;
    uint d = thread_position_in_grid.y;

    int dim_val = params[0];
    int T_kv = params[1];
    int packed_dim = params[2];
    int n_tiles = params[3];

    uint bh_idx = flat_idx / n_tiles;
    uint tile_idx = flat_idx % n_tiles;

    // Token range for this tile
    int t_start = tile_idx * TILE_T;
    int t_end = min(t_start + TILE_T, T_kv);
    int tile_len = t_end - t_start;
    if (tile_len <= 0) {
        out_accum[flat_idx * dim_val + d] = 0.0f;
        return;
    }

    // Which uint32 word and bit offset for this dimension's 4-bit index
    int word_idx = d / 8;
    int bit_offset = (d % 8) * 4;

    // Base offsets for this (batch, head)
    int w_base = bh_idx * T_kv;
    int pv_base = bh_idx * T_kv * packed_dim;

    // Shared memory for weight*norm values in this tile
    threadgroup float wn_shared[TILE_T];

    // Cooperative load: dim threads load tile_len wn values
    for (int i = (int)d; i < tile_len; i += dim_val) {
        int t = t_start + i;
        float w = weights[w_base + t];
        wn_shared[i] = (w < 1e-6f) ? 0.0f : w * v_norms[w_base + t];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;

    // Process all tokens in this tile
    for (int i = 0; i < tile_len; i++) {
        float wn = wn_shared[i];
        if (wn == 0.0f) continue;

        int t = t_start + i;
        uint word = packed_v[pv_base + t * packed_dim + word_idx];
        uint idx = (word >> bit_offset) & 0xF;
        acc += wn * centroids[idx];
    }

    out_accum[flat_idx * dim_val + d] = acc;
"""

# Cache for the compiled weighted V sum kernel
_weighted_v_sum_kernel_cache: Dict[Tuple[int, int], object] = {}


def _get_weighted_v_sum_kernel(bits: int, nr0: int = 1):
    """Get or compile the weighted V sum Metal kernel."""
    cache_key = (bits, nr0)
    if cache_key in _weighted_v_sum_kernel_cache:
        return _weighted_v_sum_kernel_cache[cache_key]

    if bits != 4:
        raise NotImplementedError(
            f"turbo_weighted_value_sum only supports 4-bit, got {bits}-bit"
        )

    source = _TURBO_WEIGHTED_V_SUM_SOURCE
    header = _TURBO_WEIGHTED_V_SUM_HEADER

    kernel = mx.fast.metal_kernel(
        name=f"turbo_weighted_v_sum_{bits}bit",
        input_names=["weights", "packed_v", "v_norms", "centroids", "params"],
        output_names=["out_accum"],
        header=header,
        source=source,
    )
    _weighted_v_sum_kernel_cache[cache_key] = kernel
    return kernel


def turbo_weighted_value_sum(
    attention_weights: mx.array,
    packed_values: mx.array,
    value_norms: mx.array,
    dim: int,
    bits: int = 4,
    seed: int = 42,
) -> mx.array:
    """Weighted sum of turbo-quantized values using pre-computed attention weights.

    For asymmetric attention (K=FP16, V=turbo4): the scoring phase uses raw FP16
    keys with standard matmul. This function handles only the V weighted sum —
    it takes post-softmax attention weights and computes the output directly from
    packed V data without decoding V to FP16 first.

    Metal kernel: for each output element (batch, head, dim_idx):
      val = sum over t of: weight[t] * centroid[packed_v[t][dim_idx]] * norm[t]
    Then inverse-rotate in Python: signs1 * WHT(signs2 * val)

    This eliminates the #1 gap in asymmetric config: V decode overhead that
    previously required full turbo_decode() every step.

    Args:
        attention_weights: Post-softmax weights, shape (B, n_q_heads, 1, T_kv).
        packed_values: Packed V indices, shape (B, n_kv_heads, T_kv, packed_dim).
        value_norms: V norms, shape (B, n_kv_heads, T_kv, 1).
        dim: Head dimension (must be power of 2, max 256).
        bits: Quantization bit-width. Default: 4.
        seed: SRHT random seed (must match encode). Default: 42.

    Returns:
        Attention output in original domain, shape (B, n_q_heads, 1, dim).

    Example:
        >>> # Asymmetric: FP16 K scoring, turbo4 V weighted sum
        >>> scores = (q @ fp16_keys.transpose(0, 1, 3, 2)) * scale
        >>> weights = mx.softmax(scores, axis=-1)
        >>> pv, vn = turbo_encode(values, bits=4)
        >>> out = turbo_weighted_value_sum(weights, pv, vn, dim=128)
    """
    if bits != 4:
        raise NotImplementedError(
            f"turbo_weighted_value_sum only supports 4-bit, got {bits}-bit"
        )
    if dim > 256:
        raise ValueError(
            f"turbo_weighted_value_sum supports dim <= 256, got dim={dim}"
        )

    B = attention_weights.shape[0]
    n_q_heads = attention_weights.shape[1]
    T_kv = attention_weights.shape[-1]
    n_kv_heads = packed_values.shape[1]
    packed_dim = packed_values.shape[-1]

    # Handle GQA: expand KV heads to match query heads if needed
    if n_kv_heads < n_q_heads:
        gqa_factor = n_q_heads // n_kv_heads
        packed_values = mx.repeat(packed_values, gqa_factor, axis=1)
        value_norms = mx.repeat(value_norms, gqa_factor, axis=1)

    # Get codebook centroids
    cb = _get_codebook(bits, dim)

    # Flatten for kernel: (B*n_q_heads, ...)
    n_bh = B * n_q_heads
    weights_flat = attention_weights.reshape(n_bh, T_kv).astype(mx.float32)
    pv_flat = packed_values.reshape(n_bh, T_kv, packed_dim)
    vn_flat = value_norms.squeeze(-1).reshape(n_bh, T_kv).astype(mx.float32)

    # Thread-per-dim kernel with T_kv tiling across threadgroups
    # Each threadgroup processes TILE_T=256 tokens for one (bh, tile) pair
    # Partials summed in Python after kernel returns
    TILE_T = 256
    n_tiles = (T_kv + TILE_T - 1) // TILE_T

    params = mx.array([dim, T_kv, packed_dim, n_tiles], dtype=mx.uint32)

    kernel = _get_weighted_v_sum_kernel(bits)

    outputs = kernel(
        inputs=[
            weights_flat,       # weights
            pv_flat,            # packed_v
            vn_flat,            # v_norms
            cb.centroids,       # centroids
            params,             # params
        ],
        output_shapes=[
            (n_bh * n_tiles, dim),    # out_accum — partials per tile
        ],
        output_dtypes=[mx.float32],
        grid=(n_bh * n_tiles, dim, 1),
        threadgroup=(1, min(dim, 256), 1),
        init_value=0.0,
        stream=mx.gpu,
    )

    # Sum partial tile results: (n_bh * n_tiles, dim) -> (n_bh, dim)
    if n_tiles > 1:
        out_rot = outputs[0].reshape(n_bh, n_tiles, dim).sum(axis=1)
    else:
        out_rot = outputs[0]  # (n_bh, dim) — in WHT domain

    # Inverse transform: signs1 * WHT(signs2 * out_rot)
    signs1 = _sign_flip_vector(dim, seed)
    signs2 = _sign_flip_vector2(dim, seed)
    out_rot = out_rot.reshape(B, n_q_heads, 1, dim)
    out_transformed = mx.hadamard_transform(out_rot * signs2)
    output = out_transformed * signs1

    return output.astype(attention_weights.dtype)


def turbo_asymmetric_attention(
    queries: mx.array,
    fp_keys: mx.array,
    packed_values: mx.array,
    value_norms: mx.array,
    dim: int,
    bits: int = 4,
    seed: int = 42,
    scale: Optional[float] = None,
) -> mx.array:
    """Asymmetric fused attention: FP16 K scoring + turbo V weighted sum.

    The optimal path for K=FP16, V=turbo4 — the recommended config. Scores are
    computed with standard matmul on raw FP16 keys (no decode needed), then the
    V weighted sum uses the Metal kernel on packed data (no V decode needed).

    This eliminates the decode overhead that was the #1 gap in asymmetric mode.

    Args:
        queries: Query tensor, shape (B, n_q_heads, 1, dim).
        fp_keys: Raw FP16 keys, shape (B, n_kv_heads, T_kv, dim).
        packed_values: Packed V indices, shape (B, n_kv_heads, T_kv, packed_dim).
        value_norms: V norms, shape (B, n_kv_heads, T_kv, 1).
        dim: Head dimension.
        bits: V quantization bit-width. Default: 4.
        seed: SRHT seed. Default: 42.
        scale: Attention scale. Default: 1/sqrt(dim).

    Returns:
        Attention output, shape (B, n_q_heads, 1, dim).

    Example:
        >>> cache = TurboKVCache(bits=4, key_bits=0)  # K=FP16, V=turbo4
        >>> # After prefill + compression:
        >>> out = turbo_asymmetric_attention(q, cache._fp_keys, cache._packed_values,
        ...                                  cache._value_norms, dim=128)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    n_kv_heads = fp_keys.shape[1]
    n_q_heads = queries.shape[1]

    # Handle GQA for K scoring: expand KV keys to match query heads
    # Cast to queries dtype for consistent precision (some layers store
    # K as FP16, others as FP32 — mismatch causes small errors that
    # compound across generation steps)
    keys_for_score = fp_keys.astype(queries.dtype)
    if n_kv_heads < n_q_heads:
        gqa_factor = n_q_heads // n_kv_heads
        keys_for_score = mx.repeat(keys_for_score, gqa_factor, axis=1)

    # Step 1: Score with K — standard matmul
    scores = (queries @ keys_for_score.transpose(0, 1, 3, 2)) * scale

    # Step 2: Softmax
    weights = mx.softmax(scores, axis=-1)

    # Step 3: Weighted sum of turbo V — Metal kernel, no V decode
    output = turbo_weighted_value_sum(
        weights, packed_values, value_norms, dim, bits=bits, seed=seed,
    )

    return output


# ---------------------------------------------------------------------------
# Fused single-dispatch asymmetric attention (K=FP16, V=turbo4)
# ---------------------------------------------------------------------------
# The CRITICAL optimization: turbo_asymmetric_attention uses 4 separate Metal
# dispatches (GQA expand, Q×K score, softmax, fused V sum) totaling ~0.704ms.
# Native SDPA does everything in ONE dispatch at ~0.151ms.
#
# This kernel fuses ALL 4 operations into a single dispatch:
#   - Thread-per-dim layout: one threadgroup per query head, dim threads
#   - GQA handled inside kernel (kv_head = query_head / gqa_factor)
#   - Q×K dot product via simd_sum across dim threads
#   - Online softmax (single pass: running max + sum correction)
#   - V weighted sum: each thread accumulates its output dim from packed V
#   - Inverse WHT butterfly + sign flips in shared memory
#   - No intermediate arrays materialized between phases
#
# Expected: match or beat native SDPA's 0.151ms for decode (T_q=1).

_FUSED_ASYMMETRIC_ATTN_HEADER = """
// Inline unpack: extract a 4-bit index from a uint32 word
inline uint unpack4(uint word, uint pos) {
    return (word >> (pos * 4)) & 0xF;
}
"""

_FUSED_ASYMMETRIC_ATTN_SOURCE_4BIT = """
    // Fused single-dispatch asymmetric attention: K=FP16, V=turbo4
    //
    // Phase 1: Q×K scoring — thread-per-token, full dot product per token
    // Phase 2: Softmax — parallel max/sum reduction
    // Phase 3: V weighted sum — thread-per-token, per-dim register accumulators
    // Phase 4: V reduction — simd_sum + shared memory cross-SIMD
    // Phase 5: Inverse WHT — butterfly in shared memory
    //
    // Scores stored to device scratch buffer (out_scores) to avoid shared memory
    // pressure. This allows unlimited T_kv without tiling overhead.
    //
    // Grid: (B*n_q_heads * TG_SIZE, 1, 1) — one threadgroup per query head
    // Threadgroup: (TG_SIZE, 1, 1) — TG_SIZE >= dim for WHT

    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint qh_idx = threadgroup_position_in_grid.x;  // query head index

    int dim_val = params[0];
    int T_kv    = params[1];
    int packed_dim = params[2];
    float scale = as_type<float>(params[3]);
    int gqa_factor = params[4];

    uint kv_head = qh_idx / gqa_factor;

    int q_base  = qh_idx * dim_val;
    int k_base  = kv_head * T_kv * dim_val;
    int pv_base = kv_head * T_kv * packed_dim;
    int vn_base = kv_head * T_kv;
    int scores_base = qh_idx * T_kv;

    uint simd_lane = thread_index_in_simdgroup;
    uint simd_id = tid / 32;
    uint n_simd = (tg_size + 31) / 32;

    // Shared memory: query cache (256) + scratch (8) + V reduction (n_simd * dim)
    threadgroup float shared_q[256];
    threadgroup float shared_scratch[8];
    threadgroup float shared_reduce[8 * 256];

    // Load query into shared memory
    if ((int)tid < dim_val) {
        shared_q[tid] = queries[q_base + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ====== Phase 1: Q×K scoring ======
    float max_score = -INFINITY;
    for (int t = (int)tid; t < T_kv; t += (int)tg_size) {
        float dot = 0.0f;
        int k_offset = k_base + t * dim_val;

        int d = 0;
        for (; d + 3 < dim_val; d += 4) {
            dot += shared_q[d]     * fp_keys[k_offset + d];
            dot += shared_q[d + 1] * fp_keys[k_offset + d + 1];
            dot += shared_q[d + 2] * fp_keys[k_offset + d + 2];
            dot += shared_q[d + 3] * fp_keys[k_offset + d + 3];
        }
        for (; d < dim_val; d++) {
            dot += shared_q[d] * fp_keys[k_offset + d];
        }

        float s = dot * scale;
        out_scores[scores_base + t] = s;
        max_score = max(max_score, s);
    }

    // ====== Phase 2: Softmax ======
    max_score = simd_max(max_score);
    if (simd_lane == 0) shared_scratch[simd_id] = max_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float gmax = shared_scratch[0];
        for (uint s = 1; s < n_simd; s++) gmax = max(gmax, shared_scratch[s]);
        shared_scratch[0] = gmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max = shared_scratch[0];

    float local_sum = 0.0f;
    for (int t = (int)tid; t < T_kv; t += (int)tg_size) {
        float e = exp(out_scores[scores_base + t] - global_max);
        out_scores[scores_base + t] = e;
        local_sum += e;
    }

    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) shared_scratch[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float gsum = 0.0f;
        for (uint s = 0; s < n_simd; s++) gsum += shared_scratch[s];
        shared_scratch[0] = gsum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0f / shared_scratch[0];

    // Normalize in-place
    for (int t = (int)tid; t < T_kv; t += (int)tg_size) {
        out_scores[scores_base + t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // ====== Phase 3: V weighted sum ======
    float v_accum[256];
    for (int dd = 0; dd < dim_val; dd++) v_accum[dd] = 0.0f;

    for (int t = (int)tid; t < T_kv; t += (int)tg_size) {
        float w = out_scores[scores_base + t];
        if (w < 1e-6f) continue;

        float wn = w * v_norms[vn_base + t];
        int pv_offset = pv_base + t * packed_dim;

        for (int pw = 0; pw < packed_dim; pw++) {
            uint word = packed_v[pv_offset + pw];
            int base_d = pw * 8;

            for (int j = 0; j < 8 && (base_d + j) < dim_val; j++) {
                uint idx = (word >> (j * 4)) & 0xF;
                v_accum[base_d + j] += wn * centroids[idx];
            }
        }
    }

    // ====== Phase 4: Reduce V accumulators ======
    for (int dd = 0; dd < dim_val; dd++) {
        float val = simd_sum(v_accum[dd]);
        if (simd_lane == 0) shared_reduce[simd_id * dim_val + dd] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)tid < dim_val) {
        float total = 0.0f;
        for (uint s = 0; s < n_simd; s++) total += shared_reduce[s * dim_val + tid];
        shared_q[tid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ====== Phase 5: Inverse WHT ======
    if ((int)tid < dim_val) {
        shared_q[tid] *= signs2[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)tid < dim_val) {
        uint log2_dim = 0;
        for (uint dd = dim_val; dd > 1; dd >>= 1) log2_dim++;

        for (uint stage = 0; stage < log2_dim; stage++) {
            uint half_block = 1u << stage;
            uint bfly_size = half_block << 1;
            uint bfly_idx = tid / bfly_size;
            uint local_idx = tid % bfly_size;
            uint base_idx = bfly_idx * bfly_size;

            float a = shared_q[base_idx + (local_idx % half_block)];
            float b = shared_q[base_idx + (local_idx % half_block) + half_block];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            shared_q[tid] = (local_idx < half_block) ? (a + b) : (a - b);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float inv_sqrt_dim = rsqrt((float)dim_val);
        output[qh_idx * dim_val + tid] = shared_q[tid] * inv_sqrt_dim * signs1[tid];
    }
"""

# Cache for the fused asymmetric attention kernel
_fused_asymmetric_kernel_cache: Dict[int, object] = {}


def _get_fused_asymmetric_kernel(bits: int = 4):
    """Get or compile the fused single-dispatch asymmetric attention kernel."""
    if bits in _fused_asymmetric_kernel_cache:
        return _fused_asymmetric_kernel_cache[bits]

    if bits != 4:
        raise NotImplementedError(
            f"Fused asymmetric attention only supports 4-bit V, got {bits}-bit"
        )

    kernel = mx.fast.metal_kernel(
        name="turbo_fused_asymmetric_sdpa_4bit",
        input_names=[
            "queries",     # [B*nq, dim] float32
            "fp_keys",     # [B*nkv, T_kv, dim] float16/32
            "packed_v",    # [B*nkv, T_kv, packed_dim] uint32
            "v_norms",     # [B*nkv, T_kv] float32
            "centroids",   # [16] float32
            "signs1",      # [dim] float32
            "signs2",      # [dim] float32
            "params",      # [5] uint32
        ],
        output_names=[
            "output",      # [B*nq, dim] float32 — final attention output
            "out_scores",  # [B*nq, T_kv] float32 — scratch for scores/weights
        ],
        header=_FUSED_ASYMMETRIC_ATTN_HEADER,
        source=_FUSED_ASYMMETRIC_ATTN_SOURCE_4BIT,
        ensure_row_contiguous=True,
        atomic_outputs=False,
    )

    _fused_asymmetric_kernel_cache[bits] = kernel
    return kernel


def turbo_fused_asymmetric_attention_single_dispatch(
    queries: mx.array,
    fp_keys: mx.array,
    packed_values: mx.array,
    value_norms: mx.array,
    dim: int,
    bits: int = 4,
    seed: int = 42,
    scale: Optional[float] = None,
) -> mx.array:
    """Fused single-dispatch asymmetric attention: K=FP16, V=turbo4.

    Replaces turbo_asymmetric_attention's 4 Metal dispatches with ONE:
    Q×K scoring, softmax, V weighted sum, and inverse WHT rotation all fused
    into a single Metal kernel. No intermediate arrays materialized.

    **Performance target:** Match or beat native SDPA's ~0.151ms (vs 0.704ms
    for the 4-dispatch version). The overhead was from 4 Metal dispatch
    round-trips, not from actual compute.

    **Thread layout:**
    - Grid: (B * n_q_heads, 1, 1) — one threadgroup per query head
    - Threadgroup: (dim, 1, 1) — one thread per output dimension
    - GQA: handled inside kernel (kv_head = query_head / gqa_factor)
    - Online softmax: single pass with running max + sum correction
    - Inverse WHT: butterfly + sign flips in shared memory

    Args:
        queries: Query tensor, shape (B, n_q_heads, 1, dim).
        fp_keys: Raw FP16 keys, shape (B, n_kv_heads, T_kv, dim).
        packed_values: Packed V indices, shape (B, n_kv_heads, T_kv, packed_dim).
        value_norms: V norms, shape (B, n_kv_heads, T_kv, 1).
        dim: Head dimension (must be power of 2, max 256).
        bits: V quantization bit-width. Default: 4.
        seed: SRHT seed. Default: 42.
        scale: Attention scale. Default: 1/sqrt(dim).

    Returns:
        Attention output, shape (B, n_q_heads, 1, dim).

    Example:
        >>> cache = TurboKVCache(bits=4, key_bits=0)  # K=FP16, V=turbo4
        >>> out = turbo_fused_asymmetric_attention_single_dispatch(
        ...     q, cache._fp_keys, cache._packed_values,
        ...     cache._value_norms, dim=128)
    """
    if bits != 4:
        raise NotImplementedError(
            f"Fused asymmetric attention only supports 4-bit V, got {bits}-bit"
        )
    if dim > 256:
        raise ValueError(
            f"Fused asymmetric attention supports dim <= 256, got dim={dim}. "
            "The kernel uses fixed-size shared memory arrays."
        )

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    B = queries.shape[0]
    n_q_heads = queries.shape[1]
    T_kv = fp_keys.shape[2]
    n_kv_heads = fp_keys.shape[1]
    packed_dim = packed_values.shape[-1]

    # GQA factor: how many query heads per KV head
    gqa_factor = n_q_heads // n_kv_heads if n_kv_heads < n_q_heads else 1

    # Get codebook centroids and sign vectors
    cb = _get_codebook(bits, dim)
    signs1 = _sign_flip_vector(dim, seed)
    signs2 = _sign_flip_vector2(dim, seed)

    # Flatten inputs for kernel
    n_bq = B * n_q_heads
    n_bkv = B * n_kv_heads

    # Queries: (B, nq, 1, dim) -> (B*nq, dim)
    q_flat = queries.reshape(n_bq, dim).astype(mx.float32)

    # Keys: (B, nkv, T_kv, dim) -> (B*nkv, T_kv, dim)
    k_flat = fp_keys.reshape(n_bkv, T_kv, dim).astype(mx.float32)

    # Packed V: (B, nkv, T_kv, packed_dim) -> (B*nkv, T_kv, packed_dim)
    pv_flat = packed_values.reshape(n_bkv, T_kv, packed_dim)

    # V norms: (B, nkv, T_kv, 1) -> (B*nkv, T_kv)
    vn_flat = value_norms.squeeze(-1).reshape(n_bkv, T_kv).astype(mx.float32)

    # Encode scale as uint32 for integer param array
    import struct
    scale_as_uint32 = struct.unpack('I', struct.pack('f', scale))[0]
    params = mx.array(
        [dim, T_kv, packed_dim, scale_as_uint32, gqa_factor], dtype=mx.uint32
    )

    kernel = _get_fused_asymmetric_kernel(bits)

    # Threadgroup size: at least `dim` threads (for WHT butterfly),
    # at least 1 SIMD group. dim=128 is the sweet spot — matches WHT
    # thread count, minimizes register pressure (128 * 256 floats = 128KB),
    # and shared_reduce fits in 4KB (4 SIMD groups * 256 * 4 bytes).
    tg_size = max(dim, 64)
    tg_size = ((tg_size + 31) // 32) * 32  # round to SIMD boundary
    tg_size = min(tg_size, 256)  # Metal threadgroup limit

    outputs = kernel(
        inputs=[
            q_flat,           # queries
            k_flat,           # fp_keys
            pv_flat,          # packed_v
            vn_flat,          # v_norms
            cb.centroids,     # centroids
            signs1,           # signs1
            signs2,           # signs2
            params,           # params
        ],
        output_shapes=[
            (n_bq, dim),      # output — final attention result
            (n_bq, T_kv),     # out_scores — scratch for scores/weights
        ],
        output_dtypes=[mx.float32, mx.float32],
        grid=(n_bq * tg_size, 1, 1),
        threadgroup=(tg_size, 1, 1),
        init_value=0.0,
        stream=mx.gpu,
    )

    output = outputs[0].reshape(B, n_q_heads, 1, dim)
    return output.astype(queries.dtype)


# ---------------------------------------------------------------------------
# Two-pass fused asymmetric attention (Eric Kryski's TurboFlash architecture)
# ---------------------------------------------------------------------------
#
# Pass 1: Each threadgroup processes a block of B=64 KV tokens in parallel.
#   - Computes Q·K scores for the block
#   - Online softmax within the block (running max + exp sum)
#   - Weighted V sum (centroid unpack from packed turbo4 V)
#   - Stores per-block partial results: {partial_output[dim], block_max, block_sum}
#
# Pass 2: One threadgroup per query head merges all block partial results.
#   - Online softmax merge across blocks (combine per-block max/sum → global)
#   - Rescale partial outputs by correction factor
#   - Sum into final output
#   - Apply inverse WHT rotation (butterfly + sign flips)
#
# Why two-pass beats single-pass: the single-pass kernel assigns one threadgroup
# per query head, serializing ALL T_kv tokens. With two-pass at B=64:
#   Pass 1: T/64 threadgroups run in parallel (massive GPU occupancy)
#   Pass 2: 1 threadgroup merges T/64 partial results (trivially fast)
#
# Eric benchmarked B=32, B=64, B=128 — B=64 is optimal for Apple Silicon.

_TWO_PASS_BLOCK_SIZE = 64

_TWO_PASS_ATTN_P1_SOURCE_4BIT = """
    // Two-pass fused asymmetric attention — PASS 1 (block scoring + partial V)
    //
    // Grid: (n_bh, n_blocks, 1)
    //   n_bh = B * n_q_heads
    //   n_blocks = ceil(T_kv / BLOCK_SIZE)
    //
    // Threadgroup: (BLOCK_SIZE, 1, 1) where BLOCK_SIZE=64
    //
    // Each threadgroup processes one block of 64 KV tokens for one query head.
    // Thread tid handles token t = block_id * BLOCK_SIZE + tid.
    //
    // Outputs per-block partial results:
    //   partial_out[bh, block, dim]  — weighted V accumulator for this block
    //   partial_max[bh, block]       — block max score (for softmax merge)
    //   partial_sum[bh, block]       — block exp sum (for softmax merge)

    uint tid = thread_position_in_threadgroup.x;
    uint bh_idx = threadgroup_position_in_grid.x;   // batch-head index
    uint block_id = threadgroup_position_in_grid.y;  // which block of 64 tokens

    int dim_val = params[0];
    int T_kv = params[1];
    int packed_dim = params[2];
    float scale = as_type<float>(params[3]);
    int gqa_factor = params[4];
    int BLOCK_SIZE = params[5];
    int n_blocks = params[6];

    uint kv_head = bh_idx / gqa_factor;

    int q_base  = bh_idx * dim_val;
    int k_base  = kv_head * T_kv * dim_val;
    int pv_base = kv_head * T_kv * packed_dim;
    int vn_base = kv_head * T_kv;

    uint simd_lane = thread_index_in_simdgroup;
    uint simd_id = tid / 32;

    // Shared memory for query cache and SIMD reductions
    threadgroup float shared_q[256];
    threadgroup float shared_scratch[4];    // max 2 SIMD groups for B=64
    threadgroup float shared_v_reduce[2 * 256];  // 2 SIMD groups * max dim

    // Load query into shared memory (stride loop: BLOCK_SIZE threads load dim elements)
    for (int d = (int)tid; d < dim_val; d += BLOCK_SIZE) {
        shared_q[d] = queries[q_base + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Token index for this thread
    int t = (int)(block_id * BLOCK_SIZE + tid);

    // ====== Q·K score for this token ======
    float score = -INFINITY;
    if (t < T_kv) {
        float dot = 0.0f;
        int k_offset = k_base + t * dim_val;

        // Unrolled 4x dot product
        int d = 0;
        for (; d + 3 < dim_val; d += 4) {
            dot += shared_q[d]     * fp_keys[k_offset + d];
            dot += shared_q[d + 1] * fp_keys[k_offset + d + 1];
            dot += shared_q[d + 2] * fp_keys[k_offset + d + 2];
            dot += shared_q[d + 3] * fp_keys[k_offset + d + 3];
        }
        for (; d < dim_val; d++) {
            dot += shared_q[d] * fp_keys[k_offset + d];
        }
        score = dot * scale;
    }

    // ====== Online softmax within block ======
    // Step 1: find block max via SIMD reduction
    float block_max = simd_max(score);
    if (simd_lane == 0) shared_scratch[simd_id] = block_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float gmax = shared_scratch[0];
        uint n_simd_groups = (BLOCK_SIZE + 31) / 32;
        for (uint s = 1; s < n_simd_groups; s++) gmax = max(gmax, shared_scratch[s]);
        shared_scratch[0] = gmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    block_max = shared_scratch[0];

    // Step 2: exp and sum
    float exp_score = (t < T_kv) ? exp(score - block_max) : 0.0f;
    float block_sum = simd_sum(exp_score);
    if (simd_lane == 0) shared_scratch[simd_id] = block_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float gsum = 0.0f;
        uint n_simd_groups = (BLOCK_SIZE + 31) / 32;
        for (uint s = 0; s < n_simd_groups; s++) gsum += shared_scratch[s];
        shared_scratch[0] = gsum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_sum = shared_scratch[0];

    // Weight for this token (unnormalized — we store block_sum for pass 2 to normalize)
    float weight = exp_score;  // NOT divided by total_sum — pass 2 handles global normalization

    // ====== V weighted sum: unpack turbo4 and accumulate ======
    // Each thread unpacks V[t] for its token and scales by weight
    float v_local[256];  // register file — one per dim element
    for (int dd = 0; dd < dim_val; dd++) v_local[dd] = 0.0f;

    if (t < T_kv && weight > 0.0f) {
        float wn = weight * v_norms[vn_base + t];
        int pv_offset = pv_base + t * packed_dim;

        for (int pw = 0; pw < packed_dim; pw++) {
            uint word = packed_v[pv_offset + pw];
            int base_d = pw * 8;

            for (int j = 0; j < 8 && (base_d + j) < dim_val; j++) {
                uint idx = (word >> (j * 4)) & 0xF;
                v_local[base_d + j] = wn * centroids[idx];
            }
        }
    }

    // ====== Reduce V accumulators across threads in block (SIMD + shared mem) ======
    // With BLOCK_SIZE=64 and dim up to 256, we may have fewer threads than dims.
    // Each SIMD group reduces internally via simd_sum, then we merge across groups.
    uint n_simd_groups = (BLOCK_SIZE + 31) / 32;
    for (int dd = 0; dd < dim_val; dd++) {
        float val = simd_sum(v_local[dd]);
        if (simd_lane == 0) shared_v_reduce[simd_id * dim_val + dd] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stride loop: each thread writes multiple dim elements (BLOCK_SIZE threads, dim dims)
    for (int dd = (int)tid; dd < dim_val; dd += BLOCK_SIZE) {
        float total = 0.0f;
        for (uint s = 0; s < n_simd_groups; s++) {
            total += shared_v_reduce[s * dim_val + dd];
        }
        // partial_out[bh_idx, block_id, dd]
        partial_out[bh_idx * n_blocks * dim_val + block_id * dim_val + dd] = total;
    }

    // Thread 0 writes block_max and block_sum
    if (tid == 0) {
        partial_max[bh_idx * n_blocks + block_id] = block_max;
        partial_sum[bh_idx * n_blocks + block_id] = total_sum;
    }
"""

_TWO_PASS_ATTN_P2_SOURCE_4BIT = """
    // Two-pass fused asymmetric attention — PASS 2 (merge + inverse WHT)
    //
    // Grid: (n_bh, 1, 1)
    // Threadgroup: (tg_size, 1, 1) where tg_size >= dim (for WHT butterfly)
    //
    // Merges partial results from Pass 1 across all blocks using online softmax
    // correction, then applies inverse WHT rotation.

    uint tid = thread_position_in_threadgroup.x;
    uint bh_idx = threadgroup_position_in_grid.x;

    int dim_val = params[0];
    int n_blocks = params[1];

    // Shared memory for merge + WHT
    threadgroup float shared_out[256];

    // ====== Step 1: Find global max across all blocks ======
    // Single thread scans — n_blocks is typically small (T/64)
    threadgroup float shared_global_max[1];
    threadgroup float shared_global_sum[1];

    if (tid == 0) {
        float gmax = -INFINITY;
        for (int b = 0; b < n_blocks; b++) {
            gmax = max(gmax, partial_max[bh_idx * n_blocks + b]);
        }
        shared_global_max[0] = gmax;

        // Compute global exp sum with correction
        float gsum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            float correction = exp(partial_max[bh_idx * n_blocks + b] - gmax);
            gsum += correction * partial_sum[bh_idx * n_blocks + b];
        }
        shared_global_sum[0] = gsum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = shared_global_max[0];
    float global_sum = shared_global_sum[0];
    float inv_global_sum = 1.0f / global_sum;

    // ====== Step 2: Merge partial outputs with softmax correction ======
    if ((int)tid < dim_val) {
        float accum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            float correction = exp(partial_max[bh_idx * n_blocks + b] - global_max);
            float block_val = partial_out[bh_idx * n_blocks * dim_val + b * dim_val + tid];
            accum += correction * block_val;
        }
        // Normalize by global softmax sum
        shared_out[tid] = accum * inv_global_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ====== Step 3: Inverse WHT rotation ======
    if ((int)tid < dim_val) {
        shared_out[tid] *= signs2[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)tid < dim_val) {
        uint log2_dim = 0;
        for (uint dd = dim_val; dd > 1; dd >>= 1) log2_dim++;

        for (uint stage = 0; stage < log2_dim; stage++) {
            uint half_block = 1u << stage;
            uint bfly_size = half_block << 1;
            uint bfly_idx = tid / bfly_size;
            uint local_idx = tid % bfly_size;
            uint base_idx = bfly_idx * bfly_size;

            float a = shared_out[base_idx + (local_idx % half_block)];
            float b = shared_out[base_idx + (local_idx % half_block) + half_block];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            shared_out[tid] = (local_idx < half_block) ? (a + b) : (a - b);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float inv_sqrt_dim = rsqrt((float)dim_val);
        output[bh_idx * dim_val + tid] = shared_out[tid] * inv_sqrt_dim * signs1[tid];
    }
"""

# Cache for the two-pass kernels
_two_pass_kernel_cache: Dict[str, object] = {}


def _get_two_pass_kernels(bits: int = 4):
    """Get or compile the two-pass asymmetric attention kernels (pass1 + pass2)."""
    cache_key = f"two_pass_{bits}bit"
    if cache_key in _two_pass_kernel_cache:
        return _two_pass_kernel_cache[cache_key]

    if bits != 4:
        raise NotImplementedError(
            f"Two-pass asymmetric attention only supports 4-bit V, got {bits}-bit"
        )

    # Pass 1: block scoring + partial V accumulation
    pass1_kernel = mx.fast.metal_kernel(
        name="turbo_two_pass_p1_4bit",
        input_names=[
            "queries",     # [B*nq, dim] float32
            "fp_keys",     # [B*nkv, T_kv, dim] float16/32
            "packed_v",    # [B*nkv, T_kv, packed_dim] uint32
            "v_norms",     # [B*nkv, T_kv] float32
            "centroids",   # [16] float32
            "params",      # [7] uint32 — dim, T_kv, packed_dim, scale, gqa, block_size, n_blocks
        ],
        output_names=[
            "partial_out",  # [n_bh, n_blocks, dim] float32
            "partial_max",  # [n_bh, n_blocks] float32
            "partial_sum",  # [n_bh, n_blocks] float32
        ],
        header=_FUSED_ASYMMETRIC_ATTN_HEADER,
        source=_TWO_PASS_ATTN_P1_SOURCE_4BIT,
        ensure_row_contiguous=True,
        atomic_outputs=False,
    )

    # Pass 2: merge partials + inverse WHT
    pass2_kernel = mx.fast.metal_kernel(
        name="turbo_two_pass_p2_4bit",
        input_names=[
            "partial_out",  # [n_bh, n_blocks, dim] float32
            "partial_max",  # [n_bh, n_blocks] float32
            "partial_sum",  # [n_bh, n_blocks] float32
            "signs1",       # [dim] float32
            "signs2",       # [dim] float32
            "params",       # [2] uint32 — dim, n_blocks
        ],
        output_names=[
            "output",       # [n_bh, dim] float32
        ],
        header="",
        source=_TWO_PASS_ATTN_P2_SOURCE_4BIT,
        ensure_row_contiguous=True,
        atomic_outputs=False,
    )

    _two_pass_kernel_cache[cache_key] = (pass1_kernel, pass2_kernel)
    return pass1_kernel, pass2_kernel


def turbo_two_pass_asymmetric_attention(
    queries: mx.array,
    fp_keys: mx.array,
    packed_values: mx.array,
    value_norms: mx.array,
    dim: int,
    bits: int = 4,
    seed: int = 42,
    scale: Optional[float] = None,
    block_size: int = _TWO_PASS_BLOCK_SIZE,
) -> mx.array:
    """Two-pass fused asymmetric attention: K=FP16, V=turbo4 (TurboFlash).

    Eric Kryski's TurboFlash architecture: two-pass with B=64 blocks.
    Pass 1 runs T/64 threadgroups in parallel (one per block), each computing
    partial softmax + weighted V sum. Pass 2 merges block results with online
    softmax correction and applies inverse WHT.

    Scales much better than single-pass because the GPU can run all pass-1
    blocks in parallel instead of serializing T_kv iterations in one threadgroup.

    Args:
        queries: Query tensor, shape (B, n_q_heads, 1, dim).
        fp_keys: Raw FP16 keys, shape (B, n_kv_heads, T_kv, dim).
        packed_values: Packed V indices, shape (B, n_kv_heads, T_kv, packed_dim).
        value_norms: V norms, shape (B, n_kv_heads, T_kv, 1).
        dim: Head dimension (must be power of 2, max 256).
        bits: V quantization bit-width. Default: 4.
        seed: SRHT seed. Default: 42.
        scale: Attention scale. Default: 1/sqrt(dim).
        block_size: KV tokens per block. Default: 64 (Eric's optimal).

    Returns:
        Attention output, shape (B, n_q_heads, 1, dim).

    Example:
        >>> cache = TurboKVCache(bits=4, key_bits=0)  # K=FP16, V=turbo4
        >>> out = turbo_two_pass_asymmetric_attention(
        ...     q, cache._fp_keys, cache._packed_values,
        ...     cache._value_norms, dim=128)
    """
    if bits != 4:
        raise NotImplementedError(
            f"Two-pass asymmetric attention only supports 4-bit V, got {bits}-bit"
        )
    if dim > 256:
        raise ValueError(
            f"Two-pass asymmetric attention supports dim <= 256, got dim={dim}. "
            "The kernel uses fixed-size shared memory arrays."
        )

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    B = queries.shape[0]
    n_q_heads = queries.shape[1]
    T_kv = fp_keys.shape[2]
    n_kv_heads = fp_keys.shape[1]
    packed_dim = packed_values.shape[-1]

    # GQA factor: how many query heads per KV head
    gqa_factor = n_q_heads // n_kv_heads if n_kv_heads < n_q_heads else 1

    # Get codebook centroids and sign vectors
    cb = _get_codebook(bits, dim)
    signs1 = _sign_flip_vector(dim, seed)
    signs2 = _sign_flip_vector2(dim, seed)

    # Flatten inputs for kernel
    n_bq = B * n_q_heads
    n_bkv = B * n_kv_heads
    n_blocks = (T_kv + block_size - 1) // block_size

    # Queries: (B, nq, 1, dim) -> (B*nq, dim)
    q_flat = queries.reshape(n_bq, dim).astype(mx.float32)

    # Keys: (B, nkv, T_kv, dim) -> (B*nkv, T_kv, dim)
    k_flat = fp_keys.reshape(n_bkv, T_kv, dim).astype(mx.float32)

    # Packed V: (B, nkv, T_kv, packed_dim) -> (B*nkv, T_kv, packed_dim)
    pv_flat = packed_values.reshape(n_bkv, T_kv, packed_dim)

    # V norms: (B, nkv, T_kv, 1) -> (B*nkv, T_kv)
    vn_flat = value_norms.squeeze(-1).reshape(n_bkv, T_kv).astype(mx.float32)

    # Encode scale as uint32 for integer param array
    import struct
    scale_as_uint32 = struct.unpack('I', struct.pack('f', scale))[0]

    # Pass 1 params: dim, T_kv, packed_dim, scale, gqa_factor, block_size, n_blocks
    p1_params = mx.array(
        [dim, T_kv, packed_dim, scale_as_uint32, gqa_factor, block_size, n_blocks],
        dtype=mx.uint32,
    )

    pass1_kernel, pass2_kernel = _get_two_pass_kernels(bits)

    # === Pass 1: block scoring + partial V accumulation ===
    # Grid: (n_bq, n_blocks, 1) — one threadgroup per (query_head, block) pair
    # Threadgroup: (block_size, 1, 1) — one thread per KV token in block
    p1_outputs = pass1_kernel(
        inputs=[
            q_flat,           # queries
            k_flat,           # fp_keys
            pv_flat,          # packed_v
            vn_flat,          # v_norms
            cb.centroids,     # centroids
            p1_params,        # params
        ],
        output_shapes=[
            (n_bq, n_blocks, dim),   # partial_out
            (n_bq, n_blocks),        # partial_max
            (n_bq, n_blocks),        # partial_sum
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(n_bq * block_size, n_blocks, 1),
        threadgroup=(block_size, 1, 1),
        init_value=0.0,
        stream=mx.gpu,
    )

    partial_out = p1_outputs[0]
    partial_max = p1_outputs[1]
    partial_sum = p1_outputs[2]

    # === Pass 2: merge partials + inverse WHT ===
    # Grid: (n_bq, 1, 1) — one threadgroup per query head
    # Threadgroup: (tg_size, 1, 1) — at least dim threads for WHT butterfly
    tg_size = max(dim, 64)
    tg_size = ((tg_size + 31) // 32) * 32  # round to SIMD boundary
    tg_size = min(tg_size, 256)

    # Pass 2 params: dim, n_blocks
    p2_params = mx.array([dim, n_blocks], dtype=mx.uint32)

    p2_outputs = pass2_kernel(
        inputs=[
            partial_out,   # partial_out
            partial_max,   # partial_max
            partial_sum,   # partial_sum
            signs1,        # signs1
            signs2,        # signs2
            p2_params,     # params
        ],
        output_shapes=[
            (n_bq, dim),   # output
        ],
        output_dtypes=[mx.float32],
        grid=(n_bq * tg_size, 1, 1),
        threadgroup=(tg_size, 1, 1),
        init_value=0.0,
        stream=mx.gpu,
    )

    output = p2_outputs[0].reshape(B, n_q_heads, 1, dim)
    return output.astype(queries.dtype)


# ---------------------------------------------------------------------------
# Fused compressed-domain attention (Metal kernel — no FP16 materialization)
# ---------------------------------------------------------------------------

# The Metal kernel operates in the WHT (rotated) domain with dual signs:
#   1. Python pre-rotates Q: Q_rot = signs2 * WHT(signs1 * Q)   — once per query
#   2. Kernel: for each KV token, unpack indices → centroid lookup → dot product
#      with Q_rot → softmax → centroid lookup for V → weighted sum
#   3. Python post-rotates output: out = signs1 * WHT(signs2 * accum)   — once
#
# This avoids materializing FP16 K/V entirely. The dual WHT and sign-flips are
# linear operators applied once to Q and once to the output, NOT per-KV-token.
# Memory bandwidth: reads packed uint32 indices + norms (4-bit: 1/8th of FP16).
# Compute: centroid lookup is a 16-entry table lookup, trivially fast.

# --- Metal kernel source for 4-bit compressed-domain attention ---
# One threadgroup per (batch, head) pair. Each threadgroup processes all T_kv
# tokens for one query head. Within the threadgroup:
#   - Phase 1 (scores): Each thread handles a range of KV tokens. For each
#     token, unpack all dim indices, lookup centroids, dot with Q_rot.
#   - Phase 2 (softmax): Parallel reduce max + sum for numerically stable softmax.
#   - Phase 3 (V weighted sum): Same unpack + lookup, multiply by attn weight,
#     accumulate across tokens.
#
# Thread layout: threadgroup_size threads, each handles ceil(T_kv / tg_size) tokens.

_TURBO_ATTN_HEADER = """
// Centroid table — embedded as constant array for minimal latency.
// Loaded into registers at kernel launch, no memory fetch during inner loop.
// These are the Beta-distribution centroids for (4-bit, 128-dim).
// For other (bits, dim) combos, the Python wrapper passes the correct table.

// Inline unpack: extract a 4-bit index from a uint32 word
inline uint unpack4(uint word, uint pos) {
    return (word >> (pos * 4)) & 0xF;
}

// Inline unpack: extract a 3-bit index from a uint32 word
inline uint unpack3(uint word, uint pos) {
    return (word >> (pos * 3)) & 0x7;
}

// Inline unpack: extract a 2-bit index from a uint32 word
inline uint unpack2(uint word, uint pos) {
    return (word >> (pos * 2)) & 0x3;
}
"""

_TURBO_ATTN_SOURCE_4BIT = """
    // Grid: (B * n_heads, 1, 1)  — one threadgroup per (batch, head) pair
    // Threadgroup: (TG_SIZE, 1, 1) where TG_SIZE divides work across T_kv tokens
    //
    // Inputs (row-contiguous, flattened to B*n_heads leading dim):
    //   q_rot:        [n_bh, dim]               — pre-rotated query (WHT domain)
    //   packed_k:     [n_bh, T_kv, packed_dim]  — packed 4-bit K indices
    //   k_norms:      [n_bh, T_kv]             — K L2 norms (already squeezed)
    //   packed_v:     [n_bh, T_kv, packed_dim]  — packed 4-bit V indices
    //   v_norms:      [n_bh, T_kv]             — V L2 norms (already squeezed)
    //   centroids:    [n_levels]                 — centroid lookup table
    //   params:       [4]                        — {dim, T_kv, packed_dim, scale_bits}
    //
    // Outputs (device buffers, indexed by bh_idx):
    //   out_accum:    [n_bh, dim]               — WHT-domain weighted sum
    //   scores:       [n_bh, T_kv]             — scratch for attention scores
    //   simd_maxes:   [n_bh, n_simd_groups]    — scratch for simd reductions

    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint bh_idx = threadgroup_position_in_grid.x;  // batch*head index

    // Read params
    int dim = params[0];
    int T_kv = params[1];
    int packed_dim = params[2];
    float scale = as_type<float>(params[3]);

    // Compute base offsets into the flattened buffers for this (batch, head)
    int q_offset = bh_idx * dim;
    int kv_base = bh_idx * T_kv;
    int pk_base = bh_idx * T_kv * packed_dim;
    int pv_base = bh_idx * T_kv * packed_dim;

    // scores and simd_maxes are global device buffers indexed per-threadgroup
    int scores_base = bh_idx * T_kv;

    uint simd_lane = thread_index_in_simdgroup;
    uint simd_id = tid / threads_per_simdgroup;
    uint n_simd = (tg_size + threads_per_simdgroup - 1) / threads_per_simdgroup;
    int smaxes_base = bh_idx * n_simd;

    // --- Phase 1: Compute Q·K scores for all T_kv tokens ---
    // Each thread processes a strided subset of tokens.
    // For each assigned token:
    //   score = norm_K * sum_d(Q_rot[d] * centroids[K_indices[d]]) * scale
    float max_score = -INFINITY;

    for (int t = tid; t < T_kv; t += tg_size) {
        float dot = 0.0f;
        int pk_offset = pk_base + t * packed_dim;

        for (int w = 0; w < packed_dim; w++) {
            uint word = packed_k[pk_offset + w];
            int base_d = w * 8;  // 8 indices per uint32 for 4-bit

            // Unroll 8 indices per word
            for (int j = 0; j < 8 && (base_d + j) < dim; j++) {
                uint idx = (word >> (j * 4)) & 0xF;
                float c = centroids[idx];
                dot += q_rot[q_offset + base_d + j] * c;
            }
        }

        float norm_k = k_norms[kv_base + t];
        float s = dot * norm_k * scale;
        scores[scores_base + t] = s;
        max_score = max(max_score, s);
    }

    // --- Phase 2: Softmax (parallel reduction) ---
    // Step 2a: reduce max across threadgroup via simd_max + cross-simd reduce
    max_score = simd_max(max_score);

    if (simd_lane == 0) {
        simd_maxes[smaxes_base + simd_id] = max_score;
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        float global_max = simd_maxes[smaxes_base];
        for (uint s = 1; s < n_simd; s++) {
            global_max = max(global_max, simd_maxes[smaxes_base + s]);
        }
        simd_maxes[smaxes_base] = global_max;
    }
    threadgroup_barrier(mem_flags::mem_device);
    float global_max = simd_maxes[smaxes_base];

    // Step 2b: compute exp(score - max) and local sum
    float local_sum = 0.0f;
    for (int t = tid; t < T_kv; t += tg_size) {
        float e = exp(scores[scores_base + t] - global_max);
        scores[scores_base + t] = e;
        local_sum += e;
    }

    // Reduce sum across threadgroup
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) {
        simd_maxes[smaxes_base + simd_id] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        float global_sum = 0.0f;
        for (uint s = 0; s < n_simd; s++) {
            global_sum += simd_maxes[smaxes_base + s];
        }
        simd_maxes[smaxes_base] = global_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);
    float inv_sum = 1.0f / simd_maxes[smaxes_base];

    // Normalize scores to attention weights
    for (int t = tid; t < T_kv; t += tg_size) {
        scores[scores_base + t] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // --- Phase 3: V weighted sum ---
    // Each thread accumulates its share of V tokens weighted by attention.
    // Local accumulator in registers — 128 floats = 512 bytes, fits easily.
    float v_accum[256];  // Max supported dim (Metal needs fixed-size arrays)
    for (int d = 0; d < dim; d++) {
        v_accum[d] = 0.0f;
    }

    for (int t = tid; t < T_kv; t += tg_size) {
        float attn_w = scores[scores_base + t];

        // Skip near-zero attention weights (fused sparse V optimization)
        if (attn_w < 1e-6f) continue;

        float norm_v = v_norms[kv_base + t];
        float w = attn_w * norm_v;

        int pv_offset = pv_base + t * packed_dim;
        for (int pw = 0; pw < packed_dim; pw++) {
            uint word = packed_v[pv_offset + pw];
            int base_d = pw * 8;

            for (int j = 0; j < 8 && (base_d + j) < dim; j++) {
                uint idx = (word >> (j * 4)) & 0xF;
                float c = centroids[idx];
                v_accum[base_d + j] += w * c;
            }
        }
    }

    // --- Phase 4: Reduce V accumulators across threads ---
    // Use simd_sum per dimension, then cross-simd reduce via device buffer.
    for (int d = 0; d < dim; d++) {
        float val = simd_sum(v_accum[d]);

        if (simd_lane == 0) {
            simd_maxes[smaxes_base + simd_id] = val;
        }
        threadgroup_barrier(mem_flags::mem_device);

        if (tid == 0) {
            float total = 0.0f;
            for (uint s = 0; s < n_simd; s++) {
                total += simd_maxes[smaxes_base + s];
            }
            out_accum[q_offset + d] = total;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
"""

# --- NR0=2 Multi-Row Amortization kernel ---
# Processes 2 queries per dispatch, sharing K/V dequant (centroid lookup + norm
# multiply) across both queries. Halves the memory bandwidth cost of reading
# packed K/V data. Expected speedup: ~30-40% for decode with 2+ pending queries.
#
# Layout: grid dispatches one threadgroup per (batch, head) pair, same as NR0=1.
# Each thread computes scores and V accumulation for BOTH queries simultaneously.

_TURBO_ATTN_SOURCE_4BIT_NR0_2 = """
    // Grid: (B * n_heads, 1, 1)  — one threadgroup per (batch, head) pair
    // Threadgroup: (TG_SIZE, 1, 1)
    //
    // Inputs:
    //   q_rot:        [n_bh, NR0, dim]          — NR0=2 pre-rotated queries (WHT domain)
    //   packed_k:     [n_bh, T_kv, packed_dim]   — packed 4-bit K indices
    //   k_norms:      [n_bh, T_kv]               — K L2 norms
    //   packed_v:     [n_bh, T_kv, packed_dim]   — packed 4-bit V indices
    //   v_norms:      [n_bh, T_kv]               — V L2 norms
    //   centroids:    [n_levels]                   — centroid lookup table
    //   params:       [4]                          — {dim, T_kv, packed_dim, scale_bits}
    //
    // Outputs:
    //   out_accum:    [n_bh, NR0, dim]           — WHT-domain weighted sums
    //   scores:       [n_bh, NR0, T_kv]          — scratch for attention scores
    //   simd_maxes:   [n_bh, NR0, n_simd_groups] — scratch for simd reductions

    uint tid = thread_position_in_threadgroup.x;
    uint tg_size = threads_per_threadgroup.x;
    uint bh_idx = threadgroup_position_in_grid.x;

    int dim = params[0];
    int T_kv = params[1];
    int packed_dim = params[2];
    float scale = as_type<float>(params[3]);

    // Base offsets — NR0=2 queries packed contiguously per (batch, head)
    int q_base = bh_idx * 2 * dim;      // q_rot[bh_idx, 0..1, :]
    int kv_base = bh_idx * T_kv;
    int pk_base = bh_idx * T_kv * packed_dim;
    int pv_base = bh_idx * T_kv * packed_dim;

    // Score buffers: [n_bh, 2, T_kv]
    int scores_base_0 = bh_idx * 2 * T_kv;
    int scores_base_1 = scores_base_0 + T_kv;

    uint simd_lane = thread_index_in_simdgroup;
    uint simd_id = tid / threads_per_simdgroup;
    uint n_simd = (tg_size + threads_per_simdgroup - 1) / threads_per_simdgroup;
    // simd_maxes: [n_bh, 2, n_simd]
    int smaxes_base_0 = bh_idx * 2 * n_simd;
    int smaxes_base_1 = smaxes_base_0 + n_simd;

    // --- Phase 1: Compute Q·K scores for BOTH queries, sharing K dequant ---
    float max_score_0 = -INFINITY;
    float max_score_1 = -INFINITY;

    for (int t = tid; t < T_kv; t += tg_size) {
        float dot_0 = 0.0f;
        float dot_1 = 0.0f;
        int pk_offset = pk_base + t * packed_dim;

        for (int w = 0; w < packed_dim; w++) {
            uint word = packed_k[pk_offset + w];  // Shared K dequant — read once
            int base_d = w * 8;

            for (int j = 0; j < 8 && (base_d + j) < dim; j++) {
                uint idx = (word >> (j * 4)) & 0xF;
                float c = centroids[idx];  // Shared centroid lookup
                int d = base_d + j;
                dot_0 += q_rot[q_base + d] * c;           // Query 0
                dot_1 += q_rot[q_base + dim + d] * c;     // Query 1
            }
        }

        float norm_k = k_norms[kv_base + t];  // Shared norm
        float s0 = dot_0 * norm_k * scale;
        float s1 = dot_1 * norm_k * scale;
        scores[scores_base_0 + t] = s0;
        scores[scores_base_1 + t] = s1;
        max_score_0 = max(max_score_0, s0);
        max_score_1 = max(max_score_1, s1);
    }

    // --- Phase 2: Softmax for both queries (parallel reduction) ---
    // 2a: Reduce max across threadgroup for query 0
    max_score_0 = simd_max(max_score_0);
    max_score_1 = simd_max(max_score_1);

    if (simd_lane == 0) {
        simd_maxes[smaxes_base_0 + simd_id] = max_score_0;
        simd_maxes[smaxes_base_1 + simd_id] = max_score_1;
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        float gmax_0 = simd_maxes[smaxes_base_0];
        float gmax_1 = simd_maxes[smaxes_base_1];
        for (uint s = 1; s < n_simd; s++) {
            gmax_0 = max(gmax_0, simd_maxes[smaxes_base_0 + s]);
            gmax_1 = max(gmax_1, simd_maxes[smaxes_base_1 + s]);
        }
        simd_maxes[smaxes_base_0] = gmax_0;
        simd_maxes[smaxes_base_1] = gmax_1;
    }
    threadgroup_barrier(mem_flags::mem_device);
    float global_max_0 = simd_maxes[smaxes_base_0];
    float global_max_1 = simd_maxes[smaxes_base_1];

    // 2b: exp(score - max) and sum
    float local_sum_0 = 0.0f;
    float local_sum_1 = 0.0f;
    for (int t = tid; t < T_kv; t += tg_size) {
        float e0 = exp(scores[scores_base_0 + t] - global_max_0);
        float e1 = exp(scores[scores_base_1 + t] - global_max_1);
        scores[scores_base_0 + t] = e0;
        scores[scores_base_1 + t] = e1;
        local_sum_0 += e0;
        local_sum_1 += e1;
    }

    local_sum_0 = simd_sum(local_sum_0);
    local_sum_1 = simd_sum(local_sum_1);
    if (simd_lane == 0) {
        simd_maxes[smaxes_base_0 + simd_id] = local_sum_0;
        simd_maxes[smaxes_base_1 + simd_id] = local_sum_1;
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        float gsum_0 = 0.0f, gsum_1 = 0.0f;
        for (uint s = 0; s < n_simd; s++) {
            gsum_0 += simd_maxes[smaxes_base_0 + s];
            gsum_1 += simd_maxes[smaxes_base_1 + s];
        }
        simd_maxes[smaxes_base_0] = gsum_0;
        simd_maxes[smaxes_base_1] = gsum_1;
    }
    threadgroup_barrier(mem_flags::mem_device);
    float inv_sum_0 = 1.0f / simd_maxes[smaxes_base_0];
    float inv_sum_1 = 1.0f / simd_maxes[smaxes_base_1];

    // Normalize scores
    for (int t = tid; t < T_kv; t += tg_size) {
        scores[scores_base_0 + t] *= inv_sum_0;
        scores[scores_base_1 + t] *= inv_sum_1;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // --- Phase 3: V weighted sum for BOTH queries, sharing V dequant ---
    float v_accum_0[256];
    float v_accum_1[256];
    for (int d = 0; d < dim; d++) {
        v_accum_0[d] = 0.0f;
        v_accum_1[d] = 0.0f;
    }

    for (int t = tid; t < T_kv; t += tg_size) {
        float attn_w_0 = scores[scores_base_0 + t];
        float attn_w_1 = scores[scores_base_1 + t];

        // Skip if BOTH queries have near-zero attention (fused sparse V)
        if (attn_w_0 < 1e-6f && attn_w_1 < 1e-6f) continue;

        float norm_v = v_norms[kv_base + t];  // Shared V norm
        float w_0 = attn_w_0 * norm_v;
        float w_1 = attn_w_1 * norm_v;

        int pv_offset = pv_base + t * packed_dim;
        for (int pw = 0; pw < packed_dim; pw++) {
            uint word = packed_v[pv_offset + pw];  // Shared V dequant
            int base_d = pw * 8;

            for (int j = 0; j < 8 && (base_d + j) < dim; j++) {
                uint idx = (word >> (j * 4)) & 0xF;
                float c = centroids[idx];  // Shared centroid lookup
                int d = base_d + j;
                v_accum_0[d] += w_0 * c;
                v_accum_1[d] += w_1 * c;
            }
        }
    }

    // --- Phase 4: Reduce V accumulators for BOTH queries ---
    int out_base_0 = bh_idx * 2 * dim;
    int out_base_1 = out_base_0 + dim;

    for (int d = 0; d < dim; d++) {
        float val_0 = simd_sum(v_accum_0[d]);
        float val_1 = simd_sum(v_accum_1[d]);

        if (simd_lane == 0) {
            simd_maxes[smaxes_base_0 + simd_id] = val_0;
            simd_maxes[smaxes_base_1 + simd_id] = val_1;
        }
        threadgroup_barrier(mem_flags::mem_device);

        if (tid == 0) {
            float total_0 = 0.0f, total_1 = 0.0f;
            for (uint s = 0; s < n_simd; s++) {
                total_0 += simd_maxes[smaxes_base_0 + s];
                total_1 += simd_maxes[smaxes_base_1 + s];
            }
            out_accum[out_base_0 + d] = total_0;
            out_accum[out_base_1 + d] = total_1;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
"""

# Cache the compiled kernel objects to avoid re-JIT on every call
_kernel_cache: Dict[str, object] = {}


def _get_turbo_attn_kernel(bits: int, nr0: int = 1):
    """Get or create the compressed-domain attention Metal kernel.

    The kernel is JIT-compiled once and cached. Currently supports 4-bit
    (the primary use case). 3-bit and 2-bit use the same kernel structure
    with different unpack widths.

    Args:
        bits: Quantization bit-width (2, 3, or 4).
        nr0: Number of queries to process per dispatch (1 or 2).
            NR0=2 shares K/V dequant across both queries, halving
            memory bandwidth for packed data reads.

    Returns:
        Compiled Metal kernel callable.
    """
    cache_key = f"turbo_attn_{bits}bit_nr{nr0}"
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    if bits == 4:
        if nr0 == 2:
            source = _TURBO_ATTN_SOURCE_4BIT_NR0_2
        else:
            source = _TURBO_ATTN_SOURCE_4BIT
    else:
        # TODO: Add 3-bit and 2-bit kernel variants
        raise NotImplementedError(
            f"Fused compressed-domain attention not yet implemented for {bits}-bit. "
            "Use turbo_attention() (decode-then-matmul) instead."
        )

    kernel = mx.fast.metal_kernel(
        name=f"turbo_sdpa_{bits}bit_nr{nr0}",
        input_names=[
            "q_rot",        # Pre-rotated query (WHT domain)
            "packed_k",     # Packed K indices
            "k_norms",      # K norms (flattened to 1D per-token)
            "packed_v",     # Packed V indices
            "v_norms",      # V norms (flattened to 1D per-token)
            "centroids",    # Centroid lookup table
            "params",       # {dim, T_kv, packed_dim, scale_as_uint32}
        ],
        output_names=[
            "out_accum",    # WHT-domain output (before inverse transform)
            "scores",       # Threadgroup-local score buffer
            "simd_maxes",   # Scratch for simd reductions
        ],
        header=_TURBO_ATTN_HEADER,
        source=source,
        ensure_row_contiguous=True,
        atomic_outputs=False,
    )

    _kernel_cache[cache_key] = kernel
    return kernel


def turbo_fused_attention(
    queries: mx.array,
    packed_keys: mx.array,
    key_norms: mx.array,
    packed_values: mx.array,
    value_norms: mx.array,
    dim: int,
    bits: int = 4,
    seed: int = 42,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
    nr0: Optional[int] = None,
) -> mx.array:
    """Compressed-domain attention — no FP16 K/V materialization.

    Uses a custom Metal kernel to compute attention directly on packed
    TurboQuant data. The key insight: the Walsh-Hadamard Transform (WHT) and
    sign-flip are linear operators that can be applied once to Q (before the
    kernel) and once to the output (after the kernel), rather than per-KV-token.

    In the WHT domain, each KV token is just a vector of centroid indices + a
    scalar norm. The dot product ``Q_rot @ centroids[indices] * norm`` replaces
    the full FP16 decode + matmul.

    **Performance characteristics:**
    - Memory: reads packed uint32 (4-bit: 1/8th of FP16 bandwidth)
    - Compute: centroid lookup (16-entry table, register-resident) + FMA
    - No intermediate FP16 K/V buffer allocated
    - Sparse V: skips dequant + accumulate for near-zero attention weights
    - NR0=2: shares K/V dequant across 2 queries, halving bandwidth cost

    **Limitations:**
    - Currently 4-bit only (3-bit and 2-bit planned)
    - T_q must be 1 or 2 (decode only — prefill uses standard SDPA)
    - T_kv limited by threadgroup memory (~16K tokens with 64KB tg mem)
    - mask not yet supported in the fused kernel (use decode-then-matmul path)

    Args:
        queries: Query tensor, shape (batch, heads, T_q, dim). T_q must be 1 or 2.
        packed_keys: Packed K indices, shape (batch, heads, T_kv, packed_dim).
        key_norms: K norms, shape (batch, heads, T_kv, 1).
        packed_values: Packed V indices, shape (batch, heads, T_kv, packed_dim).
        value_norms: V norms, shape (batch, heads, T_kv, 1).
        dim: Head dimension (must be power of 2, max 256).
        bits: Quantization bit-width. Default: 4.
        seed: SRHT random seed. Default: 42.
        scale: Attention scale factor. Default: 1/sqrt(dim).
        mask: NOT YET SUPPORTED in fused kernel. Must be None.
        nr0: Number of queries per dispatch (1 or 2). None = auto-select
            based on T_q. NR0=2 shares K/V dequant across both queries.

    Returns:
        Attention output, shape (batch, heads, T_q, dim).

    Raises:
        ValueError: If T_q > 2, mask is provided, or dim > 256.
        NotImplementedError: If bits != 4.

    Example:
        >>> q = mx.random.normal((1, 8, 1, 128))
        >>> k = mx.random.normal((1, 8, 64, 128))
        >>> pk, kn = turbo_encode(k, bits=4)
        >>> pv, vn = turbo_encode(k, bits=4)  # using k for demo
        >>> out = turbo_fused_attention(q, pk, kn, pv, vn, dim=128)
        >>> # NR0=2: process 2 queries sharing dequant work
        >>> q2 = mx.random.normal((1, 8, 2, 128))
        >>> out2 = turbo_fused_attention(q2, pk, kn, pv, vn, dim=128, nr0=2)
    """
    T_q = queries.shape[2]
    if T_q > 2:
        raise ValueError(
            f"turbo_fused_attention supports T_q=1 or T_q=2, got T_q={T_q}. "
            "Use turbo_attention() for prefill (T_q > 2)."
        )
    if mask is not None:
        raise ValueError(
            "turbo_fused_attention does not yet support attention masks. "
            "Use turbo_attention() for masked attention."
        )
    if dim > 256:
        raise ValueError(
            f"turbo_fused_attention supports dim <= 256, got dim={dim}. "
            "The kernel uses a fixed-size register array for V accumulation."
        )

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Auto-select NR0 based on T_q if not explicitly specified
    if nr0 is None:
        nr0 = T_q  # 1 query → NR0=1, 2 queries → NR0=2

    # Validate NR0/T_q compatibility
    if nr0 == 2 and T_q < 2:
        raise ValueError(
            f"NR0=2 requires T_q >= 2, got T_q={T_q}. "
            "Use NR0=1 for single-query decode."
        )

    # NR0=2 dispatch
    if nr0 == 2:
        return _turbo_fused_attention_nr0_2(
            queries, packed_keys, key_norms, packed_values,
            value_norms, dim, bits, seed, scale,
        )

    # NR0=1 (original) dispatch
    if T_q != 1:
        raise ValueError(
            f"NR0=1 requires T_q=1, got T_q={T_q}. "
            "Use NR0=2 for T_q=2 or turbo_attention() for larger T_q."
        )

    kernel = _get_turbo_attn_kernel(bits, nr0=1)
    cb = _get_codebook(bits, dim)

    B, _, T_kv, packed_dim = packed_keys.shape
    n_heads = queries.shape[1]  # Query heads (may differ from KV heads in GQA)

    # --- Step 1: Pre-rotate queries into WHT domain (dual signs) ---
    # Q_rot = signs2 * WHT(signs1 * Q)
    # This transforms the query so that dot products with centroid vectors in
    # the WHT domain give the same result as dot products with decoded K in
    # the original domain. (WHT is orthonormal → preserves inner products.)
    signs1 = _sign_flip_vector(dim, seed)
    signs2 = _sign_flip_vector2(dim, seed)
    q_flipped = queries * signs1                # (B, n_heads, 1, dim)
    q_rot = mx.hadamard_transform(q_flipped) * signs2  # (B, n_heads, 1, dim)
    q_rot = q_rot.astype(mx.float32)

    # --- Step 2: Flatten norms for kernel (remove trailing dim of 1) ---
    k_norms_flat = key_norms.squeeze(-1).astype(mx.float32)    # (B, n_heads, T_kv)
    v_norms_flat = value_norms.squeeze(-1).astype(mx.float32)  # (B, n_heads, T_kv)

    # --- Step 3: Encode scale as uint32 for passing through integer param array ---
    import struct
    scale_as_uint32 = struct.unpack('I', struct.pack('f', scale))[0]
    params = mx.array([dim, T_kv, packed_dim, scale_as_uint32], dtype=mx.uint32)

    # --- Step 4: Launch the Metal kernel ---
    # Grid: one threadgroup per (batch, head) pair
    # Threadgroup size: 64 threads (2 SIMD groups of 32)
    # — enough parallelism for T_kv >> 64, small enough for register pressure
    n_bh = B * n_heads
    tg_size = min(64, max(32, T_kv))  # At least 1 SIMD group, at most 64
    # Round to SIMD group boundary
    tg_size = ((tg_size + 31) // 32) * 32

    # Handle GQA: expand KV heads to match query heads if needed
    n_kv_heads = packed_keys.shape[1]
    if n_kv_heads < n_heads:
        gqa_factor = n_heads // n_kv_heads
        # Repeat each KV head gqa_factor times: [B, nkv, T, D] → [B, nq, T, D]
        packed_keys = mx.repeat(packed_keys, gqa_factor, axis=1)
        key_norms = mx.repeat(key_norms, gqa_factor, axis=1)
        packed_values = mx.repeat(packed_values, gqa_factor, axis=1)
        value_norms = mx.repeat(value_norms, gqa_factor, axis=1)
        k_norms_flat = key_norms.squeeze(-1).astype(mx.float32) if key_norms.ndim > 3 else key_norms.astype(mx.float32)
        v_norms_flat = value_norms.squeeze(-1).astype(mx.float32) if value_norms.ndim > 3 else value_norms.astype(mx.float32)

    # Reshape inputs to (B*n_heads, ...) for the kernel
    q_rot_flat = q_rot.reshape(n_bh, dim)
    pk_flat = packed_keys.reshape(n_bh, T_kv, packed_dim)
    kn_flat = k_norms_flat.reshape(n_bh, T_kv)
    pv_flat = packed_values.reshape(n_bh, T_kv, packed_dim)
    vn_flat = v_norms_flat.reshape(n_bh, T_kv)

    # Max number of simd groups per threadgroup (for scratch buffer)
    n_simd_groups = tg_size // 32

    outputs = kernel(
        inputs=[
            q_rot_flat,            # q_rot
            pk_flat,               # packed_k
            kn_flat,               # k_norms
            pv_flat,               # packed_v
            vn_flat,               # v_norms
            cb.centroids,          # centroids (n_levels,)
            params,                # params
        ],
        output_shapes=[
            (n_bh, dim),           # out_accum
            (n_bh, T_kv),          # scores (threadgroup scratch — will be discarded)
            (n_bh, n_simd_groups), # simd_maxes (scratch)
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(n_bh * tg_size, 1, 1),
        threadgroup=(tg_size, 1, 1),
        init_value=0.0,
        stream=mx.gpu,
    )

    out_rot = outputs[0]  # (n_bh, dim) — in WHT domain

    # --- Step 5: Inverse transform back to original domain (dual signs) ---
    # output = signs1 * WHT(signs2 * out_rot)
    # (WHT is its own inverse for orthonormal normalization)
    out_rot = out_rot.reshape(B, n_heads, 1, dim)
    out_transformed = mx.hadamard_transform(out_rot * signs2)
    output = out_transformed * signs1

    return output.astype(queries.dtype)


def _turbo_fused_attention_nr0_2(
    queries: mx.array,
    packed_keys: mx.array,
    key_norms: mx.array,
    packed_values: mx.array,
    value_norms: mx.array,
    dim: int,
    bits: int,
    seed: int,
    scale: float,
) -> mx.array:
    """NR0=2 multi-row amortization: process 2 queries sharing K/V dequant.

    Internal helper called by turbo_fused_attention when NR0=2 is selected.
    The Metal kernel reads each packed K/V word once and computes dot products
    with both queries simultaneously, halving the memory bandwidth cost.

    Args:
        queries: Shape (B, n_heads, 2, dim) — exactly 2 queries.
        packed_keys: Shape (B, n_heads, T_kv, packed_dim).
        key_norms: Shape (B, n_heads, T_kv, 1).
        packed_values: Shape (B, n_heads, T_kv, packed_dim).
        value_norms: Shape (B, n_heads, T_kv, 1).
        dim: Head dimension.
        bits: Quantization bit-width.
        seed: SRHT seed.
        scale: Attention scale factor.

    Returns:
        Attention output, shape (B, n_heads, 2, dim).
    """
    kernel = _get_turbo_attn_kernel(bits, nr0=2)
    cb = _get_codebook(bits, dim)

    B, _, T_kv, packed_dim = packed_keys.shape
    n_heads = queries.shape[1]  # Query heads (may differ from KV heads in GQA)

    # Pre-rotate both queries into WHT domain (dual signs)
    signs1 = _sign_flip_vector(dim, seed)
    signs2 = _sign_flip_vector2(dim, seed)
    q_flipped = queries * signs1                # (B, n_heads, 2, dim)
    q_rot = mx.hadamard_transform(q_flipped) * signs2  # (B, n_heads, 2, dim)
    q_rot = q_rot.astype(mx.float32)

    # Flatten norms
    k_norms_flat = key_norms.squeeze(-1).astype(mx.float32)
    v_norms_flat = value_norms.squeeze(-1).astype(mx.float32)

    # Encode scale
    import struct
    scale_as_uint32 = struct.unpack('I', struct.pack('f', scale))[0]
    params = mx.array([dim, T_kv, packed_dim, scale_as_uint32], dtype=mx.uint32)

    n_bh = B * n_heads
    tg_size = min(64, max(32, T_kv))
    tg_size = ((tg_size + 31) // 32) * 32

    # Reshape: NR0=2 queries are packed as (n_bh, 2, dim)
    q_rot_flat = q_rot.reshape(n_bh, 2, dim)
    pk_flat = packed_keys.reshape(n_bh, T_kv, packed_dim)
    kn_flat = k_norms_flat.reshape(n_bh, T_kv)
    pv_flat = packed_values.reshape(n_bh, T_kv, packed_dim)
    vn_flat = v_norms_flat.reshape(n_bh, T_kv)

    n_simd_groups = tg_size // 32

    outputs = kernel(
        inputs=[
            q_rot_flat,
            pk_flat,
            kn_flat,
            pv_flat,
            vn_flat,
            cb.centroids,
            params,
        ],
        output_shapes=[
            (n_bh, 2, dim),           # out_accum for both queries
            (n_bh, 2, T_kv),          # scores scratch for both queries
            (n_bh, 2, n_simd_groups), # simd_maxes scratch for both queries
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(n_bh * tg_size, 1, 1),
        threadgroup=(tg_size, 1, 1),
        init_value=0.0,
        stream=mx.gpu,
    )

    out_rot = outputs[0]  # (n_bh, 2, dim)

    # Inverse transform back to original domain (dual signs)
    out_rot = out_rot.reshape(B, n_heads, 2, dim)
    out_transformed = mx.hadamard_transform(out_rot * signs2)
    output = out_transformed * signs1

    return output.astype(queries.dtype)


# ---------------------------------------------------------------------------
# Model-aware config recommendation (moe-v-compression-frontier.md)
# ---------------------------------------------------------------------------


def recommend_config(
    model_type: str,
    head_dim: int,
    num_layers: int,
) -> Dict[str, Union[int, str, bool]]:
    """Suggest optimal TurboQuant config based on model architecture.

    From moe-v-compression-frontier.md: MoE models have a higher fraction of
    attention in their total decode compute (15-30%) because only a few experts
    are active per token, making FFN cheaper. Dense models have attention at <5%
    of decode — KV compression saves memory but barely affects speed.

    This means TurboQuant compression has MORE speed impact on MoE models, and
    more aggressive quantization (symmetric turbo3/turbo4) is worthwhile.

    Args:
        model_type: One of "dense", "moe", or "small" (<3B params).
        head_dim: Head dimension (e.g. 64, 128, 256).
        num_layers: Total number of transformer layers.

    Returns:
        Dict with recommended config keys:
            - bits: V quantization bit-width
            - key_bits: K quantization bit-width (0 = FP16)
            - symmetric: Whether K and V use same quantization
            - rationale: Human-readable explanation
            - speed_benefit: Expected speed improvement category

    Example:
        >>> recommend_config("moe", 128, 32)
        {'bits': 4, 'key_bits': 4, 'symmetric': True, ...}
        >>> recommend_config("dense", 128, 32)
        {'bits': 4, 'key_bits': 0, 'symmetric': False, ...}
    """
    model_type = model_type.lower().strip()

    if model_type == "small":
        # Small models (<3B): overhead of per-layer encode/decode may
        # outweigh the savings. Memory benefit exists but speed may regress.
        return {
            "bits": 4,
            "key_bits": 0,
            "symmetric": False,
            "boundary_layers": 1,
            "rationale": (
                "Small models (<3B): per-layer encode/decode overhead is a "
                "larger fraction of total compute. Use asymmetric (K=FP16, "
                "V=turbo4) for memory savings only. Speed benefit unlikely."
            ),
            "speed_benefit": "minimal",
        }
    elif model_type == "moe":
        # MoE: attention is 15-30% of decode (only a few experts active).
        # Symmetric turbo4 recommended — speed benefit is meaningful.
        return {
            "bits": 4,
            "key_bits": 4,
            "symmetric": True,
            "boundary_layers": 2,
            "rationale": (
                "MoE models: attention is 15-30% of decode compute (FFN is "
                "cheap with sparse expert routing). Symmetric turbo4 gives "
                "both memory AND speed benefits. turbo3 is viable for "
                "aggressive compression. (moe-v-compression-frontier.md)"
            ),
            "speed_benefit": "significant",
        }
    else:
        # Dense: attention is <5% of decode. KV compression helps memory
        # but speed improvement is negligible.
        return {
            "bits": 4,
            "key_bits": 0,
            "symmetric": False,
            "boundary_layers": 2,
            "rationale": (
                "Dense models: attention is <5% of decode — FFN dominates. "
                "KV compression saves memory but speed benefit is minimal. "
                "Asymmetric (K=FP16, V=turbo4) recommended for best quality "
                "per byte. (moe-v-compression-frontier.md)"
            ),
            "speed_benefit": "minimal",
        }


# ---------------------------------------------------------------------------
# TurboQuantKVCache — the main cache module
# ---------------------------------------------------------------------------


class TurboQuantKVCache(Module):
    """TurboQuant-compressed KV cache for transformer inference.

    Stores key and value projections in compressed form using the TurboQuant
    algorithm (SRHT + Lloyd-Max quantization). Supports:

    - Two-phase operation: raw FP during prefill, compressed at first decode
    - Boundary layer protection: first/last N layers stay at full precision
    - Asymmetric K/V: keys can stay FP16 while values are compressed
    - Configurable bit-width: 2, 3, or 4 bits per element

    **Quality guidance** (from empirical testing):

    K precision dominates quality via softmax amplification — small K errors
    become large attention weight errors through the exponential. V errors
    are merely averaged across tokens.

    - Best quality: ``key_bits=0, bits=4`` (K=FP16, V=turbo4 aka "turbo0v4")
    - Good balance: ``key_bits=4, bits=4`` (symmetric turbo4)
    - Aggressive: ``key_bits=0, bits=3`` (K=FP16, V=turbo3)

    Boundary layers (first 2 + last 2 by default) stay at full precision
    as they carry disproportionate signal (confirmed by dhawalc's
    TurboQuantDC independent validation).

    Args:
        bits (int): Quantization bit-width for values. Default: 4.
        key_bits (Optional[int]): Bit-width for keys. None = same as bits.
            Set to 0 or -1 to keep keys at full precision (asymmetric mode).
        seed (int): SRHT random seed. Default: 42.
        boundary_layers (int): Number of layers at start/end to keep at full
            precision. Default: 2. Override with TURBO_BOUNDARY_LAYERS env var.
        layer_idx (Optional[int]): This cache's layer index (for boundary protection).
        num_layers (Optional[int]): Total number of layers (for boundary protection).

    Example:
        >>> cache = TurboQuantKVCache(bits=4, key_bits=0)  # V=4bit, K=FP (recommended)
        >>> # During model forward pass:
        >>> keys, values = cache.update_and_fetch(keys, values)

    Usage pattern in a transformer layer::

        cache = TurboQuantKVCache(bits=4, layer_idx=layer_id, num_layers=32)
        # Prefill phase — stores raw
        k, v = cache.update_and_fetch(k_proj, v_proj)
        # ... after prefill, call cache.compress() to quantize
        # Decode phase — returns decompressed from quantized storage
        k, v = cache.update_and_fetch(new_k, new_v)
    """

    def __init__(
        self,
        bits: int = 4,
        key_bits: Optional[int] = None,
        seed: int = 42,
        boundary_layers: int = 2,
        layer_idx: Optional[int] = None,
        num_layers: Optional[int] = None,
    ):
        super().__init__()

        self.v_bits = bits
        # key_bits <= 0 means keep keys at full precision
        self.k_bits = key_bits if key_bits is not None else bits
        self.seed = seed

        # TURBO_BOUNDARY_LAYERS env var overrides the constructor arg
        env_boundary = os.environ.get("TURBO_BOUNDARY_LAYERS")
        if env_boundary is not None:
            boundary_layers = int(env_boundary)
        self.boundary_layers = boundary_layers
        self.layer_idx = layer_idx
        self.num_layers = num_layers

        # Determine if this layer is a boundary layer (stays at FP)
        self._is_boundary = False
        if layer_idx is not None and num_layers is not None:
            self._is_boundary = (
                layer_idx < boundary_layers
                or layer_idx >= num_layers - boundary_layers
            )

        # State
        self._compressed = False
        self._keys: Optional[mx.array] = None  # Raw or decoded keys
        self._values: Optional[mx.array] = None  # Raw or decoded values

        # Compressed storage
        self._packed_keys: Optional[mx.array] = None
        self._key_norms: Optional[mx.array] = None
        self._packed_values: Optional[mx.array] = None
        self._value_norms: Optional[mx.array] = None

        # Track the head dim for decode
        self._dim: Optional[int] = None

    @property
    def is_boundary_layer(self) -> bool:
        """Whether this layer stays at full precision (boundary protection)."""
        return self._is_boundary

    @property
    def is_compressed(self) -> bool:
        """Whether the cache is currently in compressed form."""
        return self._compressed

    @property
    def seq_len(self) -> int:
        """Current sequence length stored in cache."""
        if self._compressed:
            if self._packed_keys is not None:
                return self._packed_keys.shape[-2]
            if self._packed_values is not None:
                return self._packed_values.shape[-2]
        if self._keys is not None:
            return self._keys.shape[-2]
        return 0

    @property
    def compress_keys(self) -> bool:
        """Whether keys should be compressed (vs kept at FP)."""
        return self.k_bits > 0 and not self._is_boundary

    @property
    def compress_values(self) -> bool:
        """Whether values should be compressed."""
        return self.v_bits > 0 and not self._is_boundary

    def compress(self) -> None:
        """Compress the raw KV cache into TurboQuant format.

        Call this after prefill is complete, before starting decode.
        Boundary layers are left uncompressed.
        """
        if self._compressed or self._keys is None:
            return

        self._dim = self._keys.shape[-1]

        if self.compress_keys:
            self._packed_keys, self._key_norms = turbo_encode(
                self._keys, bits=self.k_bits, seed=self.seed
            )
            self._keys = None  # Free the raw storage
        # else: keys stay as self._keys (FP)

        if self.compress_values:
            self._packed_values, self._value_norms = turbo_encode(
                self._values, bits=self.v_bits, seed=self.seed
            )
            self._values = None  # Free the raw storage
        # else: values stay as self._values (FP)

        self._compressed = True

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update the cache with new KV pairs and return full KV for attention.

        Before compression: simply concatenates new KV to existing cache.
        After compression: encodes new tokens, appends to compressed storage,
        and returns decoded KV for attention computation.

        Args:
            keys: New key projections, shape (batch, heads, new_len, dim).
            values: New value projections, shape (batch, heads, new_len, dim).

        Returns:
            Tuple of (all_keys, all_values) for attention computation.
            Both have shape (batch, heads, total_len, dim).
        """
        if self._dim is None:
            self._dim = keys.shape[-1]

        if not self._compressed:
            # Pre-compression: just accumulate raw tensors
            if self._keys is None:
                self._keys = keys
                self._values = values
            else:
                self._keys = mx.concatenate([self._keys, keys], axis=-2)
                self._values = mx.concatenate([self._values, values], axis=-2)
            return self._keys, self._values

        # Post-compression: encode new tokens and append
        dim = self._dim

        # Handle keys
        if self.compress_keys:
            new_packed_k, new_k_norms = turbo_encode(
                keys, bits=self.k_bits, seed=self.seed
            )
            if self._packed_keys is not None:
                self._packed_keys = mx.concatenate(
                    [self._packed_keys, new_packed_k], axis=-2
                )
                self._key_norms = mx.concatenate(
                    [self._key_norms, new_k_norms], axis=-2
                )
            else:
                self._packed_keys = new_packed_k
                self._key_norms = new_k_norms
            # Decode all keys for attention
            all_keys = turbo_decode(
                self._packed_keys, self._key_norms, dim,
                bits=self.k_bits, seed=self.seed,
            )
        else:
            # Keys at full precision
            if self._keys is None:
                self._keys = keys
            else:
                self._keys = mx.concatenate([self._keys, keys], axis=-2)
            all_keys = self._keys

        # Handle values
        if self.compress_values:
            new_packed_v, new_v_norms = turbo_encode(
                values, bits=self.v_bits, seed=self.seed
            )
            if self._packed_values is not None:
                self._packed_values = mx.concatenate(
                    [self._packed_values, new_packed_v], axis=-2
                )
                self._value_norms = mx.concatenate(
                    [self._value_norms, new_v_norms], axis=-2
                )
            else:
                self._packed_values = new_packed_v
                self._value_norms = new_v_norms
            # Decode all values for attention
            all_values = turbo_decode(
                self._packed_values, self._value_norms, dim,
                bits=self.v_bits, seed=self.seed,
            )
        else:
            # Values at full precision
            if self._values is None:
                self._values = values
            else:
                self._values = mx.concatenate([self._values, values], axis=-2)
            all_values = self._values

        return all_keys, all_values

    def reset(self) -> None:
        """Clear all cached state."""
        self._compressed = False
        self._keys = None
        self._values = None
        self._packed_keys = None
        self._key_norms = None
        self._packed_values = None
        self._value_norms = None
        self._dim = None

    def memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage in bytes (approximate).

        Returns:
            Dict with 'keys_bytes', 'values_bytes', 'total_bytes', and
            'fp_equivalent_bytes' for comparison.
        """
        key_bytes = 0
        val_bytes = 0
        fp_bytes = 0

        seq = self.seq_len
        if seq == 0 or self._dim is None:
            return {
                "keys_bytes": 0, "values_bytes": 0,
                "total_bytes": 0, "fp_equivalent_bytes": 0,
            }

        # Estimate batch*heads from stored shapes
        if self._packed_keys is not None:
            batch_heads = math.prod(self._packed_keys.shape[:-2])
        elif self._keys is not None:
            batch_heads = math.prod(self._keys.shape[:-2])
        else:
            batch_heads = 1

        dim = self._dim
        fp_element_bytes = 2  # float16

        fp_bytes = batch_heads * seq * dim * fp_element_bytes * 2  # K + V

        if self.compress_keys and self._packed_keys is not None:
            packed_dim = self._packed_keys.shape[-1]
            key_bytes = batch_heads * seq * (packed_dim * 4 + 4)  # uint32 + norm
        elif self._keys is not None:
            key_bytes = batch_heads * seq * dim * fp_element_bytes

        if self.compress_values and self._packed_values is not None:
            packed_dim = self._packed_values.shape[-1]
            val_bytes = batch_heads * seq * (packed_dim * 4 + 4)  # uint32 + norm
        elif self._values is not None:
            val_bytes = batch_heads * seq * dim * fp_element_bytes

        return {
            "keys_bytes": key_bytes,
            "values_bytes": val_bytes,
            "total_bytes": key_bytes + val_bytes,
            "fp_equivalent_bytes": fp_bytes,
        }

    def _extra_repr(self):
        parts = [f"v_bits={self.v_bits}"]
        if self.k_bits != self.v_bits:
            parts.append(f"k_bits={self.k_bits}")
        parts.append(f"seed={self.seed}")
        if self._is_boundary:
            parts.append("boundary=True")
        if self._compressed:
            parts.append(f"compressed=True, seq_len={self.seq_len}")
        elif self._keys is not None:
            parts.append(f"raw, seq_len={self.seq_len}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# TurboKVCache — mlx-lm compatible cache for inference
# ---------------------------------------------------------------------------


def _create_causal_mask(N, offset, window_size=None):
    """Create a causal attention mask (local copy to avoid import cycles)."""
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    return mask


def _turbo_create_attention_mask(N, offset, return_array, window_size):
    """Create attention mask matching mlx-lm's cache.create_attention_mask."""
    if window_size is not None:
        return _create_causal_mask(N, offset, window_size=window_size)
    elif N == 1:
        return None
    elif return_array:
        return _create_causal_mask(N, offset, window_size=window_size)
    else:
        return "causal"


class TurboKVCache:
    """TurboQuant KV cache compatible with mlx-lm's inference loop.

    Drop-in replacement for mlx-lm's ``KVCache`` that compresses the KV cache
    using TurboQuant (SRHT + Lloyd-Max quantization). Integrates with
    ``generate_step`` / ``stream_generate`` / ``generate`` via the
    ``prompt_cache`` parameter.

    **Two-phase design:**

    1. **Prefill** (num_steps > 1): stores raw FP16 keys/values, matching our
       llama.cpp prefill fix. No quantization overhead during prompt processing.
    2. **Decode** (num_steps == 1): on the first single-token step, compresses
       the raw cache. Subsequent tokens are encoded and appended to the packed
       storage. ``update_and_fetch`` always returns full FP16 K/V so standard
       SDPA works — compression is internal only.

    No ``bits`` attribute is exposed, so ``scaled_dot_product_attention`` in
    mlx-lm's ``base.py`` takes the standard (non-quantized) path.

    Asymmetric K/V is supported: set ``key_bits=0`` to keep keys at FP16
    while values are turbo-compressed. This is the recommended config since
    K errors get amplified through softmax exponentials.

    Args:
        bits (int): Quantization bit-width for values. Default: 4.
        key_bits (Optional[int]): Bit-width for keys. ``None`` = same as
            ``bits``. Set to 0 to keep keys at full precision. Default: None.
        seed (int): SRHT random seed. Default: 42.
        min_compress_tokens (int): Minimum number of cached tokens before
            compression kicks in. Below this threshold, KV stays in raw FP16
            — the memory savings are <2MB but the encode/decode overhead
            costs ~30% decode speed. Default: 256.
        compact_threshold (int): When offset exceeds this, drop the decoded
            FP16 caches and re-decode from packed storage each step. Trades
            O(n) decode cost per step (small vs O(n²) SDPA) for ~50% less
            KV memory at long context. 0 = never compact. Default: 8192.

    Example:
        >>> import mlx_lm
        >>> from mlx.nn.layers.turbo_kv_cache import TurboKVCache
        >>> model, tokenizer = mlx_lm.load('mlx-community/Qwen3.5-2B-8bit')
        >>> n_layers = len(model.model.layers)
        >>> cache = [TurboKVCache(bits=4) for _ in range(n_layers)]
        >>> text = mlx_lm.generate(
        ...     model, tokenizer, prompt='Hello',
        ...     max_tokens=20, prompt_cache=cache, verbose=True,
        ... )
    """

    def __init__(
        self,
        bits: int = 4,
        key_bits: Optional[int] = None,
        seed: int = 42,
        min_compress_tokens: int = 256,
        fused_attention: bool = False,
        block_size: int = 0,
        compact_threshold: int = 8192,
        encode_batch_size: int = 8,
        k_compress_threshold: int = 0,
    ):
        self.v_bits = bits
        self.k_bits = key_bits if key_bits is not None else bits
        self.seed = seed
        # Adaptive K compression: when > 0 and k_bits <= 0 (asymmetric mode),
        # K stays FP16 until seq_len crosses this threshold, then all existing
        # FP16 K are batch-compressed to turbo4 and new K tokens are compressed
        # on arrival. This gives asymmetric quality at short context (where
        # precision matters) and symmetric bandwidth at long context (where
        # reading less data matters). Set via TURBO_K_COMPRESS_THRESHOLD env var.
        env_kct = os.environ.get("TURBO_K_COMPRESS_THRESHOLD")
        if env_kct is not None:
            k_compress_threshold = int(env_kct)
        self.k_compress_threshold = k_compress_threshold
        # The target K bits after adaptive compression kicks in (default: match V bits)
        self._adaptive_k_bits = bits  # turbo4 when V is turbo4
        # WHT block size: 0 = full head_dim, 32 = blocked WHT (4 blocks of 32
        # for dim=128). block_size=32 gives -0.02 PPL AND 21% faster encode.
        self.block_size = block_size
        # Deferred compression: below this threshold, keep KV in raw FP16.
        # The memory savings at short context are <2MB but the speed cost of
        # encode/decode is ~30%. Only compress when the cache exceeds this size.
        self.min_compress_tokens = min_compress_tokens
        # Compact mode: when offset exceeds this threshold, drop the decoded
        # FP16 caches (_decoded_keys/_decoded_values) and re-decode the full
        # packed cache each step. This trades O(n) decode per step for ~50%
        # less KV memory. At long context the O(n²) SDPA dominates anyway.
        # Set to 0 to disable (always keep decoded FP16 cache).
        self.compact_threshold = compact_threshold
        self._compact_mode = False  # Flipped once when threshold crossed

        # Lazy batch encode: accumulate N raw tokens before encoding them as a
        # batch. Batch encode is ~10x cheaper per token (0.021ms vs 0.219ms)
        # due to amortized WHT/quantize overhead. Set to 1 to disable (encode
        # every token immediately, original behavior). Default 8.
        self.encode_batch_size = max(1, encode_batch_size)

        # When True, skip creating decoded FP16 buffers during compression.
        # Use cache.attention() instead of update_and_fetch + SDPA to avoid
        # the double-storage problem. Requires symmetric 4-bit, Metal GPU.
        self._fused_attention = fused_attention

        # Set by patch_mlx_lm() — when True, update_and_fetch skips decode
        # and returns dummy K/V. The patched SDPA intercepts and calls
        # turbo_fused_attention on packed data directly. This eliminates the
        # double-storage problem without requiring model code changes.
        self._patched = False

        # Raw (uncompressed) storage — used during prefill
        self._raw_keys: Optional[mx.array] = None
        self._raw_values: Optional[mx.array] = None

        # Compressed storage — used during decode
        self._packed_keys: Optional[mx.array] = None
        self._key_norms: Optional[mx.array] = None
        self._packed_values: Optional[mx.array] = None
        self._value_norms: Optional[mx.array] = None

        # FP keys when key_bits <= 0 (asymmetric mode)
        self._fp_keys: Optional[mx.array] = None

        # FP values when v_bits <= 0 (no compression)
        self._fp_values: Optional[mx.array] = None

        # Cached decoded FP16 arrays — avoids re-decoding entire packed storage
        # every step. Only the newly added token(s) get decoded and concatenated.
        # This matches the llama.cpp approach: compressed storage is source of
        # truth for memory savings, decoded FP16 window is for fast attention.
        # Skipped when fused_attention=True (fused kernel reads packed directly).
        self._decoded_keys: Optional[mx.array] = None
        self._decoded_values: Optional[mx.array] = None

        # Pending raw tokens for lazy batch encode — raw FP16 arrays waiting
        # to be encoded. These are already included in _decoded_keys/_decoded_values
        # (concatenated immediately for SDPA correctness) but NOT yet in
        # _packed_keys/_packed_values. Flushed when len >= encode_batch_size.
        self._pending_raw_keys: List[mx.array] = []
        self._pending_raw_values: List[mx.array] = []

        self._is_compressed = False
        self._is_turbo_kv = True  # Flag for mlx-lm SDPA detection
        self._dim: Optional[int] = None
        self.offset = 0

        # Pre-allocated buffer offsets (for asymmetric patched fast path)
        self._fp_k_offset = 0
        self._packed_v_offset = 0

    @property
    def compress_keys(self) -> bool:
        """Whether keys should be turbo-compressed (vs kept at FP)."""
        return self.k_bits > 0

    def _maybe_compress_keys_adaptive(self) -> None:
        """Adaptively compress K from FP16 to turbo4 when context crosses threshold.

        Called during decode when k_compress_threshold > 0 and K is currently FP16.
        Batch-compresses all existing FP16 K, flips k_bits to enable compressed K
        going forward, and switches the SDPA routing from asymmetric to symmetric.

        TODO: Write a two-pass symmetric kernel so the post-switch path is as fast
        as the two-pass asymmetric kernel. Currently falls back to the single-pass
        turbo_fused_attention which is slower.
        """
        if (
            self.k_compress_threshold <= 0
            or self.k_bits > 0  # Already compressing K
            or self._fp_keys is None
            or self.offset < self.k_compress_threshold
        ):
            return

        # Flush any pending V tokens before we switch modes
        self._flush_pending()

        # Batch-compress all existing FP16 K to turbo4
        self._packed_keys, self._key_norms = turbo_encode(
            self._fp_keys, bits=self._adaptive_k_bits, seed=self.seed,
            block_size=self.block_size,
        )

        # Free FP16 K storage — this is where the memory savings come from
        self._fp_keys = None
        self._decoded_keys = None

        # Flip the mode: from this point on, new K tokens get compressed
        self.k_bits = self._adaptive_k_bits

    @property
    def compress_values(self) -> bool:
        """Whether values should be turbo-compressed."""
        return self.v_bits > 0

    def _compress_raw_cache(self) -> None:
        """Compress accumulated raw prefill cache into TurboQuant format.

        Called once on the first decode step. After this, the raw buffers are
        freed and all new tokens go through encode→pack.
        """
        if self._is_compressed or self._raw_keys is None:
            return

        self._dim = self._raw_keys.shape[-1]

        # Check if we should go straight into compact mode (e.g., prefill
        # was longer than compact_threshold — no point seeding decoded FP16
        # just to immediately drop it).
        # Also skip decoded FP16 when patched — the monkey-patched SDPA will
        # call turbo_fused_attention directly on packed data, so decoded FP16
        # would just waste memory (the whole point of patching).
        skip_decoded = (
            (self.compact_threshold > 0 and self.offset > self.compact_threshold)
            or (self._patched and self.compress_keys and self.compress_values and self.k_bits == self.v_bits == 4)
            # Asymmetric patched: disabled — always seed decoded V for SDPA fallback safety
        )
        if skip_decoded and not self._patched:
            self._compact_mode = True

        if self.compress_keys:
            self._packed_keys, self._key_norms = turbo_encode(
                self._raw_keys, bits=self.k_bits, seed=self.seed,
                block_size=self.block_size,
            )
            if not skip_decoded:
                # Decode once to seed the FP16 cache — subsequent steps only
                # decode the new token and concatenate (O(1) not O(n)).
                # Skipped in compact mode: no decoded cache maintained.
                self._decoded_keys = turbo_decode(
                    self._packed_keys, self._key_norms, self._dim,
                    bits=self.k_bits, seed=self.seed,
                    block_size=self.block_size,
                )
        else:
            self._fp_keys = self._raw_keys
            self._fp_k_offset = self._raw_keys.shape[2]

        if self.compress_values:
            self._packed_values, self._value_norms = turbo_encode(
                self._raw_values, bits=self.v_bits, seed=self.seed,
                block_size=self.block_size,
            )
            self._packed_v_offset = self._packed_values.shape[2]
            if not skip_decoded:
                # Same: decode once, then incremental.
                # Skipped in compact mode: no decoded cache maintained.
                self._decoded_values = turbo_decode(
                    self._packed_values, self._value_norms, self._dim,
                    bits=self.v_bits, seed=self.seed,
                    block_size=self.block_size,
                )
        else:
            self._fp_values = self._raw_values

        # Free raw buffers
        self._raw_keys = None
        self._raw_values = None
        self._is_compressed = True

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache with new K/V and return full (decoded) K/V for SDPA.

        During prefill (num_steps > 1), stores raw FP16 — no quantization.
        On the first decode step (num_steps == 1), compresses the raw cache.
        Subsequent decode steps encode the new token and append to packed
        storage.

        Always returns plain ``mx.array`` keys and values (not quantized
        tuples) so standard SDPA works. The compression is internal.

        Args:
            keys: Shape ``(B, n_kv_heads, num_steps, head_dim)``.
            values: Shape ``(B, n_kv_heads, num_steps, head_dim)``.

        Returns:
            ``(all_keys, all_values)`` both as ``mx.array`` with shape
            ``(B, n_kv_heads, total_seq_len, head_dim)``.
        """
        num_steps = keys.shape[2]

        if self._dim is None:
            self._dim = keys.shape[-1]

        # --- Prefill phase: accumulate raw ---
        if not self._is_compressed and num_steps > 1:
            if self._raw_keys is None:
                self._raw_keys = keys
                self._raw_values = values
            else:
                self._raw_keys = mx.concatenate(
                    [self._raw_keys, keys], axis=2,
                )
                self._raw_values = mx.concatenate(
                    [self._raw_values, values], axis=2,
                )
            self.offset = self._raw_keys.shape[2]
            return self._raw_keys, self._raw_values

        # --- Deferred compression: stay raw until we hit the token threshold ---
        # Below min_compress_tokens, the memory savings are negligible (<2MB)
        # but the encode/decode overhead costs ~30% decode speed. Keep raw FP16
        # until the cache is big enough to justify compression.
        if not self._is_compressed and self.offset < self.min_compress_tokens:
            if self._raw_keys is None:
                self._raw_keys = keys
                self._raw_values = values
            else:
                self._raw_keys = mx.concatenate(
                    [self._raw_keys, keys], axis=2,
                )
                self._raw_values = mx.concatenate(
                    [self._raw_values, values], axis=2,
                )
            self.offset = self._raw_keys.shape[2]
            return self._raw_keys, self._raw_values

        # --- Transition: first decode step triggers compression ---
        if not self._is_compressed:
            self._compress_raw_cache()

        # --- Decode phase: encode new token(s), append, return decoded ---
        self.offset += num_steps
        dim = self._dim

        # --- Adaptive K compression: switch K from FP16 to turbo4 at threshold ---
        # This gives asymmetric quality at short context and symmetric bandwidth
        # at long context. The SDPA routing auto-detects the switch via compress_keys.
        self._maybe_compress_keys_adaptive()

        # --- Patched fast path: encode-only, no decode ---
        # When patch_mlx_lm() is active, the SDPA function is monkey-patched
        # to call turbo_fused_attention directly on packed data. We still need
        # to encode new tokens into packed storage, but we skip ALL decode
        # work (no FP16 buffers created). Return the raw new keys/values as
        # dummy sentinels — the patched SDPA ignores them.
        _can_fuse_patched = (
            self._patched
            and num_steps == 1
            and self.compress_keys
            and self.compress_values
            and self.k_bits == self.v_bits == 4
            and dim is not None
            and dim <= 256
            and mx.metal.is_available()
        )
        if _can_fuse_patched:
            # Flush any pending batch-encode tokens before the fused path
            # (fused kernel reads packed storage directly, must be complete)
            self._flush_pending()

            # Encode and append keys
            new_pk, new_kn = turbo_encode(
                keys, bits=self.k_bits, seed=self.seed,
                block_size=self.block_size,
            )
            if self._packed_keys is not None:
                self._packed_keys = mx.concatenate(
                    [self._packed_keys, new_pk], axis=2,
                )
                self._key_norms = mx.concatenate(
                    [self._key_norms, new_kn], axis=2,
                )
            else:
                self._packed_keys = new_pk
                self._key_norms = new_kn

            # Encode and append values
            new_pv, new_vn = turbo_encode(
                values, bits=self.v_bits, seed=self.seed,
                block_size=self.block_size,
            )
            if self._packed_values is not None:
                self._packed_values = mx.concatenate(
                    [self._packed_values, new_pv], axis=2,
                )
                self._value_norms = mx.concatenate(
                    [self._value_norms, new_vn], axis=2,
                )
            else:
                self._packed_values = new_pv
                self._value_norms = new_vn

            # Return raw keys/values as dummy — patched SDPA will ignore these
            # and call turbo_fused_attention on packed data instead.
            # No FP16 decode buffers allocated. This is the whole point.
            return keys, values

        # --- Patched asymmetric fast path: K=FP16 (no encode), V=turbo4 (encode-only) ---
        # When asymmetric (compress_keys=False, compress_values=True) and patched,
        # the SDPA uses turbo_asymmetric_attention which scores with raw FP16 K
        # and does weighted sum on packed V. We store raw K and encode+append V.
        # No V decode buffers allocated — eliminates the #1 gap.
        _can_fuse_asymmetric = not self.compress_keys and self.compress_values
        if _can_fuse_asymmetric:
            # --- Pre-allocated FP16 decode path ---
            # Like mlx-lm's KVCache: pre-allocate in chunks, slice-assign.
            # No concat per step. No encode/decode per step.
            # Memory savings from prefill compression (99%+ of tokens).
            _STEP = 256
            n_new = keys.shape[2]
            prev = self._fp_k_offset

            # Grow K buffer if needed
            if self._fp_keys is None or (prev + n_new) > self._fp_keys.shape[2]:
                B_k, nh_k, _, d_k = keys.shape
                n_alloc = ((_STEP + n_new - 1) // _STEP) * _STEP
                new_k = mx.zeros((B_k, nh_k, n_alloc, d_k), keys.dtype)
                if self._fp_keys is not None:
                    if prev % _STEP != 0:
                        self._fp_keys = self._fp_keys[..., :prev, :]
                    self._fp_keys = mx.concatenate([self._fp_keys, new_k], axis=2)
                else:
                    self._fp_keys = new_k

            # Grow V buffer if needed
            if self._decoded_values is None or (prev + n_new) > self._decoded_values.shape[2]:
                B_v, nh_v, _, d_v = values.shape
                n_alloc = ((_STEP + n_new - 1) // _STEP) * _STEP
                new_v = mx.zeros((B_v, nh_v, n_alloc, d_v), values.dtype)
                if self._decoded_values is not None:
                    if prev % _STEP != 0:
                        self._decoded_values = self._decoded_values[..., :prev, :]
                    self._decoded_values = mx.concatenate([self._decoded_values, new_v], axis=2)
                else:
                    self._decoded_values = new_v

            # Slice-assign — no alloc, no copy
            self._fp_keys[..., prev:prev + n_new, :] = keys
            self._decoded_values[..., prev:prev + n_new, :] = values
            self._fp_k_offset = prev + n_new

            # Return sliced views
            return self._fp_keys[..., :self._fp_k_offset, :], self._decoded_values[..., :self._fp_k_offset, :]

        # --- Compact mode transition ---
        # Once we exceed compact_threshold, drop the decoded FP16 caches to
        # cut KV memory ~50%. From here on, we re-decode the full packed
        # cache each step (O(n) per step, but O(n²) SDPA dominates at long
        # context so the overhead is negligible).
        if (
            not self._compact_mode
            and self.compact_threshold > 0
            and self.offset > self.compact_threshold
        ):
            # Flush any pending raw tokens before switching to compact mode,
            # since compact mode re-decodes from packed storage each step.
            self._flush_pending()
            self._compact_mode = True
            self._decoded_keys = None
            self._decoded_values = None

        # Handle keys
        if self.compress_keys:
            if self.encode_batch_size > 1 and not self._compact_mode:
                # --- Lazy batch encode: accumulate raw keys, encode in batches ---
                # Append raw key to pending buffer (O(1) — just stores reference).
                # The decoded FP16 cache gets the raw token immediately for SDPA.
                self._pending_raw_keys.append(keys)

                # Add raw key to decoded cache for SDPA correctness
                if self._decoded_keys is not None:
                    self._decoded_keys = mx.concatenate(
                        [self._decoded_keys, keys], axis=2,
                    )
                else:
                    self._decoded_keys = keys

                # Flush pending batch when we hit encode_batch_size
                if len(self._pending_raw_keys) >= self.encode_batch_size:
                    batch_k = mx.concatenate(self._pending_raw_keys, axis=2)
                    batch_pk, batch_kn = turbo_encode(
                        batch_k, bits=self.k_bits, seed=self.seed,
                        block_size=self.block_size,
                    )
                    if self._packed_keys is not None:
                        self._packed_keys = mx.concatenate(
                            [self._packed_keys, batch_pk], axis=2,
                        )
                        self._key_norms = mx.concatenate(
                            [self._key_norms, batch_kn], axis=2,
                        )
                    else:
                        self._packed_keys = batch_pk
                        self._key_norms = batch_kn
                    self._pending_raw_keys = []

                all_keys = self._decoded_keys
            else:
                # Original per-token encode path (batch_size=1 or compact mode)
                new_pk, new_kn = turbo_encode(
                    keys, bits=self.k_bits, seed=self.seed,
                    block_size=self.block_size,
                )
                if self._packed_keys is not None:
                    self._packed_keys = mx.concatenate(
                        [self._packed_keys, new_pk], axis=2,
                    )
                    self._key_norms = mx.concatenate(
                        [self._key_norms, new_kn], axis=2,
                    )
                else:
                    self._packed_keys = new_pk
                    self._key_norms = new_kn

                if self._compact_mode:
                    # Compact mode: full decode from packed each step.
                    # No FP16 cache maintained — saves ~50% KV memory.
                    # Flush any pending raw keys first
                    if self._pending_raw_keys:
                        batch_k = mx.concatenate(self._pending_raw_keys, axis=2)
                        batch_pk, batch_kn = turbo_encode(
                            batch_k, bits=self.k_bits, seed=self.seed,
                            block_size=self.block_size,
                        )
                        self._packed_keys = mx.concatenate(
                            [self._packed_keys, batch_pk], axis=2,
                        )
                        self._key_norms = mx.concatenate(
                            [self._key_norms, batch_kn], axis=2,
                        )
                        self._pending_raw_keys = []
                    all_keys = turbo_decode(
                        self._packed_keys, self._key_norms, dim,
                        bits=self.k_bits, seed=self.seed,
                        block_size=self.block_size,
                    )
                else:
                    # Incremental decode: only decode the new token(s), concat with
                    # cached FP16. Avoids O(n) full-cache decode every step.
                    new_decoded_k = turbo_decode(
                        new_pk, new_kn, dim, bits=self.k_bits, seed=self.seed,
                        block_size=self.block_size,
                    )
                    if self._decoded_keys is not None:
                        self._decoded_keys = mx.concatenate(
                            [self._decoded_keys, new_decoded_k], axis=2,
                        )
                    else:
                        self._decoded_keys = new_decoded_k
                    all_keys = self._decoded_keys
        else:
            if self._fp_keys is not None:
                self._fp_keys = mx.concatenate([self._fp_keys, keys], axis=2)
            else:
                self._fp_keys = keys
            all_keys = self._fp_keys

        # Handle values
        if self.compress_values:
            if self.encode_batch_size > 1 and not self._compact_mode:
                # --- Lazy batch encode: accumulate raw values, encode in batches ---
                # Append raw value to pending buffer (O(1) — just stores reference).
                # The decoded FP16 cache gets the raw token immediately for SDPA.
                self._pending_raw_values.append(values)

                # Add raw value to decoded cache for SDPA correctness
                if self._decoded_values is not None:
                    self._decoded_values = mx.concatenate(
                        [self._decoded_values, values], axis=2,
                    )
                else:
                    self._decoded_values = values

                # Flush pending batch when we hit encode_batch_size
                if len(self._pending_raw_values) >= self.encode_batch_size:
                    batch_v = mx.concatenate(self._pending_raw_values, axis=2)
                    batch_pv, batch_vn = turbo_encode(
                        batch_v, bits=self.v_bits, seed=self.seed,
                        block_size=self.block_size,
                    )
                    if self._packed_values is not None:
                        self._packed_values = mx.concatenate(
                            [self._packed_values, batch_pv], axis=2,
                        )
                        self._value_norms = mx.concatenate(
                            [self._value_norms, batch_vn], axis=2,
                        )
                    else:
                        self._packed_values = batch_pv
                        self._value_norms = batch_vn
                    self._pending_raw_values = []

                all_values = self._decoded_values
            else:
                # Original per-token encode path (batch_size=1 or compact mode)
                new_pv, new_vn = turbo_encode(
                    values, bits=self.v_bits, seed=self.seed,
                    block_size=self.block_size,
                )
                if self._packed_values is not None:
                    self._packed_values = mx.concatenate(
                        [self._packed_values, new_pv], axis=2,
                    )
                    self._value_norms = mx.concatenate(
                        [self._value_norms, new_vn], axis=2,
                    )
                else:
                    self._packed_values = new_pv
                    self._value_norms = new_vn

                if self._compact_mode:
                    # Compact mode: full decode from packed each step.
                    # Flush any pending raw values first
                    if self._pending_raw_values:
                        batch_v = mx.concatenate(self._pending_raw_values, axis=2)
                        batch_pv, batch_vn = turbo_encode(
                            batch_v, bits=self.v_bits, seed=self.seed,
                            block_size=self.block_size,
                        )
                        self._packed_values = mx.concatenate(
                            [self._packed_values, batch_pv], axis=2,
                        )
                        self._value_norms = mx.concatenate(
                            [self._value_norms, batch_vn], axis=2,
                        )
                        self._pending_raw_values = []
                    all_values = turbo_decode(
                        self._packed_values, self._value_norms, dim,
                        bits=self.v_bits, seed=self.seed,
                        block_size=self.block_size,
                    )
                else:
                    # Incremental decode: only decode the new token(s)
                    new_decoded_v = turbo_decode(
                        new_pv, new_vn, dim, bits=self.v_bits, seed=self.seed,
                        block_size=self.block_size,
                    )
                    if self._decoded_values is not None:
                        self._decoded_values = mx.concatenate(
                            [self._decoded_values, new_decoded_v], axis=2,
                        )
                    else:
                        self._decoded_values = new_decoded_v
                    all_values = self._decoded_values
        else:
            if self._fp_values is not None:
                self._fp_values = mx.concatenate([self._fp_values, values], axis=2)
            else:
                self._fp_values = values
            all_values = self._fp_values

        return all_keys, all_values

    def attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        scale: Optional[float] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Update cache and compute attention in one step, using fused kernel
        when possible to avoid materializing FP16 K/V.

        This is the preferred attention path for TurboKVCache. During decode
        (T_q=1) with both K and V compressed at 4-bit, it uses the fused
        Metal kernel that operates directly on packed data. Otherwise, it
        falls back to update_and_fetch + standard SDPA.

        When the fused path is used:
        - No FP16 K/V buffer is ever allocated (solves the double-storage problem)
        - The decoded FP16 caches (_decoded_keys, _decoded_values) are NOT needed
        - Memory usage drops to purely packed storage + norms

        Args:
            queries: Query projections, shape (B, n_q_heads, T_q, dim).
            keys: New key projections, shape (B, n_kv_heads, T_q, dim).
            values: New value projections, shape (B, n_kv_heads, T_q, dim).
            scale: Attention scale. Default: 1/sqrt(dim).
            mask: Attention mask. Only used in fallback path.

        Returns:
            Attention output, shape (B, n_q_heads, T_q, dim).

        Example:
            >>> cache = TurboKVCache(bits=4, key_bits=4)
            >>> # In the model's attention layer:
            >>> output = cache.attention(q, k_proj, v_proj, scale=scale)
        """
        num_steps = keys.shape[2]
        dim = keys.shape[-1]

        if self._dim is None:
            self._dim = dim

        # Check if we should trigger compression for the fused path.
        # This handles the transition from prefill → decode when
        # fused_attention=True, compressing without creating decoded FP16.
        _wants_fuse = (
            num_steps == 1
            and self.compress_keys
            and self.compress_values
            and self.k_bits == self.v_bits == 4
            and mask is None
            and dim <= 256
            and mx.metal.is_available()
        )
        if _wants_fuse and not self._is_compressed and self.offset >= self.min_compress_tokens:
            # Trigger compression (skips decoded FP16 if fused_attention=True)
            self._compress_raw_cache()

        can_fuse = _wants_fuse and self._is_compressed

        if can_fuse:
            # Flush any pending batch-encode tokens before fused attention
            # (fused kernel reads packed storage directly, must be complete)
            self._flush_pending()

            # Encode the new token and append to packed storage
            self.offset += num_steps

            new_pk, new_kn = turbo_encode(
                keys, bits=self.k_bits, seed=self.seed,
                block_size=self.block_size,
            )
            if self._packed_keys is not None:
                self._packed_keys = mx.concatenate(
                    [self._packed_keys, new_pk], axis=2,
                )
                self._key_norms = mx.concatenate(
                    [self._key_norms, new_kn], axis=2,
                )
            else:
                self._packed_keys = new_pk
                self._key_norms = new_kn

            new_pv, new_vn = turbo_encode(
                values, bits=self.v_bits, seed=self.seed,
                block_size=self.block_size,
            )
            if self._packed_values is not None:
                self._packed_values = mx.concatenate(
                    [self._packed_values, new_pv], axis=2,
                )
                self._value_norms = mx.concatenate(
                    [self._value_norms, new_vn], axis=2,
                )
            else:
                self._packed_values = new_pv
                self._value_norms = new_vn

            # Fused attention on packed data — no FP16 materialization
            # NOTE: we do NOT update _decoded_keys/_decoded_values here.
            # The fused path doesn't need them. If the user later calls
            # update_and_fetch (fallback path), the decoded caches will be
            # stale — but that's OK because update_and_fetch rebuilds them.
            output = turbo_fused_attention(
                queries,
                self._packed_keys,
                self._key_norms,
                self._packed_values,
                self._value_norms,
                dim=dim,
                bits=self.v_bits,
                seed=self.seed,
                scale=scale,
            )

            # Handle GQA: if n_q_heads > n_kv_heads, the fused kernel already
            # handles this because it broadcasts across heads. But actually,
            # turbo_fused_attention expects Q and K to have the same n_heads.
            # GQA support will need the kernel to take a heads_ratio param.
            # TODO: Add GQA support to turbo_fused_attention
            return output

        else:
            # Fallback: update_and_fetch + standard SDPA
            all_keys, all_values = self.update_and_fetch(keys, values)

            if scale is None:
                scale = 1.0 / math.sqrt(dim)

            # Use mx.fast.scaled_dot_product_attention for the fallback
            if mask is None and num_steps == 1:
                # Decode: no mask needed
                return mx.fast.scaled_dot_product_attention(
                    queries, all_keys, all_values, scale=scale,
                )
            else:
                # Prefill or masked: use the mask
                if mask is None:
                    mask = "causal"
                return mx.fast.scaled_dot_product_attention(
                    queries, all_keys, all_values, scale=scale, mask=mask,
                )

    # --- mlx-lm _BaseCache interface ---

    @property
    def state(self):
        """Return cache tensors for mx.eval() materialization."""
        # During prefill, return raw buffers
        if not self._is_compressed:
            if self._raw_keys is not None:
                return self._raw_keys, self._raw_values
            return []

        # After compression, return all stored tensors (including decoded FP16 cache
        # and any pending raw tokens awaiting batch encode)
        parts = []
        if self._packed_keys is not None:
            parts.extend([self._packed_keys, self._key_norms])
        if self._decoded_keys is not None:
            parts.append(self._decoded_keys)
        if self._fp_keys is not None:
            parts.append(self._fp_keys)
        if self._packed_values is not None:
            parts.extend([self._packed_values, self._value_norms])
        if self._decoded_values is not None:
            parts.append(self._decoded_values)
        if self._fp_values is not None:
            parts.append(self._fp_values)
        # Include pending raw arrays so mx.eval() materializes them
        parts.extend(self._pending_raw_keys)
        parts.extend(self._pending_raw_values)
        return parts if parts else []

    @state.setter
    def state(self, v):
        if v is not None and v:
            # TODO: Implement state restore for save/load prompt cache
            pass

    @property
    def meta_state(self):
        return str(self.offset)

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            self.offset = int(v)

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        """Trim n tokens from the cache. Returns actual tokens trimmed."""
        n = min(self.offset, n)
        self.offset -= n
        # TODO: Actually trim the packed/raw buffers for correctness
        # For now this handles the common case where trim is called but
        # the cache is about to be rebuilt anyway
        if not self._is_compressed and self._raw_keys is not None:
            if n > 0:
                self._raw_keys = self._raw_keys[..., :-n, :]
                self._raw_values = self._raw_values[..., :-n, :]
        if n > 0:
            # Trim decoded FP16 caches to stay in sync
            if self._decoded_keys is not None:
                self._decoded_keys = self._decoded_keys[..., :-n, :]
            if self._decoded_values is not None:
                self._decoded_values = self._decoded_values[..., :-n, :]
            # Drop pending raw buffers on trim — they're out of sync now.
            # The decoded FP16 cache (trimmed above) is the source of truth.
            # Pending tokens will be re-accumulated from scratch.
            self._pending_raw_keys = []
            self._pending_raw_values = []
        return n

    def make_mask(self, N, return_array=False, window_size=None):
        """Create attention mask (called by mlx-lm's create_attention_mask)."""
        return _turbo_create_attention_mask(
            N, offset=self.offset, return_array=return_array,
            window_size=window_size,
        )

    def empty(self) -> bool:
        """Return True if the cache has no stored data."""
        return (
            self._raw_keys is None
            and self._packed_keys is None
            and self._fp_keys is None
        )

    @property
    def nbytes(self) -> int:
        """Approximate memory usage in bytes."""
        total = 0
        for arr in [
            self._raw_keys, self._raw_values,
            self._packed_keys, self._key_norms,
            self._packed_values, self._value_norms,
            self._decoded_keys, self._decoded_values,
            self._fp_keys, self._fp_values,
        ]:
            if arr is not None:
                total += arr.nbytes
        # Include pending raw tokens awaiting batch encode
        for arr in self._pending_raw_keys:
            total += arr.nbytes
        for arr in self._pending_raw_values:
            total += arr.nbytes
        return total

    def _flush_pending(self) -> None:
        """Force-flush any pending raw tokens into packed storage.

        Call this before operations that need packed storage to be fully
        up-to-date (e.g., serialization, switching to compact mode).
        The decoded FP16 cache already includes these tokens — this just
        ensures the packed representation is in sync.
        """
        dim = self._dim
        if self._pending_raw_keys and self.compress_keys:
            batch_k = mx.concatenate(self._pending_raw_keys, axis=2)
            batch_pk, batch_kn = turbo_encode(
                batch_k, bits=self.k_bits, seed=self.seed,
                block_size=self.block_size,
            )
            if self._packed_keys is not None:
                self._packed_keys = mx.concatenate(
                    [self._packed_keys, batch_pk], axis=2,
                )
                self._key_norms = mx.concatenate(
                    [self._key_norms, batch_kn], axis=2,
                )
            else:
                self._packed_keys = batch_pk
                self._key_norms = batch_kn
            self._pending_raw_keys = []

        if self._pending_raw_values and self.compress_values:
            batch_v = mx.concatenate(self._pending_raw_values, axis=2)
            batch_pv, batch_vn = turbo_encode(
                batch_v, bits=self.v_bits, seed=self.seed,
                block_size=self.block_size,
            )
            if self._packed_values is not None:
                self._packed_values = mx.concatenate(
                    [self._packed_values, batch_pv], axis=2,
                )
                self._value_norms = mx.concatenate(
                    [self._value_norms, batch_vn], axis=2,
                )
            else:
                self._packed_values = batch_pv
                self._value_norms = batch_vn
            self._pending_raw_values = []

    def __repr__(self):
        mode = "compact" if self._compact_mode else ("compressed" if self._is_compressed else "raw")
        k_desc = f"k={self.k_bits}bit" if self.compress_keys else "k=fp"
        v_desc = f"v={self.v_bits}bit" if self.compress_values else "v=fp"
        bs_desc = f"block={self.block_size}" if self.block_size > 0 else "block=full"
        compact_desc = f"compact@{self.compact_threshold}" if self.compact_threshold > 0 else "no-compact"
        patched = ", patched" if self._patched else ""
        pending = len(self._pending_raw_keys) + len(self._pending_raw_values)
        pending_desc = f", pending={pending}" if pending > 0 else ""
        batch_desc = f", batch_encode={self.encode_batch_size}" if self.encode_batch_size > 1 else ""
        return (
            f"TurboKVCache({k_desc}, {v_desc}, {bs_desc}, {mode}, "
            f"offset={self.offset}, dim={self._dim}, "
            f"min_compress={self.min_compress_tokens}, {compact_desc}"
            f"{batch_desc}{pending_desc}{patched})"
        )


# ---------------------------------------------------------------------------
# Monkey-patch mechanism for mlx-lm integration
# ---------------------------------------------------------------------------

# Stash for the original SDPA function, so unpatch can restore it.
_original_sdpa = None


def patch_mlx_lm(cache_list: Optional[list] = None) -> None:
    """Monkey-patch mlx-lm to use fused turbo attention when TurboKVCache is active.

    Call once before model inference. Replaces the ``scaled_dot_product_attention``
    function in ``mlx_lm.models.base`` with a turbo-aware version that detects
    ``TurboKVCache`` and routes to the fused Metal kernel.

    This eliminates the double-storage problem: ``update_and_fetch`` no longer
    materializes decoded FP16 buffers when the patched SDPA will call
    ``turbo_fused_attention`` directly on packed data.

    The patch is transparent to non-turbo caches — standard ``KVCache``,
    ``RotatingKVCache``, and quantized caches all take the original code path.

    Args:
        cache_list: Optional list of ``TurboKVCache`` instances to mark as
            patched. If ``None``, the patch still installs but individual caches
            must have ``_patched = True`` set manually. When provided, sets
            ``_patched = True`` on all ``TurboKVCache`` instances in the list.

    Example:
        >>> import mlx_lm
        >>> from mlx.nn.layers.turbo_kv_cache import TurboKVCache, patch_mlx_lm
        >>> model, tokenizer = mlx_lm.load('mlx-community/Qwen3.5-2B-8bit')
        >>> n_layers = len(model.model.layers)
        >>> cache = [TurboKVCache(bits=4, key_bits=4) for _ in range(n_layers)]
        >>> patch_mlx_lm(cache)  # Install once before generation
        >>> text = mlx_lm.generate(model, tokenizer, prompt='Hello',
        ...                        max_tokens=100, prompt_cache=cache)
    """
    global _original_sdpa

    try:
        import mlx_lm.models.base as base
    except ImportError:
        raise ImportError(
            "mlx-lm is not installed. Install with: pip install mlx-lm"
        )

    # Don't double-patch
    if _original_sdpa is not None:
        # Already patched — just mark any new caches
        if cache_list is not None:
            for c in cache_list:
                if isinstance(c, TurboKVCache):
                    c._patched = True
        return

    _original_sdpa = base.scaled_dot_product_attention

    def turbo_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        """Turbo-aware SDPA: routes to fused attention for TurboKVCache.

        When the cache is a patched TurboKVCache with compressed 4-bit K/V,
        the keys/values args (from update_and_fetch) are dummy sentinels.
        We ignore them and call turbo_fused_attention on the packed data
        stored in the cache object directly.

        For asymmetric config (K=FP16, V=turbo4), uses the asymmetric path:
        standard matmul on raw FP16 K for scoring, Metal kernel on packed V
        for the weighted sum. No V decode overhead.

        For all other cache types, delegates to the original SDPA unchanged.
        """
        # Check if this is a patched TurboKVCache that can use the fused path
        if (
            isinstance(cache, TurboKVCache)
            and cache._patched
            and cache._is_compressed
            and cache.compress_keys
            and cache.compress_values
            and cache.k_bits == cache.v_bits == 4
            and cache._packed_keys is not None
            and cache._packed_values is not None
            and queries.shape[2] <= 2  # T_q=1 or 2 (decode only)
            and mask is None  # Fused kernel doesn't support masks yet
            and cache._dim is not None
            and cache._dim <= 256
            and mx.metal.is_available()
        ):
            # Fused attention directly on packed data — no FP16 materialized
            return turbo_fused_attention(
                queries,
                cache._packed_keys,
                cache._key_norms,
                cache._packed_values,
                cache._value_norms,
                dim=cache._dim,
                bits=cache.v_bits,
                seed=cache.seed,
                scale=scale,
            )

        # Asymmetric path: when encode-decode fast path is active,
        # update_and_fetch already returned real FP16 K/V. Skip turbo dispatch
        # and let native SDPA handle it at full speed.
        if (
            isinstance(cache, TurboKVCache)
            and cache._patched
            and cache._fp_k_offset > 0  # encode-decode fast path active
        ):
            # Fall through to native SDPA with the real K/V from update_and_fetch
            return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)

        # Asymmetric fused path (legacy): custom turbo attention kernels
        # Only used when encode-decode fast path is NOT active
        if (
            isinstance(cache, TurboKVCache)
            and cache._patched
            and cache._is_compressed
            and not cache.compress_keys
            and cache.compress_values
            and cache.v_bits == 4
            and cache._fp_keys is not None
            and cache._packed_values is not None
            and queries.shape[2] == 1
            and mask is None
            and cache._dim is not None
            and cache._dim <= 256
            and mx.metal.is_available()
        ):
            # Flush any pending batch tokens so packed_values is complete
            if hasattr(cache, '_flush_pending'):
                cache._flush_pending()

            # Slice to actual offset if using pre-allocated buffers
            fp_keys = cache._fp_keys
            packed_values = cache._packed_values
            value_norms = cache._value_norms
            if hasattr(cache, '_fp_k_offset'):
                fp_keys = fp_keys[..., :cache._fp_k_offset, :]
            if hasattr(cache, '_packed_v_offset'):
                packed_values = packed_values[..., :cache._packed_v_offset, :]
                value_norms = value_norms[..., :cache._packed_v_offset, :]

            # Dispatch hierarchy: two-pass (opt-in) > single-pass > default 4-dispatch
            if os.environ.get("TURBO_ASYMMETRIC_LEGACY", "0") == "1":
                return turbo_asymmetric_attention(
                    queries,
                    fp_keys,
                    packed_values,
                    value_norms,
                    dim=cache._dim,
                    bits=cache.v_bits,
                    seed=cache.seed,
                    scale=scale,
                )
            if os.environ.get("TURBO_ASYMMETRIC_SINGLE", "0") == "1":
                return turbo_fused_asymmetric_attention_single_dispatch(
                    queries,
                    fp_keys,
                    packed_values,
                    value_norms,
                    dim=cache._dim,
                    bits=cache.v_bits,
                    seed=cache.seed,
                    scale=scale,
                )
            # Default: two-pass TurboFlash (fastest at all context lengths)
            # Set TURBO_USE_4DISPATCH=1 to fall back to original 4-dispatch
            if os.environ.get("TURBO_USE_4DISPATCH", "0") == "1":
                return turbo_asymmetric_attention(
                    queries,
                    fp_keys,
                    packed_values,
                    value_norms,
                    dim=cache._dim,
                    bits=cache.v_bits,
                    seed=cache.seed,
                    scale=scale,
                )
            _tp_bs = int(os.environ.get("TURBO_TWO_PASS_BLOCK_SIZE", "64"))
            return turbo_two_pass_asymmetric_attention(
                queries,
                fp_keys,
                packed_values,
                value_norms,
                dim=cache._dim,
                bits=cache.v_bits,
                seed=cache.seed,
                scale=scale,
            )

        # Fallback: original SDPA for non-turbo caches, prefill, masked, etc.
        return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    base.scaled_dot_product_attention = turbo_sdpa

    # Also patch all ALREADY-LOADED model modules that imported SDPA by name.
    # Python 'from X import Y' copies the reference — patching X.Y alone
    # doesn't update Y in the importing module.
    import sys as _sys
    for _name, _mod in _sys.modules.items():
        if _name.startswith('mlx_lm.models.') and hasattr(_mod, 'scaled_dot_product_attention'):
            if _mod.scaled_dot_product_attention is _original_sdpa:
                _mod.scaled_dot_product_attention = turbo_sdpa

    # Mark caches as patched so update_and_fetch skips decode
    if cache_list is not None:
        for c in cache_list:
            if isinstance(c, TurboKVCache):
                c._patched = True


def unpatch_mlx_lm(cache_list: Optional[list] = None) -> None:
    """Restore the original mlx-lm SDPA function.

    Reverses the effect of ``patch_mlx_lm()``. Also clears the ``_patched``
    flag on any caches in the provided list so ``update_and_fetch`` resumes
    normal decode behavior.

    Args:
        cache_list: Optional list of caches to un-mark. If ``None``, only
            the global SDPA is restored.
    """
    global _original_sdpa

    if _original_sdpa is None:
        return  # Nothing to unpatch

    try:
        import mlx_lm.models.base as base
        base.scaled_dot_product_attention = _original_sdpa
    except ImportError:
        pass

    _original_sdpa = None

    if cache_list is not None:
        for c in cache_list:
            if isinstance(c, TurboKVCache):
                c._patched = False


class TurboKVCacheLite:
    """KVCache wrapper that compresses V after prefill for memory savings.

    Behaves exactly like mlx-lm's KVCache during attention (zero overhead).
    On the first single-token decode step, compresses the prefill V cache
    using TurboQuant (SRHT + Lloyd-Max). The FP16 V stays for attention;
    the compressed copy is stored for memory recovery at long context.

    This is NOT a custom cache class — it delegates entirely to KVCache.
    The compression is a side-effect that doesn't affect attention.
    """

    def __init__(self, kv_cache, bits: int = 4, key_bits: int = 0,
                 seed: int = 42, encode_batch: int = 32):
        self._kv = kv_cache
        self._bits = bits
        self._key_bits = key_bits  # 0 = don't compress K, >0 = compress K too
        self._seed = seed
        self._compressed = False
        self._packed_keys: Optional[mx.array] = None
        self._key_norms: Optional[mx.array] = None
        self._packed_values: Optional[mx.array] = None
        self._value_norms: Optional[mx.array] = None
        self._encode_batch = encode_batch
        self._n_pending = 0
        self._compacted = False
        self._qv_data: Optional[mx.array] = None
        self._qv_scales: Optional[mx.array] = None
        self._qv_biases: Optional[mx.array] = None
        self._compact_pending_v: List[mx.array] = []  # Raw V tokens awaiting batch quantize
        self._compact_batch_size = 32  # Batch-quantize every N tokens

    def update_and_fetch(self, keys, values):
        if not self._compacted:
            # Pure KVCache — no compression during decode. Zero overhead.
            return self._kv.update_and_fetch(keys, values)

        # Compacted mode: dequant prefill V into KVCache buffer once,
        # then pure KVCache behavior. No tuples, no monkey-patch, no per-step
        # quantize. Native SDPA at full speed. Memory savings from packed V
        # stored in _qv_data (not used for attention — just storage).

        if not getattr(self, '_compact_v_seeded', False):
            # One-time: dequant packed V into KVCache values buffer
            _cb = getattr(self, '_compact_bits', 8)
            _cg = getattr(self, '_compact_group_size', 64)
            deq = mx.dequantize(
                self._qv_data, self._qv_scales, self._qv_biases,
                group_size=_cg, bits=_cb,
            ).astype(keys.dtype)
            n_prefill = deq.shape[2]
            _STEP = self._kv.step
            n_alloc = ((n_prefill + _STEP - 1) // _STEP) * _STEP
            self._kv.values = mx.zeros(
                (deq.shape[0], deq.shape[1], n_alloc, deq.shape[3]),
                dtype=deq.dtype,
            )
            self._kv.values[..., :n_prefill, :] = deq
            mx.eval(self._kv.values)
            self._compact_v_seeded = True

        # Pure KVCache from here — full speed
        return self._kv.update_and_fetch(keys, values)

    def compress(self) -> None:
        """Compress the current KV cache using TurboQuant.

        Call this explicitly when you want to create compressed storage
        (e.g., before a long generation, or when memory pressure is high).
        Does NOT affect the FP16 cache — attention continues at full speed.
        """
        if self._compressed or self._kv.keys is None:
            return

        offset = self._kv.offset
        if offset == 0:
            return

        # Compress V
        v = self._kv.values[..., :offset, :]
        self._packed_values, self._value_norms = turbo_encode(
            v, bits=self._bits, seed=self._seed
        )

        # Compress K if requested
        if self._key_bits > 0:
            k = self._kv.keys[..., :offset, :]
            self._packed_keys, self._key_norms = turbo_encode(
                k, bits=self._key_bits, seed=self._seed
            )

        self._compressed = True

    # Delegate everything else to the wrapped KVCache
    @property
    def offset(self):
        return self._kv.offset

    @offset.setter
    def offset(self, v):
        self._kv.offset = v

    @property
    def keys(self):
        return self._kv.keys

    @property
    def values(self):
        return self._kv.values

    @property
    def state(self):
        if self._compacted:
            # Return K + quantized V tuple
            k = self._kv.keys[..., :self._kv.offset, :] if self._kv.keys is not None else None
            return k, (self._qv_data, self._qv_scales, self._qv_biases)
        return self._kv.state

    @state.setter
    def state(self, v):
        self._kv.state = v

    def size(self):
        return self._kv.size()

    def is_trimmable(self):
        return self._kv.is_trimmable()

    def trim(self, n):
        return self._kv.trim(n)

    @property
    def nbytes(self):
        total = self._kv.keys.nbytes if self._kv.keys is not None else 0
        if self._compacted and self._qv_data is not None:
            total += self._qv_data.nbytes + self._qv_scales.nbytes + self._qv_biases.nbytes
        elif self._kv.values is not None:
            total += self._kv.values.nbytes
        return total

    def compact(self, bits: int = 8, group_size: int = 64) -> int:
        """Quantize V using mx.quantize and drop FP16 V buffer.

        K stays FP16 (critical for quality). V is quantized to the specified
        bit-width using MLX's built-in quantization.

        Args:
            bits: Quantization bit-width (4 or 8). Default 8 (safe for dense
                models with 24+ layers). 4-bit causes quality collapse on
                dense models due to error compounding across layers.
            group_size: Quantization group size. Default 64.

        Returns:
            Approximate bytes freed.
        """
        if self._kv.keys is None or self._compacted:
            return 0

        offset = self._kv.offset
        if offset == 0:
            return 0

        old_bytes = self._kv.values.nbytes
        self._compact_bits = bits
        self._compact_group_size = group_size

        # Quantize V (cast to float32 so scales/biases are float32 —
        # required by the C++ sdpa_vector_qv kernel which reads float*)
        v = self._kv.values[..., :offset, :].astype(mx.float32)
        self._qv_data, self._qv_scales, self._qv_biases = mx.quantize(
            v, group_size=group_size, bits=bits
        )
        mx.eval(self._qv_data, self._qv_scales, self._qv_biases)

        # Drop FP16 V buffer
        self._kv.values = None
        self._compacted = True

        new_bytes = self._qv_data.nbytes + self._qv_scales.nbytes + self._qv_biases.nbytes
        return old_bytes - new_bytes

    @property
    def _is_compacted(self):
        return getattr(self, '_compacted', False)

    def recover_memory(self) -> int:
        """Replace FP16 KV with lossy re-decoded version and drop compressed storage.

        Compresses K/V if not already compressed, decodes back to FP16 (lossy),
        replaces the KV buffers, and drops the compressed copies. The FP16
        buffers shrink by dropping pre-allocation padding.

        Note: this does NOT achieve 74% memory savings — it replaces FP16 with
        lossy FP16 of the same size. The savings come only from dropping the
        pre-allocation padding (~10-20%). For true compressed-only storage,
        a fused attention kernel that operates on packed data is needed.

        Attention continues on FP16 (re-decoded) so native SDPA still works.
        Not thread-safe — call only when no concurrent decode is running.

        Returns:
            Approximate bytes freed (from dropping pre-allocation padding).
        """
        if self._kv.keys is None:
            return 0

        # Compress first if not already done
        if not self._compressed:
            self.compress()

        offset = self._kv.offset
        dim = self._kv.values.shape[-1]
        old_v_bytes = self._kv.values.nbytes
        old_k_bytes = self._kv.keys.nbytes

        # Replace FP16 V with decoded-from-compressed (lossy but 74% smaller source)
        n_compressed_v = self._packed_values.shape[2]
        decoded_v = turbo_decode(
            self._packed_values, self._value_norms, dim,
            bits=self._bits, seed=self._seed,
        )
        # Rebuild: decoded prefill + any raw decode tokens after compression
        if offset > n_compressed_v:
            raw_tail = self._kv.values[..., n_compressed_v:offset, :]
            rebuilt_v = mx.concatenate([decoded_v, raw_tail], axis=2)
        else:
            rebuilt_v = decoded_v

        # Shrink buffer to exact size (no pre-allocation padding)
        self._kv.values = rebuilt_v[..., :offset, :]
        # Fix KVCache offset tracking — next update_and_fetch will re-grow
        mx.eval(self._kv.values)

        # Replace FP16 K with decoded-from-compressed if K was compressed
        if self._packed_keys is not None:
            n_compressed_k = self._packed_keys.shape[2]
            k_dim = self._kv.keys.shape[-1]  # K dim may differ from V dim
            decoded_k = turbo_decode(
                self._packed_keys, self._key_norms, k_dim,
                bits=self._key_bits, seed=self._seed,
            )
            if offset > n_compressed_k:
                raw_k_tail = self._kv.keys[..., n_compressed_k:offset, :]
                rebuilt_k = mx.concatenate([decoded_k, raw_k_tail], axis=2)
            else:
                rebuilt_k = decoded_k
            self._kv.keys = rebuilt_k[..., :offset, :]
            mx.eval(self._kv.keys)

        # Re-establish pre-allocation padding so KVCache doesn't re-grow every step
        _STEP = self._kv.step  # Usually 256
        n_alloc = ((offset + _STEP - 1) // _STEP) * _STEP
        if self._kv.values.shape[2] < n_alloc:
            B, nh, _, dv = self._kv.values.shape
            padded_v = mx.zeros((B, nh, n_alloc, dv), self._kv.values.dtype)
            padded_v[..., :offset, :] = self._kv.values
            self._kv.values = padded_v
            mx.eval(self._kv.values)
        if self._kv.keys.shape[2] < n_alloc:
            B, nh, _, dk = self._kv.keys.shape
            padded_k = mx.zeros((B, nh, n_alloc, dk), self._kv.keys.dtype)
            padded_k[..., :offset, :] = self._kv.keys
            self._kv.keys = padded_k
            mx.eval(self._kv.keys)

        # Drop compressed storage — FP16 cache now holds the decoded data
        self._packed_values = None
        self._value_norms = None
        self._packed_keys = None
        self._key_norms = None
        self._compressed = False

        new_v_bytes = self._kv.values.nbytes
        new_k_bytes = self._kv.keys.nbytes
        return (old_v_bytes + old_k_bytes) - (new_v_bytes + new_k_bytes)

    # Memory stats
    @property
    def memory_savings(self) -> float:
        """Fraction of KV memory saved by compression (0.0 to 1.0)."""
        if not self._compressed:
            return 0.0
        fp_bytes = self.fp16_size_bytes
        packed_bytes = self.compressed_size_bytes
        return 1.0 - packed_bytes / fp_bytes if fp_bytes > 0 else 0.0

    @property
    def compressed_size_bytes(self) -> int:
        """Size of compressed K+V storage in bytes."""
        total = 0
        if self._packed_values is not None:
            total += self._packed_values.nbytes + self._value_norms.nbytes
        if self._packed_keys is not None:
            total += self._packed_keys.nbytes + self._key_norms.nbytes
        return total

    @property
    def fp16_size_bytes(self) -> int:
        """Size of FP16 K+V in bytes."""
        total = 0
        if self._kv.values is not None:
            total += self._kv.values[..., : self._kv.offset, :].nbytes
        if self._kv.keys is not None:
            total += self._kv.keys[..., : self._kv.offset, :].nbytes
        return total


def make_turbo_cache(
    model,
    bits: int = 4,
    key_bits: int = 4,
    boundary: int = 2,
    seed: int = 42,
) -> list:
    """One-line TurboQuant KV cache setup for mlx-lm models.

    Uses standard KVCache for full-speed attention (zero decode overhead).
    Compresses K/V after prefill for memory savings. At long context, the
    compressed data can be used to recover memory by dropping FP16.

    Works with stock mlx-lm — no fork needed. Only requires TheTom/mlx.

    Args:
        model: The mlx-lm model (e.g., from ``mlx_lm.load()``).
        bits (int): V quantization bit-width (2, 3, or 4). Default: 4.
        key_bits (int): K quantization bit-width. 0 = don't compress K. Default: 4.
        boundary (int): Number of first/last attention layers to keep at FP16
            (no compression). Default: 2.
        seed (int): SRHT random seed. Default: 42.

    Returns:
        List of cache objects to pass as ``prompt_cache``.

    Example::

        from mlx.nn.layers.turbo_kv_cache import make_turbo_cache
        model, tokenizer = mlx_lm.load('mlx-community/Qwen2.5-7B-Instruct-8bit')
        cache = make_turbo_cache(model, bits=4)
        text = mlx_lm.generate(model, tokenizer, prompt='Hello',
                               max_tokens=100, prompt_cache=cache, verbose=True)
    """
    try:
        from mlx_lm.models.cache import make_prompt_cache, KVCache
    except ImportError:
        raise ImportError(
            "mlx-lm is required. Install with: pip install mlx-lm"
        )

    base_cache = make_prompt_cache(model)

    # Wrap turbo layers with TurboKVCacheLite (boundary layers stay as-is)
    kv_indices = [i for i, c in enumerate(base_cache) if isinstance(c, KVCache)]
    n_kv = len(kv_indices)

    for rank, idx in enumerate(kv_indices):
        if rank < boundary or rank >= n_kv - boundary:
            continue
        base_cache[idx] = TurboKVCacheLite(
            base_cache[idx], bits=bits, key_bits=key_bits, seed=seed
        )

    return base_cache


_compact_original_sdpa = None


def compact_turbo_cache(cache: list) -> int:
    """Compact all TurboKVCacheLite layers: quantize V to 4-bit, drop FP16 V.

    Installs a monkey-patched SDPA that handles FP16 K + quantized V
    using mx.quantized_matmul (C++ kernel, ~1.1-1.4x native SDPA).

    Call this when memory pressure is high and you want real KV savings.
    K stays FP16 for quality, V is quantized to 4-bit (~62% total KV savings).

    Args:
        cache: Cache list from make_turbo_cache.

    Returns:
        Approximate bytes freed.
    """
    global _compact_original_sdpa

    total_freed = 0
    for c in cache:
        if isinstance(c, TurboKVCacheLite):
            total_freed += c.compact()

    # No SDPA monkey-patch needed — compact mode dequants V on first decode
    # step and runs pure KVCache + native SDPA from there. Full speed.
    if False and _compact_original_sdpa is None:
        try:
            # Try mlx_lm first, fall back to mlx_vlm
            try:
                import mlx_lm.models.base as base
            except ImportError:
                import mlx_vlm.models.base as base
            _compact_original_sdpa = base.scaled_dot_product_attention

            def compact_sdpa(queries, keys, values, cache=None, scale=None, mask=None, sinks=None):
                # Detect compacted TurboKVCacheLite — values is a quantized tuple
                if isinstance(values, tuple) and len(values) == 3:
                    B, n_q_heads, L, D = queries.shape
                    n_kv_heads = keys.shape[1]
                    if scale is None:
                        scale = D ** -0.5

                    # C++ sdpa_vector_qv kernel — single dispatch.
                    # Bug fix: scales/biases must be float32 (kernel reads float*).
                    _cg = 64
                    if isinstance(cache, TurboKVCacheLite):
                        _cg = getattr(cache, '_compact_group_size', 64)
                    if L == 1 and D in (64, 96, 128, 256) and mx.metal.is_available():
                        try:
                            qv_data, qv_scales, qv_biases = values
                            return mx.fast.scaled_dot_product_attention_qv(
                                queries, keys,
                                qv_data,
                                qv_scales.astype(mx.float32),
                                qv_biases.astype(mx.float32),
                                scale=scale, group_size=_cg,
                            )
                        except Exception:
                            pass

                    # Fallback: manual Q×K + mx.quantized_matmul
                    n_repeats = n_q_heads // n_kv_heads
                    q_scaled = queries * scale
                    if n_repeats > 1:
                        q_scaled = q_scaled.reshape(B, n_kv_heads, n_repeats, L, D)
                        k_exp = mx.expand_dims(keys, axis=2)
                        qv = tuple(mx.expand_dims(x, axis=2) for x in values)
                    else:
                        k_exp = keys
                        qv = values
                    scores = q_scaled @ k_exp.transpose(0, 1, 2, -1, -2) if n_repeats > 1 else (
                        q_scaled @ keys.transpose(0, 1, 3, 2)
                    )
                    if mask is not None:
                        if isinstance(mask, str):
                            qL, kL = scores.shape[-2:]
                            q_idx = mx.arange(kL - qL, kL)
                            k_idx = mx.arange(kL)
                            mask = q_idx[:, None] >= k_idx[None]
                        if mask.dtype == mx.bool_:
                            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
                        else:
                            scores += mask
                    weights = mx.softmax(scores, axis=-1, precise=True)
                    # Detect bits/group_size from cache if available
                    _cb = 8; _cg = 64
                    if isinstance(cache, TurboKVCacheLite):
                        _cb = getattr(cache, '_compact_bits', 8)
                        _cg = getattr(cache, '_compact_group_size', 64)
                    out = mx.quantized_matmul(weights, *qv, transpose=False, group_size=_cg, bits=_cb)
                    if n_repeats > 1:
                        out = out.reshape(B, n_q_heads, L, D)
                    return out

                # Fallback to original SDPA
                return _compact_original_sdpa(queries, keys, values, cache, scale, mask, sinks)

            base.scaled_dot_product_attention = compact_sdpa

            # Also patch all model modules that imported the function
            # Covers both mlx_lm and mlx_vlm model modules
            import sys as _sys
            for name, mod in _sys.modules.items():
                if (name.startswith("mlx_lm.models.") or name.startswith("mlx_vlm.models.")) and hasattr(mod, "scaled_dot_product_attention"):
                    cur = getattr(mod, "scaled_dot_product_attention")
                    if cur is not compact_sdpa:  # Don't re-patch
                        setattr(mod, "scaled_dot_product_attention", compact_sdpa)
        except ImportError:
            pass

    return total_freed
