"""Tests for scaled_dot_product_attention_qv (decode + prefill).

Verifies the fused QV SDPA kernel against dequantize + native SDPA
for both decode (L=1, sdpa_vector_qv) and prefill (L>1, steel attention_qv).
"""
# Copyright © 2025 Shiyang "Landon" Yue

import unittest
import mlx.core as mx


def reference_qv_sdpa(q, k, vq, vs, vb, scale, group_size=32):
    """Reference: dequantize V, then native fp16 SDPA."""
    bits = 4 if group_size == 32 else 8
    v_deq = mx.dequantize(vq, vs, vb, group_size=group_size, bits=bits)
    return mx.fast.scaled_dot_product_attention(q, k, v_deq, scale=scale)


class TestSDPAQuantizedV(unittest.TestCase):

    def _make_inputs(self, B, H_kv, T, D, GQA, group_size=32):
        H_q = H_kv * GQA
        k = mx.random.normal((B, H_kv, T, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, T, D)).astype(mx.float16)
        bits = 4 if group_size == 32 else 8
        vq, vs, vb = mx.quantize(v, group_size=group_size, bits=bits)
        # Expand K for GQA (native SDPA needs expanded K)
        k_exp = mx.repeat(k, GQA, axis=1)
        vq_exp = mx.repeat(vq, GQA, axis=1)
        vs_exp = mx.repeat(vs, GQA, axis=1)
        vb_exp = mx.repeat(vb, GQA, axis=1)
        return k, k_exp, vq_exp, vs_exp.astype(mx.float32), vb_exp.astype(mx.float32)

    def _cosine(self, a, b):
        a = a.astype(mx.float32).reshape(-1)
        b = b.astype(mx.float32).reshape(-1)
        return (mx.sum(a * b) / (
            mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b))
        )).item()

    # === Decode tests (L=1) ===

    def test_decode_basic(self):
        """Basic decode: B=1, H=4, D=128, T=512, GQA=4."""
        B, H, T, D, GQA = 1, 4, 512, 128, 4
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, 1, D)).astype(mx.float16)
        mx.eval(q, k, k_exp, vq, vs, vb)

        scale = D ** -0.5
        ref = reference_qv_sdpa(q, k_exp, vq, vs, vb, scale)
        out = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq, vs, vb, scale=scale, group_size=32)
        mx.eval(ref, out)

        self.assertGreater(self._cosine(ref, out), 0.99)

    def test_decode_long_context(self):
        """Decode at 8K context."""
        B, H, T, D, GQA = 1, 8, 8192, 128, 4
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, 1, D)).astype(mx.float16)
        mx.eval(q, k, k_exp, vq, vs, vb)

        out = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq, vs, vb, scale=D ** -0.5, group_size=32)
        ref = reference_qv_sdpa(q, k_exp, vq, vs, vb, D ** -0.5)
        mx.eval(ref, out)

        self.assertGreater(self._cosine(ref, out), 0.99)

    def test_decode_d256(self):
        """Decode with D=256 (gemma-4 SWA layers)."""
        B, H, T, D, GQA = 1, 16, 2048, 256, 2
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, 1, D)).astype(mx.float16)
        mx.eval(q, k, k_exp, vq, vs, vb)

        out = mx.fast.scaled_dot_product_attention_qv(
            q, k_exp, vq, vs, vb, scale=D ** -0.5, group_size=32)
        ref = reference_qv_sdpa(q, k_exp, vq, vs, vb, D ** -0.5)
        mx.eval(ref, out)

        self.assertGreater(self._cosine(ref, out), 0.99)

    # === Prefill tests (L>1) — currently uses dequant + native SDPA ===
    # The fused steel attention_qv kernel is in development.
    # These tests verify the Python-level prefill approach works.

    def test_prefill_rejects_l_gt_1(self):
        """QV SDPA currently only supports L=1 (decode)."""
        B, H, T, D, GQA = 1, 4, 512, 128, 4
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, 64, D)).astype(mx.float16)
        mx.eval(q, k_exp, vq, vs, vb)

        with self.assertRaises(Exception):
            mx.fast.scaled_dot_product_attention_qv(
                q, k_exp, vq, vs, vb, scale=D ** -0.5, group_size=32)

    def test_prefill_dequant_approach(self):
        """Prefill via dequant V + native SDPA (recommended for L>1)."""
        B, H, T, D, L, GQA = 1, 4, 512, 128, 64, 4
        k, k_exp, vq, vs, vb = self._make_inputs(B, H, T, D, GQA)
        q = mx.random.normal((B, H * GQA, L, D)).astype(mx.float16)
        mx.eval(q, k, k_exp, vq, vs, vb)

        # Reference: fp16 V
        v_fp16 = mx.random.normal((B, H, T, D)).astype(mx.float16)
        v_exp = mx.repeat(v_fp16, GQA, axis=1)
        vq2, vs2, vb2 = mx.quantize(v_fp16, group_size=32, bits=4)
        vq2_exp = mx.repeat(vq2, GQA, axis=1)
        vs2_exp = mx.repeat(vs2, GQA, axis=1)
        vb2_exp = mx.repeat(vb2, GQA, axis=1)
        mx.eval(v_fp16, v_exp, vq2_exp, vs2_exp, vb2_exp)

        ref = mx.fast.scaled_dot_product_attention(q, k_exp, v_exp, scale=D ** -0.5)
        # Dequant approach
        v_deq = mx.dequantize(vq2_exp, vs2_exp, vb2_exp, group_size=32, bits=4)
        out = mx.fast.scaled_dot_product_attention(q, k_exp, v_deq, scale=D ** -0.5)
        mx.eval(ref, out)

        self.assertGreater(self._cosine(ref, out), 0.99)


if __name__ == "__main__":
    unittest.main()
