# SPDX-License-Identifier: Apache-2.0
"""Perf benchmark for materialized-MLA prefill. GLM-4.7-Flash dims, on a varlen
batch of prefill requests:

  * absorbed     : the current per-request MLX SDPA loop in kv_lora space
                   (512-wide MQA + PE-as-mask) — ``_apply_absorbed_mla_attention``.
  * materialized : materialize full per-head K/V from embed_q/unembed_out, then
                   per-request standard MHA via MLX SDPA (head_dim qk/v).

Both use only public MLX ops (no custom kernel); both match (the absorption
identity). Reports the materialized speedup for dense (fp16) and 4bit-quantized
embed_q/unembed_out — GLM-4.7-Flash ships the latter.

Run:
    .venv-vllm-metal/bin/python tools/benchmark/mla_materialized_prefill_benchmark.py
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import numpy as np

try:
    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.mla import MultiLinear
except ImportError:  # pragma: no cover
    print("mlx_lm unavailable; cannot run.")
    sys.exit(0)

H, NOPE, ROPE, KVL, VD = 32, 128, 64, 512, 128
QK = NOPE + ROPE
SCALE = QK**-0.5


def _p50(fn, w=5, it=30):
    for _ in range(w):
        mx.eval(fn())
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def _build(quantize: bool):
    mx.random.seed(0)
    eq = MultiLinear(NOPE, KVL, H)
    uo = MultiLinear(KVL, VD, H)
    eq.weight = eq.weight.astype(mx.float16)
    uo.weight = uo.weight.astype(mx.float16)
    if quantize:
        eq = eq.to_quantized(64, 4)
        uo = uo.to_quantized(64, 4)
    return eq, uo


def bench(eq, uo, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"{'nseq':>4} {'total':>6} {'abs ms':>8} {'mat ms':>8} {'speedup':>8}")
    print("-" * 40)
    for lens in [
        [768],
        [768, 256],
        [768, 256, 512, 1024],
        [2048, 1024, 512, 1536, 256, 768, 1024, 512],
    ]:
        mx.random.seed(1)
        total = sum(lens)
        qn = mx.random.normal((total, H, NOPE)).astype(mx.float16)
        qp = mx.random.normal((total, H, ROPE)).astype(mx.float16)
        kvn = mx.random.normal((total, KVL)).astype(mx.float16)
        kpe = mx.random.normal((total, ROPE)).astype(mx.float16)
        cu = [0] + [int(c) for c in np.cumsum(lens)]
        mx.eval(qn, qp, kvn, kpe)

        def absorbed(lens=lens, cu=cu, qn=qn, qp=qp, kvn=kvn, kpe=kpe, eq=eq, uo=uo):
            outs = []
            for i, length in enumerate(lens):
                s = cu[i]
                rq = eq(qn[s : s + length].transpose(1, 0, 2)[None])
                pe = (qp[s : s + length].transpose(1, 0, 2)[None] * SCALE) @ kpe[
                    s : s + length
                ][None, None].swapaxes(-1, -2)
                c = (mx.arange(length)[None] <= mx.arange(length)[:, None]).reshape(
                    1, 1, length, length
                )
                pe = mx.where(c, pe, mx.array(mx.finfo(pe.dtype).min, pe.dtype))
                kv = kvn[s : s + length][None, None]
                o = scaled_dot_product_attention(
                    rq, kv, kv, cache=None, scale=SCALE, mask=pe
                )
                outs.append(uo(o)[0].transpose(1, 0, 2))
            return mx.concatenate(outs, axis=0)

        def materialized(
            lens=lens, cu=cu, qn=qn, qp=qp, kvn=kvn, kpe=kpe, total=total, eq=eq, uo=uo
        ):
            # [1, ...] leading axis broadcasts across heads for both dense and
            # quantized embed_q/unembed_out (matches the wrapper materialization).
            kv3 = kvn.reshape(1, total, KVL)
            k_nope = eq(kv3, transpose=False)  # [H, total, qk_nope]
            values = uo(kv3)  # [H, total, v]
            kp = mx.broadcast_to(kpe[None], (H, total, ROPE))
            keys = mx.concatenate([k_nope, kp], axis=-1)  # [H, total, qk]
            queries = mx.concatenate([qn, qp], axis=-1).transpose(1, 0, 2)  # [H,t,qk]
            outs = []
            for i in range(len(lens)):
                s, e = cu[i], cu[i + 1]
                o = scaled_dot_product_attention(
                    queries[:, s:e][None],
                    keys[:, s:e][None],
                    values[:, s:e][None],
                    cache=None,
                    scale=SCALE,
                    mask="causal",
                )
                outs.append(o[0].transpose(1, 0, 2))
            return mx.concatenate(outs, axis=0)

        ta, tm = _p50(absorbed), _p50(materialized)
        print(f"{len(lens):>4} {total:>6} {ta:>8.3f} {tm:>8.3f} {ta / tm:>7.2f}x")


def main() -> None:
    print("GLM-4.7 absorbed MLA prefill — materialized MHA vs absorbed-512 MQA loop")
    print(f"H={H}, qk={QK}, v={VD}, fp16 activations, causal, p50")
    bench(*_build(quantize=False), "dense fp16 weights")
    bench(*_build(quantize=True), "4bit-quantized weights (g64)")


if __name__ == "__main__":
    main()
