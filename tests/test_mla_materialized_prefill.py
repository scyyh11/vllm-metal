# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for the materialized-MLA prefill fast path
(``VLLM_METAL_MLA_MATERIALIZED_PREFILL``). Routing absorbed-MLA prefill through
materialized full K/V + standard MHA (MLX SDPA) must match the absorbed
kv_lora-space path. No custom kernel; works on any GPU."""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import vllm_metal.paged_attention_common as pac
from vllm_metal.mlx_backend.mla_cache import MLAPagedLatentCache
from vllm_metal.paged_attention_backend.mla import MLAPagedAttentionWrapper

MultiLinear = pytest.importorskip("mlx_lm.models.mla").MultiLinear

# GLM-4.7-Flash dims (small num_heads / hidden for a fast test).
_H, _NOPE, _ROPE, _KVL, _VD, _HID, _BLK = 4, 128, 64, 512, 128, 256, 16


class _AbsorbedInner(nn.Module):
    """Absorbed-MLA stub shaped like glm4_moe_lite (MultiLinear embed_q/unembed_out)."""

    def __init__(self) -> None:
        super().__init__()
        self.q_lora_rank = None
        self.num_heads = _H
        self.q_head_dim = _NOPE + _ROPE
        self.qk_nope_head_dim = _NOPE
        self.qk_rope_head_dim = _ROPE
        self.kv_lora_rank = _KVL
        self.v_head_dim = _VD
        self.scale = (_NOPE + _ROPE) ** -0.5
        self.q_proj = nn.Linear(_HID, _H * (_NOPE + _ROPE), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(_HID, _KVL + _ROPE, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(_KVL)
        self.embed_q = MultiLinear(_NOPE, _KVL, _H)
        self.unembed_out = MultiLinear(_KVL, _VD, _H)
        self.o_proj = nn.Linear(_H * _VD, _HID, bias=False)

    def rope(self, x: mx.array, offset: int = 0) -> mx.array:
        return x


@pytest.fixture(autouse=True)
def _clear_ctx():
    pac.clear_context()
    yield
    pac.clear_context()
    os.environ.pop("VLLM_METAL_MLA_MATERIALIZED_PREFILL", None)


def _make():
    mx.random.seed(0)
    inner = _AbsorbedInner()
    inner.apply(lambda p: p.astype(mx.float16))
    cache = MLAPagedLatentCache(
        num_layers=1,
        latent_dim=_KVL + _ROPE,
        num_blocks=8,
        block_size=_BLK,
        dtype=mx.float16,
    )
    return (
        inner,
        cache,
        MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache),
    )


def test_materialized_prefill_matches_absorbed_loop() -> None:
    inner, cache, wrapper = _make()
    lens = [16, 48]  # 2 prefill requests, past=0, block-aligned
    total = sum(lens)
    cu = [0] + [int(c) for c in np.cumsum(lens)]
    ctx = pac.PagedAttentionContext(
        slot_mapping=list(range(total)),
        block_tables=[[0], [1, 2, 3]],
        context_lens=list(lens),
        cu_seqlens=cu,
        offsets=[0, 0],
    )
    x = mx.random.normal((1, total, _HID)).astype(mx.float16)

    def run(flag: str) -> mx.array:
        os.environ["VLLM_METAL_MLA_MATERIALIZED_PREFILL"] = flag
        cache.latent_caches[0] = mx.zeros_like(cache.latent_caches[0])
        pac.set_context(ctx)
        out = wrapper(x, mask=None, cache=None)
        mx.eval(out)
        pac.clear_context()
        return out

    ref = run("0")  # absorbed kv_lora-space (512-wide MQA) loop
    mat = run("1")  # materialized full-K/V MHA

    assert mat.shape == (1, total, _HID)
    np.testing.assert_allclose(np.array(mat), np.array(ref), atol=2e-2, rtol=1e-2)


def test_materialized_prefill_gate_off_by_default() -> None:
    """Without the env flag the fast path must not engage (no behavioral change)."""
    inner, cache, wrapper = _make()
    os.environ.pop("VLLM_METAL_MLA_MATERIALIZED_PREFILL", None)
    assert not wrapper._materialized_prefill_ok(
        inner,
        cache,
        pac.PagedAttentionContext(
            slot_mapping=[0, 1],
            block_tables=[[0]],
            context_lens=[2],
            cu_seqlens=[0, 2],
            offsets=[0],
        ),
    )
