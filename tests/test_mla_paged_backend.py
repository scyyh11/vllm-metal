# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.base import scaled_dot_product_attention

import vllm_metal.paged_attention_common as pac
from vllm_metal.mlx_backend.mla_cache import MLAPagedLatentCache
from vllm_metal.paged_attention_backend.mla import (
    MLAPagedAttentionBackend,
    MLAPagedAttentionWrapper,
)
from vllm_metal.paged_attention_backend.protocol import PagedAttentionBackend

# Fixture dimensions matching GLM/DeepSeek-V2 defaults
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_LATENT_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM


class TestMLAPagedLatentCache:
    def test_latent_dim_stored_correctly(self) -> None:
        cache = MLAPagedLatentCache(
            num_layers=4,
            latent_dim=_LATENT_DIM,
            num_blocks=10,
            block_size=16,
            dtype=mx.float16,
        )

        assert cache.latent_dim == _LATENT_DIM

    def test_per_layer_array_shape(self) -> None:
        cache = MLAPagedLatentCache(
            num_layers=3,
            latent_dim=288,
            num_blocks=8,
            block_size=16,
            dtype=mx.float16,
        )

        assert len(cache.latent_caches) == 3
        for arr in cache.latent_caches:
            assert arr.shape == (8, 16, 288)  # (num_blocks, block_size, latent_dim)
            assert arr.dtype == mx.float16

    def test_bfloat16_dtype_accepted(self) -> None:
        cache = MLAPagedLatentCache(
            num_layers=2,
            latent_dim=192,
            num_blocks=4,
            block_size=8,
            dtype=mx.bfloat16,
        )

        assert cache.dtype == mx.bfloat16

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dtype"):
            MLAPagedLatentCache(
                num_layers=2,
                latent_dim=_LATENT_DIM,
                num_blocks=5,
                block_size=16,
                dtype=mx.int32,
            )


class TestMLAPagedAttentionBackend:
    def _make_backend(self) -> MLAPagedAttentionBackend:
        return MLAPagedAttentionBackend(
            num_layers=4,
            latent_dim=_LATENT_DIM,
            block_size=16,
            dtype=mx.float16,
        )

    def test_implements_paged_attention_backend_protocol(self) -> None:
        backend = self._make_backend()

        assert isinstance(backend, PagedAttentionBackend)

    def test_num_blocks_raises_before_initialize(self) -> None:
        backend = self._make_backend()

        with pytest.raises(RuntimeError, match="called before initialize"):
            backend.num_blocks()

    def test_warm_up_raises_before_initialize(self) -> None:
        backend = self._make_backend()

        with pytest.raises(RuntimeError, match="called before initialize"):
            backend.warm_up()

    def test_patch_model_raises_before_initialize(self) -> None:
        backend = self._make_backend()

        with pytest.raises(RuntimeError, match="called before initialize"):
            backend.patch_model(object())

    def test_num_blocks_after_initialize(self) -> None:
        backend = self._make_backend()
        backend.initialize(50)

        assert backend.num_blocks() == 50

    def test_warm_up_after_initialize_does_not_raise(self) -> None:
        backend = self._make_backend()
        backend.initialize(10)

        backend.warm_up()

    def test_warm_up_skips_kernel_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default path: env var off → warm-up must not import or call the
        Metal kernel module (saves startup cost for users on the MLX
        path)."""
        monkeypatch.delenv("VLLM_METAL_MLA_KERNEL", raising=False)
        backend = MLAPagedAttentionBackend(
            num_layers=1,
            latent_dim=_KERNEL_KV_LORA_RANK + _KERNEL_QK_ROPE_HEAD_DIM,
            block_size=16,
            dtype=mx.float16,
        )
        backend.initialize(4)

        called = MagicMock()
        # Patch into the metal module so any unexpected call surfaces.
        from vllm_metal import metal as vm_metal

        monkeypatch.setattr(vm_metal, "metal_mla_paged_attention", called)
        monkeypatch.setattr(vm_metal, "metal_mla_paged_attention_decode_2pass", called)
        backend.warm_up()
        assert called.call_count == 0

    def test_warm_up_runs_kernel_when_env_set_and_shape_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var on + cache at kernel-instantiated shape → warm-up
        dispatches every (HEADS_PER_TG, PARTITION_SIZE) instantiation the
        dispatcher can reach in production. Single-pass: G ∈ {1, 2}.
        2pass: hpt picked by ``num_heads`` (≥32 → hpt32 else hpt8), ps
        picked by ``max_ctx`` (>1024 → ps128 else ps64) → 2×2 grid."""
        monkeypatch.setenv("VLLM_METAL_MLA_KERNEL", "1")
        backend = MLAPagedAttentionBackend(
            num_layers=1,
            latent_dim=_KERNEL_KV_LORA_RANK + _KERNEL_QK_ROPE_HEAD_DIM,
            block_size=16,
            dtype=mx.float16,
        )
        backend.initialize(4)

        single_pass = MagicMock()
        two_pass = MagicMock()
        from vllm_metal import metal as vm_metal

        monkeypatch.setattr(vm_metal, "metal_mla_paged_attention", single_pass)
        monkeypatch.setattr(
            vm_metal, "metal_mla_paged_attention_decode_2pass", two_pass
        )
        backend.warm_up()

        # Single-pass: G ∈ {1, 2}.
        assert single_pass.call_count == 2
        gs = sorted(call.kwargs["heads_per_tg"] for call in single_pass.call_args_list)
        assert gs == [1, 2]

        # 2pass: cover the 2 × 2 grid. The dispatcher reads num_heads and
        # max(context_lens) to pick (hpt, ps) internally — check those.
        assert two_pass.call_count == 4
        observed: set[tuple[int, int]] = set()
        for call in two_pass.call_args_list:
            num_heads = int(call.kwargs["q_nope"].shape[1])
            max_ctx = int(call.kwargs["context_lens"].max())
            hpt_class = 32 if num_heads >= 32 else 8
            ps_class = 128 if max_ctx > 1024 else 64
            observed.add((hpt_class, ps_class))
        assert observed == {(8, 64), (8, 128), (32, 64), (32, 128)}

    def test_warm_up_skips_kernel_when_shape_mismatches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var on but cache shape isn't the kernel-instantiated one →
        skip warm-up (kernel would never be reachable through
        ``_can_use_kernel`` anyway)."""
        monkeypatch.setenv("VLLM_METAL_MLA_KERNEL", "1")
        # 288 doesn't match the kernel's required 512+64=576 latent_dim.
        backend = MLAPagedAttentionBackend(
            num_layers=1,
            latent_dim=288,
            block_size=16,
            dtype=mx.float16,
        )
        backend.initialize(4)

        called = MagicMock()
        from vllm_metal import metal as vm_metal

        monkeypatch.setattr(vm_metal, "metal_mla_paged_attention", called)
        monkeypatch.setattr(vm_metal, "metal_mla_paged_attention_decode_2pass", called)
        backend.warm_up()
        assert called.call_count == 0

    def test_initialize_allocates_cache_with_correct_shape(self) -> None:
        backend = self._make_backend()

        backend.initialize(20)

        assert backend._cache is not None
        assert backend._cache.num_blocks == 20
        assert backend._cache.latent_dim == _LATENT_DIM
        assert backend._cache.num_layers == 4


class _FakeAttn(nn.Module):
    pass


class _FakeLayer:
    def __init__(self) -> None:
        self.self_attn = _FakeAttn()


class _FakeModel:
    """Minimal stand-in for a model with .model.layers."""

    def __init__(self, num_layers: int) -> None:
        self.model = SimpleNamespace(layers=[_FakeLayer() for _ in range(num_layers)])


class TestPatchModelAttentionMla:
    def _make_backend(self, num_layers: int) -> MLAPagedAttentionBackend:
        backend = MLAPagedAttentionBackend(
            num_layers=num_layers,
            latent_dim=_LATENT_DIM,
            block_size=16,
            dtype=mx.float16,
        )
        backend.initialize(5)
        return backend

    def test_replaces_all_attention_layers(self) -> None:
        model = _FakeModel(num_layers=3)

        n = self._make_backend(num_layers=3).patch_model(model)

        assert n == 3
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, MLAPagedAttentionWrapper)

    def test_wrapped_layer_has_correct_index(self) -> None:
        model = _FakeModel(num_layers=2)

        self._make_backend(num_layers=2).patch_model(model)

        for idx, layer in enumerate(model.model.layers):
            assert layer.self_attn._mla_layer_idx == idx

    def test_already_patched_layers_update_cache_reference(self) -> None:
        model = _FakeModel(num_layers=1)
        backend_a = self._make_backend(num_layers=1)
        backend_b = self._make_backend(num_layers=1)
        backend_a.patch_model(model)

        n = backend_b.patch_model(model)

        assert n == 1
        assert model.model.layers[0].self_attn._mla_latent_cache is backend_b._cache

    def test_returns_correct_patch_count(self) -> None:
        for n_layers in (1, 4, 10):
            model = _FakeModel(num_layers=n_layers)

            count = self._make_backend(num_layers=n_layers).patch_model(model)

            assert count == n_layers


class TestMLAPagedAttentionWrapperFallback:
    def test_delegates_to_inner_when_no_paged_context(self) -> None:
        sentinel = object()
        inner = MagicMock(return_value=sentinel)
        latent_cache = MagicMock(spec=MLAPagedLatentCache)

        wrapper = MLAPagedAttentionWrapper(
            inner, layer_idx=0, latent_cache=latent_cache
        )

        x = mx.zeros((1, 3, 64))
        result = wrapper(x, mask=None, cache=None)

        inner.assert_called_once_with(x, mask=None, cache=None)
        assert result is sentinel

    def test_passes_mask_and_cache_to_inner(self) -> None:
        inner = MagicMock(return_value=mx.zeros((1, 2, 32)))
        latent_cache = MagicMock(spec=MLAPagedLatentCache)
        wrapper = MLAPagedAttentionWrapper(
            inner, layer_idx=1, latent_cache=latent_cache
        )
        x = mx.zeros((1, 2, 32))
        mask = object()
        cache = object()

        wrapper(x, mask=mask, cache=cache)

        inner.assert_called_once_with(x, mask=mask, cache=cache)


_HIDDEN = 32
_NUM_HEADS = 2
_NOPE_DIM = 8  # qk_nope_head_dim
_ROPE_DIM = 4  # qk_rope_head_dim
_KV_RANK = 16  # kv_lora_rank
_V_DIM = 8  # v_head_dim
_Q_LORA_RANK = 12  # q_lora_rank

# Production-shape fixtures matching the Metal kernel instantiation
# (kernels_v2/mla.metal — kv_lora_rank=512, qk_rope_head_dim=64).
# Other dims kept small so wrapper-level tests stay fast.
_KERNEL_KV_LORA_RANK = 512
_KERNEL_QK_ROPE_HEAD_DIM = 64
_KERNEL_NOPE_DIM = 32
_KERNEL_V_DIM = 32
_KERNEL_NUM_HEADS = 2
_KERNEL_HIDDEN = _KERNEL_NUM_HEADS * _KERNEL_V_DIM


class _MinimalMLAInner(nn.Module):
    """Minimal MLA attention stub with correct shapes for paged path tests."""

    def __init__(self) -> None:
        super().__init__()
        self.q_lora_rank = None
        self.num_heads = _NUM_HEADS
        self.q_head_dim = _NOPE_DIM + _ROPE_DIM
        self.qk_nope_head_dim = _NOPE_DIM
        self.qk_rope_head_dim = _ROPE_DIM
        self.kv_lora_rank = _KV_RANK
        self.scale = 1.0 / math.sqrt(_KV_RANK)

        self.q_proj = nn.Linear(_HIDDEN, _NUM_HEADS * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(_HIDDEN, _KV_RANK + _ROPE_DIM, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(_KV_RANK)
        self.embed_q = nn.Linear(_NOPE_DIM, _KV_RANK, bias=False)
        self.unembed_out = nn.Linear(_KV_RANK, _V_DIM, bias=False)
        self.o_proj = nn.Linear(_NUM_HEADS * _V_DIM, _HIDDEN, bias=False)

    def rope(self, x: mx.array, offset: int = 0) -> mx.array:
        # Identity RoPE: preserves shape, sufficient for testing shape logic.
        return x


class _AbsorbedKernelInner(nn.Module):
    """Absorbed-MLA stub at kernel-instantiated shapes (kv_lora_rank=512,
    qk_rope_head_dim=64). Triggers the wrapper's fast path on decode."""

    def __init__(self, num_heads: int = _KERNEL_NUM_HEADS) -> None:
        super().__init__()
        self.q_lora_rank = None
        self.num_heads = num_heads
        self.q_head_dim = _KERNEL_NOPE_DIM + _KERNEL_QK_ROPE_HEAD_DIM
        self.qk_nope_head_dim = _KERNEL_NOPE_DIM
        self.qk_rope_head_dim = _KERNEL_QK_ROPE_HEAD_DIM
        self.kv_lora_rank = _KERNEL_KV_LORA_RANK
        self.scale = 1.0 / math.sqrt(_KERNEL_KV_LORA_RANK)

        hidden = num_heads * _KERNEL_V_DIM
        self.q_proj = nn.Linear(hidden, num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden,
            _KERNEL_KV_LORA_RANK + _KERNEL_QK_ROPE_HEAD_DIM,
            bias=False,
        )
        self.kv_a_layernorm = nn.LayerNorm(_KERNEL_KV_LORA_RANK)
        self.embed_q = nn.Linear(_KERNEL_NOPE_DIM, _KERNEL_KV_LORA_RANK, bias=False)
        self.unembed_out = nn.Linear(_KERNEL_KV_LORA_RANK, _KERNEL_V_DIM, bias=False)
        self.o_proj = nn.Linear(num_heads * _KERNEL_V_DIM, hidden, bias=False)

    def rope(self, x: mx.array, offset: int = 0) -> mx.array:
        return x


def _absorbed_dense_reference(
    inner: _AbsorbedKernelInner,
    full: mx.array,  # [1, total_len, hidden] — concatenated past + new
    *,
    cache_dtype: mx.Dtype,
) -> mx.array:
    """Dense absorbed-MLA forward over the full sequence; the test slices
    out the last (decode) token. Mirrors the wrapper's math:
      score = scale * (embed_q(q_nope) · kv_norm + q_pe · k_pe)
      out   = unembed_out(softmax(score) @ kv_norm)
    Cache roundtrip is a no-op when input and cache_dtype both fp16, so
    no extra quantization steps are needed beyond a single .astype().
    """
    _, total_len, _ = full.shape

    q = inner.q_proj(full)
    q = q.reshape(1, total_len, inner.num_heads, inner.q_head_dim).transpose(0, 2, 1, 3)
    q_nope, q_pe = mx.split(q, [inner.qk_nope_head_dim], axis=-1)

    kv_out = inner.kv_a_proj_with_mqa(full)
    compressed_kv, k_pe = mx.split(kv_out, [inner.kv_lora_rank], axis=-1)
    kv_norm = inner.kv_a_layernorm(compressed_kv).astype(cache_dtype)
    k_pe = (
        k_pe.astype(cache_dtype)
        .reshape(1, total_len, 1, inner.qk_rope_head_dim)
        .transpose(0, 2, 1, 3)
    )

    scale = inner.scale
    # PE branch broadcasts across heads (k_pe head dim is 1).
    pe_scores = (q_pe * scale) @ k_pe.swapaxes(-1, -2)

    q_nope_proj = inner.embed_q(q_nope)
    kv = kv_norm.reshape(1, 1, total_len, inner.kv_lora_rank)

    # Causal mask for prefill positions; pe_scores already gates softmax via
    # SDPA's mask argument (matches _apply_absorbed_mla_attention).
    if total_len > 1:
        rows = mx.arange(total_len).reshape(-1, 1)
        cols = mx.arange(total_len).reshape(1, -1)
        valid = (cols <= rows).reshape(1, 1, total_len, total_len)
        fill = mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
        pe_scores = mx.where(valid, pe_scores, fill)

    out = scaled_dot_product_attention(
        q_nope_proj, kv, kv, cache=None, scale=scale, mask=pe_scores
    )
    out = inner.unembed_out(out)
    out = out.transpose(0, 2, 1, 3).reshape(1, total_len, -1)
    return inner.o_proj(out)


class _MiniCPM3StyleInner(nn.Module):
    """MLA stub shaped like MiniCPM3: softmax_scale and kv_b_proj only."""

    def __init__(self) -> None:
        super().__init__()
        self.q_lora_rank = _Q_LORA_RANK
        self.num_heads = _NUM_HEADS
        self.q_head_dim = _NOPE_DIM + _ROPE_DIM
        self.qk_nope_head_dim = _NOPE_DIM
        self.qk_rope_head_dim = _ROPE_DIM
        self.kv_lora_rank = _KV_RANK
        self.softmax_scale = 0.37

        self.q_a_proj = nn.Linear(_HIDDEN, _Q_LORA_RANK, bias=False)
        self.q_a_layernorm = nn.LayerNorm(_Q_LORA_RANK)
        self.q_b_proj = nn.Linear(
            _Q_LORA_RANK, _NUM_HEADS * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(_HIDDEN, _KV_RANK + _ROPE_DIM, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(_KV_RANK)
        self.kv_b_proj = nn.Linear(
            _KV_RANK, _NUM_HEADS * (_NOPE_DIM + _V_DIM), bias=False
        )
        self.o_proj = nn.Linear(_NUM_HEADS * _V_DIM, _HIDDEN, bias=False)

    def rope(self, x: mx.array, offset: int = 0) -> mx.array:
        return x


def _minicpm3_dense_reference(
    inner: _MiniCPM3StyleInner,
    x: mx.array,
    *,
    cache_dtype: mx.Dtype,
) -> mx.array:
    _, seq_len, _ = x.shape

    q = inner.q_b_proj(inner.q_a_layernorm(inner.q_a_proj(x)))
    q = q.reshape(1, seq_len, inner.num_heads, inner.q_head_dim).transpose(0, 2, 1, 3)
    q_nope, q_pe = mx.split(q, [inner.qk_nope_head_dim], axis=-1)

    kv_out = inner.kv_a_proj_with_mqa(x)
    compressed_kv, k_pe = mx.split(kv_out, [inner.kv_lora_rank], axis=-1)
    kv_norm = inner.kv_a_layernorm(compressed_kv).astype(cache_dtype)
    k_pe = k_pe.reshape(1, seq_len, 1, inner.qk_rope_head_dim).transpose(0, 2, 1, 3)
    k_pe = k_pe.astype(cache_dtype)

    kv = inner.kv_b_proj(kv_norm)
    kv = kv.reshape(1, seq_len, inner.num_heads, -1).transpose(0, 2, 1, 3)
    k_nope, values = mx.split(kv, [inner.qk_nope_head_dim], axis=-1)
    k_pe = mx.broadcast_to(
        k_pe,
        (1, inner.num_heads, seq_len, inner.qk_rope_head_dim),
    )

    queries = mx.concatenate([q_nope, q_pe], axis=-1)
    keys = mx.concatenate([k_nope, k_pe], axis=-1)
    attn_mask = None
    if seq_len > 1:
        rows = mx.arange(seq_len).reshape(-1, 1)
        cols = mx.arange(seq_len).reshape(1, -1)
        valid = (cols <= rows).reshape(1, 1, seq_len, seq_len)
        fill = mx.array(mx.finfo(queries.dtype).min, queries.dtype)
        attn_mask = mx.where(valid, mx.array(0, queries.dtype), fill)

    out = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=inner.softmax_scale,
        mask=attn_mask,
    )
    out = out.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)
    return inner.o_proj(out)


class TestMLAPagedAttentionWrapperPagedPath:
    """Exercises the paged attention computation path (PagedAttentionContext set)."""

    @pytest.fixture(autouse=True)
    def _clear_ctx(self) -> Generator[None, None, None]:
        pac.clear_context()
        yield
        pac.clear_context()

    def _make_cache(self) -> MLAPagedLatentCache:
        return MLAPagedLatentCache(
            num_layers=1,
            latent_dim=_KV_RANK + _ROPE_DIM,
            num_blocks=4,
            block_size=4,
            dtype=mx.float16,
        )

    def test_decode_output_shape(self) -> None:
        # 1 request, 3 cached tokens, 1 new decode token
        inner = _MinimalMLAInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 1],
                offsets=[3],
            )
        )

        out = wrapper(
            mx.random.normal((1, 1, _HIDDEN)).astype(mx.float16), mask=None, cache=None
        )
        mx.eval(out)

        assert out.shape == (1, 1, _HIDDEN)

    def test_prefill_output_shape(self) -> None:
        # 1 request, 0 past tokens, 4 new prefill tokens
        inner = _MinimalMLAInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2, 3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 4],
                offsets=[0],
            )
        )

        out = wrapper(
            mx.random.normal((1, 4, _HIDDEN)).astype(mx.float16), mask=None, cache=None
        )
        mx.eval(out)

        assert out.shape == (1, 4, _HIDDEN)

    def test_cache_written_at_correct_slot(self) -> None:
        # Scatter-write: only the assigned slot is non-zero after the call
        inner = _MinimalMLAInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[2],
                block_tables=[[0]],
                context_lens=[3],
                cu_seqlens=[0, 1],
                offsets=[2],
            )
        )

        wrapper(
            mx.random.normal((1, 1, _HIDDEN)).astype(mx.float16), mask=None, cache=None
        )

        # block 0, position 2 should now hold the new latent
        written = cache.latent_caches[0][0, 2, :]
        untouched = cache.latent_caches[0][0, 0, :]

        assert bool(mx.any(written != 0))
        assert not bool(mx.any(untouched != 0))

    def test_two_decode_requests_combined_output_shape(self) -> None:
        # Two decode requests in one batch — outputs must be concatenated along seq axis.
        inner = _MinimalMLAInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        # Request A: 2 past tokens, decode token at slot 2 in block 0
        # Request B: 1 past token,  decode token at slot 5 in block 1
        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[2, 5],
                block_tables=[[0], [1]],
                context_lens=[3, 2],
                cu_seqlens=[0, 1, 2],
                offsets=[2, 1],
            )
        )

        x = mx.random.normal((1, 2, _HIDDEN)).astype(mx.float16)
        out = wrapper(x, mask=None, cache=None)
        mx.eval(out)

        assert out.shape == (1, 2, _HIDDEN)

    def test_causal_mask_token0_output_independent_of_later_tokens(self) -> None:
        # Token 0 in a prefill can only attend to itself (causal mask).
        # Changing tokens 1-3 must not change token 0's output.
        inner = _MinimalMLAInner()

        # Run 1: prefill with input_a
        cache_a = self._make_cache()
        wrapper_a = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache_a)
        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2, 3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 4],
                offsets=[0],
            )
        )
        mx.random.seed(0)
        token0 = mx.random.normal((1, 1, _HIDDEN)).astype(mx.float16)
        other = mx.random.normal((1, 3, _HIDDEN)).astype(mx.float16)
        input_a = mx.concatenate([token0, other], axis=1)
        out_a = wrapper_a(input_a, mask=None, cache=None)
        mx.eval(out_a)

        pac.clear_context()

        # Run 2: same token 0, completely different tokens 1-3
        cache_b = self._make_cache()
        wrapper_b = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache_b)
        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2, 3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 4],
                offsets=[0],
            )
        )
        mx.random.seed(99)
        different_other = mx.random.normal((1, 3, _HIDDEN)).astype(mx.float16)
        input_b = mx.concatenate([token0, different_other], axis=1)
        out_b = wrapper_b(input_b, mask=None, cache=None)
        mx.eval(out_b)

        # Token 0 output must be identical — it attends only to position 0
        assert bool(mx.all(out_a[0, 0, :] == out_b[0, 0, :]))

    def test_minicpm3_style_prefill_matches_dense_reference(self) -> None:
        inner = _MiniCPM3StyleInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2, 3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 4],
                offsets=[0],
            )
        )

        mx.random.seed(7)
        x = mx.random.normal((1, 4, _HIDDEN)).astype(mx.float16)
        out = wrapper(x, mask=None, cache=None)
        expected = _minicpm3_dense_reference(inner, x, cache_dtype=cache.dtype)
        mx.eval(out, expected)

        assert bool(mx.allclose(out, expected, rtol=1e-3, atol=1e-3))

    def test_minicpm3_style_decode_matches_dense_reference(self) -> None:
        inner = _MiniCPM3StyleInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        mx.random.seed(11)
        past = mx.random.normal((1, 3, _HIDDEN)).astype(mx.float16)
        new = mx.random.normal((1, 1, _HIDDEN)).astype(mx.float16)

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2],
                block_tables=[[0]],
                context_lens=[3],
                cu_seqlens=[0, 3],
                offsets=[0],
            )
        )
        wrapper(past, mask=None, cache=None)
        pac.clear_context()

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 1],
                offsets=[3],
            )
        )
        out = wrapper(new, mask=None, cache=None)
        dense = _minicpm3_dense_reference(
            inner,
            mx.concatenate([past, new], axis=1),
            cache_dtype=cache.dtype,
        )
        expected = dense[:, -1:, :]
        mx.eval(out, expected)

        assert bool(mx.allclose(out, expected, rtol=1e-3, atol=1e-3))


class TestMLAAbsorbedKernelPath:
    """Wrapper-level tests for the absorbed-MLA kernel fast path.

    Production shapes (kv_lora_rank=512, qk_rope_head_dim=64, block_size=16)
    so MLAPagedAttentionWrapper._can_use_kernel returns True on decode and
    metal_mla_paged_attention is dispatched. Confirms the wrapper-side
    plumbing — embed_q reshape, block_tables packing, unembed_out + o_proj
    flow — agrees numerically with the dense reference end-to-end."""

    @pytest.fixture(autouse=True)
    def _clear_ctx(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> Generator[None, None, None]:
        # Opt into the experimental MLA kernel path. Production default is
        # MLX SDPA per RFC #360; this class specifically exercises the
        # kernel fast path, so it sets the env var for the duration of
        # each test.
        monkeypatch.setenv("VLLM_METAL_MLA_KERNEL", "1")
        pac.clear_context()
        yield
        pac.clear_context()

    def _make_cache(self, block_size: int = 16) -> MLAPagedLatentCache:
        return MLAPagedLatentCache(
            num_layers=1,
            latent_dim=_KERNEL_KV_LORA_RANK + _KERNEL_QK_ROPE_HEAD_DIM,
            num_blocks=8,
            block_size=block_size,
            dtype=mx.float16,
        )

    def test_can_use_kernel_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sanity: the gate accepts production shapes + decode and rejects
        when prerequisites aren't met. Also: kernel path stays off when
        ``VLLM_METAL_MLA_KERNEL`` is unset (RFC #360 default)."""
        inner = _AbsorbedKernelInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        decode_ctx = SimpleNamespace(
            cu_seqlens=[0, 1, 2],  # two decode requests
            context_lens=[3, 5],
            block_tables=[[0], [1]],
        )
        assert wrapper._can_use_kernel(inner, cache, decode_ctx)

        # Without the opt-in env var, the gate must reject even when every
        # other prerequisite (shape, dtype, decode-only) is satisfied.
        monkeypatch.delenv("VLLM_METAL_MLA_KERNEL", raising=False)
        assert not wrapper._can_use_kernel(inner, cache, decode_ctx)
        monkeypatch.setenv("VLLM_METAL_MLA_KERNEL", "1")

        prefill_ctx = SimpleNamespace(
            cu_seqlens=[0, 4],  # one request, 4 query tokens
            context_lens=[4],
            block_tables=[[0]],
        )
        assert not wrapper._can_use_kernel(inner, cache, prefill_ctx)

        # fp32 cache: MLAPagedLatentCache accepts it, but the kernel is
        # only instantiated for half / bfloat16_t. Gate must reject so the
        # fast path doesn't dispatch a non-existent specialisation.
        fp32_cache = MLAPagedLatentCache(
            num_layers=1,
            latent_dim=_KERNEL_KV_LORA_RANK + _KERNEL_QK_ROPE_HEAD_DIM,
            num_blocks=4,
            block_size=16,
            dtype=mx.float32,
        )
        assert not wrapper._can_use_kernel(inner, fp32_cache, decode_ctx)

    def test_can_use_kernel_does_not_route_to_slow_path_on_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate must not introduce shape-based fall-throughs to the
        MLX slow path — the project goal is to replace MLX entirely,
        so every shape (including the historically-MLX-favoured B=1
        long ctx) has to go through the kernel for the gap to be
        visible and attackable."""
        monkeypatch.setenv("VLLM_METAL_MLA_KERNEL", "1")
        inner = _AbsorbedKernelInner()
        cache = self._make_cache()
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        # B=1 short, B=1 long, B=2 long — all must enter the kernel.
        for ctx_len, n_blocks in [(128, 8), (512, 32), (2048, 128)]:
            ctx_obj = SimpleNamespace(
                cu_seqlens=[0, 1],
                context_lens=[ctx_len],
                block_tables=[list(range(n_blocks))],
            )
            assert wrapper._can_use_kernel(inner, cache, ctx_obj), (
                f"gate rejected B=1 ctx={ctx_len} — must enter kernel"
            )

    def test_fp32_cache_routes_to_slow_path(self) -> None:
        """End-to-end: a wrapper with an fp32 latent cache must complete
        decode without dispatching the (non-existent) fp32 kernel
        specialisation. Catches regressions if the gate later loses the
        dtype check."""
        inner = _AbsorbedKernelInner()
        cache = MLAPagedLatentCache(
            num_layers=1,
            latent_dim=_KERNEL_KV_LORA_RANK + _KERNEL_QK_ROPE_HEAD_DIM,
            num_blocks=8,
            block_size=16,
            dtype=mx.float32,
        )
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        # Single-request decode: prefill 3 past tokens, then decode 1.
        mx.random.seed(21)
        past = mx.random.normal((1, 3, _KERNEL_HIDDEN)).astype(mx.float32)
        new = mx.random.normal((1, 1, _KERNEL_HIDDEN)).astype(mx.float32)

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2],
                block_tables=[[0]],
                context_lens=[3],
                cu_seqlens=[0, 3],
                offsets=[0],
            )
        )
        wrapper(past, mask=None, cache=None)
        pac.clear_context()

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 1],
                offsets=[3],
            )
        )
        # Must not raise — slow path handles fp32 caches just fine.
        out = wrapper(new, mask=None, cache=None)
        mx.eval(out)
        assert out.shape == (1, 1, _KERNEL_HIDDEN)

    def test_decode_matches_dense_reference(self) -> None:
        """Run prefill (slow path) + decode (fast path) and compare the
        decode token's output against a dense MLX reference."""
        inner = _AbsorbedKernelInner()
        block_size = 16
        cache = self._make_cache(block_size=block_size)
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        # 5 past tokens + 1 decode token; total ctx_len=6 fits one block.
        mx.random.seed(7)
        past = mx.random.normal((1, 5, _KERNEL_HIDDEN)).astype(mx.float16)
        new = mx.random.normal((1, 1, _KERNEL_HIDDEN)).astype(mx.float16)

        # Prefill writes positions 0..4 into block 0 — slow path (num_new=5).
        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2, 3, 4],
                block_tables=[[0]],
                context_lens=[5],
                cu_seqlens=[0, 5],
                offsets=[0],
            )
        )
        wrapper(past, mask=None, cache=None)
        pac.clear_context()

        # Decode at position 5 — fast path (num_new=1, absorbed, prod shapes).
        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[5],
                block_tables=[[0]],
                context_lens=[6],
                cu_seqlens=[0, 1],
                offsets=[5],
            )
        )
        out = wrapper(new, mask=None, cache=None)
        dense = _absorbed_dense_reference(
            inner,
            mx.concatenate([past, new], axis=1),
            cache_dtype=cache.dtype,
        )
        expected = dense[:, -1:, :]
        mx.eval(out, expected)

        assert bool(mx.allclose(out, expected, rtol=1e-3, atol=1e-3))

    def test_two_decode_requests_match_dense_reference(self) -> None:
        """Batched decode (two requests, different ctx_lens) — fast path
        with a multi-row block_tables, exercises the wrapper's per-request
        padding into the 2D int32 array consumed by the kernel."""
        inner = _AbsorbedKernelInner()
        block_size = 16
        cache = self._make_cache(block_size=block_size)
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        # Prefill request A (4 tokens at block 0) and request B (3 tokens at
        # block 1) separately on the slow path so each request lands in its
        # own block. Then decode both in one batched fast-path call.
        mx.random.seed(13)
        past_a = mx.random.normal((1, 4, _KERNEL_HIDDEN)).astype(mx.float16)
        past_b = mx.random.normal((1, 3, _KERNEL_HIDDEN)).astype(mx.float16)
        new_a = mx.random.normal((1, 1, _KERNEL_HIDDEN)).astype(mx.float16)
        new_b = mx.random.normal((1, 1, _KERNEL_HIDDEN)).astype(mx.float16)

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2, 3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 4],
                offsets=[0],
            )
        )
        wrapper(past_a, mask=None, cache=None)
        pac.clear_context()

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[16, 17, 18],  # block 1 starts at slot 16
                block_tables=[[1]],
                context_lens=[3],
                cu_seqlens=[0, 3],
                offsets=[0],
            )
        )
        wrapper(past_b, mask=None, cache=None)
        pac.clear_context()

        # Batched decode: A at slot 4 (ctx_len=5), B at slot 19 (ctx_len=4).
        decode_x = mx.concatenate([new_a, new_b], axis=1)
        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[4, 19],
                block_tables=[[0], [1]],
                context_lens=[5, 4],
                cu_seqlens=[0, 1, 2],
                offsets=[4, 3],
            )
        )
        out = wrapper(decode_x, mask=None, cache=None)
        mx.eval(out)

        # Reference: each request's last token via _absorbed_dense_reference
        dense_a = _absorbed_dense_reference(
            inner,
            mx.concatenate([past_a, new_a], axis=1),
            cache_dtype=cache.dtype,
        )
        dense_b = _absorbed_dense_reference(
            inner,
            mx.concatenate([past_b, new_b], axis=1),
            cache_dtype=cache.dtype,
        )
        expected = mx.concatenate([dense_a[:, -1:, :], dense_b[:, -1:, :]], axis=1)
        mx.eval(expected)

        assert bool(mx.allclose(out, expected, rtol=1e-3, atol=1e-3))

    def test_kernel_fast_path_uses_single_pass_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Wrapper contract: with the MLX sdpa_vector–style decode kernel,
        the fast path always dispatches single-pass and never the
        partitioned variant. Bench showed split-K never wins over the
        new single-pass kernel; this test locks that in. The single-pass
        path now goes through ``metal_mla_paged_attention_primitive``
        (the lazy MLX Primitive variant added in Stage 9) — patch that
        instead of the eager binding."""
        from vllm_metal import metal as vm_metal

        single_pass = MagicMock(
            return_value=mx.zeros((1, 2, _KERNEL_KV_LORA_RANK), dtype=mx.float16)
        )
        partitioned = MagicMock()
        monkeypatch.setattr(
            vm_metal, "metal_mla_paged_attention_primitive", single_pass
        )
        monkeypatch.setattr(
            vm_metal,
            "metal_mla_paged_attention_partitioned",
            partitioned,
        )

        inner = _AbsorbedKernelInner()
        cache = MLAPagedLatentCache(
            num_layers=1,
            latent_dim=_KERNEL_KV_LORA_RANK + _KERNEL_QK_ROPE_HEAD_DIM,
            num_blocks=200,
            block_size=32,
            dtype=mx.float16,
        )
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        # Probe across the (low B*H, long ctx) corner that previously
        # routed to split-K, plus a high-batch case for good measure.
        cases = [
            # (n_seqs, ctx_len, blocks_per_seq) — long ctx + low batch
            (1, 4096, 128),
            # short ctx + low batch
            (1, 64, 1),
        ]
        for n_seqs, ctx_len, n_blocks_per_seq in cases:
            single_pass.reset_mock()
            partitioned.reset_mock()
            bts = [
                list(range(i * n_blocks_per_seq, (i + 1) * n_blocks_per_seq))
                for i in range(n_seqs)
            ]
            ctx_obj = SimpleNamespace(
                cu_seqlens=list(range(n_seqs + 1)),
                context_lens=[ctx_len] * n_seqs,
                block_tables=bts,
            )
            mx.random.seed(0)
            q_nope = mx.random.normal(
                (1, _KERNEL_NUM_HEADS, n_seqs, _KERNEL_NOPE_DIM)
            ).astype(mx.float16)
            q_pe = mx.random.normal(
                (1, _KERNEL_NUM_HEADS, n_seqs, _KERNEL_QK_ROPE_HEAD_DIM)
            ).astype(mx.float16)

            wrapper._kernel_fast_path(
                inner=inner,
                latent_cache=cache,
                layer_idx=0,
                q_nope=q_nope,
                q_pe=q_pe,
                ctx=ctx_obj,
                seq_len=n_seqs,
            )

            assert single_pass.call_count == 1, (
                f"single-pass not called for n_seqs={n_seqs}, ctx_len={ctx_len}"
            )
            assert partitioned.call_count == 0, (
                f"partitioned unexpectedly called for n_seqs={n_seqs}, "
                f"ctx_len={ctx_len}"
            )

    def test_queued_decodes_dont_alias_kernel_output(self) -> None:
        """Regression: two queued wrapper calls with the same seq_len must
        produce independent outputs. An earlier single-slot scratch cache
        reused the kernel's ``out_kvr`` buffer across calls; because
        downstream ``unembed_out`` / ``o_proj`` are lazy, the second
        kernel's write would overwrite the first call's pending graph
        before it evaluated. Failure mode: out_1 ends up equal to out_2
        (both reading the last kernel write). Fresh per-call allocation
        prevents this — this test fails if the cache is reintroduced
        without a safety signal."""
        inner = _AbsorbedKernelInner()
        cache = self._make_cache(block_size=16)
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        mx.random.seed(11)
        past = mx.random.normal((1, 4, _KERNEL_HIDDEN)).astype(mx.float16)
        new_x1 = mx.random.normal((1, 1, _KERNEL_HIDDEN)).astype(mx.float16)
        # Very different content from x1 — if the buffer aliases, out_1
        # would read x2's attention result instead of its own.
        new_x2 = mx.random.normal((1, 1, _KERNEL_HIDDEN)).astype(mx.float16) * 5

        # Prefill 4 past tokens via slow path (writes slots 0..3 into block 0).
        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[0, 1, 2, 3],
                block_tables=[[0]],
                context_lens=[4],
                cu_seqlens=[0, 4],
                offsets=[0],
            )
        )
        wrapper(past, mask=None, cache=None)
        pac.clear_context()
        # Evaluate the prefill so the cache state for the decode reference
        # is committed independently of the queued calls below.
        mx.eval(cache.latent_caches[0])

        # Reference for call 1: dense forward over past || x1, last token.
        full_for_call_1 = mx.concatenate([past, new_x1], axis=1)
        correct_out_1 = _absorbed_dense_reference(
            inner, full_for_call_1, cache_dtype=cache.dtype
        )[:, -1:, :]
        mx.eval(correct_out_1)

        # Now queue the two decode calls without eval'ing in between.
        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[4],
                block_tables=[[0]],
                context_lens=[5],
                cu_seqlens=[0, 1],
                offsets=[4],
            )
        )
        out_1 = wrapper(new_x1, mask=None, cache=None)
        pac.clear_context()

        pac.set_context(
            pac.PagedAttentionContext(
                slot_mapping=[5],
                block_tables=[[0]],
                context_lens=[6],
                cu_seqlens=[0, 1],
                offsets=[5],
            )
        )
        out_2 = wrapper(new_x2, mask=None, cache=None)
        pac.clear_context()

        # Now eval both. If the kernel's output buffer is shared, out_1's
        # lazy graph reads the post-call-2 buffer state and matches out_2;
        # without aliasing, out_1 matches its own reference and differs
        # materially from out_2.
        mx.eval(out_1, out_2)

        assert bool(mx.allclose(out_1, correct_out_1, rtol=1e-3, atol=1e-3))
        assert not bool(mx.allclose(out_1, out_2, rtol=1e-3, atol=1e-3))

    def test_pick_fa_variant_routing(self) -> None:
        """Verify ``_pick_fa_variant`` matches the bench-driven routing.

        Returns ``"fa"`` only when FA wide beats single-pass and 2pass
        head-to-head; otherwise ``None`` so the wrapper picks
        single-pass / 2pass."""
        pick = MLAPagedAttentionWrapper._pick_fa_variant

        # H % 8 != 0 → never FA.
        assert pick(2, 8, 128, 16) is None
        assert pick(7, 8, 2048, 16) is None
        assert pick(20, 8, 2048, 16) is None

        # B = 1: single-pass dominates regardless of ctx/H.
        assert pick(8, 1, 128, 16) is None
        assert pick(16, 1, 2048, 16) is None
        assert pick(128, 1, 8192, 16) is None

        # B ≥ 2, num_heads < 32: single-pass still wins.
        assert pick(8, 8, 128, 16) is None
        assert pick(16, 8, 2048, 16) is None
        assert pick(24, 16, 4096, 16) is None

        # B ≥ 2, num_heads ≥ 32, ctx < 2048: fa wide.
        assert pick(32, 2, 128, 16) == "fa"
        assert pick(40, 8, 128, 16) == "fa"
        # H=40 keeps fa at long ctx too; the medium-ctx 2pass redirect
        # is limited to H ∈ {96, 128}.
        assert pick(40, 8, 8192, 16) == "fa"
        # H=64 keeps fa at ctx ≤ 4096; only longer contexts redirect.
        assert pick(64, 4, 2048, 16) == "fa"
        assert pick(64, 8, 4096, 16) == "fa"
        # Long ctx + high-batch + 64 ≤ H ≤ 128: fa wide loses to 2pass.
        assert pick(64, 8, 8192, 16) is None
        # H ∈ {96, 128} B ≥ 4 ctx ≥ 2048 bs=16: fa loses to 2pass at
        # ctx=2048 too, not just ctx>4096.
        assert pick(96, 8, 2048, 16) is None
        assert pick(128, 8, 2048, 16) is None
        assert pick(128, 32, 2048, 16) is None
        # Long ctx unchanged for H=96/128.
        assert pick(96, 8, 8192, 16) is None
        assert pick(128, 8, 8192, 16) is None
        # Guard: ctx < 2048 keeps fa for H=96.
        assert pick(96, 8, 128, 16) == "fa"
        # H=128 B≥4 bs=16 (any ctx) → None; even ctx<2048 routes to
        # 2pass because the lazy 2pass path integrates better.
        assert pick(128, 8, 128, 16) is None
        assert pick(128, 8, 1024, 16) is None
        assert pick(128, 16, 128, 16) is None
        # Guard: H=128 B < 4 still picks fa.
        assert pick(128, 2, 128, 16) == "fa"
        # Guard: B < 4 keeps fa for H=96 ctx ≥ 2048.
        assert pick(96, 2, 2048, 16) == "fa"

        # B ≥ 16: high-batch always routes to 2pass (launch overhead
        # amortizes regardless of H/ctx). Without this gate fa would
        # grab H ∈ {40, 64 ctx≤4096, 96 ctx<2048, ...} cells before
        # ``_should_use_2pass`` is even checked.
        assert pick(40, 16, 128, 16) is None
        assert pick(64, 16, 4096, 16) is None
        assert pick(96, 16, 128, 16) is None
        assert pick(40, 32, 2048, 16) is None

        # block_size=32 caches have no FA instantiation — must fall
        # through regardless of (H, B, ctx) so the wrapper picks
        # 2pass / single-pass instead of raising from the FA dispatcher.
        assert pick(64, 8, 2048, 32) is None
        assert pick(128, 8, 2048, 32) is None
        assert pick(96, 8, 2048, 32) is None

    def test_should_use_2pass_routing(self) -> None:
        """Verify ``_should_use_2pass`` matches the bench-driven
        routing rules: high-batch, long-ctx, H=96/128 medium-ctx,
        B=1 long-ctx H≥40, and H=128 short-ctx bs=16.
        """
        use = MLAPagedAttentionWrapper._should_use_2pass

        # High batch routes to 2pass regardless of ctx/H.
        assert use(8, 16, 128, 16) is True
        assert use(128, 32, 2048, 16) is True

        # 64 ≤ H ≤ 128 B≥4 long-ctx → 2pass.
        assert use(64, 8, 8192, 16) is True
        assert use(128, 8, 8192, 16) is True
        assert use(96, 4, 5000, 16) is True

        # H ∈ {96, 128} B≥4 ctx≥2048 bs=16 → 2pass.
        assert use(96, 4, 2048, 16) is True
        assert use(96, 8, 2048, 16) is True
        assert use(128, 8, 2048, 16) is True
        # Guard: bs=32 NOT covered for H=96 (audit showed
        # forced 2pass at bs=32 either loses or is in noise band).
        assert use(96, 8, 2048, 32) is False
        # Guard: H=64 NOT covered at ctx=2048.
        assert use(64, 8, 2048, 16) is False
        # Guard: B<4 NOT covered at ctx=2048.
        assert use(96, 2, 2048, 16) is False
        assert use(128, 2, 2048, 16) is False

        # H=128 B≥4 bs=16 → 2pass at any ctx.
        assert use(128, 4, 128, 16) is True
        assert use(128, 8, 128, 16) is True
        assert use(128, 16, 1024, 16) is True
        # Guard: bs=32 short/medium ctx stays on single-pass.
        assert use(128, 8, 128, 32) is False
        assert use(128, 8, 2048, 32) is False

        # B=1 H≥40 ctx≥8192 → 2pass.
        assert use(40, 1, 8192, 16) is True
        assert use(64, 1, 8192, 16) is True
        assert use(96, 1, 8192, 16) is True
        assert use(128, 1, 8192, 16) is True
        # bs is not constrained — both bs values benefit.
        assert use(40, 1, 8192, 32) is True
        assert use(96, 1, 8192, 32) is True
        # Guard: H<40 keeps single-pass at long ctx.
        assert use(16, 1, 8192, 16) is False
        # Guard: ctx<8192 keeps single-pass at B=1.
        assert use(96, 1, 2048, 16) is False
        assert use(128, 1, 4096, 16) is False
        assert use(128, 1, 128, 16) is False

        # Existing low-batch low-H short-ctx fallthroughs unchanged.
        assert use(16, 1, 128, 16) is False
        assert use(16, 8, 2048, 16) is False
        assert use(32, 8, 4096, 16) is False

    def test_kernel_fast_path_routes_per_workload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Wrapper routes to the bench-best kernel per cell:

        - B = 1 + (H<40 OR ctx<8192): single-pass.
        - B = 1 + H≥40 + ctx≥8192: 2pass.
        - B ≥ 2 with num_heads < 32: single-pass.
        - B ≥ 2 with num_heads ≥ 32 and ctx < 2048: FA wide.
        - B ≥ 4 with H ∈ {96, 128} and ctx ≥ 2048 bs=16: 2pass.
        - B ≥ 16: 2pass.
        """
        from vllm_metal import metal as vm_metal

        # All four lazy Primitive entry points — mocked so we can assert
        # which one fired per cell. Mocks return mx.zeros so the
        # downstream unembed_out chain still has a valid array.
        fa = MagicMock(
            side_effect=lambda **kw: mx.zeros(
                kw["q_nope"].shape, dtype=kw["q_nope"].dtype
            )
        )
        fa_part = MagicMock()
        two_pass = MagicMock()
        pr_mma = MagicMock(
            side_effect=lambda **kw: mx.zeros(
                (kw["q_combined"].shape[0], kw["q_combined"].shape[1], 512),
                dtype=kw["q_combined"].dtype,
            )
        )
        single_pass = MagicMock(
            side_effect=lambda **kw: mx.zeros(
                kw["q_nope"].shape, dtype=kw["q_nope"].dtype
            )
        )
        monkeypatch.setattr(
            vm_metal, "metal_mla_paged_attention_decode_fa_primitive", fa
        )
        monkeypatch.setattr(
            vm_metal, "metal_mla_paged_attention_decode_fa_partitioned", fa_part
        )
        monkeypatch.setattr(
            vm_metal,
            "metal_mla_paged_attention_decode_2pass_primitive",
            two_pass,
        )
        monkeypatch.setattr(
            vm_metal,
            "metal_mla_paged_attention_decode_pr_mma_primitive",
            pr_mma,
        )
        monkeypatch.setattr(
            vm_metal, "metal_mla_paged_attention_primitive", single_pass
        )

        def _make_wrapper(num_heads: int):
            inner = _AbsorbedKernelInner(num_heads=num_heads)
            cache = MLAPagedLatentCache(
                num_layers=1,
                latent_dim=_KERNEL_KV_LORA_RANK + _KERNEL_QK_ROPE_HEAD_DIM,
                num_blocks=2048,
                block_size=16,
                dtype=mx.float16,
            )
            return (
                inner,
                cache,
                MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache),
            )

        cases = [
            # (num_heads, n_seqs, ctx_len, expected mock)
            (8, 1, 128, single_pass),  # B=1 H<40 → single
            (8, 1, 2048, single_pass),  # B=1 H<40 → single
            (8, 8, 128, single_pass),  # H<32 → single
            (16, 8, 2048, single_pass),  # H<32 long ctx → single
            (40, 8, 128, fa),  # H≥32 short → fa wide
            (64, 4, 2048, fa),  # H=64 ctx=2048 → fa wide
            (96, 8, 128, fa),  # H=96 ctx<2048 → fa wide
            # H ∈ {96, 128} B≥4 ctx≥2048 bs=16 → 2pass.
            (96, 8, 2048, two_pass),
            (96, 8, 8192, two_pass),
            (128, 8, 2048, two_pass),
            # H=128 B≥4 bs=16 ctx<2048 → 2pass.
            (128, 4, 128, two_pass),
            (128, 8, 128, two_pass),
            # H=128 ctx>4096 hits pr_mma before 2pass.
            (128, 8, 8192, pr_mma),
            # B=1 H≥40 ctx≥8192 → 2pass.
            (40, 1, 8192, two_pass),
            (64, 1, 8192, two_pass),
            (96, 1, 8192, two_pass),
            (128, 1, 8192, two_pass),
            # Guard: B=1 H≥40 ctx<8192 still single-pass.
            (40, 1, 2048, single_pass),
            (96, 1, 2048, single_pass),
            (128, 1, 4096, single_pass),
            # Guard: B=1 H<40 long ctx still single-pass.
            (16, 1, 8192, single_pass),
            # B ≥ 16: high-batch always 2pass (the bug fix — without
            # the B≥16 guard in _pick_fa_variant these would have
            # routed to fa wide).
            (40, 16, 128, two_pass),
            (64, 16, 4096, two_pass),
            (96, 16, 128, two_pass),
            (40, 32, 2048, two_pass),
        ]
        for num_heads, n_seqs, ctx_len, expected in cases:
            for m in (fa, fa_part, two_pass, pr_mma, single_pass):
                m.reset_mock()
            inner, cache, wrapper = _make_wrapper(num_heads)
            n_blocks = (ctx_len + 16 - 1) // 16
            bts = [list(range(i * n_blocks, (i + 1) * n_blocks)) for i in range(n_seqs)]
            ctx_obj = SimpleNamespace(
                cu_seqlens=list(range(n_seqs + 1)),
                context_lens=[ctx_len] * n_seqs,
                block_tables=bts,
            )
            mx.random.seed(0)
            q_nope = mx.random.normal((1, num_heads, n_seqs, _KERNEL_NOPE_DIM)).astype(
                mx.float16
            )
            q_pe = mx.random.normal(
                (1, num_heads, n_seqs, _KERNEL_QK_ROPE_HEAD_DIM)
            ).astype(mx.float16)

            wrapper._kernel_fast_path(
                inner=inner,
                latent_cache=cache,
                layer_idx=0,
                q_nope=q_nope,
                q_pe=q_pe,
                ctx=ctx_obj,
                seq_len=n_seqs,
            )

            assert expected.call_count == 1, (
                f"expected {expected._mock_name or 'kernel'} not called for "
                f"H={num_heads}, B={n_seqs}, ctx={ctx_len}"
            )
            others = {fa, fa_part, two_pass, pr_mma, single_pass} - {expected}
            for m in others:
                assert m.call_count == 0, (
                    f"unexpected kernel fired for H={num_heads}, B={n_seqs}, "
                    f"ctx={ctx_len}"
                )

    def test_kernel_fast_path_block_size_32_skips_fa(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bs=32 cache passes ``_can_use_kernel`` (2pass / single-pass
        support both bs ∈ {16, 32}), but FA's dispatchers are only
        instantiated for bs=16. Verify the routing predicate falls
        through to 2pass / single-pass instead of raising from the FA
        dispatcher when bs=32."""
        from vllm_metal import metal as vm_metal

        fa = MagicMock()
        fa_part = MagicMock()
        two_pass = MagicMock()
        # Single-pass is now invoked through the Primitive variant.
        single_pass = MagicMock(
            side_effect=lambda **kw: mx.zeros(
                kw["q_nope"].shape, dtype=kw["q_nope"].dtype
            )
        )
        monkeypatch.setattr(vm_metal, "metal_mla_paged_attention_decode_fa", fa)
        monkeypatch.setattr(
            vm_metal, "metal_mla_paged_attention_decode_fa_partitioned", fa_part
        )
        monkeypatch.setattr(
            vm_metal,
            "metal_mla_paged_attention_decode_2pass_primitive",
            two_pass,
        )
        monkeypatch.setattr(
            vm_metal, "metal_mla_paged_attention_primitive", single_pass
        )

        inner = _AbsorbedKernelInner(num_heads=8)
        cache = MLAPagedLatentCache(
            num_layers=1,
            latent_dim=_KERNEL_KV_LORA_RANK + _KERNEL_QK_ROPE_HEAD_DIM,
            num_blocks=256,
            block_size=32,
            dtype=mx.float16,
        )
        wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

        # Two cells that *would* route to fa / fa_partitioned at bs=16
        # (H=8 short ctx → fa; H=8 long ctx low-batch → fa_partitioned).
        for ctx_len in (128, 2048):
            for m in (fa, fa_part, two_pass, single_pass):
                m.reset_mock()
            n_blocks = (ctx_len + 32 - 1) // 32
            ctx_obj = SimpleNamespace(
                cu_seqlens=[0, 1],
                context_lens=[ctx_len],
                block_tables=[list(range(n_blocks))],
            )
            mx.random.seed(0)
            q_nope = mx.random.normal((1, 8, 1, _KERNEL_NOPE_DIM)).astype(mx.float16)
            q_pe = mx.random.normal((1, 8, 1, _KERNEL_QK_ROPE_HEAD_DIM)).astype(
                mx.float16
            )

            wrapper._kernel_fast_path(
                inner=inner,
                latent_cache=cache,
                layer_idx=0,
                q_nope=q_nope,
                q_pe=q_pe,
                ctx=ctx_obj,
                seq_len=1,
            )

            assert fa.call_count == 0, f"FA fired for bs=32 ctx={ctx_len}"
            assert fa_part.call_count == 0, (
                f"FA-partitioned fired for bs=32 ctx={ctx_len}"
            )
            # Either 2pass or single-pass must service the call.
            assert two_pass.call_count + single_pass.call_count == 1

    def test_kernel_fast_path_bs32_medium_ctx_guards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """bs=32 medium-ctx H=96/128 guards stay off 2pass."""
        from vllm_metal import metal as vm_metal

        fa = MagicMock()
        fa_part = MagicMock()
        two_pass = MagicMock()
        single_pass = MagicMock(
            side_effect=lambda **kw: mx.zeros(
                kw["q_nope"].shape, dtype=kw["q_nope"].dtype
            )
        )
        monkeypatch.setattr(
            vm_metal, "metal_mla_paged_attention_decode_fa_primitive", fa
        )
        monkeypatch.setattr(
            vm_metal, "metal_mla_paged_attention_decode_fa_partitioned", fa_part
        )
        monkeypatch.setattr(
            vm_metal,
            "metal_mla_paged_attention_decode_2pass_primitive",
            two_pass,
        )
        monkeypatch.setattr(
            vm_metal, "metal_mla_paged_attention_primitive", single_pass
        )

        def _make_wrapper(num_heads: int, block_size: int):
            inner = _AbsorbedKernelInner(num_heads=num_heads)
            cache = MLAPagedLatentCache(
                num_layers=1,
                latent_dim=_KERNEL_KV_LORA_RANK + _KERNEL_QK_ROPE_HEAD_DIM,
                num_blocks=2048,
                block_size=block_size,
                dtype=mx.float16,
            )
            return (
                inner,
                cache,
                MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache),
            )

        # Also mock pr_mma since H=128 long-ctx routes to it at bs=32 too.
        pr_mma = MagicMock(
            side_effect=lambda **kw: mx.zeros(
                (kw["q_combined"].shape[0], kw["q_combined"].shape[1], 512),
                dtype=kw["q_combined"].dtype,
            )
        )
        monkeypatch.setattr(
            vm_metal,
            "metal_mla_paged_attention_decode_pr_mma_primitive",
            pr_mma,
        )

        # bs=32 cells: the medium-ctx 2pass redirect requires bs=16,
        # so these stay on
        # single-pass even though their bs=16 sibling routes to 2pass.
        # H=128 B≥4 ctx>4096 routes to pr_mma at bs=32 too.
        cases = [
            # (num_heads, n_seqs, ctx_len, expected mock)
            (96, 8, 2048, single_pass),
            (128, 8, 2048, single_pass),
            # B=1 long-ctx rule is bs-agnostic — should still go 2pass.
            (96, 1, 8192, two_pass),
            (128, 1, 8192, two_pass),
            # High-batch fallthrough rule still fires for bs=32.
            (128, 16, 2048, two_pass),
            # H=128 B≥4 ctx>4096 bs=32 → pr_mma.
            (128, 8, 8192, pr_mma),
        ]
        for num_heads, n_seqs, ctx_len, expected in cases:
            for m in (fa, fa_part, two_pass, pr_mma, single_pass):
                m.reset_mock()
            inner, cache, wrapper = _make_wrapper(num_heads, block_size=32)
            n_blocks = (ctx_len + 32 - 1) // 32
            bts = [list(range(i * n_blocks, (i + 1) * n_blocks)) for i in range(n_seqs)]
            ctx_obj = SimpleNamespace(
                cu_seqlens=list(range(n_seqs + 1)),
                context_lens=[ctx_len] * n_seqs,
                block_tables=bts,
            )
            mx.random.seed(0)
            q_nope = mx.random.normal((1, num_heads, n_seqs, _KERNEL_NOPE_DIM)).astype(
                mx.float16
            )
            q_pe = mx.random.normal(
                (1, num_heads, n_seqs, _KERNEL_QK_ROPE_HEAD_DIM)
            ).astype(mx.float16)

            wrapper._kernel_fast_path(
                inner=inner,
                latent_cache=cache,
                layer_idx=0,
                q_nope=q_nope,
                q_pe=q_pe,
                ctx=ctx_obj,
                seq_len=n_seqs,
            )

            assert expected.call_count == 1, (
                f"expected {expected._mock_name or 'kernel'} not called for "
                f"bs=32 H={num_heads} B={n_seqs} ctx={ctx_len}"
            )
            others = {fa, fa_part, two_pass, pr_mma, single_pass} - {expected}
            for m in others:
                assert m.call_count == 0, (
                    f"unexpected kernel fired for bs=32 H={num_heads} "
                    f"B={n_seqs} ctx={ctx_len}"
                )
