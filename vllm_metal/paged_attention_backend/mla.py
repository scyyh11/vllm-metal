# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from vllm.logger import init_logger

from vllm_metal import envs
from vllm_metal.metal_kernel_backend.packed_prefill_compat import apply_packed_rope
from vllm_metal.mlx_backend.mla_cache import MLAPagedLatentCache
from vllm_metal.paged_attention_common import find_attn_attr, find_layers, get_context

logger = init_logger(__name__)

# Default rope head dim for GLM/DeepSeek-V2 lineage models.
# Used as fallback when qk_rope_head_dim is absent from model config.
MLA_DEFAULT_QK_ROPE_HEAD_DIM = 64


class MLAPagedAttentionWrapper(nn.Module):
    """Wraps an MLA attention module to use a paged latent cache.

    MLA (GLM/DeepSeek/MiniCPM3 lineage) compresses KV into a latent before caching:

        latent = [kv_norm || k_pe_roped]  # kv_lora_rank + qk_rope_head_dim dims

    Each call scatter-writes the new tokens' latents into the scheduled cache
    slots, then gather-reads all past latents per request via block tables.

    Some models expose absorbed MLA helpers: embed_q projects q_nope into
    kv_lora_rank space, and unembed_out maps the output back to v_head_dim.
    MiniCPM3 instead keeps kv_b_proj as the public K/V reconstruction path.
    This wrapper handles both layouts while sharing the paged latent cache.

    When no PagedAttentionContext is active the original module is called as-is.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        latent_cache: MLAPagedLatentCache,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_mla_layer_idx", layer_idx)
        object.__setattr__(self, "_mla_latent_cache", latent_cache)
        is_absorbed = hasattr(inner, "embed_q") and hasattr(inner, "unembed_out")
        object.__setattr__(self, "_is_absorbed", is_absorbed)
        if is_absorbed:
            object.__setattr__(
                self, "_apply_mla_attention", self._apply_absorbed_mla_attention
            )
        else:
            object.__setattr__(
                self, "_apply_mla_attention", self._apply_kv_b_proj_attention
            )
        # Note: an earlier revision cached the kernel's ``out_kvr`` buffer
        # across calls (single-slot, keyed on (seq_len, num_heads, dtype))
        # to elide a ~150μs B=1 ``mx.zeros`` zero-fill. That was unsound:
        # ``unembed_out(out_kvr)`` and downstream ``o_proj`` build a lazy
        # MLX graph that captures the buffer by reference, not value. The
        # dispatcher's ``mx.synchronize()`` only waits for the kernel's
        # GPU write; it does not force the lazy host graph to evaluate.
        # If two wrapper calls queue before the first output is consumed,
        # the second kernel's write to the same buffer corrupts the first
        # call's pending graph. Allocating fresh per-call avoids the
        # aliasing entirely. Re-introducing a cache requires a safety
        # signal that MLX doesn't expose today.

    def _attention_scale(self) -> float:
        inner = self._inner
        scale = getattr(inner, "scale", None)
        if scale is None:
            scale = inner.softmax_scale
        return scale

    @staticmethod
    def _causal_valid_mask(
        *,
        num_new: int,
        ctx_len: int,
        past_len: int,
    ) -> mx.array | None:
        if num_new == 1:
            return None
        rows = mx.arange(num_new).reshape(-1, 1)
        cols = mx.arange(ctx_len).reshape(1, -1)
        return (cols <= (past_len + rows)).reshape(1, 1, num_new, ctx_len)

    def _apply_absorbed_mla_attention(
        self,
        *,
        rq_nope: mx.array,
        rq_pe: mx.array,
        all_kv_norm: mx.array,
        k_pe: mx.array,
        causal_mask: mx.array | None,
    ) -> mx.array:
        inner = self._inner
        scale = self._attention_scale()

        # PE branch: q_pe · k_pe contributes an additive score bias.
        # Passing this as the `mask` to scaled_dot_product_attention adds it
        # to the nope scores before softmax, matching the original model exactly.
        pe_scores = (rq_pe * scale) @ k_pe.swapaxes(-1, -2)
        if causal_mask is not None:
            fill = mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
            pe_scores = mx.where(causal_mask, pe_scores, fill)

        # Nope branch: embed_q absorbs q_nope into kv_lora_rank space;
        # kv_norm is shared across heads as k=v (single-head broadcast).
        ctx_len = all_kv_norm.shape[0]
        rq_nope_proj = inner.embed_q(rq_nope)
        kv = all_kv_norm.reshape(1, 1, ctx_len, inner.kv_lora_rank)

        out = scaled_dot_product_attention(
            rq_nope_proj, kv, kv, cache=None, scale=scale, mask=pe_scores
        )
        return inner.unembed_out(out)  # recover v_head_dim from kv_lora_rank

    def _apply_kv_b_proj_attention(
        self,
        *,
        rq_nope: mx.array,
        rq_pe: mx.array,
        all_kv_norm: mx.array,
        k_pe: mx.array,
        causal_mask: mx.array | None,
    ) -> mx.array:
        inner = self._inner
        scale = self._attention_scale()
        ctx_len = all_kv_norm.shape[0]

        # MiniCPM3-style MLA keeps a single kv_b_proj instead of pre-split
        # embed_q/unembed_out modules. Rebuild K/V from the cached latent using
        # the model's own projection to preserve quantized Linear behavior and
        # the source model's layout.
        kv = inner.kv_b_proj(all_kv_norm.reshape(1, ctx_len, inner.kv_lora_rank))
        kv = kv.reshape(1, ctx_len, inner.num_heads, -1).transpose(0, 2, 1, 3)
        k_nope, values = mx.split(kv, [inner.qk_nope_head_dim], axis=-1)
        k_pe = mx.broadcast_to(
            k_pe,
            (1, inner.num_heads, ctx_len, inner.qk_rope_head_dim),
        )
        queries = mx.concatenate([rq_nope, rq_pe], axis=-1)
        keys = mx.concatenate([k_nope, k_pe], axis=-1)
        attn_mask = None
        if causal_mask is not None:
            fill = mx.array(mx.finfo(queries.dtype).min, queries.dtype)
            attn_mask = mx.where(causal_mask, mx.array(0, queries.dtype), fill)

        return scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=scale, mask=attn_mask
        )

    # Production shapes the Metal MLA kernel is instantiated for. Anything
    # else falls back to the per-request MLX path below.
    _KERNEL_KV_LORA_RANK = 512
    _KERNEL_QK_ROPE_HEAD_DIM = 64
    _KERNEL_BLOCK_SIZES = (16, 32)
    # MLAPagedLatentCache also accepts mx.float32, but the kernel is only
    # instantiated for half / bfloat16_t — fp32 caches must stay on MLX.
    _KERNEL_DTYPES = (mx.float16, mx.bfloat16)

    # Split-K + reduce path is intentionally NOT auto-routed. With the
    # MLX sdpa_vector–style decode kernel (1024 threads, 32 simdgroups
    # already striding the ctx axis at 1-token granularity), the single-pass
    # path saturates the simdgroup-internal parallelism on its own; bench on
    # M5 Max shows split-K never wins after the rewrite (it just adds two
    # dispatches + reduce shmem overhead). The partitioned kernel + Python
    # entry are kept available for future workloads where per-TG cost grows
    # — e.g. an MLA-TQ dequant fused into score path, where each token's
    # work is heavier and ctx-split occupancy gain can re-emerge.

    def _can_use_kernel(
        self,
        inner: nn.Module,
        latent_cache: MLAPagedLatentCache,
        ctx: Any,
    ) -> bool:
        """Fast-path gate. Activates when:
        - ``VLLM_METAL_MLA_KERNEL=1`` (experimental opt-in; off by default
          per RFC #360 — MLX SDPA remains the production default until the
          kernel ≥ MLX across the production grid),
        - absorbed MLA layout (embed_q / unembed_out present),
        - every request has exactly one query token (decode-only — varlen
          prefill lands in P2),
        - inner dims match the kernel instantiation (kv_lora_rank=512,
          qk_rope_head_dim=64),
        - the cache block size and dtype match an instantiated specialisation.

        The gate intentionally does NOT route around shapes where the
        kernel currently loses to MLX. The project goal is to replace
        MLX entirely, so every reachable (B, H, ctx) cell must be
        beaten on the kernel side — falling through to the MLX slow
        path masks the real gaps."""
        if not envs.VLLM_METAL_MLA_KERNEL:
            return False
        if not self._is_absorbed:
            return False
        if inner.kv_lora_rank != self._KERNEL_KV_LORA_RANK:
            return False
        if inner.qk_rope_head_dim != self._KERNEL_QK_ROPE_HEAD_DIM:
            return False
        if latent_cache.block_size not in self._KERNEL_BLOCK_SIZES:
            return False
        if latent_cache.dtype not in self._KERNEL_DTYPES:
            return False
        cu = ctx.cu_seqlens
        for i in range(len(ctx.context_lens)):
            if cu[i + 1] - cu[i] != 1:
                return False
        return True

    @staticmethod
    def _pick_heads_per_tg(num_heads: int, batch_size: int) -> int:
        """Pick HEADS_PER_TG (G) for the kernel based on production shape.

        G=2 packs 2 query heads into one threadgroup so each K/V load is
        reused for 2 dot products. Bench on M5 Max shows ~1.5–2× speedup
        vs G=1 once the GPU is saturated (B*H ≳ 30 launched threadgroups);
        small (B=1, num_heads small) workloads under-saturate the cores
        and G=1's higher per-TG parallelism (NUM_THREADS=1024 vs 512)
        wins instead. G=4 was instantiated too but its per-thread state
        (4× q/v slices) over-pressures the register file and ends up
        slower than G=1 across the board, so we don't auto-select it.

        Falls back to G=1 when num_heads is odd (kernel requires
        num_heads % G == 0)."""
        if num_heads % 2 != 0:
            return 1
        # B=1 with small head counts under-fills the GPU; G=1's wider TGs
        # absorb the latency better there. Empirical breakeven on M5 Max.
        if batch_size == 1 and num_heads < 32:
            return 1
        return 2

    @staticmethod
    def _pick_fa_variant(
        num_heads: int, batch_size: int, max_ctx: int, block_size: int
    ) -> str | None:
        """Pick the FA decode kernel variant for this workload, or
        ``None`` to fall through to the 2pass / single-pass kernels.

        FA's simdgroup_matrix tile requires ``num_heads % 8 == 0``;
        ``_can_use_kernel`` accepts both bs ∈ {16, 32}, but the FA
        dispatchers are only instantiated for bs=16, so route bs=32
        through 2pass / single-pass.

        Bench-driven routing:

        - **B = 1**: FA under-saturates, so return ``None`` and let
          ``_should_use_2pass`` catch the long-ctx H≥40 case; otherwise
          the wrapper falls through to single-pass.
        - **B ≥ 16**: 2pass amortizes launch overhead regardless of
          H/ctx (matches ``_should_use_2pass``'s first rule). Return
          ``None`` so the wrapper falls through to 2pass instead of
          routing high-batch traffic through FA wide.
        - **B ≥ 2, num_heads < 32**: single-pass primitive still
          wins — ``num_head_groups`` too small to fill FA wide's
          per-TG simdgroup budget. Return ``None``.
        - **B ≥ 4, 64 ≤ num_heads ≤ 128, ctx > 4096**: 2pass
          primitive wins. FA wide's outer K-iter count grows
          linearly with ctx; at ctx=8192 the bandwidth-bound region
          dominates and partitioned 2pass splits the K axis (kernel-
          only: 2pass 3500 μs vs fa-wide 4312 μs at H=96 B=8
          ctx=8192, -19%). Return ``None`` so ``_should_use_2pass``
          picks 2pass up.
        - **B ≥ 2, num_heads ≥ 32**: FA wide primitive otherwise
          wins at short / medium ctx (350 μs short, 880-1380 μs at
          ctx≤2048 vs single-pass 400-1700 μs). Return ``"fa"``.

        FA-partitioned never wins on the current production grid.
        The wrapper does not route to it. The kernel + Python entry
        remain available for parity tests and bench harnesses only.
        """
        if num_heads % 8 != 0:
            return None
        if block_size != 16:
            return None
        if batch_size == 1:
            return None
        # B ≥ 16: 2pass amortizes launch overhead regardless of H/ctx.
        # `_should_use_2pass` returns True for every B ≥ 16 cell; without
        # this gate FA wide would grab high-batch cells (H ∈ {40, 64,
        # 96 ctx<2048}, ...) before the 2pass branch is even reached.
        if batch_size >= 16:
            return None
        if num_heads < 32:
            return None
        if max_ctx > 4096 and batch_size >= 4 and 64 <= num_heads <= 128:
            return None  # → _should_use_2pass picks 2pass
        # H ∈ {96, 128} B≥4 ctx≥2048 bs=16: fa loses to 2pass at
        # ctx=2048 too, so push fa out and let _should_use_2pass pick up
        # the route.
        if batch_size >= 4 and num_heads in (96, 128) and max_ctx >= 2048:
            return None
        # H=128 B≥4 bs=16 (any ctx): fa and 2pass are close kernel-only,
        # but 2pass integrates better with the post-kernel lazy graph.
        # Narrow to H=128; H=96 keeps the ctx≥2048 rule above.
        if batch_size >= 4 and num_heads == 128:
            return None
        return "fa"

    @staticmethod
    def _should_use_pr_mma(
        num_heads: int, batch_size: int, max_ctx: int, block_size: int
    ) -> bool:
        """Route to the per-request MMA paged decode at the parity-gate
        cells where the MMA-based score+SV path beats 2pass.

        Narrow predicate intentionally — the kernel is instantiated for
        H=128 ctx>4096 B≥4 at bs ∈ {16, 32}. Other shapes stay on the
        established single-pass / FA / 2pass routing.

        Bench shows the MMA main kernel is about 15 % faster than 2pass
        main at both bs=16 and bs=32; the loop body is bs-agnostic.
        """
        return (
            num_heads == 128
            and batch_size >= 4
            and max_ctx > 4096
            and block_size in (16, 32)
        )

    @staticmethod
    def _should_use_2pass(
        num_heads: int,
        batch_size: int,
        max_ctx: int = 0,
        block_size: int = 16,
    ) -> bool:
        """Route to the MLX-style 2pass kernel where the (seq,
        partition) launch grid pays off.

        Four routes hit this:

        - **B ≥ 16**: launch overhead amortizes regardless of ctx.
        - **B ≥ 4 AND 64 ≤ num_heads ≤ 128 AND ctx > 4096**: long-ctx
          medium-to-high-H regime. FA wide's outer K-iter count
          grows linearly with ctx (e.g. ctx=8192 → 128 outer iters
          per TG); 2pass splits the K axis across more launched TGs
          and amortizes via partition parallelism.
        - **B ≥ 4 AND H ∈ {96, 128} AND ctx ≥ 2048 AND bs=16**:
          fa wide loses to 2pass at ctx=2048 too — the
          partition-parallelism gain shows up one ctx-bucket earlier
          than the broader long-ctx rule. ``_pick_fa_variant`` returns
          None here so fa doesn't grab the route first.
          Gated on bs=16 because forced-2pass at bs=32 either loses
          or is in the noise band.
        - **B == 1 AND H ≥ 40 AND ctx ≥ 8192**: 2pass beats
          single-pass at B=1 long-ctx once H≥32 fills the
          HEADS_PER_TG=32 wide-TG layout.

        The wrapper routes 2pass through
        ``metal_mla_paged_attention_decode_2pass_primitive`` (lazy
        MLX Primitive). The eager
        ``metal_mla_paged_attention_decode_2pass`` binding is kept
        for bench tools and 2pass parity tests.
        """
        if batch_size >= 16:
            return True
        if batch_size >= 4 and 64 <= num_heads <= 128 and max_ctx > 4096:
            return True
        if (
            batch_size >= 4
            and num_heads in (96, 128)
            and max_ctx >= 2048
            and block_size == 16
        ):
            return True
        # H=128 B≥4 bs=16: extend to all ctx. Combined with the
        # predicate-order pr_mma rule which catches ctx>4096, this rule
        # serves ctx ∈ [16, 4096] for H=128.
        if batch_size >= 4 and num_heads == 128 and block_size == 16:
            return True
        if batch_size == 1 and num_heads >= 40 and max_ctx >= 8192:
            return True
        return False

    def _kernel_fast_path(
        self,
        inner: nn.Module,
        latent_cache: MLAPagedLatentCache,
        layer_idx: int,
        q_nope: mx.array,  # [1, num_heads, seq_len, qk_nope_head_dim]
        q_pe: mx.array,  # [1, num_heads, seq_len, qk_rope_head_dim] (post-RoPE)
        ctx: Any,
        seq_len: int,
    ) -> mx.array:
        """Decode fast path: project q_nope through embed_q, dispatch the
        kernel for the whole batch in one call, recover v_head_dim through
        unembed_out, and concatenate for o_proj. Replaces the per-request
        Python loop entirely when the gate above accepts."""
        # Use the Metal kernel wrapper lazily so the fallback path doesn't
        # take an unnecessary import dependency.
        from vllm_metal.metal import (
            metal_mla_paged_attention_decode_2pass_primitive,
            metal_mla_paged_attention_decode_fa_primitive,
            metal_mla_paged_attention_decode_pr_mma_primitive,
            metal_mla_paged_attention_primitive,
        )

        # The kernel is instantiated for half / bfloat16_t only; cast Q to
        # the cache dtype so we hit a real specialisation. In production
        # this is a no-op (weights are already fp16/bf16); test fixtures
        # with default fp32 Linear weights need the cast to match the
        # latent-cache dtype the cache was scatter-written in.
        target_dtype = latent_cache.dtype

        # embed_q maps qk_nope_head_dim -> kv_lora_rank along the last dim.
        # Same shape contract as in _apply_absorbed_mla_attention.
        q_nope_proj = inner.embed_q(q_nope).astype(target_dtype)
        q_pe_t = q_pe.astype(target_dtype)
        # [1, num_heads, seq_len, kv_lora_rank] -> [seq_len, num_heads, kv_lora_rank]
        q_nope_kernel = q_nope_proj.transpose(0, 2, 1, 3).reshape(
            seq_len, inner.num_heads, inner.kv_lora_rank
        )
        q_pe_kernel = q_pe_t.transpose(0, 2, 1, 3).reshape(
            seq_len, inner.num_heads, inner.qk_rope_head_dim
        )

        # Pad block_tables (list[list[int]]) into a 2D [num_seqs, max_blocks]
        # int32 array. The kernel reads block_table_row[0..n_context_blocks-1];
        # padding entries beyond n_context_blocks are never read.
        import numpy as np

        bts = ctx.block_tables
        num_seqs = len(bts)
        max_blocks = max(len(bt) for bt in bts)
        bt_np = np.zeros((num_seqs, max_blocks), dtype=np.int32)
        for i, bt in enumerate(bts):
            bt_np[i, : len(bt)] = bt
        block_tables_mx = mx.array(bt_np)

        context_lens_mx = mx.array(list(ctx.context_lens), dtype=mx.uint32)
        cu_seqlens_q_mx = mx.array(list(ctx.cu_seqlens), dtype=mx.int32)

        max_ctx = max(ctx.context_lens) if ctx.context_lens else 0
        fa_variant = self._pick_fa_variant(
            inner.num_heads, seq_len, max_ctx, latent_cache.block_size
        )

        # use_wide=True selects the BQ=8/BK=64/WN=8 → 256-thread variant
        # of the FA kernels. Wide is the production pick: it is no worse
        # in the gate cells and 30-40% faster at long ctx where doubling
        # per-iter K throughput pays off.
        if fa_variant == "fa":
            # Same kernel body as the eager binding, but returns a
            # deferred ``mx.array`` so the call joins the wrapper's lazy
            # graph instead of forcing a dispatch boundary.
            out_kvr = metal_mla_paged_attention_decode_fa_primitive(
                q_nope=q_nope_kernel,
                q_pe=q_pe_kernel,
                latent_cache=latent_cache.latent_caches[layer_idx],
                block_tables=block_tables_mx,
                context_lens=context_lens_mx,
                cu_seqlens_q=cu_seqlens_q_mx,
                scale=self._attention_scale(),
                use_wide=True,
            )
        elif self._should_use_pr_mma(
            inner.num_heads, seq_len, max_ctx, latent_cache.block_size
        ):
            # Per-request MMA paged decode (lazy MLX Primitive). Narrow
            # routing: H=128, B≥4, ctx>4096, bs∈{16,32}. Concatenate
            # q_nope + q_pe into q_combined first because the kernel reads
            # Q as a single D=KVR+PE vector.
            q_combined_kernel = mx.concatenate([q_nope_kernel, q_pe_kernel], axis=-1)
            out_kvr = metal_mla_paged_attention_decode_pr_mma_primitive(
                q_combined=q_combined_kernel,
                latent_cache=latent_cache.latent_caches[layer_idx],
                block_tables=block_tables_mx,
                context_lens=context_lens_mx,
                cu_seqlens_q=cu_seqlens_q_mx,
                scale=self._attention_scale(),
            )
        elif self._should_use_2pass(
            inner.num_heads, seq_len, max_ctx, latent_cache.block_size
        ):
            # Lazy MLX Primitive variant. Joins the wrapper graph and
            # folds scratch (exp_sums / max_logits / tmp_out) into the
            # kernel's own eval_gpu so they never cross the Python
            # boundary.
            out_kvr = metal_mla_paged_attention_decode_2pass_primitive(
                q_nope=q_nope_kernel,
                q_pe=q_pe_kernel,
                latent_cache=latent_cache.latent_caches[layer_idx],
                block_tables=block_tables_mx,
                context_lens=context_lens_mx,
                cu_seqlens_q=cu_seqlens_q_mx,
                scale=self._attention_scale(),
            )
        else:
            # Single-pass via the lazy MLX Primitive variant. The kernel
            # call joins the wrapper's lazy graph rather than forcing
            # an mx.eval boundary — saves ~200 μs per dispatch at
            # B=1/H≤64 cells where MLX per-call overhead dominates.
            # Single-pass is the routing pick at every cell where this
            # branch is reached (per ``_pick_fa_variant`` /
            # ``_should_use_2pass`` — B=1 or B≥2 H<32, B<16).
            out_kvr = metal_mla_paged_attention_primitive(
                q_nope=q_nope_kernel,
                q_pe=q_pe_kernel,
                latent_cache=latent_cache.latent_caches[layer_idx],
                block_tables=block_tables_mx,
                context_lens=context_lens_mx,
                cu_seqlens_q=cu_seqlens_q_mx,
                scale=self._attention_scale(),
                heads_per_tg=self._pick_heads_per_tg(inner.num_heads, seq_len),
            )

        # Recover v_head_dim and assemble [1, seq_len, num_heads * v_head_dim]
        # for o_proj — matching the slow path's exit shape.
        out_for_unembed = out_kvr.reshape(
            1, seq_len, inner.num_heads, inner.kv_lora_rank
        ).transpose(0, 2, 1, 3)
        out_unembedded = inner.unembed_out(out_for_unembed)
        return out_unembedded.transpose(0, 2, 1, 3).reshape(1, seq_len, -1)

    def _slow_path_per_request(
        self,
        inner: nn.Module,
        latent_cache: MLAPagedLatentCache,
        layer_idx: int,
        q_nope: mx.array,
        q_pe: mx.array,
        ctx: Any,
    ) -> mx.array:
        """Per-request gather + MLX SDPA. Used when the kernel fast-path
        gate rejects (kv_b_proj path, varlen prefill, non-instantiated
        shapes). Identical behaviour to the original wrapper body before
        step 12 — preserved for correctness fallback."""
        block_tables_mx = [mx.array(bt, dtype=mx.int32) for bt in ctx.block_tables]

        outputs = []
        for req_idx, ctx_len in enumerate(ctx.context_lens):
            req_start = ctx.cu_seqlens[req_idx]
            req_end = ctx.cu_seqlens[req_idx + 1]
            num_new = req_end - req_start
            past_len = ctx_len - num_new

            n_blocks = math.ceil(ctx_len / latent_cache.block_size)
            blocks = block_tables_mx[req_idx][:n_blocks]
            all_latent = latent_cache.latent_caches[layer_idx][blocks].reshape(
                -1, latent_cache.latent_dim
            )[:ctx_len]

            all_kv_norm = all_latent[:, : inner.kv_lora_rank]
            all_k_pe = all_latent[:, inner.kv_lora_rank :]

            rq_nope = q_nope[:, :, req_start:req_end, :]
            rq_pe = q_pe[:, :, req_start:req_end, :]

            k_pe_r = all_k_pe.reshape(1, 1, ctx_len, inner.qk_rope_head_dim)
            causal_mask = self._causal_valid_mask(
                num_new=num_new, ctx_len=ctx_len, past_len=past_len
            )

            out = self._apply_mla_attention(
                rq_nope=rq_nope,
                rq_pe=rq_pe,
                all_kv_norm=all_kv_norm,
                k_pe=k_pe_r,
                causal_mask=causal_mask,
            )

            out = out.transpose(0, 2, 1, 3).reshape(1, num_new, -1)
            outputs.append(out)

        return mx.concatenate(outputs, axis=1) if len(outputs) > 1 else outputs[0]

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)
        if not ctx.block_tables:
            raise RuntimeError(
                "MLAPagedAttentionWrapper called with empty block_tables"
            )

        inner = self._inner
        layer_idx: int = self._mla_layer_idx
        latent_cache: MLAPagedLatentCache = self._mla_latent_cache

        _, seq_len, _ = x.shape  # B=1, seq_len = total new tokens across all requests

        # Query path — q_lora_rank is None for models without query compression
        if inner.q_lora_rank is None:
            q = inner.q_proj(x)
        else:
            q = inner.q_b_proj(inner.q_a_layernorm(inner.q_a_proj(x)))
        q = q.reshape(1, seq_len, inner.num_heads, inner.q_head_dim).transpose(
            0, 2, 1, 3
        )
        q_nope, q_pe = mx.split(q, [inner.qk_nope_head_dim], axis=-1)

        # KV path — kv_a_proj produces both the lora latent and the rope key in one shot
        kv_out = inner.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe_raw = mx.split(kv_out, [inner.kv_lora_rank], axis=-1)
        kv_norm = inner.kv_a_layernorm(compressed_kv)  # what ends up in the cache
        k_pe = k_pe_raw.reshape(1, seq_len, 1, inner.qk_rope_head_dim).transpose(
            0, 2, 1, 3
        )

        # RoPE is applied per request segment so each request starts at its own position
        q_pe, k_pe = apply_packed_rope(
            inner,
            q_pe,
            k_pe,
            ctx.cu_seqlens,
            offsets=ctx.offsets or None,
        )

        # Concatenate kv_norm and the roped k_pe into a single per-token latent,
        # then scatter-write it into the cache at the scheduler-assigned slots.
        # MLX arrays are functional, so the indexed update returns a new array
        # that we explicitly reassign back into the cache list.
        k_pe_seq = k_pe.transpose(0, 2, 1, 3).reshape(
            1, seq_len, inner.qk_rope_head_dim
        )
        latent_new = mx.concatenate([kv_norm, k_pe_seq], axis=-1)
        latent_flat = latent_new.reshape(seq_len, latent_cache.latent_dim).astype(
            latent_cache.dtype
        )

        flat = latent_cache.latent_caches[layer_idx].reshape(
            -1, latent_cache.latent_dim
        )
        flat[mx.array(ctx.slot_mapping, dtype=mx.int64)] = latent_flat
        latent_cache.latent_caches[layer_idx] = flat.reshape(
            latent_cache.num_blocks, latent_cache.block_size, latent_cache.latent_dim
        )

        if self._can_use_kernel(inner, latent_cache, ctx):
            final = self._kernel_fast_path(
                inner, latent_cache, layer_idx, q_nope, q_pe, ctx, seq_len
            )
        else:
            final = self._slow_path_per_request(
                inner, latent_cache, layer_idx, q_nope, q_pe, ctx
            )
        return inner.o_proj(final)


class MLAPagedAttentionBackend:
    """Paged attention backend for MLA models.

    Implements the PagedAttentionBackend protocol. Uses MLX-native
    scatter/gather (no vendored C++/Metal kernel) because MLA latents
    do not fit the standard (num_heads, head_dim) kernel layout.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        latent_dim: int,
        block_size: int,
        dtype: mx.Dtype,
    ) -> None:
        self._num_layers = num_layers
        self._latent_dim = latent_dim
        self._block_size = block_size
        self._dtype = dtype
        self._cache: MLAPagedLatentCache | None = None

    def _require_initialized(self, caller: str) -> MLAPagedLatentCache:
        if self._cache is None:
            raise RuntimeError(f"{caller}() called before initialize()")
        return self._cache

    def initialize(self, num_blocks: int) -> None:
        self._cache = MLAPagedLatentCache(
            num_layers=self._num_layers,
            latent_dim=self._latent_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
        )

    def patch_model(self, model: Any) -> int:
        cache = self._require_initialized("patch_model")
        return self._patch_model(model, cache)

    def _patch_model(self, model: Any, latent_cache: MLAPagedLatentCache) -> int:
        patched = 0

        for layer_idx, layer in enumerate(find_layers(model)):
            attn_attr = find_attn_attr(layer)
            if attn_attr is None:
                continue

            attn = getattr(layer, attn_attr)
            if isinstance(attn, MLAPagedAttentionWrapper):
                # Already patched — refresh cache reference (e.g. after re-initialisation)
                object.__setattr__(attn, "_mla_latent_cache", latent_cache)
                patched += 1
                continue

            setattr(
                layer,
                attn_attr,
                MLAPagedAttentionWrapper(attn, layer_idx, latent_cache),
            )
            patched += 1

        return patched

    def warm_up(self) -> None:
        cache = self._require_initialized("warm_up")
        if not envs.VLLM_METAL_MLA_KERNEL:
            # Default path: MLX-native attention only — MLX ops JIT on
            # first use, no Metal shader warm-up needed.
            logger.info(
                "MLA paged attention (MLX SDPA default): skipping Metal kernel "
                "warm-up. Set VLLM_METAL_MLA_KERNEL=1 to enable the experimental "
                "kernel path (RFC #360)."
            )
            return
        self._warm_kernel(cache)

    def _warm_kernel(self, cache: MLAPagedLatentCache) -> None:
        """JIT-compile the MLA Metal decode kernels via tiny synthetic
        dispatches. Without this, the first live decode pays the
        ``get_ops()`` + Metal compile cost on the hot path and any shader
        / instantiation failure surfaces well after startup. The kernel is
        only instantiated for the production absorbed-MLA shape
        (``kv_lora_rank=512``, ``qk_rope_head_dim=64``, fp16 / bf16,
        ``block_size`` ∈ {16, 32}); other caches are unreachable through
        ``_can_use_kernel`` so they don't need warming.

        Coverage matrix (must match what the dispatcher can route to in
        production):

        - Single-pass: ``HEADS_PER_TG`` ∈ {1, 2} — picked by
          ``_pick_heads_per_tg`` based on ``num_heads`` and batch.
        - 2pass: ``(HEADS_PER_TG, PARTITION_SIZE)`` ∈
          {(8, 64), (8, 128), (32, 64), (32, 128)}. The dispatcher
          (``paged_ops.cpp``) picks ``HEADS_PER_TG = 32`` when
          ``num_heads ≥ 32`` else 8; ``_pick_mla_decode_2pass_partition``
          picks ``PARTITION_SIZE = 128`` when ``max_ctx > 1024`` else 64.
          Warm both axes so first live H≥32 or ctx>1024 decode is hot.
        - FA wide: ``block_size == 16`` only; the routing predicate gates
          on bs=16 so a bs=32 cache skips this branch.
        - pr_mma: warmed at the cache's ``block_size`` (∈ {16, 32}) with
          ``num_heads == 32`` (HPT match) and ``ctx > 4096`` to JIT the
          combined main+reduce dispatch ahead of the first live H=128
          B≥4 long-ctx decode.
        """
        if (
            self._latent_dim != 576  # 512 + 64
            or self._dtype not in (mx.float16, mx.bfloat16)
            or self._block_size not in (16, 32)
        ):
            logger.info(
                "MLA Metal kernel: skipping warm-up — cache "
                "(latent_dim=%d, dtype=%s, block_size=%d) doesn't match the "
                "kernel instantiation (576, fp16/bf16, 16/32)",
                self._latent_dim,
                self._dtype,
                self._block_size,
            )
            return

        from vllm_metal.metal import (
            metal_mla_paged_attention,
            metal_mla_paged_attention_decode_2pass,
            metal_mla_paged_attention_decode_fa,
            metal_mla_paged_attention_decode_pr_mma_primitive,
        )

        kv_lora_rank = 512
        qk_rope_head_dim = 64
        # ctx threshold > 1024 trips the ps128 branch in
        # _pick_mla_decode_2pass_partition; pick the smallest value that
        # does so to keep the synthetic cache tiny.
        long_ctx = 1025
        long_blocks = math.ceil(long_ctx / self._block_size)
        # Synthetic latent cache large enough for the long-ctx warm-up
        # dispatches. The model's own cache may have been initialised
        # with too few blocks (or for a test, with just a handful), so we
        # allocate our own — only used during warm-up, freed after.
        synth_cache = mx.zeros(
            (long_blocks, self._block_size, self._latent_dim),
            dtype=self._dtype,
        )

        def _run(num_heads: int, ctx_len: int, fn, **extra) -> None:
            n_blocks = math.ceil(ctx_len / self._block_size)
            q_nope = mx.zeros((1, num_heads, kv_lora_rank), dtype=self._dtype)
            q_pe = mx.zeros((1, num_heads, qk_rope_head_dim), dtype=self._dtype)
            out = mx.zeros((1, num_heads, kv_lora_rank), dtype=self._dtype)
            block_tables = mx.array([list(range(n_blocks))], dtype=mx.int32)
            context_lens = mx.array([ctx_len], dtype=mx.uint32)
            cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
            fn(
                q_nope=q_nope,
                q_pe=q_pe,
                latent_cache=synth_cache,
                out=out,
                block_tables=block_tables,
                context_lens=context_lens,
                cu_seqlens_q=cu_seqlens_q,
                scale=1.0,
                **extra,
            )

        # Single-pass: G ∈ {1, 2}. Shape doesn't influence JIT beyond the
        # already-fixed kv_lora_rank / qk_rope_head_dim / block_size /
        # dtype, so the smallest layout is enough.
        short_ctx = self._block_size
        _run(
            num_heads=2, ctx_len=short_ctx, fn=metal_mla_paged_attention, heads_per_tg=1
        )
        _run(
            num_heads=2, ctx_len=short_ctx, fn=metal_mla_paged_attention, heads_per_tg=2
        )

        # 2pass: (HEADS_PER_TG, PARTITION_SIZE) grid via (num_heads, max_ctx).
        for num_heads in (2, 32):  # → hpt8, hpt32
            for ctx_len in (short_ctx, long_ctx):  # → ps64, ps128
                _run(
                    num_heads=num_heads,
                    ctx_len=ctx_len,
                    fn=metal_mla_paged_attention_decode_2pass,
                )

        # FA wide. FA's kernel template requires num_heads % 8 == 0 and
        # block_size=16 only; skip warming if block_size doesn't match.
        # _pick_fa_variant also gates on block_size=16 so a bs=32 cache
        # falls through to 2pass / single-pass at routing time. Wide
        # (BK=64/WN=8) is the production default; narrow kept for
        # benchmarking only and not warmed.
        if self._block_size == 16:
            _run(
                num_heads=8,
                ctx_len=short_ctx,
                fn=metal_mla_paged_attention_decode_fa,
                use_wide=True,
            )

        # pr_mma (per-request MMA paged decode). Routing fires at
        # num_heads == 128 ∧ batch_size ≥ 4 ∧ max_ctx > 4096 ∧
        # block_size ∈ {16, 32}. Warm one cell per cache block_size with
        # H=32 (HPT match) and ctx just above the 4096 threshold; this
        # JITs the combined main+reduce dispatch and the bs-specific
        # specialisation. Block tables are shared across the synthetic
        # B=4 sequences so the synth cache stays small.
        pr_ctx = 4097
        pr_blocks = math.ceil(pr_ctx / self._block_size)
        if pr_blocks > synth_cache.shape[0]:
            synth_cache = mx.zeros(
                (pr_blocks, self._block_size, self._latent_dim),
                dtype=self._dtype,
            )
        pr_num_heads = 32
        pr_batch = 4
        q_combined = mx.zeros(
            (pr_batch, pr_num_heads, kv_lora_rank + qk_rope_head_dim),
            dtype=self._dtype,
        )
        pr_block_tables = mx.array([list(range(pr_blocks))] * pr_batch, dtype=mx.int32)
        pr_context_lens = mx.array([pr_ctx] * pr_batch, dtype=mx.uint32)
        pr_cu_seqlens_q = mx.array(list(range(pr_batch + 1)), dtype=mx.int32)
        pr_out = metal_mla_paged_attention_decode_pr_mma_primitive(
            q_combined=q_combined,
            latent_cache=synth_cache,
            block_tables=pr_block_tables,
            context_lens=pr_context_lens,
            cu_seqlens_q=pr_cu_seqlens_q,
            scale=1.0,
        )
        mx.eval(pr_out)

        mx.synchronize()
        logger.info(
            "MLA Metal kernel: warm-up complete (single-pass G ∈ {1, 2}; "
            "2pass (hpt, ps) ∈ {(8, 64), (8, 128), (32, 64), (32, 128)}; "
            "FA wide; pr_mma at bs=%d)",
            self._block_size,
        )

    def num_blocks(self) -> int:
        return self._require_initialized("num_blocks").num_blocks
