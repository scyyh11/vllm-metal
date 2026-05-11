# SPDX-License-Identifier: Apache-2.0
"""Native paged-attention Metal kernels dispatched through MLX.

Usage::

    from vllm_metal.metal import get_ops
    ops = get_ops()
    ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    ops.paged_attention_v1(out, query, key_cache, value_cache, ...)
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
from pathlib import Path
from types import ModuleType

from vllm_metal.metal.constants import PARTITION_SIZE, PARTITION_THRESHOLD

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_KERNELS_DIR = _THIS_DIR / "kernels_v1"
_KERNELS_V2_DIR = _THIS_DIR / "kernels_v2"

# Cached after first get_ops() call.  The Metal shaders are JIT-compiled once
# and held in MLX's library cache for the lifetime of the process.  Editing
# .metal source files requires restarting the Python interpreter to pick up
# changes (the .cpp extension itself is rebuilt automatically by build.py when
# paged_ops.cpp is newer than the .so).
_ops_module: ModuleType | None = None

# The MLA paged-attention shader set is loaded lazily on the first MLA
# entrypoint call (see _ensure_mla_library). Initialising it inside
# get_ops() would compile experimental MLA Metal source for every
# paged-attention user, so non-MLA models pay the compile cost and a
# compile error in MLA would block unrelated paged-attention paths.
_mla_library_initialized: bool = False


def _read_metal_source(path: Path) -> str:
    """Read a .metal file and strip local #include directives."""
    text = path.read_text()
    # Remove #include "..." for our vendored files (keep <metal_stdlib> etc.)
    text = re.sub(r'#include\s+"[^"]*"', "", text)
    return text


def _read_v2_metal_source(filename: str) -> str:
    """Read a kernels_v2 .metal source file."""
    return _read_metal_source(_KERNELS_V2_DIR / filename)


def _build_reshape_cache_source() -> str:
    """Concatenate float8 + utils + reshape_and_cache into a single source."""
    parts = [
        _read_metal_source(_KERNELS_DIR / "float8.metal"),
        _read_metal_source(_KERNELS_DIR / "utils.metal"),
        _read_metal_source(_KERNELS_DIR / "reshape_and_cache.metal"),
    ]
    return "\n".join(parts)


def _build_paged_attention_source() -> str:
    """Concatenate float8 + utils + paged_attention into a single source."""
    parts = [
        f"#define VLLM_METAL_PARTITION_SIZE {PARTITION_SIZE}",
        _read_metal_source(_KERNELS_DIR / "float8.metal"),
        _read_metal_source(_KERNELS_DIR / "utils.metal"),
        _read_metal_source(_KERNELS_DIR / "pagedattention.metal"),
    ]
    return "\n".join(parts)


def _build_v2_paged_attention_source() -> str:
    """Concatenate float8 + utils + turboquant + v2 paged_attention (online softmax)."""
    parts = [
        f"#define VLLM_METAL_PARTITION_SIZE {PARTITION_SIZE}",
        _read_metal_source(_KERNELS_V2_DIR / "float8.metal"),
        _read_metal_source(_KERNELS_V2_DIR / "utils.metal"),
        _read_metal_source(_KERNELS_V2_DIR / "turboquant.metal"),
        _read_metal_source(_KERNELS_V2_DIR / "pagedattention.metal"),
    ]
    return "\n".join(parts)


def _build_gdn_source() -> str:
    """GDN linear attention kernel source."""
    parts = [
        _read_metal_source(_KERNELS_V2_DIR / "utils.metal"),
        _read_metal_source(_KERNELS_V2_DIR / "gdn_linear_attention.metal"),
    ]
    return "\n".join(parts)


def _build_mla_paged_attention_source() -> str:
    """Concatenate utils + mla into a single source for the MLA library."""
    parts = [
        _read_metal_source(_KERNELS_V2_DIR / "utils.metal"),
        _read_metal_source(_KERNELS_V2_DIR / "mla.metal"),
    ]
    return "\n".join(parts)


def metal_unified_attention(
    q,  # [total_q_tokens, num_q_heads, head_size]
    k,  # [num_blocks, block_size, num_kv_heads, head_size]
    v,  # [num_blocks, block_size, num_kv_heads, head_size]
    out,  # [total_q_tokens, num_q_heads, head_size]
    cu_seqlens_q,  # [num_seqs + 1], int32
    seqused_k,  # [num_seqs], int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    block_table,  # [num_seqs, max_blocks_per_seq], int32
    softcap: float,
) -> None:
    """Unified varlen paged attention for Metal.

    Supports variable-length queries (prefill + decode) with online softmax,
    paged KV cache, causal masking, sliding window, and soft capping.

    Grid: one threadgroup per (head, query_token). Each threadgroup uses
    binary search on cu_seqlens_q to find its sequence and computes causal
    attention against the paged KV cache.
    """
    assert causal, "Only causal attention is supported"
    import mlx.core as mx

    # Extract dimensions from cache shape
    # k shape: [num_blocks, block_size, num_kv_heads, head_size]
    num_kv_heads = k.shape[2]
    block_size = k.shape[1]

    # Convert window_size tuple to a single sliding_window int.
    # window_size = (left, right) where left = sw-1, right = 0 for causal.
    # sliding_window = left + 1 = total window size. -1 = disabled.
    if window_size == (-1, -1):
        sliding_window = -1
    else:
        sliding_window = window_size[0] + 1

    ops = get_ops()

    # Ensure all inputs are evaluated before raw Metal dispatch
    mx.eval(out, q, k, v, block_table, seqused_k, cu_seqlens_q)
    max_num_partitions = max(1, (max_seqlen_k + PARTITION_SIZE - 1) // PARTITION_SIZE)
    use_partitioning = (
        PARTITION_SIZE % block_size == 0
        and max_seqlen_q == 1
        and max_seqlen_k >= PARTITION_THRESHOLD
        and max_num_partitions > 1
    )

    if use_partitioning:
        exp_sums = mx.zeros(
            (q.shape[0], q.shape[1], max_num_partitions), dtype=mx.float32
        )
        max_logits = mx.zeros(
            (q.shape[0], q.shape[1], max_num_partitions), dtype=mx.float32
        )
        tmp_out = mx.zeros(
            (q.shape[0], q.shape[1], max_num_partitions, q.shape[2]),
            dtype=q.dtype,
        )
        mx.eval(exp_sums, max_logits, tmp_out)
        ops.paged_attention_v2_online_partitioned(
            out,
            q,
            k,
            v,
            num_kv_heads,
            softmax_scale,
            softcap,
            block_table,
            seqused_k,
            cu_seqlens_q,
            block_size,
            max_seqlen_k,
            sliding_window,
            exp_sums,
            max_logits,
            tmp_out,
        )
        mx.synchronize()
    else:
        ops.paged_attention_v2_online(
            out,
            q,
            k,
            v,
            num_kv_heads,
            softmax_scale,
            softcap,
            block_table,
            seqused_k,
            cu_seqlens_q,
            block_size,
            max_seqlen_k,
            sliding_window,
        )
        mx.synchronize()


def metal_mla_paged_attention(
    q_nope,  # [total_q_tokens, num_heads, kv_lora_rank]
    q_pe,  # [total_q_tokens, num_heads, qk_rope_head_dim]
    latent_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    out,  # [total_q_tokens, num_heads, kv_lora_rank]
    block_tables,  # [num_seqs, max_blocks_per_seq], int32
    context_lens,  # [num_seqs], uint32
    cu_seqlens_q,  # [num_seqs + 1], int32
    scale: float,
    heads_per_tg: int = 1,
) -> None:
    """Paged Multi-head Latent Attention (RFC #360).

    Phase 1 step 8 (multi-block decode): the kernel iterates the per-sequence
    block table with NUM_WARPS-strided online softmax and a cross-warp merge,
    so any ``ctx_len`` that fits into the allocated block_tables row is fine.
    Decode-only is still required (one query token per sequence) — varlen
    prefill lands in P2.

    Q is expected to be already projected through ``embed_q`` (so q_nope is
    in kv_lora_rank space) and ``q_pe`` is RoPE-applied. The caller is
    responsible for ``unembed_out`` on the result to recover v_head_dim.

    ``heads_per_tg`` (G) controls cross-head KV amortization: each
    threadgroup processes ``G`` consecutive query heads sharing the same
    latent KV. Total dispatched threadgroups drop from ``B×H`` to
    ``B×ceil(H/G)``, so total KV bandwidth is amortized G×. ``num_heads``
    must be divisible by G. Currently instantiated for G ∈ {1, 4}; G=1 is
    the existing single-head-per-TG kernel.
    """
    import mlx.core as mx
    import numpy as np

    # Shape contract — fail fast at the Python boundary so the C++ error
    # path isn't the only line of defence.
    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]

    mx.eval(out, q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q)

    # P1 decode-only guard: q_token_idx == seq_idx is hard-wired in the
    # kernel. Reading cu_seqlens_q back to host is cheap (tens of int32 per
    # layer per step) and the guard goes away when P2 makes the kernel
    # cu_seqlens_q-aware via find_seq_idx.
    cu_q = np.asarray(cu_seqlens_q)
    deltas = np.diff(cu_q)
    if np.any(deltas != 1):
        bad = int(np.argmax(deltas != 1))
        raise NotImplementedError(
            "MLA kernel (P1) supports decode only — one query token per "
            f"sequence. Got request {bad} with {int(deltas[bad])} query "
            "tokens. Multi-token prefill / varlen support lands in P2."
        )

    # block_tables row-width guard. The kernel walks block_table_row[0..
    # num_context_blocks-1] for each sequence; if the caller-allocated row
    # is too narrow we'd silently read into the next sequence's row or off
    # the end of the buffer. Caller-side capacity bug (ValueError, not
    # NotImplementedError — this isn't a feature gap).
    ctx = np.asarray(context_lens)
    max_blocks_per_seq = int(block_tables.shape[1])
    required_blocks = (ctx + block_size - 1) // block_size
    if np.any(required_blocks > max_blocks_per_seq):
        bad = int(np.argmax(required_blocks > max_blocks_per_seq))
        raise ValueError(
            f"MLA: block_tables row width ({max_blocks_per_seq}) too small "
            f"for request {bad}: ctx_len={int(ctx[bad])} requires "
            f"{int(required_blocks[bad])} blocks at block_size={block_size}."
        )

    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention(
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_size,
        scale,
        heads_per_tg,
    )
    mx.synchronize()


def metal_mla_paged_attention_primitive(
    q_nope,  # [total_q_tokens, num_heads, kv_lora_rank]
    q_pe,  # [total_q_tokens, num_heads, qk_rope_head_dim]
    latent_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    block_tables,  # [num_seqs, max_blocks_per_seq], int32
    context_lens,  # [num_seqs], uint32
    cu_seqlens_q,  # [num_seqs + 1], int32
    scale: float,
    heads_per_tg: int = 1,
):
    """Primitive variant of :func:`metal_mla_paged_attention` — returns
    a lazy ``mx.array`` whose evaluation triggers the kernel dispatch.

    Same shape contract and decode-only guard as the eager binding; the
    only difference is the call participates in MLX's lazy graph
    instead of forcing a ``mx.eval`` boundary inside this entry. That
    saves ~200 μs at small workloads (B=1, H≤64) where MLX dispatch
    overhead is a meaningful fraction of total wrapper time.

    The wrapper switches to this variant when the router picks the
    single-pass kernel; the eager binding is kept for bench tools and
    correctness tests that need explicit in-place semantics.
    """
    import numpy as np

    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]

    # Decode-only guard mirrors the eager entry. Reads cu_seqlens_q to
    # host (cheap — a few int32 per layer per step). The guard is on
    # the Python side because the kernel hard-wires q_token_idx ==
    # seq_idx; lifting this is P2.
    cu_q = np.asarray(cu_seqlens_q)
    deltas = np.diff(cu_q)
    if np.any(deltas != 1):
        bad = int(np.argmax(deltas != 1))
        raise NotImplementedError(
            "MLA kernel (primitive) supports decode only — one query "
            f"token per sequence. Got request {bad} with "
            f"{int(deltas[bad])} query tokens."
        )

    ctx = np.asarray(context_lens)
    max_blocks_per_seq = int(block_tables.shape[1])
    required_blocks = (ctx + block_size - 1) // block_size
    if np.any(required_blocks > max_blocks_per_seq):
        bad = int(np.argmax(required_blocks > max_blocks_per_seq))
        raise ValueError(
            f"MLA: block_tables row width ({max_blocks_per_seq}) too small "
            f"for request {bad}: ctx_len={int(ctx[bad])} requires "
            f"{int(required_blocks[bad])} blocks at block_size={block_size}."
        )

    import mlx.core as mx

    total_q_tokens = int(q_nope.shape[0])
    num_heads = int(q_nope.shape[1])
    kv_lora_rank = int(q_nope.shape[2])
    # ``mx.zeros`` here is lazy — the C++ side replaces ``out``'s
    # descriptor with the Primitive output before the zeros ever
    # evaluate, so the memset is never scheduled.
    out = mx.zeros((total_q_tokens, num_heads, kv_lora_rank), dtype=q_nope.dtype)

    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_primitive(
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_size,
        scale,
        heads_per_tg,
        out,
    )
    return out


# Hard-coded for now: matches the only instantiated reduce kernel
# (see kernels_v2/mla.metal). If we add more PARTITION_SIZE specializations
# later, this becomes a parameter.
MLA_PARTITION_SIZE = 512


def metal_mla_paged_attention_partitioned(
    q_nope,  # [total_q_tokens, num_heads, kv_lora_rank]
    q_pe,  # [total_q_tokens, num_heads, qk_rope_head_dim]
    latent_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    out,  # [total_q_tokens, num_heads, kv_lora_rank]
    block_tables,  # [num_seqs, max_blocks_per_seq], int32
    context_lens,  # [num_seqs], uint32
    cu_seqlens_q,  # [num_seqs + 1], int32
    scale: float,
    heads_per_tg: int = 1,
) -> None:
    """Paged MLA with split-K + reduce (RFC #360 Phase 3 — for long contexts /
    low-batch decode where the single-pass kernel under-fills the GPU).

    Same shape contract and decode-only constraints as
    ``metal_mla_paged_attention``. Internally allocates per-partition
    ``exp_sums`` / ``max_logits`` / ``tmp_out`` scratch and dispatches the
    main kernel with ``PARTITION_SIZE=512`` followed by the reduce kernel.

    Caller is responsible for the partitioning routing decision (e.g., based
    on max ctx_len and total threadgroup count); this function unconditionally
    runs the partitioned path.
    """
    import mlx.core as mx
    import numpy as np

    # Shape contract — identical to the non-partitioned entry.
    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]
    if MLA_PARTITION_SIZE % block_size != 0:
        raise ValueError(
            f"MLA partitioned: PARTITION_SIZE ({MLA_PARTITION_SIZE}) must be "
            f"divisible by block_size ({block_size})."
        )

    mx.eval(out, q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q)

    cu_q = np.asarray(cu_seqlens_q)
    deltas = np.diff(cu_q)
    if np.any(deltas != 1):
        bad = int(np.argmax(deltas != 1))
        raise NotImplementedError(
            "MLA partitioned kernel (P1) supports decode only — one query "
            f"token per sequence. Got request {bad} with {int(deltas[bad])} "
            "query tokens."
        )

    ctx = np.asarray(context_lens)
    max_blocks_per_seq = int(block_tables.shape[1])
    required_blocks = (ctx + block_size - 1) // block_size
    if np.any(required_blocks > max_blocks_per_seq):
        bad = int(np.argmax(required_blocks > max_blocks_per_seq))
        raise ValueError(
            f"MLA partitioned: block_tables row width ({max_blocks_per_seq}) "
            f"too small for request {bad}: ctx_len={int(ctx[bad])} requires "
            f"{int(required_blocks[bad])} blocks at block_size={block_size}."
        )

    total_q_tokens = int(q_nope.shape[0])
    num_heads = int(q_nope.shape[1])
    kv_lora_rank = int(q_nope.shape[2])
    max_ctx = int(ctx.max())
    max_num_partitions = max(
        1, (max_ctx + MLA_PARTITION_SIZE - 1) // MLA_PARTITION_SIZE
    )

    # Scratch buffers. Zero-initialized so partitions that return early (no
    # blocks to process for their seq) leave a no-op contribution.
    exp_sums = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions), dtype=mx.float32
    )
    max_logits = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions), dtype=mx.float32
    )
    tmp_out = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions, kv_lora_rank),
        dtype=q_nope.dtype,
    )
    mx.eval(exp_sums, max_logits, tmp_out)

    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_partitioned(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_size,
        scale,
        MLA_PARTITION_SIZE,
        max_num_partitions,
        heads_per_tg,
    )
    mx.synchronize()


# Decode-2pass kernel partition sizes (mirrored in mla.metal as
# instantiate_mla_2pass + the matching reduce specializations).
# 256 is kept as a bench knob between 128 and 512; auto-pick still
# returns {64, 128}.
_MLA_DECODE_2PASS_SIZES = (64, 128, 256, 512)


def _pick_mla_decode_2pass_partition(max_ctx: int) -> int:
    """Pick PARTITION_SIZE for the 2pass decode kernel.

    Smaller partition → more TGs → better GPU fill on small ctx. Larger
    partition → less reduce overhead at long ctx. Mirrors MLX's choice in
    `mlx/backend/metal/scaled_dot_product_attention.cpp::sdpa_vector_2pass`
    (devc=='s' branch): 64 by default, 128 once ctx > 1024."""
    if max_ctx <= 1024:
        return 64
    return 128


def metal_mla_paged_attention_decode_2pass(
    q_nope,  # [total_q_tokens, num_heads, kv_lora_rank]
    q_pe,  # [total_q_tokens, num_heads, qk_rope_head_dim]
    latent_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    out,  # [total_q_tokens, num_heads, kv_lora_rank]
    block_tables,  # [num_seqs, max_blocks_per_seq], int32
    context_lens,  # [num_seqs], uint32
    cu_seqlens_q,  # [num_seqs + 1], int32  (only used for decode-only validation)
    scale: float,
    partition_size: int | None = None,
) -> None:
    """MLX sdpa_vector_2pass-style cross-head amortization for absorbed MLA
    decode (RFC #360, follow-up to the G-batched single-pass kernel).

    Each TG handles one (seq, ctx-partition) pair with 32*num_heads threads
    arranged as 32-lane × num_heads-head simdgroups. All heads in the TG
    read the same K cache tokens — the L1/L2 cache serves the H-1 repeats
    so total KV bandwidth is amortized H× across the whole launch.

    Same shape contract and decode-only constraints as
    `metal_mla_paged_attention`. Internally allocates `exp_sums` /
    `max_logits` / `tmp_out` scratch and dispatches the main kernel
    followed by the existing reduce kernel.

    `partition_size` defaults to a heuristic based on max ctx_len; pass an
    explicit value to override.
    """
    import mlx.core as mx
    import numpy as np

    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]

    mx.eval(out, q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q)

    cu_q = np.asarray(cu_seqlens_q)
    deltas = np.diff(cu_q)
    if np.any(deltas != 1):
        bad = int(np.argmax(deltas != 1))
        raise NotImplementedError(
            "MLA decode-2pass kernel supports decode only — one query token "
            f"per sequence. Got request {bad} with {int(deltas[bad])} query "
            "tokens."
        )

    ctx = np.asarray(context_lens)
    max_blocks_per_seq = int(block_tables.shape[1])
    required_blocks = (ctx + block_size - 1) // block_size
    if np.any(required_blocks > max_blocks_per_seq):
        bad = int(np.argmax(required_blocks > max_blocks_per_seq))
        raise ValueError(
            f"MLA decode-2pass: block_tables row width ({max_blocks_per_seq}) "
            f"too small for request {bad}: ctx_len={int(ctx[bad])} requires "
            f"{int(required_blocks[bad])} blocks at block_size={block_size}."
        )

    total_q_tokens = int(q_nope.shape[0])
    num_heads = int(q_nope.shape[1])
    kv_lora_rank = int(q_nope.shape[2])
    max_ctx = int(ctx.max())

    if partition_size is None:
        partition_size = _pick_mla_decode_2pass_partition(max_ctx)
    if partition_size not in _MLA_DECODE_2PASS_SIZES:
        raise ValueError(
            f"MLA decode-2pass: partition_size must be in "
            f"{_MLA_DECODE_2PASS_SIZES}; got {partition_size}"
        )
    if partition_size % block_size != 0:
        raise ValueError(
            f"MLA decode-2pass: partition_size ({partition_size}) must be "
            f"divisible by block_size ({block_size})."
        )

    max_num_partitions = max(1, (max_ctx + partition_size - 1) // partition_size)

    exp_sums = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions), dtype=mx.float32
    )
    max_logits = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions), dtype=mx.float32
    )
    tmp_out = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions, kv_lora_rank),
        dtype=q_nope.dtype,
    )
    mx.eval(exp_sums, max_logits, tmp_out)

    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_decode_2pass(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        block_size,
        scale,
        partition_size,
        max_num_partitions,
    )
    mx.synchronize()


def metal_mla_paged_attention_decode_2pass_main_only(
    q_nope,
    q_pe,
    latent_cache,
    out_exp_sums,  # caller-allocated [T, H, max_partitions] fp32
    out_max_logits,  # caller-allocated [T, H, max_partitions] fp32
    out_tmp_out,  # caller-allocated [T, H, max_partitions, kv_lora_rank] T
    block_tables,
    context_lens,
    cu_seqlens_q,
    scale: float,
    partition_size: int,
    max_num_partitions: int,
) -> None:
    """Bench-only — runs *only* the partitioned main kernel of the 2pass
    decode dispatch. Writes the three scratch arrays; does NOT run the
    reduce kernel.

    Caller pre-allocates the scratch arrays so the per-call cost is
    pure kernel + dispatch + sync. Pair with
    :func:`metal_mla_paged_attention_decode_2pass_reduce_only` to time
    the two phases separately.
    """
    import mlx.core as mx

    block_size = latent_cache.shape[1]
    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_decode_2pass_main_only(
        out_exp_sums,
        out_max_logits,
        out_tmp_out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        block_size,
        scale,
        partition_size,
        max_num_partitions,
    )
    mx.synchronize()


def metal_mla_paged_attention_decode_pr_mma_primitive(
    q_combined,  # [T, H, KVR+PE=576] T (caller concats q_nope + q_pe)
    latent_cache,  # [num_blocks, block_size, KVR+PE] T
    block_tables,  # [num_seqs, max_blocks_per_seq] int32
    context_lens,  # [num_seqs] uint32
    cu_seqlens_q,  # [num_seqs + 1] int32 — host-side decode-only validation
    scale: float,
):
    """Lazy MLX Primitive variant of the per-request MMA paged decode.

    Joins the wrapper's lazy graph rather than forcing an mx.eval boundary.
    Allocates scratch (exp_sums / max_logits / tmp_out) inside the
    primitive's eval_gpu — scratch never crosses the Python boundary.

    Only the narrow target shape is instantiated:
    KVR=512, PE=64, block_size ∈ {16, 32}, partition_size=128, hpt=32,
    wm=4, wn=8 for fp16 + bf16. The wrapper routes here only at H=128,
    B≥4, ctx>4096, bs ∈ {16, 32}.

    Returns a lazy array — wrapper graph eval triggers the actual GPU
    dispatch via the primitive.
    """
    import mlx.core as mx
    import numpy as np

    block_size = latent_cache.shape[1]
    if block_size not in (16, 32):
        raise ValueError(
            f"pr_mma primitive: only block_size ∈ {{16, 32}} instantiated; "
            f"got {block_size}"
        )

    cu_q = np.asarray(cu_seqlens_q)
    deltas = np.diff(cu_q)
    if np.any(deltas != 1):
        bad = int(np.argmax(deltas != 1))
        raise NotImplementedError(
            "pr_mma primitive: decode-only contract — one query token per "
            f"sequence. Got request {bad} with {int(deltas[bad])} query tokens."
        )

    ctx = np.asarray(context_lens)
    max_blocks_per_seq = int(block_tables.shape[1])
    required_blocks = (ctx + block_size - 1) // block_size
    if np.any(required_blocks > max_blocks_per_seq):
        bad = int(np.argmax(required_blocks > max_blocks_per_seq))
        raise ValueError(
            f"pr_mma primitive: block_tables row width ({max_blocks_per_seq}) "
            f"too small for request {bad}: ctx_len={int(ctx[bad])} requires "
            f"{int(required_blocks[bad])} blocks at block_size={block_size}."
        )

    max_ctx = int(ctx.max())
    partition_size = 128  # only ps=128 instantiated
    max_num_partitions = max(1, (max_ctx + partition_size - 1) // partition_size)

    total_q_tokens = int(q_combined.shape[0])
    num_heads = int(q_combined.shape[1])
    # Output shape: [total_q_tokens, num_heads, KV_LORA_RANK=512]
    out = mx.zeros((total_q_tokens, num_heads, 512), dtype=q_combined.dtype)

    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_decode_pr_mma_primitive(
        q_combined,
        latent_cache,
        block_tables,
        context_lens,
        block_size,
        scale,
        partition_size,
        max_num_partitions,
        out,
    )
    return out


def metal_mla_paged_attention_decode_pr_mma_main(
    q_combined,  # [num_seqs, num_heads, KVR+PE=576] T
    latent_cache,
    out_exp_sums,  # [num_seqs, num_heads, max_partitions] fp32
    out_max_logits,  # [num_seqs, num_heads, max_partitions] fp32
    out_tmp_out,  # [num_seqs, num_heads, max_partitions, KVR] T
    block_tables,
    context_lens,
    scale: float,
    partition_size: int,
    max_num_partitions: int,
) -> None:
    """Bench-only — per-request MMA paged main kernel.

    Full main kernel for the per-request MMA decode path. Writes the
    same scratch contract as ``paged_mla_attention_decode_2pass_main_only``;
    the existing reduce kernel
    (:func:`metal_mla_paged_attention_decode_2pass_reduce_only`) merges
    its partials.

    fp16 + bf16 × KVR=512 / PE=64 / bs ∈ {16, 32} / ps=128 / hpt=32 /
    wm=4 / wn=8 instantiated. Production routing uses the lazy
    primitive variant (``metal_mla_paged_attention_decode_pr_mma_primitive``);
    this eager entry exists for the bench harnesses that need to time
    the main kernel alone or run parity vs the 2pass main.
    """
    import mlx.core as mx

    block_size = latent_cache.shape[1]
    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_decode_pr_mma_main(
        out_exp_sums,
        out_max_logits,
        out_tmp_out,
        q_combined,
        latent_cache,
        block_tables,
        context_lens,
        block_size,
        scale,
        partition_size,
        max_num_partitions,
    )
    mx.synchronize()


def metal_mla_paged_attention_decode_2pass_reduce_only(
    out,
    exp_sums,
    max_logits,
    tmp_out,
    context_lens,
    partition_size: int,
    max_num_partitions: int,
) -> None:
    """Bench-only — runs *only* the reduce kernel of the 2pass decode
    dispatch. Reads (exp_sums, max_logits, tmp_out) from a prior
    ``metal_mla_paged_attention_decode_2pass_main_only`` call and
    writes ``out``.
    """
    import mlx.core as mx

    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_decode_2pass_reduce_only(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        context_lens,
        partition_size,
        max_num_partitions,
    )
    mx.synchronize()


def metal_mla_paged_attention_decode_2pass_primitive(
    q_nope,
    q_pe,
    latent_cache,
    block_tables,
    context_lens,
    cu_seqlens_q,
    scale: float,
    partition_size: int | None = None,
):
    """Lazy MLX Primitive variant of :func:`metal_mla_paged_attention_decode_2pass`.

    Returns the output array as a deferred Primitive node so the
    partitioned kernel + reduce kernel call participates in the
    wrapper's lazy graph. Scratch (exp_sums / max_logits / tmp_out)
    is allocated **inside** the primitive's eval_gpu — it never
    crosses the Python boundary, eliminating both the per-call
    ``mx.eval`` sync boundary and the three ``mx.zeros`` scratch
    allocations.

    Wired into the wrapper for routes where partition parallelism wins
    over FA wide. Same decode-only contract as the eager binding.
    """
    import mlx.core as mx
    import numpy as np

    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]

    cu_q = np.asarray(cu_seqlens_q)
    deltas = np.diff(cu_q)
    if np.any(deltas != 1):
        bad = int(np.argmax(deltas != 1))
        raise NotImplementedError(
            "MLA 2pass kernel (primitive) supports decode only — one "
            f"query token per sequence. Got request {bad} with "
            f"{int(deltas[bad])} query tokens."
        )

    ctx = np.asarray(context_lens)
    max_blocks_per_seq = int(block_tables.shape[1])
    required_blocks = (ctx + block_size - 1) // block_size
    if np.any(required_blocks > max_blocks_per_seq):
        bad = int(np.argmax(required_blocks > max_blocks_per_seq))
        raise ValueError(
            f"MLA 2pass kernel: block_tables row width "
            f"({max_blocks_per_seq}) too small for request {bad}: "
            f"ctx_len={int(ctx[bad])} requires "
            f"{int(required_blocks[bad])} blocks at block_size={block_size}."
        )

    total_q_tokens = int(q_nope.shape[0])
    num_heads = int(q_nope.shape[1])
    kv_lora_rank = int(q_nope.shape[2])
    max_ctx = int(ctx.max())

    if partition_size is None:
        partition_size = _pick_mla_decode_2pass_partition(max_ctx)
    if partition_size not in _MLA_DECODE_2PASS_SIZES:
        raise ValueError(
            f"MLA 2pass kernel: partition_size must be in "
            f"{_MLA_DECODE_2PASS_SIZES}; got {partition_size}"
        )

    max_num_partitions = max(1, (max_ctx + partition_size - 1) // partition_size)

    out = mx.zeros((total_q_tokens, num_heads, kv_lora_rank), dtype=q_nope.dtype)

    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_decode_2pass_primitive(
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        block_size,
        scale,
        partition_size,
        max_num_partitions,
        out,
    )
    return out


def metal_mla_paged_attention_decode_fa(
    q_nope,  # [total_q_tokens, num_heads, kv_lora_rank]
    q_pe,  # [total_q_tokens, num_heads, qk_rope_head_dim]
    latent_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    out,  # [total_q_tokens, num_heads, kv_lora_rank]
    block_tables,  # [num_seqs, max_blocks_per_seq], int32
    context_lens,  # [num_seqs], uint32
    cu_seqlens_q,  # [num_seqs + 1], int32 (decode-only validation)
    scale: float,
    use_wide: bool = False,
) -> None:
    """Paged Flash-Attention decode kernel using ``simdgroup_matrix<T, 8, 8>``
    MMAs.

    Full FA pipeline (Q load → simdgroup_matrix QK →
    online softmax merged across BK tiles → SV → normalize) with
    multi-block context. Partial tail (last K-tile with valid_cols <
    BK) is handled inside the kernel.

    Shape and decode-only contract match
    ``metal_mla_paged_attention_decode_2pass``. Additionally
    ``num_heads`` must be a multiple of 8 (one ``simdgroup_matrix`` M
    dim).
    """
    import mlx.core as mx
    import numpy as np

    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]
    num_heads = int(q_nope.shape[1])
    if num_heads % 8 != 0:
        raise ValueError(
            f"MLA FA decode kernel: num_heads ({num_heads}) must be a "
            "multiple of 8 (the simdgroup_matrix M dim)."
        )

    mx.eval(out, q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q)

    cu_q = np.asarray(cu_seqlens_q)
    deltas = np.diff(cu_q)
    if np.any(deltas != 1):
        bad = int(np.argmax(deltas != 1))
        raise NotImplementedError(
            "MLA FA decode kernel supports decode only — one query token per "
            f"sequence. Got request {bad} with {int(deltas[bad])} query "
            "tokens."
        )

    ctx = np.asarray(context_lens)
    max_blocks_per_seq = int(block_tables.shape[1])
    required_blocks = (ctx + block_size - 1) // block_size
    if np.any(required_blocks > max_blocks_per_seq):
        bad = int(np.argmax(required_blocks > max_blocks_per_seq))
        raise ValueError(
            f"MLA FA decode kernel: block_tables row width ({max_blocks_per_seq}) "
            f"too small for request {bad}: ctx_len={int(ctx[bad])} requires "
            f"{int(required_blocks[bad])} blocks at block_size={block_size}."
        )

    if np.any(ctx <= 0):
        bad = int(np.argmax(ctx <= 0))
        raise ValueError(
            "MLA FA decode kernel: every context_len must be positive; got "
            f"ctx_len={int(ctx[bad])} for request {bad}."
        )

    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_decode_fa(
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_size,
        scale,
        use_wide,
    )
    mx.synchronize()


def metal_mla_paged_attention_decode_fa_primitive(
    q_nope,
    q_pe,
    latent_cache,
    block_tables,
    context_lens,
    cu_seqlens_q,
    scale: float,
    use_wide: bool = False,
):
    """Lazy MLX Primitive variant of :func:`metal_mla_paged_attention_decode_fa`.

    Returns the output array as a deferred Primitive node so the
    kernel call participates in the wrapper's lazy graph — saves the
    per-call ``mx.eval`` boundary the eager binding forces. Same
    shape contract and decode-only guard.

    Mirrors the single-pass primitive's design. The wrapper switches
    eligible FA routes to this entry; the eager binding is kept for
    bench tools and parity tests.
    """
    import mlx.core as mx
    import numpy as np

    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]
    num_heads = int(q_nope.shape[1])
    if num_heads % 8 != 0:
        raise ValueError(
            f"MLA FA decode kernel: num_heads ({num_heads}) must be a "
            "multiple of 8 (the simdgroup_matrix M dim)."
        )

    cu_q = np.asarray(cu_seqlens_q)
    deltas = np.diff(cu_q)
    if np.any(deltas != 1):
        bad = int(np.argmax(deltas != 1))
        raise NotImplementedError(
            "MLA FA decode kernel (primitive) supports decode only — one "
            f"query token per sequence. Got request {bad} with "
            f"{int(deltas[bad])} query tokens."
        )

    ctx = np.asarray(context_lens)
    max_blocks_per_seq = int(block_tables.shape[1])
    required_blocks = (ctx + block_size - 1) // block_size
    if np.any(required_blocks > max_blocks_per_seq):
        bad = int(np.argmax(required_blocks > max_blocks_per_seq))
        raise ValueError(
            f"MLA FA decode kernel: block_tables row width "
            f"({max_blocks_per_seq}) too small for request {bad}: "
            f"ctx_len={int(ctx[bad])} requires "
            f"{int(required_blocks[bad])} blocks at block_size={block_size}."
        )

    # The FA kernel early-returns for ctx == 0 without writing the
    # output tile, but the primitive backs `out` with `allocator::malloc`
    # rather than the lazy zero-init descriptor — empty contexts would
    # leave uninitialized data in `out`. Mirror the eager wrapper's
    # ctx > 0 contract here.
    if np.any(ctx <= 0):
        bad = int(np.argmax(ctx <= 0))
        raise ValueError(
            "MLA FA decode kernel (primitive): every context_len must be "
            f"positive; got ctx_len={int(ctx[bad])} for request {bad}."
        )

    total_q_tokens = int(q_nope.shape[0])
    kv_lora_rank = int(q_nope.shape[2])
    out = mx.zeros((total_q_tokens, num_heads, kv_lora_rank), dtype=q_nope.dtype)

    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_decode_fa_primitive(
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_size,
        scale,
        use_wide,
        out,
    )
    return out


def metal_mla_paged_attention_decode_fa_partitioned(
    q_nope,  # [total_q_tokens, num_heads, kv_lora_rank]
    q_pe,  # [total_q_tokens, num_heads, qk_rope_head_dim]
    latent_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    out,  # [total_q_tokens, num_heads, kv_lora_rank]
    block_tables,  # [num_seqs, max_blocks_per_seq], int32
    context_lens,  # [num_seqs], uint32
    cu_seqlens_q,  # [num_seqs + 1], int32 (decode-only validation)
    scale: float,
    partition_size: int | None = None,
    use_wide: bool = False,
) -> None:
    """Split-K + reduce variant of the paged FA decode kernel. Each
    (seq, head_group, partition) threadgroup processes one
    ``partition_size``-token slice of ctx and writes a normalized partial;
    the same reduce kernel used by ``metal_mla_paged_attention_decode_2pass``
    merges across partitions.

    Targets long-ctx decode where the non-partitioned FA is
    bandwidth-bound on the K stream. Same shape + decode-only contract
    as the other FA / 2pass entries.
    """
    import mlx.core as mx
    import numpy as np

    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]
    num_heads = int(q_nope.shape[1])
    if num_heads % 8 != 0:
        raise ValueError(
            f"MLA FA partitioned kernel: num_heads ({num_heads}) must be a "
            "multiple of 8 (the simdgroup_matrix M dim)."
        )

    mx.eval(out, q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q)

    cu_q = np.asarray(cu_seqlens_q)
    deltas = np.diff(cu_q)
    if np.any(deltas != 1):
        bad = int(np.argmax(deltas != 1))
        raise NotImplementedError(
            "MLA FA partitioned kernel supports decode only — one query token "
            f"per sequence. Got request {bad} with {int(deltas[bad])} query "
            "tokens."
        )

    ctx = np.asarray(context_lens)
    max_blocks_per_seq = int(block_tables.shape[1])
    required_blocks = (ctx + block_size - 1) // block_size
    if np.any(required_blocks > max_blocks_per_seq):
        bad = int(np.argmax(required_blocks > max_blocks_per_seq))
        raise ValueError(
            f"MLA FA partitioned: block_tables row width ({max_blocks_per_seq}) "
            f"too small for request {bad}: ctx_len={int(ctx[bad])} requires "
            f"{int(required_blocks[bad])} blocks at block_size={block_size}."
        )

    if np.any(ctx <= 0):
        bad = int(np.argmax(ctx <= 0))
        raise ValueError(
            "MLA FA partitioned: every context_len must be positive; got "
            f"ctx_len={int(ctx[bad])} for request {bad}."
        )

    total_q_tokens = int(q_nope.shape[0])
    kv_lora_rank = int(q_nope.shape[2])
    max_ctx = int(ctx.max())

    # Mirrors _pick_mla_decode_2pass_partition: 64 for ctx ≤ 1024, 128
    # for longer. Caller can override.
    if partition_size is None:
        partition_size = _pick_mla_decode_2pass_partition(max_ctx)
    if partition_size not in _MLA_DECODE_2PASS_SIZES:
        raise ValueError(
            f"MLA FA partitioned: partition_size must be in "
            f"{_MLA_DECODE_2PASS_SIZES}; got {partition_size}"
        )

    max_num_partitions = max(1, (max_ctx + partition_size - 1) // partition_size)

    exp_sums = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions), dtype=mx.float32
    )
    max_logits = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions), dtype=mx.float32
    )
    tmp_out = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions, kv_lora_rank),
        dtype=q_nope.dtype,
    )
    mx.eval(exp_sums, max_logits, tmp_out)

    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_decode_fa_partitioned(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        block_size,
        scale,
        partition_size,
        max_num_partitions,
        use_wide,
    )
    mx.synchronize()


def get_ops() -> ModuleType:
    """JIT-build and import the native paged_ops extension.

    The Metal shader sources are read, pre-processed (includes inlined),
    and passed to the C++ extension which JIT-compiles them via
    ``mlx::core::metal::Device::get_library()``.

    Returns:
        The ``_paged_ops`` module with ``reshape_and_cache()`` and
        ``paged_attention_v1()``.
    """
    global _ops_module
    if _ops_module is not None:
        return _ops_module

    # 1. JIT-build the C++ extension if needed
    from vllm_metal.metal.build import build

    so_path = build()

    # 2. Import the built extension
    spec = importlib.util.spec_from_file_location("_paged_ops", str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load extension from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 3. Initialise Metal libraries (JIT-compile shaders)
    reshape_src = _build_reshape_cache_source()
    paged_attn_src = _build_paged_attention_source()
    mod.init_libraries(reshape_src, paged_attn_src)

    # 4. Initialise v2 library (online softmax kernel)
    v2_src = _build_v2_paged_attention_source()
    mod.init_v2_library(v2_src)

    # 5. Initialise GDN linear attention library
    gdn_src = _build_gdn_source()
    mod.init_gdn_library(gdn_src)

    # The MLA paged-attention library is loaded lazily on first use via
    # _ensure_mla_library so that VLLM_METAL_MLA_KERNEL stays a true
    # opt-in and a compile error in the experimental MLA shader cannot
    # block unrelated paged-attention paths.

    _ops_module = mod
    logger.info("Native paged-attention Metal kernels loaded")
    return mod


def _ensure_mla_library(mod: ModuleType) -> None:
    """JIT-compile the MLA paged-attention shaders on first MLA use.

    The MLA kernel set targets DeepSeek-style absorbed-MLA decode and
    is opt-in via ``VLLM_METAL_MLA_KERNEL``. Initialising it eagerly
    inside :func:`get_ops` would compile the experimental MLA Metal
    source for every paged-attention user, and any compile error in
    that source would then prevent unrelated paged-attention paths
    from loading. MLA entrypoints call this helper immediately after
    :func:`get_ops` to keep the MLA library off the default load path.
    """
    global _mla_library_initialized
    if _mla_library_initialized:
        return
    mla_src = _build_mla_paged_attention_source()
    mod.init_mla_library(mla_src)
    _mla_library_initialized = True
    logger.info("MLA paged-attention Metal kernels loaded")
