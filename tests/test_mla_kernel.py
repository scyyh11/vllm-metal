# SPDX-License-Identifier: Apache-2.0
"""Direct unit tests for the MLA Metal kernel (RFC #360).

Phase 1 step 7 — single-block correctness. The kernel handles
``ctx_len ≤ BLOCK_SIZE`` for one threadgroup per (head, q_token), with
two-pass softmax and no warp merge. Multi-block + online softmax tests
land alongside step 8.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.metal import (
    MLA_PARTITION_SIZE,
    metal_mla_paged_attention,
    metal_mla_paged_attention_decode_2pass,
    metal_mla_paged_attention_decode_2pass_primitive,
    metal_mla_paged_attention_decode_fa,
    metal_mla_paged_attention_decode_fa_partitioned,
    metal_mla_paged_attention_decode_fa_primitive,
    metal_mla_paged_attention_partitioned,
    metal_mla_paged_attention_primitive,
)

# Production shapes — only kv_lora_rank=512, qk_rope_head_dim=64 are
# instantiated in mla.metal.
_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64
_LATENT_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM


def _tolerance(dtype: mx.Dtype) -> tuple[float, float]:
    """Per-dtype (rtol, atol).

    bf16 has 7 mantissa bits — 1 ULP near magnitude 1.0 is ~7.8e-3, so a
    512-dim dot product accumulated through fp32 then cast back to bf16
    routinely shows 1–2 ULP drift vs an einsum-ordered reference. fp16
    has 10 mantissa bits and is much tighter.
    """
    if dtype == mx.bfloat16:
        return 1e-2, 2e-2
    return 1e-3, 1e-3


def _absorbed_mla_dense_reference(
    q_nope: mx.array,  # [num_q, num_heads, kv_lora_rank], fp32
    q_pe: mx.array,  # [num_q, num_heads, qk_rope_head_dim], fp32
    kv_norm: mx.array,  # [num_q, ctx_len, kv_lora_rank], fp32
    k_pe: mx.array,  # [num_q, ctx_len, qk_rope_head_dim], fp32
    scale: float,
) -> mx.array:
    """Pure-MLX absorbed-MLA single attention step.

    All inputs are expected fp32. Returns fp32 output of shape
    [num_q, num_heads, kv_lora_rank].
    """
    nope_scores = mx.einsum("qhd,qtd->qht", q_nope, kv_norm)
    pe_scores = mx.einsum("qhd,qtd->qht", q_pe, k_pe)
    scores = scale * (nope_scores + pe_scores)
    weights = mx.softmax(scores, axis=-1)
    out = mx.einsum("qht,qtd->qhd", weights, kv_norm)
    return out


def _make_inputs(
    *,
    num_seqs: int,
    num_heads: int,
    ctx_len: int,
    block_size: int,
    dtype: mx.Dtype,
    seed: int = 0,
):
    """Build a decode input set that fits ``ctx_len`` worth of valid
    context per sequence. Each sequence is allocated
    ``ceil(ctx_len / block_size)`` blocks; out-of-context slots in the
    last block hold garbage that the kernel must ignore via the ctx_len
    bound. Block tables are contiguous per-seq for simplicity."""
    mx.random.seed(seed)

    n_blocks_per_seq = max(1, (ctx_len + block_size - 1) // block_size)
    num_blocks = n_blocks_per_seq * num_seqs

    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    q_nope = mx.random.normal(shape=(num_seqs, num_heads, _KV_LORA_RANK)).astype(dtype)
    q_pe = mx.random.normal(shape=(num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(
        dtype
    )
    latent_cache = mx.random.normal(shape=(num_blocks, block_size, _LATENT_DIM)).astype(
        dtype
    )

    block_tables_np = np.arange(num_blocks, dtype=np.int32).reshape(
        num_seqs, n_blocks_per_seq
    )
    block_tables = mx.array(block_tables_np)

    context_lens = mx.array([ctx_len] * num_seqs, dtype=mx.uint32)
    cu_seqlens_q = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)

    return (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    )


def _expected_output(
    q_nope: mx.array,
    q_pe: mx.array,
    latent_cache: mx.array,
    block_tables_np: np.ndarray,
    ctx_lens: list[int],
    scale: float,
) -> mx.array:
    """Run the dense reference per-request, gathering the valid-context
    window across however many blocks the request occupies. Casts to
    fp32 for the reference math then back to the original dtype."""
    num_seqs = q_nope.shape[0]
    block_size = latent_cache.shape[1]
    in_dtype = q_nope.dtype

    outs = []
    for i in range(num_seqs):
        ctx_len = ctx_lens[i]
        n_blocks = (ctx_len + block_size - 1) // block_size
        # Concatenate all valid blocks then slice down to ctx_len.
        gathered = mx.concatenate(
            [latent_cache[int(block_tables_np[i, b]), :, :] for b in range(n_blocks)],
            axis=0,
        )[:ctx_len, :].astype(mx.float32)
        kv_norm = gathered[:, :_KV_LORA_RANK].reshape(1, ctx_len, _KV_LORA_RANK)
        k_pe = gathered[:, _KV_LORA_RANK:].reshape(1, ctx_len, _QK_ROPE_HEAD_DIM)
        out_i = _absorbed_mla_dense_reference(
            q_nope[i : i + 1, :, :].astype(mx.float32),
            q_pe[i : i + 1, :, :].astype(mx.float32),
            kv_norm,
            k_pe,
            scale,
        )
        outs.append(out_i)
    return mx.concatenate(outs, axis=0).astype(in_dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_decode_single_block(dtype: mx.Dtype, block_size: int) -> None:
    """ctx_len strictly less than block_size — exercises the masked-tail
    code path: out-of-context slots in the BLOCK_SIZE-wide score buffer
    must not contribute to the softmax."""
    ctx_len = max(1, block_size // 2)  # 8 or 16
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=4,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )
    mx.synchronize()

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len, ctx_len],
        scale=0.125,
    )

    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"single-block mismatch (dtype={dtype}, block_size={block_size}, "
        f"ctx_len={ctx_len}): max_abs_diff={max_abs:.5f}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_decode_full_block(dtype: mx.Dtype, block_size: int) -> None:
    """ctx_len == block_size — every slot in the block is valid context.
    Catches off-by-one errors in the ctx_len boundary check."""
    ctx_len = block_size
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=3,
        num_heads=4,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )
    mx.synchronize()

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len] * 3,
        scale=0.125,
    )

    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"full-block mismatch (dtype={dtype}, block_size={block_size}): "
        f"max_abs_diff={max_abs:.5f}"
    )


def test_decode_single_token_context(dtype: mx.Dtype = mx.float16) -> None:
    """ctx_len = 1 — degenerate case. softmax of a single score is
    always 1.0, output should equal kv_norm[0, :]. Catches obviously-wrong
    softmax / accumulation bugs."""
    block_size = 16
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=1,
        num_heads=2,
        ctx_len=1,
        block_size=block_size,
        dtype=dtype,
    )

    metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )
    mx.synchronize()

    # Expected: out[0, h, :] = kv_norm[block_idx, 0, :KV_LORA_RANK] for all h
    expected_row = latent_cache[
        int(block_tables_np[0, 0]), 0, :_KV_LORA_RANK
    ]  # [KV_LORA_RANK]
    expected = mx.broadcast_to(
        expected_row.reshape(1, 1, _KV_LORA_RANK), out.shape
    ).astype(dtype)

    rtol, atol = _tolerance(dtype)
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_decode_two_blocks_with_partial_tail(dtype: mx.Dtype, block_size: int) -> None:
    """ctx_len = block_size + 1 — minimal multi-block case. Exercises the
    block-table walk (vs. step 7's hardcoded block 0), the partial last
    block, and the cross-warp merge with one warp doing real work and
    seven idle (their state must be the identity element)."""
    ctx_len = block_size + 1
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=4,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )
    mx.synchronize()

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len, ctx_len],
        scale=0.125,
    )
    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"two-block mismatch (dtype={dtype}, block_size={block_size}, "
        f"ctx_len={ctx_len}): max_abs_diff={max_abs:.5f}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_decode_one_block_per_warp(dtype: mx.Dtype, block_size: int) -> None:
    """ctx_len = block_size * NUM_WARPS — every warp processes exactly one
    block, all NUM_WARPS=8 warp states participate in the merge."""
    num_warps = 8
    ctx_len = block_size * num_warps
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=1,
        num_heads=4,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )
    mx.synchronize()

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len],
        scale=0.125,
    )
    rtol, atol = _tolerance(dtype)
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_decode_many_blocks_per_warp(dtype: mx.Dtype) -> None:
    """ctx_len = 4096 — each warp does ctx_len / (block_size * NUM_WARPS)
    iterations, tests the warp-strided loop's iteration count and the
    online softmax accumulator stability over many blocks."""
    block_size = 32
    ctx_len = 4096
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=1,
        num_heads=2,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )
    mx.synchronize()

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len],
        scale=0.125,
    )
    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"long-context mismatch (dtype={dtype}, ctx_len={ctx_len}): "
        f"max_abs_diff={max_abs:.5f}"
    )


def test_decode_mixed_ctx_batch() -> None:
    """Batch with three sequences at different ctx_lens — exercises the
    per-seq context_lens read and ensures the kernel doesn't accidentally
    use one sequence's ctx_len for another (which would happen if the
    block-iteration bound was hoisted out of the threadgroup-local lookup)."""
    block_size = 32
    ctx_lens = [1, 200, 1024]  # single-block, multi-block, many-warp
    max_ctx = max(ctx_lens)
    n_blocks_per_seq = (max_ctx + block_size - 1) // block_size
    num_seqs = len(ctx_lens)
    num_heads = 4
    dtype = mx.float16

    mx.random.seed(13)
    num_blocks = n_blocks_per_seq * num_seqs
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    q_nope = mx.random.normal(shape=(num_seqs, num_heads, _KV_LORA_RANK)).astype(dtype)
    q_pe = mx.random.normal(shape=(num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(
        dtype
    )
    latent_cache = mx.random.normal(shape=(num_blocks, block_size, _LATENT_DIM)).astype(
        dtype
    )
    block_tables_np = np.arange(num_blocks, dtype=np.int32).reshape(
        num_seqs, n_blocks_per_seq
    )
    block_tables = mx.array(block_tables_np)
    context_lens = mx.array(ctx_lens, dtype=mx.uint32)
    cu_seqlens_q = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)

    metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )
    mx.synchronize()

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=ctx_lens,
        scale=0.125,
    )
    rtol, atol = _tolerance(dtype)
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item()


def test_rejects_multi_token_query() -> None:
    """Step 7 supports decode only (one query token per sequence). Any
    request with >1 query tokens must raise — without the guard the
    kernel mis-aligns q_token_idx with seq_idx and reads OOB metadata."""
    block_size = 16
    (
        out_unused,
        q_nope_unused,
        q_pe_unused,
        latent_cache,
        block_tables,
        context_lens,
        _cu_unused,
        _bt_np,
    ) = _make_inputs(
        num_seqs=1,
        num_heads=2,
        ctx_len=4,
        block_size=block_size,
        dtype=mx.float16,
    )
    # 1 sequence with 2 query tokens (prefill of length 2).
    cu_seqlens_q = mx.array([0, 2], dtype=mx.int32)
    out = mx.zeros((2, 2, _KV_LORA_RANK), dtype=mx.float16)
    q_nope = mx.zeros((2, 2, _KV_LORA_RANK), dtype=mx.float16)
    q_pe = mx.zeros((2, 2, _QK_ROPE_HEAD_DIM), dtype=mx.float16)

    with pytest.raises(NotImplementedError, match="one query token per sequence"):
        metal_mla_paged_attention(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
        )


def test_rejects_undersized_block_tables() -> None:
    """If block_tables.shape[1] is too narrow to cover the requested
    ctx_len, the kernel would read past the row into the next sequence's
    row (or off the end of the buffer). The wrapper must catch this
    before dispatch — programmer error, ValueError."""
    block_size = 16
    ctx_len = block_size * 3  # needs 3 blocks per seq
    num_seqs = 1
    num_heads = 2
    dtype = mx.float16

    mx.random.seed(0)
    # Allocate a too-narrow block_tables: only 1 column, but ctx_len needs 3.
    num_blocks = 8  # plenty of physical blocks; the issue is row width
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    q_nope = mx.random.normal(shape=(num_seqs, num_heads, _KV_LORA_RANK)).astype(dtype)
    q_pe = mx.random.normal(shape=(num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(
        dtype
    )
    latent_cache = mx.random.normal(shape=(num_blocks, block_size, _LATENT_DIM)).astype(
        dtype
    )
    block_tables = mx.zeros((num_seqs, 1), dtype=mx.int32)  # one column!
    context_lens = mx.array([ctx_len], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)

    with pytest.raises(ValueError, match="block_tables row width"):
        metal_mla_paged_attention(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
        )


def test_unsupported_kv_lora_rank_raises() -> None:
    """Phase 1 only instantiates kv_lora_rank=512. Anything else must
    raise at dispatch time, not silently dispatch a wrong kernel."""
    out = mx.zeros((1, 2, 16), dtype=mx.float16)
    q_nope = mx.zeros((1, 2, 16), dtype=mx.float16)
    q_pe = mx.zeros((1, 2, 4), dtype=mx.float16)
    latent_cache = mx.zeros((1, 16, 20), dtype=mx.float16)
    block_tables = mx.zeros((1, 1), dtype=mx.int32)
    context_lens = mx.array([1], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)

    with pytest.raises(RuntimeError, match="kv_lora_rank=512"):
        metal_mla_paged_attention(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
        )


# ---------------------------------------------------------------------------
# Split-K + reduce (RFC #360 Phase 3): metal_mla_paged_attention_partitioned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_partitioned_single_partition_matches_dense(dtype: mx.Dtype) -> None:
    """ctx_len ≤ PARTITION_SIZE — only one partition runs and the reduce
    kernel hits its single-partition fast path (plain copy from tmp_out).
    Output must match the dense reference, just like the non-partitioned
    kernel."""
    ctx_len = MLA_PARTITION_SIZE  # exactly one partition (boundary case)
    block_size = 32
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=4,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    metal_mla_paged_attention_partitioned(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )
    mx.synchronize()

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len] * 2,
        scale=0.125,
    )

    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"partitioned single-partition mismatch (dtype={dtype}): "
        f"max_abs_diff={max_abs:.5f}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_partitioned_two_partitions_matches_dense(
    dtype: mx.Dtype, block_size: int
) -> None:
    """ctx_len just over PARTITION_SIZE — exercises the reduce kernel's
    cross-partition merge with a small, deterministic partition count."""
    ctx_len = MLA_PARTITION_SIZE + block_size  # 528 (bs=16) or 544 (bs=32)
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=4,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    metal_mla_paged_attention_partitioned(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )
    mx.synchronize()

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len] * 2,
        scale=0.125,
    )

    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"partitioned 2-partition mismatch (dtype={dtype}, bs={block_size}): "
        f"max_abs_diff={max_abs:.5f}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_partitioned_many_partitions_matches_dense(dtype: mx.Dtype) -> None:
    """Long ctx — 16 partitions, full reduce stress including cross-partition
    online softmax merge accuracy at scale."""
    ctx_len = 8192  # 16 partitions of 512
    block_size = 32
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=2,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    metal_mla_paged_attention_partitioned(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )
    mx.synchronize()

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len] * 2,
        scale=0.125,
    )

    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"partitioned 16-partition mismatch (dtype={dtype}): max_abs_diff={max_abs:.5f}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_partitioned_mixed_ctx_lens_matches_dense(dtype: mx.Dtype) -> None:
    """Heterogeneous ctx across requests — caller's max_num_partitions is
    sized for the longest seq, so shorter seqs leave their tail partitions
    empty (kernel returns early; reduce loops only over the actual count).
    Verifies the early-return / loop-bound contract works correctly."""
    block_size = 32
    ctx_lens = [128, 1024, 2560]  # 1, 2, 5 partitions respectively
    n_seqs = len(ctx_lens)
    n_heads = 2

    max_ctx = max(ctx_lens)
    n_blocks_per_seq = (max_ctx + block_size - 1) // block_size
    num_blocks = n_blocks_per_seq * n_seqs

    mx.random.seed(0)
    out = mx.zeros((n_seqs, n_heads, _KV_LORA_RANK), dtype=dtype)
    q_nope = mx.random.normal(shape=(n_seqs, n_heads, _KV_LORA_RANK)).astype(dtype)
    q_pe = mx.random.normal(shape=(n_seqs, n_heads, _QK_ROPE_HEAD_DIM)).astype(dtype)
    latent_cache = mx.random.normal(shape=(num_blocks, block_size, _LATENT_DIM)).astype(
        dtype
    )

    block_tables_np = np.arange(num_blocks, dtype=np.int32).reshape(
        n_seqs, n_blocks_per_seq
    )
    block_tables = mx.array(block_tables_np)
    context_lens = mx.array(ctx_lens, dtype=mx.uint32)
    cu_seqlens_q = mx.array(list(range(n_seqs + 1)), dtype=mx.int32)

    metal_mla_paged_attention_partitioned(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )
    mx.synchronize()

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=ctx_lens,
        scale=0.125,
    )

    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"partitioned mixed-ctx mismatch (dtype={dtype}): "
        f"max_abs_diff={max_abs:.5f}, ctx_lens={ctx_lens}"
    )


def test_partitioned_rejects_multi_token_query() -> None:
    """Same decode-only contract as the non-partitioned entry."""
    out = mx.zeros((2, 2, _KV_LORA_RANK), dtype=mx.float16)
    q_nope = mx.zeros((2, 2, _KV_LORA_RANK), dtype=mx.float16)
    q_pe = mx.zeros((2, 2, _QK_ROPE_HEAD_DIM), dtype=mx.float16)
    latent_cache = mx.zeros((1, 16, _LATENT_DIM), dtype=mx.float16)
    block_tables = mx.zeros((1, 1), dtype=mx.int32)
    context_lens = mx.array([1], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 2], dtype=mx.int32)  # 2-token query

    with pytest.raises(NotImplementedError, match="decode only"):
        metal_mla_paged_attention_partitioned(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
        )


# ---------------------------------------------------------------------------
# Cross-head amortization (HEADS_PER_TG > 1)
# ---------------------------------------------------------------------------
# G=4 packs 4 query heads into one threadgroup so each K/V load is reused
# across 4 dot products (4× KV bandwidth amortization). num_heads must be
# divisible by G; num_threads is 256 instead of 1024 so the per-thread
# register footprint stays comparable.


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("num_heads", [4, 8, 128])
def test_g4_matches_dense(dtype: mx.Dtype, block_size: int, num_heads: int) -> None:
    """G=4 single-pass kernel must match the dense reference within fp16
    tolerance — same as G=1, just with a different parallelism layout."""
    ctx_len = 384  # spans multiple blocks at both bs=16 and bs=32
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=num_heads,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
        heads_per_tg=4,
    )

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len, ctx_len],
        scale=0.125,
    )

    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"G=4 mismatch (dtype={dtype}, bs={block_size}, H={num_heads}): "
        f"max_abs_diff={max_abs:.5f}"
    )


def test_g4_matches_g1() -> None:
    """G=4 and G=1 must produce identical outputs (up to fp32 rounding) on
    the same workload — different parallelism, same math."""
    ctx_len = 1024
    (
        out_g4,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        _,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=128,
        ctx_len=ctx_len,
        block_size=16,
        dtype=mx.float16,
    )
    out_g1 = mx.zeros_like(out_g4)

    metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out_g4,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
        heads_per_tg=4,
    )
    metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out_g1,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
        heads_per_tg=1,
    )

    diff = mx.abs(out_g4.astype(mx.float32) - out_g1.astype(mx.float32))
    max_abs = mx.max(diff).item()
    # Same dtype, same data, same math — only parallelism differs. Reduction
    # order can differ (different number of simdgroups merging), so allow
    # a small tolerance from accumulation reordering.
    assert max_abs < 1e-2, f"G=4 vs G=1 divergence: max_abs_diff={max_abs:.5f}"


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_g4_partitioned_matches_dense(dtype: mx.Dtype) -> None:
    """G=4 + split-K (Phase 3 long-ctx infra) — must match the dense
    reference, mirroring the G=1 partitioned tests above."""
    ctx_len = 1536  # 3 partitions at PARTITION_SIZE=512
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=8,
        ctx_len=ctx_len,
        block_size=16,
        dtype=dtype,
    )

    metal_mla_paged_attention_partitioned(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
        heads_per_tg=4,
    )

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len, ctx_len],
        scale=0.125,
    )

    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), f"G=4 partitioned mismatch (dtype={dtype}): max_abs_diff={max_abs:.5f}"


def test_g_invalid_raises() -> None:
    """num_heads not divisible by G should raise at the dispatch boundary,
    not silently produce garbage."""
    out = mx.zeros((1, 5, _KV_LORA_RANK), dtype=mx.float16)
    q_nope = mx.zeros((1, 5, _KV_LORA_RANK), dtype=mx.float16)
    q_pe = mx.zeros((1, 5, _QK_ROPE_HEAD_DIM), dtype=mx.float16)
    latent_cache = mx.zeros((1, 16, _LATENT_DIM), dtype=mx.float16)
    block_tables = mx.zeros((1, 1), dtype=mx.int32)
    context_lens = mx.array([1], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)

    with pytest.raises(RuntimeError, match="divisible by heads_per_tg"):
        metal_mla_paged_attention(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
            heads_per_tg=4,  # 5 % 4 != 0
        )


# ---------------------------------------------------------------------------
# Decode 2pass kernel — MLX sdpa_vector_2pass-style cross-head amortization.
# ---------------------------------------------------------------------------
# One TG per (seq, partition) with 32*num_heads threads. All H heads share
# K cache reads — total KV bandwidth amortized H× across the launch.


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("num_heads", [8, 32, 96, 128])
def test_decode_2pass_matches_dense(
    dtype: mx.Dtype, block_size: int, num_heads: int
) -> None:
    """2pass kernel must match the dense reference for all production head
    counts within fp16 / bf16 tolerance."""
    ctx_len = 768
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=num_heads,
        ctx_len=ctx_len,
        block_size=block_size,
        dtype=dtype,
    )

    metal_mla_paged_attention_decode_2pass(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len, ctx_len],
        scale=0.125,
    )

    rtol, atol = _tolerance(dtype)
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"2pass mismatch (dtype={dtype}, bs={block_size}, H={num_heads}): "
        f"max_abs_diff={max_abs:.5f}"
    )


@pytest.mark.parametrize("partition_size", [64, 128, 256, 512])
def test_decode_2pass_partition_sizes_match(partition_size: int) -> None:
    """Each instantiated PARTITION_SIZE must produce the correct output
    on the same workload — different ctx-split, same math."""
    ctx_len = 1536
    (
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_tables_np,
    ) = _make_inputs(
        num_seqs=2,
        num_heads=32,
        ctx_len=ctx_len,
        block_size=16,
        dtype=mx.float16,
    )

    metal_mla_paged_attention_decode_2pass(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
        partition_size=partition_size,
    )

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=[ctx_len, ctx_len],
        scale=0.125,
    )
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    assert max_abs < 5e-3, (
        f"2pass partition_size={partition_size} mismatch: max_abs_diff={max_abs:.5f}"
    )


def test_decode_2pass_mixed_ctx_lens() -> None:
    """Mixed ctx_lens — short / medium / long sequences. Empty-tail
    partitions must contribute nothing to the reduce."""
    ctx_lens = [128, 1024, 2500, 64]
    num_seqs = len(ctx_lens)
    num_heads = 32
    block_size = 16
    max_ctx = max(ctx_lens)
    n_blocks_per_seq = (max_ctx + block_size - 1) // block_size
    num_blocks = n_blocks_per_seq * num_seqs

    mx.random.seed(0)
    q_nope = mx.random.normal(shape=(num_seqs, num_heads, _KV_LORA_RANK)).astype(
        mx.float16
    )
    q_pe = mx.random.normal(shape=(num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(
        mx.float16
    )
    latent_cache = mx.random.normal(shape=(num_blocks, block_size, _LATENT_DIM)).astype(
        mx.float16
    )

    block_tables_np = np.arange(num_blocks, dtype=np.int32).reshape(
        num_seqs, n_blocks_per_seq
    )
    block_tables = mx.array(block_tables_np)
    context_lens = mx.array(ctx_lens, dtype=mx.uint32)
    cu_seqlens_q = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=mx.float16)

    metal_mla_paged_attention_decode_2pass(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=0.125,
    )

    expected = _expected_output(
        q_nope,
        q_pe,
        latent_cache,
        block_tables_np,
        ctx_lens=ctx_lens,
        scale=0.125,
    )
    diff = mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    max_abs = mx.max(diff).item()
    rtol, atol = _tolerance(mx.float16)
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item(), (
        f"2pass mixed-ctx mismatch: max_abs_diff={max_abs:.5f}, ctx_lens={ctx_lens}"
    )


def test_decode_2pass_rejects_multi_token_query() -> None:
    """Same decode-only contract as the other entries."""
    out = mx.zeros((2, 8, _KV_LORA_RANK), dtype=mx.float16)
    q_nope = mx.zeros((2, 8, _KV_LORA_RANK), dtype=mx.float16)
    q_pe = mx.zeros((2, 8, _QK_ROPE_HEAD_DIM), dtype=mx.float16)
    latent_cache = mx.zeros((1, 16, _LATENT_DIM), dtype=mx.float16)
    block_tables = mx.zeros((1, 1), dtype=mx.int32)
    context_lens = mx.array([1], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 2], dtype=mx.int32)

    with pytest.raises(NotImplementedError, match="decode only"):
        metal_mla_paged_attention_decode_2pass(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
        )


# ---------------------------------------------------------------------------
# Mixed-dtype rejection: every MLA dispatcher picks one Metal specialization
# from q_nope.dtype() but the kernel template binds the same `T` to all
# fp16/bf16 buffers (q_nope, q_pe, latent_cache, out, tmp_out). If they
# disagree the shader silently reinterprets bytes — e.g. a bf16 cache read
# as fp16 — and produces corrupt attention. Validate that we reject up
# front instead.
# ---------------------------------------------------------------------------


def _mixed_dtype_inputs(dtype_q: mx.Dtype, dtype_kv: mx.Dtype):
    """Build a minimal valid input set with mismatched query / cache dtypes.
    Production shapes (kv_lora_rank=512) so the dispatcher gets past the
    shape check and reaches dtype validation."""
    block_size = 16
    num_seqs = 1
    num_heads = 4
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype_q)
    q_nope = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype_q)
    q_pe = mx.zeros((num_seqs, num_heads, _QK_ROPE_HEAD_DIM), dtype=dtype_q)
    latent_cache = mx.zeros((1, block_size, _LATENT_DIM), dtype=dtype_kv)
    block_tables = mx.zeros((num_seqs, 1), dtype=mx.int32)
    context_lens = mx.array([1], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
    return out, q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q


def test_mla_rejects_mixed_dtypes_single_pass() -> None:
    """fp16 queries vs bf16 latent cache must raise, not silently corrupt."""
    out, q_nope, q_pe, latent_cache, btab, ctx_lens, cu_q = _mixed_dtype_inputs(
        dtype_q=mx.float16, dtype_kv=mx.bfloat16
    )
    with pytest.raises(RuntimeError, match="must share the same dtype"):
        metal_mla_paged_attention(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=btab,
            context_lens=ctx_lens,
            cu_seqlens_q=cu_q,
            scale=0.125,
        )


def test_mla_rejects_mixed_dtypes_partitioned() -> None:
    out, q_nope, q_pe, latent_cache, btab, ctx_lens, cu_q = _mixed_dtype_inputs(
        dtype_q=mx.bfloat16, dtype_kv=mx.float16
    )
    with pytest.raises(RuntimeError, match="must share the same dtype"):
        metal_mla_paged_attention_partitioned(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=btab,
            context_lens=ctx_lens,
            cu_seqlens_q=cu_q,
            scale=0.125,
        )


def test_mla_rejects_mixed_dtypes_decode_2pass() -> None:
    out, q_nope, q_pe, latent_cache, btab, ctx_lens, cu_q = _mixed_dtype_inputs(
        dtype_q=mx.float16, dtype_kv=mx.bfloat16
    )
    with pytest.raises(RuntimeError, match="must share the same dtype"):
        metal_mla_paged_attention_decode_2pass(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=btab,
            context_lens=ctx_lens,
            cu_seqlens_q=cu_q,
            scale=0.125,
        )


# ============================================================
# Paged FA decode kernel.
# Single-block tests cover ctx == BK == 32 (one single-iter K-tile,
# no multi-block online softmax). Multi-block tests cover the production
# online-softmax merge path.
# ============================================================


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_decode_fa_matches_dense_single_block(dtype: mx.Dtype) -> None:
    """Stage 3a parity: B=1, H=8, ctx == BK == 16 (one page). Kernel's
    QK + softmax + SV must match the dense reference within the
    standard tolerance."""
    block_size = 16
    num_seqs = 1
    num_heads = 8
    ctx_len = 16  # == BK
    n_blocks = ctx_len // block_size

    mx.random.seed(7)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(dtype)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(dtype)
    latent_cache = mx.random.normal((n_blocks, block_size, _LATENT_DIM)).astype(dtype)
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    block_tables = mx.array([[0]], dtype=mx.int32)
    context_lens = mx.array([ctx_len], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    metal_mla_paged_attention_decode_fa(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
    )
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)).item())

    # Dense reference: gather the two blocks into a contiguous
    # [num_seqs, ctx_len, latent_dim] tensor, split kv_norm + k_pe,
    # and run the same math the wrapper's slow path does.
    flat = latent_cache.reshape(n_blocks * block_size, _LATENT_DIM)
    gathered = flat[:ctx_len].reshape(num_seqs, ctx_len, _LATENT_DIM)
    kv_norm = gathered[:, :, :_KV_LORA_RANK]
    k_pe = gathered[:, :, _KV_LORA_RANK:]
    expected = _absorbed_mla_dense_reference(
        q_nope.astype(mx.float32),
        q_pe.astype(mx.float32),
        kv_norm.astype(mx.float32),
        k_pe.astype(mx.float32),
        scale,
    ).astype(dtype)
    mx.eval(expected)

    rtol, atol = _tolerance(dtype)
    max_abs = mx.max(
        mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    ).item()
    assert bool(mx.allclose(out, expected, rtol=rtol, atol=atol).item()), (
        f"FA single-block mismatch (dtype={dtype}): max_abs_diff={max_abs:.5f}"
    )


def test_decode_fa_multi_head_group_and_multi_seq() -> None:
    """Stage 3a parity at non-degenerate grid: 2 seqs × (H=16 → 2 head
    groups) = 4 TGs. Catches grid-index swaps (seq vs head_group) and
    cross-TG aliasing in the buffer indexing."""
    block_size = 16
    num_seqs = 2
    num_heads = 16  # 2 head groups
    ctx_len = 16
    n_blocks = num_seqs  # one block per seq

    mx.random.seed(19)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(mx.float16)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(mx.float16)
    latent_cache = mx.random.normal((n_blocks, block_size, _LATENT_DIM)).astype(
        mx.float16
    )
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=mx.float16)
    block_tables = mx.array([[0], [1]], dtype=mx.int32)
    context_lens = mx.array([ctx_len, ctx_len], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1, 2], dtype=mx.int32)
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    metal_mla_paged_attention_decode_fa(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
    )
    mx.eval(out)

    # Per-seq dense reference.
    flat = latent_cache.reshape(n_blocks * block_size, _LATENT_DIM)
    kv_norm = mx.stack(
        [
            flat[i * block_size : (i + 1) * block_size, :_KV_LORA_RANK]
            for i in range(num_seqs)
        ],
        axis=0,
    )
    k_pe = mx.stack(
        [
            flat[i * block_size : (i + 1) * block_size, _KV_LORA_RANK:]
            for i in range(num_seqs)
        ],
        axis=0,
    )
    expected = _absorbed_mla_dense_reference(
        q_nope.astype(mx.float32),
        q_pe.astype(mx.float32),
        kv_norm.astype(mx.float32),
        k_pe.astype(mx.float32),
        scale,
    ).astype(mx.float16)
    mx.eval(expected)

    rtol, atol = _tolerance(mx.float16)
    max_abs = mx.max(
        mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    ).item()
    assert bool(mx.allclose(out, expected, rtol=rtol, atol=atol).item()), (
        f"FA multi-TG mismatch: max_abs_diff={max_abs:.5f}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("ctx_len", [24, 32, 64, 128])
def test_decode_fa_matches_dense_multi_block(dtype: mx.Dtype, ctx_len: int) -> None:
    """Stage 3b parity: multi-block ctx with online softmax merge across
    K tiles. ``ctx_len`` choices cover both an exact-multiple of BK=16
    (32, 64, 128) and a partial tail (24 → one full tile + 8 valid
    cols in the second tile). The partial-tail case stresses the
    in-kernel mask that zeroes out S cols beyond ctx_len."""
    block_size = 16
    num_seqs = 1
    num_heads = 8
    n_blocks = (ctx_len + block_size - 1) // block_size

    mx.random.seed(101 + ctx_len)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(dtype)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(dtype)
    latent_cache = mx.random.normal((n_blocks, block_size, _LATENT_DIM)).astype(dtype)
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    block_tables = mx.array([list(range(n_blocks))], dtype=mx.int32)
    context_lens = mx.array([ctx_len], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    metal_mla_paged_attention_decode_fa(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
    )
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)).item())

    flat = latent_cache.reshape(n_blocks * block_size, _LATENT_DIM)
    gathered = flat[:ctx_len].reshape(num_seqs, ctx_len, _LATENT_DIM)
    kv_norm = gathered[:, :, :_KV_LORA_RANK]
    k_pe = gathered[:, :, _KV_LORA_RANK:]
    expected = _absorbed_mla_dense_reference(
        q_nope.astype(mx.float32),
        q_pe.astype(mx.float32),
        kv_norm.astype(mx.float32),
        k_pe.astype(mx.float32),
        scale,
    ).astype(dtype)
    mx.eval(expected)

    rtol, atol = _tolerance(dtype)
    max_abs = mx.max(
        mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    ).item()
    assert bool(mx.allclose(out, expected, rtol=rtol, atol=atol).item()), (
        f"FA multi-block mismatch (dtype={dtype}, ctx={ctx_len}): max_abs_diff={max_abs:.5f}"
    )


@pytest.mark.parametrize("num_heads", [64, 128])
def test_decode_fa_matches_dense_production_shape(num_heads: int) -> None:
    """Stage 3b parity on the production grid: B=4 × H ∈ {64, 128} ×
    ctx=2048, fp16. This is the cell where the sdpa_vector-style
    kernels lose 0.4× to MLX (status doc §1.3) — getting parity here
    means the FA kernel is ready for the Stage 4 bench."""
    block_size = 16
    num_seqs = 4
    ctx_len = 2048
    n_blocks_per_seq = ctx_len // block_size
    n_blocks = n_blocks_per_seq * num_seqs

    mx.random.seed(2048 + num_heads)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(mx.float16)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(mx.float16)
    latent_cache = mx.random.normal((n_blocks, block_size, _LATENT_DIM)).astype(
        mx.float16
    )
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=mx.float16)
    bt_np = np.arange(n_blocks, dtype=np.int32).reshape(num_seqs, n_blocks_per_seq)
    block_tables = mx.array(bt_np)
    context_lens = mx.array([ctx_len] * num_seqs, dtype=mx.uint32)
    cu_seqlens_q = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    metal_mla_paged_attention_decode_fa(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
    )
    mx.eval(out)

    # Per-seq reference.
    outs = []
    for i in range(num_seqs):
        flat = mx.concatenate(
            [latent_cache[int(bt_np[i, b])] for b in range(n_blocks_per_seq)],
            axis=0,
        )
        kv_norm = flat[:, :_KV_LORA_RANK].reshape(1, ctx_len, _KV_LORA_RANK)
        k_pe = flat[:, _KV_LORA_RANK:].reshape(1, ctx_len, _QK_ROPE_HEAD_DIM)
        ref_i = _absorbed_mla_dense_reference(
            q_nope[i : i + 1].astype(mx.float32),
            q_pe[i : i + 1].astype(mx.float32),
            kv_norm.astype(mx.float32),
            k_pe.astype(mx.float32),
            scale,
        )
        outs.append(ref_i)
    expected = mx.concatenate(outs, axis=0).astype(mx.float16)
    mx.eval(expected)

    rtol, atol = _tolerance(mx.float16)
    max_abs = mx.max(
        mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    ).item()
    assert bool(mx.allclose(out, expected, rtol=rtol, atol=atol).item()), (
        f"FA production-shape mismatch (H={num_heads}, ctx={ctx_len}): "
        f"max_abs_diff={max_abs:.5f}"
    )


def test_decode_fa_matches_dense_mixed_ctx_batch() -> None:
    """Multi-seq batch with different ctx_lens per request. Confirms
    that each TG reads its own seq's context_lens / block_tables row
    independently and doesn't leak state across seqs."""
    block_size = 16
    num_seqs = 3
    num_heads = 8
    ctx_lens = [16, 48, 24]
    max_blocks = max((c + block_size - 1) // block_size for c in ctx_lens)
    pool_blocks = sum((c + block_size - 1) // block_size for c in ctx_lens)

    mx.random.seed(307)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(mx.float16)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(mx.float16)
    latent_cache = mx.random.normal((pool_blocks, block_size, _LATENT_DIM)).astype(
        mx.float16
    )
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=mx.float16)

    # Pack block tables — each seq gets its own contiguous range. Pad
    # to max_blocks; padding entries past each seq's required blocks
    # are never read.
    bt_np = np.zeros((num_seqs, max_blocks), dtype=np.int32)
    cursor = 0
    for i, c in enumerate(ctx_lens):
        n = (c + block_size - 1) // block_size
        bt_np[i, :n] = np.arange(cursor, cursor + n)
        cursor += n
    block_tables = mx.array(bt_np)
    context_lens = mx.array(ctx_lens, dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1, 2, 3], dtype=mx.int32)
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    metal_mla_paged_attention_decode_fa(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
    )
    mx.eval(out)

    # Per-seq reference (each seq has its own ctx_len, gathers its own blocks).
    outs = []
    for i, c in enumerate(ctx_lens):
        n = (c + block_size - 1) // block_size
        seq_blocks = mx.concatenate(
            [latent_cache[int(bt_np[i, b])] for b in range(n)], axis=0
        )[:c, :]
        kv_norm = seq_blocks[:, :_KV_LORA_RANK].reshape(1, c, _KV_LORA_RANK)
        k_pe = seq_blocks[:, _KV_LORA_RANK:].reshape(1, c, _QK_ROPE_HEAD_DIM)
        ref_i = _absorbed_mla_dense_reference(
            q_nope[i : i + 1].astype(mx.float32),
            q_pe[i : i + 1].astype(mx.float32),
            kv_norm.astype(mx.float32),
            k_pe.astype(mx.float32),
            scale,
        )
        outs.append(ref_i)
    expected = mx.concatenate(outs, axis=0).astype(mx.float16)
    mx.eval(expected)

    rtol, atol = _tolerance(mx.float16)
    max_abs = mx.max(
        mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    ).item()
    assert bool(mx.allclose(out, expected, rtol=rtol, atol=atol).item()), (
        f"FA mixed-ctx mismatch: max_abs_diff={max_abs:.5f}"
    )


def test_decode_fa_indirect_block_tables() -> None:
    """Block table points at a non-zero physical block. Catches absolute-
    vs-relative block index bugs in the K loader."""
    block_size = 16
    num_seqs = 1
    num_heads = 8
    ctx_len = 16
    num_blocks_pool = 8  # bigger than what the seq needs

    mx.random.seed(31)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(mx.float16)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(mx.float16)
    latent_cache = mx.random.normal((num_blocks_pool, block_size, _LATENT_DIM)).astype(
        mx.float16
    )
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=mx.float16)
    # Seq's logical block 0 lives at physical block 5.
    block_tables = mx.array([[5]], dtype=mx.int32)
    context_lens = mx.array([ctx_len], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    metal_mla_paged_attention_decode_fa(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
    )
    mx.eval(out)

    # Reference reads from physical block 5.
    referenced = latent_cache[5].reshape(1, block_size, _LATENT_DIM)
    kv_norm = referenced[:, :, :_KV_LORA_RANK]
    k_pe = referenced[:, :, _KV_LORA_RANK:]
    expected = _absorbed_mla_dense_reference(
        q_nope.astype(mx.float32),
        q_pe.astype(mx.float32),
        kv_norm.astype(mx.float32),
        k_pe.astype(mx.float32),
        scale,
    ).astype(mx.float16)
    mx.eval(expected)

    rtol, atol = _tolerance(mx.float16)
    max_abs = mx.max(
        mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    ).item()
    assert bool(mx.allclose(out, expected, rtol=rtol, atol=atol).item()), (
        f"FA indirect block_tables mismatch: max_abs_diff={max_abs:.5f}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "ctx_len,partition_size",
    [(64, 64), (128, 64), (256, 128), (2048, 128), (4096, 512)],
)
def test_decode_fa_partitioned_matches_dense(
    dtype: mx.Dtype, ctx_len: int, partition_size: int
) -> None:
    """Stage 6.2 parity: partitioned FA + reduce kernel chain matches
    the dense reference. Cells span:
    - ctx=64, ps=64 (single partition)
    - ctx=128, ps=64 (2 partitions)
    - ctx=256, ps=128 (2 partitions)
    - ctx=2048, ps=128 (16 partitions — main long-ctx target)
    - ctx=4096, ps=512 (8 partitions, sanity at production scale)
    """
    block_size = 16
    num_seqs = 1
    num_heads = 8
    n_blocks = (ctx_len + block_size - 1) // block_size

    mx.random.seed(401 + ctx_len + partition_size)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(dtype)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(dtype)
    latent_cache = mx.random.normal((n_blocks, block_size, _LATENT_DIM)).astype(dtype)
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    block_tables = mx.array([list(range(n_blocks))], dtype=mx.int32)
    context_lens = mx.array([ctx_len], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    metal_mla_paged_attention_decode_fa_partitioned(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
        partition_size=partition_size,
    )
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)).item())

    flat = latent_cache.reshape(n_blocks * block_size, _LATENT_DIM)
    gathered = flat[:ctx_len].reshape(num_seqs, ctx_len, _LATENT_DIM)
    kv_norm = gathered[:, :, :_KV_LORA_RANK]
    k_pe = gathered[:, :, _KV_LORA_RANK:]
    expected = _absorbed_mla_dense_reference(
        q_nope.astype(mx.float32),
        q_pe.astype(mx.float32),
        kv_norm.astype(mx.float32),
        k_pe.astype(mx.float32),
        scale,
    ).astype(dtype)
    mx.eval(expected)

    rtol, atol = _tolerance(dtype)
    max_abs = mx.max(
        mx.abs(out.astype(mx.float32) - expected.astype(mx.float32))
    ).item()
    assert bool(mx.allclose(out, expected, rtol=rtol, atol=atol).item()), (
        f"FA partitioned mismatch (dtype={dtype}, ctx={ctx_len}, ps={partition_size}): "
        f"max_abs_diff={max_abs:.5f}"
    )


def test_decode_fa_partitioned_matches_non_partitioned() -> None:
    """The partitioned FA + reduce chain must produce numerically
    equivalent output to the non-partitioned FA on the same input
    (within numerical noise from the partition-then-merge softmax)."""
    block_size = 16
    num_seqs = 2
    num_heads = 16
    ctx_len = 512
    n_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    n_blocks = n_blocks_per_seq * num_seqs

    mx.random.seed(509)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(mx.float16)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(mx.float16)
    latent_cache = mx.random.normal((n_blocks, block_size, _LATENT_DIM)).astype(
        mx.float16
    )
    bt_np = np.arange(n_blocks, dtype=np.int32).reshape(num_seqs, n_blocks_per_seq)
    block_tables = mx.array(bt_np)
    context_lens = mx.array([ctx_len] * num_seqs, dtype=mx.uint32)
    cu_seqlens_q = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    out_partitioned = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=mx.float16)
    metal_mla_paged_attention_decode_fa_partitioned(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out_partitioned,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
        partition_size=128,
    )

    out_non_partitioned = mx.zeros(
        (num_seqs, num_heads, _KV_LORA_RANK), dtype=mx.float16
    )
    metal_mla_paged_attention_decode_fa(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out_non_partitioned,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
    )

    mx.eval(out_partitioned, out_non_partitioned)
    rtol, atol = _tolerance(mx.float16)
    max_abs = mx.max(
        mx.abs(
            out_partitioned.astype(mx.float32) - out_non_partitioned.astype(mx.float32)
        )
    ).item()
    assert bool(
        mx.allclose(out_partitioned, out_non_partitioned, rtol=rtol, atol=atol).item()
    ), f"FA partitioned vs non-partitioned mismatch: max_abs_diff={max_abs:.5f}"


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "num_heads,ctx_len",
    [
        (8, 128),  # short ctx, single head_group
        (16, 512),  # short ctx boundary
        (16, 2048),  # long ctx
        (40, 1024),  # MiniCPM3 H + mid ctx
        (96, 2048),  # GLM-full H + long ctx
        (128, 2048),  # DeepSeek H + long ctx
    ],
)
def test_decode_fa_wide_matches_narrow(
    num_heads: int, ctx_len: int, dtype: mx.Dtype
) -> None:
    """The wide (BK=64, WN=8) FA instantiation must match the narrow
    (BK=32, WN=4) anchor across the production grid. Output equivalence
    within fp16/bf16 noise — the two tile shapes process the same SV
    math, just with different per-iter K throughput."""
    block_size = 16
    num_seqs = 2
    n_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    n_blocks = n_blocks_per_seq * num_seqs

    mx.random.seed(1117)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(dtype)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(dtype)
    latent_cache = mx.random.normal((n_blocks, block_size, _LATENT_DIM)).astype(dtype)
    bt_np = np.arange(n_blocks, dtype=np.int32).reshape(num_seqs, n_blocks_per_seq)
    block_tables = mx.array(bt_np)
    context_lens = mx.array([ctx_len] * num_seqs, dtype=mx.uint32)
    cu_seqlens_q = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    out_narrow = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    metal_mla_paged_attention_decode_fa(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out_narrow,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
        use_wide=False,
    )
    out_wide = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    metal_mla_paged_attention_decode_fa(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out_wide,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
        use_wide=True,
    )
    mx.eval(out_narrow, out_wide)
    rtol, atol = _tolerance(dtype)
    max_abs = mx.max(
        mx.abs(out_narrow.astype(mx.float32) - out_wide.astype(mx.float32))
    ).item()
    assert bool(mx.allclose(out_narrow, out_wide, rtol=rtol, atol=atol).item()), (
        f"FA wide vs narrow mismatch: max_abs_diff={max_abs:.5f}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "num_heads,ctx_len,partition_size",
    [
        (16, 512, 128),
        (16, 2048, 128),
        (96, 4096, 512),
        (128, 2048, 128),
    ],
)
def test_decode_fa_partitioned_wide_matches_narrow(
    num_heads: int, ctx_len: int, partition_size: int, dtype: mx.Dtype
) -> None:
    """Partitioned FA wide (BK=64/WN=8) vs narrow (BK=32/WN=4) parity.
    Reduce kernel is shared so the only diff is per-partition tile
    shape; output should match within reduction-merge noise."""
    block_size = 16
    num_seqs = 2
    n_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    n_blocks = n_blocks_per_seq * num_seqs

    mx.random.seed(2222)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(dtype)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(dtype)
    latent_cache = mx.random.normal((n_blocks, block_size, _LATENT_DIM)).astype(dtype)
    bt_np = np.arange(n_blocks, dtype=np.int32).reshape(num_seqs, n_blocks_per_seq)
    block_tables = mx.array(bt_np)
    context_lens = mx.array([ctx_len] * num_seqs, dtype=mx.uint32)
    cu_seqlens_q = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    out_narrow = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    metal_mla_paged_attention_decode_fa_partitioned(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out_narrow,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
        partition_size=partition_size,
        use_wide=False,
    )
    out_wide = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    metal_mla_paged_attention_decode_fa_partitioned(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=latent_cache,
        out=out_wide,
        block_tables=block_tables,
        context_lens=context_lens,
        cu_seqlens_q=cu_seqlens_q,
        scale=scale,
        partition_size=partition_size,
        use_wide=True,
    )
    mx.eval(out_narrow, out_wide)
    rtol, atol = _tolerance(dtype)
    max_abs = mx.max(
        mx.abs(out_narrow.astype(mx.float32) - out_wide.astype(mx.float32))
    ).item()
    assert bool(mx.allclose(out_narrow, out_wide, rtol=rtol, atol=atol).item()), (
        f"FA partitioned wide vs narrow mismatch: max_abs_diff={max_abs:.5f}"
    )


def test_decode_fa_rejects_unsupported_num_heads() -> None:
    """``num_heads`` not a multiple of BQ=8 must raise — the kernel has
    no partial-tile handling along the M axis."""
    block_size = 16
    q_nope = mx.zeros((1, 7, _KV_LORA_RANK), dtype=mx.float16)
    q_pe = mx.zeros((1, 7, _QK_ROPE_HEAD_DIM), dtype=mx.float16)
    latent_cache = mx.zeros((1, block_size, _LATENT_DIM), dtype=mx.float16)
    out = mx.zeros((1, 7, _KV_LORA_RANK), dtype=mx.float16)
    block_tables = mx.zeros((1, 1), dtype=mx.int32)
    context_lens = mx.array([block_size], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)

    with pytest.raises(ValueError, match="multiple of 8"):
        metal_mla_paged_attention_decode_fa(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
        )


def test_decode_fa_rejects_multi_token_query() -> None:
    """Same decode-only contract as the other MLA kernels."""
    block_size = 16
    q_nope = mx.zeros((2, 8, _KV_LORA_RANK), dtype=mx.float16)
    q_pe = mx.zeros((2, 8, _QK_ROPE_HEAD_DIM), dtype=mx.float16)
    latent_cache = mx.zeros((1, block_size, _LATENT_DIM), dtype=mx.float16)
    out = mx.zeros((2, 8, _KV_LORA_RANK), dtype=mx.float16)
    block_tables = mx.zeros((1, 1), dtype=mx.int32)
    context_lens = mx.array([block_size], dtype=mx.uint32)
    # Single request with 2 query tokens — fails the cu_seqlens delta=1 check.
    cu_seqlens_q = mx.array([0, 2], dtype=mx.int32)

    with pytest.raises(NotImplementedError, match="decode only"):
        metal_mla_paged_attention_decode_fa(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
        )


def test_mla_rejects_out_dtype_mismatch() -> None:
    """Catches the case where everything reads correctly but the output
    buffer has the wrong dtype — the kernel would write fp16 bytes into a
    bf16 buffer (or vice versa)."""
    block_size = 16
    num_seqs = 1
    num_heads = 4
    q_nope = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=mx.float16)
    q_pe = mx.zeros((num_seqs, num_heads, _QK_ROPE_HEAD_DIM), dtype=mx.float16)
    latent_cache = mx.zeros((1, block_size, _LATENT_DIM), dtype=mx.float16)
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=mx.bfloat16)  # mismatch
    block_tables = mx.zeros((num_seqs, 1), dtype=mx.int32)
    context_lens = mx.array([1], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
    with pytest.raises(RuntimeError, match="must share the same dtype"):
        metal_mla_paged_attention(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
        )


# ---------------------------------------------------------------------------
# Primitive correctness / graph tests (Stage 9 + Stage 10)
# ---------------------------------------------------------------------------


def _make_paged_inputs(
    *, num_seqs: int, num_heads: int, ctx_len: int, dtype: mx.Dtype, seed: int = 0
):
    """Build a self-contained synthetic MLA decode workload — q_nope,
    q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q. Used
    by the primitive parity / graph tests below."""
    block_size = 16
    n_blocks_per_seq = math.ceil(ctx_len / block_size)
    n_blocks = n_blocks_per_seq * num_seqs

    mx.random.seed(seed)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(dtype)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(dtype)
    latent_cache = mx.random.normal((n_blocks, block_size, _LATENT_DIM)).astype(dtype)
    bt_np = np.arange(n_blocks, dtype=np.int32).reshape(num_seqs, n_blocks_per_seq)
    block_tables = mx.array(bt_np)
    context_lens = mx.array([ctx_len] * num_seqs, dtype=mx.uint32)
    cu_seqlens_q = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)
    return q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "num_seqs,num_heads,ctx_len,heads_per_tg",
    [
        (1, 16, 128, 1),  # B=1 short ctx, hpt=1
        (1, 40, 2048, 2),  # B=1 long ctx, hpt=2 (H % 2 == 0)
        (8, 16, 128, 2),  # B=8 short ctx
        (8, 40, 2048, 2),  # B=8 long ctx, single-pass routing
        (8, 96, 128, 2),  # B=8 H % 2 == 0
    ],
)
def test_single_pass_primitive_matches_eager(
    num_seqs: int, num_heads: int, ctx_len: int, heads_per_tg: int, dtype: mx.Dtype
) -> None:
    """The lazy Primitive variant of the single-pass MLA decode kernel
    must produce bit-exact output vs the eager binding across every
    cell the wrapper actually routes to it (B=1 all H, or B≥2 H<32)."""
    q_nope, q_pe, lc, bt, ctx, cu = _make_paged_inputs(
        num_seqs=num_seqs, num_heads=num_heads, ctx_len=ctx_len, dtype=dtype
    )
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    out_eager = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    metal_mla_paged_attention(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=lc,
        out=out_eager,
        block_tables=bt,
        context_lens=ctx,
        cu_seqlens_q=cu,
        scale=scale,
        heads_per_tg=heads_per_tg,
    )
    out_prim = metal_mla_paged_attention_primitive(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=lc,
        block_tables=bt,
        context_lens=ctx,
        cu_seqlens_q=cu,
        scale=scale,
        heads_per_tg=heads_per_tg,
    )
    mx.eval(out_eager, out_prim)
    max_abs = mx.max(
        mx.abs(out_eager.astype(mx.float32) - out_prim.astype(mx.float32))
    ).item()
    # Same kernel body, same inputs — must be bit-exact (0 diff).
    assert max_abs == 0.0, (
        f"single-pass primitive ≠ eager at "
        f"num_seqs={num_seqs}, H={num_heads}, ctx={ctx_len}, "
        f"hpt={heads_per_tg}, dtype={dtype}: max_abs={max_abs}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("use_wide", [True, False])
@pytest.mark.parametrize(
    "num_seqs,num_heads,ctx_len",
    [
        (2, 16, 128),  # short ctx, low H
        (2, 64, 2048),  # long ctx
        (8, 128, 2048),  # production DeepSeek shape
    ],
)
def test_fa_primitive_matches_eager(
    num_seqs: int, num_heads: int, ctx_len: int, use_wide: bool, dtype: mx.Dtype
) -> None:
    """FA decode lazy Primitive must be bit-exact vs the eager binding
    on both wide (BK=64/WN=8) and narrow (BK=32/WN=4) tile shapes."""
    q_nope, q_pe, lc, bt, ctx, cu = _make_paged_inputs(
        num_seqs=num_seqs, num_heads=num_heads, ctx_len=ctx_len, dtype=dtype
    )
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    out_eager = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    metal_mla_paged_attention_decode_fa(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=lc,
        out=out_eager,
        block_tables=bt,
        context_lens=ctx,
        cu_seqlens_q=cu,
        scale=scale,
        use_wide=use_wide,
    )
    out_prim = metal_mla_paged_attention_decode_fa_primitive(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=lc,
        block_tables=bt,
        context_lens=ctx,
        cu_seqlens_q=cu,
        scale=scale,
        use_wide=use_wide,
    )
    mx.eval(out_eager, out_prim)
    max_abs = mx.max(
        mx.abs(out_eager.astype(mx.float32) - out_prim.astype(mx.float32))
    ).item()
    assert max_abs == 0.0, (
        f"FA primitive ≠ eager at num_seqs={num_seqs}, H={num_heads}, "
        f"ctx={ctx_len}, use_wide={use_wide}, dtype={dtype}: "
        f"max_abs={max_abs}"
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "num_seqs,num_heads,ctx_len",
    [
        (1, 16, 512),  # short ctx single-pass routing fallthrough
        (8, 64, 4096),  # B≥4 H=64 long-ish ctx — Stage 12 hot path
        (8, 96, 8192),  # production GLM-full long ctx
        (8, 128, 2048),  # H=128 medium ctx
    ],
)
def test_2pass_primitive_matches_eager(
    num_seqs: int, num_heads: int, ctx_len: int, dtype: mx.Dtype
) -> None:
    """2pass primitive vs eager binding parity. Scratch (exp_sums /
    max_logits / tmp_out) is allocated inside the primitive's
    eval_gpu in the primitive variant vs Python-side in the eager
    variant — output must still be bit-exact since the kernel body
    is unchanged."""
    q_nope, q_pe, lc, bt, ctx, cu = _make_paged_inputs(
        num_seqs=num_seqs, num_heads=num_heads, ctx_len=ctx_len, dtype=dtype
    )
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    out_eager = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype)
    metal_mla_paged_attention_decode_2pass(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=lc,
        out=out_eager,
        block_tables=bt,
        context_lens=ctx,
        cu_seqlens_q=cu,
        scale=scale,
    )
    out_prim = metal_mla_paged_attention_decode_2pass_primitive(
        q_nope=q_nope,
        q_pe=q_pe,
        latent_cache=lc,
        block_tables=bt,
        context_lens=ctx,
        cu_seqlens_q=cu,
        scale=scale,
    )
    mx.eval(out_eager, out_prim)
    max_abs = mx.max(
        mx.abs(out_eager.astype(mx.float32) - out_prim.astype(mx.float32))
    ).item()
    assert max_abs == 0.0, (
        f"2pass primitive ≠ eager at num_seqs={num_seqs}, H={num_heads}, "
        f"ctx={ctx_len}, dtype={dtype}: max_abs={max_abs}"
    )


@pytest.mark.parametrize(
    "kernel_fn",
    [
        metal_mla_paged_attention_primitive,
        metal_mla_paged_attention_decode_fa_primitive,
        metal_mla_paged_attention_decode_2pass_primitive,
    ],
    ids=["single_pass", "fa", "2pass"],
)
def test_primitive_two_calls_dont_alias(kernel_fn) -> None:
    """Regression test for the lazy-graph aliasing footgun that killed
    the old ``out_kvr`` cache (RFC #362 P1 review). Two consecutive
    primitive calls with different inputs must each return a distinct
    lazy array; evaluating both must yield each call's own answer, not
    the second call overwriting the first.

    The eager binding aliased buffers when callers cached ``out``
    across calls; the primitive variant is immune by construction
    because each call's output array carries its own Primitive node,
    but we lock that property explicitly."""
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    q_nope_1, q_pe_1, lc_1, bt_1, ctx_1, cu_1 = _make_paged_inputs(
        num_seqs=1, num_heads=16, ctx_len=128, dtype=mx.float16, seed=11
    )
    q_nope_2, q_pe_2, lc_2, bt_2, ctx_2, cu_2 = _make_paged_inputs(
        num_seqs=1, num_heads=16, ctx_len=128, dtype=mx.float16, seed=22
    )

    kwargs_common: dict[str, Any] = {"scale": scale}
    if kernel_fn is metal_mla_paged_attention_decode_fa_primitive:
        kwargs_common["use_wide"] = True
    elif kernel_fn is metal_mla_paged_attention_primitive:
        kwargs_common["heads_per_tg"] = 1
    # 2pass primitive takes no extra kwargs.

    # Independent reference for each input, evaluated immediately.
    ref_1 = kernel_fn(
        q_nope=q_nope_1,
        q_pe=q_pe_1,
        latent_cache=lc_1,
        block_tables=bt_1,
        context_lens=ctx_1,
        cu_seqlens_q=cu_1,
        **kwargs_common,
    )
    mx.eval(ref_1)
    ref_2 = kernel_fn(
        q_nope=q_nope_2,
        q_pe=q_pe_2,
        latent_cache=lc_2,
        block_tables=bt_2,
        context_lens=ctx_2,
        cu_seqlens_q=cu_2,
        **kwargs_common,
    )
    mx.eval(ref_2)

    # Now the aliasing scenario: queue both lazy calls without an
    # eval between, then eval together.
    out_1 = kernel_fn(
        q_nope=q_nope_1,
        q_pe=q_pe_1,
        latent_cache=lc_1,
        block_tables=bt_1,
        context_lens=ctx_1,
        cu_seqlens_q=cu_1,
        **kwargs_common,
    )
    out_2 = kernel_fn(
        q_nope=q_nope_2,
        q_pe=q_pe_2,
        latent_cache=lc_2,
        block_tables=bt_2,
        context_lens=ctx_2,
        cu_seqlens_q=cu_2,
        **kwargs_common,
    )
    mx.eval(out_1, out_2)

    assert bool(mx.allclose(out_1, ref_1, rtol=0, atol=0).item()), (
        "out_1 doesn't match call-1 reference — possible aliasing"
    )
    assert bool(mx.allclose(out_2, ref_2, rtol=0, atol=0).item()), (
        "out_2 doesn't match call-2 reference"
    )
    # Sanity: the two calls' outputs differ (else this test is vacuous).
    assert not bool(mx.allclose(out_1, out_2, rtol=1e-3, atol=1e-3).item()), (
        "out_1 == out_2 — inputs weren't different enough; test is vacuous"
    )


@pytest.mark.parametrize(
    "kernel_fn",
    [
        metal_mla_paged_attention_primitive,
        metal_mla_paged_attention_decode_fa_primitive,
        metal_mla_paged_attention_decode_2pass_primitive,
    ],
    ids=["single_pass", "fa", "2pass"],
)
def test_primitive_respects_cache_dependency(kernel_fn) -> None:
    """A pre-attention scatter into the latent cache must be visible
    to the primitive kernel through MLX's lazy graph. The test stages
    a scatter that writes a sentinel into the cache, then runs the
    primitive over the post-scatter cache and asserts the kernel saw
    the new content.

    The wrapper's hot path runs: kv_a_proj → kv_a_layernorm →
    scatter into ``latent_caches[layer_idx]`` → primitive attention.
    The scatter is an in-place setitem; if MLX failed to serialise it
    before the primitive's kernel dispatch, attention would read
    stale data."""
    scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    num_seqs, num_heads, ctx_len = 1, 16, 64
    block_size = 16
    n_blocks_per_seq = math.ceil(ctx_len / block_size)
    n_blocks = n_blocks_per_seq * num_seqs

    mx.random.seed(33)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(mx.float16)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(mx.float16)
    cache_zero = mx.zeros((n_blocks, block_size, _LATENT_DIM), dtype=mx.float16)
    bt = mx.array(
        np.arange(n_blocks, dtype=np.int32).reshape(num_seqs, n_blocks_per_seq)
    )
    cu = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)
    context_lens = mx.array([ctx_len] * num_seqs, dtype=mx.uint32)

    kwargs_common: dict[str, Any] = {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "block_tables": bt,
        "context_lens": context_lens,
        "cu_seqlens_q": cu,
        "scale": scale,
    }
    if kernel_fn is metal_mla_paged_attention_decode_fa_primitive:
        kwargs_common["use_wide"] = True
    elif kernel_fn is metal_mla_paged_attention_primitive:
        kwargs_common["heads_per_tg"] = 1
    # 2pass primitive takes no extra kwargs.

    out_zero = kernel_fn(latent_cache=cache_zero, **kwargs_common)
    mx.eval(out_zero)

    # Scatter a non-zero pattern, then run the primitive on the
    # SAME cache variable in the same lazy chain.
    flat = cache_zero.reshape(-1, _LATENT_DIM)
    sentinel = mx.ones((flat.shape[0], _LATENT_DIM), dtype=mx.float16) * 0.5
    flat[mx.arange(flat.shape[0], dtype=mx.int64)] = sentinel
    cache_after = flat.reshape(n_blocks, block_size, _LATENT_DIM)
    out_after = kernel_fn(latent_cache=cache_after, **kwargs_common)
    mx.eval(out_after)

    # Attention over the scattered cache must differ materially from
    # the zero-cache version. If MLX hadn't honoured the dependency,
    # the kernel might have read pre-scatter content — symptom would
    # be a smaller-than-expected diff.
    diff = mx.max(
        mx.abs(out_zero.astype(mx.float32) - out_after.astype(mx.float32))
    ).item()
    assert diff > 1e-4, (
        f"primitive read stale cache: diff={diff:.4e}; "
        f"dependency ordering may be broken"
    )
