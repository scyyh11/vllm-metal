# SPDX-License-Identifier: Apache-2.0
"""Direct unit tests for the MLA Metal kernel (RFC #360).

Single-pass + split-K + 2pass + FA decode kernels. The single-pass
kernel handles ``ctx_len`` of any size that fits the caller-provided
block_tables row; the partitioned (split-K) variant chunks ctx across
multiple TGs and merges partials with the reduce kernel; the 2pass
variant uses an MLX sdpa_vector_2pass-style cross-head amortized layout;
the FA variant uses ``simdgroup_matrix<T, 8, 8>`` MMAs over the same
paged latent cache.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.metal import (
    MLA_PARTITION_SIZE,
    metal_mla_paged_attention,
    metal_mla_paged_attention_decode_2pass,
    metal_mla_paged_attention_decode_fa,
    metal_mla_paged_attention_decode_fa_partitioned,
    metal_mla_paged_attention_partitioned,
    mla_bf16_fa_available,
)


def _skip_if_no_bf16_fa(dtype: mx.Dtype) -> None:
    """Skip a bf16 FA test on devices without native bfloat support.
    The bf16 FA kernels are gated at metallib compile time on such
    devices (mla.metal `#if defined(__HAVE_BFLOAT__)`); the Python /
    C++ wrappers reject bf16 FA calls up-front so callers do not hit a
    raw `get_kernel` miss. Tests that parametrize over `dtype` should
    call this at function entry to mark bf16 as `skipped` rather than
    `failed` on those targets."""
    if dtype == mx.bfloat16 and not mla_bf16_fa_available():
        pytest.skip("bf16 FA kernels require native bfloat support on the Metal device")


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
        num_seqs=1,
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

    expected = _expected_output(
        q_nope, q_pe, latent_cache, block_tables_np, ctx_lens=[ctx_len], scale=0.125
    )

    rtol, atol = _tolerance(dtype)
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32])
def test_partitioned_two_partitions_matches_dense(
    dtype: mx.Dtype, block_size: int
) -> None:
    """ctx_len spanning two partitions — exercises the reduce kernel's full
    online-softmax merge path. Each partition computes its own (max, lse,
    partial out); reduce normalizes against the global max."""
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
        num_seqs=1,
        num_heads=8,
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

    expected = _expected_output(
        q_nope, q_pe, latent_cache, block_tables_np, ctx_lens=[ctx_len], scale=0.125
    )

    rtol, atol = _tolerance(dtype)
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_partitioned_many_partitions_matches_dense(dtype: mx.Dtype) -> None:
    """Long context spanning many partitions — stresses the reduce kernel's
    cross-partition merge across a large max_num_partitions count."""
    ctx_len = MLA_PARTITION_SIZE * 6 + 17  # 3089 — last partition is partial
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
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_partitioned_mixed_ctx_lens_matches_dense(dtype: mx.Dtype) -> None:
    """Mixed ctx_lens — sequences with different partition counts in one
    batch. Each seq's reduce kernel sees only its own num_partitions; the
    unused partition slots in the scratch buffers must be zero-init
    sentinels."""
    ctx_lens = [100, MLA_PARTITION_SIZE * 2, MLA_PARTITION_SIZE - 1, 1500]
    block_size = 16
    num_seqs = len(ctx_lens)
    num_heads = 4
    dtype_in = dtype

    mx.random.seed(0)
    max_ctx = max(ctx_lens)
    n_blocks_per_seq = (max_ctx + block_size - 1) // block_size
    num_blocks = n_blocks_per_seq * num_seqs

    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=dtype_in)
    q_nope = mx.random.normal(shape=(num_seqs, num_heads, _KV_LORA_RANK)).astype(
        dtype_in
    )
    q_pe = mx.random.normal(shape=(num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(
        dtype_in
    )
    latent_cache = mx.random.normal(shape=(num_blocks, block_size, _LATENT_DIM)).astype(
        dtype_in
    )
    block_tables_np = np.arange(num_blocks, dtype=np.int32).reshape(
        num_seqs, n_blocks_per_seq
    )
    block_tables = mx.array(block_tables_np)
    context_lens = mx.array(ctx_lens, dtype=mx.uint32)
    cu_seqlens_q = mx.array(list(range(num_seqs + 1)), dtype=mx.int32)

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

    expected = _expected_output(
        q_nope, q_pe, latent_cache, block_tables_np, ctx_lens=ctx_lens, scale=0.125
    )

    rtol, atol = _tolerance(dtype)
    assert mx.allclose(
        out.astype(mx.float32), expected.astype(mx.float32), rtol=rtol, atol=atol
    ).item()


def test_partitioned_rejects_multi_token_query() -> None:
    """Same decode-only guard as the non-partitioned entry."""
    block_size = 16
    latent_cache = mx.zeros((4, block_size, _LATENT_DIM), dtype=mx.float16)
    block_tables = mx.zeros((1, 4), dtype=mx.int32)
    context_lens = mx.array([8], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 2], dtype=mx.int32)
    out = mx.zeros((2, 2, _KV_LORA_RANK), dtype=mx.float16)
    q_nope = mx.zeros((2, 2, _KV_LORA_RANK), dtype=mx.float16)
    q_pe = mx.zeros((2, 2, _QK_ROPE_HEAD_DIM), dtype=mx.float16)

    with pytest.raises(NotImplementedError, match="one query token per sequence"):
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
    _skip_if_no_bf16_fa(dtype)
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
    _skip_if_no_bf16_fa(dtype)
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
    independently and doesn't leak state across seqs. Padding entries
    past each seq's valid blocks are set to -1 (0xFFFFFFFF on the
    Metal uint32_t side) so any unclamped pbi -> OOB load on
    latent_cache trips a GPU fault rather than silently aliasing to
    block 0."""
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

    # Pack block tables — each seq gets its own contiguous range, with
    # padding entries past its valid blocks set to -1 to force the
    # k_base_for clamp to use num_valid_blocks (not max_num_blocks_per_seq).
    bt_np = np.full((num_seqs, max_blocks), -1, dtype=np.int32)
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
    _skip_if_no_bf16_fa(dtype)
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


def test_decode_fa_partitioned_mixed_ctx_with_poison_padding() -> None:
    """Partitioned FA counterpart to test_decode_fa_matches_dense_mixed_ctx_batch:
    different ctx_lens per request, block_tables padded with -1 past each
    seq's valid blocks. Exercises the partitioned k_base_for clamp — an
    unclamped pbi would dereference 0xFFFFFFFF as a physical-block index
    and OOB load on latent_cache."""
    block_size = 16
    num_seqs = 3
    num_heads = 8
    ctx_lens = [64, 192, 128]
    partition_size = 64
    max_blocks = max((c + block_size - 1) // block_size for c in ctx_lens)
    pool_blocks = sum((c + block_size - 1) // block_size for c in ctx_lens)

    mx.random.seed(613)
    q_nope = mx.random.normal((num_seqs, num_heads, _KV_LORA_RANK)).astype(mx.float16)
    q_pe = mx.random.normal((num_seqs, num_heads, _QK_ROPE_HEAD_DIM)).astype(mx.float16)
    latent_cache = mx.random.normal((pool_blocks, block_size, _LATENT_DIM)).astype(
        mx.float16
    )
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=mx.float16)

    bt_np = np.full((num_seqs, max_blocks), -1, dtype=np.int32)
    cursor = 0
    for i, c in enumerate(ctx_lens):
        n = (c + block_size - 1) // block_size
        bt_np[i, :n] = np.arange(cursor, cursor + n)
        cursor += n
    block_tables = mx.array(bt_np)
    context_lens = mx.array(ctx_lens, dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1, 2, 3], dtype=mx.int32)
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
        f"FA partitioned mixed-ctx (poison padding) mismatch: "
        f"max_abs_diff={max_abs:.5f}"
    )


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
    _skip_if_no_bf16_fa(dtype)
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
    _skip_if_no_bf16_fa(dtype)
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


def test_decode_fa_partitioned_rejects_unsupported_partition_size() -> None:
    """FA partitioned only instantiates ps ∈ {64, 128, 512}. Caller
    passing ps=256 (which is an MLA 2pass-supported size, easy to
    confuse) must be rejected at the Python wrapper with a clear error
    rather than slipping through to a C++ `get_kernel` miss."""
    block_size = 16
    num_seqs = 1
    num_heads = 8
    ctx_len = 64
    q_nope = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=mx.float16)
    q_pe = mx.zeros((num_seqs, num_heads, _QK_ROPE_HEAD_DIM), dtype=mx.float16)
    latent_cache = mx.zeros((1, block_size, _LATENT_DIM), dtype=mx.float16)
    out = mx.zeros((num_seqs, num_heads, _KV_LORA_RANK), dtype=mx.float16)
    block_tables = mx.zeros((num_seqs, 4), dtype=mx.int32)
    context_lens = mx.array([ctx_len], dtype=mx.uint32)
    cu_seqlens_q = mx.array([0, 1], dtype=mx.int32)
    with pytest.raises(ValueError, match="partition_size must be in"):
        metal_mla_paged_attention_decode_fa_partitioned(
            q_nope=q_nope,
            q_pe=q_pe,
            latent_cache=latent_cache,
            out=out,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens_q=cu_seqlens_q,
            scale=0.125,
            partition_size=256,  # not in _MLA_DECODE_FA_PARTITION_SIZES
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
