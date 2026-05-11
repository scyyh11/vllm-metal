# SPDX-License-Identifier: Apache-2.0
"""Wrapper-level benchmark for MLA paged attention (RFC #360 follow-up).

Times end-to-end ``MLAPagedAttentionWrapper.__call__`` for absorbed-MLA
decode workloads, comparing the experimental Metal kernel path
(``VLLM_METAL_MLA_KERNEL=1``) against the MLX SDPA slow path (default).

This reports the production-level cost around the kernel. Kernel-only
microbenchmarks can be useful during development, but they ignore
wrapper-side work that production actually pays:

- q/kv projections, RoPE, ``kv_a_layernorm``,
- block_tables packing into a 2D ``int32`` MLX array,
- per-request Python loop in the slow path,
- output reshape + ``o_proj``.

Either path's overhead can dominate at small batch / short context, so
flipping the production default to the kernel needs the wrapper-level
table, not the kernel-time table — see RFC #360 review (LxYuan,
2026-05-10).

Examples:
  python -m tools.benchmark.mla_wrapper_benchmark
  python -m tools.benchmark.mla_wrapper_benchmark --heads 16 --batch 8 --ctx 2048
  python -m tools.benchmark.mla_wrapper_benchmark --dtype bfloat16 --warmup 3 --iters 30
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

if __package__ in (None, ""):
    raise SystemExit("Run as a module: python -m tools.benchmark.mla_wrapper_benchmark")

import vllm_metal.paged_attention_common as pac
from vllm_metal.mlx_backend.mla_cache import MLAPagedLatentCache
from vllm_metal.paged_attention_backend.mla import MLAPagedAttentionWrapper

# Match the kernel's instantiation in kernels_v2/mla.metal.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM

# Production absorbed-MLA shapes (DeepSeek-V2/V3, GLM-4.5 lineage). Held
# fixed across the grid so the softmax temperature, q/kv projection cost,
# and o_proj cost are realistic instead of synthetic.
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128

DTYPE_MAP = {"float16": mx.float16, "bfloat16": mx.bfloat16}

# Production grid agreed on under RFC #360 review (LxYuan, 2026-05-10):
# H ∈ {16, 64, 128}, B ∈ {1, 8, 32}, ctx ∈ {128, 2048, 8192}, fp16 + bf16.
DEFAULTS: dict[str, object] = {
    "batch": (1, 8, 32),
    "ctx": (128, 2048, 8192),
    "heads": (16, 64, 128),
    "dtype": ("float16", "bfloat16"),
    "block_size": 16,
    "warmup": 5,
    "iters": 20,
    "seed": 0,
}


class _AbsorbedMLAStub(nn.Module):
    """Synthetic absorbed-MLA inner module at kernel-instantiated shapes.

    Provides the attribute surface ``MLAPagedAttentionWrapper`` reads
    (``embed_q``, ``unembed_out``, ``q_proj``, ``kv_a_proj_with_mqa``,
    ``kv_a_layernorm``, ``o_proj``, ``rope``, ``num_heads``, ``q_head_dim``,
    ``qk_nope_head_dim``, ``qk_rope_head_dim``, ``kv_lora_rank``,
    ``q_lora_rank``, ``scale``).  Identity RoPE — bench cares about timing
    and shape, not numerics.
    """

    def __init__(self, num_heads: int, hidden: int) -> None:
        super().__init__()
        self.q_lora_rank = None
        self.num_heads = num_heads
        self.q_head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
        self.qk_nope_head_dim = QK_NOPE_HEAD_DIM
        self.qk_rope_head_dim = QK_ROPE_HEAD_DIM
        self.kv_lora_rank = KV_LORA_RANK
        self.scale = 1.0 / math.sqrt(self.q_head_dim)

        self.q_proj = nn.Linear(hidden, num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, KV_LORA_RANK + QK_ROPE_HEAD_DIM, bias=False
        )
        self.kv_a_layernorm = nn.LayerNorm(KV_LORA_RANK)
        self.embed_q = nn.Linear(QK_NOPE_HEAD_DIM, KV_LORA_RANK, bias=False)
        self.unembed_out = nn.Linear(KV_LORA_RANK, V_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(num_heads * V_HEAD_DIM, hidden, bias=False)

    def rope(self, x: mx.array, offset: int = 0) -> mx.array:
        return x


@dataclass(frozen=True)
class Workload:
    batch_size: int
    ctx_len: int
    num_heads: int
    dtype: mx.Dtype
    dtype_name: str
    block_size: int
    seed: int

    @property
    def num_blocks_per_seq(self) -> int:
        return math.ceil(self.ctx_len / self.block_size)

    @property
    def num_blocks(self) -> int:
        return self.num_blocks_per_seq * self.batch_size

    @property
    def hidden(self) -> int:
        return self.num_heads * V_HEAD_DIM


def _cast_module(module: nn.Module, dtype: mx.Dtype) -> None:
    """Cast every parameter in ``module`` to ``dtype`` so weight loads
    don't dominate the first iteration's wall time."""
    for name, value in module.parameters().items():
        if isinstance(value, mx.array):
            module.update({name: value.astype(dtype)})


@dataclass
class _BenchSetup:
    wrapper: MLAPagedAttentionWrapper
    decode_input: mx.array
    decode_ctx: pac.PagedAttentionContext


def build_setup(workload: Workload) -> _BenchSetup:
    mx.random.seed(workload.seed)

    cache = MLAPagedLatentCache(
        num_layers=1,
        latent_dim=LATENT_DIM,
        num_blocks=workload.num_blocks,
        block_size=workload.block_size,
        dtype=workload.dtype,
    )
    # Pre-fill the cache with random data. Only timing matters here; the
    # decode call writes its own latent at slot_mapping and reads
    # ctx_len-1 random rows back, which exercises the same dispatch as
    # production.
    cache.latent_caches[0] = mx.random.normal(
        shape=(workload.num_blocks, workload.block_size, LATENT_DIM)
    ).astype(workload.dtype)

    inner = _AbsorbedMLAStub(num_heads=workload.num_heads, hidden=workload.hidden)
    _cast_module(inner, workload.dtype)
    wrapper = MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache)

    # One new token per request; land it at the last position of the seq's
    # last block so context_lens lines up with the pre-filled rows.
    slot_mapping: list[int] = []
    block_tables: list[list[int]] = []
    context_lens: list[int] = []
    offsets: list[int] = []
    for i in range(workload.batch_size):
        seq_blocks = list(
            range(
                i * workload.num_blocks_per_seq, (i + 1) * workload.num_blocks_per_seq
            )
        )
        block_tables.append(seq_blocks)
        context_lens.append(workload.ctx_len)
        last_pos = workload.ctx_len - 1
        last_block = seq_blocks[last_pos // workload.block_size]
        last_slot = last_block * workload.block_size + (last_pos % workload.block_size)
        slot_mapping.append(last_slot)
        offsets.append(last_pos)

    cu_seqlens = list(range(workload.batch_size + 1))

    decode_input = mx.random.normal(
        shape=(1, workload.batch_size, workload.hidden)
    ).astype(workload.dtype)
    mx.eval(decode_input, cache.latent_caches[0])

    return _BenchSetup(
        wrapper=wrapper,
        decode_input=decode_input,
        decode_ctx=pac.PagedAttentionContext(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            cu_seqlens=cu_seqlens,
            offsets=offsets,
        ),
    )


def _time(
    fn: Callable[[], mx.array], warmup: int, iters: int
) -> tuple[float, float, float]:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)

    timings_ms: list[float] = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    return (
        statistics.fmean(timings_ms),
        float(np.percentile(timings_ms, 50)),
        float(np.percentile(timings_ms, 95)),
    )


@dataclass
class _PathResult:
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    error: str = ""


def _time_path(
    setup: _BenchSetup, kernel_enabled: bool, warmup: int, iters: int
) -> _PathResult:
    os.environ["VLLM_METAL_MLA_KERNEL"] = "1" if kernel_enabled else "0"
    pac.set_context(setup.decode_ctx)
    try:
        mean_ms, p50_ms, p95_ms = _time(
            lambda: setup.wrapper(setup.decode_input, mask=None, cache=None),
            warmup,
            iters,
        )
    except Exception as e:  # noqa: BLE001
        return _PathResult(None, None, None, f"{type(e).__name__}: {e}")
    finally:
        pac.clear_context()
    return _PathResult(mean_ms, p50_ms, p95_ms)


def _format_workload(w: Workload) -> str:
    return (
        f"B={w.batch_size:<3} ctx={w.ctx_len:<6} "
        f"H={w.num_heads:<4} dt={w.dtype_name:<8}"
    )


def _format_path(name: str, r: _PathResult) -> str:
    if r.mean_ms is None:
        return f"{name}=ERR ({r.error})"
    return f"{name}={r.mean_ms:7.3f}ms (p50={r.p50_ms:.2f} p95={r.p95_ms:.2f})"


def parse_csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def parse_csv_dtypes(s: str) -> tuple[str, ...]:
    out: list[str] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        if x not in DTYPE_MAP:
            raise SystemExit(f"unknown dtype {x!r}; supported: {list(DTYPE_MAP)}")
        out.append(x)
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Wrapper-level MLA bench (RFC #360 review #2)."
    )
    ap.add_argument(
        "--batch",
        type=parse_csv_ints,
        default=DEFAULTS["batch"],
        help="comma-separated batch sizes (default: 1,8,32)",
    )
    ap.add_argument(
        "--ctx",
        type=parse_csv_ints,
        default=DEFAULTS["ctx"],
        help="comma-separated ctx_lens (default: 128,2048,8192)",
    )
    ap.add_argument(
        "--heads",
        type=parse_csv_ints,
        default=DEFAULTS["heads"],
        help="comma-separated num_heads values (default: 16,64,128)",
    )
    ap.add_argument(
        "--dtype",
        type=parse_csv_dtypes,
        default=DEFAULTS["dtype"],
        help="comma-separated dtypes (default: float16,bfloat16)",
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=DEFAULTS["block_size"],
        choices=(16, 32),
        help="cache block size (default: 16)",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=DEFAULTS["warmup"],
        help="warmup iterations (default: 5)",
    )
    ap.add_argument(
        "--iters",
        type=int,
        default=DEFAULTS["iters"],
        help="measured iterations (default: 20)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"],
        help="rng seed (default: 0)",
    )
    args = ap.parse_args(argv)

    print(
        f"# warmup={args.warmup} iters={args.iters} "
        f"block_size={args.block_size} seed={args.seed}"
    )
    print(
        f"# kernel instantiation: kv_lora_rank={KV_LORA_RANK} "
        f"qk_rope_head_dim={QK_ROPE_HEAD_DIM}"
    )
    print(
        f"# qk_nope_head_dim={QK_NOPE_HEAD_DIM} v_head_dim={V_HEAD_DIM} "
        f"scale=1/sqrt({QK_NOPE_HEAD_DIM}+{QK_ROPE_HEAD_DIM})"
    )
    print(
        "# wrapper end-to-end timing — includes q/kv projections, RoPE, "
        "cache scatter, block_tables packing, attention, o_proj"
    )
    print()

    grid = [
        Workload(
            batch_size=b,
            ctx_len=ctx,
            num_heads=h,
            dtype=DTYPE_MAP[dt],
            dtype_name=dt,
            block_size=args.block_size,
            seed=args.seed,
        )
        for dt in args.dtype
        for h in args.heads
        for ctx in args.ctx
        for b in args.batch
    ]

    for w in grid:
        try:
            setup = build_setup(w)
        except Exception as e:  # noqa: BLE001
            print(f"{_format_workload(w)}  setup ERR: {type(e).__name__}: {e}")
            continue

        mlx_r = _time_path(
            setup, kernel_enabled=False, warmup=args.warmup, iters=args.iters
        )
        ker_r = _time_path(
            setup, kernel_enabled=True, warmup=args.warmup, iters=args.iters
        )

        if mlx_r.mean_ms is not None and ker_r.mean_ms is not None:
            speedup = f"kernel/mlx={mlx_r.mean_ms / ker_r.mean_ms:.2f}x"
        else:
            speedup = "kernel/mlx=N/A"

        print(
            f"{_format_workload(w)} | "
            f"{_format_path('mlx', mlx_r)} | "
            f"{_format_path('kernel', ker_r)} | {speedup}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
