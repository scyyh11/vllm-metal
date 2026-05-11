# SPDX-License-Identifier: Apache-2.0
"""MLA grid regression gate.

Runs the canonical (H × B × ctx) wrapper-level grid and compares
against a stored baseline. Catches kernel-level regressions in
already-close cells (≥ 0.95×) before they merge.

Workflow:

    # Establish a fresh baseline (e.g. after a routing or kernel change
    # that the user has validated).
    uv run python -m tools.benchmark.mla_grid_regression --save \
        /tmp/mla_regression_baseline.json

    # Check current state against the saved baseline (exits non-zero
    # on regression).
    uv run python -m tools.benchmark.mla_grid_regression --check \
        /tmp/mla_regression_baseline.json

Cell judgement is conservative: a cell is "regressed" only when
*both* its kernel_ms grew by more than ``--ms-tolerance`` (default
10%) **and** its speedup (mlx_ms / kernel_ms) dropped by more than
``--speedup-tolerance`` (default 0.05 = 5 percentage points). This
avoids flagging cells where MLX itself happens to run slower on
the machine that round.

An additional ``--min-speedup`` flag (default 0.95) enforces an
absolute floor regardless of baseline: any cell at less than
0.95× MLX after a run is reported even if it's within tolerance
of the baseline. This catches the case where someone re-saves a
baseline that already lost ground.

The full gate covers 120 cells. Baseline JSON files are local artifacts:
attach the important benchmark summary to the PR rather than committing
generated result snapshots.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    raise SystemExit("Run as a module: python -m tools.benchmark.mla_grid_regression")

from tools.benchmark.mla_wrapper_benchmark import (
    DTYPE_MAP,
    Workload,
    _time_path,
    build_setup,
)

# Default-on gate grid. Five dimensions:
#
#   - num_heads ∈ {16, 40, 64, 96, 128}    GLM-Air, MiniCPM3, GLM-Z1,
#                                           GLM-full, DeepSeek-V3
#   - batch_size ∈ {1, 8}                  interactive + low-serving
#   - ctx ∈ {128, 2048, 8192}              short + medium + long
#   - dtype ∈ {float16, bfloat16}          both kernel-instantiated
#   - block_size ∈ {16, 32}                both cache-instantiated
#
# Full cross-product = 120 cells. Adding cells here means promising
# to defend them across future kernel changes; remove only with a
# documented reason (see ``--restrict-*`` flags to subset for
# quicker dev iteration).
HEADS_GRID: tuple[int, ...] = (16, 40, 64, 96, 128)
BATCH_GRID: tuple[int, ...] = (1, 8)
CTX_GRID: tuple[int, ...] = (128, 2048, 8192)
DTYPE_GRID: tuple[str, ...] = ("float16", "bfloat16")
BLOCK_SIZE_GRID: tuple[int, ...] = (16, 32)


def _all_cells(
    heads: tuple[int, ...] = HEADS_GRID,
    batches: tuple[int, ...] = BATCH_GRID,
    ctxs: tuple[int, ...] = CTX_GRID,
    dtypes: tuple[str, ...] = DTYPE_GRID,
    block_sizes: tuple[int, ...] = BLOCK_SIZE_GRID,
):
    """Cartesian product of the grid dimensions, in run order
    (outermost = dtype, then block_size, H, B, ctx). The order
    minimises cache rebuilds in build_setup."""
    for dt in dtypes:
        for bs in block_sizes:
            for h in heads:
                for b in batches:
                    for c in ctxs:
                        yield h, b, c, dt, bs


def _cell_key(h: int, b: int, c: int, dt: str, bs: int) -> str:
    return f"H={h},B={b},ctx={c},dt={dt},bs={bs}"


def run_grid(
    warmup: int,
    iters: int,
    seed: int,
    heads: tuple[int, ...] = HEADS_GRID,
    batches: tuple[int, ...] = BATCH_GRID,
    ctxs: tuple[int, ...] = CTX_GRID,
    dtypes: tuple[str, ...] = DTYPE_GRID,
    block_sizes: tuple[int, ...] = BLOCK_SIZE_GRID,
    outer_runs: int = 1,
    aggregate: str = "median",
    quiet_inner: bool = False,
) -> dict[str, Any]:
    """Run the regression grid.

    With ``outer_runs == 1`` the function behaves exactly like the
    original single-shot harness. With ``outer_runs > 1`` it loops the
    whole grid that many times and aggregates the per-cell timings
    via ``aggregate`` ∈ {``"median"`` (default), ``"mean"``}.

    The aggregated speedup is what drives the gate (``--check`` /
    ``--min-speedup``). Per-run samples are also persisted in the
    saved JSON so reviewers can sanity-check the aggregate.

    Picking K: at K=1 a single 60-iter snapshot has ~1-2% wrapper noise
    at near-parity cells (P7-C showed individual H=128 B=1 cells
    flipping between 0.97 × and 1.01 ×). K=3 smooths that out for
    most cells; K=5 is the recommended setting for the strict
    default-on gate. K is capped by overall runtime — full 120-cell
    × K=5 takes ~50 min on M5 Max.
    """
    grid = list(_all_cells(heads, batches, ctxs, dtypes, block_sizes))
    print(
        f"# regression grid: {len(grid)} cells "
        f"(dtypes={list(dtypes)} bs={list(block_sizes)} H={list(heads)} "
        f"B={list(batches)} ctx={list(ctxs)}) "
        f"warmup={warmup} iters={iters} outer_runs={outer_runs} "
        f"aggregate={aggregate}"
    )

    # Per-cell, per-run timing samples — collected across all outer
    # runs, aggregated at the end.
    per_cell_samples: dict[str, dict[str, list[float]]] = {}
    for outer in range(outer_runs):
        if outer_runs > 1:
            print(f"# outer run {outer + 1}/{outer_runs}")
        for h, b, c, dt_name, bs in grid:
            dtype = DTYPE_MAP[dt_name]
            # Use seed + outer offset so each outer run sees slightly
            # different cache contents and decode_input — averages
            # over both wrapper-side timing noise and data-dependent
            # kernel jitter.
            workload = Workload(
                batch_size=b,
                ctx_len=c,
                num_heads=h,
                dtype=dtype,
                dtype_name=dt_name,
                block_size=bs,
                seed=seed + outer,
            )
            setup = build_setup(workload)
            mlx = _time_path(setup, kernel_enabled=False, warmup=warmup, iters=iters)
            kernel = _time_path(setup, kernel_enabled=True, warmup=warmup, iters=iters)
            mlx_ms = mlx.mean_ms
            kernel_ms = kernel.mean_ms
            if mlx_ms is None or kernel_ms is None:
                raise RuntimeError(
                    f"cell ({h}, {b}, {c}, {dt_name}, bs{bs}) outer={outer} "
                    f"failed: mlx={mlx.error or 'ok'}, "
                    f"kernel={kernel.error or 'ok'}"
                )
            speedup = mlx_ms / kernel_ms
            key = _cell_key(h, b, c, dt_name, bs)
            slot = per_cell_samples.setdefault(
                key, {"mlx_ms": [], "kernel_ms": [], "speedup": []}
            )
            slot["mlx_ms"].append(mlx_ms)
            slot["kernel_ms"].append(kernel_ms)
            slot["speedup"].append(speedup)

            if not quiet_inner or outer_runs == 1:
                marker = "✓" if speedup >= 0.99 else (" " if speedup >= 0.95 else "✗")
                print(
                    f" {marker} H={h:>3} B={b} ctx={c:>4} {dt_name[:2]}{bs:>2}  "
                    f"mlx={mlx_ms:6.3f}ms kernel={kernel_ms:6.3f}ms "
                    f"speedup={speedup:5.2f}x"
                )

    agg_fn = statistics.median if aggregate == "median" else statistics.fmean

    cells: dict[str, dict[str, Any]] = {}
    for key, slot in per_cell_samples.items():
        speedups = slot["speedup"]
        cell_entry: dict[str, Any] = {
            "mlx_ms": agg_fn(slot["mlx_ms"]),
            "kernel_ms": agg_fn(slot["kernel_ms"]),
            "speedup": agg_fn(speedups),
        }
        if outer_runs > 1:
            cell_entry["speedup_stdev"] = statistics.stdev(speedups)
            cell_entry["speedup_min"] = min(speedups)
            cell_entry["speedup_max"] = max(speedups)
            cell_entry["samples"] = {
                "mlx_ms": slot["mlx_ms"],
                "kernel_ms": slot["kernel_ms"],
                "speedup": speedups,
            }
        cells[key] = cell_entry

    if outer_runs > 1:
        # One-line summary per cell after aggregation, so users still
        # see the gate-relevant numbers without scrolling through
        # every outer run.
        print(f"# aggregated ({aggregate}) over {outer_runs} runs:")
        for h, b, c, dt_name, bs in grid:
            key = _cell_key(h, b, c, dt_name, bs)
            entry = cells[key]
            speedup = entry["speedup"]
            stdev = entry["speedup_stdev"]
            marker = "✓" if speedup >= 0.99 else (" " if speedup >= 0.95 else "✗")
            print(
                f" {marker} H={h:>3} B={b} ctx={c:>4} {dt_name[:2]}{bs:>2}  "
                f"speedup={speedup:5.2f}× "
                f"(stdev={stdev:.3f}, "
                f"range {entry['speedup_min']:.2f}-{entry['speedup_max']:.2f})"
            )

    return {
        "warmup": warmup,
        "iters": iters,
        "outer_runs": outer_runs,
        "aggregate": aggregate,
        "timestamp_unix": int(time.time()),
        "dimensions": {
            "heads": list(heads),
            "batches": list(batches),
            "ctxs": list(ctxs),
            "dtypes": list(dtypes),
            "block_sizes": list(block_sizes),
        },
        "cells": cells,
    }


def compare(
    baseline: dict[str, Any],
    current: dict[str, Any],
    ms_tolerance: float,
    speedup_tolerance: float,
) -> list[str]:
    """Return a list of regression descriptions (empty == no regressions).

    Walks ``current["cells"]`` so the comparison reflects whatever grid
    the current run actually covered; baseline cells that aren't in the
    current run are silently skipped (the user may have subset the grid
    via ``--restrict-*`` flags).
    """
    regressions: list[str] = []
    for cell_key, c_cell in current["cells"].items():
        b_cell = baseline["cells"].get(cell_key)
        if b_cell is None:
            continue
        ms_grew = c_cell["kernel_ms"] / b_cell["kernel_ms"] - 1
        speedup_dropped = b_cell["speedup"] - c_cell["speedup"]
        if ms_grew > ms_tolerance and speedup_dropped > speedup_tolerance:
            regressions.append(
                f"{cell_key}: kernel_ms {b_cell['kernel_ms']:.3f} → "
                f"{c_cell['kernel_ms']:.3f} (+{ms_grew * 100:.1f}%); "
                f"speedup {b_cell['speedup']:.2f}x → "
                f"{c_cell['speedup']:.2f}x "
                f"(-{speedup_dropped * 100:.1f} pp)"
            )
    return regressions


def _parse_csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


def _parse_csv_strs(s: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--restrict-dtypes",
        type=_parse_csv_strs,
        default=DTYPE_GRID,
        help="Comma-separated dtype subset (default: all).",
    )
    parser.add_argument(
        "--restrict-block-sizes",
        type=_parse_csv_ints,
        default=BLOCK_SIZE_GRID,
        help="Comma-separated block_size subset (default: 16,32).",
    )
    parser.add_argument(
        "--restrict-heads",
        type=_parse_csv_ints,
        default=HEADS_GRID,
        help="Comma-separated num_heads subset (default: 16,40,64,96,128).",
    )
    parser.add_argument(
        "--restrict-batches",
        type=_parse_csv_ints,
        default=BATCH_GRID,
        help="Comma-separated batch subset (default: 1,8).",
    )
    parser.add_argument(
        "--restrict-ctxs",
        type=_parse_csv_ints,
        default=CTX_GRID,
        help="Comma-separated ctx subset (default: 128,2048,8192).",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outer-runs",
        type=int,
        default=1,
        help="Repeat the whole grid this many times and aggregate the "
        "per-cell speedup (default 1 = legacy single-shot). The "
        "strict default-on CI gate uses 5 (~50 min on M5 Max) to "
        "smooth out wrapper-noise crossings around 1.00 ×.",
    )
    parser.add_argument(
        "--aggregate",
        choices=("median", "mean"),
        default="median",
        help="How to combine per-outer-run speedups for the gate "
        "column (default median — robust to single-run outliers).",
    )
    parser.add_argument(
        "--quiet-inner",
        action="store_true",
        help="With --outer-runs > 1, suppress per-cell prints inside "
        "each outer run; only show the aggregated summary.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Write current grid to this JSON path (creates a new baseline).",
    )
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="Compare current grid to this baseline JSON; exit non-zero "
        "if any cell regresses outside tolerance.",
    )
    parser.add_argument(
        "--ms-tolerance",
        type=float,
        default=0.10,
        help="Per-cell kernel_ms growth tolerance (default 10%%).",
    )
    parser.add_argument(
        "--speedup-tolerance",
        type=float,
        default=0.05,
        help="Per-cell speedup drop tolerance, in absolute speedup "
        "units (default 0.05 = 5 percentage points).",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=0.95,
        help="Absolute speedup floor — any cell below this is "
        "reported as a regression (default 0.95). Independent "
        "of --check baseline. Set to 0 to disable.",
    )
    args = parser.parse_args(argv)

    for dt in args.restrict_dtypes:
        if dt not in DTYPE_MAP:
            print(
                f"# unknown dtype {dt!r}; supported: {list(DTYPE_MAP)}", file=sys.stderr
            )
            return 2
    for bs in args.restrict_block_sizes:
        if bs not in BLOCK_SIZE_GRID:
            print(
                f"# unknown block_size {bs}; supported: {list(BLOCK_SIZE_GRID)}",
                file=sys.stderr,
            )
            return 2

    if args.outer_runs < 1:
        print(f"# --outer-runs must be >= 1; got {args.outer_runs}", file=sys.stderr)
        return 2
    current = run_grid(
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        heads=args.restrict_heads,
        batches=args.restrict_batches,
        ctxs=args.restrict_ctxs,
        dtypes=args.restrict_dtypes,
        block_sizes=args.restrict_block_sizes,
        outer_runs=args.outer_runs,
        aggregate=args.aggregate,
        quiet_inner=args.quiet_inner,
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(current, indent=2) + "\n")
        print(f"# wrote baseline → {args.save}")

    fail = False
    if args.check is not None:
        if not args.check.is_file():
            print(f"# baseline not found: {args.check}", file=sys.stderr)
            return 2
        baseline = json.loads(args.check.read_text())
        regressions = compare(
            baseline, current, args.ms_tolerance, args.speedup_tolerance
        )
        if regressions:
            print()
            print(f"# {len(regressions)} regression(s) vs {args.check}:")
            for r in regressions:
                print(f"  ✗ {r}")
            fail = True
        else:
            print(f"# no regressions vs {args.check}")

    if args.min_speedup > 0:
        below_floor = [
            (k, c["speedup"])
            for k, c in current["cells"].items()
            if c["speedup"] < args.min_speedup
        ]
        if below_floor:
            print()
            print(
                f"# {len(below_floor)} cell(s) below absolute floor "
                f"(speedup < {args.min_speedup}):"
            )
            for cell_key, speedup in below_floor:
                print(f"  ✗ {cell_key}: speedup={speedup:.2f}x")
            fail = True
        else:
            print(f"# all cells ≥ {args.min_speedup}× MLX")

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
