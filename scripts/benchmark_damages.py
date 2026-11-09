"""Profile calculate_direct_damages using reproducible synthetic split data.

Run from a development environment with snail and psutil installed:
    python scripts/benchmark_damages.py
    python scripts/benchmark_damages.py --assets 1000 --samples 1000
    python scripts/benchmark_damages.py --profile /tmp/damages.prof
    python scripts/benchmark_damages.py --ead-only --no-profile
    python scripts/benchmark_damages.py --no-profile --metrics /tmp/damages.json

Data generation, imports and result checks are excluded from measurements.
The first invocation measures wall time and RSS. A separate invocation reports
cProfile cumulative time (including callees) and direct time (excluding callees),
without the memory sampler. --no-profile skips this second invocation.
An optional .prof file can be opened with pstats or other profile viewers.

Process RSS is sampled every 10 ms during the measured call, including native
NumPy/SciPy memory. The reported peak is approximate and may miss shorter spikes.
Peak increase is relative to RSS after input generation; RSS includes allocator
retention and is not a count of allocated bytes. Run each workload in a fresh
process for comparisons. --metrics saves timings, sizes and RSS measurements as
JSON. Memory sampling adds a small overhead even with --no-profile.

The default call exercises multiple scenarios and weighted damage curves,
variable split counts, per-asset unit costs, all uncertainty terms, asset/event
quantiles, asset EAD, and per-scenario totals. It needs no geometry or raster data.
"""

import argparse
import cProfile
import gc
import json
import platform
import pstats
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy
import pandas
import psutil

from snail.damages import (
    DirectDamageResult,
    PiecewiseLinearDamageCurve,
    calculate_direct_damages,
)

RETURN_PERIODS = (20, 50, 100, 200, 500, 1500)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
MIB = 1024**2
MEMORY_INTERVAL = 0.01


def positive_int(value: str) -> int:
    """Parse a positive integer for workload dimensions."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def make_workload(
    n_assets: int, n_scenarios: int, max_splits: int, seed: int
) -> tuple[pandas.DataFrame, dict[str, PiecewiseLinearDamageCurve]]:
    """Generate shared asset quantities and differing scenario/event depths."""
    rng = numpy.random.default_rng(seed)
    split_counts = rng.integers(1, max_splits + 1, size=n_assets)
    asset_ids = numpy.repeat(numpy.arange(n_assets), split_counts)
    quantities = rng.uniform(0.005, 0.2, size=len(asset_ids))
    unit_costs = rng.lognormal(numpy.log(300_000), 0.3, size=n_assets)
    base_depth = rng.gamma(2.0, 0.6, size=len(asset_ids))
    base_depth[rng.random(len(asset_ids)) < 0.25] = 0.0

    frames = []
    for scenario_index in range(n_scenarios):
        frame = pandas.DataFrame(
            {
                "asset_id": asset_ids,
                "scenario": f"scenario_{scenario_index}",
                "quantity": quantities,
                "unit_cost": unit_costs[asset_ids],
            }
        )
        for rp in RETURN_PERIODS:
            event_factor = 0.4 + numpy.log(rp) / numpy.log(RETURN_PERIODS[-1])
            scenario_factor = 1.0 + 0.15 * scenario_index
            frame[f"rp{rp}"] = base_depth * event_factor * scenario_factor
        frames.append(frame)

    depths = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
    curves = {
        name: PiecewiseLinearDamageCurve(
            pandas.DataFrame({"intensity": depths, "damage": fractions})
        )
        for name, fractions in {
            "low": [0.0, 0.0, 0.01, 0.08, 0.2, 0.5, 0.8],
            "central": [0.0, 0.01, 0.05, 0.2, 0.45, 0.8, 1.0],
            "high": [0.0, 0.03, 0.15, 0.4, 0.7, 1.0, 1.0],
        }.items()
    }
    return pandas.concat(frames, ignore_index=True), curves


def run(
    splits: pandas.DataFrame,
    curves: dict[str, PiecewiseLinearDamageCurve],
    n_samples: int,
    seed: int,
    ead_only: bool,
) -> DirectDamageResult:
    """Run the measured invocation, including validation and summarisation."""
    return calculate_direct_damages(
        splits,
        asset_id="asset_id",
        scenario="scenario",
        quantity="quantity",
        intensity_columns={rp: f"rp{rp}" for rp in RETURN_PERIODS},
        return_periods={rp: float(rp) for rp in RETURN_PERIODS},
        curves=curves,
        curve_weights={"low": 0.2, "central": 0.5, "high": 0.3},
        cost_mean="unit_cost",
        cost_asset_cv=0.2,
        cost_systemic_cv=0.25,
        hazard_cv={rp: 0.15 for rp in RETURN_PERIODS},
        eta_sd=0.15,
        n_samples=n_samples,
        quantiles=None if ead_only else QUANTILES,
        ead_quantiles=QUANTILES,
        seed=seed,
        aggregate_assets=True,
    )


def check_result(
    result: DirectDamageResult, n_assets: int, n_scenarios: int, ead_only: bool
) -> None:
    """Check that the workload produced complete, finite quantile summaries."""
    expected_rows = {
        "quantiles": None if ead_only else n_assets * n_scenarios * len(RETURN_PERIODS),
        "ead_quantiles": n_assets * n_scenarios,
        "total_quantiles": None if ead_only else n_scenarios * len(RETURN_PERIODS),
        "total_ead_quantiles": n_scenarios,
    }
    columns = [f"q_{round(q * 100):02d}" for q in QUANTILES]
    for field, n_rows in expected_rows.items():
        frame = getattr(result, field)
        if n_rows is None:
            assert frame is None, field
            continue
        assert frame is not None and len(frame) == n_rows, field
        values = frame[columns].to_numpy()
        assert numpy.isfinite(values).all() and (values >= 0).all(), field
        assert (numpy.diff(values, axis=1) >= 0).all(), field
    assert (result.total_ead_quantiles["n_assets"] == n_assets).all()


@contextmanager
def monitor_rss() -> Iterator[dict[str, int]]:
    """Sample resident process memory during the call, excluding setup/checks."""
    process = psutil.Process()
    before = process.memory_info().rss
    memory = {"rss_before_bytes": before, "peak_rss_bytes": before}
    stopped = threading.Event()

    def sample() -> None:
        while not stopped.wait(MEMORY_INTERVAL):
            memory["peak_rss_bytes"] = max(
                memory["peak_rss_bytes"], process.memory_info().rss
            )

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        yield memory
    finally:
        stopped.set()
        sampler.join()
        memory["rss_after_bytes"] = process.memory_info().rss
        memory["peak_rss_bytes"] = max(
            memory["peak_rss_bytes"], memory["rss_after_bytes"]
        )
        memory["peak_rss_increase_bytes"] = memory["peak_rss_bytes"] - before


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=positive_int, default=500)
    parser.add_argument("--scenarios", type=positive_int, default=3)
    parser.add_argument(
        "--splits",
        type=positive_int,
        default=8,
        help="maximum splits per asset (counts are sampled uniformly from 1 to this value)",
    )
    parser.add_argument("--samples", type=positive_int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ead-only", action="store_true", help="omit per-event quantile summaries"
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="skip the separate cProfile invocation",
    )
    parser.add_argument(
        "--profile", type=Path, help="save cProfile statistics to this file"
    )
    parser.add_argument(
        "--top", type=positive_int, default=20, help="rows per profile table"
    )
    parser.add_argument(
        "--metrics", type=Path, help="save timing and memory measurements as JSON"
    )
    args = parser.parse_args()
    if args.no_profile and args.profile is not None:
        parser.error("--profile cannot be combined with --no-profile")

    splits, curves = make_workload(args.assets, args.scenarios, args.splits, args.seed)
    print(
        f"Python {platform.python_version()}, NumPy {numpy.__version__}, "
        f"pandas {pandas.__version__}"
    )
    print(
        f"{args.assets:,} assets x {args.scenarios} scenarios = "
        f"{args.assets * args.scenarios:,} groups; {len(splits):,} split rows; "
        f"{len(RETURN_PERIODS)} events; {len(curves)} curves; {args.samples:,} samples"
    )
    input_bytes = int(splits.memory_usage(deep=True).sum())
    print(f"Input DataFrame: {input_bytes / MIB:.2f} MiB")
    print(
        f"Summaries: {'EAD only' if args.ead_only else 'events and EAD'}; totals enabled"
    )
    print("Timing calculate_direct_damages; setup excluded.", flush=True)

    profiler = None if args.no_profile else cProfile.Profile()
    gc.collect()
    with monitor_rss() as memory:
        started = time.perf_counter()
        result = run(splits, curves, args.samples, args.seed, args.ead_only)
        elapsed = time.perf_counter() - started
    print(f"Elapsed: {elapsed:.3f} s (RSS sampling enabled; no cProfile overhead)")
    print(
        f"Process RSS: before {memory['rss_before_bytes'] / MIB:.2f} MiB; "
        f"sampled peak {memory['peak_rss_bytes'] / MIB:.2f} MiB; "
        f"peak increase {memory['peak_rss_increase_bytes'] / MIB:.2f} MiB; "
        f"after {memory['rss_after_bytes'] / MIB:.2f} MiB"
    )

    check_result(result, args.assets, args.scenarios, args.ead_only)
    result_bytes = int(
        sum(
            frame.memory_usage(deep=True).sum()
            for frame in (
                result.quantiles,
                result.ead_quantiles,
                result.total_quantiles,
                result.total_ead_quantiles,
            )
            if frame is not None
        )
    )
    print(f"Result DataFrames: {result_bytes / MIB:.2f} MiB")
    print("Result checks passed. Total EAD summaries (first 10 scenarios):")
    print(result.total_ead_quantiles.head(10).to_string(index=False))
    if profiler is not None:
        print("Profiling a separate invocation; RSS sampling is stopped.", flush=True)
        started = time.perf_counter()
        profiler.runcall(run, splits, curves, args.samples, args.seed, args.ead_only)
        profiled_elapsed = time.perf_counter() - started
        print(
            f"Profiled elapsed: {profiled_elapsed:.3f} s (includes cProfile overhead)"
        )
    if args.metrics is not None:
        metrics = {
            "assets": args.assets,
            "scenarios": args.scenarios,
            "max_splits": args.splits,
            "samples": args.samples,
            "seed": args.seed,
            "events": len(RETURN_PERIODS),
            "curves": len(curves),
            "split_rows": len(splits),
            "ead_only": args.ead_only,
            "profiled": profiler is not None,
            "elapsed_seconds": elapsed,
            "memory_interval_seconds": MEMORY_INTERVAL,
            "input_bytes": input_bytes,
            "result_bytes": result_bytes,
            **memory,
        }
        if profiler is not None:
            metrics["profiled_seconds"] = profiled_elapsed
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\nMetrics saved to {args.metrics}")
    if profiler is not None:
        if args.profile is not None:
            args.profile.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(args.profile))
            print(f"\nProfile saved to {args.profile}")
        stats = pstats.Stats(profiler).strip_dirs()
        print("\nCumulative time (includes callees; rows overlap):")
        stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(args.top)
        print("\nDirect time (excludes callees):")
        stats.sort_stats(pstats.SortKey.TIME).print_stats(args.top)


if __name__ == "__main__":
    main()
