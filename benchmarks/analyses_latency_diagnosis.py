"""Isolated Analyses latency, memory and cProfile diagnostic runner.

Each invocation builds the deterministic 300k acceptance fixture in a fresh
process. --mode untraced measures cold and warm compute; --mode traced measures
Python allocation peaks separately. Chart and Plotly hashes detect changes.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import resource
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

from benchmarks.analyses_300k import (
    Project, _assert_chart_values, _expected_cell_statistics, cases, populate,
)
from geometric_coder.analyses import analysis_figure, compute_analysis


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str,
    ).encode()).hexdigest()


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run(args):
    setup_start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "analyses.sqlite"
        expected = populate(path, args.units)
        project = Project(path)
        fixture_seconds = time.perf_counter() - setup_start
        data = {
            "sha": os.environ.get("GITHUB_SHA", "unknown"),
            "source_revision": os.environ.get("SOURCE_SHA", "unknown"),
            "units": args.units,
            "round": args.round,
            "mode": args.mode,
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "fixture_seed": "deterministic i modulo 20/25/17/100; no PRNG",
            "fixture_seconds": round(fixture_seconds, 6),
            "rss_after_fixture_mib": round(rss_mib(), 3),
            "charts": {},
        }
        selected = [args.case] if args.case else list(cases())
        oracle = None
        for name in selected:
            spec = cases()[name]
            samples = []
            repeats = args.repeats if args.mode == "untraced" and not args.profile else 1
            for rep in range(repeats):
                rss_before = rss_mib()
                if args.mode == "traced":
                    tracemalloc.start()
                profile = cProfile.Profile() if args.profile else None
                if profile:
                    profile.enable()
                started = time.perf_counter_ns()
                result = compute_analysis(project, spec)
                compute_seconds = (time.perf_counter_ns() - started) / 1e9
                if profile:
                    profile.disable()
                peak = None
                if args.mode == "traced":
                    _, peak_bytes = tracemalloc.get_traced_memory()
                    peak = round(peak_bytes / 1024**2, 3)
                    tracemalloc.stop()
                rss_after_compute = rss_mib()
                started = time.perf_counter_ns()
                fig = analysis_figure(result)
                figure_seconds = (time.perf_counter_ns() - started) / 1e9
                started = time.perf_counter_ns()
                fig_json = fig.to_json()
                serialization_seconds = (time.perf_counter_ns() - started) / 1e9
                if oracle is None:
                    oracle = _expected_cell_statistics(args.units)
                _assert_chart_values(name, result, args.units, oracle)
                assert result["coverage"]["total"] == args.units
                assert result["coverage"]["observed"][0] == expected["score"]
                sample = {
                    "phase": "cold" if rep == 0 else "warm",
                    "compute_seconds": round(compute_seconds, 6),
                    "figure_seconds": round(figure_seconds, 6),
                    "to_json_seconds": round(serialization_seconds, 6),
                    "tracemalloc_peak_mib": peak,
                    "rss_before_mib": round(rss_before, 3),
                    "rss_after_compute_mib": round(rss_after_compute, 3),
                    "result_sha256": digest(result),
                    "figure_sha256": hashlib.sha256(fig_json.encode()).hexdigest(),
                    "result_bytes": len(json.dumps(result).encode()),
                    "plotly_bytes": len(fig_json.encode()),
                }
                samples.append(sample)
                if profile:
                    profile_path = args.output.with_suffix("." + name + ".prof")
                    profile.dump_stats(str(profile_path))
                    summary = args.output.with_suffix("." + name + ".profile.txt")
                    with summary.open("w") as stream:
                        pstats.Stats(profile, stream=stream).sort_stats(
                            "cumulative",
                        ).print_stats(45)
                        pstats.Stats(profile, stream=stream).sort_stats(
                            "tottime",
                        ).print_stats(25)
                print(json.dumps({
                    "event": "sample", "units": args.units,
                    "round": args.round, "mode": args.mode,
                    "chart": name, **sample,
                }), flush=True)
            data["charts"][name] = samples
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "event": "complete", "units": args.units, "round": args.round,
            "mode": args.mode, "fixture_seconds": data["fixture_seconds"],
            "charts": len(data["charts"]),
        }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--mode", choices=["untraced", "traced"], required=True)
    parser.add_argument("--case", choices=list(cases()))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    assert opts.units > 0 and opts.repeats > 0
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    run(opts)
