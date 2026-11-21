"""Small, dependency-light reporting helpers for tiered backend benchmarks.

Wall clock includes tracemalloc instrumentation overhead. Per-call tracing is
restarted after fixture setup; these allocations are not process RSS.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any


def environment() -> dict[str, Any]:
    def version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
            timeout=3,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "commit": commit or os.environ.get("GITHUB_SHA"),
        "ref": os.environ.get("GITHUB_REF_NAME"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "dependencies": {
            name: version(name)
            for name in ("geometric-coder", "numpy", "pandas", "scipy", "dash",
                         "plotly", "text-analysis-lab")
        },
    }


def _percentile_95(values: list[float]) -> float | None:
    # Report a sample percentile only at >=20 warm observations.
    if len(values) < 20:
        return None
    ordered = sorted(values)
    return round(ordered[math.ceil(0.95 * len(ordered)) - 1], 5)


def measure(
    name: str,
    action: Callable[[], Any],
    validate: Callable[[Any], None],
    describe: Callable[[Any], dict[str, Any]],
    *,
    warm_repeats: int,
) -> dict[str, Any]:
    """Validate every result; measure first-call proxy separately from warm.

    The first call shares the fixture process and is NOT a cold-process result.
    Describing payload/counts and deterministic assertions are outside timing.
    """
    if warm_repeats < 1:
        raise ValueError("warm_repeats must be positive")
    wall: list[float] = []
    peaks: list[int] = []
    details: dict[str, Any] | None = None
    for _ in range(warm_repeats + 1):
        if tracemalloc.is_tracing():
            raise RuntimeError("Benchmark requires exclusive tracemalloc control")
        tracemalloc.start()
        try:
            baseline, _ = tracemalloc.get_traced_memory()
            start = time.perf_counter()
            result = action()
            elapsed_ms = (time.perf_counter() - start) * 1000
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        validate(result)
        current_details = describe(result)
        if details is None:
            details = current_details
        else:
            assert details == current_details, (name, "result shape changed across calls")
        wall.append(round(elapsed_ms, 5))
        peaks.append(max(0, peak - baseline))
    assert details is not None
    return {
        "status": "ok",
        "operation": name,
        "first_call_proxy": {
            "state": "first_call_same_process_not_cold",
            "wall_ms": wall[0],
            "python_allocation_peak_bytes": peaks[0],
        },
        "warm": {
            "state": "warm_same_process",
            "repetitions": warm_repeats,
            "wall_ms": wall[1:],
            "p50_wall_ms": round(statistics.median(wall[1:]), 5),
            "p95_wall_ms": _percentile_95(wall[1:]),
            "p95_status": "measured" if warm_repeats >= 20 else "insufficient_samples",
            "python_allocation_peak_bytes_max": max(peaks[1:]),
        },
        "result": details,
        "metrics": {
            "wall_clock": "measured_with_tracemalloc_overhead",
            "python_allocations": "per_call_tracemalloc_peak_above_start_baseline",
            "process_rss": "not_measured",
            "sql_io": "not_measured",
            "network_bytes": "not_measured",
            "browser_paint": "not_measured",
        },
    }


def skipped(name: str, reason: str) -> dict[str, str]:
    return {"status": "unavailable", "operation": name, "reason": reason}


def save_report(report: dict[str, Any], path: Path | None) -> str:
    data = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
    return data
