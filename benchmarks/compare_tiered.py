"""Validate comparability of tiered benchmark reports; never infer a speedup.

Example:
  python -m benchmarks.compare_tiered --baseline a-base.json \
      --candidate a-head.json --control a-head-repeat.json --output a-compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported benchmark schema")
    return report


def compare(
    baseline: dict[str, Any], candidate: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    """One paired comparison and one candidate same-revision control.

    Comparisons are rejected if hardware, dependency and fixture metadata differ.
    A control pair is one crude noise observation, not a confidence interval.
    """
    for label, report in (("baseline", baseline), ("candidate", candidate),
                          ("control", control)):
        if not report.get("environment", {}).get("commit"):
            raise ValueError(f"{label}: missing exact revision SHA")
        if report["fixture"]["mode"] != "synthetic_external_provider_no_teal_io":
            raise ValueError(f"{label}: only deterministic synthetic comparison supported")
    if candidate["environment"]["commit"] != control["environment"]["commit"]:
        raise ValueError("control must use the exact candidate revision")
    if baseline["environment"]["commit"] == candidate["environment"]["commit"]:
        raise ValueError("use separate SHAs for baseline/candidate; control is same-revision")

    common_fields = ("python", "platform", "cpu_count", "dependencies")
    for field in common_fields:
        values = [r["environment"][field] for r in (baseline, candidate, control)]
        if values[0] != values[1] or values[1] != values[2]:
            raise ValueError(f"mismatched execution environment: {field}")
    for field in ("mode", "dataset_id", "seed", "actual_units",
                  "synthetic_fixture_version"):
        values = [r["fixture"][field] for r in (baseline, candidate, control)]
        if len(set(values)) != 1:
            raise ValueError(f"mismatched fixture: {field}")
    for field in ("tier", "units", "page_size", "warm_repeats"):
        values = [r["execution"][field] for r in (baseline, candidate, control)]
        if len(set(values)) != 1:
            raise ValueError(f"mismatched execution configuration: {field}")

    a = baseline["execution"]["operations"]
    b = candidate["execution"]["operations"]
    c = control["execution"]["operations"]
    if not (a.keys() == b.keys() == c.keys()):
        raise ValueError("operation sets differ")
    rows = []
    for name in sorted(a):
        statuses = [r[name]["status"] for r in (a, b, c)]
        if len(set(statuses)) != 1:
            raise ValueError(f"{name}: mismatched capability status")
        if statuses[0] != "ok":
            rows.append({
                "operation": name, "status": statuses[0],
                "reason": a[name].get("reason"),
            })
            continue
        results = [r[name]["result"] for r in (a, b, c)]
        if not (results[0] == results[1] == results[2]):
            raise ValueError(f"{name}: result size/identity checks differ")
        warm = [r[name]["warm"] for r in (a, b, c)]
        first = [r[name]["first_call_proxy"] for r in (a, b, c)]
        if any(len(x["wall_ms"]) != x["repetitions"] for x in warm):
            raise ValueError(f"{name}: inconsistent warm repetitions")
        if any(x["state"] != "warm_same_process" for x in warm):
            raise ValueError(f"{name}: warm state label mismatch")
        if any(x["state"] != "first_call_same_process_not_cold" for x in first):
            raise ValueError(f"{name}: first-call proxy label mismatch")
        p50 = [x["p50_wall_ms"] for x in warm]
        p95 = [x["p95_wall_ms"] for x in warm]
        noise = abs(p50[2] - p50[1])
        rows.append({
            "operation": name,
            "status": "ok",
            "result": results[0],
            "baseline_p50_ms": p50[0],
            "candidate_p50_ms": p50[1],
            "control_p50_ms": p50[2],
            "baseline_p95_ms": p95[0],
            "candidate_p95_ms": p95[1],
            "control_p95_ms": p95[2],
            "candidate_minus_baseline_p50_ms": round(p50[1] - p50[0], 5),
            "no_change_p50_absolute_difference_ms": round(noise, 5),
            "python_allocation_peaks_bytes": [
                x["python_allocation_peak_bytes_max"] for x in warm
            ],
            "warm_repetitions_per_run": [x["repetitions"] for x in warm],
        })
    return {
        "schema_version": 1,
        "comparison_type": "paired_revisions_and_same_revision_control",
        "baseline_sha": baseline["environment"]["commit"],
        "candidate_sha": candidate["environment"]["commit"],
        "control_sha": control["environment"]["commit"],
        "fixture": baseline["fixture"],
        "environment": baseline["environment"],
        "tier": baseline["execution"]["tier"],
        "operations": rows,
        "interpretation": (
            "This infrastructure-only PR makes no application speed claim. "
            "A single same-revision control difference estimates run-to-run noise, "
            "not a confidence interval. Synthetic providers exclude TeAL I/O. "
            "Times include tracemalloc overhead; cold process, RSS, browser paint, "
            "network and SQL I/O are unmeasured."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(_load(args.baseline), _load(args.candidate), _load(args.control))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {args.output}: {len(report['operations'])} operations")


if __name__ == "__main__":
    main()
