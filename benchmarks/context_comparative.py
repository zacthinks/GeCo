"""Controlled, real-GeCo context benchmark for two pinned revisions.

Executed as a subprocess driver from an optional GitHub Actions job.
Uses the same immutable fixture snapshot (copied to disposable projects)
and the same driver source for both code revisions.

Command:
  python benchmarks/context_comparative.py orchestrate \
    --baseline-dir /path/to/baseline --candidate-dir /path/to/candidate \
    --output-dir /path/to/results --samples 20 --rounds 2

The old matched-200 path must load full context before selecting the exact
nearest 200 rows; it represents the best old-API composition of the *same*
visible result, not a claim that old UI originally sliced.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
import tracemalloc


BASELINE = "e5971fe1eb29d98f5f89bdb11bcb5389cab9e063"
# Exact PR #83 implementation head at measurement configuration.
CANDIDATE = os.environ.get("GECO_BENCHMARK_CANDIDATE_SHA", "c1620944cf82fd0fb3c83fab71d19ea68e93bd20")
SEED = 20261121
CAP = 200
SCENARIOS = {
    "10k_center": {"fixture": "10k_short", "focal": 5001, "level": 0, "window": 3},
    "10k_edge": {"fixture": "10k_short", "focal": 1, "level": 0, "window": 3},
    "50k_center": {"fixture": "50k_long", "focal": 25001, "level": 0, "window": 1},
    "50k_edge": {"fixture": "50k_long", "focal": 1, "level": 0, "window": 1},
}
VARIANTS = ("baseline_unlimited", "baseline_matched200", "candidate_bounded200")
STAGES = ("api", "reader")


def stable_digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def fixture_create(name: str, destination: Path) -> None:
    import pandas as pd
    from geometric_coder import GeometricCoder

    if name == "10k_short":
        count, docs, length = 10000, 25, 180
    elif name == "50k_long":
        count, docs, length = 50000, 1, 700
    else:
        raise ValueError(name)
    if destination.exists():
        raise RuntimeError(f"Fixture already exists: {destination}")
    subjects = ("interview", "classroom", "policy", "participant", "researcher",
                "observation", "curriculum", "discussion")
    rows = []
    for index in range(count):
        doc = f"document-{index // (count // docs):03d}"
        # Per-record deterministic text, with nontrivial sentence/metadata bytes.
        phrase = (
            f"Observation {index:06d} from {doc}. "
            f"The {subjects[(index * 17 + SEED) % len(subjects)]} described "
            f"changes over time, uncertainty in interpretation, and mixed evidence. "
        )
        text = (phrase * (1 + length // len(phrase)))[:length]
        rows.append({
            "document": doc,
            "sentence": index % (count // docs),
            "text": text,
            "category": ("health", "education", "economics")[index % 3],
            "wave": (index * 7 + SEED) % 8,
            "source": f"field-note-{index % 117}",
        })
    frame = pd.DataFrame.from_records(rows)
    GeometricCoder.create(
        project_dir=destination, data=frame, keys=["document", "sentence"],
        text="text", metadata=["category", "wave", "source"], geometries=None,
    )
    print(json.dumps({
        "fixture": name, "units": count, "documents": docs, "text_chars_per_unit": length,
        "seed": SEED, "project_dir": str(destination),
    }), flush=True)


def _peak_rss_bytes() -> int:
    # Linux ru_maxrss is in KiB. This is process-wide peak, not delta RSS.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    higher = math.ceil(position)
    return ordered[lower] + (ordered[higher] - ordered[lower]) * (position - lower)


def sample(args: argparse.Namespace) -> None:
    from geometric_coder import GeometricCoder
    import geometric_coder.storage.database as database_module

    is_candidate = args.variant.startswith("candidate")
    project = GeometricCoder.open(Path(args.project_dir))
    case = SCENARIOS[args.scenario]
    focal = case["focal"]
    level = case["level"]
    window = case["window"]
    focal_position = int(project.unit(focal)["row_position"])
    key_columns = list(project.metadata["key_columns"])

    if args.stage == "reader":
        from dash import dcc, html
        from dash._utils import to_json
        from geometric_coder.ui.app import _render_context_units, _metadata_table

    def select_old_nearest(rows: list[dict]) -> list[dict]:
        return sorted(
            sorted(rows, key=lambda row: (
                abs(int(row["row_position"]) - focal_position),
                int(row["row_position"]),
            ))[:CAP],
            key=lambda row: int(row["row_position"]),
        )

    def operation():
        if is_candidate:
            rows = project.context(focal, level=level, window=window, max_units=CAP)
        else:
            rows = project.context(focal, level=level, window=window)
            if args.variant == "baseline_matched200":
                rows = select_old_nearest(rows)
        rendered = None
        if args.stage == "reader":
            # Real GeCo UI rendering helper and Dash serialization; HTTP transport
            # and browser layout/paint are explicitly outside the measurement.
            children = _render_context_units(
                html, dcc, rows, focal_unit_id=focal,
                key_columns=key_columns, selected_unit_ids=[focal],
                checkbox_type="explore-span-checkbox", span_selection_enabled=True,
            )
            focal_metadata = project.unit(focal)["metadata"]
            payload = {
                "key": " / ".join(
                    f"{key}: {value}" for key, value in project.unit(focal)["user_key"].items()
                ),
                "context": children,
                "metadata": _metadata_table(html, focal_metadata),
            }
            rendered = to_json(payload)
        return rows, rendered

    # First invocation is after project-open, but NOT a cold-process claim.
    gc.collect()
    rss_before = _peak_rss_bytes()
    started = time.perf_counter_ns()
    first_rows, first_payload = operation()
    first_ms = (time.perf_counter_ns() - started) / 1e6
    del first_rows, first_payload

    warm_ms = []
    for _ in range(args.samples):
        gc.collect()  # outside the measurement
        started = time.perf_counter_ns()
        rows, payload = operation()
        warm_ms.append((time.perf_counter_ns() - started) / 1e6)
        del rows, payload
    rss_after_warm = _peak_rss_bytes()

    # Instrument decoding outside the wall-time measurements.
    decoded = {"rows": 0, "column_bytes": 0}
    original = database_module._unit_row

    def instrument(row):
        decoded["rows"] += 1
        decoded["column_bytes"] += sum(
            len(str(row[column]).encode("utf-8"))
            for column in ("text", "user_key_json", "metadata_json")
        )
        return original(row)

    database_module._unit_row = instrument
    try:
        rows, payload = operation()
    finally:
        database_module._unit_row = original
    ids = [int(row["unit_id"]) for row in rows]
    positions = [int(row["row_position"]) for row in rows]
    if ids != sorted(ids) or positions != sorted(positions):
        raise AssertionError("GeCo source order changed")
    if focal not in ids:
        raise AssertionError("Context excluded the focal unit")
    context_fingerprint = stable_digest([
        {"unit_id": row["unit_id"], "row_position": row["row_position"],
         "user_key": row["user_key"], "text": row["text"],
         "metadata": row["metadata"]}
        for row in rows
    ])
    render_fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest() if payload else None
    render_bytes = len(payload.encode("utf-8")) if payload else None
    visible_info = {
        "first_row_position": positions[0], "last_row_position": positions[-1],
        "focal_row_position": focal_position,
        "before_focal": sum(pos < focal_position for pos in positions),
        "after_focal": sum(pos > focal_position for pos in positions),
    }
    del rows, payload
    gc.collect()

    # Python traced allocations in a separate measurement, not in the timed path.
    # tracemalloc is not RSS.
    tracemalloc.start()
    gc.collect()
    rows, payload = operation()
    _, peak_allocated = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del rows, payload
    rss_after_all = _peak_rss_bytes()

    result = {
        "scenario": args.scenario, "fixture": case["fixture"],
        "variant": args.variant, "stage": args.stage, "round": args.round,
        "revision": CANDIDATE if is_candidate else BASELINE,
        "python": sys.version.split()[0], "seed": SEED, "cap": CAP,
        "focal_unit_id": focal, "level": level, "window": window,
        "samples": args.samples, "first_ms": first_ms,
        "warm_ms": warm_ms, "p50_ms": _percentile(warm_ms, .50),
        "p95_ms": _percentile(warm_ms, .95),
        "min_ms": min(warm_ms), "max_ms": max(warm_ms),
        "returned_rows": len(ids), "decoded_rows": decoded["rows"],
        "decoded_text_key_metadata_bytes": decoded["column_bytes"],
        "rendered_payload_bytes": render_bytes,
        "returned_digest": context_fingerprint,
        "rendered_digest": render_fingerprint,
        "tracemalloc_peak_bytes": peak_allocated,
        "peak_rss_before_bytes": rss_before,
        "peak_rss_after_warm_bytes": rss_after_warm,
        "peak_rss_after_all_bytes": rss_after_all,
        "visibility": visible_info,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "scenario": args.scenario, "variant": args.variant, "stage": args.stage,
        "round": args.round, "first_ms": round(first_ms, 3),
        "p50_ms": round(result["p50_ms"], 3),
        "p95_ms": round(result["p95_ms"], 3),
        "returned": len(ids), "decoded": decoded["rows"],
        "decoded_column_bytes": decoded["column_bytes"],
        "rendered_bytes": render_bytes,
    }), flush=True)


def launch(args: argparse.Namespace, revision_dir: Path, *flags: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(revision_dir / "src")
    env.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    })
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *flags],
                   check=True, env=env)


def orchestrate(args: argparse.Namespace) -> None:
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    baseline = Path(args.baseline_dir).resolve()
    candidate = Path(args.candidate_dir).resolve()
    fixture_root = root / "fixtures"
    fixture_root.mkdir(exist_ok=True)
    output = root / "runs"
    output.mkdir(exist_ok=True)
    for revision_dir, expected in ((baseline, BASELINE), (candidate, CANDIDATE)):
        sha = subprocess.check_output(
            ["git", "-C", str(revision_dir), "rev-parse", "HEAD"], text=True
        ).strip()
        if sha != expected:
            raise RuntimeError(f"Incorrect pinned revision: {sha} != {expected}")
    for fixture in ("10k_short", "50k_long"):
        destination = fixture_root / fixture
        launch(args, baseline, "fixture", "--name", fixture,
               "--destination", str(destination))
    ordering = (
        [("baseline_unlimited", baseline), ("baseline_matched200", baseline),
         ("candidate_bounded200", candidate)],
        [("candidate_bounded200", candidate),
         ("baseline_matched200", baseline), ("baseline_unlimited", baseline)],
    )
    for round_number in range(1, args.rounds + 1):
        sequence = ordering[(round_number - 1) % 2]
        for scenario, case in SCENARIOS.items():
            for variant, revision_dir in sequence:
                for stage in STAGES:
                    source = fixture_root / case["fixture"]
                    scratch = root / "scratch" / f"{scenario}_{variant}_{stage}_r{round_number}"
                    scratch.parent.mkdir(exist_ok=True)
                    shutil.copytree(source, scratch)
                    dest = output / f"{scenario}_{variant}_{stage}_r{round_number}.json"
                    launch(
                        args, revision_dir, "sample",
                        "--scenario", scenario, "--variant", variant,
                        "--stage", stage, "--round", str(round_number),
                        "--samples", str(args.samples), "--project-dir", str(scratch),
                        "--output", str(dest),
                    )
                    shutil.rmtree(scratch)
    results = [json.loads(path.read_text()) for path in sorted(output.glob("*.json"))]
    expected_count = args.rounds * len(SCENARIOS) * len(VARIANTS) * len(STAGES)
    if len(results) != expected_count:
        raise AssertionError(f"Expected {expected_count} results; got {len(results)}")
    # Matched old/new must return identical row data and Dash payloads.
    for scenario in SCENARIOS:
        for stage in STAGES:
            rows = [r for r in results if r["scenario"] == scenario and r["stage"] == stage]
            olds = [r for r in rows if r["variant"] == "baseline_matched200"]
            news = [r for r in rows if r["variant"] == "candidate_bounded200"]
            unlimited = [r for r in rows if r["variant"] == "baseline_unlimited"]
            for old, new, full in zip(sorted(olds, key=lambda r:r["round"]),
                                      sorted(news, key=lambda r:r["round"]),
                                      sorted(unlimited, key=lambda r:r["round"])):
                assert old["returned_rows"] == new["returned_rows"] == CAP
                assert old["returned_digest"] == new["returned_digest"], scenario
                assert old["rendered_digest"] == new["rendered_digest"], (scenario,stage)
                assert full["returned_rows"] > CAP
                assert old["decoded_rows"] == full["decoded_rows"]
                # The reader stage additionally loads focal unit/key metadata twice.
                assert new["decoded_rows"] == CAP + (2 if stage == "reader" else 0)
    raw_csv = root / "warm_samples.csv"
    with raw_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scenario", "stage", "variant", "round", "sample", "wall_ms"))
        for r in results:
            for index, elapsed in enumerate(r["warm_ms"], 1):
                writer.writerow((r["scenario"], r["stage"], r["variant"],
                                 r["round"], index, f"{elapsed:.6f}"))
    compact = []
    for case in SCENARIOS:
        for stage in STAGES:
            for variant in VARIANTS:
                selected = [r for r in results if
                            r["scenario"] == case and r["stage"] == stage
                            and r["variant"] == variant]
                all_warm = [x for row in selected for x in row["warm_ms"]]
                representative = selected[0]
                compact.append({
                    "scenario": case, "stage": stage, "variant": variant,
                    "warm_n": len(all_warm), "rounds": args.rounds,
                    "p50_ms": _percentile(all_warm, .50),
                    "p95_ms": _percentile(all_warm, .95),
                    "first_ms": [r["first_ms"] for r in selected],
                    "returned_rows": representative["returned_rows"],
                    "decoded_rows": representative["decoded_rows"],
                    "decoded_column_bytes": representative["decoded_text_key_metadata_bytes"],
                    "rendered_payload_bytes": representative["rendered_payload_bytes"],
                    "tracemalloc_peak_bytes": [r["tracemalloc_peak_bytes"] for r in selected],
                    "peak_process_rss_bytes": [r["peak_rss_after_all_bytes"] for r in selected],
                })
    environment = {
        "baseline": BASELINE, "candidate": CANDIDATE, "seed": SEED,
        "cap": CAP, "scenarios": SCENARIOS, "rounds": args.rounds,
        "warm_samples_per_round": args.samples,
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version, "cpu_count": os.cpu_count(),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "method": "GeometricCoder.context API; additionally real Dash UI context renderer and JSON serialization",
        "memory_notes": "tracemalloc peak is Python allocations; ru_maxrss is process-wide high-water mark, not per-operation RSS delta",
        "first_use_notes": "first invocation after GeometricCoder.open in a new process; not guaranteed cold OS cache",
        "limitations": ["No browser paint or HTTP transport", "Synthetic but real persisted GeCo projects",
                        "Old matched output is post-sliced outside the old public API"],
    }
    (root / "summary.json").write_text(json.dumps({
        "environment": environment, "comparisons": compact,
    }, indent=2) + "\n")
    (root / "raw_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("MATCHED OUTPUT DIGESTS VERIFIED ON BOTH REVISIONS", flush=True)
    print(json.dumps(compact, indent=2), flush=True)


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture")
    fixture.add_argument("--name", required=True)
    fixture.add_argument("--destination", type=Path, required=True)
    sampling = commands.add_parser("sample")
    sampling.add_argument("--scenario", choices=SCENARIOS, required=True)
    sampling.add_argument("--variant", choices=VARIANTS, required=True)
    sampling.add_argument("--stage", choices=STAGES, required=True)
    sampling.add_argument("--round", type=int, required=True)
    sampling.add_argument("--samples", type=int, default=20)
    sampling.add_argument("--project-dir", required=True)
    sampling.add_argument("--output", required=True)
    orchestrator = commands.add_parser("orchestrate")
    orchestrator.add_argument("--baseline-dir", required=True)
    orchestrator.add_argument("--candidate-dir", required=True)
    orchestrator.add_argument("--output-dir", required=True)
    orchestrator.add_argument("--samples", type=int, default=20)
    orchestrator.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.command == "fixture":
        fixture_create(args.name, args.destination)
    elif args.command == "sample":
        sample(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    cli()
