"""Controlled, full Develop prediction-plot benchmark across two GeCo revisions.

Only the candidate worktree needs this driver; each worker imports the selected
revision's package through PYTHONPATH. No model fitting or browser painting is timed.

Commands (after installing GeCo's base dependencies in one Python environment):
    python benchmarks/compare_prediction_plot_workflow.py prepare --root /tmp/geco-plots
    python benchmarks/compare_prediction_plot_workflow.py compare \
        --root /tmp/geco-plots --baseline /path/to/base --candidate /path/to/head

The SQLite fixture is a disposable GeCo schema, with persisted classifier fits
and precomputed scores (not actual trained estimators). Observation enumeration,
fit/spec lookup, persisted score reads, annotations, compaction, pagination,
Plotly construction, and click index all run through real GeCo methods.
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
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any


SEED = 20261121
BASE_SHA = "e5971fe1eb29d98f5f89bdb11bcb5389cab9e063"
FIXTURES = {5000: (2048, 768), 50000: (1024, 512)}
SPAN_COUNT = 12
ACTIVE_TEACHING = 4
DELETED_TEACHING = 2
PAGE_SIZE = 5000


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fixture_dir(root: Path, n: int) -> Path:
    return root / "fixtures" / f"n{n}"


def _text(index: int, length: int) -> str:
    phrases = (
        "The participant reflects on access to services and a changing neighborhood. ",
        "The interview discusses schooling, migration and experiences of institutions. ",
        "The respondent contrasts informal support with a formal policy process. ",
        "The account describes work routines, uncertainty, and social relationships. ",
    )
    preface = f"Interview {index // 15:06d}, passage {index:06d}. "
    sample = phrases[index % len(phrases)] + phrases[(index * 7 + 1) % len(phrases)]
    return (preface + sample * (1 + length // len(sample)))[:length]


def prepare(root: Path, n: int) -> dict[str, Any]:
    from geometric_coder.storage.database import ProjectDatabase

    text_length, meta_length = FIXTURES[n]
    folder = _fixture_dir(root, n)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "project.sqlite3"
    if path.exists():
        path.unlink()
    db = ProjectDatabase(path)
    db.initialize(
        modality="text", text_column="text",
        key_columns=["document", "paragraph", "segment"],
        metadata_columns=["region", "wave", "note"],
    )
    stamp = "2026-11-21T00:00:00Z"
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO codes(code_id,name,definition,created_at) VALUES(1,?,?,?)",
            ("benchmark code", "synthetic plot benchmark", stamp),
        )
        conn.execute(
            """INSERT INTO geometries(
                geometry_id,name,class_name,modality,public,config_json,
                dependency_names_json,matrix_path,matrix_sparse,state_path,
                supports_query,supports_text_transform,storage_kind,external_ref_json,created_at
            ) VALUES (1,'fixture','CountGeometry','text',1,'{}','[]','unused.npz',0,
                     NULL,0,0,'local',NULL,?)""",
            (stamp,),
        )
        for fit in range(1, 4):
            conn.execute(
                """INSERT INTO classifier_specs(
                    classifier_spec_id,code_id,name,geometry_id,algorithm,
                    hyperparameters_json,status,created_at
                ) VALUES (?,1,?,1,'logistic_regression','{}','active',?)""",
                (fit, f"Classifier {fit}", stamp),
            )
            conn.execute(
                """INSERT INTO classifier_fits(
                    classifier_fit_id,classifier_spec_id,code_id,classifier_name,
                    geometry_id,algorithm,hyperparameters_json,training_snapshot_json,
                    artifact_path,score_kind,created_at
                ) VALUES (?,?,1,?,1,'logistic_regression','{}','{}',
                          'not-loaded.joblib','probability',?)""",
                (fit, fit, f"Classifier {fit}", stamp),
            )

        chunk = 1000
        for start in range(1, n + 1, chunk):
            observations = []
            units = []
            for identifier in range(start, min(n + 1, start + chunk)):
                passage = _text(identifier, text_length)
                key = json.dumps(
                    {
                        "document": f"doc-{identifier // 150:05d}",
                        "paragraph": (identifier // 5) % 30,
                        "segment": identifier % 5,
                    },
                    separators=(",", ":"),
                )
                metadata = json.dumps(
                    {
                        "region": ("north", "south", "east", "west")[identifier % 4],
                        "wave": 2020 + (identifier % 6),
                        "note": ("Interview memo " + str(identifier) + " ")
                        + ("m" * max(0, meta_length - 75)),
                    },
                    separators=(",", ":"),
                )
                observations.append((identifier, "atomic", passage, stamp))
                units.append((identifier, identifier, identifier - 1, passage, key, metadata))
            conn.executemany(
                "INSERT INTO observations(observation_id,kind,text,created_at)"
                " VALUES(?,?,?,?)", observations,
            )
            conn.executemany(
                "INSERT INTO units(unit_id,observation_id,row_position,text,"
                "user_key_json,metadata_json) VALUES(?,?,?,?,?,?)", units,
            )

        for j in range(SPAN_COUNT):
            observation_id = n + 1 + j
            first = 2 + j * 9
            conn.execute(
                "INSERT INTO observations VALUES (?,?,?,?)",
                (observation_id, "span", _text(observation_id, text_length), stamp),
            )
            conn.executemany(
                "INSERT INTO span_members(span_observation_id,unit_id,member_order)"
                " VALUES(?,?,?)",
                [(observation_id, first, 0), (observation_id, first + 1, 1)],
            )
        for j in range(ACTIVE_TEACHING + DELETED_TEACHING):
            oid = n + 1 + SPAN_COUNT + j
            status = "active" if j < ACTIVE_TEACHING else "deleted"
            conn.execute(
                "INSERT INTO observations VALUES (?,?,?,?)",
                (oid, "teaching_example", _text(oid, text_length), stamp),
            )
            conn.execute(
                """INSERT INTO teaching_examples(
                    observation_id,code_id,label,status,note,created_at,updated_at
                ) VALUES (?,1,?,?,?, ?,?)""",
                (oid, "positive" if j % 2 else "negative",
                 status, "authored teaching example", stamp, stamp),
            )
        all_ids = list(range(1, n + SPAN_COUNT + ACTIVE_TEACHING + DELETED_TEACHING + 1))
        for fit in range(1, 4):
            for start in range(0, len(all_ids), chunk):
                score_rows = []
                for oid in all_ids[start:start + chunk]:
                    position = (
                        math.sin(oid * (0.011 + fit * 0.003))
                        + math.cos(oid * (0.037 + fit * 0.007))
                    ) / 4 + 0.5
                    probability = 0.04 + 0.92 * position
                    margin = math.log(probability / (1 - probability))
                    score_rows.append((
                        fit, oid, "positive" if probability >= 0.5 else "negative",
                        probability, margin, abs(probability - 0.5),
                    ))
                conn.executemany(
                    """INSERT INTO classifier_predictions(
                        classifier_fit_id,observation_id,predicted_label,
                        probability,decision_score,uncertainty
                    ) VALUES(?,?,?,?,?,?)""",
                    score_rows,
                )
        for index in range(1, n + 1, max(1, n // 70)):
            conn.execute(
                """INSERT INTO annotation_events(
                    observation_id,code_id,value,origin,created_at
                ) VALUES (?,1,?,'human_direct',?)""",
                (index, ("positive", "negative", "unsure")[index % 3], stamp),
            )
        conn.execute(
            """INSERT INTO annotation_events(
                observation_id,code_id,value,origin,created_at
            ) VALUES (?,1,'positive','human_direct',?)""",
            (n + 1, stamp),
        )

    with db.connect() as conn:
        atomic = conn.execute(
            """SELECT SUM(LENGTH(CAST(o.text AS BLOB))) AS text_bytes,
                      SUM(LENGTH(CAST(u.metadata_json AS BLOB))) AS metadata_bytes,
                      SUM(LENGTH(CAST(u.user_key_json AS BLOB))) AS key_bytes
               FROM observations o JOIN units u ON u.observation_id=o.observation_id"""
        ).fetchone()
        derived = conn.execute(
            """SELECT SUM(LENGTH(CAST(text AS BLOB))) AS text_bytes
               FROM observations WHERE kind!='atomic'"""
        ).fetchone()
    stats = {
        "rows_atomic": n, "spans": SPAN_COUNT,
        "active_teaching": ACTIVE_TEACHING,
        "deleted_teaching": DELETED_TEACHING,
        "seed": SEED, "text_target_bytes": text_length,
        "metadata_target_bytes": meta_length,
        "atomic_text_bytes": int(atomic["text_bytes"]),
        "atomic_metadata_bytes": int(atomic["metadata_bytes"]),
        "atomic_key_bytes": int(atomic["key_bytes"]),
        "derived_text_bytes": int(derived["text_bytes"]),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "db_size_bytes": path.stat().st_size,
        "schema_created_with_revision": "candidate",
        "fit_source": "persisted precomputed synthetic scores; no estimator trained",
    }
    (folder / "fixture.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def _pipeline(project: Any, fit_count: int, variant: str, score_type: str = "probability"):
    from geometric_coder.ui.app import (
        _build_prediction_geometry_figure,
        _compact_prediction_geometry,
        _prediction_geometry_page_index,
        _prediction_geometry_visible_points,
        deterministic_page_with_inclusions,
    )

    start = time.perf_counter_ns()
    kw = {"code_id": 1, "classifier_fit_ids": list(range(1, fit_count + 1)),
          "score_type": score_type}
    if variant == "candidate":
        kw["compact_for_ui"] = True
    geometry = project.classifier_prediction_geometry(**kw)
    stage_geometry = time.perf_counter_ns()
    geometry = _compact_prediction_geometry(geometry)
    stage_compact = time.perf_counter_ns()
    visible = _prediction_geometry_visible_points(geometry, ["spans", "teaching"])
    page = deterministic_page_with_inclusions(
        len(visible), page_size=PAGE_SIZE, page_index=0,
        seed=0, include_positions=(),
    )
    selected = [visible[int(position)] for position in page.positions]
    figure = _build_prediction_geometry_figure(
        geometry=geometry, points=selected, page_index=page.page_index,
    )
    click_index = _prediction_geometry_page_index(figure)
    end = time.perf_counter_ns()
    durations = {
        "geometry_ms": (stage_geometry - start) / 1e6,
        "compaction_ms": (stage_compact - stage_geometry) / 1e6,
        "figure_ms": (end - stage_compact) / 1e6,
        "total_ms": (end - start) / 1e6,
    }
    return durations, geometry, figure, click_index


def _rss_hwm_bytes() -> int | None:
    path = Path("/proc/self/status")
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def _fingerprint(geometry: dict[str, Any], figure: Any, click_index: dict):
    points = geometry["points"]
    # Stable serialization exercises all observation identities and coordinates,
    # including derived observation labels, not just the 5k figure page.
    point_string = json.dumps(
        [(p["observation_id"], p["unit_id"], p["kind"], p["display_key"],
          p["assignment"], p["x"], p["y"]) for p in points],
        ensure_ascii=False, separators=(",", ":"),
    )
    figure_string = figure.to_json()
    click_string = json.dumps(sorted(
        (int(key), int(location[0]), int(location[1]))
        for key, location in click_index.items()
    ))
    return {
        "method": geometry["method"], "point_count": len(points),
        "point_sha256": _digest(point_string),
        "figure_sha256": _digest(figure_string),
        "figure_payload_bytes": len(figure_string.encode("utf-8")),
        "click_sha256": _digest(click_string),
        "figure_click_count": len(click_index),
        "first_observation_id": int(points[0]["observation_id"]),
        "last_observation_id": int(points[-1]["observation_id"]),
    }


def _read_volume(fixture: dict[str, Any], variant: str, fit_count: int):
    # Logical UTF-8 text/metadata fields selected by the real SQL queries.
    # This is NOT disk bytes read, nor a measurement of SQLite page-cache I/O.
    n = fixture["rows_atomic"]
    spans = fixture["spans"]
    text_full = fixture["atomic_text_bytes"] + fixture["derived_text_bytes"]
    score_join_text = text_full * fit_count
    list_text = 0 if variant == "candidate" else (
        fixture["atomic_text_bytes"]
        + fixture["derived_text_bytes"]
        - fixture["deleted_teaching"] * fixture["text_target_bytes"]
    )
    list_metadata = (
        0 if variant == "candidate" else fixture["atomic_metadata_bytes"]
    )
    # Full display key generation makes 2 unit reads + 1 observation per span.
    # Unit text is duplicated from the corresponding atomic observation.
    extra_text = (
        0 if variant == "candidate" else
        3 * spans * fixture["text_target_bytes"]
    )
    return {
        "logical_selected_text_bytes": score_join_text + list_text + extra_text,
        "logical_selected_metadata_bytes": list_metadata,
        "logical_total_text_metadata_bytes": (
            score_join_text + list_text + list_metadata + extra_text
        ),
        "plot_observation_rows": n + spans + fixture["active_teaching"],
        "prediction_rows": (
            (n + spans + fixture["active_teaching"] + fixture["deleted_teaching"])
            * fit_count
        ),
        "estimated_metadata_json_decodes": 0 if variant == "candidate" else (
            n + 2 * spans
        ),
        "estimated_source_key_json_decodes": (
            n + 2 * spans if variant == "candidate" else n + 2 * spans
        ),
        "read_bytes_scope": "logical selected UTF-8 text/metadata SQL columns; not physical I/O",
    }


def worker(args: argparse.Namespace) -> None:
    from geometric_coder.project import GeometricCoder

    fixture_dir = _fixture_dir(args.root, args.size)
    project = GeometricCoder(fixture_dir)
    fixture = json.loads((fixture_dir / "fixture.json").read_text())
    first, geometry, figure, click = _pipeline(project, args.fits, args.variant)
    signature = _fingerprint(geometry, figure, click)
    # Validate decision-score semantics once per process, separately from warm
    # probability timing. Both revisions must return the same exact fingerprints.
    _, decision_geometry, decision_figure, decision_click = _pipeline(
        project, args.fits, args.variant, score_type="decision_score",
    )
    decision_signature = _fingerprint(
        decision_geometry, decision_figure, decision_click,
    )
    del geometry, figure, click, decision_geometry, decision_figure, decision_click
    samples = []
    for index in range(args.repetitions):
        gc.collect()
        durations, geometry, figure, click = _pipeline(
            project, args.fits, args.variant,
        )
        samples.append({"sample": index, **durations})
        del geometry, figure, click
    gc.collect()
    tracemalloc.start()
    _, measured_geometry, measured_figure, measured_click = _pipeline(
        project, args.fits, args.variant,
    )
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del measured_geometry, measured_figure, measured_click
    result = {
        "size": args.size, "fits": args.fits,
        "variant": args.variant, "round": args.round,
        "first_after_import_ms": first, "warm": samples,
        "correctness_probability": signature,
        "correctness_decision_score": decision_signature,
        "tracemalloc_peak_bytes_separate_pass": python_peak,
        "process_rss_high_water_bytes": _rss_hwm_bytes(),
        "logical_read_estimates": _read_volume(fixture, args.variant, args.fits),
        "fixture_sha256": fixture["sha256"],
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": __import__("numpy").__version__,
        "plotly": __import__("plotly").__version__,
        "pid": os.getpid(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def _percentile(numbers: list[float], p: float) -> float:
    values = sorted(numbers)
    position = (len(values) - 1) * p / 100
    a = math.floor(position)
    b = math.ceil(position)
    return values[a] + (values[b] - values[a]) * (position - a)


def compare(args: argparse.Namespace) -> None:
    reports = args.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    all_data = []
    for n in (5000, 50000):
        fixture = _fixture_dir(args.root, n) / "fixture.json"
        if not fixture.exists():
            raise RuntimeError(f"Missing fixture: {fixture}. Run prepare first.")
    for n in (5000, 50000):
        for fits in (1, 2, 3):
            for round_number in range(1, args.rounds + 1):
                order = ("baseline", "candidate") if round_number % 2 else (
                    "candidate", "baseline"
                )
                paired = {}
                for variant in order:
                    src = args.baseline if variant == "baseline" else args.candidate
                    dest = reports / f"n{n}_f{fits}_round{round_number}_{variant}.json"
                    env = dict(os.environ)
                    env["PYTHONPATH"] = str(src.resolve() / "src")
                    env["OPENBLAS_NUM_THREADS"] = "1"
                    env["OMP_NUM_THREADS"] = "1"
                    env["MKL_NUM_THREADS"] = "1"
                    env["NUMEXPR_NUM_THREADS"] = "1"
                    env["PYTHONHASHSEED"] = "0"
                    command = [
                        sys.executable, str(Path(__file__).resolve()),
                        "worker", "--root", str(args.root), "--size", str(n),
                        "--fits", str(fits), "--variant", variant,
                        "--round", str(round_number),
                        "--repetitions", str(args.repetitions),
                        "--output", str(dest),
                    ]
                    print("RUN", n, fits, round_number, variant, flush=True)
                    subprocess.run(command, env=env, check=True)
                    data = json.loads(dest.read_text())
                    all_data.append(data)
                    paired[variant] = data
                for key in ("correctness_probability", "correctness_decision_score"):
                    if paired["baseline"][key] != paired["candidate"][key]:
                        raise AssertionError(
                            f"Result mismatch at n={n}, fits={fits}, round={round_number}:"
                            f" {key}\nBASE {paired['baseline'][key]}"
                            f"\nCAND {paired['candidate'][key]}"
                        )
    rows = []
    summaries = []
    for n in (5000, 50000):
        for fits in (1, 2, 3):
            group = [x for x in all_data if x["size"] == n and x["fits"] == fits]
            summary = {"size": n, "fits": fits, "warm_samples_per_variant":
                       args.rounds * args.repetitions,
                       "independent_rounds": args.rounds}
            for variant in ("baseline", "candidate"):
                members = [x for x in group if x["variant"] == variant]
                firsts = [x["first_after_import_ms"]["total_ms"] for x in members]
                summary[variant + "_first_total_ms"] = firsts
                summary[variant + "_tracemalloc_peak_bytes"] = [
                    x["tracemalloc_peak_bytes_separate_pass"] for x in members
                ]
                summary[variant + "_rss_hwm_bytes"] = [
                    x["process_rss_high_water_bytes"] for x in members
                ]
                summary[variant + "_read"] = members[0]["logical_read_estimates"]
                for stage in ("geometry_ms", "compaction_ms", "figure_ms", "total_ms"):
                    samples = [sample[stage] for m in members for sample in m["warm"]]
                    summary[f"{variant}_{stage}_p50"] = _percentile(samples, 50)
                    summary[f"{variant}_{stage}_p95"] = _percentile(samples, 95)
                    summary[f"{variant}_{stage}_stdev"] = statistics.stdev(samples)
                summary[variant + "_figure_bytes"] = (
                    members[0]["correctness_probability"]["figure_payload_bytes"]
                )
            summary["overall_p50_speedup"] = (
                summary["baseline_total_ms_p50"] /
                summary["candidate_total_ms_p50"]
            )
            summaries.append(summary)
            for member in group:
                for sample in member["warm"]:
                    rows.append({
                        "size": n, "fits": fits,
                        "variant": member["variant"], "round": member["round"],
                        **sample,
                    })
    with (reports / "raw_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "baseline_sha": args.baseline_sha,
        "candidate_sha": args.candidate_sha,
        "baseline_path": str(args.baseline),
        "candidate_path": str(args.candidate),
        "runner": platform.platform(),
        "python": sys.version,
        "seed": SEED,
        "page_size": PAGE_SIZE,
        "rounds": args.rounds,
        "warm_repetitions_per_round": args.repetitions,
        "first_use": "first full invocation per fresh process, not cold OS disk cache",
        "processes_per_variant_case": args.rounds,
        "fixture_details": [
            json.loads((_fixture_dir(args.root, n) / "fixture.json").read_text())
            for n in (5000, 50000)
        ],
        "results": summaries,
        "correctness": "exact sha256 parity of all point identity/coordinates/labels,"
                       " final Plotly JSON (includes hovertext/customdata),"
                       " and click mapping for probability and decision-score",
        "not_measured": [
            "actual SQLite page-cache/disk bytes", "network delivery latency",
            "browser WebGL paint", "end-to-end human interaction delay",
        ],
    }
    (reports / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("RESULTS", json.dumps(summaries, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", type=Path, required=True)
    work = sub.add_parser("worker")
    work.add_argument("--root", type=Path, required=True)
    work.add_argument("--size", type=int, required=True)
    work.add_argument("--fits", type=int, required=True)
    work.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    work.add_argument("--round", type=int, required=True)
    work.add_argument("--repetitions", type=int, required=True)
    work.add_argument("--output", type=Path, required=True)
    comp = sub.add_parser("compare")
    comp.add_argument("--root", type=Path, required=True)
    comp.add_argument("--baseline", type=Path, required=True)
    comp.add_argument("--candidate", type=Path, required=True)
    comp.add_argument("--baseline-sha", required=True)
    comp.add_argument("--candidate-sha", required=True)
    comp.add_argument("--rounds", type=int, default=3)
    comp.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    if args.action == "prepare":
        args.root.mkdir(parents=True, exist_ok=True)
        for size in (5000, 50000):
            print("PREPARE", size, json.dumps(prepare(args.root, size)), flush=True)
    elif args.action == "worker":
        worker(args)
    else:
        if args.rounds * args.repetitions < 20:
            parser.error("At least 20 warm samples per case are required")
        compare(args)


if __name__ == "__main__":
    main()
