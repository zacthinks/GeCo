"""Reproducible *backend* recommendation benchmark for Phase 1E.

Run: uv run python benchmarks/phase1e_recommendations.py --sizes 1000 10000
Reports JSON, not a CI speed assertion. No browser/low-powered-laptop claims.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tracemalloc
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from geometric_coder import CountGeometry, GeometricCoder


def _measure(fn, repeats: int) -> dict[str, float]:
    wall = []
    cpu = []
    for _ in range(repeats):
        t0, c0 = time.perf_counter(), time.process_time()
        fn()
        wall.append((time.perf_counter() - t0) * 1000)
        cpu.append((time.process_time() - c0) * 1000)
    ordered = sorted(wall)
    return {
        "median_ms": round(statistics.median(wall), 5),
        "p95_ms": round(ordered[max(0, int(np.ceil(0.95 * len(ordered))) - 1)], 5),
        "median_cpu_ms": round(statistics.median(cpu), 5),
    }


def run(size: int, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    tokens = ("clear", "helpful", "useful", "ambiguous", "uncertain",
              "weak", "bad", "strong", "positive", "negative")
    texts = [" ".join(tokens[int(x)] for x in rng.integers(0, len(tokens), 6))
             for _ in range(size)]
    with tempfile.TemporaryDirectory(prefix="geco-phase1e-") as root:
        project = GeometricCoder.create(
            project_dir=Path(root) / "bench.geco",
            data=pd.DataFrame({"row_id": range(size), "text": texts}),
            keys=["row_id"], text="text", metadata=[],
            geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
        )
        code = project.create_code("quality")
        units = project.database.list_observations()
        for idx in range(10):
            project.annotate(int(units[idx]["observation_id"]), code, "positive")
            project.annotate(int(units[idx + 10]["observation_id"]), code, "negative")
        classifier = int(project.classifier_specs(code_id=code)[0]["classifier_spec_id"])
        t0 = time.perf_counter()
        fit = project.train_classifier(code_id=code, classifier_spec_id=classifier)
        training_ms = (time.perf_counter() - t0) * 1000
        session = project.create_session("benchmark")
        visited_session = project.create_session("visited benchmark")
        fit_id = int(fit["classifier_fit_id"])
        strategies = ["likely_positive", "likely_negative", "classifier_uncertainty"]

        # First call includes freshness-cache establishment, unlike steady state.
        first = {}
        warm = {}
        for strategy in strategies:
            first[strategy] = _measure(lambda: project.focus_recommendation(
                code_id=code, session_id=session, strategy=strategy,
                active_classifier_spec_id=classifier, auto_retrain=False,
            ), 1)
            warm[strategy] = _measure(lambda: project.focus_recommendation(
                code_id=code, session_id=session, strategy=strategy,
                active_classifier_spec_id=classifier, auto_retrain=False,
            ), repeats)

        # Realistic post-visit selection: advance through ranked candidates in
        # a separate session, then measure lookups with accumulated exclusions.
        visit_target = min(100, size // 10)
        for _ in range(visit_target):
            recommended = project.focus_recommendation(
                code_id=code, session_id=visited_session,
                strategy="likely_positive", active_classifier_spec_id=classifier,
                auto_retrain=False,
            )
            if recommended is None:
                break
            project.record_visit(visited_session, int(recommended["unit_id"]), method="bench")
        visited = _measure(lambda: project.focus_recommendation(
            code_id=code, session_id=visited_session,
            strategy="likely_positive", active_classifier_spec_id=classifier,
            auto_retrain=False,
        ), repeats)

        # Check post-judgment stale snapshot in an unchanged session; do not
        # accidentally benchmark a hidden retrain on the click path.
        project.annotate(int(units[21]["observation_id"]), code, "positive")
        stale_result = project.focus_recommendation(
            code_id=code, session_id=session, strategy="likely_positive",
            active_classifier_spec_id=classifier, auto_retrain=False,
        )
        assert stale_result is not None and stale_result["stale_classifiers"]
        stale = _measure(lambda: project.focus_recommendation(
            code_id=code, session_id=session, strategy="likely_positive",
            active_classifier_spec_id=classifier, auto_retrain=False,
        ), repeats)

        # Reopen tests cold per-instance freshness state and supported saved
        # snapshots without confusing it with another training operation.
        reopened = GeometricCoder.open(project.project_dir)
        reopened_first = _measure(lambda: reopened.focus_recommendation(
            code_id=code, session_id=session, strategy="likely_positive",
            active_classifier_spec_id=classifier, auto_retrain=False,
        ), 1)

        # Compare a prepared fixed committee against the single-classifier
        # lookup. Member training/preparation is timed separately, never folded
        # into recommendation click latency.
        geometry_id = int(project.database.get_classifier_spec(classifier)["geometry_id"])
        member = project.create_classifier_spec(
            code_id=code, name="phase1e tree", geometry_id=geometry_id,
            algorithm="decision_tree",
        )
        member_start = time.perf_counter()
        project.train_classifiers(code_id=code, classifier_spec_ids=[classifier, member])
        member_training_ms = (time.perf_counter() - member_start) * 1000
        committee = project.create_classifier_committee(
            code_id=code, name="phase1e fixed", classifier_spec_ids=[classifier, member],
            aggregation="mean",
        )
        prepared_start = time.perf_counter()
        project.prepare_selected_committee(code_id=code, committee_id=committee)
        preparation_ms = (time.perf_counter() - prepared_start) * 1000
        committee_first = _measure(lambda: project.focus_recommendation(
            code_id=code, session_id=session, strategy="likely_positive",
            committee_id=committee, recommendation_source="committee", auto_retrain=False,
        ), 1)
        committee_warm = _measure(lambda: project.focus_recommendation(
            code_id=code, session_id=session, strategy="likely_positive",
            committee_id=committee, recommendation_source="committee", auto_retrain=False,
        ), repeats)

        tracemalloc.start()
        _measure(lambda: project.focus_recommendation(
            code_id=code, session_id=session, strategy="likely_positive",
            active_classifier_spec_id=classifier, auto_retrain=False,
        ), repeats)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        with project.database.connect() as connection:
            indexes = [
                str(row["name"])
                for row in connection.execute("PRAGMA index_list('classifier_predictions')")
            ]
            connection.execute("PRAGMA optimize")
        result = {
            "size": size,
            "seed": seed,
            "training_ms": round(training_ms, 3),
            "fit_id": fit_id,
            "first": first,
            "warm": warm,
            "visited_count": visit_target,
            "visited_recommendation": visited,
            "stale_snapshot": stale,
            "reopened_first": reopened_first,
            "committee_member_training_ms": round(member_training_ms, 3),
            "fixed_committee_prepare_ms": round(preparation_ms, 3),
            "fixed_committee_first": committee_first,
            "fixed_committee_warm": committee_warm,
            "tracemalloc_peak_bytes": peak,
            "indexes": indexes,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000])
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.sizes or min(args.sizes) < 40 or args.repeats < 2:
        parser.error("size must be >=40, and repeats >=2")
    report = {
        "runner": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "limitations": [
            "backend project API only, not Dash click-to-focal-display",
            "GitHub-hosted runner results cannot represent constrained researcher laptops",
            "no historical pre-refactor baseline provided; compare revisions on same runner",
            "fixed committee mean included; learned committee is tested separately",
            "tracemalloc captures Python allocations, not total resident memory",
        ],
        "runs": [run(s, args.repeats, args.seed) for s in args.sizes],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
