"""Phase 1E diagnostic benchmark: committee scaling and exhausted visits.

Not a timing gate or an end-to-end browser measurement. Uses real project
APIs and reported SQL statement counts, rather than an assumed query count.
Run: uv run python benchmarks/phase1e_diagnostics.py --sizes 1000 10000
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geometric_coder import CountGeometry, GeometricCoder


def _measure(fn, repeat: int) -> dict[str, float]:
    durations = []
    for _ in range(repeat):
        before = time.perf_counter()
        result = fn()
        assert result is not None
        durations.append((time.perf_counter() - before) * 1000)
    values = sorted(durations)
    return {
        "median_ms": round(statistics.median(values), 4),
        "p95_ms": round(values[max(0, int(np.ceil(.95 * len(values))) - 1)], 4),
    }


def _case(size: int, seed: int, repeats: int) -> dict:
    rng = np.random.default_rng(seed)
    words = ["good", "clear", "bad", "weak", "excellent", "helpful", "uncertain"]
    data = pd.DataFrame({
        "id": list(range(size)),
        "text": [
            " ".join(words[int(j)] for j in rng.integers(0, len(words), 6))
            for _ in range(size)
        ],
    })
    with tempfile.TemporaryDirectory() as tmp:
        project = GeometricCoder.create(
            project_dir=Path(tmp) / "diag.geco", data=data, keys=["id"],
            text="text", metadata=[],
            geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
        )
        code = project.create_code("benchmark")
        obs = project.database.list_observations()
        for i in range(10):
            project.annotate(int(obs[i]["observation_id"]), code, "positive")
            project.annotate(int(obs[i + 10]["observation_id"]), code, "negative")
        spec = int(project.classifier_specs(code_id=code)[0]["classifier_spec_id"])
        geometry = int(project.database.get_classifier_spec(spec)["geometry_id"])
        member = project.create_classifier_spec(
            code_id=code, name="tree diagnostic member",
            geometry_id=geometry, algorithm="decision_tree",
        )
        project.train_classifiers(code_id=code, classifier_spec_ids=[spec, member])
        fixed = project.create_classifier_committee(
            code_id=code, name="diagnostic fixed", classifier_spec_ids=[spec, member],
            aggregation="mean",
        )
        learned = project.create_classifier_committee(
            code_id=code, name="diagnostic learned", classifier_spec_ids=[spec, member],
            aggregation="logistic_stack",
        )
        project.prepare_selected_committee(code_id=code, committee_id=fixed)
        project.train_classifier_committee(code_id=code, committee_id=learned)
        results = {"size": size, "cases": {}}
        for source, committee in [
            ("classifier", None), ("fixed", fixed), ("learned", learned)
        ]:
            session = project.create_session(f"{source} session")
            options = dict(
                code_id=code, session_id=session, strategy="likely_positive",
                auto_retrain=False,
            )
            if committee is None:
                options["active_classifier_spec_id"] = spec
            else:
                options["committee_id"] = committee
                options["recommendation_source"] = "committee"
            project.focus_recommendation(**options)  # warm per-source metadata
            fresh = _measure(lambda: project.focus_recommendation(**options), repeats)

            # A nearly exhausted session has only one eligible recommendation.
            # Repeated visits are setup, not part of measured click latency.
            all_units = [int(row["unit_id"]) for row in project.database.list_units()]
            annotated = {
                int(a["observation_id"]) for a in project.current_annotations(code)
            }
            eligible = [
                unit for unit in all_units
                if int(project.observation_id_for_unit(unit)) not in annotated
            ]
            for unit in eligible[:-1]:
                project.record_visit(session, unit, method="phase1e benchmark")
            last = _measure(lambda: project.focus_recommendation(**options), repeats)
            found = project.focus_recommendation(**options)
            assert found is not None
            assert found["unit_id"] == eligible[-1]
            results["cases"][source] = {
                "fresh": fresh, "near_exhausted": last,
                "visited": len(eligible) - 1,
            }
        return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[200, 1000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, default=Path("phase1e-diagnostics.json"))
    args = parser.parse_args()
    report = {
        "limitations": [
            "backend API, not Dash click-to-render",
            "no historical baseline or constrained laptop test",
            "nearly exhausted session setup can have O(N) visit writes",
        ],
        "runs": [_case(size, args.seed, args.repeats) for size in args.sizes],
    }
    result = json.dumps(report, indent=2)
    print(result, flush=True)
    args.output.write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
