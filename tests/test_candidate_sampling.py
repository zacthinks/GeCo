"""Regression contract for bounded Code observation and Scale pair candidates."""
from __future__ import annotations

import itertools

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.candidate_sampling import (
    MAX_EXHAUSTIVE_SCALE_PAIRS,
    _unrank_pair,
    resolve_candidate_budget,
)


def _project(tmp_path, n=12):
    project = GeometricCoder.create(
        project_dir=tmp_path / "candidates.geco",
        data=pd.DataFrame({
            "key": list(range(n)),
            "text": [f"example {i}" for i in range(n)],
            "group": ["yes" if i % 2 == 0 else "no" for i in range(n)],
        }),
        keys=["key"], text="text", metadata=["group"], geometries=None,
    )
    return project, project.create_code("Code"), project.create_scale("Scale")


def _all(run, *, acknowledge=True):
    tasks = []
    for batch in run.batches():
        tasks.extend(batch.tasks)
        if acknowledge:
            run.acknowledge_evaluated(batch)
    return tasks


@pytest.mark.parametrize("n", [2, 3, 7, 12])
def test_exhaustive_pairs_are_exactly_combinations(tmp_path, n):
    project, _, scale = _project(tmp_path, n)
    run = project.candidate_run(scale, budget=0, seed=5, batch_size=3)
    tasks = _all(run)
    ids = project.database.acquisition_eligible_atomic_ids("scale", scale)
    expected = set(itertools.combinations(ids, 2))
    assert set(tasks) == expected
    assert len(tasks) == n * (n - 1) // 2
    assert run.requested_budget == 0
    assert run.exhausted
    assert run.evaluated_count == len(tasks)
    assert all(a < b for a, b in tasks)
    assert len(set(tasks)) == len(tasks)


def test_code_and_scale_budget_ten(tmp_path):
    project, code, scale = _project(tmp_path)
    for construct in (code, scale):
        run = project.candidate_run(construct, budget=10, seed=7, batch_size=4)
        candidates = _all(run)
        assert len(candidates) == 10
        assert run.evaluated_count == 10
        assert len(set(candidates)) == 10
        assert all(len(task) == (1 if construct == code else 2) for task in candidates)
        assert run.provenance()["requested_budget"] == 10
    assert resolve_candidate_budget("code", None) == 0
    assert resolve_candidate_budget("scale", None) == 1000


def test_acknowledgment_is_distinct_from_sampling_and_cannot_double_count(tmp_path):
    project, _, scale = _project(tmp_path, 4)
    run = project.candidate_run(scale, budget=3, seed=0)
    batch = next(run.batches())
    assert run.evaluated_count == 0
    run.acknowledge_evaluated(batch, batch.tasks[:1])
    assert run.evaluated_count == 1
    with pytest.raises(ValueError):
        run.acknowledge_evaluated(batch)
    assert run.sampled_count == 3


def test_seed_stability_change_and_evidence_revision(tmp_path):
    project, _, scale = _project(tmp_path, 30)
    one = project.candidate_run(scale, budget=10, seed=22)
    two = project.candidate_run(scale, budget=10, seed=22)
    three = project.candidate_run(scale, budget=10, seed=23)
    first = _all(one)
    assert _all(two) == first
    assert _all(three) != first
    assert one.eligibility_fingerprint == two.eligibility_fingerprint
    project.record_scale_comparison(scale, *first[0], "unsure")
    revised = project.candidate_run(scale, budget=10, seed=22)
    assert first[0] not in _all(revised)
    assert revised.eligibility_fingerprint != one.eligibility_fingerprint


def test_all_versions_and_unsure_are_excluded(tmp_path):
    project, _, scale = _project(tmp_path, 5)
    ids = list(project.database.acquisition_eligible_atomic_ids("scale", scale))
    project.record_scale_comparison(scale, ids[1], ids[0], "unsure")
    project.update_scale(scale, "Updated scale", "Version 2")
    project.record_scale_comparison(scale, ids[2], ids[3], "parity")
    run = project.candidate_run(scale, budget=0)
    tasks = _all(run)
    assert len(tasks) == 8
    assert (ids[0], ids[1]) not in tasks
    assert (ids[2], ids[3]) not in tasks
    assert run.exhausted


def test_code_to_scale_bootstrap_excluded(tmp_path):
    project, code, _ = _project(tmp_path, 6)
    ids = list(project.database.acquisition_eligible_atomic_ids("code", code))
    for i, obs in enumerate(ids):
        project.annotate(obs, code, "positive" if i < 3 else "negative")
    preview = project.code_scale_bootstrap_preview(code)
    created = project.create_scale_from_code(
        "Bootstrapped", "", code, expected_preview=preview,
        submission_key="a" * 32,
    )
    scale = created["scale_id"]
    tasks = _all(project.candidate_run(scale, budget=0))
    assert created["comparisons_created"] == 9
    assert len(tasks) == 15 - 9
    assert all(not ((a in ids[:3] and b in ids[3:]) or
                    (b in ids[:3] and a in ids[3:])) for a, b in tasks)


def test_code_current_annotations_and_session_visits(tmp_path):
    project, code, _ = _project(tmp_path, 5)
    ids = list(project.database.acquisition_eligible_atomic_ids("code", code))
    session = project.create_session("Candidate session")
    project.annotate(ids[0], code, "unsure")
    project.record_visit(session, 2, method="test")
    tasks = _all(project.candidate_run(code, budget=0, session_id=session))
    assert len(tasks) == 3
    assert (ids[0],) not in tasks
    assert (ids[1],) not in tasks


def test_scale_eligibility_and_fingerprint_revisions(tmp_path):
    project, code, scale = _project(tmp_path, 6)
    base = project.candidate_run(scale, budget=0)
    project.update_scale_eligibility(scale, {
        "kind": "filtered",
        "metadata_clauses": [
            {"field": "group", "type": "categorical",
             "operator": "in", "value": ["yes"]}
        ],
    })
    filtered = project.candidate_run(scale, budget=0)
    assert filtered.eligible_observation_count == 3
    assert len(_all(filtered)) == 3
    assert filtered.eligibility_fingerprint != base.eligibility_fingerprint
    project.update_scale_eligibility(scale, {"kind": "all"})
    restored = project.candidate_run(scale, budget=0)
    assert restored.eligible_observation_count == 6


def test_exhaustive_refuses_large_pair_space_without_materializing(tmp_path, monkeypatch):
    project, _, scale = _project(tmp_path, 3)
    from array import array
    monkeypatch.setattr(
        project.database, "acquisition_eligible_atomic_ids",
        lambda *args, **kwargs: array("q", range(300000)),
    )
    with pytest.raises(ValueError, match="Exhaustive.*safety limit"):
        project.candidate_run(scale, budget=0)
    run = project.candidate_run(scale, budget=10, seed=12)
    assert len(_all(run)) == 10
    assert run.inspected_task_count <= 128
    assert MAX_EXHAUSTIVE_SCALE_PAIRS < 300000 * 299999 // 2


def test_partial_exhaustion_and_no_judgment_writes(tmp_path):
    project, _, scale = _project(tmp_path, 3)
    ids = list(project.database.acquisition_eligible_atomic_ids("scale", scale))
    project.record_scale_comparison(scale, ids[0], ids[1], "parity")
    project.record_scale_comparison(scale, ids[0], ids[2], "parity")
    before = len(project.scale_comparisons(scale))
    run = project.candidate_run(scale, budget=10)
    assert _all(run) == [(ids[1], ids[2])]
    assert run.evaluated_count == 1
    assert run.exhausted
    assert len(project.scale_comparisons(scale)) == before


def test_saved_budget_controls_both_construct_types(tmp_path):
    from geometric_coder.ui.app import _save_ui_settings
    project, code, scale = _project(tmp_path, 8)
    _save_ui_settings(project.project_dir, candidate_evaluation_budget=3)
    for construct in (code, scale):
        run = project.candidate_run(construct)
        assert len(_all(run)) == 3
        assert run.requested_budget == 3
    _save_ui_settings(project.project_dir, candidate_evaluation_budget=0)
    assert project.candidate_run(scale).requested_budget == 0


@pytest.mark.parametrize("n", range(2, 25))
def test_integer_pair_unranking(n):
    assert [_unrank_pair(rank, n) for rank in range(n * (n - 1) // 2)] == list(
        itertools.combinations(range(n), 2)
    )


@pytest.mark.parametrize("n", [0, 1])
def test_empty_and_singleton_eligible_scale_population(tmp_path, monkeypatch, n):
    from array import array
    project, _, scale = _project(tmp_path, 3)
    monkeypatch.setattr(
        project.database, "acquisition_eligible_atomic_ids",
        lambda *args, **kwargs: array("q", range(1, n + 1)),
    )
    run = project.candidate_run(scale, budget=0)
    assert _all(run) == []
    assert run.possible_task_count == 0
    assert run.evaluated_count == 0
    assert run.exhausted
    assert run.unavailable_reason
