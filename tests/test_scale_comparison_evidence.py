"""Scale comparison evidence and eligibility regression tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import GeometricCoder


def _project(tmp_path: Path, *, external: bool = False) -> GeometricCoder:
    data = pd.DataFrame({
        "row_id": [10, 11, 12, 13],
        "text": ["one", "two", "three", "four"],
        "group": ["alpha", "alpha", "beta", "beta"],
        "score": [1, 2, 3, 4],
    })
    kwargs = dict(
        project_dir=tmp_path / "comparisons.geco", data=data,
        keys=["row_id"], text="text", metadata=["group", "score"],
    )
    return (
        GeometricCoder.create_external(**kwargs, external_provider=object())
        if external else GeometricCoder.create(**kwargs, geometries=None)
    )


def _ids(project: GeometricCoder) -> list[int]:
    return [
        project.database.observation_for_unit(i)
        for i in range(1, 5)
    ]


@pytest.mark.parametrize("external", [False, True])
def test_orientation_parity_skip_and_reopening(tmp_path: Path, external: bool) -> None:
    project = _project(tmp_path, external=external)
    scale = project.create_scale("Clarity")
    a, b, c, _ = _ids(project)
    session = int(project.sessions()[0]["session_id"])
    first = project.record_scale_comparison(
        scale, a, b, "first_higher", reviewer_key="r1", session_id=session,
        origin="human_direct", selection_provenance={"strategy": "manual", "seed": 42},
    )
    second = project.record_scale_comparison(
        scale, b, a, "second_higher", reviewer_key="r2",
    )
    parity = project.record_scale_comparison(scale, b, c, "parity")
    skip = project.record_scale_comparison(scale, a, c, "unsure")
    assert len({first, second, parity, skip}) == 4
    events = project.scale_comparisons(scale)
    assert len(events) == 4
    assert events[0]["observation_low_id"] == a
    assert events[0]["observation_high_id"] == b
    assert events[0]["presented_first_id"] == a
    assert events[1]["presented_first_id"] == b
    assert [event["canonical_outcome"] for event in events] == [
        "low_higher", "low_higher", "parity", "unsure",
    ]
    assert events[0]["selection_provenance"] == {"strategy": "manual", "seed": 42}
    assert events[0]["session_id"] == session
    assert events[0]["reviewer_key"] == "r1"
    assert events[3]["training_usable"] is False
    assert [item["comparison_event_id"] for item in project.scale_training_comparisons(scale)] == [
        first, second, parity
    ]
    assert project.codes() == []
    reopened = GeometricCoder.open(
        project.project_dir, external_provider=object() if external else None,
    )
    assert reopened.scale_comparisons(scale) == events
    assert reopened.scale_training_comparisons(scale) == project.scale_training_comparisons(scale)


def test_corrections_are_append_only_with_one_terminal_event_per_chain(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    scale = project.create_scale("Power")
    a, b, c, _ = _ids(project)
    initial = project.record_scale_comparison(
        scale, a, b, "parity", reviewer_key="same",
    )
    changed = project.record_scale_comparison(
        scale, b, a, "first_higher", reviewer_key="same",
        supersedes_event_id=initial,
    )
    again = project.record_scale_comparison(
        scale, a, b, "second_higher", reviewer_key="same",
        supersedes_event_id=changed,
    )
    independent = project.record_scale_comparison(
        scale, a, b, "first_higher", reviewer_key="other",
    )
    assert len(project.scale_comparisons(scale)) == 4
    effective = project.scale_comparisons(scale, effective_only=True)
    assert [event["comparison_event_id"] for event in effective] == [again, independent]
    assert effective[0]["canonical_outcome"] == "high_higher"
    assert effective[1]["canonical_outcome"] == "low_higher"
    with pytest.raises(ValueError, match="already corrected"):
        project.record_scale_comparison(
            scale, a, b, "parity", reviewer_key="same",
            supersedes_event_id=initial,
        )
    with pytest.raises(ValueError, match="reviewer"):
        project.record_scale_comparison(
            scale, a, b, "parity", reviewer_key="wrong",
            supersedes_event_id=again,
        )
    with pytest.raises(ValueError, match="canonical pair"):
        project.record_scale_comparison(
            scale, a, c, "parity", reviewer_key="same",
            supersedes_event_id=again,
        )
    assert len(project.scale_comparisons(scale)) == 4


def test_definition_versions_isolate_new_evidence_and_old_corrections(tmp_path: Path) -> None:
    project = _project(tmp_path)
    scale = project.create_scale("Draft")
    a, b, _, _ = _ids(project)
    first = project.record_scale_comparison(scale, a, b, "parity")
    project.update_scale(scale, "New meaning", "new definition")
    second = project.record_scale_comparison(scale, a, b, "second_higher")
    assert [item["comparison_event_id"] for item in project.scale_comparisons(
        scale, version_number=1,
    )] == [first]
    assert [item["comparison_event_id"] for item in project.scale_comparisons(
        scale, version_number=2,
    )] == [second]
    assert [item["comparison_event_id"] for item in project.scale_training_comparisons(
        scale,
    )] == [second]
    with pytest.raises(ValueError, match="Scale version"):
        project.record_scale_comparison(
            scale, a, b, "first_higher", supersedes_event_id=first,
        )
    assert len(project.scale_comparisons(scale)) == 2
    with pytest.raises(KeyError, match="scale version"):
        project.scale_comparisons(scale, version_number=3)


def test_idempotent_retries_are_payload_checked_even_after_version_change(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    scale = project.create_scale("Fidelity")
    a, b, _, _ = _ids(project)
    kwargs = {"reviewer_key": "R", "selection_provenance": {"strategy": "manual"}}
    first = project.record_scale_comparison(
        scale, a, b, "first_higher", submission_key="request-1", **kwargs,
    )
    assert project.record_scale_comparison(
        scale, a, b, "first_higher", submission_key="request-1", **kwargs,
    ) == first
    project.update_scale(scale, "New fidelity", "more precise")
    assert project.record_scale_comparison(
        scale, a, b, "first_higher", submission_key="request-1", **kwargs,
    ) == first
    with pytest.raises(ValueError, match="submission_key"):
        project.record_scale_comparison(
            scale, b, a, "second_higher", submission_key="request-1", **kwargs,
        )
    assert len(project.scale_comparisons(scale)) == 1


def test_reject_invalid_observations_and_provenance_atomically(tmp_path: Path) -> None:
    project = _project(tmp_path)
    scale = project.create_scale("Focus")
    a, b, _, _ = _ids(project)
    with pytest.raises(ValueError, match="itself"):
        project.record_scale_comparison(scale, a, a, "parity")
    with pytest.raises(KeyError, match="observation_id"):
        project.record_scale_comparison(scale, a, 999999, "parity")
    with pytest.raises(ValueError, match="outcome"):
        project.record_scale_comparison(scale, a, b, "maybe")
    with pytest.raises(ValueError, match="JSON-compatible"):
        project.record_scale_comparison(
            scale, a, b, "parity", selection_provenance={"bad": object()},
        )
    with pytest.raises(KeyError, match="session_id"):
        project.record_scale_comparison(scale, a, b, "parity", session_id=999999)
    with pytest.raises(KeyError, match="supersedes"):
        project.record_scale_comparison(
            scale, a, b, "parity", supersedes_event_id=999999,
        )
    with pytest.raises(KeyError, match="scale_id"):
        project.record_scale_comparison(999999, a, b, "parity")
    assert project.scale_comparisons(scale) == []
    with project.database.connect() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_eligibility_metadata_and_human_positive_code_filter(tmp_path: Path) -> None:
    project = _project(tmp_path)
    scale = project.create_scale("Independence")
    a, b, c, d = _ids(project)
    assert project.eligible_scale_observation_ids(scale, [d, b, a, c]) == [d, b, a, c]
    assert project.scale_eligibility(scale)["eligibility_revision"] == 1

    code = project.create_code("Include")
    project.annotate(a, code, "positive", origin="human_direct")
    project.annotate(b, code, "negative", origin="human_direct")
    project.annotate(c, code, "positive", origin="human_active_learning")
    project.annotate(d, code, "positive", origin="model_apply")
    config = {
        "kind": "filtered",
        "metadata_clauses": [
            {"field": "score", "type": "numeric", "operator": "gte", "value": 1},
            {"field": "group", "type": "categorical", "operator": "in", "value": ["alpha", "beta"]},
        ],
        "code_filters": [{"code_id": code, "value": "positive", "source": "human"}],
    }
    v = project.update_scale_eligibility(scale, config)
    assert v == 2
    assert project.scale_eligibility(scale)["eligibility_revision"] == 2
    assert project.eligible_scale_observation_ids(scale, [a, b, c, d]) == [a, c]
    assert project.update_scale_eligibility(scale, config) == 2
    assert project.scale_versions(scale)[0]["version_number"] == 1
    dependency_before = project.scale_eligibility(scale)["code_revisions"][code]
    project.annotate(a, code, "negative", origin="human_direct")
    assert project.eligible_scale_observation_ids(scale, [a, b, c, d]) == [c]
    assert project.scale_eligibility(scale)["code_revisions"][code] != dependency_before
    assert project.scale_eligibility(scale)["eligibility_revision"] == 2
    with pytest.raises(KeyError, match="code_id"):
        project.update_scale_eligibility(
            scale, {"kind": "filtered", "code_filters": [
                {"code_id": 999999, "value": "positive", "source": "human"}
            ]},
        )
    with pytest.raises(ValueError, match="human-positive"):
        project.update_scale_eligibility(
            scale, {"kind": "filtered", "code_filters": [
                {"code_id": code, "value": "positive", "source": "predicted"}
            ]},
        )
    assert project.scale_eligibility(scale)["eligibility_revision"] == 2


@pytest.mark.parametrize("external", [False, True])
def test_training_selection_respects_current_eligibility_without_dropping_history(
    tmp_path: Path, external: bool,
) -> None:
    project = _project(tmp_path, external=external)
    scale = project.create_scale("Warmth")
    a, b, c, d = _ids(project)
    first = project.record_scale_comparison(scale, a, b, "first_higher")
    second = project.record_scale_comparison(scale, b, c, "parity")
    ignored = project.record_scale_comparison(scale, c, d, "unsure")
    assert len(project.scale_training_comparisons(scale)) == 2
    project.update_scale_eligibility(scale, {
        "kind": "filtered", "metadata_clauses": [
            {"field": "group", "type": "categorical", "operator": "in", "value": ["alpha"]},
        ],
    })
    assert [row["comparison_event_id"] for row in project.scale_training_comparisons(scale)] == [first]
    assert [row["comparison_event_id"] for row in project.scale_comparisons(scale)] == [
        first, second, ignored
    ]
    reopened = GeometricCoder.open(
        project.project_dir, external_provider=object() if external else None,
    )
    assert [row["comparison_event_id"] for row in reopened.scale_training_comparisons(scale)] == [first]
    assert reopened.scale_eligibility(scale)["eligibility_revision"] == 2
    assert reopened.database.project_metadata()["schema_version"] == 19


def test_invalid_eligibility_is_rejected_without_mutating_version(tmp_path: Path) -> None:
    project = _project(tmp_path)
    scale = project.create_scale("Quality")
    invalid = [
        {"kind": "unsupported"},
        {"kind": "filtered"},
        {"kind": "all", "code_id": 1},
        {"kind": "filtered", "metadata_clauses": [
            {"field": "missing", "type": "text", "operator": "eq", "value": "x"},
        ]},
        {"kind": "filtered", "metadata_clauses": [
            {"field": "score", "type": "numeric", "operator": "gte", "value": "not-a-number"},
        ]},
        {"kind": "filtered", "metadata_clauses": [
            {"field": "group", "type": "text", "operator": "regex", "value": "["},
        ]},
    ]
    for config in invalid:
        with pytest.raises((ValueError, KeyError, TypeError)):
            project.update_scale_eligibility(scale, config)
    assert project.scale_eligibility(scale)["eligibility"] == {"kind": "all"}
    assert project.scale_eligibility(scale)["eligibility_revision"] == 1
    assert project.scale(scale)["version_number"] == 1
    assert project.database.project_metadata()["schema_version"] == 19
