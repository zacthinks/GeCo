"""Phase 5: bounded Apply pages, filtered counts, and review scope integrity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import CountGeometry, GeometricCoder
from geometric_coder.ui.app import _apply_sort_spec, _verified_apply_page


def _draft(tmp_path: Path) -> tuple[GeometricCoder, int]:
    data = pd.DataFrame({
        "row_id": list(range(64)),
        "text": [
            (f"helpful positive clear testimony {index}" if index < 32
             else f"weak negative confusing testimony {index}")
            for index in range(64)
        ],
    })
    project = GeometricCoder.create(
        project_dir=tmp_path / "phase5.geco",
        data=data, keys=["row_id"], text="text", metadata=[],
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    code_id = project.create_code("quality")
    project.ensure_default_classifier_specs(code_id=code_id)
    spec_id = int(project.classifier_specs(code_id=code_id)[0]["classifier_spec_id"])
    observations = project.database.list_observations()
    for index in (0, 1):
        project.annotate(int(observations[index]["observation_id"]), code_id, "positive")
    for index in (32, 33):
        project.annotate(int(observations[index]["observation_id"]), code_id, "negative")
    project.train_classifier(code_id=code_id, classifier_spec_id=spec_id)
    run_id = project.create_apply_run(
        code_id=code_id, source_kind="classifier", source_id=spec_id,
    )
    return project, run_id


def test_bounded_sorted_pages_match_filtered_count(tmp_path: Path) -> None:
    project, run_id = _draft(tmp_path)
    total = project.count_apply_proposals(run_id)
    assert total >= 50
    all_rows = project.apply_proposals(run_id)
    assert len(all_rows) == total
    first = project.apply_proposals(run_id, limit=25, offset=0)
    second = project.apply_proposals(run_id, limit=25, offset=25)
    third = project.apply_proposals(run_id, limit=25, offset=50)
    assert [row["unit_id"] for row in first + second + third] == [
        row["unit_id"] for row in all_rows
    ]
    assert len({row["unit_id"] for row in first + second + third}) == total

    ascending = project.apply_proposals(
        run_id, limit=25, sort_by=[("probability", "asc")],
    )
    assert [row["probability"] for row in ascending] == sorted(
        row["probability"] for row in ascending
    )
    for label in ("positive", "negative"):
        count = project.count_apply_proposals(run_id, proposed_label=label)
        rows = project.apply_proposals(run_id, proposed_label=label)
        assert count == len(rows)

    with pytest.raises(ValueError, match="Unsupported Apply proposal sort"):
        project.apply_proposals(
            run_id, limit=25, sort_by=[("metadata_json; DROP TABLE units", "asc")],
        )


def test_page_identity_rejects_stale_filters_and_wrong_drafts(tmp_path: Path) -> None:
    project, run_id = _draft(tmp_path)
    rows = project.apply_proposals(run_id, limit=25)
    count = project.count_apply_proposals(run_id)
    snapshot = {
        "draft_id": run_id, "page_index": 0, "page_count": 3,
        "page_size": 25, "count": count,
        "unit_ids": [int(row["unit_id"]) for row in rows],
        "proposed": "", "decision": "", "range": [0.0, 1.0],
        "sort": [], "refresh": 4,
    }
    args = dict(
        project=project, draft_id=run_id, snapshot=snapshot,
        proposed_filter="", decision_filter="", probability_range=[0.0, 1.0],
        page_size=25, column_state=None, refresh=4,
    )
    assert _verified_apply_page(**args) == snapshot["unit_ids"]
    with pytest.raises(ValueError, match="page changed"):
        _verified_apply_page(**{**args, "draft_id": run_id + 999})
    with pytest.raises(ValueError, match="page changed"):
        _verified_apply_page(**{**args, "proposed_filter": "positive"})
    with pytest.raises(ValueError, match="page changed"):
        _verified_apply_page(**{**args, "refresh": 5})
    assert _apply_sort_spec([
        {"colId": "probability", "sort": "asc", "sortIndex": 0},
    ]) == [("probability", "asc")]


def test_current_page_commit_and_undo_leave_other_proposals_intact(tmp_path: Path) -> None:
    project, run_id = _draft(tmp_path)
    rows = project.apply_proposals(run_id, limit=25)
    ids = [int(row["unit_id"]) for row in rows]
    changed = project.bulk_review_apply_proposals(
        apply_run_id=run_id, decision="accept", unit_ids=ids,
    )
    assert changed == 25
    assert project.count_apply_proposals(run_id, decision="accept") == 25
    result = project.commit_apply_run(run_id, unit_ids=ids, scope="current_page")
    assert result["processed"] == 25
    assert project.count_apply_proposals(run_id) == 60 - 25
    assert project.count_apply_proposals(run_id, decision="pending") == 35
    restored = project.undo_last_apply_commit(run_id)
    assert restored["restored_proposals"] == 25
    assert project.count_apply_proposals(run_id) == 60


def test_all_scope_does_not_mean_visible_page(tmp_path: Path) -> None:
    project, run_id = _draft(tmp_path)
    visible = project.apply_proposals(run_id, limit=25)
    assert len(visible) == 25
    count = project.bulk_review_apply_proposals(
        apply_run_id=run_id, decision="unreviewed",
    )
    assert count == project.count_apply_proposals(run_id)
    assert count > len(visible)
    reviewed = project.commit_apply_run(run_id, scope="all_reviewed")
    assert reviewed["processed"] == count
    assert reviewed["removed_unreviewed"] == count
    assert reviewed["committed"] == 0
