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


def test_hidden_workspace_skips_apply_sources_and_presentation_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dash import no_update

    from geometric_coder.ui.app import _memo_presentation_is_active, create_app

    project, _run_id = _draft(tmp_path)
    app = create_app(project)

    assert _memo_presentation_is_active("?presentation=memo&memo_id=1")
    assert not _memo_presentation_is_active("")
    assert not _memo_presentation_is_active("?presentation=other")

    callbacks = app.callback_map
    source_entries = [
        entry["callback"].__wrapped__
        for name, entry in callbacks.items()
        if "apply-source-dropdown.options" in name
        and "apply-source-dropdown.value" in name
    ]
    assert len(source_entries) == 1

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Hidden Apply must not inspect classifier status")

    monkeypatch.setattr(project, "classifier_status", forbidden)
    hidden = source_entries[0](None, 0, 0, 0, 0, 0, "explore", None)
    assert hidden == (no_update, no_update)

    map_entries = [
        entry["callback"].__wrapped__
        for name, entry in callbacks.items()
        if "memo-presentation-map.figure" in name
        and "memo-presentation-page-status.children" in name
    ]
    assert len(map_entries) == 1
    unused = map_entries[0](
        None, 0, 0, "__all__", "5000", 1, None, "", None
    )
    assert unused == (no_update, no_update)


def test_paging_ui_renders_only_one_server_page_at_a_time(tmp_path: Path) -> None:
    from geometric_coder.ui.app import create_app

    project, _run_id = _draft(tmp_path)
    app = create_app(project)
    callbacks = app.callback_map
    matches = [
        entry["callback"].__wrapped__
        for name, entry in callbacks.items()
        if "apply-table.rowData" in name
        and "apply-run-summary.children" in name
    ]
    assert len(matches) == 1
    grid_rows, summary, pager, previous_disabled, next_disabled, snapshot = matches[0](
        _run_id, "", "", [0.0, 1.0], 0, 25, None, 0
    )
    assert len(grid_rows) == 25
    assert snapshot["unit_ids"] == [row["unit_id"] for row in grid_rows]
    assert snapshot["page_count"] == 3
    assert previous_disabled is True and next_disabled is False
    assert "60 matching" in pager
    assert f"Draft {_run_id}" in summary
    last = matches[0](_run_id, "", "", [0.0, 1.0], 99, 25, None, 0)
    assert len(last[0]) == 10
    assert last[5]["page_index"] == 2
    assert last[3] is False and last[4] is True


def test_paging_index_is_installed_on_current_schema_reopen(tmp_path: Path) -> None:
    project, _run_id = _draft(tmp_path)
    with project.database.connect() as connection:
        connection.execute("DROP INDEX apply_proposals_active_probability_rank_idx")
    reopened = GeometricCoder.open(project.project_dir)
    with reopened.database.connect() as connection:
        names = {
            str(row["name"]) for row in connection.execute(
                "PRAGMA index_list('apply_proposals')"
            )
        }
    assert "apply_proposals_active_probability_rank_idx" in names


def test_presentation_map_active_uses_page_rows_without_explore_visits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from types import SimpleNamespace
    import geometric_coder.ui.app as ui

    project, _run_id = _draft(tmp_path)
    geometry_id = int(project.database.list_geometries(public_only=True)[0]["geometry_id"])
    coordinates = np.column_stack((
        np.linspace(-1.0, 1.0, 64),
        np.linspace(1.0, -1.0, 64),
    ))
    artifact = project.artifacts.save_view("phase5_presentation", coordinates)
    view_id = project.database.register_view(
        geometry_id=geometry_id, name="phase5_presentation",
        method="pca", parameters={}, artifact_path=artifact,
    )
    app = ui.create_app(project)
    callbacks = [
        entry["callback"].__wrapped__
        for name, entry in app.callback_map.items()
        if "memo-presentation-map.figure" in name
        and "memo-presentation-page-status.children" in name
    ]
    assert len(callbacks) == 1
    recorded: list[list[int]] = []
    actual = project.view_coordinate_rows

    def traced(view: int, positions: list[int]) -> np.ndarray:
        assert view == view_id
        recorded.append(positions)
        return actual(view, positions)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Presentation must not read full views or record Explore visits")

    monkeypatch.setattr(project, "view_coordinate_rows", traced)
    monkeypatch.setattr(project, "view_coordinates", forbidden)
    monkeypatch.setattr(project, "record_visit", forbidden)
    monkeypatch.setattr(ui, "callback_context", SimpleNamespace(triggered_id="memo-presentation-view"))
    figure, message = callbacks[0](
        view_id, 0, 0, "__all__", "20", 1, "",
        "?presentation=memo&memo_id=1", None,
    )
    assert recorded and len(recorded[-1]) <= 20
    assert figure is not None
    assert "Page 1" in message
