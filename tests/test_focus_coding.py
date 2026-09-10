from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import GeometricCoder


def _project(tmp_path: Path) -> GeometricCoder:
    project = GeometricCoder.create(
        project_dir=tmp_path / "focus.geco",
        data=pd.DataFrame(
            {
                "doc_id": [10, 20, 30],
                "text": ["alpha", "beta", "gamma"],
                "group": ["A", "B", "C"],
            }
        ),
        keys=["doc_id"],
        text="text",
        metadata=["group"],
    )
    project.create_code("qualitative", "Qualitative evidence")
    project.create_code("quantitative", "Quantitative evidence")
    return project


def test_focus_task_freezes_order_codes_and_versions(tmp_path: Path) -> None:
    project = _project(tmp_path)
    codes = project.codes()
    session_id = project.configure_focus(
        codes=[codes[1]["code_id"], codes[0]["code_id"]],
        unit_ids=[3, 1, 2],
        allow_unsure=False,
    )
    task = project.focus_task(session_id)
    assert task["required_unit_ids"] == [3, 1, 2]
    assert [row["code_id"] for row in task["required_codes"]] == [
        int(codes[1]["code_id"]),
        int(codes[0]["code_id"]),
    ]
    frozen_versions = [row["version_number"] for row in task["required_codes"]]
    assert frozen_versions == [1, 1]

    project.update_code(
        int(codes[0]["code_id"]),
        name="qualitative",
        description="Changed later",
    )
    task_after_edit = project.focus_task(session_id)
    assert [row["version_number"] for row in task_after_edit["required_codes"]] == [1, 1]
    assert project.code(int(codes[0]["code_id"]), 1)["description"] == "Qualitative evidence"



def test_focus_user_keys_define_cross_system_order(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_ids = [int(row["code_id"]) for row in project.codes()]
    session_id = project.configure_focus(
        codes=code_ids,
        user_keys=[{"doc_id": 30}, {"doc_id": 10}],
        allow_unsure=False,
    )
    task = project.focus_task(session_id)
    assert task["required_unit_ids"] == [3, 1]
    assert task["required_user_keys"] == [{"doc_id": 30}, {"doc_id": 10}]


def test_focus_progress_treats_unsure_as_unresolved_when_forbidden(tmp_path: Path) -> None:
    project = _project(tmp_path)
    codes = project.codes()
    code_ids = [int(row["code_id"]) for row in codes]
    session_id = project.configure_focus(codes=code_ids, allow_unsure=False)

    project.annotate(1, code_ids[0], "positive", origin="human_focus_coder")
    project.annotate(1, code_ids[1], "unsure", origin="human_focus_coder")
    progress = project.focus_progress(session_id)

    assert progress["total_units"] == 3
    assert progress["total_judgments"] == 6
    assert progress["resolved_judgments"] == 1
    assert progress["unsure_judgments"] == 1
    assert progress["complete_units"] == 0
    assert progress["unresolved_count"] == 5
    assert not progress["complete"]


def test_focus_progress_allows_unsure_when_protocol_allows_it(tmp_path: Path) -> None:
    project = _project(tmp_path)
    codes = [int(row["code_id"]) for row in project.codes()]
    session_id = project.configure_focus(codes=codes, allow_unsure=True)
    for unit in project.units():
        observation_id = int(unit["observation_id"])
        project.annotate(observation_id, codes[0], "positive", origin="human_focus_coder")
        project.annotate(observation_id, codes[1], "unsure", origin="human_focus_coder")

    progress = project.focus_progress(session_id)
    assert progress["complete"]
    assert progress["complete_units"] == 3
    assert progress["resolved_judgments"] == 6
    assert progress["unsure_judgments"] == 3


def test_done_enters_unresolved_review_then_closes_session(tmp_path: Path) -> None:
    project = _project(tmp_path)
    codes = [int(row["code_id"]) for row in project.codes()]
    session_id = project.configure_focus(codes=codes, allow_unsure=False)

    # Complete the first two documents, leaving the third unresolved.
    for unit in project.units()[:2]:
        observation_id = int(unit["observation_id"])
        for code_id in codes:
            project.annotate(observation_id, code_id, "negative", origin="human_focus_coder")

    first_done = project.complete_focus(session_id)
    assert first_done["status"] == "open"
    assert first_done["review_unresolved"] is True
    assert first_done["current_unit_id"] == 3
    assert project.focus_task(session_id)["review_unresolved"] is True

    third = project.unit(3)
    for code_id in codes:
        project.annotate(int(third["observation_id"]), code_id, "positive", origin="human_focus_coder")

    second_done = project.complete_focus(session_id)
    assert second_done["complete"]
    assert second_done["status"] == "complete"
    task = project.focus_task(session_id)
    assert task["status"] == "complete"
    assert task["completed_at"] is not None

    project.reopen_focus(session_id)
    assert project.focus_task(session_id)["status"] == "open"


def test_configure_focus_is_idempotent_for_same_protocol(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_ids = [int(row["code_id"]) for row in project.codes()]
    first = project.configure_focus(codes=code_ids, allow_unsure=False)
    second = project.configure_focus(codes=code_ids, allow_unsure=False)
    assert second == first
    with pytest.raises(ValueError, match="different focus task"):
        project.configure_focus(codes=code_ids[:1], allow_unsure=False)


def test_public_launch_focus_coder_uses_single_process_default(monkeypatch, tmp_path: Path) -> None:
    import geometric_coder.ui as ui

    project = _project(tmp_path)
    project.configure_focus(codes=[int(row["code_id"]) for row in project.codes()])
    captured: dict[str, object] = {}

    def fake_launch(project_arg, **kwargs):
        assert project_arg is project
        captured.update(kwargs)

    monkeypatch.setattr(ui, "launch_focus_coder", fake_launch)
    project.launch_focus_coder(port=8123, debug=True)

    assert captured["port"] == 8123
    assert captured["debug"] is True
    assert captured["use_reloader"] is False
    assert captured["session_id"] == project.sessions()[0]["session_id"]


def test_focus_mode_dash_layout_hides_unsure_when_forbidden(tmp_path: Path) -> None:
    pytest.importorskip("dash")
    project = _project(tmp_path)
    project.configure_focus(
        codes=[int(row["code_id"]) for row in project.codes()],
        allow_unsure=False,
    )
    from geometric_coder.ui import create_app

    app = create_app(project, mode="focus")
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    layout_text = client.get("/_dash-layout").get_data(as_text=True)
    assert "GeCo" in layout_text
    assert "Focus Coding Session" in layout_text
    assert "Finite blind-coding task" not in layout_text
    assert "Next Incomplete" in layout_text
    assert "Auto-advance" in layout_text
    assert '"value":"unsure"' not in layout_text



def test_auto_advance_targets_next_incomplete_only_on_completion_transition() -> None:
    from geometric_coder.ui.focus_coding import _auto_advance_target

    required = [1, 2, 3, 4]
    before = {
        "unresolved": [
            {"unit_id": 2},
            {"unit_id": 2},
            {"unit_id": 4},
        ]
    }
    after = {"unresolved": [{"unit_id": 4}]}
    assert _auto_advance_target(required, before, after, 2) == 4

    # Editing a document that was already complete should not throw the coder forward.
    assert _auto_advance_target(required, before, after, 1) is None

    # If the current document is still incomplete, stay on it.
    still_incomplete = {"unresolved": [{"unit_id": 2}, {"unit_id": 4}]}
    assert _auto_advance_target(required, before, still_incomplete, 2) is None

    # Finishing the final outstanding document leaves the coder in place for Done.
    assert _auto_advance_target(required, {"unresolved": [{"unit_id": 4}]}, {"unresolved": []}, 4) is None

def test_focus_launch_reopens_if_completed_matrix_becomes_unresolved(monkeypatch, tmp_path: Path) -> None:
    import geometric_coder.ui as ui

    project = _project(tmp_path)
    codes = [int(row["code_id"]) for row in project.codes()]
    session_id = project.configure_focus(codes=codes, allow_unsure=False)
    for unit in project.units():
        observation_id = int(unit["observation_id"])
        for code_id in codes:
            project.annotate(observation_id, code_id, "negative", origin="human_focus_coder")
    assert project.complete_focus(session_id)["status"] == "complete"

    first = project.units()[0]
    project.annotate(
        int(first["observation_id"]),
        codes[0],
        "unsure",
        origin="human_explore",
    )
    assert project.focus_task(session_id)["status"] == "complete"
    assert not project.focus_progress(session_id)["complete"]

    monkeypatch.setattr(ui, "launch_focus_coder", lambda *args, **kwargs: None)
    project.launch_focus_coder(session_id=session_id)
    assert project.focus_task(session_id)["status"] == "open"
