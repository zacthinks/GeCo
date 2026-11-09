"""Random Develop navigation must remain bounded on large linked corpora."""

from __future__ import annotations

import pandas as pd
import pytest

from geometric_coder import GeometricCoder


def _project(tmp_path, count=200):
    project = GeometricCoder.create(
        project_dir=tmp_path / "random-fast.geco",
        data=pd.DataFrame({
            "key": list(range(count)),
            "text": [f"document {i}" for i in range(count)],
        }),
        keys=["key"], text="text", metadata=[], geometries=None,
    )
    code = project.create_code("Example")
    session = project.create_session("Develop")
    return project, code, session


def test_random_never_fetches_all_units_or_annotations(tmp_path, monkeypatch):
    project, code, session = _project(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Random attempted to materialize the full corpus")

    monkeypatch.setattr(project, "units", forbidden)
    monkeypatch.setattr(project.database, "list_units", forbidden)
    monkeypatch.setattr(project.database, "current_annotations", forbidden)
    monkeypatch.setattr(project.database, "seen_unit_ids", forbidden)
    monkeypatch.setattr(project, "geometry_matrix", forbidden)

    result = project.focus_recommendation(
        code_id=code, session_id=session, strategy="random", random_seed=13,
    )
    assert result is not None
    assert 1 <= result["unit_id"] <= 200
    assert result["strategy"] == "random"
    assert result["fallback"] is False


def test_random_excludes_seen_and_current_annotations(tmp_path):
    project, code, session = _project(tmp_path, 8)
    project.record_visit(session, 1, method="test")
    for unit_id in (2, 3, 4):
        project.annotate(
            project.observation_id_for_unit(unit_id),
            code,
            ("positive" if unit_id == 2 else
             "negative" if unit_id == 3 else "unsure"),
        )
    selected = []
    for _ in range(4):
        result = project.focus_recommendation(
            code_id=code, session_id=session, strategy="random", random_seed=19,
        )
        assert result is not None
        unit_id = int(result["unit_id"])
        assert unit_id in {5, 6, 7, 8}
        assert unit_id not in selected
        selected.append(unit_id)
        project.record_visit(session, unit_id, method="test")
    assert set(selected) == {5, 6, 7, 8}
    assert project.focus_recommendation(
        code_id=code, session_id=session, strategy="random",
    ) is None


def test_random_has_independent_session_and_code_eligibility(tmp_path):
    project, code, session = _project(tmp_path, 4)
    other_code = project.create_code("Other")
    other_session = project.create_session("Other session")
    project.annotate(project.observation_id_for_unit(1), code, "unsure")
    project.record_visit(session, 2, method="test")

    assert project.database.random_available_unit(
        code_id=code, session_id=session, random_seed=7,
    ) in {3, 4}
    for unit_id in (3, 4):
        project.record_visit(session, unit_id, method="test")
    assert project.database.random_available_unit(
        code_id=code, session_id=session, random_seed=7,
    ) is None
    assert project.database.random_available_unit(
        code_id=other_code, session_id=other_session, random_seed=7,
    ) in {1, 2, 3, 4}


def test_random_finds_last_eligible_unit_without_large_candidate_list(tmp_path):
    project, code, session = _project(tmp_path, 100)
    for unit_id in range(1, 100):
        project.record_visit(session, unit_id, method="test")
    result = project.focus_recommendation(
        code_id=code, session_id=session, strategy="random", random_seed=100,
    )
    assert result is not None
    assert result["unit_id"] == 100


def test_random_has_stable_session_cursor_after_reopen(tmp_path):
    from geometric_coder.explore import deterministic_page
    project, code, session = _project(tmp_path, 73)
    # The same session/code pair follows one permutation, with no replacement.
    seed = ((session * 2654435761) ^ (code * 2246822519)) & 0xFFFFFFFF
    expected = (deterministic_page(73, page_size=73, seed=seed).positions + 1).tolist()
    selected = []
    for index in range(15):
        if index == 8:
            project = GeometricCoder.open(project.project_dir)
        result = project.focus_recommendation(
            code_id=code, session_id=session, strategy="random",
        )
        assert result is not None
        selected.append(result["unit_id"])
        project.record_visit(
            session, result["unit_id"], method="focus_random",
            active_code_id=code,
        )
    assert selected == expected[:15]


def test_random_skips_already_visited_and_annotated_in_permutation_order(tmp_path):
    from geometric_coder.explore import deterministic_page
    project, code, session = _project(tmp_path, 47)
    seed = ((session * 2654435761) ^ (code * 2246822519)) & 0xFFFFFFFF
    ordered = (deterministic_page(47, page_size=47, seed=seed).positions + 1).tolist()
    project.record_visit(session, ordered[0], method="other")
    project.annotate(project.observation_id_for_unit(ordered[1]), code, "unsure")
    result = project.focus_recommendation(
        code_id=code, session_id=session, strategy="random",
    )
    assert result["unit_id"] == ordered[2]


def test_random_cursor_has_an_index_on_reopened_old_workspace(tmp_path):
    project, _, _ = _project(tmp_path)
    with project.database.connect() as connection:
        connection.execute("DROP INDEX navigation_random_cursor_idx")
    reopened = GeometricCoder.open(project.project_dir)
    with reopened.database.connect() as connection:
        indexes = {
            str(row["name"])
            for row in connection.execute("PRAGMA index_list('navigation_events')")
        }
    assert "navigation_random_cursor_idx" in indexes
