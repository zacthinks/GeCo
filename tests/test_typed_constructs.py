"""Typed Construct identity and schema migration regression contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.storage import database as storage


def _project(tmp_path: Path, *, external: bool = False) -> GeometricCoder:
    data = pd.DataFrame({"row_id": [1, 2, 3, 4], "text": ["a", "b", "c", "d"]})
    if external:
        return GeometricCoder.create_external(
            project_dir=tmp_path / "project.geco", data=data,
            keys=["row_id"], text="text", metadata=[], external_provider=object(),
        )
    return GeometricCoder.create(
        project_dir=tmp_path / "project.geco", data=data,
        keys=["row_id"], text="text", metadata=[], geometries=None,
    )


def _revert_construct_tables(path: Path) -> None:
    """Simulate real schema 18: remove all tables that were not present then."""
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE scale_comparison_events")
        connection.execute("DROP TABLE scale_versions")
        connection.execute("DROP TABLE scales")
        connection.execute("DROP TABLE constructs")
        connection.execute(
            "UPDATE project SET schema_version = 18 WHERE project_id = 1"
        )


def _structure(path: Path) -> dict[str, list[tuple]]:
    with sqlite3.connect(path) as connection:
        names = ("constructs", "scales", "scale_versions", "scale_comparison_events")
        result = {}
        for name in names:
            result[name] = [
                tuple(row) for row in connection.execute(f"PRAGMA table_info({name})")
            ]
            result[name + "_fk"] = [
                tuple(row) for row in connection.execute(f"PRAGMA foreign_key_list({name})")
            ]
            result[name + "_index"] = [
                tuple(row) for row in connection.execute(f"PRAGMA index_list({name})")
            ]
        return result


def test_fresh_project_has_schema_19_and_no_initial_constructs(tmp_path: Path) -> None:
    project = _project(tmp_path)
    assert project.database.project_metadata()["schema_version"] == 19
    assert project.constructs() == []
    assert project.codes() == []
    assert _structure(project.project_dir / project.DATABASE_FILENAME)


def test_interleaved_code_scale_code_ids_and_backward_compatible_code_shapes(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    code_a = project.create_code("First", "first definition")
    original = project.code(code_a)
    scale_id = project.create_scale("Intensity", "continuous dimension")
    code_b = project.create_code("Second", "second definition")
    assert len({code_a, scale_id, code_b}) == 3
    assert code_a < scale_id < code_b
    assert [row["construct_id"] for row in project.constructs()] == [
        code_a, scale_id, code_b
    ]
    assert [row["type"] for row in project.constructs()] == [
        "code", "scale", "code"
    ]
    assert project.code(code_a) == original
    assert [row["code_id"] for row in project.codes()] == [code_a, code_b]
    assert project.construct(code_a)["version_id"] == original["code_version_id"]
    assert project.construct(code_a)["description"] == "first definition"
    assert project.construct(scale_id)["scale_id"] == scale_id
    assert project.construct(scale_id)["eligibility"] == {"kind": "all"}
    assert project.construct(scale_id)["eligibility_revision"] == 1
    assert project.scale_versions(scale_id)[0]["version_number"] == 1
    with project.database.connect() as connection:
        ids = [
            tuple(row) for row in connection.execute(
                "SELECT construct_id, type FROM constructs ORDER BY construct_id"
            )
        ]
        assert ids == [(code_a, "code"), (scale_id, "scale"), (code_b, "code")]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_scale_without_code_supports_version_history_and_explicit_eligibility(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    scale = project.create_scale("Quality", "initial", eligibility={"kind": "all"})
    assert project.codes() == []
    v1 = project.scale(scale)
    assert v1["name"] == "Quality"
    assert v1["description"] == "initial"
    v2_id = project.update_scale(scale, "Argument quality", "more precise")
    v2 = project.scale(scale)
    assert v2["scale_version_id"] == v2_id
    assert v2["version_number"] == 2
    assert v2["name"] == "Argument quality"
    assert project.scale(scale, 1) == v1 | {
        "updated_at": v2["updated_at"]
    }
    assert [r["name"] for r in project.scale_versions(scale)] == [
        "Quality", "Argument quality"
    ]
    assert project.construct(scale, 1)["name"] == "Quality"
    assert project.construct(scale, 2)["description"] == "more precise"
    assert project.scale(scale)["eligibility_revision"] == 1
    with pytest.raises(KeyError):
        project.construct(scale, 3)
    with pytest.raises(KeyError):
        project.scale_versions(123456)


def test_code_versions_still_are_authoritative_for_construct_projection(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    code = project.create_code("Theme", "old")
    project.update_code(code, name="Theme revised", description="new")
    assert [v["name"] for v in project.code_versions(code)] == [
        "Theme", "Theme revised"
    ]
    assert project.construct(code, 1)["name"] == "Theme"
    assert project.construct(code, 2)["description"] == "new"
    assert project.construct(code)["version_number"] == 2
    assert "construct_id" not in project.code(code)
    with pytest.raises(KeyError):
        project.construct(987654)


@pytest.mark.parametrize("external", [False, True])
def test_real_schema_18_upgrade_preserves_dependent_rows_and_identity(
    tmp_path: Path, external: bool,
) -> None:
    project = _project(tmp_path, external=external)
    first = project.create_code("Analysis", "first")
    project.update_code(first, name="Analysis 2", description="second")
    second = project.create_code("Other", "other")
    observation = project.observation_id_for_unit(1)
    project.annotate(observation, first, "positive")
    geometry_id = project.database.register_geometry(
        name="migration_geometry", class_name="CountGeometry", modality="text",
        public=True, config={}, dependency_names=[], matrix_path="migration.npz",
        matrix_sparse=True, state_path=None, supports_query=True,
        supports_text_transform=True,
    )
    spec_id = project.create_classifier_spec(
        code_id=first, name="Preserved classifier", geometry_id=geometry_id,
    )
    fit_id = project.database.register_classifier_fit(
        code_id=first, classifier_spec_id=spec_id,
        classifier_name="Preserved classifier", geometry_id=geometry_id,
        algorithm="logistic_l2", hyperparameters={"regularization": 1.0},
        training_snapshot={"annotations": []}, artifact_path="models/preserved.joblib",
        score_kind="probability",
        predictions=[{
            "observation_id": observation, "predicted_label": "positive",
            "probability": 0.8, "decision_score": None, "uncertainty": 0.4,
        }],
    )
    session_id = project.sessions()[0]["session_id"]
    project.patch_session_state(int(session_id), {"saved_key": "untouched"})
    with project.database.connect() as connection:
        base = {}
        for table in ("codes", "code_versions", "annotation_events", "sessions",
                      "classifier_specs", "classifier_fits", "classifier_predictions",
                      "classifier_committees", "apply_runs", "saved_analyses"):
            base[table] = [
                tuple(row) for row in connection.execute(f"SELECT * FROM {table}")
            ]
    path = project.project_dir / project.DATABASE_FILENAME
    fresh_shape = _structure(path)
    _revert_construct_tables(path)
    reopened = GeometricCoder.open(
        project.project_dir,
        external_provider=object() if external else None,
    )
    assert reopened.database.project_metadata()["schema_version"] == 19
    assert _structure(path) == fresh_shape
    assert [
        (c["construct_id"], c["type"]) for c in reopened.constructs()
    ] == [(first, "code"), (second, "code")]
    assert reopened.code_versions(first)[0]["name"] == "Analysis"
    assert reopened.current_annotation(observation_id=observation, code_id=first)["value"] == "positive"
    assert reopened.database.get_classifier_spec(spec_id)["code_id"] == first
    assert reopened.database.get_classifier_fit(fit_id)["artifact_path"] == "models/preserved.joblib"
    assert reopened.database.classifier_predictions(fit_id)[0]["probability"] == pytest.approx(0.8)
    assert reopened.sessions()[0]["state"]["saved_key"] == "untouched"
    with reopened.database.connect() as connection:
        for table, values in base.items():
            assert [tuple(r) for r in connection.execute(f"SELECT * FROM {table}")] == values
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    reopened_again = GeometricCoder.open(
        project.project_dir,
        external_provider=object() if external else None,
    )
    assert reopened_again.constructs() == reopened.constructs()
    next_scale = reopened.create_scale("Independent")
    next_code = reopened.create_code("Third")
    assert len({first, second, next_scale, next_code}) == 4


def test_schema_18_upgrade_rolls_back_all_ddl_and_backfill(tmp_path: Path, monkeypatch) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("Existing")
    path = project.project_dir / project.DATABASE_FILENAME
    _revert_construct_tables(path)
    original = storage._MIGRATIONS[18]

    def fail_after_upgrade(connection: sqlite3.Connection) -> None:
        original(connection)
        raise RuntimeError("forced construct migration failure")

    monkeypatch.setitem(storage._MIGRATIONS, 18, fail_after_upgrade)
    with pytest.raises(RuntimeError, match="forced construct migration failure"):
        GeometricCoder.open(project.project_dir)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT schema_version FROM project").fetchone()[0] == 18
        for name in ("constructs", "scales", "scale_versions", "scale_comparison_events"):
            assert connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (name,),
            ).fetchone() is None
        assert connection.execute("SELECT code_id FROM codes").fetchone()[0] == code_id
    monkeypatch.setitem(storage._MIGRATIONS, 18, original)
    assert GeometricCoder.open(project.project_dir).construct(code_id)["type"] == "code"


def test_new_scale_inputs_fail_atomically(tmp_path: Path) -> None:
    project = _project(tmp_path)
    with pytest.raises(ValueError):
        project.create_scale("  ")
    with pytest.raises(TypeError):
        project.create_scale("Broken", eligibility=["not", "mapping"])
    with pytest.raises(ValueError):
        project.create_scale("Broken", eligibility={"kind": None})
    with pytest.raises(TypeError):
        project.create_scale("Broken", eligibility={"kind": "all", "x": object()})
    assert project.constructs() == []
    scale = project.create_scale("Valid")
    with pytest.raises(ValueError):
        project.update_scale(scale, "  ", "invalid")
    assert len(project.scale_versions(scale)) == 1
