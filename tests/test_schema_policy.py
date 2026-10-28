from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.storage import SCHEMA_VERSION
from geometric_coder.storage import database as storage_database


def _make_workspace(tmp_path: Path, *, external: bool = False) -> GeometricCoder:
    data = pd.DataFrame({"row_id": [0, 1], "text": ["alpha", "beta"]})
    if external:
        return GeometricCoder.create_external(
            project_dir=tmp_path / "external.geco", data=data, keys=["row_id"],
            text="text", metadata=[], external_provider=object(),
        )
    return GeometricCoder.create(
        project_dir=tmp_path / "schema.geco", data=data, keys=["row_id"],
        text="text", metadata=[], geometries=None,
    )


def _simulate_existing_schema_13(path: Path) -> None:
    """A schema-13 database had no topic_models table."""
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE topic_models")
        connection.execute("DROP TABLE representation_legacy_refs")
        connection.execute("DROP TABLE representations")
        connection.execute(
            "UPDATE project SET schema_version = 13 WHERE project_id = 1"
        )


def _version_and_topics(path: Path) -> tuple[int, bool]:
    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT schema_version FROM project WHERE project_id = 1"
        ).fetchone()[0]
        has_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='topic_models'"
        ).fetchone() is not None
    return int(version), has_table


def test_current_schema_and_upgrade_registry() -> None:
    assert SCHEMA_VERSION == 15
    assert set(storage_database._MIGRATIONS) == {13, 14}


@pytest.mark.parametrize("external", [False, True])
def test_open_schema_13_upgrades_and_preserves_research_state(
    tmp_path: Path, external: bool
) -> None:
    project = _make_workspace(tmp_path, external=external)
    code_id = project.create_code("Interview theme")
    session = project.sessions()[0]
    project.patch_session_state(int(session["session_id"]), {"marker": "persisted"})
    unit_id = int(project.database.list_units()[0]["unit_id"])
    project.annotate_span([unit_id], code_id, "positive")
    path = project.project_dir / project.DATABASE_FILENAME
    with sqlite3.connect(path) as connection:
        baseline = {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("codes", "code_versions", "units", "sessions", "annotation_events")
        }
    _simulate_existing_schema_13(path)
    assert _version_and_topics(path) == (13, False)

    provider = object() if external else None
    reopened = GeometricCoder.open(project.project_dir, external_provider=provider)
    assert _version_and_topics(path) == (15, True)
    assert reopened.topic_models() == []
    with sqlite3.connect(path) as connection:
        for table, rows in baseline.items():
            assert connection.execute(f"SELECT * FROM {table}").fetchall() == rows
    assert reopened.code(code_id)["name"] == "Interview theme"
    assert reopened.sessions()[0]["state"]["marker"] == "persisted"
    # Reopening at the latest schema is a no-op and preserves the data.
    again = GeometricCoder.open(project.project_dir, external_provider=provider)
    assert again.topic_models() == []


def test_latest_schema_skips_migration_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_workspace(tmp_path)
    def unexpected(_connection: sqlite3.Connection) -> None:
        raise AssertionError("Current-schema open attempted a migration")
    monkeypatch.setitem(storage_database._MIGRATIONS, 13, unexpected)
    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.topic_models() == []
    assert _version_and_topics(project.project_dir / project.DATABASE_FILENAME) == (15, True)


def test_13_to_14_upgrade_is_transactional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_workspace(tmp_path)
    path = project.project_dir / project.DATABASE_FILENAME
    _simulate_existing_schema_13(path)
    original = storage_database._MIGRATIONS[13]
    def broken(connection: sqlite3.Connection) -> None:
        original(connection)
        raise RuntimeError("simulated migration failure")
    monkeypatch.setitem(storage_database._MIGRATIONS, 13, broken)
    with pytest.raises(RuntimeError, match="simulated migration failure"):
        GeometricCoder.open(project.project_dir)
    assert _version_and_topics(path) == (13, False)
    monkeypatch.setitem(storage_database._MIGRATIONS, 13, original)
    assert GeometricCoder.open(project.project_dir).topic_models() == []
    assert _version_and_topics(path) == (15, True)


def test_unsupported_older_and_newer_versions_fail_without_changes(
    tmp_path: Path,
) -> None:
    project = _make_workspace(tmp_path)
    path = project.project_dir / project.DATABASE_FILENAME
    for version in (12, 16):
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE project SET schema_version = ? WHERE project_id = 1",
                (version,),
            )
        with pytest.raises(RuntimeError, match="migration|newer schema"):
            GeometricCoder.open(project.project_dir)
        assert _version_and_topics(path) == (version, True)


def test_external_workspace_needs_provider_before_migrating(tmp_path: Path) -> None:
    project = _make_workspace(tmp_path, external=True)
    path = project.project_dir / project.DATABASE_FILENAME
    _simulate_existing_schema_13(path)
    with pytest.raises(Exception, match="external_provider"):
        GeometricCoder.open(project.project_dir)
    assert _version_and_topics(path) == (13, False)


def test_14_to_15_preserves_legacy_ids_and_view_parent(tmp_path: Path) -> None:
    project = _make_workspace(tmp_path)
    database = project.database
    geometry_id = database.register_geometry(
        name="base", class_name="CountGeometry", modality="text", public=True,
        config={}, dependency_names=[], matrix_path="base.npz", matrix_sparse=True,
        state_path=None, supports_query=True, supports_text_transform=True,
    )
    child_id = database.register_geometry(
        name="reduced", class_name="SVDGeometry", modality="text", public=True,
        config={}, dependency_names=["base"], matrix_path="reduced.npy",
        matrix_sparse=False, state_path=None, supports_query=False,
        supports_text_transform=False,
    )
    view_id = database.register_view(
        geometry_id=child_id, name="map", method="umap", parameters={},
        artifact_path="map.npy",
    )
    path = project.project_dir / project.DATABASE_FILENAME
    with sqlite3.connect(path) as con:
        con.execute("DROP TABLE representation_legacy_refs")
        con.execute("DROP TABLE representations")
        con.execute("UPDATE project SET schema_version=14 WHERE project_id=1")
    reopened = GeometricCoder.open(project.project_dir)
    with reopened.database.connect() as con:
        rows = con.execute(
            "SELECT representation_id, legacy_kind, legacy_id, "
            "parent_representation_id, analytic_enabled, dimensions "
            "FROM representations ORDER BY representation_id"
        ).fetchall()
        assert len(rows) == 3
        ids = {(r["legacy_kind"], r["legacy_id"]): r["representation_id"] for r in rows}
        assert rows[0]["legacy_id"] == geometry_id
        child = next(r for r in rows if r["legacy_kind"] == "geometry"
                     and r["legacy_id"] == child_id)
        view = next(r for r in rows if r["legacy_kind"] == "view"
                    and r["legacy_id"] == view_id)
        assert child["parent_representation_id"] == ids["geometry", geometry_id]
        assert view["parent_representation_id"] == ids["geometry", child_id]
        assert child["analytic_enabled"] == 1
        assert view["analytic_enabled"] == 0 and view["dimensions"] == 2
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    GeometricCoder.open(project.project_dir)
    with reopened.database.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM representations").fetchone()[0] == 3


def test_14_to_15_migration_rolls_back_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _make_workspace(tmp_path)
    path = project.project_dir / project.DATABASE_FILENAME
    with sqlite3.connect(path) as con:
        con.execute("DROP TABLE representation_legacy_refs")
        con.execute("DROP TABLE representations")
        con.execute("UPDATE project SET schema_version=14 WHERE project_id=1")
    original = storage_database._MIGRATIONS[14]
    def fail_after_schema_change(con: sqlite3.Connection) -> None:
        original(con)
        raise RuntimeError("forced rollback")
    monkeypatch.setitem(storage_database._MIGRATIONS, 14, fail_after_schema_change)
    with pytest.raises(RuntimeError, match="forced rollback"):
        GeometricCoder.open(project.project_dir)
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT schema_version FROM project").fetchone()[0] == 14
        assert con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='representations'"
        ).fetchone() is None


def test_unified_registry_transitive_views_and_cycle_guards(tmp_path: Path) -> None:
    project = _make_workspace(tmp_path)
    db = project.database
    a = db.register_geometry(
        name="A", class_name="count", modality="text", public=True,
        config={}, dependency_names=[], matrix_path="", matrix_sparse=False,
        state_path=None, supports_query=True, supports_text_transform=True,
    )
    b = db.register_geometry(
        name="B", class_name="svd", modality="text", public=True,
        config={}, dependency_names=["A"], matrix_path="", matrix_sparse=False,
        state_path=None, supports_query=True, supports_text_transform=True,
    )
    c = db.register_view(
        geometry_id=b, name="C", method="umap", parameters={},
        artifact_path="view.npy",
    )
    ra = db.representation_for_legacy("geometry", a)
    rb = db.representation_for_legacy("geometry", b)
    rc = db.representation_for_legacy("view", c)
    assert rb["parent_representation_id"] == ra["representation_id"]
    assert rc["parent_representation_id"] == rb["representation_id"]
    db.record_representation_dimensions(ra["representation_id"], 50)
    db.record_representation_dimensions(rb["representation_id"], 500)
    assert db.get_representation(rc["representation_id"])["dimensions"] == 2
    assert [x["name"] for x in db.eligible_view_representations(
        ra["representation_id"]
    )] == ["C"]
    assert [view["name"] for view in project.views(a, include_descendants=True)] == ["C"]
    assert project.views(a) == []
    assert [view["name"] for view in project.views(b)] == ["C"]
    db.record_representation_dimensions(rb["representation_id"], 500)
    with pytest.raises(ValueError, match="dimensions"):
        db.record_representation_dimensions(rb["representation_id"], 499)
    with pytest.raises(ValueError, match="parent"):
        db.assign_representation_parent(ra["representation_id"], rc["representation_id"])
    with pytest.raises(ValueError, match="parent"):
        db.assign_representation_parent(rc["representation_id"], ra["representation_id"])


def test_unified_registry_can_resolve_out_of_order_parent(tmp_path: Path) -> None:
    db = _make_workspace(tmp_path).database
    child_id = db.register_geometry(
        name="child", class_name="external", modality="text", public=True,
        config={}, dependency_names=[], matrix_path="", matrix_sparse=False,
        state_path=None, supports_query=False, supports_text_transform=False,
    )
    parent_id = db.register_geometry(
        name="parent", class_name="external", modality="text", public=True,
        config={}, dependency_names=[], matrix_path="", matrix_sparse=False,
        state_path=None, supports_query=False, supports_text_transform=False,
    )
    child = db.representation_for_legacy("geometry", child_id)
    parent = db.representation_for_legacy("geometry", parent_id)
    assert child["parent_representation_id"] is None
    db.assign_representation_parent(child["representation_id"], parent["representation_id"])
    assert db.get_representation(child["representation_id"])["parent_representation_id"] == parent["representation_id"]


def test_one_dimensional_display_jitter_is_stable_and_nonanalytic() -> None:
    import numpy as np
    from geometric_coder.ui.figures import build_geometry_figure

    units = [{"unit_id": 20, "user_key": {"id": 20}},
             {"unit_id": 10, "user_key": {"id": 10}}]
    original = np.array([[2.5], [4.5]], dtype=float)
    figure = build_geometry_figure(
        coordinates=original, units=units, eligible=[True, True]
    )
    active = figure.data[1]
    assert list(active.x) == [2.5, 4.5]
    jitter_by_unit = dict(zip(active.customdata, active.y, strict=True))
    reversed_figure = build_geometry_figure(
        coordinates=original[::-1], units=units[::-1], eligible=[True, True]
    )
    reversed_active = reversed_figure.data[1]
    assert dict(zip(reversed_active.customdata, reversed_active.y, strict=True)) == jitter_by_unit
    assert len(set(jitter_by_unit.values())) == 2
    assert original.shape == (2, 1)


def test_unified_registry_deduplicates_same_provider_artifact(tmp_path: Path) -> None:
    project = _make_workspace(tmp_path, external=True)
    db = project.database
    external_ref = {"artifact_id": "frozen-matrix-42"}
    geometry_id = db.register_geometry(
        name="two_dimensional", class_name="external", modality="text",
        public=True, config={"type": "external"}, dependency_names=[],
        matrix_path="", matrix_sparse=False, state_path=None,
        supports_query=False, supports_text_transform=False,
        storage_kind="external", external_ref=external_ref,
    )
    view_id = db.register_view(
        geometry_id=geometry_id, name="same_matrix", method="external",
        parameters={}, artifact_path="", storage_kind="external",
        external_ref=external_ref,
    )
    geometry = db.representation_for_legacy("geometry", geometry_id)
    view = db.representation_for_legacy("view", view_id)
    assert geometry["representation_id"] == view["representation_id"]
    assert len(db.list_representations()) == 1
    assert db.list_representations()[0]["analytic_enabled"]
    assert db.list_representations()[0]["dimensions"] == 2

    # Recreate a genuine schema-14 database to test the migration separately.
    with sqlite3.connect(project.project_dir / project.DATABASE_FILENAME) as con:
        con.execute("DROP TABLE representation_legacy_refs")
        con.execute("DROP TABLE representations")
        con.execute("UPDATE project SET schema_version=14 WHERE project_id=1")
    reopened = GeometricCoder.open(project.project_dir, external_provider=object())
    migrated_geometry = reopened.database.representation_for_legacy("geometry", geometry_id)
    migrated_view = reopened.database.representation_for_legacy("view", view_id)
    assert migrated_view["representation_id"] == migrated_geometry["representation_id"]
    assert len(reopened.database.list_representations()) == 1


def test_native_identity_view_reuses_geometry_identity(tmp_path: Path) -> None:
    db = _make_workspace(tmp_path).database
    geometry_id = db.register_geometry(
        name="native_2d", class_name="count", modality="text", public=True,
        config={}, dependency_names=[], matrix_path="fitted.npy",
        matrix_sparse=False, state_path=None, supports_query=True,
        supports_text_transform=True,
    )
    view_id = db.register_view(
        geometry_id=geometry_id, name="identity", method="identity",
        parameters={}, artifact_path="identity.npy",
    )
    assert (db.representation_for_legacy("view", view_id)["representation_id"]
            == db.representation_for_legacy("geometry", geometry_id)["representation_id"])
    assert len(db.list_representations()) == 1


def test_numeric_lineage_reconciles_out_of_order_teal_artifacts(tmp_path: Path) -> None:
    from geometric_coder.ui.representation_manager import (
        _teal_numeric_ancestors, reconcile_teal_numeric_parents,
    )

    project = _make_workspace(tmp_path, external=True)
    db = project.database
    operations = {
        "embeddings": "op_embed", "umap_50": "op_50",
        "umap_2": "op_2", "concat": "op_concat",
    }
    sources = {
        "op_embed": ["text"], "op_50": ["embeddings"],
        "op_2": ["umap_50"], "op_concat": ["embeddings", "counts"],
    }
    types = {
        "text": "text", "counts": "sparse_matrix",
        "embeddings": "dense_matrix", "umap_50": "dense_matrix",
        "umap_2": "dense_matrix", "concat": "dense_matrix",
    }
    catalog = SimpleNamespace(
        operation_for_artifact=lambda artifact_id: (
            {"operation_id": operations[artifact_id]}
            if artifact_id in operations else None
        ),
        operation_sources=lambda operation_id: [
            {"source_artifact_id": value} for value in sources[operation_id]
        ],
    )
    teal = SimpleNamespace(
        catalog=catalog,
        get_artifact=lambda artifact_id: SimpleNamespace(
            artifact_type=SimpleNamespace(value=types[artifact_id])
        ),
    )
    assert _teal_numeric_ancestors(teal, "umap_2") == ("umap_50", "embeddings")
    assert _teal_numeric_ancestors(teal, "embeddings") == ()
    assert _teal_numeric_ancestors(teal, "concat") == ()

    def register(name: str) -> int:
        gid = db.register_geometry(
            name=name, class_name="external", modality="text", public=True,
            config={}, dependency_names=[], matrix_path="", matrix_sparse=False,
            state_path=None, supports_query=False, supports_text_transform=False,
            storage_kind="external", external_ref={"artifact_id": name},
        )
        return db.representation_for_legacy("geometry", gid)["representation_id"]

    root_id = register("embeddings")
    leaf_id = register("umap_2")
    assert reconcile_teal_numeric_parents(project, teal) == 1
    assert db.get_representation(leaf_id)["parent_representation_id"] == root_id
    middle_id = register("umap_50")
    assert reconcile_teal_numeric_parents(project, teal) == 2
    assert db.get_representation(middle_id)["parent_representation_id"] == root_id
    assert db.get_representation(leaf_id)["parent_representation_id"] == middle_id
    concat_id = register("concat")
    assert reconcile_teal_numeric_parents(project, teal) == 0
    assert db.get_representation(concat_id)["parent_representation_id"] is None


def test_one_dimensional_representation_is_selectable_without_a_saved_view(tmp_path: Path) -> None:
    import numpy as np

    project = _make_workspace(tmp_path)
    file_path = project.project_dir / "one_dimensional.npy"
    np.save(file_path, np.array([[3.0], [7.0]], dtype=np.float32))
    geometry_id = project.database.register_geometry(
        name="one_dimensional", class_name="custom", modality="text",
        public=True, config={}, dependency_names=[], matrix_path="one_dimensional.npy",
        matrix_sparse=False, state_path=None, supports_query=False,
        supports_text_transform=False,
    )
    representation = project.database.representation_for_legacy("geometry", geometry_id)
    project.database.record_representation_dimensions(representation["representation_id"], 1)
    assert project.views(geometry_id) == []
    eligible = project.views(geometry_id, include_descendants=True)
    assert len(eligible) == 1
    assert eligible[0]["view_id"] == -int(representation["representation_id"])
    coordinates = project.view_coordinates(int(eligible[0]["view_id"]))
    assert coordinates.shape == (2, 1)
    assert list(coordinates[:, 0]) == [3.0, 7.0]
