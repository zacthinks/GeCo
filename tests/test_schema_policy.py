from __future__ import annotations

import sqlite3
from pathlib import Path

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
    assert SCHEMA_VERSION == 14
    assert set(storage_database._MIGRATIONS) == {13}


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
    assert _version_and_topics(path) == (14, True)
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
    assert _version_and_topics(project.project_dir / project.DATABASE_FILENAME) == (14, True)


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
    assert _version_and_topics(path) == (14, True)


def test_unsupported_older_and_newer_versions_fail_without_changes(
    tmp_path: Path,
) -> None:
    project = _make_workspace(tmp_path)
    path = project.project_dir / project.DATABASE_FILENAME
    for version in (12, 15):
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
