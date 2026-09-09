from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import CountGeometry, GeometricCoder
from geometric_coder.storage import SCHEMA_VERSION


def test_current_schema_is_clean_pre1_boundary() -> None:
    assert SCHEMA_VERSION == 13
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    assert not list(scripts_dir.glob("*migrate*"))


def test_schema_mismatch_requires_recreation_not_migration(tmp_path: Path) -> None:
    project = GeometricCoder.create(
        project_dir=tmp_path / "schema.geco",
        data=pd.DataFrame({"row_id": [0, 1], "text": ["alpha", "beta"]}),
        keys=["row_id"],
        text="text",
        metadata=[],
        geometries={"lexical": CountGeometry(lemmatize=False)},
    )
    database_path = project.project_dir / project.DATABASE_FILENAME
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE project SET schema_version = 12 WHERE project_id = 1")
        connection.commit()

    with pytest.raises(RuntimeError, match="does not carry backward-compatibility or migration architecture"):
        GeometricCoder.open(project.project_dir)
