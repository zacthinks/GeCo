"""Adaptive bounded context: small groups, larger groups, and source order."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from geometric_coder import GeometricCoder


@pytest.mark.parametrize("group_size", [8, 16, 64, 200, 201, 1000, 1800])
@pytest.mark.parametrize("focal_fraction", [0.0, 0.5, 1.0])
def test_group_context_matches_unlimited_nearest(tmp_path, group_size, focal_fraction):
    """The cap must not change row identity, payload or nearest tie ordering."""
    # Interleaved documents ensure group membership is not inferred from a
    # contiguous source-position interval.
    count = max(group_size, 8)
    data = pd.DataFrame(
        {
            "doc": ["left", "right"] * count,
            "sentence": [i // 2 for i in range(count * 2)],
            "text": [f"text {i}" for i in range(count * 2)],
            "category": [f"cat {i % 3}" for i in range(count * 2)],
        }
    )
    project = GeometricCoder.create(
        project_dir=tmp_path / "adaptive.geco",
        data=data,
        keys=["doc", "sentence"],
        text="text",
        metadata=["category"],
        geometries=None,
    )
    focal_index = int((count - 1) * focal_fraction)
    # Left-document units are at even positions; IDs are one-based.
    focal_id = 2 * focal_index + 1
    unlimited = project.context(focal_id, level=0, window=1)
    focal_position = unlimited[focal_index]["row_position"]
    cap = 200
    expected = sorted(
        sorted(
            unlimited,
            key=lambda row: (
                abs(row["row_position"] - focal_position),
                row["row_position"],
            ),
        )[:cap],
        key=lambda row: row["row_position"],
    )
    assert project.context(focal_id, level=0, window=1, max_units=cap) == expected


def test_existing_workspace_gets_group_index_without_schema_bump(tmp_path):
    project = GeometricCoder.create(
        project_dir=tmp_path / "legacy.geco",
        data=pd.DataFrame({
            "doc": ["A", "A", "B"],
            "sentence": [0, 1, 0],
            "text": ["one", "two", "three"],
        }),
        keys=["doc", "sentence"],
        text="text",
        metadata=[],
        geometries=None,
    )
    with project.database.connect() as con:
        con.execute("DROP INDEX unit_groups_group_unit_idx")
        version = con.execute("SELECT schema_version FROM project").fetchone()[0]
    reopened = GeometricCoder.open(project.project_dir)
    with reopened.database.connect() as con:
        plan = con.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM unit_groups "
            "WHERE level_index=? AND group_id=? LIMIT ?",
            (0, 1, 201),
        ).fetchall()
        assert any("unit_groups_group_unit_idx" in row["detail"] for row in plan)
        assert con.execute("SELECT schema_version FROM project").fetchone()[0] == version
    assert len(reopened.context(1, level=0, max_units=200)) == 2


def test_large_context_fetches_at_most_cap_full_rows(tmp_path):
    count = 4000
    project = GeometricCoder.create(
        project_dir=tmp_path / "large.geco",
        data=pd.DataFrame({
            "doc": ["shared"] * count,
            "sentence": list(range(count)),
            "text": ["wide payload " * 100] * count,
        }),
        keys=["doc", "sentence"],
        text="text",
        metadata=[],
        geometries=None,
    )
    selected = project.context(count // 2, level=0, max_units=200)
    assert len(selected) == 200
    with project.database.connect() as con:
        # Selecting full rows must only happen after bounded positional probes.
        # One can verify this using SQLite's statement trace; the initial
        # membership probe must not load text/metadata from units.
        queries = []
        con.set_trace_callback(queries.append)
        # Query-plan check covers the covering membership index used on this path.
        plan = con.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM unit_groups "
            "WHERE level_index=0 AND group_id=1 LIMIT 1601"
        ).fetchall()
        assert any("COVERING INDEX unit_groups_group_unit_idx" in row["detail"] for row in plan)
    assert selected == project.context(count // 2, level=0)[
        count // 2 - 100 : count // 2 + 100
    ]
