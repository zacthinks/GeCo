"""Regression coverage for bounded hierarchy context and SQL materialization."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.storage import database as database_module


def _project(tmp_path, *, rows=300):
    return GeometricCoder.create(
        project_dir=tmp_path / "context.geco",
        data=pd.DataFrame({
            "doc": ["shared"] * rows,
            "sentence": list(range(rows)),
            "text": [f"Sentence {index}: " + "text " * 10 for index in range(rows)],
            "category": ["even" if index % 2 == 0 else "odd" for index in range(rows)],
        }),
        keys=["doc", "sentence"], text="text",
        metadata=["category"], geometries=None,
    )


@pytest.mark.parametrize("cap,expected", [
    (1, [153]), (2, [152, 153]), (5, list(range(151, 156))),
    (6, list(range(150, 156))), (19, list(range(144, 163))),
])
def test_nearest_context_includes_focal_and_preserves_source_order(tmp_path, cap, expected):
    project = _project(tmp_path)
    # Unit IDs are one-based; row positions are zero-based.
    focal_id = 154
    full = project.context(focal_id, level=0, window=1)
    assert len(full) == 300
    bounded = project.context(focal_id, level=0, window=1, max_units=cap)
    assert [row["row_position"] for row in bounded] == expected
    assert [row["unit_id"] for row in bounded] == [index + 1 for index in expected]
    assert bounded == [full[index] for index in expected]
    assert [row["row_position"] for row in project.context(
        focal_id, level=1, window=1, max_units=cap,
    )] == [153]


@pytest.mark.parametrize("focal,cap,expected", [
    (1, 5, [0, 1, 2, 3, 4]),
    (300, 5, [295, 296, 297, 298, 299]),
    (2, 6, [0, 1, 2, 3, 4, 5]),
    (299, 6, [294, 295, 296, 297, 298, 299]),
    (154, 500, list(range(300))),
])
def test_context_edges_and_cap_larger_than_group(tmp_path, focal, cap, expected):
    project = _project(tmp_path)
    assert [row["row_position"] for row in project.context(
        focal, level=0, max_units=cap,
    )] == expected


def test_context_sibling_window_and_interleaved_source_order(tmp_path):
    # Input rows are interleaved, but hierarchy groups follow first-seen keys.
    records = []
    for index in range(12):
        records.append({
            "doc": ("A", "B", "C")[index % 3],
            "paragraph": index // 3,
            "sentence": index,
            "text": f"record {index}",
        })
    project = GeometricCoder.create(
        project_dir=tmp_path / "mixed.geco",
        data=pd.DataFrame(records), keys=["doc", "paragraph", "sentence"],
        text="text", metadata=[], geometries=None,
    )
    # At the document level, a window of 2 includes A and B, not C.
    full = project.context(5, level=0, window=2)
    assert [row["row_position"] for row in full] == [0, 1, 3, 4, 6, 7, 9, 10]
    bounded = project.context(5, level=0, window=2, max_units=5)
    assert [row["row_position"] for row in bounded] == [1, 3, 4, 6, 7]
    # A paragraph-level window is restricted to the focal document.
    paragraph = project.context(5, level=1, window=3)
    assert [row["row_position"] for row in paragraph] == [1, 4, 7]
    assert project.context(5, level=1, window=3, max_units=2) == paragraph[:2]
    assert [row["row_position"] for row in project.context(
        5, level=2, window=3, max_units=2,
    )] == [4]


@pytest.mark.parametrize("invalid", [0, -1, 1.5, "2", True])
def test_invalid_max_units_rejected(tmp_path, invalid):
    project = _project(tmp_path, rows=3)
    with pytest.raises(ValueError, match="max_units"):
        project.context(1, level=0, max_units=invalid)
    with pytest.raises(ValueError, match="max_units"):
        project.database.context_units(unit_id=1, level_index=0, max_units=invalid)
    assert len(project.context(1, level=0)) == 3


def test_bounded_path_decodes_only_selected_units(tmp_path, monkeypatch):
    project = _project(tmp_path, rows=420)
    decoded = []
    original = database_module._unit_row

    def record(row: sqlite3.Row):
        decoded.append(int(row["row_position"]))
        return original(row)

    monkeypatch.setattr(database_module, "_unit_row", record)
    bounded = project.context(211, level=0, max_units=17)
    assert len(bounded) == 17
    assert decoded == [row["row_position"] for row in bounded]
    assert len(decoded) < 420


def test_ui_context_calls_are_bounded_at_every_reader():
    """Every current UI context reader must use the capped, lookahead helper."""
    from pathlib import Path
    import ast
    import geometric_coder.ui.app as app_module
    import geometric_coder.ui.scale_comparisons as comparison_module
    import geometric_coder.ui.scale_prediction as prediction_module
    import geometric_coder.ui.context_expansion as expansion_module

    counts = {}
    for module in (app_module, comparison_module, prediction_module):
        path = Path(module.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "read_context"
        ]
        counts[path.name] = len(calls)
        assert not any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "context"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "project"
            for node in ast.walk(tree)
        ), path.name
    assert counts == {"app.py": 5, "scale_comparisons.py": 1, "scale_prediction.py": 1}
    source = ast.parse(Path(expansion_module.__file__).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(source) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "context"
    ]
    assert len(calls) == 1
    assert "max_units" in {kw.arg for kw in calls[0].keywords}
