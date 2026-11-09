"""Regression checks for bounded document and visualization pagination."""

import numpy as np
import pandas as pd

from geometric_coder import GeometricCoder
from geometric_coder.explore import (
    deterministic_page,
    deterministic_page_rank,
)
from geometric_coder.ui.app import _LazyTexts, _LazyUnits, _metadata_filtered_page
from geometric_coder.explore import combine_filters


def _project(tmp_path, count=113):
    return GeometricCoder.create(
        project_dir=tmp_path / "bounded.geco",
        data=pd.DataFrame({
            "key": list(range(count)),
            "text": [f"document {index}" for index in range(count)],
            "category": ["a" if index % 3 == 0 else "b" for index in range(count)],
        }),
        keys=["key"],
        text="text",
        metadata=["category"],
        geometries=None,
    )


def test_bounded_position_reads_preserve_order_duplicates_and_text(tmp_path):
    project = _project(tmp_path)
    db = project.database
    assert db.unit_count() == 113
    requested = [91, 1, 91, 0]
    rows = db.unit_rows_at_positions(requested)
    assert [row["row_position"] for row in rows] == requested
    assert [row["text"] for row in rows] == [
        "document 91", "document 1", "document 91", "document 0",
    ]
    assert db.unit_rows_at_positions([]) == []
    assert db.unit_positions_for_ids([92, 2, 999]) == {92: 91, 2: 1}
    assert db.unit_position_for_id(999) is None


def test_default_paging_never_calls_full_unit_list(tmp_path, monkeypatch):
    project = _project(tmp_path, 100)
    def reject_full_load(*_args, **_kwargs):
        raise AssertionError("Unfiltered pagination loaded the entire corpus")
    monkeypatch.setattr(project.database, "list_units", reject_full_load)
    lazy = _LazyUnits(project.database)
    texts = _LazyTexts(lazy)
    assert len(lazy) == 100
    assert len(texts) == 100
    mask = combine_filters(texts)
    assert mask.eligible.shape == (100,)
    positions, page, eligible, errors = _metadata_filtered_page(
        lazy, [], operator="and", page_size=11, page_number=3,
    )
    assert len(positions) == 11
    assert page.page_index == 2
    assert eligible.all()
    assert errors == []
    assert [row["row_position"] for row in lazy.at_positions(positions.tolist())] == positions.tolist()


def test_filtered_paging_streams_only_metadata(tmp_path, monkeypatch):
    project = _project(tmp_path)
    lazy = _LazyUnits(project.database)
    def reject_full_load(*_args, **_kwargs):
        raise AssertionError("Metadata filter fetched every full document")
    monkeypatch.setattr(project.database, "list_units", reject_full_load)
    filters = [{
        "field": "category", "type": "categorical",
        "operator": "in", "value": ["a"], "value2": None,
    }]
    positions, page, mask, errors = _metadata_filtered_page(
        lazy, filters, operator="and", page_size=7, page_number=2,
    )
    assert errors == []
    assert int(mask.sum()) == 38
    assert page.page_count == 6
    assert len(positions) == 7
    assert all(position % 3 == 0 for position in positions)


def test_large_page_generation_is_bounded_and_inverse_is_exact(monkeypatch):
    def reject_rng(*_args, **_kwargs):
        raise AssertionError("Cannot allocate a corpus-wide permutation")
    monkeypatch.setattr("geometric_coder.explore.np.random.default_rng", reject_rng)
    n = 10_000_000
    p0 = deterministic_page(n, page_size=23, page_index=0, seed=7)
    p1 = deterministic_page(n, page_size=23, page_index=1, seed=7)
    assert len(p0.positions) == len(p1.positions) == 23
    assert set(p0.positions).isdisjoint(p1.positions)
    for rank, position in enumerate(p0.positions):
        assert deterministic_page_rank(n, int(position), seed=7) == rank
    for rank, position in enumerate(p1.positions, start=23):
        assert deterministic_page_rank(n, int(position), seed=7) == rank
