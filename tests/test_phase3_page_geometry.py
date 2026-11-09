"""Phase 3: bounded Explore geometry rows preserve navigation ranking."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from geometric_coder import GeometricCoder
from geometric_coder.explore import recommend_page_unit


@pytest.mark.parametrize("strategy", ["nearest", "farthest"])
@pytest.mark.parametrize("use_sparse", [False, True])
def test_page_matrix_ranks_same_atomic_unit_as_full_corpus_matrix(strategy, use_sparse):
    unit_ids = [10, 20, 30, 40, 50, 60]
    positions = np.array([5, 1, 3, 0], dtype=int)
    dense = np.array([[0.0, 0.0], [1.0, 0.0], [9.0, 9.0],
                      [3.0, 0.0], [8.0, 8.0], [6.0, 0.0]])
    matrix = sparse.csr_matrix(dense) if use_sparse else dense
    requested = matrix[positions]
    common = {
        "strategy": strategy,
        "unit_ids": unit_ids,
        "page_positions": positions,
        "seen_unit_ids": {40},
        "eligible": [True, True, True, True, True, True],
        "focal_unit_id": 20,
    }
    expected = recommend_page_unit(**common, matrix=matrix)
    actual = recommend_page_unit(**common, page_matrix=requested)
    assert actual == expected
    with pytest.raises(ValueError, match="either matrix or page_matrix"):
        recommend_page_unit(**common, matrix=matrix, page_matrix=requested)
    with pytest.raises(ValueError, match="one row"):
        recommend_page_unit(**common, page_matrix=requested[:2])


def test_dense_local_geometry_rows_use_mmap_and_preserve_order(tmp_path, monkeypatch):
    project = GeometricCoder.create(
        project_dir=tmp_path / "dense-rows.geco",
        data=pd.DataFrame({"row_id": [8, 3, 9, 4], "text": ["a", "b", "c", "d"]}),
        keys=["row_id"], text="text",
        geometries=None,
    )
    values = np.array([[2.0, 4.0], [8.0, 1.0], [5.0, 7.0], [0.0, 2.0]])
    np.save(project.project_dir / "dense-nav.npy", values)
    geometry_id = project.database.register_geometry(
        name="dense_nav", class_name="CountGeometry", modality="text",
        public=True, config={}, dependency_names=[],
        matrix_path="dense-nav.npy", matrix_sparse=False,
        state_path=None, supports_query=False, supports_text_transform=False,
    )
    positions = [3, 1, 3, 0]
    captured = []
    original_load = np.load

    def load(*args, **kwargs):
        captured.append(kwargs.get("mmap_mode"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", load)
    result = project.geometry_page_rows(geometry_id, positions)
    assert result.tolist() == values[positions].tolist()
    assert captured == ["r"]
    assert project.geometry_page_rows(geometry_id, []).shape == (0, 2)


def test_external_provider_reads_only_requested_page_keys(tmp_path):
    class Provider:
        def __init__(self):
            self.calls = []

        def geometry_matrix(self, external_ref, ordered_user_keys):
            self.calls.append((external_ref, ordered_user_keys))
            lookup = {9: [10.0], 2: [20.0], 7: [30.0]}
            return np.array([lookup[key["id"]] for key in ordered_user_keys], dtype=float)

        def view_coordinates(self, external_ref, ordered_user_keys):
            raise AssertionError("No view should be requested")

    provider = Provider()
    project = GeometricCoder.create_external(
        project_dir=tmp_path / "provider-rows.geco",
        data=pd.DataFrame({
            "id": [9, 2, 7], "text": ["alpha", "beta", "gamma"]
        }),
        keys=["id"], text="text", metadata=[], external_provider=provider,
    )
    geometry_id = project.register_external_geometry(
        name="external_dense", external_ref={"artifact": "external"},
    )
    # Registration uses full-key validation; a page read must not.
    provider.calls.clear()
    rows = project.geometry_page_rows(geometry_id, [2, 0])
    assert rows.tolist() == [[30.0], [10.0]]
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == {"artifact": "external"}
    assert provider.calls[0][1] == [{"id": 7}, {"id": 9}]
    assert project.geometry_page_rows(geometry_id, []).shape == (0, 1)
    assert len(provider.calls) == 1
