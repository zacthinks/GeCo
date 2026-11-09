"""Phase 4: bounded local view rows retain canonical coordinate alignment."""

import numpy as np
import pandas as pd
import pytest

from geometric_coder import GeometricCoder


def _local(tmp_path):
    project = GeometricCoder.create(
        project_dir=tmp_path / "code-center-local.geco",
        data=pd.DataFrame({
            "key": [10, 30, 20, 40],
            "text": ["one", "two", "three", "four"],
        }),
        keys=["key"], text="text",
        geometries=None,
    )
    geometry_matrix = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
    np.save(project.project_dir / "phase4_geometry.npy", geometry_matrix)
    gid = project.database.register_geometry(
        name="phase4", class_name="CountGeometry", modality="text",
        public=True, config={}, dependency_names=[],
        matrix_path="phase4_geometry.npy", matrix_sparse=False, state_path=None,
        supports_query=False, supports_text_transform=False,
    )
    coords = np.array([[0.1, 0.3], [0.4, 0.7], [0.9, 0.2], [0.8, 0.6]])
    view_path = project.artifacts.save_view("phase4_test", coords)
    vid = project.database.register_view(
        geometry_id=gid, name="phase4_test", method="pca",
        parameters={}, artifact_path=view_path,
    )
    return project, gid, vid, coords


def test_local_view_page_rows_preserve_duplicates_order_and_dtype(tmp_path, monkeypatch):
    project, _, vid, coords = _local(tmp_path)
    captured = []
    original_load = np.load

    def traced_load(*args, **kwargs):
        captured.append(kwargs.get("mmap_mode"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", traced_load)
    positions = [3, 0, 3, 2]
    result = project.view_coordinate_rows(vid, positions)
    np.testing.assert_allclose(result, coords[positions].astype(np.float32))
    assert result.dtype == np.float32
    assert captured == ["r"]
    assert project.view_coordinate_rows(vid, []).shape == (0, 2)
    with pytest.raises(ValueError, match="one-dimensional"):
        project.view_coordinate_rows(vid, [[0, 1]])


def test_atomic_point_reads_only_its_local_view_row(tmp_path, monkeypatch):
    project, _, vid, coords = _local(tmp_path)
    def forbidden_full_view(*args, **kwargs):
        raise AssertionError("Atomic focal lookup must not load full coordinates")
    monkeypatch.setattr(project, "view_coordinates", forbidden_full_view)
    oid = project.observation_id_for_unit(3)
    result = project.observation_view_coordinates(oid, vid)
    np.testing.assert_allclose(result, coords[2:3].astype(np.float32))


def test_unified_local_views_and_analytic_rows_match_full_view(tmp_path):
    project, gid, vid, coords = _local(tmp_path)
    positions = [2, 0, 2]
    rep_view = project.database.representation_for_legacy("view", vid)
    rep_geometry = project.database.representation_for_legacy("geometry", gid)
    # Direct fixture registrations do not infer dimensions from the artifact.
    project.database.record_representation_dimensions(
        int(rep_geometry["representation_id"]), 2
    )
    np.testing.assert_allclose(
        project.view_coordinate_rows(-int(rep_view["representation_id"]), positions),
        project.view_coordinates(vid)[positions],
    )
    np.testing.assert_allclose(
        project.view_coordinate_rows(-int(rep_geometry["representation_id"]), positions),
        project.geometry_matrix(gid)[positions].astype(np.float32),
    )


def test_external_view_reads_only_page_keys_in_requested_order(tmp_path):
    class Provider:
        def __init__(self):
            self.view_calls = []

        def view_coordinates(self, external_ref, ordered_user_keys):
            self.view_calls.append((external_ref, ordered_user_keys))
            lookup = {8: [0.0, 1.0], 2: [2.0, 3.0], 7: [4.0, 5.0]}
            return np.array([lookup[key["key"]] for key in ordered_user_keys])

    provider = Provider()
    project = GeometricCoder.create_external(
        project_dir=tmp_path / "code-center-external.geco",
        data=pd.DataFrame({"key": [8, 2, 7], "text": ["a", "b", "c"]}),
        keys=["key"], text="text", metadata=[], external_provider=provider,
    )
    geometry_id = project.database.register_geometry(
        name="external", class_name="External", modality="text",
        public=True, config={}, dependency_names=[],
        matrix_path="", matrix_sparse=False, state_path=None,
        supports_query=False, supports_text_transform=False,
        storage_kind="external", external_ref={"matrix": 1},
    )
    view_id = project.database.register_view(
        geometry_id=geometry_id, name="external_view", method="external",
        parameters={}, artifact_path="", storage_kind="external",
        external_ref={"view": 2},
    )
    result = project.view_coordinate_rows(view_id, [2, 0])
    assert result.tolist() == [[4.0, 5.0], [0.0, 1.0]]
    assert provider.view_calls == [
        ({"view": 2}, [{"key": 7}, {"key": 8}])
    ]
