from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.exceptions import ConfigurationError


class Provider:
    def geometry_matrix(self, external_ref: Any, ordered_user_keys: list[dict[str, Any]]) -> np.ndarray:
        assert external_ref == {"artifact": "minilm"}
        return np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)

    def view_coordinates(self, external_ref: Any, ordered_user_keys: list[dict[str, Any]]) -> np.ndarray:
        assert external_ref == {"artifact": "minilm_umap"}
        return np.asarray([[0.0, 0.0], [1.0, 1.0], [0.2, 0.1]], dtype=np.float32)

    def transform_query(self, external_ref: Any, query: str) -> np.ndarray:
        assert external_ref == {"artifact": "minilm"}
        assert query == "qualitative interview methods"
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


def _external_project(tmp_path: Path) -> GeometricCoder:
    data = pd.DataFrame(
        {"row_id": [0, 1, 2], "text": ["alpha", "beta", "gamma"]}
    )
    return GeometricCoder.create_external(
        project_dir=tmp_path / "external.geco",
        data=data,
        keys=["row_id"],
        text="text",
        metadata=[],
        external_provider=Provider(),
    )


def test_public_external_resource_inspection_and_query(tmp_path: Path) -> None:
    project = _external_project(tmp_path)
    geometry_id = project.register_external_geometry(
        name="minilm",
        external_ref={"artifact": "minilm"},
        supports_query=True,
    )
    view_id = project.register_external_view(
        geometry_id=geometry_id,
        name="minilm_umap",
        external_ref={"artifact": "minilm_umap"},
    )

    assert project.has_geometry("minilm")
    assert not project.has_geometry("missing")
    assert project.has_view("minilm_umap")
    assert not project.has_view("missing")

    geometry = next(row for row in project.geometries() if row["name"] == "minilm")
    assert geometry["geometry_id"] == geometry_id
    assert geometry["storage_kind"] == "external"
    assert geometry["supports_query"] is True

    view = next(row for row in project.views() if row["name"] == "minilm_umap")
    assert view["view_id"] == view_id
    assert view["geometry_id"] == geometry_id
    assert view["storage_kind"] == "external"

    transformed = project.transform_query("minilm", "qualitative interview methods")
    assert transformed.shape == (1, 2)
    scores = project.semantic_search(
        "qualitative interview methods", geometry_names=["minilm"]
    )["minilm"]
    assert len(scores) == 3
    assert scores[0] > scores[1]


def test_external_registration_reuse_is_idempotent_but_not_mutating(tmp_path: Path) -> None:
    project = _external_project(tmp_path)
    geometry_id = project.register_external_geometry(
        name="minilm",
        external_ref={"artifact": "minilm"},
        supports_query=True,
    )
    assert project.register_external_geometry(
        name="minilm",
        external_ref={"artifact": "minilm"},
        supports_query=True,
        if_exists="reuse",
    ) == geometry_id

    view_id = project.register_external_view(
        geometry_id=geometry_id,
        name="minilm_umap",
        external_ref={"artifact": "minilm_umap"},
    )
    assert project.register_external_view(
        geometry_id=geometry_id,
        name="minilm_umap",
        external_ref={"artifact": "minilm_umap"},
        if_exists="reuse",
    ) == view_id

    with pytest.raises(ConfigurationError, match="does not match"):
        project.register_external_geometry(
            name="minilm",
            external_ref={"artifact": "other"},
            supports_query=True,
            if_exists="reuse",
        )


def test_transform_query_reports_capability_errors(tmp_path: Path) -> None:
    project = _external_project(tmp_path)
    project.register_external_geometry(
        name="no_query",
        external_ref={"artifact": "minilm"},
        supports_query=False,
    )
    with pytest.raises(ConfigurationError, match="does not support semantic query"):
        project.transform_query("no_query", "qualitative interview methods")
