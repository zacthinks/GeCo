"""Smoke checks intended to run in an environment with standard GeCo dependencies."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from geometric_coder import CountGeometry, GeometricCoder


def _require_ui() -> None:
    pytest.importorskip("dash")
    pytest.importorskip("dash_ag_grid")
    pytest.importorskip("plotly")
    pytest.importorskip("umap")


def _component_by_id(component: Any, component_id: str) -> Any:
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    if children is None:
        raise LookupError(component_id)
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if child is None or isinstance(child, (str, int, float, bool)):
            continue
        try:
            return _component_by_id(child, component_id)
        except LookupError:
            pass
    raise LookupError(component_id)


def _real_local_data() -> pd.DataFrame:
    """Small AERA-shaped table that exercises real local geometry/view creation."""
    texts = [
        "Teachers discussed classroom assessment and formative feedback.",
        "Students described collaborative learning during science instruction.",
        "The study examined qualitative interview methods in teacher education.",
        "Researchers analyzed school leadership and organizational change.",
        "The paper studied mathematics learning with digital instructional tools.",
        "Participants reflected on equity and access in higher education.",
        "The analysis considered reading comprehension and literacy development.",
        "Survey responses described professional learning among school principals.",
        "The project investigated multilingual learners in urban classrooms.",
        "Authors examined educational policy implementation across districts.",
        "The session focused on social-emotional learning and student belonging.",
        "Researchers compared instructional coaching practices across schools.",
    ]
    return pd.DataFrame(
        {
            "record_id": [index // 2 + 1 for index in range(len(texts))],
            "sentence_id": [index % 2 + 1 for index in range(len(texts))],
            "text": texts,
            "record_type": ["paper"] * len(texts),
            "session_title": [f"Session {index // 4 + 1}" for index in range(len(texts))],
        }
    )


def _assert_dash_http_surface(app: Any) -> None:
    """Exercise the actual Flask routes a browser hits before callbacks run."""
    client = app.server.test_client()
    for path in ("/", "/_dash-layout", "/_dash-dependencies"):
        response = client.get(path)
        assert response.status_code == 200, (
            f"Dash route {path!r} returned HTTP {response.status_code}: "
            f"{response.get_data(as_text=True)[:1000]}"
        )


class _ExternalSmokeProvider:
    def geometry_matrix(
        self, external_ref: Any, ordered_user_keys: list[dict[str, Any]]
    ) -> np.ndarray:
        assert external_ref == {"artifact": "geometry"}
        return np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)

    def view_coordinates(
        self, external_ref: Any, ordered_user_keys: list[dict[str, Any]]
    ) -> np.ndarray:
        assert external_ref == {"artifact": "view"}
        return np.asarray([[0.0, 0.0], [1.0, 1.0], [0.2, 0.1]], dtype=np.float32)


def test_standard_install_real_local_project_serves_dash_routes(tmp_path: Path) -> None:
    """A real locally computed geometry/view must survive the browser HTTP surface."""
    _require_ui()
    from geometric_coder.ui import create_app

    project = GeometricCoder.create(
        project_dir=tmp_path / "real-local-smoke.geco",
        data=_real_local_data(),
        keys=["record_id", "sentence_id"],
        text="text",
        metadata=["record_type", "session_title"],
        geometries={
            "lexical": CountGeometry(
                lemmatize=False,
                lowercase=True,
                weighting="tfidf",
            )
        },
    )
    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.geometries()
    assert reopened.views()

    app = create_app(reopened)
    assert app.layout is not None
    geometry_dropdown = _component_by_id(app.layout, "geometry-dropdown")
    view_dropdown = _component_by_id(app.layout, "view-dropdown")
    assert geometry_dropdown.value is not None
    assert view_dropdown.value is not None
    _assert_dash_http_surface(app)


def test_external_interface_initializes_registered_geometry_and_view(tmp_path: Path) -> None:
    """Keep a narrow external-provider smoke separate from the real local smoke."""
    _require_ui()
    from geometric_coder.ui import create_app

    provider = _ExternalSmokeProvider()
    project = GeometricCoder.create_external(
        project_dir=tmp_path / "external-smoke.geco",
        data=pd.DataFrame(
            {"row_id": [0, 1, 2], "text": ["alpha", "beta", "gamma"]}
        ),
        keys=["row_id"],
        text="text",
        metadata=[],
        external_provider=provider,
    )
    geometry_id = project.register_external_geometry(
        name="external_geometry",
        external_ref={"artifact": "geometry"},
    )
    view_id = project.register_external_view(
        geometry_id=geometry_id,
        name="external_view",
        external_ref={"artifact": "view"},
    )
    reopened = GeometricCoder.open(project.project_dir, external_provider=provider)
    app = create_app(reopened)
    geometry_dropdown = _component_by_id(app.layout, "geometry-dropdown")
    view_dropdown = _component_by_id(app.layout, "view-dropdown")
    assert geometry_dropdown.value == geometry_id
    assert view_dropdown.value == view_id
    assert view_dropdown.options == [{"label": "external_view", "value": view_id}]
    _assert_dash_http_surface(app)
