"""Smoke checks intended to run in an environment with only standard GeCo dependencies."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from geometric_coder import CountGeometry, GeometricCoder
from geometric_coder.ui import create_app


def test_standard_install_can_construct_interface(tmp_path: Path) -> None:
    # These imports are deliberately inside the smoke test: all three are core
    # application dependencies, not test/dev extras.
    import dash  # noqa: F401
    import dash_ag_grid  # noqa: F401
    import plotly  # noqa: F401
    import umap  # noqa: F401

    project = GeometricCoder.create(
        project_dir=tmp_path / "smoke.geco",
        data=pd.DataFrame(
            {
                "row_id": [0, 1, 2],
                "text": ["alpha beta", "beta gamma", "gamma delta"],
            }
        ),
        keys=["row_id"],
        text="text",
        metadata=[],
        geometries={"lexical": CountGeometry(lemmatize=False)},
    )
    reopened = GeometricCoder.open(project.project_dir)
    app = create_app(reopened)
    assert app.layout is not None
