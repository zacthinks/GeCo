"""Unit tests for the GeCo-owned metadata FeatureMatrix orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from geometric_coder.exceptions import ConfigurationError
from geometric_coder.ui import representation_manager as rm


@pytest.mark.parametrize(
    ("method", "kwargs", "expected"),
    [
        ("raw", {}, {"kind": "raw"}),
        ("zscore", {}, {"kind": "zscore"}),
        ("minmax", {}, {"kind": "minmax"}),
        ("onehot", {}, "onehot"),
        ("fixed_width", {"width": 10, "anchor": 2000}, {
            "kind": "bin", "strategy": "fixed_width", "width": 10.,
            "anchor": 2000.,
        }),
        ("fixed_width", {"width": "0.1", "anchor": "-0.2"}, {
            "kind": "bin", "strategy": "fixed_width", "width": 0.1,
            "anchor": -0.2,
        }),
        ("quantile", {"n_bins": 10}, {
            "kind": "bin", "strategy": "quantile", "n_bins": 10,
        }),
        ("equal_width", {"n_bins": 4}, {
            "kind": "bin", "strategy": "equal_width", "n_bins": 4,
        }),
        ("boundaries", {"boundaries": "[0, 10, 20]"}, {
            "kind": "bin", "strategy": "boundaries", "boundaries": [0., 10., 20.],
        }),
    ],
)
def test_feature_matrix_request_is_teal_native(method, kwargs, expected):
    assert rm._metadata_encoding({"transform": method, **kwargs}) == expected


@pytest.mark.parametrize(
    "settings",
    [
        {"transform": "quantile", "n_bins": 0},
        {"transform": "quantile", "n_bins": "4.5"},
        {"transform": "fixed_width", "width": 0},
        {"transform": "fixed_width", "width": float("nan")},
        {"transform": "fixed_width", "width": 10, "anchor": float("inf")},
        {"transform": "boundaries", "boundaries": "[10, 0]"},
        {"transform": "boundaries", "boundaries": "[0, 0]"},
        {"transform": "boundaries", "boundaries": "[0, \"wrong\"]"},
        {"transform": "unrecognized"},
    ],
)
def test_invalid_feature_matrix_settings_rejected(settings):
    with pytest.raises(ValueError):
        rm._metadata_encoding(settings)


def test_metadata_requires_explicit_source_and_selected_fields(monkeypatch):
    coder = SimpleNamespace(
        has_geometry=Mock(return_value=False),
        has_view=Mock(return_value=False),
    )
    monkeypatch.setattr(rm, "teal_linked_project", lambda unused: None)
    with pytest.raises(ValueError, match="at least one"):
        rm.create_metadata_geometry(
            coder, name="Augmented", source_representation_id=7,
            metadata_specs={},
        )
    with pytest.raises(ConfigurationError, match="TeAL-linked"):
        rm.create_metadata_geometry(
            coder, name="Augmented", source_representation_id=7,
            metadata_specs={"source": {"transform": "onehot"}},
        )


def test_metadata_ui_is_part_of_existing_representation_wizard():
    from pathlib import Path

    from geometric_coder.ui import representation_ui

    source = Path(representation_ui.__file__).read_text(encoding="utf-8")
    assert '"Add metadata features"' in source
    assert '"representation-metadata-fields"' in source
    assert "create_metadata_geometry(" in source
    assert '"representation-wizard-submit"' in source



def test_metadata_wizard_preserves_edited_controls_when_fields_change(tmp_path):
    """Exercise the registered Dash callback with its real component tree.

    The old callback reconstructed all controls with default values when a
    second metadata field was selected, changing the fitted representation.
    """
    import pandas as pd

    from geometric_coder import CountGeometry, GeometricCoder
    from geometric_coder.ui import create_app

    coder = GeometricCoder.create(
        project_dir=tmp_path / "callback.geco",
        data=pd.DataFrame({
            "row_id": [1, 2, 3],
            "text": ["alpha beta", "gamma delta", "alpha gamma"],
        }),
        keys=["row_id"], text="text", metadata=[],
        geometries={"tokens": CountGeometry(lemmatize=False)},
    )
    app = create_app(coder)
    callbacks = [
        entry["callback"].__wrapped__
        for key, entry in app.callback_map.items()
        if "representation-metadata-settings.children" in key
        and "representation-metadata-config.data" in key
    ]
    assert len(callbacks) == 1
    render = callbacks[0]
    discovered = [
        {"value": "publication_year", "numeric": True, "default_transform": "zscore"},
        {"value": "publication_source", "numeric": False, "default_transform": "onehot"},
    ]

    def component(root, identifier):
        if getattr(root, "id", None) == identifier:
            return root
        for child in getattr(root, "children", []) or []:
            if isinstance(child, (str, int, float, bool)):
                continue
            found = component(child, identifier)
            if found is not None:
                return found
        return None

    def render_fields(selected, state, mounted=None, source=10):
        mounted = mounted or {}
        def ids(kind):
            return [{"type": f"representation-metadata-{kind}", "name": field}
                    for field in mounted]
        return render(
            selected, discovered, source, state,
            ids("transform"), [c["transform"] for c in mounted.values()],
            ids("width"), [c["width"] for c in mounted.values()],
            ids("anchor"), [c["anchor"] for c in mounted.values()],
            ids("count"), [c["n_bins"] for c in mounted.values()],
            ids("boundaries"), [c["boundaries"] for c in mounted.values()],
        )

    fields, state = render_fields(["publication_year"], None)
    assert component(fields[0], {
        "type": "representation-metadata-transform", "name": "publication_year",
    }).value == "zscore"

    edited = {
        "publication_year": {
            "transform": "fixed_width", "width": 10,
            "anchor": 2000, "n_bins": 8, "boundaries": "[1990, 2000, 2010]",
        },
    }
    fields, state = render_fields(
        ["publication_year", "publication_source"], state, edited,
    )
    assert component(fields[0], {
        "type": "representation-metadata-transform", "name": "publication_year",
    }).value == "fixed_width"
    assert component(fields[0], {
        "type": "representation-metadata-width", "name": "publication_year",
    }).value == 10
    assert component(fields[0], {
        "type": "representation-metadata-anchor", "name": "publication_year",
    }).value == 2000
    assert component(fields[1], {
        "type": "representation-metadata-transform", "name": "publication_source",
    }).value == "onehot"

    # Removing and re-adding year also restores its full configuration.
    fields, state = render_fields(["publication_source"], state, edited)
    fields, state = render_fields(["publication_source", "publication_year"], state)
    assert component(fields[1], {
        "type": "representation-metadata-transform", "name": "publication_year",
    }).value == "fixed_width"
    assert component(fields[1], {
        "type": "representation-metadata-anchor", "name": "publication_year",
    }).value == 2000

    # A different geometry must not inherit a coincidentally named field.
    fields, state = render_fields(["publication_year"], state, source=11)
    assert component(fields[0], {
        "type": "representation-metadata-transform", "name": "publication_year",
    }).value == "zscore"
    assert component(fields[0], {
        "type": "representation-metadata-anchor", "name": "publication_year",
    }).value == 0
