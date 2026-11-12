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
