"""TeAL LDA creation guardrails; no fitting in standalone GeCo."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from geometric_coder.exceptions import ConfigurationError
from geometric_coder.ui.topic_creation import (
    create_lda_topic_model, lda_geometry_options, teal_linked_project,
)


def test_creation_is_unavailable_without_teal_provider():
    coder = SimpleNamespace(is_external_backed=False, external_provider=None)
    assert teal_linked_project(coder) is None
    assert lda_geometry_options(coder) == []
    coder.topic_models = lambda: []
    with pytest.raises(ConfigurationError, match="TeAL-linked"):
        create_lda_topic_model(coder, name="LDA", geometry_id=1, n_topics=5)


def test_creation_rejects_invalid_values_before_fitting():
    coder = SimpleNamespace(
        is_external_backed=False, external_provider=None,
        topic_models=lambda: [{"name": "Existing"}],
    )
    with pytest.raises(ValueError, match="name"):
        create_lda_topic_model(coder, name=" ", geometry_id=1, n_topics=5)
    with pytest.raises(ValueError, match="at least 2"):
        create_lda_topic_model(coder, name="Fresh", geometry_id=1, n_topics=1)
    with pytest.raises(ValueError, match="Maximum iterations"):
        create_lda_topic_model(coder, name="Fresh", geometry_id=1, n_topics=5, max_iter=0)
    with pytest.raises(ValueError, match="already exists"):
        create_lda_topic_model(coder, name="Existing", geometry_id=1, n_topics=5)


def test_ui_creation_uses_public_teal_pipeline():
    source = (Path(__file__).resolve().parents[1] / "src/geometric_coder/ui/topic_creation.py").read_text()
    app = (Path(__file__).resolve().parents[1] / "src/geometric_coder/ui/app.py").read_text()
    assert "teal.translate(" in source
    assert "LDA(n_components=n_topics" in source
    assert "ArgMax(), membership" in source
    assert "linked.add_topic_model(" in source
    assert 'id="topic-create-submit"' in app
    assert 'running=[(Output("topic-create-submit", "disabled"), True, False)]' in app


@pytest.mark.parametrize("recipe", [
    "identity", "lda", "gaussian_mixture", "kmeans", "minibatch_kmeans",
    "spherical_kmeans", "kmedoids", "spectral", "agglomerative",
    "dbscan", "optics", "hdbscan",
])
def test_teal_recipe_inventory_has_parameter_contract(recipe):
    from geometric_coder.ui.topic_creation import recipe_options, PRIMARY_PARAMETERS
    assert recipe in {row["value"] for row in recipe_options()}
    assert recipe in PRIMARY_PARAMETERS


def test_installed_teal_signature_is_complete_when_available():
    pytest.importorskip("text_analysis_lab")
    from geometric_coder.ui.topic_creation import (
        recipe_fields, parameter_specs, split_parameter_specs, MODEL_CLASSES,
    )
    for recipe, _, _ in MODEL_CLASSES:
        names = set(recipe_fields(recipe))
        common, advanced = split_parameter_specs(recipe)
        assert names == {entry["name"] for entry in parameter_specs(recipe)}
        assert names == {entry["name"] for entry in common + advanced}
        assert not {entry["name"] for entry in common} & {entry["name"] for entry in advanced}


def test_identity_has_no_fit_parameters():
    from geometric_coder.ui.topic_creation import recipe_fields
    assert recipe_fields("identity") == ()


def test_creation_wizard_is_separate_and_always_uses_existing_geometry():
    source = (Path(__file__).resolve().parents[1] / "src/geometric_coder/ui/app.py").read_text()
    assert 'id="topic-create-modal"' in source
    assert 'id="topic-create-algorithm"' in source
    assert 'id="topic-create-geometry"' in source
    assert 'id="topic-create-open"' in source
    assert 'id="topic-create-close"' in source
    assert 'split_parameter_specs(algorithm or "identity")' in source



def test_geometry_options_include_standalone_registrations():
    from geometric_coder.ui.topic_creation import topic_geometry_options
    coder = SimpleNamespace(
        is_external_backed=False,
        geometries=lambda **_: [
            {"geometry_id": 1, "name": "Counts"},
            {"geometry_id": 2, "name": "SVD coordinates"},
        ],
    )
    assert topic_geometry_options(coder) == [
        {"label": "Counts", "value": 1},
        {"label": "SVD coordinates", "value": 2},
    ]


def test_kmeans_metric_is_fixed_and_other_parameters_visible():
    pytest.importorskip("text_analysis_lab")
    from geometric_coder.ui.topic_creation import recipe_fields
    assert "metric" not in recipe_fields("kmeans")
    assert {"n_init", "init", "tol", "random_state"}.issubset(recipe_fields("kmeans"))
    assert {"batch_size", "reassignment_ratio", "max_no_improvement"}.issubset(
        recipe_fields("minibatch_kmeans")
    )
    assert {"n_init", "init", "tol"}.issubset(recipe_fields("spherical_kmeans"))
    assert "metric" in recipe_fields("kmedoids")


def test_creation_wizard_has_collapsed_advanced_controls():
    source = (Path(__file__).resolve().parents[1] / "src/geometric_coder/ui/app.py").read_text()
    assert 'html.Summary("Advanced parameters")' in source
    assert 'id="topic-create-advanced"' in source
    assert "split_parameter_specs(algorithm or \"identity\")" in source


def test_standalone_wizard_shows_common_parameters_without_teal():
    from geometric_coder.ui.topic_creation import standalone_primary_specs
    specs = standalone_primary_specs("kmeans")
    assert {p["name"] for p in specs} == {"n_clusters", "init", "max_iter"}
    assert standalone_primary_specs("identity") == []


def test_advanced_accordion_is_conditionally_visible():
    source = (Path(__file__).resolve().parents[1] / "src/geometric_coder/ui/app.py").read_text()
    assert 'id="topic-create-advanced-wrap"' in source
    assert 'if advanced else {"display": "none"}' in source
    assert 'standalone_primary_specs(algorithm or "identity")' in source


def test_categorical_topic_parameters_use_documented_choices():
    from geometric_coder.ui.topic_creation import (
        CATEGORICAL_CHOICES, _parse_parameter, standalone_primary_specs,
    )
    assert CATEGORICAL_CHOICES["kmedoids"]["metric"] == (
        "euclidean", "manhattan", "cosine",
    )
    for recipe, fields in CATEGORICAL_CHOICES.items():
        for name, values in fields.items():
            assert len(values) == len(set(values)), (recipe, name)
            assert "precomputed" not in values

    fields = {p["name"]: p for p in standalone_primary_specs("kmedoids")}
    metric = fields["metric"]
    assert metric["kind"] == "choice"
    assert metric["label"] == "Distance metric"
    assert metric["default"] == "euclidean"
    assert {item["value"] for item in metric["options"]} == {
        "euclidean", "manhattan", "cosine",
    }
    assert _parse_parameter("metric", "cosine", metric) == "cosine"
    with pytest.raises(ValueError, match="valid value"):
        _parse_parameter("metric", "unknown", metric)


def test_categorical_defaults_and_complete_signatures_with_installed_teal():
    pytest.importorskip("text_analysis_lab")
    from geometric_coder.ui.topic_creation import (
        MODEL_CLASSES, CATEGORICAL_CHOICES, parameter_specs, recipe_fields,
    )
    for recipe, _, _ in MODEL_CLASSES:
        specs = {field["name"]: field for field in parameter_specs(recipe)}
        assert set(specs) == set(recipe_fields(recipe))
        for name in CATEGORICAL_CHOICES.get(recipe, {}):
            field = specs[name]
            assert field["kind"] == "choice"
            assert field["default"] in {opt["value"] for opt in field["options"]}

    # sklearn's n_init can be 'auto' OR a positive integer; do not turn it
    # into an incorrectly finite categorical list.
    assert next(p for p in parameter_specs("kmeans") if p["name"] == "n_init")["kind"] == "text"


def test_topic_manager_copy_and_wizard_choice_rendering():
    root = Path(__file__).resolve().parents[1] / "src/geometric_coder/ui"
    app = (root / "app.py").read_text()
    modeling = (root / "topic_modeling.py").read_text()
    css = (root / "assets/geco.css").read_text()
    assert 'elif spec["kind"] == "choice":' in app
    assert 'options=spec["options"]' in app
    assert "Topic label saved locally" not in modeling
    assert "Model name saved locally" not in modeling
    assert "TeAL artifacts remain unchanged" not in app
    assert "Original TeAL component IDs" not in app
    assert ".topic-create-advanced { margin-top:" in css
