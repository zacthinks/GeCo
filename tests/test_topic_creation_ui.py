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
    from geometric_coder.ui.topic_creation import recipe_fields, recipe_options, RECIPE_FIELDS
    assert recipe in {row["value"] for row in recipe_options()}
    allowed = {row[0] for row in RECIPE_FIELDS}
    assert set(recipe_fields(recipe)).issubset(allowed)


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
    assert "recipe_fields(algorithm or \"identity\")" in source
