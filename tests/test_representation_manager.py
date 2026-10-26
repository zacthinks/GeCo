"""Representation manager contracts: named scientific artifacts and separate 2D views."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from geometric_coder.exceptions import ConfigurationError
from geometric_coder.ui import representation_manager as rm


def test_recipe_catalog_is_generic_and_does_not_duplicate_topic_models():
    recipes = {row["value"] for row in rm.geometry_recipe_options()}
    assert {"tfidf", "normalize", "svd", "umap"} <= recipes
    assert "lda" not in recipes


def test_creation_rejects_invalid_names_and_unlinked_projects(monkeypatch):
    coder = SimpleNamespace(has_geometry=lambda name: False)
    with pytest.raises(ValueError, match="geometry name"):
        rm.create_geometry(coder, name="  ", source_geometry_id=1, recipe="umap")
    monkeypatch.setattr(rm, "teal_linked_project", lambda coder: None)
    with pytest.raises(ConfigurationError, match="TeAL-linked"):
        rm.create_geometry(coder, name="Test", source_geometry_id=1, recipe="umap")


def test_view_creation_rejects_non_2d_configuration_without_teal(monkeypatch):
    coder = SimpleNamespace(has_view=lambda name: False)
    with pytest.raises(ValueError, match="2D view name"):
        rm.create_view(coder, name=" ", geometry_id=1)
    monkeypatch.setattr(rm, "teal_linked_project", lambda coder: Path("/fake"))
    with pytest.raises(ValueError, match="fixed at two"):
        rm.create_view(coder, name="view", geometry_id=1, parameters={"n_components": 5})


@pytest.fixture
def fake_link(monkeypatch):
    teal_project = pytest.importorskip("text_analysis_lab.core.project")
    artifact = SimpleNamespace(
        artifact_id="base-matrix",
        artifact_type=SimpleNamespace(value="dense_matrix"),
        primary_key=["doc_id"],
        get_matrix=Mock(return_value=__import__("numpy").ones((1, 2))),
    )
    derived = SimpleNamespace(
        artifact_id="saved-matrix",
        artifact_type=SimpleNamespace(value="dense_matrix"),
        primary_key=["doc_id"],
        get_matrix=Mock(return_value=__import__("numpy").ones((1, 2))),
    )
    linked = SimpleNamespace(
        documents=SimpleNamespace(primary_key=["doc_id"]),
        add_geometry=Mock(return_value=47),
        add_projection=Mock(return_value=48),
    )
    project = SimpleNamespace(
        get_artifact=Mock(return_value=artifact),
        translate=Mock(return_value={"output": derived}),
        geco=SimpleNamespace(
            list=Mock(return_value=[
                {"mode": "explore", "name": "linked",
                 "workspace_path": "/fake/workspace.geco"}
            ]),
            open=Mock(return_value=nullcontext(linked)),
        ),
    )
    monkeypatch.setattr(teal_project.Project, "open", classmethod(lambda cls, path: nullcontext(project)))
    monkeypatch.setattr(rm, "teal_linked_project", lambda coder: Path("/fake/project"))
    monkeypatch.setattr(rm, "_operator_for_recipe", lambda recipe, parameters: ("operator", recipe, parameters))
    coder = SimpleNamespace(
        project_dir=Path("/fake/workspace.geco"),
        has_geometry=Mock(return_value=False),
        has_view=Mock(return_value=False),
        geometries=Mock(return_value=[
            {"geometry_id": 3, "name": "base",
             "external_ref": {"artifact_id": "base-matrix"}}
        ]),
        register_external_geometry=Mock(return_value=49),
    )
    return coder, project, linked, artifact, derived


def test_fit_only_geometry_is_navigation_only_and_does_not_claim_replay(fake_link):
    coder, project, linked, artifact, derived = fake_link
    created = rm.create_geometry(
        coder, name="UMAP-8", source_geometry_id=3, recipe="umap",
        parameters={"n_components": 8}, navigation_only=True,
    )
    assert created.geometry_id == 49
    assert created.artifact_id == "saved-matrix"
    assert created.replayable is False
    coder.register_external_geometry.assert_called_once_with(
        name="UMAP-8", external_ref={"artifact_id": "saved-matrix"},
        supports_query=False, supports_text_transform=False,
    )
    linked.add_geometry.assert_not_called()
    project.translate.assert_called_once()


def test_replayable_registration_uses_teal_link_validation(fake_link):
    coder, project, linked, artifact, derived = fake_link
    created = rm.create_geometry(
        coder, name="Normalized", source_geometry_id=3,
        recipe="normalize", navigation_only=False,
    )
    assert created.geometry_id == 47
    assert created.replayable
    linked.add_geometry.assert_called_once_with("Normalized", derived)
    coder.register_external_geometry.assert_not_called()


def test_identity_view_is_separate_from_geometry_and_has_no_fit(fake_link):
    coder, project, linked, artifact, derived = fake_link
    view_id = rm.create_view(coder, name="Base 2D", geometry_id=3, method="identity")
    assert view_id == 48
    project.translate.assert_not_called()
    linked.add_projection.assert_called_once_with("Base 2D", artifact, geometry="base")


def test_projected_view_uses_fixed_two_dimensions(fake_link):
    coder, project, linked, artifact, derived = fake_link
    view_id = rm.create_view(coder, name="Map", geometry_id=3,
                             method="umap", parameters={"n_neighbors": 12})
    assert view_id == 48
    assert project.translate.call_args.args[0] == (
        "operator", "umap", {"n_neighbors": 12, "n_components": 2}
    )
    linked.add_projection.assert_called_once_with("Map", derived, geometry="base")


def test_manager_is_exposed_without_changing_topic_model_creation():
    root = Path(__file__).resolve().parents[1] / "src/geometric_coder/ui"
    app = (root / "app.py").read_text()
    ui = (root / "representation_ui.py").read_text()
    assert 'id="representation-manager-open"' in app
    assert "register_representation_callbacks(" in app
    assert 'id="representation-create-submit"' in ui
    assert 'id="representation-view-submit"' in ui
    assert 'Output("topic-create-geometry", "options")' in ui
    assert "create_topic_model(" in app
