"""Representation manager contracts: named scientific artifacts and separate 2D views."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
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
    coder = SimpleNamespace(has_geometry=lambda name: False, has_view=lambda name: False)
    with pytest.raises(ValueError, match="view representation name"):
        rm.create_view(coder, name=" ", geometry_id=1)
    monkeypatch.setattr(rm, "teal_linked_project", lambda coder: Path("/fake"))
    with pytest.raises(ValueError, match="dimensionality is fixed"):
        rm.create_view(coder, name="view", geometry_id=1, parameters={"n_components": 5})


@pytest.fixture
def fake_link(monkeypatch):
    # Stub only the optional TeAL project import: routing tests must run in
    # GeCo's ordinary CI even when TeAL is not installed.
    artifact = SimpleNamespace(
        artifact_id="base-matrix",
        artifact_type=SimpleNamespace(value="dense_matrix"),
        primary_key=["doc_id"],
        get_matrix=Mock(return_value=__import__("numpy").ones((1, 2))),
        n_rows=1,
    )
    derived = SimpleNamespace(
        artifact_id="saved-matrix",
        artifact_type=SimpleNamespace(value="dense_matrix"),
        primary_key=["doc_id"],
        get_matrix=Mock(return_value=__import__("numpy").ones((1, 8))),
        n_rows=1,
        aliases=[],
        add_alias=Mock(),
        remove_alias=Mock(),
    )
    linked = SimpleNamespace(
        documents=SimpleNamespace(primary_key=["doc_id"]),
        manifest={"text_artifact_id": "source-text"},
        add_geometry=Mock(return_value=47),
        add_projection=Mock(return_value=48),
    )
    project = SimpleNamespace(
        get_artifact=Mock(return_value=artifact),
        pipeline=Mock(),
        translate=Mock(return_value={"output": derived}),
        geco=SimpleNamespace(
            list=Mock(return_value=[
                {"mode": "explore", "name": "linked",
                 "workspace_path": "/fake/workspace.geco"}
            ]),
            open=Mock(return_value=nullcontext(linked)),
        ),
    )
    pkg = ModuleType("text_analysis_lab")
    pkg.__path__ = []
    errors = ModuleType("text_analysis_lab.core.errors")
    errors.PipelineError = type("PipelineError", (Exception,), {})
    core = ModuleType("text_analysis_lab.core")
    core.__path__ = []
    module = ModuleType("text_analysis_lab.core.project")
    module.Project = SimpleNamespace(open=Mock(return_value=nullcontext(project)))
    monkeypatch.setitem(sys.modules, "text_analysis_lab", pkg)
    monkeypatch.setitem(sys.modules, "text_analysis_lab.core", core)
    monkeypatch.setitem(sys.modules, "text_analysis_lab.core.errors", errors)
    monkeypatch.setitem(sys.modules, "text_analysis_lab.core.project", module)
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
        database=SimpleNamespace(
            list_observations=Mock(return_value=[]),
            list_representations=Mock(return_value=[]),
            representation_for_legacy=Mock(side_effect=lambda _kind, ident: {
                "representation_id": 100 + int(ident)
            }),
            assign_representation_parent=Mock(),
        ),
        views=Mock(return_value=[]),
        register_external_view=Mock(return_value=52),
    )
    return coder, project, linked, artifact, derived


def test_fit_only_geometry_is_navigation_only_and_does_not_claim_replay(fake_link):
    coder, project, linked, artifact, derived = fake_link
    from text_analysis_lab.core.errors import PipelineError
    project.pipeline.side_effect = PipelineError("No reusable state")
    created = rm.create_geometry(
        coder, name="UMAP-8", source_geometry_id=3, recipe="umap",
        parameters={"n_components": 8},
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
    coder.database.assign_representation_parent.assert_called_once_with(149, 103)


def test_derived_observations_block_registration_before_teal_fit(fake_link):
    coder, project, linked, artifact, derived = fake_link
    coder.database.list_observations.return_value = [{"kind": "span"}]
    with pytest.raises(ConfigurationError, match="backfill"):
        rm.create_geometry(coder, name="Cannot register", source_geometry_id=3,
                           recipe="umap")
    project.translate.assert_not_called()
    linked.add_geometry.assert_not_called()
    coder.register_external_geometry.assert_not_called()


def test_replayable_registration_uses_teal_link_validation(fake_link):
    coder, project, linked, artifact, derived = fake_link
    created = rm.create_geometry(
        coder, name="Normalized", source_geometry_id=3,
        recipe="normalize",
    )
    assert created.geometry_id == 47
    assert created.replayable
    linked.add_geometry.assert_called_once_with("Normalized", derived)
    coder.register_external_geometry.assert_not_called()


def test_identity_view_is_implicit_and_does_not_fit(fake_link):
    coder, project, linked, artifact, derived = fake_link
    with pytest.raises(ValueError, match="implicit"):
        rm.create_view(coder, name="Base 2D", geometry_id=3, method="identity")
    project.translate.assert_not_called()
    linked.add_projection.assert_not_called()


def test_projected_view_is_registered_once_as_view_only(fake_link):
    coder, project, linked, artifact, derived = fake_link
    derived.get_matrix.return_value = __import__("numpy").ones((1, 2))
    representation_id = rm.create_view(
        coder, name="Map", geometry_id=3,
        method="umap", parameters={"n_neighbors": 12},
    )
    assert representation_id == 149
    assert project.translate.call_args.args[0] == (
        "operator", "umap", {"n_neighbors": 12, "n_components": 2}
    )
    coder.register_external_geometry.assert_called_once_with(
        name="Map", external_ref={"artifact_id": "saved-matrix"},
        supports_query=False, supports_text_transform=False,
        analytic_enabled=False,
    )
    coder.database.assign_representation_parent.assert_called_once_with(149, 103)
    derived.add_alias.assert_called_once_with("Map")
    linked.add_projection.assert_not_called()


def test_one_dimensional_view_is_navigation_only(fake_link):
    coder, project, linked, artifact, derived = fake_link
    derived.get_matrix.return_value = __import__("numpy").ones((1, 1))
    representation_id = rm.create_view(
        coder, name="Map1D", geometry_id=3, method="svd", dimensions=1,
    )
    assert representation_id == 149
    assert project.translate.call_args.args[0] == (
        "operator", "svd", {"n_components": 1}
    )
    coder.register_external_geometry.assert_called_once()
    assert coder.register_external_geometry.call_args.kwargs["analytic_enabled"] is False



def test_manager_is_exposed_without_changing_topic_model_creation():
    root = Path(__file__).resolve().parents[1] / "src/geometric_coder/ui"
    app = (root / "app.py").read_text()
    ui = (root / "representation_ui.py").read_text()
    assert 'id="representation-manager-open"' in app
    assert "register_representation_callbacks(" in app
    assert 'id="representation-wizard-submit"' in ui
    assert 'id="representation-detail-add-view"' in ui
    assert 'id="representation-filter-min"' in ui
    assert 'id="representation-filter-source"' in ui
    assert 'Output("topic-create-geometry", "options")' in ui
    assert "create_topic_model(" in app


def test_unified_representation_filters():
    rows = [
        {"key": "geometry:1", "name": "counts", "dimensions": 500, "sources": []},
        {"key": "geometry:2", "name": "umap_8", "dimensions": 8, "sources": ["counts"]},
        {"key": "view:1", "name": "umap_2", "dimensions": 2, "sources": ["umap_8"]},
    ]
    assert [r["name"] for r in rm.filter_representations(rows, min_dimensions=3)] == [
        "counts", "umap_8",
    ]
    assert [r["name"] for r in rm.filter_representations(rows, min_dimensions=2, max_dimensions=2)] == ["umap_2"]
    assert [r["name"] for r in rm.filter_representations(rows, source="counts")] == ["umap_8"]
    assert rm.source_filter_options(rows) == [
        {"label": "counts", "value": "counts"},
        {"label": "umap_8", "value": "umap_8"},
    ]
    with pytest.raises(ValueError, match="exceed"):
        rm.filter_representations(rows, min_dimensions=10, max_dimensions=2)


def test_unregistration_blocks_dependent_views():
    from geometric_coder.ui.representation_manager import unregister_representation
    connection = Mock()

    def execute(sql, _params=()):
        # SELECT sessions uses fetchall(), so positional fetchone mocks would
        # accidentally assign the view dependency to classifier_specs.
        result = Mock()
        result.fetchall.return_value = []
        result.fetchone.return_value = {"view_id": 3} if "FROM views" in sql else None
        return result

    connection.execute.side_effect = execute
    context = Mock()
    context.__enter__ = Mock(return_value=connection)
    context.__exit__ = Mock(return_value=False)
    coder = SimpleNamespace(
        database=SimpleNamespace(
            get_geometry=Mock(return_value={"geometry_id": 2, "name": "source"}),
            connect=Mock(return_value=context),
        ),
        geometries=lambda **_: [{"geometry_id": 1}, {"geometry_id": 2}],
    )
    with pytest.raises(ConfigurationError, match="views still references"):
        unregister_representation(coder, key="geometry:2")

    connection.execute.assert_any_call(
        "SELECT 1 FROM views WHERE geometry_id = ? LIMIT 1", (2,)
    )


def test_unified_inventory_shows_one_row_for_shared_artifact(monkeypatch):
    coder = SimpleNamespace(
        geometries=lambda **_: [{"geometry_id": 2, "name": "B"}],
        database=SimpleNamespace(
            list_geometries=Mock(return_value=[{"geometry_id": 2, "name": "B"}]),
            list_representations=Mock(return_value=[{
                "representation_id": 7, "legacy_kind": "geometry",
                "legacy_id": 2, "name": "B", "dimensions": 2,
                "analytic_enabled": True, "parent_representation_id": None,
                "storage_kind": "external",
                "external_ref": {"artifact_id": "shared"},
            }]),
            get_geometry=Mock(return_value={
                "supports_query": False, "supports_text_transform": False,
            }),
        ),
    )
    monkeypatch.setattr(rm, "teal_linked_project", lambda _coder: None)
    inventory = rm.representation_inventory(coder)
    assert len(inventory) == 1
    assert inventory[0]["key"] == "representation:7"
    assert inventory[0]["dimensions"] == 2
    assert inventory[0]["analytic_enabled"]
    assert inventory[0]["source"] == ""



def test_register_view_as_geometry_reuses_artifact_without_fitting():
    coder = SimpleNamespace(
        database=SimpleNamespace(get_view=Mock(return_value={
            "view_id": 11, "geometry_id": 3, "name": "Map2D",
            "storage_kind": "external", "external_ref": {"artifact_id": "saved-2d"},
        })),
        geometries=Mock(return_value=[]),
        has_geometry=Mock(return_value=False),
        register_external_geometry=Mock(return_value=21),
        register_external_view=Mock(return_value=22),
        views=Mock(return_value=[]),
    )
    assert rm.register_view_as_geometry(coder, view_id=11) == 21
    coder.register_external_geometry.assert_called_once_with(
        name="Map2D", external_ref={"artifact_id": "saved-2d"},
        supports_query=False, supports_text_transform=False,
    )
    coder.register_external_view.assert_not_called()
    coder.database.get_view.return_value["storage_kind"] = "local"
    with pytest.raises(ConfigurationError, match="provider-backed"):
        rm.register_view_as_geometry(coder, view_id=11)


def test_inventory_includes_view_only_numeric_representation(monkeypatch):
    coder = SimpleNamespace(
        database=SimpleNamespace(
            list_geometries=Mock(return_value=[{"geometry_id": 8, "name": "view_only"}]),
            list_representations=Mock(return_value=[{
                "representation_id": 28, "legacy_kind": "geometry",
                "legacy_id": 8, "name": "view_only", "dimensions": 2,
                "analytic_enabled": False, "parent_representation_id": None,
                "storage_kind": "external", "external_ref": {"artifact_id": "view-only"},
            }]),
            get_geometry=Mock(return_value={
                "supports_query": False, "supports_text_transform": False,
            }),
        ),
    )
    monkeypatch.setattr(rm, "teal_linked_project", lambda _coder: None)
    inventory = rm.representation_inventory(coder)
    assert len(inventory) == 1
    assert inventory[0]["name"] == "view_only"
    assert inventory[0]["analytic_enabled"] is False
    assert inventory[0]["can_view"] is True
