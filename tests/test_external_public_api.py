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


class TransformingProvider(Provider):
    def transform_texts(self, external_ref: Any, texts: list[str]) -> np.ndarray:
        assert external_ref == {"artifact": "minilm"}
        rows = []
        for text in texts:
            value = float(len(str(text)))
            rows.append([value, 1.0])
        return np.asarray(rows, dtype=np.float32)


def _transforming_external_project(tmp_path: Path) -> GeometricCoder:
    data = pd.DataFrame(
        {"row_id": [0, 1, 2], "text": ["alpha", "beta", "gamma"]}
    )
    return GeometricCoder.create_external(
        project_dir=tmp_path / "transforming-external.geco",
        data=data,
        keys=["row_id"],
        text="text",
        metadata=[],
        external_provider=TransformingProvider(),
    )


def test_external_text_transform_capability_is_per_geometry(tmp_path: Path) -> None:
    project = _transforming_external_project(tmp_path)
    replayable_id = project.register_external_geometry(
        name="replayable",
        external_ref={"artifact": "minilm"},
        supports_query=True,
        supports_text_transform=True,
    )
    record = project.database.get_geometry(replayable_id)
    assert record["supports_text_transform"] is True
    assert project.geometries()[0]["supports_text_transform"] is True
    assert project.can_transform_new_observations() is True

    incompatible_id = project.register_external_geometry(
        name="not_replayable",
        external_ref={"artifact": "minilm"},
        supports_query=False,
        supports_text_transform=False,
    )
    assert project.can_transform_new_observations() is True
    assert project.transformable_geometry_ids() == {replayable_id}
    assert incompatible_id not in project.transformable_geometry_ids()


def test_external_text_transform_declaration_requires_provider_method(tmp_path: Path) -> None:
    project = _external_project(tmp_path)
    with pytest.raises(ConfigurationError, match="does not implement transform_texts"):
        project.register_external_geometry(
            name="replayable",
            external_ref={"artifact": "minilm"},
            supports_text_transform=True,
        )


def test_external_registration_reuse_checks_text_transform_capability(tmp_path: Path) -> None:
    project = _transforming_external_project(tmp_path)
    geometry_id = project.register_external_geometry(
        name="minilm",
        external_ref={"artifact": "minilm"},
        supports_query=True,
        supports_text_transform=True,
    )
    assert project.register_external_geometry(
        name="minilm",
        external_ref={"artifact": "minilm"},
        supports_query=True,
        supports_text_transform=True,
        if_exists="reuse",
    ) == geometry_id
    with pytest.raises(ConfigurationError, match="does not match"):
        project.register_external_geometry(
            name="minilm",
            external_ref={"artifact": "minilm"},
            supports_query=True,
            supports_text_transform=False,
            if_exists="reuse",
        )


def test_semantic_search_reuses_matrix_when_transforming_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _external_project(tmp_path)
    project.register_external_geometry(
        name="minilm",
        external_ref={"artifact": "minilm"},
        supports_query=True,
    )
    real_geometry_matrix = project.geometry_matrix
    fetched: list[int | str] = []

    def counted_geometry_matrix(geometry: int | str):
        fetched.append(geometry)
        return real_geometry_matrix(geometry)

    monkeypatch.setattr(project, "geometry_matrix", counted_geometry_matrix)
    result = project.semantic_search(
        "qualitative interview methods", geometry_names=["minilm"]
    )
    assert len(result["minilm"]) == 3
    assert len(fetched) == 1

    # Direct transforms still validate width against the registered matrix.
    project.transform_query("minilm", "qualitative interview methods")
    assert len(fetched) == 2


def test_mixed_geometry_derived_evidence_is_geometry_specific(tmp_path: Path) -> None:
    """Unsupported geometries keep their atomic evidence and remain current."""
    project = _transforming_external_project(tmp_path)
    available_id = project.register_external_geometry(
        name="transformable", external_ref={"artifact": "minilm"},
        supports_text_transform=True,
    )
    unavailable_id = project.register_external_geometry(
        name="fixed", external_ref={"artifact": "minilm"},
        supports_text_transform=False,
    )
    code_id = project.create_code("mixed geometry coding")
    available_spec = project.create_classifier_spec(
        code_id=code_id, name="transformable classifier",
        geometry_id=available_id,
    )
    unavailable_spec = project.create_classifier_spec(
        code_id=code_id, name="fixed classifier",
        geometry_id=unavailable_id,
    )
    units = project.database.list_units()
    project.annotate(int(units[0]["observation_id"]), code_id, "positive")
    project.annotate(int(units[2]["observation_id"]), code_id, "negative")
    project.train_classifiers(
        code_id=code_id,
        classifier_spec_ids=[available_spec, unavailable_spec],
    )
    teacher = project.create_teaching_example(
        code_id=code_id, text="new teaching observation",
        label="positive", assess_before_training=True,
    )
    span, _event = project.annotate_span(
        [int(units[0]["unit_id"]), int(units[1]["unit_id"])],
        code_id, "negative",
    )
    for observation_id in (teacher, span):
        assert project.database.observation_vector_record(
            observation_id, available_id
        ) is not None
        assert project.database.observation_vector_record(
            observation_id, unavailable_id
        ) is None

    status = {
        int(row["classifier_spec_id"]): row["status"]
        for row in project.classifier_status(code_id=code_id)["classifiers"]
    }
    assert status[available_spec] == "stale"
    assert status[unavailable_spec] == "current"
    supported_fit = project.train_classifier(
        code_id=code_id, classifier_spec_id=available_spec,
    )
    fixed_fit = project.train_classifier(
        code_id=code_id, classifier_spec_id=unavailable_spec,
    )
    assert fixed_fit["training_selection"]["reason"] == "current_fit_reused"
    supported_ids = {
        int(row["observation_id"]) for row in
        project.database.classifier_predictions(int(supported_fit["classifier_fit_id"]))
    }
    fixed_ids = {
        int(row["observation_id"]) for row in
        project.database.classifier_predictions(int(fixed_fit["classifier_fit_id"]))
    }
    assert {teacher, span} <= supported_ids
    assert {teacher, span}.isdisjoint(fixed_ids)
    assert project.database.classifier_predictions_complete(
        int(fixed_fit["classifier_fit_id"])
    )
    assert not project.database.missing_classifier_prediction_observation_ids(
        int(fixed_fit["classifier_fit_id"])
    )
    assert all(
        row["status"] == "current"
        for row in project.classifier_status(code_id=code_id)["classifiers"]
    )

    committee_id = project.create_classifier_committee(
        code_id=code_id, name="mixed family committee",
        classifier_spec_ids=[available_spec, unavailable_spec],
        aggregation="logistic_stack",
    )
    committee_fit = project.train_classifier_committee(
        code_id=code_id, committee_id=committee_id,
    )
    committee_predictions = project.database.classifier_committee_predictions(
        int(committee_fit["committee_fit_id"])
    )
    assert {teacher, span}.isdisjoint({
        int(row["observation_id"]) for row in committee_predictions
    })
    assert project.classifier_committee_status(
        code_id=code_id, committee_id=committee_id
    )["status"] == "current"


def test_untransformable_external_project_still_disallows_derived_text(
    tmp_path: Path,
) -> None:
    project = _external_project(tmp_path)
    project.register_external_geometry(
        name="fixed", external_ref={"artifact": "minilm"},
        supports_text_transform=False,
    )
    assert not project.can_transform_new_observations()
    code_id = project.create_code("unsupported")
    with pytest.raises(ConfigurationError, match="No registered geometry"):
        project.create_teaching_example(
            code_id=code_id, text="not representable", label="positive"
        )


def test_partial_compatibility_keeps_teaching_controls_enabled(
    tmp_path: Path,
) -> None:
    """One unsupported geometry must not gray out the shared teaching manager."""
    from geometric_coder.ui.app import create_app

    project = _transforming_external_project(tmp_path)
    project.register_external_geometry(
        name="transformable", external_ref={"artifact": "minilm"},
        supports_text_transform=True,
    )
    project.register_external_geometry(
        name="fixed", external_ref={"artifact": "minilm"},
        supports_text_transform=False,
    )
    code_id = project.create_code("some supported")
    app = create_app(project)
    choices = [
        entry["callback"].__wrapped__
        for output, entry in app.callback_map.items()
        if "code-center-teaching-button.disabled" in output
        and "focus-teaching-button.disabled" in output
    ]
    assert len(choices) == 1
    assert choices[0](code_id, code_id) == (False, False)


def test_automatic_training_skips_geometry_missing_a_class(tmp_path: Path) -> None:
    """New-text-only labels cannot force fitting a fixed geometry with one class."""
    project = _transforming_external_project(tmp_path)
    compatible_id = project.register_external_geometry(
        name="transformable", external_ref={"artifact": "minilm"},
        supports_text_transform=True,
    )
    incompatible_id = project.register_external_geometry(
        name="fixed", external_ref={"artifact": "minilm"},
        supports_text_transform=False,
    )
    code_id = project.create_code("unequal evidence")
    compatible_spec = project.create_classifier_spec(
        code_id=code_id, name="can use teacher",
        geometry_id=compatible_id,
    )
    incompatible_spec = project.create_classifier_spec(
        code_id=code_id, name="atomic only",
        geometry_id=incompatible_id,
    )
    unit = project.database.list_units()[0]
    project.annotate(int(unit["observation_id"]), code_id, "negative")
    project.create_teaching_example(
        code_id=code_id, text="authored positive training item",
        label="positive",
    )
    assert project.training_label_counts(
        code_id, geometry_id=compatible_id
    ) == {"positive": 1, "negative": 1}
    assert project.training_label_counts(
        code_id, geometry_id=incompatible_id
    ) == {"positive": 0, "negative": 1}

    result = project.train_classifiers_after_evidence_change(
        code_id=code_id,
        classifier_spec_ids=[compatible_spec, incompatible_spec],
    )
    assert result["status"] == "trained"
    assert list(result["trained_fit_ids"]) == [compatible_spec]
    assert result["awaiting_classifier_spec_ids"] == [incompatible_spec]
    assert project.classifier_status(
        code_id=code_id, classifier_spec_ids=[incompatible_spec]
    )["classifiers"][0]["status"] == "not_trained"
