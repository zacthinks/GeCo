"""Derived annotations are durable at authoring time and vectors are lazy."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geometric_coder import GeometricCoder


class RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def geometry_matrix(self, external_ref: Any, ordered_user_keys: Any) -> np.ndarray:
        return np.asarray([[1., 0.], [0., 1.], [0.8, 0.2]], dtype=np.float32)

    def transform_texts(self, external_ref: Any, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.asarray([[float(len(text)), 1.] for text in texts], dtype=np.float32)


def make_project(tmp_path: Path) -> tuple[GeometricCoder, RecordingProvider, int, int, int]:
    provider = RecordingProvider()
    project = GeometricCoder.create_external(
        project_dir=tmp_path / "lazy.geco",
        data=pd.DataFrame({"row_id": [0, 1, 2], "text": ["alpha", "beta", "gamma"]}),
        keys=["row_id"], text="text", metadata=[], external_provider=provider,
    )
    replayable = project.register_external_geometry(
        name="replayable", external_ref={"id": "a"}, supports_text_transform=True,
    )
    fixed = project.register_external_geometry(
        name="fixed", external_ref={"id": "b"}, supports_text_transform=False,
    )
    code_id = project.create_code("lazy derived")
    return project, provider, replayable, fixed, code_id


def test_spans_and_teaching_are_immediate_then_batched_for_one_geometry(tmp_path: Path) -> None:
    project, provider, replayable, fixed, code_id = make_project(tmp_path)
    spec = project.create_classifier_spec(
        code_id=code_id, name="lazy classifier", geometry_id=replayable,
    )
    units = project.database.list_units()
    project.annotate(int(units[2]["observation_id"]), code_id, "negative")
    teaching_id = project.create_teaching_example(
        code_id=code_id, text="teaching text", label="positive",
    )
    span_id, _ = project.annotate_span(
        [int(units[0]["unit_id"]), int(units[1]["unit_id"])],
        code_id, "positive",
    )
    assert provider.calls == []
    assert project.training_label_counts(
        code_id, geometry_id=replayable
    ) == {"positive": 2, "negative": 1}
    assert project.training_label_counts(
        code_id, geometry_id=fixed
    ) == {"positive": 0, "negative": 1}
    assert project.database.observation_vector_record(teaching_id, replayable) is None
    result = project.train_classifier(code_id=code_id, classifier_spec_id=spec)
    assert len(provider.calls) == 1
    assert set(provider.calls[0]) == {"teaching text", "alpha\nbeta"}
    for observation_id in (teaching_id, span_id):
        assert project.database.observation_vector_record(observation_id, replayable)
        assert project.database.observation_vector_record(observation_id, fixed) is None
    project.materialize_derived_vectors(replayable)
    assert len(provider.calls) == 1
    assert project.database.classifier_predictions_complete(int(result["classifier_fit_id"]))


def test_diagnostics_only_materialize_the_tested_observation(tmp_path: Path) -> None:
    project, provider, replayable, _fixed, code_id = make_project(tmp_path)
    spec = project.create_classifier_spec(
        code_id=code_id, name="tested model", geometry_id=replayable,
    )
    units = project.database.list_units()
    project.annotate(int(units[0]["observation_id"]), code_id, "positive")
    project.annotate(int(units[1]["observation_id"]), code_id, "negative")
    project.train_classifier(code_id=code_id, classifier_spec_id=spec)
    ordinary = project.create_teaching_example(
        code_id=code_id, text="ordinary", label="positive",
    )
    assert provider.calls == []
    tested = project.create_teaching_example(
        code_id=code_id, text="diagnostic", label="negative",
        assess_before_training=True,
    )
    assert provider.calls == [["diagnostic"]]
    assert project.database.observation_vector_record(tested, replayable)
    assert project.database.observation_vector_record(ordinary, replayable) is None
    assert any(
        row["event_type"] == "teaching_example_pretraining"
        and int(row["observation_id"]) == tested
        for row in project.database.list_classifier_evaluations(code_id=code_id)
    )


def test_unsupported_geometry_never_requires_derived_vectors(tmp_path: Path) -> None:
    project, provider, replayable, fixed, code_id = make_project(tmp_path)
    teaching_id = project.create_teaching_example(
        code_id=code_id, text="durable", label="negative",
    )
    assert project.materialize_derived_vectors(fixed) == []
    assert project.database.observation_vector_record(teaching_id, fixed) is None
    assert provider.calls == []
    project.materialize_derived_vectors(replayable)
    assert provider.calls == [["durable"]]
    project.materialize_derived_vectors(replayable)
    assert len(provider.calls) == 1
