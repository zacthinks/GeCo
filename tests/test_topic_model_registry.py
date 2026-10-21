"""R1 Topic Model registration is GeCo-local metadata over opaque provider refs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.exceptions import ConfigurationError
from geometric_coder.storage import SCHEMA_VERSION


class _Provider:
    """R1 must not call a provider to register metadata."""


def _project(tmp_path: Path) -> GeometricCoder:
    return GeometricCoder.create_external(
        project_dir=tmp_path / "topic_registry.geco",
        data=pd.DataFrame(
            {"document_id": ["a", "b", "c"], "turn": [1, 2, 3], "text": ["one", "two", "three"]}
        ),
        keys=["document_id", "turn"],
        text="text",
        metadata=[],
        external_provider=_Provider(),
    )


def test_register_two_models_and_preserve_opaque_refs_without_provider_calls(tmp_path: Path) -> None:
    project = _project(tmp_path)
    lda_id = project.register_topic_model(
        name="LDA", membership_ref={"artifact_id": "m01"},
        description_ref={"artifact_id": "m02"},
        source_ref={"artifact_id": "documents"},
        membership_kind="soft_probabilities", n_components=2,
        component_ids=["topic_0", "topic_1"],
    )
    cluster_id = project.register_topic_model(
        name="HDBSCAN", membership_ref={"artifact_id": "m03"},
        membership_kind="hard_exclusive", n_components=3,
        component_ids=["cluster_0", "cluster_1", "cluster_2"],
    )
    assert lda_id != cluster_id
    assert [m["name"] for m in project.topic_models()] == ["LDA", "HDBSCAN"]
    lda = project.topic_model(lda_id)
    assert lda["membership_ref"] == {"artifact_id": "m01"}
    assert lda["description_ref"] == {"artifact_id": "m02"}
    assert lda["source_ref"] == {"artifact_id": "documents"}
    assert lda["source_key_columns"] == ["document_id", "turn"]
    assert lda["source_grain"] == "atomic"
    assert lda["component_ids"] == ["topic_0", "topic_1"]
    assert lda["topic_names"] == ["Topic 0", "Topic 1"]
    assert lda["selected_component"] is None
    assert project.topic_model("HDBSCAN")["description_ref"] is None
    assert project.topic_model(cluster_id)["membership_kind"] == "hard_exclusive"


def test_rename_topics_and_display_persist_without_mutating_source(tmp_path: Path) -> None:
    project = _project(tmp_path)
    artifact = {"artifact_id": "lda", "selected_columns": [0, 1]}
    model_id = project.register_topic_model(
        name="LDA", membership_ref=artifact,
        membership_kind="soft_probabilities", n_components=2,
    )
    project.rename_topic_model(model_id, name="Interview themes")
    project.rename_topic(model_id, component_index=1, name="Work")
    project.set_topic_model_display(model_id, mode="gradient", selected_component=1)
    reopened = GeometricCoder.open(project.project_dir, external_provider=_Provider())
    model = reopened.topic_model(model_id)
    assert model["name"] == "Interview themes"
    assert model["topic_names"] == ["Topic 0", "Work"]
    assert model["component_ids"] == ["topic_0", "topic_1"]
    assert model["membership_ref"] == artifact
    assert model["display_mode"] == "gradient"
    assert model["selected_component"] == 1
    reopened.set_topic_model_display(model_id, mode="categorical")
    assert reopened.topic_model(model_id)["display_mode"] == "categorical"


def test_if_exists_reuse_requires_matching_immutable_identity(tmp_path: Path) -> None:
    project = _project(tmp_path)
    spec = dict(name="saved", membership_ref="artifact-7",
                membership_kind="soft_probabilities", n_components=2)
    model_id = project.register_topic_model(**spec)
    project.rename_topic(model_id, component_index=0, name="Locally renamed")
    assert project.register_topic_model(**spec, if_exists="reuse") == model_id
    assert project.topic_model(model_id)["topic_names"][0] == "Locally renamed"
    with pytest.raises(ConfigurationError, match="already"):
        project.register_topic_model(**spec)
    with pytest.raises(ConfigurationError, match="different immutable"):
        project.register_topic_model(**{**spec, "membership_ref": "artifact-8"}, if_exists="reuse")
    assert len(project.topic_models()) == 1


@pytest.mark.parametrize("patch", [
    {"name": ""},
    {"name": "   "},
    {"membership_ref": None},
    {"membership_ref": object()},
    {"membership_kind": "soft"},
    {"n_components": 0},
    {"n_components": True},
    {"component_ids": ["a"]},
    {"component_ids": ["a", "a"]},
    {"topic_names": ["one"]},
    {"if_exists": "overwrite"},
])
def test_invalid_registration_is_atomic(tmp_path: Path, patch: dict) -> None:
    project = _project(tmp_path)
    kwargs = dict(
        name="valid", membership_ref={"artifact_id": "m"},
        membership_kind="soft_probabilities", n_components=2,
    )
    with pytest.raises((ConfigurationError, ValueError)):
        project.register_topic_model(**{**kwargs, **patch})
    assert project.topic_models() == []


def test_rename_conflicts_and_invalid_display_leave_state_unchanged(tmp_path: Path) -> None:
    project = _project(tmp_path)
    a = project.register_topic_model(
        name="A", membership_ref="a", membership_kind="soft_probabilities", n_components=2,
    )
    b = project.register_topic_model(
        name="B", membership_ref="b", membership_kind="hard_exclusive", n_components=2,
    )
    with pytest.raises(ConfigurationError, match="already exists"):
        project.rename_topic_model(b, name="A")
    with pytest.raises(ConfigurationError, match="gradient"):
        project.set_topic_model_display(b, mode="gradient", selected_component=0)
    with pytest.raises(IndexError):
        project.set_topic_model_display(a, mode="categorical", selected_component=9)
    with pytest.raises(ConfigurationError, match="required"):
        project.set_topic_model_display(a, mode="gradient")
    with pytest.raises(IndexError):
        project.rename_topic(a, component_index=2, name="Missing")
    assert project.topic_model(a)["display_mode"] == "categorical"
    assert project.topic_model(b)["name"] == "B"


def test_registry_has_no_matrix_copy_or_human_coding_side_effects(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("Substantive theme")
    project.register_topic_model(
        name="clusters", membership_ref="opaque", membership_kind="hard_exclusive",
        n_components=2,
    )
    assert project.code(code_id)["name"] == "Substantive theme"
    assert len(project.codes()) == 1
    with sqlite3.connect(project.project_dir / project.DATABASE_FILENAME) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "topic_models" in tables
        assert not any(x.startswith("topic_model_vectors") for x in tables)
        assert connection.execute("SELECT schema_version FROM project").fetchone()[0] == SCHEMA_VERSION


def test_native_project_can_continue_without_topic_provider(tmp_path: Path) -> None:
    project = GeometricCoder.create(
        project_dir=tmp_path / "native.geco",
        data=pd.DataFrame({"row_id": [1, 2], "text": ["a", "b"]}),
        keys=["row_id"], text="text", metadata=[], geometries=None,
    )
    assert project.topic_models() == []
    with pytest.raises(ConfigurationError, match="provider-backed"):
        project.register_topic_model(
            name="unsupported", membership_ref="a",
            membership_kind="hard_exclusive", n_components=2,
        )
    assert project.topic_models() == []
