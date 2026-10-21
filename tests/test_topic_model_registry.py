"""R1 stores model artifact IDs and editable topic aliases, without numerical data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.exceptions import ConfigurationError
from geometric_coder.storage import SCHEMA_VERSION


class _Provider:
    """R1 registration must not call a provider."""


def _project(tmp_path: Path) -> GeometricCoder:
    return GeometricCoder.create_external(
        project_dir=tmp_path / "topic_registry.geco",
        data=pd.DataFrame({
            "document_id": ["a", "b", "c"], "turn": [1, 2, 3],
            "text": ["one", "two", "three"],
        }),
        keys=["document_id", "turn"], text="text", metadata=[],
        external_provider=_Provider(),
    )


def test_required_two_artifacts_and_optional_descriptions(tmp_path: Path) -> None:
    project = _project(tmp_path)
    lda_id = project.register_topic_model(
        name="LDA", membership_artifact_id="doc-topic",
        dominant_artifact_id="doc-argmax",
        description_artifact_id="topic-words",
        topic_labels={"topic_0": "Apple"},
    )
    cluster_id = project.register_topic_model(
        name="HDBSCAN", membership_artifact_id="hard-memberships",
        dominant_artifact_id="hard-argmax",
    )
    assert lda_id != cluster_id
    assert [m["name"] for m in project.topic_models()] == ["LDA", "HDBSCAN"]
    lda = project.topic_model(lda_id)
    assert lda == {
        "topic_model_id": lda_id,
        "name": "LDA",
        "membership_artifact_id": "doc-topic",
        "dominant_artifact_id": "doc-argmax",
        "description_artifact_id": "topic-words",
        "topic_labels": {"topic_0": "Apple"},
    }
    assert project.topic_model("HDBSCAN")["description_artifact_id"] is None


def test_local_renames_reopen_and_preserve_original_topic_ids(tmp_path: Path) -> None:
    project = _project(tmp_path)
    model_id = project.register_topic_model(
        name="model", membership_artifact_id="membership",
        dominant_artifact_id="argmax",
    )
    project.rename_topic_model(model_id, name="Interview themes")
    project.rename_topic(model_id, component_id="topic_0", name="Apple")
    project.rename_topic(model_id, component_id="topic_1", name="Work")
    reopened = GeometricCoder.open(project.project_dir, external_provider=_Provider())
    assert reopened.topic_model(model_id)["name"] == "Interview themes"
    assert reopened.topic_model(model_id)["topic_labels"] == {
        "topic_0": "Apple", "topic_1": "Work"
    }
    assert reopened.topic_model(model_id)["membership_artifact_id"] == "membership"
    assert reopened.topic_model(model_id)["dominant_artifact_id"] == "argmax"
    # R1 only stores aliases. R2 will inspect original feature names in TeAL.


def test_reuse_keeps_renamed_labels_and_detects_incompatible_refs(tmp_path: Path) -> None:
    project = _project(tmp_path)
    spec = dict(name="model", membership_artifact_id="a",
                dominant_artifact_id="a-argmax")
    model_id = project.register_topic_model(**spec)
    project.rename_topic(model_id, component_id="topic_0", name="Custom")
    assert project.register_topic_model(**spec, if_exists="reuse") == model_id
    assert project.topic_model(model_id)["topic_labels"]["topic_0"] == "Custom"
    with pytest.raises(ConfigurationError, match="already"):
        project.register_topic_model(**spec)
    with pytest.raises(ConfigurationError, match="different artifact"):
        project.register_topic_model(
            **{**spec, "dominant_artifact_id": "different"}, if_exists="reuse"
        )
    with pytest.raises(ConfigurationError, match="different artifact"):
        project.register_topic_model(
            **{**spec, "description_artifact_id": "new"}, if_exists="reuse"
        )
    assert len(project.topic_models()) == 1


@pytest.mark.parametrize("patch", [
    {"name": ""}, {"name": "  "},
    {"membership_artifact_id": ""}, {"membership_artifact_id": None},
    {"dominant_artifact_id": ""}, {"dominant_artifact_id": None},
    {"membership_artifact_id": {"artifact_id": "x"}},
    {"dominant_artifact_id": ["x"]},
    {"membership_artifact_id": "same", "dominant_artifact_id": "same"},
    {"description_artifact_id": ""},
    {"description_artifact_id": "memberships"},
    {"description_artifact_id": "dominant"},
    {"topic_labels": ["invalid"]},
    {"topic_labels": {"": "Apple"}},
    {"topic_labels": {"topic_0": "  "}},
    {"if_exists": "overwrite"},
])
def test_registration_validation_and_atomicity(tmp_path: Path, patch: dict) -> None:
    project = _project(tmp_path)
    kwargs = dict(
        name="model", membership_artifact_id="memberships",
        dominant_artifact_id="dominant",
    )
    with pytest.raises((ConfigurationError, ValueError)):
        project.register_topic_model(**{**kwargs, **patch})
    assert project.topic_models() == []


def test_rename_conflicts_and_invalid_labels(tmp_path: Path) -> None:
    project = _project(tmp_path)
    a = project.register_topic_model(
        name="A", membership_artifact_id="membership_a", dominant_artifact_id="max_a",
    )
    b = project.register_topic_model(
        name="B", membership_artifact_id="membership_b", dominant_artifact_id="max_b",
    )
    with pytest.raises(ConfigurationError, match="already exists"):
        project.rename_topic_model(b, name="A")
    with pytest.raises(ConfigurationError, match="non-empty"):
        project.rename_topic(a, component_id="", name="Bad")
    with pytest.raises(ConfigurationError, match="non-empty"):
        project.rename_topic(a, component_id="topic_0", name="")
    with pytest.raises(KeyError):
        project.rename_topic(9999, component_id="topic_0", name="Missing")
    assert project.topic_model(a)["topic_labels"] == {}
    assert project.topic_model(b)["name"] == "B"


def test_no_matrix_copy_or_human_coding_mutation(tmp_path: Path) -> None:
    project = _project(tmp_path)
    code_id = project.create_code("Substantive theme")
    project.register_topic_model(
        name="model", membership_artifact_id="membership",
        dominant_artifact_id="argmax",
    )
    assert project.code(code_id)["name"] == "Substantive theme"
    assert len(project.codes()) == 1
    path = project.project_dir / project.DATABASE_FILENAME
    with sqlite3.connect(path) as connection:
        schema = {
            row[1] for row in connection.execute("PRAGMA table_info(topic_models)")
        }
        assert schema == {
            "topic_model_id", "name", "membership_artifact_id",
            "dominant_artifact_id", "description_artifact_id", "topic_labels_json",
        }
        assert connection.execute(
            "SELECT schema_version FROM project"
        ).fetchone()[0] == SCHEMA_VERSION


def test_native_workspaces_continue_to_work(tmp_path: Path) -> None:
    project = GeometricCoder.create(
        project_dir=tmp_path / "native.geco",
        data=pd.DataFrame({"row_id": [1, 2], "text": ["a", "b"]}),
        keys=["row_id"], text="text", metadata=[], geometries=None,
    )
    assert project.topic_models() == []
    with pytest.raises(ConfigurationError, match="provider-backed"):
        project.register_topic_model(
            name="model", membership_artifact_id="membership",
            dominant_artifact_id="argmax",
        )
