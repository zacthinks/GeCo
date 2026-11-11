"""Persisted TeAL + GeCo representation creation and UI integration.

This test runs in the dedicated cross-repository CI job with the current TeAL
checkout installed. It is skipped in GeCo-only test environments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

teal = pytest.importorskip("text_analysis_lab")
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.translators import CountVectorizer

from geometric_coder.ui import create_app
from geometric_coder.ui.app import (
    _display_view_records, _presentation_view_records, _presentation_view_options,
)
from geometric_coder.ui.representation_manager import (
    create_geometry, create_view, representation_inventory, rename_representation,
    set_representation_analytic_enabled, unregister_representation,
)
from geometric_coder.ui.topic_creation import topic_geometry_options


def _component_by_id(component, component_id):
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    if children is None:
        raise LookupError(component_id)
    for child in children if isinstance(children, (tuple, list)) else [children]:
        if child is not None and not isinstance(child, (str, int, float, bool)):
            try:
                return _component_by_id(child, component_id)
            except LookupError:
                pass
    raise LookupError(component_id)


def _seed_documents(project):
    docs = pd.DataFrame({
        "row_id": list(range(12)),
        "text": [
            "apple banana orange pear", "pear apple banana kiwi",
            "car train highway bus", "bus train car railroad",
            "biology cells protein gene", "protein cells gene enzyme",
            "finance revenue profit shares", "shares revenue earnings profit",
            "climate warming carbon emissions", "carbon climate weather warming",
            "music rhythm piano melody", "melody piano music rhythm",
        ],
    })
    writer = create_artifact_writer(
        artifact_type="table",
        artifact_dir=project.storage.artifact_dir("integration_documents"),
        artifact_id="integration_documents",
        label="documents", lineage_mode="new_key", basis_artifact_ids=(),
    )
    writer.write({"keys": docs[["row_id"]], "data": docs[["text"]]})
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id="integration_documents", artifact_type="table",
        label="documents", lineage_mode="new_key", status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact("integration_documents")


def test_live_teal_representations_survive_reopen_and_ui_callbacks(tmp_path: Path) -> None:
    project_path = tmp_path / "teal_project"
    project = teal.Project.create(project_path, name="geco_representation_integration")
    try:
        documents = _seed_documents(project)
        counts = project.translate(
            CountVectorizer(text_field="text", min_df=1), documents
        )["output"]
        assert sparse.issparse(counts.get_matrix())
        linked = project.geco.create(
            "linked", documents=documents, text_field="text", geometry=counts,
        )
        coder = linked.coder
        base_id = int(coder.geometries()[0]["geometry_id"])
        base_rep = coder.database.representation_for_legacy(
            "geometry", base_id
        )["representation_id"]
        weighted = create_geometry(
            coder, name="tfidf_weighted", recipe="tfidf",
            source_representation_id=base_rep,
        )
        assert weighted.replayable
        reduced = create_geometry(
            coder, name="svd_four", recipe="svd",
            source_geometry_id=weighted.geometry_id,
            parameters={"n_components": 4, "random_state": 7},
        )
        reduced_id = coder.database.representation_for_legacy(
            "geometry", reduced.geometry_id
        )["representation_id"]
        view_id = create_view(
            coder, name="svd_two", method="svd",
            source_representation_id=reduced_id,
            dimensions=2, parameters={"random_state": 7},
        )
        one_id = create_view(
            coder, name="svd_one", method="svd",
            source_representation_id=reduced_id,
            dimensions=1, parameters={"random_state": 7},
        )
        matrix = coder.representation_matrix(reduced_id)
        assert matrix.shape == (12, 4)
        assert coder.representation_matrix(view_id).shape == (12, 2)
        assert coder.representation_matrix(one_id).shape == (12, 1)
        assert {row["value"] for row in topic_geometry_options(coder)} >= {
            base_id, weighted.geometry_id, reduced.geometry_id,
        }
        assert any(
            record["representation_id"] == view_id
            and not record["analytic_enabled"]
            for record in representation_inventory(coder)
        )
        assert {record["representation_id"] for record in
                coder.representation_display_candidates(base_rep)} >= {view_id, one_id}
        rename_representation(coder, key=f"representation:{view_id}", name="svd_two_renamed")
        assert coder.database.get_representation(view_id)["name"] == "svd_two_renamed"
        session_id = int(coder.sessions()[0]["session_id"])
        coder.patch_session_state(session_id, {
            "geometry_id": reduced.geometry_id,
            "view_id": -one_id,
        })
        # Active saved view is a real dependency, even for view-only geometry
        # registrations without separate legacy view rows.
        with pytest.raises(Exception, match="saved session"):
            unregister_representation(coder, key=f"representation:{one_id}")
        linked.close()
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        with reopened.geco.open("linked") as linked:
            coder = linked.coder
            assert coder.database.get_representation(view_id)["name"] == "svd_two_renamed"
            assert coder.database.get_representation(one_id)["dimensions"] == 1
            assert coder.database.get_representation(reduced_id)["dimensions"] == 4
            assert coder.database.get_session(session_id)["state"]["view_id"] == -one_id
            assert coder.representation_matrix(reduced_id).shape == (12, 4)
            options = _display_view_records(coder, reduced.geometry_id)
            assert -one_id in {int(row["view_id"]) for row in options}
            assert -view_id in {int(row["view_id"]) for row in options}
            presentation = _presentation_view_records(coder, coder.geometries())
            assert -one_id in {int(row["view_id"]) for row in presentation}
            assert -one_id in {
                int(row["value"]) for row in _presentation_view_options(
                    presentation, coder.geometries()
                )
            }
            app = create_app(coder)
            assert _component_by_id(app.layout, "view-dropdown").value == -one_id
            assert _component_by_id(app.layout, "representation-manager-modal") is not None
            assert _component_by_id(app.layout, "memo-presentation-view").value == -one_id
            assert _component_by_id(app.layout, "code-center-view") is not None
            client = app.server.test_client()
            for path in ("/", "/_dash-layout", "/_dash-dependencies"):
                assert client.get(path).status_code == 200
            assert coder.view_coordinates(-one_id).shape == (12, 1)
    finally:
        reopened.close()


def test_spacy_source_text_dtm_registers_with_original_keys(tmp_path: Path) -> None:
    """Real TeAL spaCy-token/subset/count pipeline, no model download."""
    spacy = pytest.importorskip("spacy")
    from geometric_coder.ui.text_geometry import create_text_geometry

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    model_dir = tmp_path / "local_spacy"
    nlp.to_disk(model_dir)

    project_path = tmp_path / "text_source_project"
    project = teal.Project.create(project_path, name="text_source_representation")
    try:
        documents = _seed_documents(project)
        baseline = project.translate(
            CountVectorizer(text_field="text", min_df=1), documents,
        )["output"]
        linked = project.geco.create(
            "linked_text", documents=documents, text_field="text", geometry=baseline,
        )
        coder = linked.coder
        result = create_text_geometry(
            coder, name="spacy_tokens", method="spacy_dtm",
            model=str(model_dir),
            token_field="text", token_flags={"is_alpha": "true"},
            ngram_min=1, ngram_max=1, min_df=1,
        )
        assert result.geometry_id > 0
        assert result.artifact_id
        vector = project.get_artifact(result.artifact_id)
        assert vector.primary_key == documents.primary_key
        assert vector.n_rows == documents.n_rows
        assert sparse.issparse(vector.get_matrix())
        assert "apple" in vector.get_data_columns()
        assert coder.geometry_matrix(result.geometry_id).shape[0] == documents.n_rows
        linked.close()
    finally:
        project.close()

    reopened = teal.Project.open(project_path)
    try:
        with reopened.geco.open("linked_text") as linked:
            names = {record["name"] for record in linked.coder.geometries()}
            assert "spacy_tokens" in names
    finally:
        reopened.close()


def test_sentence_transformer_source_text_uses_teal_encoder_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use a deterministic synthetic encoder, but real TeAL artifacts and link."""
    from text_analysis_lab.translators.sentence_transformer_encoder import (
        SentenceTransformerEncoder,
    )
    from geometric_coder.ui.text_geometry import create_text_geometry

    def fake_runtime(self, *, device):
        self.resolved_revision = "a" * 40
        return object()

    def fake_translation(self, texts, *, device, model_batch_size,
                         context_examples=None):
        batch = [texts] if isinstance(texts, str) else list(texts)
        values = np.array(
            [[float(len(str(item))), float(len(str(item).split()))] for item in batch],
            dtype=np.float32,
        )
        return values, [2] * len(batch), 256

    monkeypatch.setattr(SentenceTransformerEncoder, "_runtime_component", fake_runtime)
    monkeypatch.setattr(SentenceTransformerEncoder, "_translate_texts", fake_translation)

    path = tmp_path / "sentence_encoder_project"
    project = teal.Project.create(path, name="source_text_embeddings")
    try:
        documents = _seed_documents(project)
        baseline = project.translate(CountVectorizer(), documents)["output"]
        linked = project.geco.create(
            "source_text", documents=documents, text_field="text", geometry=baseline,
        )
        created = create_text_geometry(
            linked.coder, name="test_embeddings", method="sentence_transformer",
            model="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
            revision="a" * 40, normalize=True, truncation="error",
        )
        assert created.replayable
        matrix = project.get_artifact(created.artifact_id)
        assert matrix.primary_key == documents.primary_key
        assert matrix.get_matrix().shape == (12, 2)
        assert linked.coder.geometry_matrix(created.geometry_id).shape == (12, 2)
        linked.close()
    finally:
        project.close()
