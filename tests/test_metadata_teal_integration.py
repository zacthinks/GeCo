"""Real TeAL 0.12 metadata-augmented geometry and GeCo predictor acceptance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

teal = pytest.importorskip("text_analysis_lab")
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from text_analysis_lab.integrations.geco import _geco_predictor_from_export
from text_analysis_lab.translators import CountVectorizer, FeatureMatrix

from geometric_coder.ui.representation_manager import (
    create_metadata_geometry, create_view, metadata_feature_options,
    representation_inventory, unregister_representation,
)


def _frame(artifact):
    return artifact.query(
        key_columns=True, data_columns=True, metadata_columns=False,
        order_by="_position", include_position=True, form="table",
    )


def test_metadata_featurematrix_linked_lifecycle_and_mixed_committee(tmp_path):
    path = tmp_path / "teal"
    csv = tmp_path / "input.csv"
    pd.DataFrame({
        "text": [
            "clear useful helpful", "excellent clear helpful", "useful good clear",
            "bad confusing weak", "difficult unclear bad", "weak terrible confusing",
            "strong helpful", "bad weak",
        ],
        "publication_year": [1990, 1999, 2000, 2009, 2010, 2019, 2020, 2025],
        "publication_source": ["A", "B", "A", None, "B", "C", "A", "C"],
        "document_length": [100, 120, 140, 200, 210, 220, 160, 230],
    }).to_csv(csv, index=False)
    project = teal.Project.create(path, name="metadata-geco")
    try:
        documents = project.read_csv(
            csv, text_fields="text",
            metadata_fields=["publication_year", "publication_source", "document_length"],
        )
        dtm = project.translate(CountVectorizer(text_field="text"), documents)["output"]
        linked = project.geco.create(
            "linked", documents=documents, text_field="text", geometry=dtm,
        )
        coder = linked.coder
        first_id = int(coder.geometries()[0]["geometry_id"])
        parent = coder.database.representation_for_legacy("geometry", first_id)
        source_representation_id = int(parent["representation_id"])
        discovered = metadata_feature_options(
            coder, source_representation_id=source_representation_id,
        )
        assert {row["value"] for row in discovered} >= {
            "publication_year", "publication_source", "document_length",
        }
        assert any(
            row["default_transform"] == "onehot"
            for row in discovered if row["value"] == "publication_source"
        )
        specs = {
            "publication_year": {
                "transform": "fixed_width", "width": 10, "anchor": 2000,
            },
            "publication_source": {"transform": "onehot"},
            "document_length": {"transform": "zscore"},
        }
        created = create_metadata_geometry(
            coder, name="tfidf_plus_metadata",
            source_representation_id=source_representation_id,
            metadata_specs=specs,
        )
        assert created.replayable is False
        registered = coder.database.get_geometry(created.geometry_id)
        assert not registered["supports_query"] and not registered["supports_text_transform"]
        child = coder.database.representation_for_legacy(
            "geometry", created.geometry_id,
        )
        assert int(child["parent_representation_id"]) == source_representation_id
        augmented = project.get_artifact(created.artifact_id)
        assert augmented.artifact_type.value == "sparse_matrix"
        assert "data" not in augmented.components  # TeAL owns a keys-only lazy recipe.
        source_matrix = dtm.get_matrix()
        values = augmented.get_matrix()
        assert sparse.issparse(values)
        assert sparse.issparse(coder.geometry_matrix(created.geometry_id))
        np.testing.assert_allclose(
            coder.geometry_matrix(created.geometry_id).toarray(),
            values.toarray(), rtol=0, atol=0,
        )
        np.testing.assert_allclose(
            values[:, :source_matrix.shape[1]].toarray(),
            source_matrix.toarray(),
        )
        assert augmented.get_data_columns()[:source_matrix.shape[1]] == dtm.get_data_columns()
        assert values.shape[1] == source_matrix.shape[1] + 4 + 3 + 1
        assert len(augmented.get_data_columns()) == values.shape[1]
        rows = representation_inventory(coder)
        entry = next(x for x in rows if x["artifact_id"] == created.artifact_id)
        assert entry["source"] and entry["new_text"] is False and entry["analytic_enabled"]
        with pytest.raises(Exception):
            unregister_representation(coder, key=f"representation:{source_representation_id}")
        with pytest.raises(Exception):
            coder.transform_query(created.geometry_id, "arbitrary query text")

        view_rep = create_view(
            coder, name="Augmented two dimensions",
            source_representation_id=int(child["representation_id"]),
            method="svd", dimensions=2,
        )
        assert coder.representation_matrix(view_rep).shape == (8, 2)

        code_id = coder.create_code("quality")
        observations = coder.database.list_observations()
        for idx in (0, 1, 2):
            coder.annotate(int(observations[idx]["observation_id"]), code_id, "positive")
        for idx in (3, 4, 5):
            coder.annotate(int(observations[idx]["observation_id"]), code_id, "negative")
        baseline_id = coder.create_classifier_spec(
            code_id=code_id, name="base lexical", geometry_id=first_id,
            algorithm="logistic_l2", hyperparameters=None,
        )
        metadata_id = coder.create_classifier_spec(
            code_id=code_id, name="metadata augmented", geometry_id=created.geometry_id,
            algorithm="logistic_l2", hyperparameters={"feature_scaling": "fixed"},
        )
        coder.train_classifiers(
            code_id=code_id, classifier_spec_ids=[baseline_id, metadata_id],
        )
        assert coder.classifier_status(code_id=code_id)["classifiers"]
        committee_id = coder.create_classifier_committee(
            code_id=code_id, name="different geometries",
            classifier_spec_ids=[baseline_id, metadata_id], aggregation="mean",
        )
        frozen = coder.export_predictor(
            next(ref for ref in coder.predictors() if ref.kind == "committee" and ref.id == committee_id)
        )
        assert len(frozen.sources) == 2
        source_matrices = [coder.geometry_matrix(x.geometry_id) for x in frozen.sources]
        expected = frozen.predict_proba(source_matrices)
        predictor = _geco_predictor_from_export(frozen)
        artifact_by_id = {
            dtm.artifact_id: dtm, augmented.artifact_id: augmented,
        }
        ordered = [
            artifact_by_id[source.external_ref["artifact_id"]]
            if source.external_ref else (
                augmented if source.geometry_id == created.geometry_id else dtm
            )
            for source in frozen.sources
        ]
        result = project.translate(predictor, ordered, batch_size=3)["output"]
        np.testing.assert_allclose(_frame(result)["probability"], expected, atol=1e-12)
        op_id = predictor.operator_id
        ids = [artifact.artifact_id for artifact in ordered]
        augmented_id = created.artifact_id
        linked.close()
    finally:
        project.close()

    reopened = teal.Project.open(path)
    try:
        with reopened.geco.open("linked") as linked:
            coder = linked.coder
            assert coder.database.get_geometry(created.geometry_id)["supports_query"] is False
            actual = coder.geometry_matrix(created.geometry_id)
            assert sparse.issparse(actual)
            np.testing.assert_allclose(actual.toarray(), values.toarray())
        operator = reopened.get_operator(op_id)
        result = reopened.translate(
            operator, [reopened.get_artifact(i) for i in ids],
            batch_size=2,
        )["output"]
        np.testing.assert_allclose(_frame(result)["probability"], expected, atol=1e-12)
        assert reopened.get_artifact(augmented_id).get_data_columns() == augmented.get_data_columns()
    finally:
        reopened.close()


def test_metadata_source_only_no_geco_sqlite_fallback(tmp_path):
    path = tmp_path / "lab"
    csv = tmp_path / "data.csv"
    pd.DataFrame({
        "text": ["alpha beta", "beta alpha", "gamma delta"],
        "extra": ["A", "B", "C"],
    }).to_csv(csv, index=False)
    project = teal.Project.create(path, name="no-silent-metadata")
    try:
        docs = project.read_csv(
            csv, text_fields="text", metadata_fields=["extra"],
        )
        dtm = project.translate(CountVectorizer(text_field="text"), docs)["output"]
        linked = project.geco.create(
            "linked", documents=docs, text_field="text", geometry=dtm,
        )
        coder = linked.coder
        source_id = int(coder.database.representation_for_legacy(
            "geometry", coder.geometries()[0]["geometry_id"]
        )["representation_id"])
        with pytest.raises(ValueError, match="does not expose metadata"):
            create_metadata_geometry(
                coder, name="invalid_metadata", source_representation_id=source_id,
                metadata_specs={"nonexistent": {"transform": "onehot"}},
            )
        assert not coder.has_geometry("invalid_metadata")
        linked.close()
    finally:
        project.close()
