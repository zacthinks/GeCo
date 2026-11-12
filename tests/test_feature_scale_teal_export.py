"""End-to-end GeCo scaled classifier to persisted TeAL GeCoPredictor contract."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

pytest.importorskip("text_analysis_lab")
pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

import text_analysis_lab as teal
from text_analysis_lab.core.writer import create_artifact_writer
from text_analysis_lab.integrations.geco import _geco_predictor_from_export

from geometric_coder import CountGeometry, GeometricCoder, SVDGeometry


def _seed_matrix(project, artifact_id, values):
    n_rows, n_cols = values.shape
    writer = create_artifact_writer(
        artifact_type="sparse_matrix" if sparse.issparse(values) else "dense_matrix",
        artifact_dir=project.storage.artifact_dir(artifact_id),
        artifact_id=artifact_id,
        label=artifact_id,
        lineage_mode="new_key",
        basis_artifact_ids=(),
    )
    writer.write({
        "keys": pd.DataFrame({"row_id": list(range(n_rows))}),
        "data": {"values": values, "columns": [f"x{i}" for i in range(n_cols)]},
    })
    writer.finalize()
    project.catalog.register_artifact(
        artifact_id=artifact_id,
        artifact_type="sparse_matrix" if sparse.issparse(values) else "dense_matrix",
        label=artifact_id,
        lineage_mode="new_key",
        status="complete",
        basis_artifact_ids=(),
    )
    return project.get_artifact(artifact_id)


@pytest.mark.parametrize("committee", [False, True])
def test_real_geco_scaled_predictor_reopens_in_teal(tmp_path, committee):
    text = [
        "excellent clear helpful", "excellent useful strong", "helpful clear useful",
        "bad confusing weak", "terrible unclear weak", "bad difficult confusing",
        "excellent clear", "bad weak",
    ]
    lexical = CountGeometry(lemmatize=False, weighting="tfidf")
    geometries = {"lexical": lexical}
    if committee:
        geometries["svd"] = SVDGeometry(lexical, n_components=2)
    coder = GeometricCoder.create(
        project_dir=tmp_path / "coder.geco",
        data=pd.DataFrame({"row_id": range(8), "text": text}),
        keys=["row_id"], text="text", metadata=[], geometries=geometries,
    )
    code_id = coder.create_code("quality")
    observations = coder.database.list_observations()
    for position in range(3):
        coder.annotate(int(observations[position]["observation_id"]), code_id, "positive")
    for position in range(3, 6):
        coder.annotate(int(observations[position]["observation_id"]), code_id, "negative")

    geometry_ids = {g["name"]: int(g["geometry_id"]) for g in coder.geometries()}
    first = coder.create_classifier_spec(
        code_id=code_id, name="lexical scaler",
        geometry_id=geometry_ids["lexical"], algorithm="logistic_l2",
        hyperparameters={"feature_scaling": "fixed"},
    )
    coder.train_classifier(code_id=code_id, classifier_spec_id=first)
    ids = [first]
    if committee:
        second = coder.create_classifier_spec(
            code_id=code_id, name="svd scaler",
            geometry_id=geometry_ids["svd"], algorithm="logistic_l1",
            hyperparameters={"feature_scaling": "tune", "feature_scale_gamma": 0.5},
        )
        coder.train_classifier(code_id=code_id, classifier_spec_id=second)
        ids.append(second)
        committee_id = coder.create_classifier_committee(
            code_id=code_id, name="mixed scales",
            classifier_spec_ids=ids, aggregation="mean",
        )
        ref = next(x for x in coder.predictors() if x.kind == "committee" and x.id == committee_id)
    else:
        ref = next(x for x in coder.predictors() if x.kind == "classifier" and x.id == first)
    frozen = coder.export_predictor(ref)
    source_matrices = [coder.geometry_matrix(source.geometry_id) for source in frozen.sources]
    expected = frozen.predict_proba(source_matrices)
    assert all(
        b"geometric_coder.preprocessing" not in pickle.dumps(member.estimator)
        for member in frozen.members
    )

    predictor = _geco_predictor_from_export(frozen)
    path = tmp_path / "teal"
    project = teal.Project.create(path, name="portable_geco_scale")
    try:
        sources = [
            _seed_matrix(project, f"geometry_{index}", source)
            for index, source in enumerate(source_matrices)
        ]
        output = project.translate(predictor, sources, batch_size=3)["output"]
        values = output.query(
            key_columns=True, data_columns=True, metadata_columns=False,
            order_by="_position", include_position=True, form="table",
        )
        np.testing.assert_allclose(values["probability"], expected, rtol=1e-12, atol=1e-12)
        operator_id = predictor.operator_id
        artifact_ids = [source.artifact_id for source in sources]
    finally:
        project.close()

    # The saved TeAL operator assets contain sklearn pipeline state only for the
    # feature normalizer and do not need its GeCo implementation on reload.
    reopened = teal.Project.open(path)
    try:
        loaded = reopened.get_operator(operator_id)
        assert b"geometric_coder.preprocessing" not in pickle.dumps(loaded.member_models)
        reused = reopened.translate(
            loaded, [reopened.get_artifact(i) for i in artifact_ids],
            batch_size=2,
        )["output"]
        actual = reused.query(
            key_columns=True, data_columns=True, metadata_columns=False,
            order_by="_position", include_position=True, form="table",
        )
        np.testing.assert_allclose(actual["probability"], expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(actual["prediction"], expected >= 0.5)
    finally:
        reopened.close()
