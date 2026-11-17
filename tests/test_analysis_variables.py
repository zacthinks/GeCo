"""Phase 1 typed analytical-variable contract and legacy analysis isolation."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from geometric_coder import CountGeometry, GeometricCoder
from geometric_coder.analysis_variables import (
    MAX_RESOLUTION_OBSERVATIONS, resolve_analytical_variables, validate_variable,
)


class SimpleScaler:
    """Tiny saved test estimator: returned scores depend only on selected rows."""

    def __init__(self, **_kwargs):
        self.success_ = True
        self.message_ = "converged"
        self.posterior_ = None

    def fit(self, X, pairs, outcomes, *, reference_X=None):
        return self

    def predict_scores(self, X):
        column = X[:, 0]
        return np.asarray(column.toarray() if hasattr(column, "toarray") else column).ravel()


@pytest.fixture
def sample(tmp_path):
    project = GeometricCoder.create(
        project_dir=tmp_path / "typed.geco",
        data=pd.DataFrame({
            "key": [101, 103, 107, 109, 113, 127],
            "text": [
                "excellent helpful", "great excellent", "useful clear",
                "bad confusing", "terrible unclear", "weak bad",
            ],
            "group": ["A", "A", "B", "B", "C", "C"],
            "score": [1, 2, 3, 4, 5, 6],
            "literal": ["one", "two", "three", "four", "five", "six"],
        }),
        keys=["key"], text="text", metadata=["group", "score", "literal"],
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    ids = [row["observation_id"] for row in project.database.list_units()]
    code = project.create_code("Sentiment")
    project.annotate(ids[0], code, "positive")
    project.annotate(ids[3], code, "negative")
    project.annotate(ids[1], code, "unsure")
    project.ensure_default_classifier_specs(code_id=code)
    predictor = int(project.classifier_specs(code_id=code)[0]["classifier_spec_id"])
    project.train_classifier(code_id=code, classifier_spec_id=predictor)

    matrix = np.asarray([
        [6.0, 1.0], [5.0, 0.0], [4.0, 1.0],
        [3.0, 0.0], [2.0, 1.0], [1.0, 0.0],
    ])
    path, sparse = project.artifacts.save_geometry_matrix("scalar_fixture", matrix)
    geometry = project.database.register_geometry(
        name="scalar_fixture", class_name="SyntheticGeometry", modality="text",
        public=True, config={}, dependency_names=[], matrix_path=path,
        matrix_sparse=sparse, state_path=None, supports_query=False,
        supports_text_transform=False,
    )
    representation = project.database.representation_for_legacy("geometry", geometry)
    scale = project.create_scale("Quality")
    project.record_scale_comparison(scale, ids[0], ids[1], "first_higher")
    project.record_scale_comparison(scale, ids[2], ids[3], "second_higher")
    project.configure_scale_model(scale, int(representation["representation_id"]))
    assert project.train_scale(scale, estimator_factory=SimpleScaler)["status"] == "current"
    return project, code, predictor, scale, ids


def _code(code, source="labels", value="binary", predictor=None):
    reference = {"type": "code", "code_id": code, "source": source, "value": value}
    if predictor is not None:
        reference["predictor_id"] = predictor
    return reference


def test_mixed_columns_retain_requested_atomic_identity_and_duplicate_order(sample):
    project, code, predictor, scale, ids = sample
    requested = [ids[5], ids[0], ids[2], ids[0], ids[3]]
    references = [
        _code(code, "predictions", "probability", predictor),
        {"type": "scale", "scale_id": scale},
        {"type": "metadata", "field": "score", "value": "numeric"},
        {"type": "metadata", "field": "group"},
    ]
    result = resolve_analytical_variables(project, references, requested)
    assert result["observation_ids"] == requested
    assert result["row_count"] == 5
    assert len(result["columns"]) == 4
    probability, position, number, group = result["columns"]
    assert probability["kind"] == "numeric"
    assert position["kind"] == "numeric"
    assert group["kind"] == "categorical"
    assert position["values"] == [1.0, 6.0, 4.0, 6.0, 3.0]
    assert position["provenance"]["scaler_fit_id"] is not None
    assert position["provenance"]["fit_status"] == "current"
    assert number["values"] == [6.0, 1.0, 3.0, 1.0, 4.0]
    assert group["values"] == ["C", "A", "B", "A", "B"]
    assert probability["values"][1] == probability["values"][3]
    assert probability["coverage"] == {"observed": 5, "missing": 0, "by_reason": {}}


def test_code_predictions_binary_probability_and_hybrid_unsure(sample):
    project, code, predictor, _scale, ids = sample
    references = [
        _code(code),
        _code(code, "predictions", "binary", predictor),
        _code(code, "predictions", "probability", predictor),
        _code(code, "hybrid", "probability", predictor),
        _code(code, "hybrid", "binary", predictor),
    ]
    labels, binary, probability, hybrid, hybrid_binary = (
        resolve_analytical_variables(project, references, ids)["columns"]
    )
    assert labels["values"] == [1.0, None, None, 0.0, None, None]
    assert labels["missing_reasons"][:3] == [None, "unsure", "unreviewed"]
    assert set(binary["values"]).issubset({0.0, 1.0})
    assert all(0.0 <= value <= 1.0 for value in probability["values"])
    assert probability["values"] != binary["values"]
    assert hybrid["values"][0] == 1.0 and hybrid["values"][3] == 0.0
    assert hybrid["values"][1] is None
    assert hybrid["missing_reasons"][1] == "unsure"
    assert hybrid["values"][2] == probability["values"][2]
    assert hybrid_binary["values"][2] == binary["values"][2]
    assert hybrid["coverage"]["missing"] == 1
    assert hybrid["coverage"]["by_reason"] == {"unsure": 1}
    assert hybrid["provenance"]["classifier_spec_id"] == predictor


def test_untrained_code_predictor_is_missing_but_human_hybrid_remains(sample):
    project, code, _predictor, _scale, ids = sample
    # A second configured classifier exists but has never been trained.
    specs = project.classifier_specs(code_id=code)
    if len(specs) < 2:
        pytest.skip("No second default classifier specification is available.")
    other = next(int(s["classifier_spec_id"]) for s in specs[1:])
    columns = resolve_analytical_variables(project, [
        _code(code, "predictions", "binary", other),
        _code(code, "hybrid", "binary", other),
    ], ids)["columns"]
    assert columns[0]["values"] == [None] * len(ids)
    assert columns[0]["coverage"]["by_reason"] == {"not_trained": len(ids)}
    assert columns[1]["values"][0] == 1.0
    assert columns[1]["values"][3] == 0.0
    assert columns[1]["missing_reasons"][1] == "unsure"
    assert columns[1]["missing_reasons"][2] == "not_trained"


def test_scale_absent_fit_freshness_version_and_eligibility(sample):
    project, code, _predictor, scale, ids = sample
    missing_scale = project.create_scale("Not yet fitted")
    absent = resolve_analytical_variables(
        project, [{"type": "scale", "scale_id": missing_scale}], ids
    )["columns"][0]
    assert absent["values"] == [None] * len(ids)
    assert absent["provenance"]["scaler_fit_id"] is None
    assert absent["provenance"]["fit_status"] == "not_configured"
    assert absent["coverage"]["by_reason"] == {"not_configured_fit": len(ids)}
    prior = resolve_analytical_variables(
        project, [{"type": "scale", "scale_id": scale}], ids
    )["columns"][0]
    project.update_scale(scale, "Quality revised", "new definition")
    blocked = resolve_analytical_variables(
        project, [{"type": "scale", "scale_id": scale}], ids
    )["columns"][0]
    assert blocked["provenance"]["fit_status"] == "stale"
    assert blocked["provenance"]["scale_version_id"] != prior["provenance"]["scale_version_id"]
    assert blocked["coverage"]["by_reason"] == {"stale_fit": len(ids)}
    allowed = resolve_analytical_variables(
        project, [{"type": "scale", "scale_id": scale}], ids, allow_stale_scales=True
    )["columns"][0]
    assert allowed["values"] == prior["values"]
    assert allowed["provenance"]["scaler_fit_id"] == prior["provenance"]["scaler_fit_id"]
    assert any("stale" in warning for warning in allowed["warnings"])

    project.update_scale_eligibility(scale, {
        "kind": "filtered",
        "code_filters": [{"code_id": code, "source": "human", "value": "positive"}],
    })
    selected = resolve_analytical_variables(
        project, [{"type": "scale", "scale_id": scale}], ids,
        allow_stale_scales=True,
    )["columns"][0]
    assert selected["values"][0] == 6.0
    assert selected["missing_reasons"][1:] == ["ineligible"] * 5


def test_numeric_metadata_strict_explicit_string_parse_and_bad_json_values(sample):
    project, _code_id, _predictor, _scale, ids = sample
    changes = {
        ids[1]: "7.5", ids[2]: "malformed", ids[3]: None,
        ids[4]: math.inf, ids[5]: False,
    }
    with project.database.connect() as connection:
        for observation_id, value in changes.items():
            record = connection.execute(
                "SELECT metadata_json FROM units WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
            metadata = json.loads(record["metadata_json"])
            metadata["score"] = value
            connection.execute(
                "UPDATE units SET metadata_json = ? WHERE observation_id = ?",
                (json.dumps(metadata), observation_id),
            )
    raw, parsed, categorical = resolve_analytical_variables(project, [
        {"type": "metadata", "field": "score", "value": "numeric"},
        {"type": "metadata", "field": "score", "value": "numeric", "numeric_strings": True},
        {"type": "metadata", "field": "score"},
    ], ids)["columns"]
    assert raw["values"] == [1.0, None, None, None, None, None]
    assert parsed["values"] == [1.0, 7.5, None, None, None, None]
    assert parsed["missing_reasons"][2:] == [
        "invalid_numeric", "missing_metadata", "invalid_numeric", "invalid_numeric"
    ]
    assert categorical["values"][:3] == [1, "7.5", "malformed"]
    assert categorical["missing_reasons"][4] == "invalid_metadata"
    assert categorical["values"][5] is False


def test_validation_unknown_variables_and_non_atomic_ids(sample):
    project, code, predictor, scale, ids = sample
    for reference in [
        {"type": "code", "code_id": 999999},
        {"type": "scale", "scale_id": 999999},
        {"type": "metadata", "field": "key"},
        _code(code, "predictions", "binary", 999999),
        _code(code, "labels", "probability"),
        {"type": "scale", "scale_id": scale, "value": "probability"},
        {"type": "metadata", "field": "literal", "value": "numeric", "numeric_strings": "yes"},
    ]:
        with pytest.raises(ValueError):
            validate_variable(project, reference)
    with pytest.raises(ValueError, match="Unknown or non-atomic"):
        resolve_analytical_variables(project, [_code(code)], [ids[0], 999999999])
    with pytest.raises(ValueError, match="at most"):
        resolve_analytical_variables(
            project, [_code(code)], [ids[0]] * (MAX_RESOLUTION_OBSERVATIONS + 1)
        )


def test_bounded_reads_and_no_training_or_evidence_mutations(sample, monkeypatch):
    project, code, predictor, scale, ids = sample
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Resolver cannot train models or fetch the full corpus")
    monkeypatch.setattr(project.database, "list_units", forbidden)
    monkeypatch.setattr(project, "train_classifier", forbidden)
    monkeypatch.setattr(project, "train_scale", forbidden)
    before = {}
    with project.database.connect() as connection:
        for table in ("annotation_events", "scale_comparison_events", "scaler_fits"):
            before[table] = connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
    result = resolve_analytical_variables(project, [
        _code(code, "hybrid", "probability", predictor),
        {"type": "scale", "scale_id": scale},
        {"type": "metadata", "field": "group"},
    ], ids[:3])
    assert result["row_count"] == 3
    with project.database.connect() as connection:
        for table, count in before.items():
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == count


def test_existing_v1_saved_analysis_still_live_after_typed_resolution(sample):
    project, code, predictor, scale, ids = sample
    legacy = {"kind": "bar", "variables": [_code(code)]}
    analysis_id = project.save_analysis("Legacy", legacy)
    original = project.run_analysis(analysis_id)
    assert project.analysis(analysis_id)["spec"]["schema_version"] == 1
    resolve_analytical_variables(project, [
        _code(code, "hybrid", "probability", predictor),
        {"type": "scale", "scale_id": scale},
    ], ids)
    assert project.run_analysis(analysis_id) == original
    project.annotate(ids[2], code, "positive")
    assert project.run_analysis(analysis_id)["counts"][0][0] == original["counts"][0][0] + 1
    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.analysis(analysis_id)["spec"]["schema_version"] == 1
    assert reopened.run_analysis(analysis_id)["counts"] == project.run_analysis(analysis_id)["counts"]


def test_linked_geometry_scores_are_key_aligned_and_bounded(tmp_path, monkeypatch):
    class KeyedProvider:
        def geometry_matrix(self, external_ref, keys):
            assert external_ref == {"geometry": "keyed"}
            return np.asarray([[float(key["key"]), 1.0] for key in keys])

    project = GeometricCoder.create_external(
        project_dir=tmp_path / "linked.geco",
        data=pd.DataFrame({"key": [501, 101, 401, 201], "text": ["a", "b", "c", "d"]}),
        keys=["key"], text="text", metadata=[],
        external_provider=KeyedProvider(),
    )
    geometry = project.database.register_geometry(
        name="keyed", class_name="ExternalGeometry", modality="text",
        public=True, config={}, dependency_names=[], matrix_path="",
        matrix_sparse=False, state_path=None, supports_query=False,
        supports_text_transform=False, storage_kind="external",
        external_ref={"geometry": "keyed"},
    )
    representation = project.database.representation_for_legacy("geometry", geometry)
    scale = project.create_scale("Linked position")
    ids = [row["observation_id"] for row in project.database.list_units()]
    project.record_scale_comparison(scale, ids[0], ids[1], "first_higher")
    project.record_scale_comparison(scale, ids[2], ids[3], "second_higher")
    project.configure_scale_model(scale, int(representation["representation_id"]))
    project.train_scale(scale, estimator_factory=SimpleScaler)
    monkeypatch.setattr(project.database, "list_units", lambda: 1 / 0)
    requested = [ids[3], ids[0], ids[3]]
    result = resolve_analytical_variables(
        project, [{"type": "scale", "scale_id": scale}], requested
    )
    assert result["columns"][0]["values"] == [201.0, 501.0, 201.0]
