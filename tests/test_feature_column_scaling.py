"""Classifier feature scaling and portable exported fitted-estimator regressions."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from geometric_coder.classifiers import fit_classifier, normalize_hyperparameters
from geometric_coder.preprocessing import (
    FeatureScaleNormalizer,
    portable_feature_scale_estimator,
)


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("kind", ["dense", "csr", "csc"])
@pytest.mark.parametrize("norm", ["rms", "l2"])
def test_column_scale_matches_formula_and_sparse(gamma, kind, norm):
    x = np.array([[3., 0., 1e-15], [0., 4., 0.], [0., 0., 0.]])
    source = x if kind == "dense" else getattr(sparse, kind + "_matrix")(x)
    transformer = FeatureScaleNormalizer(gamma=gamma, norm=norm).fit(source)
    result = transformer.transform(source)
    actual = result if isinstance(result, np.ndarray) else result.toarray()
    rms = np.sqrt(np.mean(x**2, axis=0))
    scales = rms if norm == "rms" else rms * np.sqrt(x.shape[0])
    safe = np.where(scales <= 1e-12, 1.0, scales)
    np.testing.assert_allclose(actual, x / safe**gamma, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(transformer.column_scales_, scales)
    np.testing.assert_allclose(clone(transformer).fit(source).transform(source).toarray()
                               if sparse.issparse(source) else clone(transformer).fit(source).transform(source),
                               actual)
    assert sparse.issparse(result) == sparse.issparse(source)


@pytest.mark.parametrize("kind", ["csr", "csc"])
def test_noncanonical_duplicates_and_sparse_width(kind):
    x = np.array([[1., 0., 2.], [0., 3., 0.]])
    matrix = getattr(sparse, kind + "_matrix")(x)
    # Explicit duplicate coordinates, including cancellation, are mathematically
    # equivalent to a canonical sparse matrix.
    if kind == "csr":
        matrix = sparse.csr_matrix(
            ([100., -99., 2., 3.], [0, 0, 2, 1], [0, 3, 4]),
            shape=(2, 3),
        )
    else:
        matrix = sparse.csc_matrix(
            ([100., -99., 3., 2.], [0, 0, 1, 0], [0, 2, 3, 4]),
            shape=(2, 3),
        )
    assert not matrix.has_canonical_format
    before = matrix.data.copy()
    normalizer = FeatureScaleNormalizer().fit(matrix)
    np.testing.assert_allclose(
        normalizer.column_scales_,
        FeatureScaleNormalizer().fit(x).column_scales_,
    )
    np.testing.assert_array_equal(matrix.data, before)
    wide = sparse.csr_matrix(([1., 2.], ([0, 1], [5, 99999])), shape=(2, 100000))
    out = FeatureScaleNormalizer().fit_transform(wide)
    assert sparse.issparse(out) and out.shape == wide.shape and out.nnz == wide.nnz


def test_neutral_fitted_state_matches_teal_when_available():
    teal = pytest.importorskip("text_analysis_lab.preprocessing")
    x = np.array([[4., 0.], [0., 3.], [2., 0.]])
    for gamma in (0., 0.5, 1.):
        for norm in ("rms", "l2"):
            ge = FeatureScaleNormalizer(gamma=gamma, norm=norm).fit(x)
            te = teal.FeatureScaleNormalizer(gamma=gamma, norm=norm).fit(x)
            assert ge.to_fitted_state() == te.to_fitted_state()
            np.testing.assert_array_equal(ge.transform(x), te.transform(x))
            rebuilt = teal.FeatureScaleNormalizer.from_fitted_state(ge.to_fitted_state())
            np.testing.assert_array_equal(rebuilt.transform(x), te.transform(x))


@pytest.mark.parametrize("algorithm", ["logistic_l1", "logistic_l2", "logistic_elasticnet", "linear_svm"])
def test_scaled_models_preserve_predictions_after_portable_export(algorithm):
    x = sparse.csr_matrix(np.array([
        [4., 0.], [3., 1.], [5., 0.],
        [0., 4.], [1., 3.], [0., 5.],
    ]))
    y = [1, 1, 1, 0, 0, 0]
    for gamma in (0., 0.5, 1.):
        hp = {"feature_scaling": "tune", "feature_scale_gamma": gamma}
        fitted = fit_classifier(
            algorithm=algorithm, hyperparameters=hp,
            training_matrix=x, labels=y, scoring_matrix=x,
        )
        assert isinstance(fitted.estimator, Pipeline) if algorithm.startswith("logistic") else True
        portable = portable_feature_scale_estimator(fitted.estimator)
        assert b"geometric_coder.preprocessing" not in pickle.dumps(portable)
        np.testing.assert_allclose(
            portable.predict_proba(x), fitted.estimator.predict_proba(x), rtol=1e-12, atol=1e-12,
        )
        if hasattr(fitted.estimator, "decision_function"):
            np.testing.assert_allclose(
                portable.decision_function(x), fitted.estimator.decision_function(x),
                rtol=1e-12, atol=1e-12,
            )


def test_old_defaults_and_family_guard():
    assert "feature_scaling" not in normalize_hyperparameters("logistic_l2", None)
    assert normalize_hyperparameters("logistic_l2", {"feature_scaling": "fixed"})["feature_scaling"] == "fixed"
    with pytest.raises(ValueError, match="not supported"):
        normalize_hyperparameters("rbf_svm", {"feature_scaling": "tune"})
    with pytest.raises(ValueError, match="not supported"):
        normalize_hyperparameters("knn", {"feature_scaling": "fixed"})


def test_fold_isolated_scaler_fit(monkeypatch):
    import geometric_coder.preprocessing as preprocessing

    seen = []
    original = preprocessing.FeatureScaleNormalizer.fit

    def recording(self, x, y=None):
        seen.append(int(x.shape[0]))
        return original(self, x, y)

    monkeypatch.setattr(preprocessing.FeatureScaleNormalizer, "fit", recording)
    x = np.array([
        [1., 0.], [2., 0.], [3., 0.],
        [0., 1.], [0., 2.], [0., 3.],
    ])
    for indices in ([0, 1, 3, 4], [0, 2, 3, 5]):
        fit_classifier(
            algorithm="logistic_l2",
            hyperparameters={"feature_scaling": "fixed"},
            training_matrix=x[indices],
            labels=[1, 1, 0, 0],
            scoring_matrix=x,
        )
    assert seen == [4, 4]


def test_geco_joint_cv_persistence_staleness_and_retrain(tmp_path):
    import pandas as pd

    from geometric_coder import CountGeometry, GeometricCoder

    data = pd.DataFrame({
        "row_id": list(range(8)),
        "text": [
            "excellent clear helpful", "excellent useful strong", "helpful clear useful",
            "bad confusing weak", "terrible unclear weak", "bad difficult confusing",
            "excellent clear", "bad weak",
        ],
    })
    project = GeometricCoder.create(
        project_dir=tmp_path / "scaling.geco",
        data=data, keys=["row_id"], text="text", metadata=[],
        geometries={"lexical": CountGeometry(lemmatize=False, weighting="tfidf")},
    )
    code = project.create_code("quality")
    geometry_id = int(project.geometries()[0]["geometry_id"])
    spec_id = project.create_classifier_spec(
        code_id=code, name="scaled", geometry_id=geometry_id,
        algorithm="logistic_l2", hyperparameters={"feature_scaling": "tune"},
    )
    rows = project.database.list_observations()
    for position in (0, 1, 2):
        project.annotate(int(rows[position]["observation_id"]), code, "positive")
    for position in (3, 4, 5):
        project.annotate(int(rows[position]["observation_id"]), code, "negative")

    # Per-classifier Tune must request CV on a genuine fit without the global
    # tuning checkbox. All old regularization candidates are retained.
    fit = project.train_classifier(
        code_id=code, classifier_spec_id=spec_id, tune=False, folds=2,
    )
    plan = fit["training_selection"]
    assert plan["tuned"]
    assert len(plan["results"]) == 39
    assert {r["hyperparameters"]["feature_scale_gamma"] for r in plan["results"]} == {
        0.0, 0.5, 1.0,
    }
    saved = project.artifacts.load_model(project.project_dir, fit["artifact_path"])
    assert isinstance(saved, Pipeline)
    assert saved.named_steps["scale"].n_samples_seen_ == 6
    assert saved.named_steps["scale"].gamma in (0.0, 0.5, 1.0)
    assert project.classifier_status(code_id=code)["classifiers"][0]["status"] == "current"

    reopened = GeometricCoder.open(project.project_dir)
    assert reopened.classifier_status(code_id=code)["classifiers"][0]["status"] == "current"
    reopened.update_classifier_spec(
        spec_id, name="scaled",
        hyperparameters={"regularization": 1.0, "feature_scaling": "fixed"},
    )
    assert reopened.classifier_status(code_id=code)["classifiers"][0]["status"] == "stale"
    fixed = reopened.train_classifier(code_id=code, classifier_spec_id=spec_id)
    assert fixed["classifier_fit_id"] != fit["classifier_fit_id"]
    model = reopened.artifacts.load_model(reopened.project_dir, fixed["artifact_path"])
    assert model.named_steps["scale"].gamma == 1.0
    assert reopened.classifier_status(code_id=code)["classifiers"][0]["status"] == "current"
