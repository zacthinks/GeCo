from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from geometric_coder.classifiers import (
    SUPPORTED_CLASSIFIER_ALGORITHMS,
    classifier_algorithm_label,
    classifier_capabilities,
    default_hyperparameters,
    fit_classifier,
    normalize_hyperparameters,
    score_classifier,
    tuning_candidates,
)


@pytest.fixture
def toy_matrix() -> sparse.csr_matrix:
    return sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 0.1, 0.0],
                [0.1, 0.2, 0.0],
                [0.2, 0.1, 0.1],
                [1.0, 0.9, 0.8],
                [0.9, 1.1, 0.7],
                [1.2, 0.8, 1.0],
            ],
            dtype=float,
        )
    )


def test_every_supported_family_fits_and_returns_probabilities(
    toy_matrix: sparse.csr_matrix,
) -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
    for algorithm in SUPPORTED_CLASSIFIER_ALGORITHMS:
        fitted = fit_classifier(
            algorithm=algorithm,
            hyperparameters=default_hyperparameters(algorithm),
            training_matrix=toy_matrix,
            labels=labels,
            scoring_matrix=toy_matrix,
        )
        assert classifier_algorithm_label(algorithm)
        capabilities = classifier_capabilities(algorithm)
        assert capabilities.probability_output is True
        assert capabilities.supports_sparse is True
        assert fitted.outputs.predicted_labels.shape == (6,)
        assert fitted.outputs.probabilities is not None
        assert fitted.outputs.probabilities.shape == (6,)
        assert np.all((fitted.outputs.probabilities >= 0.0) & (fitted.outputs.probabilities <= 1.0))
        assert fitted.outputs.uncertainty.shape == (6,)


def test_complement_nb_rejects_negative_geometry() -> None:
    matrix = sparse.csr_matrix(np.asarray([[-1.0, 0.0], [0.0, 1.0]], dtype=float))
    with pytest.raises(ValueError, match="requires a nonnegative geometry"):
        fit_classifier(
            algorithm="complement_nb",
            hyperparameters=None,
            training_matrix=matrix,
            labels=np.asarray([0, 1]),
            scoring_matrix=matrix,
        )


def test_family_tuning_grids_are_nonempty_and_knn_respects_fold_size() -> None:
    for algorithm in SUPPORTED_CLASSIFIER_ALGORITHMS:
        candidates = tuning_candidates(
            algorithm,
            default_hyperparameters(algorithm),
            max_neighbors=4,
        )
        assert candidates
        if algorithm == "knn":
            assert max(int(row["n_neighbors"]) for row in candidates) <= 4



@pytest.mark.parametrize(
    ("algorithm", "field", "expected_count"),
    [
        ("logistic_l2", "regularization", 13),
        ("logistic_l1", "regularization", 13),
        ("linear_svm", "regularization", 13),
        ("complement_nb", "alpha", 13),
        ("logistic_elasticnet", "regularization", 25),
    ],
)
def test_continuous_parameter_search_is_bounded_and_non_grid(
    algorithm: str, field: str, expected_count: int,
) -> None:
    base = default_hyperparameters(algorithm)
    first = tuning_candidates(algorithm, base)
    second = tuning_candidates(algorithm, base)
    assert first == second
    assert len(first) == expected_count
    assert first[0] == base
    assert len({float(candidate[field]) for candidate in first[1:]}) == expected_count - 1
    assert any(
        not np.isclose(np.log10(float(candidate[field])), round(np.log10(float(candidate[field]))))
        for candidate in first[1:]
    )
    if algorithm == "logistic_elasticnet":
        assert len({float(candidate["l1_ratio"]) for candidate in first[1:]}) == 24
        assert all(0 < float(candidate["l1_ratio"]) < 1 for candidate in first[1:])


def test_finite_classifier_search_still_uses_small_discrete_spaces() -> None:
    tree = tuning_candidates("decision_tree", default_hyperparameters("decision_tree"))
    assert len(tree) == 15
    neighbors = tuning_candidates(
        "knn", default_hyperparameters("knn"), max_neighbors=5,
    )
    assert neighbors
    assert {int(row["n_neighbors"]) for row in neighbors} <= {1, 3, 5}


def test_rbf_svm_fits_nonlinear_pattern_and_monotone_probabilities() -> None:
    # Center versus ring is not linearly separable in the original two axes.
    x = np.asarray([
        [0.0, 0.0], [0.1, 0.0], [-0.1, 0.0], [0.0, 0.1],
        [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0],
    ])
    labels = np.asarray([1, 1, 1, 1, 0, 0, 0, 0])
    for matrix in (x, sparse.csr_matrix(x)):
        fit = fit_classifier(
            algorithm="rbf_svm",
            hyperparameters={"regularization": 0.1, "gamma": "scale"},
            training_matrix=matrix,
            labels=labels,
            scoring_matrix=matrix,
        )
        assert fit.outputs.decision_scores is not None
        assert fit.outputs.probabilities is not None
        margins = fit.outputs.decision_scores
        scores = fit.outputs.probabilities
        assert min(margins[:4]) > max(margins[4:])
        assert np.all(np.diff(scores[np.argsort(margins)]) >= -1e-12)
        assert np.all((0 <= scores) & (scores <= 1))
        assert fit.estimator.calibration_method in {
            "sigmoid_cv", "monotone_sigmoid_fallback"
        }


def test_svm_calibration_handles_minority_class_with_one_label() -> None:
    matrix = np.asarray([[0.0], [0.3], [1.5], [2.0]])
    for algorithm in ("linear_svm", "rbf_svm"):
        fit = fit_classifier(
            algorithm=algorithm, hyperparameters=None,
            training_matrix=matrix, labels=[0, 0, 0, 1],
            scoring_matrix=matrix,
        )
        assert fit.estimator.calibration_method == "monotone_sigmoid_fallback"
        assert np.all(np.isfinite(fit.outputs.probabilities))
        assert fit.outputs.decision_scores is not None


def test_rbf_candidate_search_is_bounded_reproducible_and_joint() -> None:
    data = sparse.csr_matrix(np.asarray([
        [0., 1.], [1., 0.], [2., 1.], [1., 2.],
    ]))
    first = tuning_candidates(
        "rbf_svm", default_hyperparameters("rbf_svm"), training_matrix=data,
    )
    second = tuning_candidates(
        "rbf_svm", default_hyperparameters("rbf_svm"), training_matrix=data,
    )
    assert first == second
    assert len(first) == 25
    assert first[0]["gamma"] == "scale"
    assert len({float(row["gamma"]) for row in first[1:]}) == 24
    assert len({float(row["regularization"]) for row in first[1:]}) == 24
    assert all(float(row["gamma"]) > 0 for row in first[1:])


def test_svm_explicit_probability_false_retains_native_margin() -> None:
    matrix = np.asarray([[0.], [1.], [2.], [3.]])
    fit = fit_classifier(
        algorithm="linear_svm",
        hyperparameters={"probability": False},
        training_matrix=matrix, labels=[0, 0, 1, 1],
        scoring_matrix=matrix,
    )
    assert fit.outputs.probabilities is None
    assert fit.outputs.decision_scores is not None


@pytest.mark.parametrize("as_sparse", [False, True])
def test_random_forest_nonlinear_deterministic_threshold(as_sparse):
    # Signed XOR quadrants with repeated observations, not linearly separable.
    rng = np.random.default_rng(12)
    centers = np.repeat([[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]], 40, axis=0)
    x = centers + rng.normal(0, 0.1, centers.shape)
    y = np.repeat([0, 1, 1, 0], 40)
    matrix = sparse.csr_matrix(x) if as_sparse else x
    params = {"threshold": 0.8}
    first = fit_classifier(algorithm="random_forest", hyperparameters=params,
                           training_matrix=matrix, labels=y, scoring_matrix=matrix)
    second = fit_classifier(algorithm="random_forest", hyperparameters=params,
                            training_matrix=matrix, labels=y, scoring_matrix=matrix)
    outputs = first.outputs
    p = outputs.probabilities
    assert np.all(np.isfinite(p))
    assert np.all((0 <= p) & (p <= 1))
    np.testing.assert_array_equal(p, second.outputs.probabilities)
    np.testing.assert_array_equal(p, first.estimator.predict_proba(matrix)[:, 1])
    np.testing.assert_array_equal(outputs.predicted_labels, p >= 0.8)
    np.testing.assert_allclose(outputs.uncertainty, 1 - 2 * np.abs(p - 0.5))
    assert np.mean((p >= 0.5) == y) > 0.95
    assert outputs.score_kind == "probability"
    assert outputs.decision_scores is None
    assert first.estimator.n_jobs == 1
    assert classifier_capabilities("random_forest").supports_dense
    assert not classifier_capabilities("random_forest").decision_score_output


@pytest.mark.parametrize("field,value", [
    ("n_estimators", 0), ("n_estimators", 1.5), ("n_estimators", True),
    ("max_depth", 0), ("max_depth", 2.5), ("min_samples_leaf", 0),
    ("min_samples_leaf", False), ("max_features", 0),
    ("max_features", 1.5), ("max_features", float("nan")),
    ("max_features", "auto"), ("max_features", True),
    ("class_weight", "uniform"), ("random_state", -1),
    ("random_state", 2**32), ("threshold", 1.1),
    ("feature_scaling", "fixed"), ("feature_scaling", "tune"),
])
def test_random_forest_invalid_configuration(field, value):
    with pytest.raises(ValueError):
        normalize_hyperparameters("random_forest", {field: value})


@pytest.mark.parametrize("features", ["sqrt", "log2", None, 2, 0.5, 1.0])
def test_random_forest_supported_feature_settings(features):
    p = normalize_hyperparameters("random_forest", {
        "max_features": features, "max_depth": None, "class_weight": None,
    })
    assert p["max_features"] == features
    assert type(p["max_features"]) is type(features)
    assert p["max_depth"] is None
    assert p["class_weight"] is None


def test_random_forest_joint_search_keeps_saved_anchor():
    base = normalize_hyperparameters("random_forest", {
        "n_estimators": 17, "max_depth": 11, "min_samples_leaf": 3,
        "max_features": 0.8, "class_weight": None, "random_state": 7,
        "threshold": 0.7,
    })
    candidates = tuning_candidates("random_forest", base)
    assert candidates == tuning_candidates("random_forest", base)
    assert candidates[0] == base
    assert len(candidates) == 9
    assert {p["max_depth"] for p in candidates[1:]} == {4, 8, 16, None}
    assert {p["min_samples_leaf"] for p in candidates[1:]} == {1, 2, 4}
    assert {p["max_features"] for p in candidates[1:]} == {"sqrt", "log2", 0.5, None}
    for p in candidates:
        for key in ("n_estimators", "class_weight", "random_state", "threshold"):
            assert p[key] == base[key]


def test_random_forest_small_minority_and_positive_class_selection():
    matrix = sparse.csr_matrix([[0.], [0.1], [0.2], [3.]])
    fitted = fit_classifier(
        algorithm="random_forest", hyperparameters=None, training_matrix=matrix,
        labels=[0, 0, 0, 1], scoring_matrix=matrix,
    )
    assert np.all(np.isfinite(fitted.outputs.probabilities))

    class Reordered:
        classes_ = np.array([1, 0])

        def predict_proba(self, matrix):
            return np.tile([0.7, 0.3], (matrix.shape[0], 1))

    outputs = score_classifier(
        estimator=Reordered(), algorithm="random_forest",
        hyperparameters={"threshold": 0.8}, matrix=matrix,
    )
    np.testing.assert_allclose(outputs.probabilities, 0.7)
    np.testing.assert_array_equal(outputs.predicted_labels, np.zeros(4))
