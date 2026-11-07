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
