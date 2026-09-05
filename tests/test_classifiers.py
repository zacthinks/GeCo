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
