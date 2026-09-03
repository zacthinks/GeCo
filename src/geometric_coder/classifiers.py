"""First-class classifier specifications, fitting, and scoring for GeCo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression

from geometric_coder.geometry.base import Matrix

ClassifierAlgorithm = Literal["logistic_l2"]


class ProbabilisticEstimator(Protocol):
    """Minimal persisted-estimator interface used by current GeCo classifiers."""

    def predict(self, matrix: Matrix) -> np.ndarray: ...

    def predict_proba(self, matrix: Matrix) -> np.ndarray: ...

    def decision_function(self, matrix: Matrix) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ClassifierOutputs:
    """Parallel prediction outputs for a fitted classifier."""

    predicted_labels: np.ndarray
    probabilities: np.ndarray | None
    decision_scores: np.ndarray | None
    uncertainty: np.ndarray
    score_kind: str


@dataclass(frozen=True, slots=True)
class FittedClassifier:
    """One fitted estimator and its outputs on a requested observation matrix."""

    estimator: Any
    outputs: ClassifierOutputs


def default_hyperparameters(algorithm: str) -> dict[str, Any]:
    """Return stable defaults for one supported classifier algorithm."""
    if algorithm == "logistic_l2":
        return {
            "regularization": 1.0,
            "class_weight": "balanced",
            "max_iter": 2_000,
            "random_state": 0,
            "threshold": 0.5,
        }
    raise ValueError(f"Unsupported classifier algorithm: {algorithm!r}")


def normalize_hyperparameters(
    algorithm: str, hyperparameters: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge and validate user hyperparameters for a supported algorithm."""
    values = default_hyperparameters(algorithm)
    values.update(dict(hyperparameters or {}))
    if algorithm == "logistic_l2":
        regularization = float(values["regularization"])
        if regularization <= 0:
            raise ValueError("regularization lambda must be greater than zero")
        threshold = float(values.get("threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in the closed interval [0, 1]")
        values["regularization"] = regularization
        values["max_iter"] = int(values.get("max_iter", 2_000))
        values["random_state"] = int(values.get("random_state", 0))
        values["threshold"] = threshold
        class_weight = values.get("class_weight", "balanced")
        if class_weight not in {"balanced", None}:
            raise ValueError("class_weight must be 'balanced' or null")
        values["class_weight"] = class_weight
        return values
    raise ValueError(f"Unsupported classifier algorithm: {algorithm!r}")


def fit_classifier(
    *,
    algorithm: str,
    hyperparameters: dict[str, Any] | None,
    training_matrix: Matrix,
    labels: list[int] | np.ndarray,
    scoring_matrix: Matrix,
) -> FittedClassifier:
    """Fit one supported classifier and score the supplied observation matrix."""
    parameters = normalize_hyperparameters(algorithm, hyperparameters)
    y = np.asarray(labels, dtype=int)
    if y.ndim != 1 or training_matrix.shape[0] != y.size:
        raise ValueError("training_matrix and labels must contain parallel rows")
    if y.size < 2 or set(y.tolist()) != {0, 1}:
        raise ValueError("Classifiers require at least one positive and one negative label")

    if algorithm == "logistic_l2":
        estimator = LogisticRegression(
            C=1.0 / float(parameters["regularization"]),
            solver="liblinear",
            class_weight=parameters["class_weight"],
            max_iter=int(parameters["max_iter"]),
            random_state=int(parameters["random_state"]),
        )
        estimator.fit(training_matrix, y)
        return FittedClassifier(
            estimator=estimator,
            outputs=score_classifier(
                estimator=estimator,
                algorithm=algorithm,
                hyperparameters=parameters,
                matrix=scoring_matrix,
            ),
        )
    raise ValueError(f"Unsupported classifier algorithm: {algorithm!r}")


def score_classifier(
    *,
    estimator: Any,
    algorithm: str,
    hyperparameters: dict[str, Any] | None,
    matrix: Matrix,
) -> ClassifierOutputs:
    """Produce explicit labels, probabilities, decision scores, and uncertainty."""
    parameters = normalize_hyperparameters(algorithm, hyperparameters)
    if algorithm == "logistic_l2":
        probabilities = np.asarray(estimator.predict_proba(matrix)[:, 1], dtype=float)
        decision_scores = np.asarray(estimator.decision_function(matrix), dtype=float)
        threshold = float(parameters["threshold"])
        labels = np.where(probabilities >= threshold, 1, 0).astype(int)
        uncertainty = 1.0 - np.minimum(1.0, np.abs(probabilities - 0.5) * 2.0)
        return ClassifierOutputs(
            predicted_labels=labels,
            probabilities=probabilities,
            decision_scores=decision_scores,
            uncertainty=np.asarray(uncertainty, dtype=float),
            score_kind="probability",
        )
    raise ValueError(f"Unsupported classifier algorithm: {algorithm!r}")


def primary_scores(outputs: ClassifierOutputs) -> np.ndarray:
    """Return probability-like scores used for recommendation and geometry views."""
    if outputs.probabilities is not None:
        return np.asarray(outputs.probabilities, dtype=float)
    if outputs.decision_scores is None:
        raise ValueError("Classifier exposes neither probabilities nor decision scores.")
    values = np.asarray(outputs.decision_scores, dtype=float)
    if values.size == 0:
        return values
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    if upper <= lower:
        return np.full(values.shape, 0.5, dtype=float)
    return (values - lower) / (upper - lower)
