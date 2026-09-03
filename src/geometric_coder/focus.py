"""Code-specific active-learning helpers for GeCo Develop workspace."""

from __future__ import annotations

import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression

from geometric_coder.apply import ApplyAggregation, aggregate_probabilities
from geometric_coder.geometry.base import Matrix

FocusStrategy = Literal[
    "random",
    "classifier_uncertainty",
    "geometry_uncertainty",
    "committee_uncertainty",
    "classifier_disagreement",
    "geometry_disagreement",
    "likely_positive",
    "likely_negative",
]


@dataclass(frozen=True, slots=True)
class FittedFocusModel:
    """One fitted code classifier and its corpus-wide probabilities."""

    geometry_id: int
    geometry_name: str
    regularization: float
    estimator: LogisticRegression
    probabilities: np.ndarray


@dataclass(frozen=True, slots=True)
class FocusRecommendation:
    """One recommended unit and the score that selected it."""

    unit_id: int
    strategy: str
    score: float | None
    probabilities: dict[str, float]


def fit_focus_model(
    *,
    matrix: Matrix,
    training_positions: Sequence[int] | None = None,
    training_matrix: Matrix | None = None,
    labels: Sequence[int],
    regularization: float,
    geometry_id: int,
    geometry_name: str,
) -> FittedFocusModel:
    """Fit the single GeCo Develop classifier family: L2 logistic regression."""
    if regularization <= 0:
        raise ValueError("regularization lambda must be greater than zero")
    y = np.asarray(labels, dtype=int)
    if training_matrix is None:
        if training_positions is None:
            raise ValueError("training_positions are required without training_matrix")
        positions = np.asarray(training_positions, dtype=int)
        if positions.ndim != 1 or y.ndim != 1 or positions.size != y.size:
            raise ValueError(
                "training_positions and labels must be parallel one-dimensional arrays"
            )
        fitted_matrix = matrix[positions]
    else:
        if y.ndim != 1 or training_matrix.shape[0] != y.size:
            raise ValueError("training_matrix and labels must contain parallel rows")
        fitted_matrix = training_matrix
    if y.size < 2 or set(y.tolist()) != {0, 1}:
        raise ValueError("Develop models require at least one positive and one negative label")

    estimator = LogisticRegression(
        C=1.0 / float(regularization),
        solver="liblinear",
        class_weight="balanced",
        max_iter=2_000,
        random_state=0,
    )
    estimator.fit(fitted_matrix, y)
    probabilities = estimator.predict_proba(matrix)[:, 1]
    return FittedFocusModel(
        geometry_id=int(geometry_id),
        geometry_name=str(geometry_name),
        regularization=float(regularization),
        estimator=estimator,
        probabilities=np.asarray(probabilities, dtype=float),
    )


def recommend_focus_unit(
    *,
    strategy: FocusStrategy,
    unit_ids: Sequence[int],
    seen_unit_ids: Collection[int],
    annotated_unit_ids: Collection[int],
    probabilities_by_geometry: Mapping[str, Sequence[float]] | None = None,
    selected_geometry: str | None = None,
    committee_aggregation: ApplyAggregation = "mean",
    random_seed: int | None = None,
) -> FocusRecommendation | None:
    """Recommend an unseen, unreviewed unit using one Develop strategy."""
    ids = np.asarray(unit_ids, dtype=int)
    if ids.ndim != 1:
        raise ValueError("unit_ids must be one-dimensional")
    eligible = np.ones(ids.size, dtype=bool)
    if seen_unit_ids:
        eligible &= ~np.isin(ids, np.fromiter(seen_unit_ids, dtype=int))
    if annotated_unit_ids:
        eligible &= ~np.isin(ids, np.fromiter(annotated_unit_ids, dtype=int))
    positions = np.flatnonzero(eligible)
    if positions.size == 0:
        return None

    if strategy == "random":
        rng = random.Random(random_seed)
        position = int(rng.choice(positions.tolist()))
        return FocusRecommendation(
            unit_id=int(ids[position]),
            strategy=strategy,
            score=None,
            probabilities={},
        )

    if not probabilities_by_geometry:
        raise ValueError(f"{strategy} requires fitted geometry probabilities")
    probability_names = list(probabilities_by_geometry)
    probability_matrix = np.column_stack(
        [np.asarray(probabilities_by_geometry[name], dtype=float) for name in probability_names]
    )
    if probability_matrix.shape[0] != ids.size:
        raise ValueError("Every probability vector must contain one value per unit")

    if strategy in {"classifier_uncertainty", "geometry_uncertainty"}:
        if selected_geometry is None:
            raise ValueError("geometry_uncertainty requires selected_geometry")
        try:
            geometry_index = probability_names.index(selected_geometry)
        except ValueError:
            raise KeyError(f"Unknown selected geometry: {selected_geometry!r}") from None
        score_vector = -np.abs(probability_matrix[:, geometry_index] - 0.5)
        position = _argmax_eligible(score_vector, positions)
    elif strategy == "committee_uncertainty":
        aggregate = aggregate_probabilities(
            probabilities_by_geometry, aggregation=committee_aggregation
        )
        score_vector = -np.abs(aggregate - 0.5)
        position = _argmax_eligible(score_vector, positions)
    elif strategy in {"classifier_disagreement", "geometry_disagreement"}:
        score_vector = np.ptp(probability_matrix, axis=1)
        position = _argmax_eligible(score_vector, positions)
    elif strategy == "likely_positive":
        score_vector = (
            aggregate_probabilities(
                probabilities_by_geometry, aggregation=committee_aggregation
            )
            if probability_matrix.shape[1] > 1
            else probability_matrix[:, 0]
        )
        position = _argmax_eligible(score_vector, positions)
    elif strategy == "likely_negative":
        score_vector = (
            aggregate_probabilities(
                probabilities_by_geometry, aggregation=committee_aggregation
            )
            if probability_matrix.shape[1] > 1
            else probability_matrix[:, 0]
        )
        position = _argmin_eligible(score_vector, positions)
    else:
        raise ValueError(f"Unknown Develop strategy: {strategy!r}")

    return FocusRecommendation(
        unit_id=int(ids[position]),
        strategy=strategy,
        score=float(score_vector[position]),
        probabilities={
            name: float(probability_matrix[position, index])
            for index, name in enumerate(probability_names)
        },
    )


def _normalized_rank(values: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] ranks, with larger values receiving larger scores."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    if values.size <= 1:
        return np.ones(values.size, dtype=float)
    return ranks / float(values.size - 1)


def _argmax_eligible(scores: np.ndarray, positions: np.ndarray) -> int:
    candidate_scores = np.asarray(scores, dtype=float)[positions]
    return int(positions[int(np.nanargmax(candidate_scores))])


def _argmin_eligible(scores: np.ndarray, positions: np.ndarray) -> int:
    candidate_scores = np.asarray(scores, dtype=float)[positions]
    return int(positions[int(np.nanargmin(candidate_scores))])
