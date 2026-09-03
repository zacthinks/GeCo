"""Corpus-wide proposal generation and review helpers for GeCo Apply mode."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

ApplyAggregation = Literal[
    "mean",
    "median",
    "minimum",
    "maximum",
    "harmonic_mean",
    "geometric_mean",
]
ApplyDecision = Literal[
    "pending",
    "accept",
    "positive",
    "negative",
    "unsure",
    "unreviewed",
]


@dataclass(frozen=True, slots=True)
class ApplyProposal:
    """One aggregated machine proposal for an atomic unit."""

    unit_id: int
    probability: float
    proposed_label: Literal["positive", "negative"]


def aggregate_probabilities(
    probabilities_by_geometry: Mapping[str, Sequence[float]],
    *,
    aggregation: ApplyAggregation = "mean",
) -> np.ndarray:
    """Aggregate parallel corpus-wide probability vectors.

    Apply mode intentionally keeps the aggregation rule simple and inspectable.
    Model-specific probabilities remain persisted separately, so the committee
    score never erases its component predictions.
    """
    if not probabilities_by_geometry:
        raise ValueError("At least one geometry probability vector is required.")
    names = list(probabilities_by_geometry)
    matrix = np.column_stack(
        [np.asarray(probabilities_by_geometry[name], dtype=float) for name in names]
    )
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Probability vectors must contain at least one unit.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Probability vectors may not contain NaN or infinite values.")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("Probabilities must lie in the closed interval [0, 1].")
    if aggregation == "mean":
        return np.mean(matrix, axis=1)
    if aggregation == "median":
        return np.median(matrix, axis=1)
    if aggregation == "minimum":
        return np.min(matrix, axis=1)
    if aggregation == "maximum":
        return np.max(matrix, axis=1)
    if aggregation == "harmonic_mean":
        # The exact harmonic mean is zero when any member assigns zero probability.
        positive = np.all(matrix > 0.0, axis=1)
        result = np.zeros(matrix.shape[0], dtype=float)
        result[positive] = matrix.shape[1] / np.sum(1.0 / matrix[positive], axis=1)
        return result
    if aggregation == "geometric_mean":
        # Work in log space while preserving exact zeros.
        positive = np.all(matrix > 0.0, axis=1)
        result = np.zeros(matrix.shape[0], dtype=float)
        result[positive] = np.exp(np.mean(np.log(matrix[positive]), axis=1))
        return result
    raise ValueError(f"Unknown Apply aggregation: {aggregation!r}")


def build_apply_proposals(
    *,
    unit_ids: Sequence[int],
    probabilities: Sequence[float],
    eligible_unit_ids: Sequence[int],
    threshold: float = 0.5,
) -> list[ApplyProposal]:
    """Create thresholded proposals for an explicit eligibility set."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in the closed interval [0, 1]")
    ids = np.asarray(unit_ids, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if ids.ndim != 1 or scores.ndim != 1 or ids.size != scores.size:
        raise ValueError("unit_ids and probabilities must be parallel one-dimensional arrays")
    eligible = {int(unit_id) for unit_id in eligible_unit_ids}
    return [
        ApplyProposal(
            unit_id=int(unit_id),
            probability=float(probability),
            proposed_label=("positive" if float(probability) >= threshold else "negative"),
        )
        for unit_id, probability in zip(ids, scores, strict=True)
        if int(unit_id) in eligible
    ]
