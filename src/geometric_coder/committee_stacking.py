"""Cross-fitted probability features for GeCo's learned classifier committees.

Each member receives its own representation and selected fitted configuration.
Temporary fold estimators are discarded; only held-out probabilities are returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from geometric_coder.classifiers import fit_classifier
from geometric_coder.geometry.base import Matrix


@dataclass(frozen=True)
class MemberTrainingData:
    """Geometry-specific labeled evidence for one committee member."""

    algorithm: str
    hyperparameters: Mapping[str, Any]
    observation_ids: tuple[int, ...]
    labels: np.ndarray
    matrix: Matrix


def out_of_fold_probabilities(
    *,
    observation_ids: Sequence[int],
    labels: np.ndarray,
    members: Sequence[MemberTrainingData],
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """Return rows-by-members held-out positive-class probabilities.

    Shared stratified folds are defined over jointly represented labeled IDs. Each
    member can have additional eligible labeled IDs outside that intersection.
    Every held-out ID is excluded from *all* of that member's training data, including
    the scaler fit and any inner SVM probability calibration. Hyperparameters are
    previously selected and are not tuned within these folds (not nested CV).
    """
    ids = tuple(int(value) for value in observation_ids)
    y = np.asarray(labels, dtype=int)
    if not members or not ids or len(set(ids)) != len(ids):
        raise ValueError("OOF stacking requires members and unique labeled observation IDs.")
    if y.ndim != 1 or y.size != len(ids) or not np.isin(y, (0, 1)).all():
        raise ValueError("OOF stacking requires parallel binary observation labels.")
    counts = np.bincount(y, minlength=2)
    if int(counts.min()) < 2 or n_splits < 2 or n_splits > int(counts.min()):
        raise ValueError(
            "OOF committee training requires at least two Present and two Absent "
            "jointly represented examples, with a feasible stratified fold count."
        )

    folds = list(
        StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state,
        ).split(np.zeros(len(ids), dtype=int), y)
    )
    result = np.full((len(ids), len(members)), np.nan, dtype=float)
    for member_index, member in enumerate(members):
        member_ids = tuple(int(value) for value in member.observation_ids)
        member_y = np.asarray(member.labels, dtype=int)
        if len(member_ids) != len(set(member_ids)) or (
            member_y.shape != (len(member_ids),)
            or member.matrix.shape[0] != len(member_ids)
        ):
            raise ValueError("Committee member training rows must have unique, aligned IDs.")
        position = {value: idx for idx, value in enumerate(member_ids)}
        if any(value not in position or member_y[position[value]] != y[idx]
               for idx, value in enumerate(ids)):
            raise ValueError("Committee members must cover all shared labels consistently.")

        for _, held_out in folds:
            held_ids = {ids[int(idx)] for idx in held_out}
            train_positions = np.asarray(
                [idx for idx, value in enumerate(member_ids) if value not in held_ids],
                dtype=int,
            )
            score_positions = np.asarray([position[ids[int(i)]] for i in held_out], dtype=int)
            train_y = member_y[train_positions]
            if np.unique(train_y).size != 2:
                raise ValueError(
                    "A committee member has insufficient training examples after "
                    "excluding the shared held-out fold."
                )
            fitted = fit_classifier(
                algorithm=member.algorithm,
                hyperparameters=dict(member.hyperparameters),
                training_matrix=member.matrix[train_positions],
                labels=train_y,
                scoring_matrix=member.matrix[score_positions],
            )
            probabilities = fitted.outputs.probabilities
            if probabilities is None:
                raise ValueError(
                    f"Committee member {member.algorithm!r} must provide probabilities; "
                    "decision margins cannot be used as stacking probabilities."
                )
            probabilities = np.asarray(probabilities, dtype=float)
            if probabilities.shape != (len(held_out),) or (
                not np.isfinite(probabilities).all()
                or np.any((probabilities < 0.0) | (probabilities > 1.0))
            ):
                raise ValueError("Committee member produced invalid held-out probabilities.")
            result[held_out, member_index] = probabilities

    if not np.isfinite(result).all():
        raise RuntimeError("OOF stacking did not score every eligible observation and member.")
    return result
