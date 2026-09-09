"""Neutral frozen predictor exports for GeCo -> external-system handoff.

The objects in this module deliberately know nothing about TeAL.  GeCo freezes the
prediction procedure it fitted; an integration layer may then translate this neutral
representation into its own durable execution abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy import sparse

from geometric_coder.apply import aggregate_probabilities
from geometric_coder.classifiers import score_classifier
from geometric_coder.geometry.base import Matrix

PredictorKind = Literal["classifier", "committee"]


@dataclass(frozen=True, slots=True)
class GeCoPredictorRef:
    """Stable reference to one active GeCo object that can become a predictor."""

    kind: PredictorKind
    id: int
    code_id: int
    name: str


@dataclass(frozen=True, slots=True)
class PredictorSourceSpec:
    """One ordered unique geometry required by a frozen predictor."""

    source_index: int
    geometry_id: int
    geometry_name: str
    n_features: int | None
    storage_kind: str
    external_ref: Any | None = None


@dataclass(frozen=True, slots=True)
class FrozenPredictorMember:
    """One fitted classifier member plus its routing and scoring semantics."""

    classifier_spec_id: int
    classifier_fit_id: int
    classifier_name: str
    fit_classifier_name: str
    source_index: int
    algorithm: str
    hyperparameters: Mapping[str, Any]
    score_kind: str
    positive_class: int
    estimator: Any


@dataclass(frozen=True, slots=True)
class PredictorBatchOutput:
    """Canonical batch output of a frozen GeCo predictor."""

    prediction: np.ndarray
    probability: np.ndarray | None


@dataclass(frozen=True, slots=True)
class FrozenPredictorExport:
    """A fitted GeCo prediction procedure independent of any external runtime.

    ``inputs`` are always an ordered sequence whose positions correspond exactly to
    ``sources``.  Sparse and dense matrices remain separate; members route directly to
    the source they consume.  Repeated member use of one geometry therefore does not
    duplicate the input matrix.
    """

    format_version: int
    ref: GeCoPredictorRef
    code_name: str
    sources: tuple[PredictorSourceSpec, ...]
    members: tuple[FrozenPredictorMember, ...]
    aggregation: str | None
    stacker: Any | None
    stacker_fit_id: int | None
    positive_class: int
    threshold: float
    output_fields: tuple[str, ...]
    stale_at_export: bool
    provenance: Mapping[str, Any]

    def _validated_inputs(self, inputs: Sequence[Matrix]) -> tuple[Matrix, ...]:
        matrices = tuple(inputs)
        if len(matrices) != len(self.sources):
            raise ValueError(
                f"Predictor requires {len(self.sources)} source matrix/matrices, "
                f"but received {len(matrices)}."
            )

        row_count: int | None = None
        for source, matrix in zip(self.sources, matrices, strict=True):
            if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
                raise ValueError(
                    f"Source {source.source_index} ({source.geometry_name!r}) must be a 2D matrix."
                )
            rows = int(matrix.shape[0])
            columns = int(matrix.shape[1])
            if row_count is None:
                row_count = rows
            elif rows != row_count:
                raise ValueError("All predictor source matrices must contain the same number of rows.")
            if source.n_features is not None and columns != int(source.n_features):
                raise ValueError(
                    f"Source {source.source_index} ({source.geometry_name!r}) requires "
                    f"{source.n_features} features, but received {columns}."
                )
        return matrices

    @staticmethod
    def _member_outputs(member: FrozenPredictorMember, matrix: Matrix):
        return score_classifier(
            estimator=member.estimator,
            algorithm=str(member.algorithm),
            hyperparameters=dict(member.hyperparameters),
            matrix=matrix,
        )

    @classmethod
    def _member_primary_score(
        cls, member: FrozenPredictorMember, matrix: Matrix
    ) -> np.ndarray:
        outputs = cls._member_outputs(member, matrix)
        if outputs.probabilities is not None:
            return np.asarray(outputs.probabilities, dtype=float)
        if outputs.decision_scores is not None:
            values = np.asarray(outputs.decision_scores, dtype=float)
            if values.ndim != 1:
                raise ValueError("GeCo predictor member decision scores must be one-dimensional.")
            return values
        raise ValueError(
            f"Classifier {member.classifier_name!r} exposes neither probabilities nor decision scores."
        )

    def predict_batch(self, inputs: Sequence[Matrix]) -> PredictorBatchOutput:
        """Apply the frozen predictor to one row-aligned batch of source matrices."""
        matrices = self._validated_inputs(inputs)
        if not self.members:
            raise RuntimeError("Frozen predictor contains no fitted members.")

        # The one-member case preserves the fitted classifier's own prediction
        # semantics, including classifiers that expose labels/decision scores but no
        # calibrated probabilities.
        if self.ref.kind == "classifier":
            if len(self.members) != 1:
                raise RuntimeError("A classifier export must contain exactly one fitted member.")
            member = self.members[0]
            outputs = self._member_outputs(member, matrices[member.source_index])
            probabilities = (
                np.asarray(outputs.probabilities, dtype=float)
                if outputs.probabilities is not None
                else None
            )
            return PredictorBatchOutput(
                prediction=np.asarray(outputs.predicted_labels, dtype=int),
                probability=probabilities,
            )

        member_scores = [
            self._member_primary_score(member, matrices[member.source_index])
            for member in self.members
        ]
        if not member_scores:
            raise RuntimeError("A committee export must contain fitted members.")

        if self.aggregation == "logistic_stack":
            if self.stacker is None:
                raise RuntimeError("Learned committee export is missing its fitted stacker.")
            stack_matrix = np.column_stack(member_scores)
            stack_outputs = score_classifier(
                estimator=self.stacker,
                algorithm="logistic_l2",
                hyperparameters={"regularization": 1.0, "threshold": self.threshold},
                matrix=stack_matrix,
            )
            if stack_outputs.probabilities is None:
                raise RuntimeError("Learned committee stacker did not return probabilities.")
            probabilities = np.asarray(stack_outputs.probabilities, dtype=float)
        else:
            if self.aggregation is None:
                raise RuntimeError("Committee export is missing an aggregation rule.")
            probabilities = aggregate_probabilities(
                {
                    f"member_{index}": values
                    for index, values in enumerate(member_scores)
                },
                aggregation=self.aggregation,  # type: ignore[arg-type]
            )

        predictions = np.where(probabilities >= float(self.threshold), 1, 0).astype(int)
        return PredictorBatchOutput(
            prediction=predictions,
            probability=np.asarray(probabilities, dtype=float),
        )

    def predict(self, inputs: Sequence[Matrix]) -> np.ndarray:
        """Return canonical fitted predictions for one batch."""
        return self.predict_batch(inputs).prediction

    def predict_proba(self, inputs: Sequence[Matrix]) -> np.ndarray:
        """Return positive-class probabilities where the frozen predictor defines them."""
        output = self.predict_batch(inputs)
        if output.probability is None:
            raise ValueError("This frozen predictor does not define probability output.")
        return output.probability

    def manifest(self) -> dict[str, Any]:
        """Return JSON-round-trippable structural/provenance state without fitted assets."""
        return {
            "format_version": int(self.format_version),
            "ref": {
                "kind": str(self.ref.kind),
                "id": int(self.ref.id),
                "code_id": int(self.ref.code_id),
                "name": str(self.ref.name),
            },
            "code_name": str(self.code_name),
            "sources": [
                {
                    "source_index": int(source.source_index),
                    "geometry_id": int(source.geometry_id),
                    "geometry_name": str(source.geometry_name),
                    "n_features": (
                        int(source.n_features) if source.n_features is not None else None
                    ),
                    "storage_kind": str(source.storage_kind),
                    "external_ref": source.external_ref,
                }
                for source in self.sources
            ],
            "members": [
                {
                    "classifier_spec_id": int(member.classifier_spec_id),
                    "classifier_fit_id": int(member.classifier_fit_id),
                    "classifier_name": str(member.classifier_name),
                    "fit_classifier_name": str(member.fit_classifier_name),
                    "source_index": int(member.source_index),
                    "algorithm": str(member.algorithm),
                    "hyperparameters": dict(member.hyperparameters),
                    "score_kind": str(member.score_kind),
                    "positive_class": int(member.positive_class),
                }
                for member in self.members
            ],
            "aggregation": self.aggregation,
            "stacker_fit_id": self.stacker_fit_id,
            "positive_class": int(self.positive_class),
            "threshold": float(self.threshold),
            "output_fields": list(self.output_fields),
            "stale_at_export": bool(self.stale_at_export),
            "provenance": dict(self.provenance),
        }


def estimator_feature_width(estimator: Any) -> int | None:
    """Return a fitted estimator's expected feature width when it exposes one."""
    width = getattr(estimator, "n_features_in_", None)
    if width is None:
        return None
    try:
        return int(width)
    except (TypeError, ValueError):
        return None


def matrix_kind(matrix: Matrix) -> str:
    """Small debugging helper retained for integration tests."""
    return "sparse" if sparse.issparse(matrix) else "dense"
