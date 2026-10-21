"""Provider-neutral row packets for registered Topic Model artifacts.

The optional external read_artifact method returns matched rows aligned by
positions in the requested GeCo user-key list. It may use TeAL WITH joins to
broadcast coarse-grained artifacts to compatible fine-grained observations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from geometric_coder.exceptions import ConfigurationError
from geometric_coder.geometry.base import Matrix


@dataclass(frozen=True)
class ArtifactRows:
    """Ordinary external matrix/table plus optional key matching information."""

    kind: Literal["matrix", "table"]
    values: Any
    columns: Sequence[str]
    positions: Sequence[int] | None = None
    row_ids: Sequence[Any] | None = None


@dataclass(frozen=True)
class TopicModelRows:
    """A keyed and checked view of two required topic artifacts."""

    topic_model_id: int
    topic_ids: tuple[str, ...]
    matched_positions: tuple[int, ...]
    memberships: Matrix
    dominant: pd.DataFrame
    descriptions: ArtifactRows | None
    topic_labels: dict[str, str]


def _columns(rows: ArtifactRows, *, role: str) -> tuple[str, ...]:
    columns = tuple(rows.columns)
    if not columns or any(
        not isinstance(name, str) or not name.strip() for name in columns
    ) or len(set(columns)) != len(columns):
        raise ConfigurationError(
            f"Topic Model {role} needs unique nonempty feature names."
        )
    return columns


def _matrix(rows: ArtifactRows, *, role: str) -> Matrix:
    if rows.kind != "matrix":
        raise ConfigurationError(f"Topic Model {role} must be a matrix artifact.")
    values = rows.values
    if sparse.issparse(values):
        result: Matrix = sparse.csr_matrix(values)
        numeric = np.issubdtype(result.dtype, np.number)
        finite = numeric and np.isfinite(result.data).all()
    else:
        result = np.asarray(values)
        numeric = np.issubdtype(result.dtype, np.number)
        finite = numeric and np.isfinite(result).all()
    if result.ndim != 2 or result.shape[1] != len(_columns(rows, role=role)):
        raise ConfigurationError(
            f"Topic Model {role} matrix shape and feature names disagree."
        )
    if not finite:
        raise ConfigurationError(f"Topic Model {role} has non-finite/non-numeric values.")
    return result


def _keyed(
    rows: ArtifactRows, *, count: int, role: str
) -> tuple[tuple[int, ...], np.ndarray]:
    if rows.positions is None:
        raise ConfigurationError(
            f"Topic Model {role} must report matched requested-key positions."
        )
    positions = tuple(rows.positions)
    if any(
        isinstance(p, (bool, np.bool_))
        or not isinstance(p, (int, np.integer))
        or p < 0
        or p >= count
        for p in positions
    ) or len(set(positions)) != len(positions):
        raise ConfigurationError(
            f"Topic Model {role} returned invalid or duplicate key positions."
        )
    order = np.argsort(np.asarray(positions, dtype=np.int64), kind="stable")
    return tuple(int(positions[int(i)]) for i in order), order


def validate_topic_model_rows(
    model: dict[str, Any],
    membership: ArtifactRows,
    dominant: ArtifactRows,
    *,
    requested_count: int,
    description: ArtifactRows | None = None,
) -> TopicModelRows:
    """Validate source rows/coverage without accessing TeAL internals."""
    if not isinstance(membership, ArtifactRows) or not isinstance(dominant, ArtifactRows):
        raise ConfigurationError("Provider returned an invalid Topic Model artifact packet.")
    topic_ids = _columns(membership, role="membership")
    matrix = _matrix(membership, role="membership")
    positions, order = _keyed(
        membership, count=requested_count, role="membership"
    )
    if matrix.shape[0] != len(positions):
        raise ConfigurationError("Membership row count differs from key coverage.")
    matrix = matrix[order]

    if dominant.kind != "table" or not isinstance(dominant.values, pd.DataFrame):
        raise ConfigurationError("Topic Model ArgMax must be a keyed table.")
    if not {"max_feature", "max_value"}.issubset(dominant.values.columns):
        raise ConfigurationError(
            "Topic Model ArgMax needs max_feature and max_value columns."
        )
    d_positions, d_order = _keyed(dominant, count=requested_count, role="ArgMax")
    if len(d_positions) != len(dominant.values):
        raise ConfigurationError("ArgMax row count differs from key coverage.")
    if positions != d_positions:
        raise ConfigurationError("Membership and ArgMax coverage do not match.")
    frame = dominant.values.iloc[d_order].reset_index(drop=True).copy()
    for label in frame["max_feature"]:
        if not pd.isna(label) and (
            not isinstance(label, str) or label not in topic_ids
        ):
            raise ConfigurationError("ArgMax references an unknown topic component.")
    try:
        scores = pd.to_numeric(frame["max_value"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("ArgMax contains nonnumeric scores.") from exc
    if not np.isfinite(scores).all():
        raise ConfigurationError("ArgMax contains non-finite scores.")
    if any(
        pd.isna(label) and score != 0
        for label, score in zip(frame["max_feature"], scores, strict=True)
    ):
        raise ConfigurationError("Unassigned ArgMax rows must have zero max_value.")

    if description is not None:
        if not isinstance(description, ArtifactRows):
            raise ConfigurationError("Topic description packet is invalid.")
        desc = _matrix(description, role="description")
        if desc.shape[0] != len(topic_ids):
            raise ConfigurationError(
                "Topic description row count does not match membership width."
            )
        if description.row_ids is not None:
            row_ids = tuple(description.row_ids)
            if len(row_ids) != len(topic_ids) or len(set(map(str, row_ids))) != len(row_ids):
                raise ConfigurationError("Description has invalid topic row IDs.")
            suffixes = [re.fullmatch(r".*_(\d+)", name) for name in topic_ids]
            if all(suffixes) and all(str(k).isdigit() for k in row_ids):
                expected = tuple(int(m.group(1)) for m in suffixes if m is not None)
                if tuple(int(k) for k in row_ids) != expected:
                    raise ConfigurationError(
                        "Description topic IDs do not match membership columns."
                    )

    return TopicModelRows(
        topic_model_id=int(model["topic_model_id"]),
        topic_ids=topic_ids,
        matched_positions=positions,
        memberships=matrix,
        dominant=frame,
        descriptions=description,
        topic_labels=dict(model["topic_labels"]),
    )
