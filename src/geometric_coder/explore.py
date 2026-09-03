"""Pure exploration, filtering, pagination, and recommendation helpers for GeCo."""

from __future__ import annotations

import random
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances

from geometric_coder.geometry.base import Matrix

NavigationStrategy = Literal[
    "nearest",
    "farthest",
    "random",
    "most_similar",
    "least_similar",
    "random_in_filter",
]


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Combined literal and semantic eligibility for plotted units."""

    eligible: np.ndarray
    literal_matches: np.ndarray
    semantic_in_range: np.ndarray


@dataclass(frozen=True, slots=True)
class PageSlice:
    """One deterministic page of a randomly permuted corpus."""

    positions: np.ndarray
    page_index: int
    page_count: int
    page_size: int


class InvalidRegexError(ValueError):
    """Raised when a literal-search regular expression cannot be compiled."""


def literal_match_mask(
    texts: Sequence[str],
    query: str,
    *,
    case_sensitive: bool = False,
    regex: bool = False,
) -> np.ndarray:
    """Return a Boolean mask for substring or regular-expression matches."""
    if not query:
        return np.ones(len(texts), dtype=bool)
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error as error:
            raise InvalidRegexError(str(error)) from error
        return np.fromiter((pattern.search(text) is not None for text in texts), dtype=bool)

    needle = query if case_sensitive else query.casefold()
    if case_sensitive:
        return np.fromiter((needle in text for text in texts), dtype=bool)
    return np.fromiter((needle in text.casefold() for text in texts), dtype=bool)


def combine_filters(
    texts: Sequence[str],
    *,
    literal_query: str = "",
    case_sensitive: bool = False,
    regex: bool = False,
    semantic_scores: Sequence[float] | None = None,
    semantic_range: tuple[float, float] | None = None,
) -> FilterResult:
    """Combine literal and semantic filters using conjunctive eligibility."""
    literal_matches = literal_match_mask(
        texts,
        literal_query,
        case_sensitive=case_sensitive,
        regex=regex,
    )
    semantic_in_range = np.ones(len(texts), dtype=bool)
    if semantic_scores is not None:
        scores = np.asarray(semantic_scores, dtype=float)
        if scores.shape != (len(texts),):
            raise ValueError("semantic_scores must contain one value per text")
        if semantic_range is None:
            lower, upper = float(np.nanmin(scores)), float(np.nanmax(scores))
        else:
            lower, upper = semantic_range
        if lower > upper:
            lower, upper = upper, lower
        semantic_in_range = np.isfinite(scores) & (scores >= lower) & (scores <= upper)
    return FilterResult(
        eligible=literal_matches & semantic_in_range,
        literal_matches=literal_matches,
        semantic_in_range=semantic_in_range,
    )


def deterministic_page(
    total_units: int,
    *,
    page_size: int = 5_000,
    page_index: int = 0,
    seed: int = 0,
) -> PageSlice:
    """Return one stable random page of corpus row positions.

    The corpus is shuffled with a fixed seed and divided into equally sized pages.
    ``page_index`` is zero-based and is clamped to the available range.
    """
    if total_units < 0:
        raise ValueError("total_units must be nonnegative")
    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    page_count = max(1, ceil(total_units / page_size))
    bounded_index = min(max(int(page_index), 0), page_count - 1)
    if total_units == 0:
        positions = np.asarray([], dtype=int)
    else:
        permutation = np.random.default_rng(seed).permutation(total_units)
        start = bounded_index * page_size
        positions = np.asarray(permutation[start : start + page_size], dtype=int)
    return PageSlice(
        positions=positions,
        page_index=bounded_index,
        page_count=page_count,
        page_size=page_size,
    )




def deterministic_page_with_inclusions(
    total_units: int,
    *,
    page_size: int = 5_000,
    page_index: int = 0,
    seed: int = 0,
    include_positions: Collection[int] = (),
) -> PageSlice:
    """Return a deterministic page augmented with required corpus positions.

    The ordinary random page remains the sampling frame. Any valid positions in
    ``include_positions`` are appended exactly once, which is useful for views
    that must retain all substantively selected units while still sampling the
    rest of a very large corpus.
    """
    page = deterministic_page(
        total_units,
        page_size=page_size,
        page_index=page_index,
        seed=seed,
    )
    if not include_positions:
        return page

    requested = np.asarray(sorted({int(position) for position in include_positions}), dtype=int)
    if requested.size and (requested.min() < 0 or requested.max() >= total_units):
        raise IndexError("include_positions contain an out-of-range corpus position")
    existing = set(int(position) for position in page.positions)
    extras = np.asarray(
        [int(position) for position in requested if int(position) not in existing],
        dtype=int,
    )
    positions = (
        np.concatenate([page.positions, extras])
        if extras.size
        else page.positions.copy()
    )
    return PageSlice(
        positions=positions,
        page_index=page.page_index,
        page_count=page.page_count,
        page_size=page.page_size,
    )


def recommend_unit(
    *,
    strategy: NavigationStrategy,
    unit_ids: Sequence[int],
    seen_unit_ids: Collection[int],
    eligible: Sequence[bool] | None = None,
    matrix: Matrix | None = None,
    focal_unit_id: int | None = None,
    semantic_scores: Sequence[float] | None = None,
    random_seed: int | None = None,
) -> int | None:
    """Recommend one unseen eligible unit under a navigation strategy."""
    ids = np.asarray(unit_ids, dtype=int)
    if ids.ndim != 1:
        raise ValueError("unit_ids must be one-dimensional")
    mask = np.ones(ids.size, dtype=bool)
    if eligible is not None:
        eligible_array = np.asarray(eligible, dtype=bool)
        if eligible_array.shape != mask.shape:
            raise ValueError("eligible must contain one value per unit")
        mask &= eligible_array
    if seen_unit_ids:
        mask &= ~np.isin(ids, np.fromiter(seen_unit_ids, dtype=int))
    # Navigation should always move away from the current focal unit, even when
    # revisits are allowed. Otherwise nearest or random navigation may simply
    # return the point the researcher is already reading.
    if focal_unit_id is not None:
        mask &= ids != int(focal_unit_id)
    candidate_positions = np.flatnonzero(mask)
    if candidate_positions.size == 0:
        return None

    if strategy in {"random", "random_in_filter"}:
        rng = random.Random(random_seed)
        return int(ids[rng.choice(candidate_positions.tolist())])

    if strategy in {"most_similar", "least_similar"}:
        if semantic_scores is None:
            raise ValueError(
                f"{strategy.replace('_', ' ').title()} requires an active semantic search."
            )
        scores = np.asarray(semantic_scores, dtype=float)
        if scores.shape != ids.shape:
            raise ValueError("semantic_scores must contain one value per unit")
        candidate_scores = scores[candidate_positions]
        finite = np.isfinite(candidate_scores)
        if not finite.any():
            return None
        finite_positions = candidate_positions[finite]
        finite_scores = candidate_scores[finite]
        relative_index = (
            int(np.argmax(finite_scores))
            if strategy == "most_similar"
            else int(np.argmin(finite_scores))
        )
        return int(ids[finite_positions[relative_index]])

    if matrix is None:
        raise ValueError(f"{strategy} requires a geometry matrix")
    if matrix.shape[0] != ids.size:
        raise ValueError("matrix row count must equal the number of unit IDs")

    if strategy in {"nearest", "farthest"}:
        if focal_unit_id is None:
            raise ValueError(
                f"Select a focal point before requesting the {strategy} unseen unit."
            )
        focal_positions = np.flatnonzero(ids == focal_unit_id)
        if focal_positions.size != 1:
            raise KeyError(f"Unknown or duplicate focal unit ID: {focal_unit_id}")
        focal_position = int(focal_positions[0])
        candidate_matrix = _matrix_rows(matrix, candidate_positions)
        focal_matrix = _matrix_rows(matrix, np.asarray([focal_position], dtype=int))
        distances = pairwise_distances(
            candidate_matrix,
            focal_matrix,
            metric="cosine",
        ).reshape(-1)
        finite = np.isfinite(distances)
        if not finite.any():
            return None
        finite_positions = candidate_positions[finite]
        finite_distances = distances[finite]
        relative_index = (
            int(np.argmin(finite_distances))
            if strategy == "nearest"
            else int(np.argmax(finite_distances))
        )
        return int(ids[finite_positions[relative_index]])

    raise ValueError(f"Unknown navigation strategy: {strategy!r}")



def recommend_page_unit(
    *,
    strategy: NavigationStrategy,
    unit_ids: Sequence[int],
    page_positions: Sequence[int],
    seen_unit_ids: Collection[int],
    eligible: Sequence[bool] | None = None,
    matrix: Matrix | None = None,
    focal_unit_id: int | None = None,
    semantic_scores: Sequence[float] | None = None,
    random_seed: int | None = None,
) -> int | None:
    """Recommend strictly within one deterministic Explore page.

    ``eligible``, ``matrix``, and ``semantic_scores`` are corpus-wide inputs.
    This helper slices all of them to the same page before delegating to
    :func:`recommend_unit`, preventing accidental navigation to another page.
    """
    all_ids = np.asarray(unit_ids, dtype=int)
    positions = np.asarray(page_positions, dtype=int)
    if all_ids.ndim != 1:
        raise ValueError("unit_ids must be one-dimensional")
    if positions.ndim != 1:
        raise ValueError("page_positions must be one-dimensional")
    if positions.size and (positions.min() < 0 or positions.max() >= all_ids.size):
        raise IndexError("page_positions contain an out-of-range corpus position")
    page_ids = all_ids[positions]
    page_eligible = None
    if eligible is not None:
        all_eligible = np.asarray(eligible, dtype=bool)
        if all_eligible.shape != all_ids.shape:
            raise ValueError("eligible must contain one value per corpus unit")
        page_eligible = all_eligible[positions]
    page_matrix = None if matrix is None else _matrix_rows(matrix, positions)
    page_semantic = None
    if semantic_scores is not None:
        all_scores = np.asarray(semantic_scores, dtype=float)
        if all_scores.shape != all_ids.shape:
            raise ValueError("semantic_scores must contain one value per corpus unit")
        page_semantic = all_scores[positions]
    return recommend_unit(
        strategy=strategy,
        unit_ids=page_ids,
        seen_unit_ids=seen_unit_ids,
        eligible=page_eligible,
        matrix=page_matrix,
        focal_unit_id=focal_unit_id,
        semantic_scores=page_semantic,
        random_seed=random_seed,
    )

def matrix_row_count(matrix: Matrix) -> int:
    """Return a matrix row count while retaining sparse compatibility."""
    if not isinstance(matrix, np.ndarray) and not sparse.issparse(matrix):
        raise TypeError("matrix must be a NumPy array or SciPy sparse matrix")
    return int(matrix.shape[0])


def _matrix_rows(matrix: Matrix, positions: np.ndarray) -> Matrix:
    """Take rows while preserving a two-dimensional matrix for dense inputs."""
    if sparse.issparse(matrix):
        return matrix[positions]
    dense = np.asarray(matrix)
    return dense[positions, :]
