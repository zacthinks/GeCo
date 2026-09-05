"""Two-dimensional projection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD

from geometric_coder.exceptions import ConfigurationError
from geometric_coder.geometry.base import Matrix, ViewSpec


@dataclass(slots=True)
class ViewFit:
    """Coordinates plus the fitted state needed to place new observations."""

    coordinates: np.ndarray
    state: dict[str, Any]

    @property
    def supports_transform(self) -> bool:
        return bool(self.state)


def compute_view(matrix: Matrix, spec: ViewSpec) -> np.ndarray:
    """Compute a two-dimensional view from a geometry matrix."""
    return fit_view(matrix, spec).coordinates


def fit_view(matrix: Matrix, spec: ViewSpec) -> ViewFit:
    """Fit a two-dimensional view and retain a transformer for new observations."""
    method = spec.method.lower()
    parameters = dict(spec.parameters)
    parameters["n_components"] = 2

    if method == "auto":
        method = "truncated_svd" if sparse.issparse(matrix) else "pca"

    if method in {"truncated_svd", "svd"}:
        return _svd_view(matrix, parameters)
    if method == "pca":
        return _pca_view(matrix, parameters)
    if method == "umap":
        return _umap_view(matrix, parameters)
    if method == "identity":
        return ViewFit(
            coordinates=_pad_to_two(np.asarray(matrix)),
            state={"method": "identity"},
        )
    raise ValueError(f"Unknown view method: {spec.method!r}")


def transform_view(matrix: Matrix, state: dict[str, Any]) -> np.ndarray:
    """Project new geometry rows through a fitted view state."""
    method = str(state.get("method", ""))
    if method == "identity":
        dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        return _pad_to_two(dense)
    transformer = state.get("transformer")
    if transformer is None or not hasattr(transformer, "transform"):
        raise ValueError("This view does not retain a transformer for new observations.")
    transformed_input = (
        matrix.toarray()
        if method == "pca" and sparse.issparse(matrix)
        else matrix
    )
    transformed = transformer.transform(transformed_input)
    return np.asarray(transformed, dtype=np.float32)


def _svd_view(matrix: Matrix, parameters: dict[str, Any]) -> ViewFit:
    if min(matrix.shape) <= 2:
        dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
        return ViewFit(_pad_to_two(dense), {"method": "identity"})
    reducer = TruncatedSVD(**parameters)
    coordinates = np.asarray(reducer.fit_transform(matrix), dtype=np.float32)
    return ViewFit(coordinates, {"method": "truncated_svd", "transformer": reducer})


def _pca_view(matrix: Matrix, parameters: dict[str, Any]) -> ViewFit:
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    if min(dense.shape) <= 2:
        return ViewFit(_pad_to_two(dense), {"method": "identity"})
    reducer = PCA(**parameters)
    coordinates = np.asarray(reducer.fit_transform(dense), dtype=np.float32)
    return ViewFit(coordinates, {"method": "pca", "transformer": reducer})


def _umap_view(matrix: Matrix, parameters: dict[str, Any]) -> ViewFit:
    try:
        from umap import UMAP
    except ImportError as error:
        raise ConfigurationError(
            "UMAP is part of the standard GeCo installation but could not be imported. "
            "Reinstall geometric-coder with its default dependencies."
        ) from error
    if matrix.shape[0] < 4:
        return _pca_view(matrix, {"n_components": 2})
    reducer = UMAP(**parameters)
    coordinates = np.asarray(reducer.fit_transform(matrix), dtype=np.float32)
    return ViewFit(coordinates, {"method": "umap", "transformer": reducer})


def _pad_to_two(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("Projection input must be a two-dimensional matrix.")
    if matrix.shape[1] >= 2:
        return np.asarray(matrix[:, :2], dtype=np.float32)
    if matrix.shape[1] == 1:
        first = np.asarray(matrix[:, 0], dtype=np.float32)
        second = np.zeros(matrix.shape[0], dtype=np.float32)
        return np.column_stack([first, second])
    return np.zeros((matrix.shape[0], 2), dtype=np.float32)
