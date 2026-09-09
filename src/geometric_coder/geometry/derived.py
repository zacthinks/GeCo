"""Geometries derived from existing geometries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD

from geometric_coder.geometry.base import Geometry, Matrix, ViewSpec


@dataclass(frozen=True, slots=True)
class SVDOptions:
    n_components: int = 100
    algorithm: str = "randomized"
    n_iter: int = 5
    random_state: int | None = 0


class SVDGeometry(Geometry):
    """A truncated-SVD/LSA geometry derived from another geometry."""

    default_view = ViewSpec("pca", {"n_components": 2, "random_state": 0})

    def __init__(self, source: Geometry, **kwargs: Any) -> None:
        super().__init__()
        self.source = source
        self.options = SVDOptions(**kwargs)
        self._reducer: TruncatedSVD | None = None

    @property
    def dependencies(self) -> tuple[Geometry, ...]:
        return (self.source,)

    @property
    def supports_query(self) -> bool:
        return self._reducer is not None and self.source.supports_query

    @property
    def supports_text_transform(self) -> bool:
        return self._reducer is not None and self.source.supports_text_transform

    def _fit_transform(self, texts: list[str]) -> Matrix:
        del texts
        source_matrix = self.source.matrix
        max_components = max(1, min(source_matrix.shape) - 1)
        n_components = min(self.options.n_components, max_components)
        self._reducer = TruncatedSVD(
            n_components=n_components,
            algorithm=self.options.algorithm,
            n_iter=self.options.n_iter,
            random_state=self.options.random_state,
        )
        return np.asarray(self._reducer.fit_transform(source_matrix), dtype=np.float32)

    def transform_texts(self, texts: list[str]) -> Matrix:
        if self._reducer is None:
            raise RuntimeError("SVD geometry must be fitted before transformation.")
        source_matrix = self.source.transform_texts(texts)
        return np.asarray(self._reducer.transform(source_matrix), dtype=np.float32)

    def configuration(self) -> dict[str, Any]:
        return {"type": "svd", **asdict(self.options)}

    def artifact_state(self) -> dict[str, Any]:
        return {"reducer": self._reducer}

    def load_artifact_state(self, state: dict[str, Any]) -> None:
        reducer = state.get("reducer")
        if not isinstance(reducer, TruncatedSVD):
            raise TypeError("Invalid SVD-geometry artifact state.")
        self._reducer = reducer
