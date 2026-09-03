"""Core abstractions for GeCo geometries and two-dimensional views."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import numpy as np
from scipy import sparse

from geometric_coder.exceptions import ConfigurationError

Matrix = np.ndarray | sparse.spmatrix


@dataclass(frozen=True, slots=True)
class ViewSpec:
    """A reproducible two-dimensional projection specification."""

    method: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


class Geometry(ABC):
    """A reproducible mapping from atomic observations into a numerical space.

    Geometry objects are configuration-bearing computational objects. Public
    registry names are assigned by :class:`GeometryRegistry`; dependencies are
    represented directly as object references.
    """

    modality: ClassVar[str] = "text"
    default_view: ClassVar[ViewSpec] = ViewSpec("auto")

    def __init__(self) -> None:
        self._matrix: Matrix | None = None
        self._fitted = False

    @property
    def dependencies(self) -> tuple[Geometry, ...]:
        """Return direct geometry dependencies."""
        return ()

    @property
    def matrix(self) -> Matrix:
        """Return the fitted geometry matrix."""
        if self._matrix is None:
            raise RuntimeError("Geometry has not been fitted.")
        return self._matrix

    @property
    def is_fitted(self) -> bool:
        """Whether the geometry has been fitted."""
        return self._fitted

    @property
    def supports_query(self) -> bool:
        """Whether new text queries can be transformed into this space."""
        return False

    def validate_modality(self, modality: str) -> None:
        """Validate that the geometry can operate on the project modality."""
        if modality != self.modality:
            raise ConfigurationError(
                f"{type(self).__name__} supports modality {self.modality!r}, "
                f"not {modality!r}."
            )

    def validate_runtime(self) -> None:
        """Validate optional runtime resources before project creation begins.

        Subclasses should override this when they depend on external models or
        other resources that are not installed as ordinary Python packages.
        The default implementation has no additional requirements.
        """

    def fit_transform(self, texts: list[str]) -> Matrix:
        """Fit the geometry and return its matrix."""
        matrix = self._fit_transform(texts)
        if matrix.shape[0] != len(texts):
            raise RuntimeError(
                f"{type(self).__name__} returned {matrix.shape[0]} rows for "
                f"{len(texts)} observations."
            )
        self._matrix = matrix
        self._fitted = True
        return matrix

    @abstractmethod
    def _fit_transform(self, texts: list[str]) -> Matrix:
        """Implement fitting and transformation."""

    def transform_texts(self, texts: list[str]) -> Matrix:
        """Transform new observations through the frozen fitted geometry."""
        del texts
        raise NotImplementedError(
            f"{type(self).__name__} does not support new observation transformation."
        )

    def transform_query(self, query: str) -> Matrix:
        """Transform one query into this geometry."""
        return self.transform_texts([query])

    @abstractmethod
    def configuration(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""

    def artifact_state(self) -> dict[str, Any]:
        """Return fitted state to serialize separately from the matrix.

        Subclasses may override this to expose fitted estimators. The artifact
        store serializes this mapping with joblib.
        """
        return {}

    def load_artifact_state(self, state: dict[str, Any]) -> None:
        """Restore fitted state loaded from disk."""
        if state:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement fitted-state restoration."
            )

    def restore_matrix(self, matrix: Matrix) -> None:
        """Attach a previously persisted matrix."""
        self._matrix = matrix
        self._fitted = True
