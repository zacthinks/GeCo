"""Persistence for sparse matrices, dense matrices, and fitted estimators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse

from geometric_coder.geometry.base import Matrix


class ArtifactStore:
    """Project-local sidecar storage for large numerical artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.geometry_dir = root / "geometries"
        self.view_dir = root / "views"
        self.model_dir = root / "models"
        self.observation_dir = root / "observations"
        for directory in (
            self.geometry_dir, self.view_dir, self.model_dir, self.observation_dir
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(name: str) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in name
        )

    def save_geometry_matrix(self, name: str, matrix: Matrix) -> tuple[str, bool]:
        """Save a geometry matrix and return its project-relative path and sparsity."""
        safe = self._safe_name(name)
        if sparse.issparse(matrix):
            path = self.geometry_dir / f"{safe}.npz"
            sparse.save_npz(path, sparse.csr_matrix(matrix))
            return str(path.relative_to(self.root.parent)), True
        path = self.geometry_dir / f"{safe}.npy"
        np.save(path, np.asarray(matrix), allow_pickle=False)
        return str(path.relative_to(self.root.parent)), False

    def load_matrix(self, project_root: Path, relative_path: str, is_sparse: bool) -> Matrix:
        """Load a previously persisted matrix."""
        path = project_root / relative_path
        if is_sparse:
            return sparse.load_npz(path)
        return np.load(path, allow_pickle=False)

    def load_view(self, project_root: Path, relative_path: str) -> np.ndarray:
        """Load persisted two-dimensional view coordinates."""
        coordinates = np.load(project_root / relative_path, allow_pickle=False)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(f"View artifact must have shape (n, 2), got {coordinates.shape}.")
        return np.asarray(coordinates, dtype=np.float32)

    def load_geometry_state(
        self,
        project_root: Path,
        relative_path: str | None,
    ) -> dict[str, Any]:
        """Load fitted estimator state for a geometry."""
        if relative_path is None:
            return {}
        state = joblib.load(project_root / relative_path)
        if not isinstance(state, dict):
            raise TypeError("Persisted geometry state must be a dictionary.")
        return state

    def save_geometry_state(self, name: str, state: dict[str, Any]) -> str | None:
        """Persist fitted state when a geometry exposes any."""
        if not state:
            return None
        path = self.geometry_dir / f"{self._safe_name(name)}.joblib"
        joblib.dump(state, path)
        return str(path.relative_to(self.root.parent))

    def save_view(self, name: str, coordinates: np.ndarray) -> str:
        """Save two-dimensional view coordinates."""
        path = self.view_dir / f"{self._safe_name(name)}.npy"
        np.save(path, np.asarray(coordinates), allow_pickle=False)
        return str(path.relative_to(self.root.parent))

    def save_observation_vector(
        self, observation_token: str, geometry_name: str, matrix: Matrix
    ) -> tuple[str, bool]:
        """Persist one derived observation vector."""
        safe = self._safe_name(f"{observation_token}_{geometry_name}")
        if sparse.issparse(matrix):
            path = self.observation_dir / f"{safe}.npz"
            sparse.save_npz(path, sparse.csr_matrix(matrix))
            return str(path.relative_to(self.root.parent)), True
        path = self.observation_dir / f"{safe}.npy"
        np.save(path, np.asarray(matrix), allow_pickle=False)
        return str(path.relative_to(self.root.parent)), False

    def save_view_state(self, name: str, state: dict[str, Any]) -> str | None:
        """Persist a fitted two-dimensional view transformer."""
        if not state:
            return None
        path = self.view_dir / f"{self._safe_name(name)}.joblib"
        joblib.dump(state, path)
        return str(path.relative_to(self.root.parent))

    def load_view_state(
        self, project_root: Path, relative_path: str | None
    ) -> dict[str, Any]:
        """Load a fitted view-transform state."""
        if relative_path is None:
            return {}
        state = joblib.load(project_root / relative_path)
        if not isinstance(state, dict):
            raise TypeError("Persisted view state must be a dictionary.")
        return state

    def save_model(self, name: str, model: Any) -> str:
        """Persist a fitted model and return its project-relative path."""
        path = self.model_dir / f"{self._safe_name(name)}.joblib"
        joblib.dump(model, path)
        return str(path.relative_to(self.root.parent))

    def load_model(self, project_root: Path, relative_path: str) -> Any:
        """Load one persisted fitted model."""
        return joblib.load(project_root / relative_path)
