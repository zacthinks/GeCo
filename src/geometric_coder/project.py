"""The main GeCo project object shared by Python, notebooks, and the web UI."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

from geometric_coder.classifiers import (
    effective_hyperparameters,
    fit_classifier,
    normalize_hyperparameters,
    score_classifier,
    tuning_candidates,
)
from geometric_coder.external import ExternalDataProvider
from geometric_coder.exceptions import (
    ConfigurationError,
    ProjectExistsError,
    ProjectNotFoundError,
)
from geometric_coder.geometry import CountGeometry, SVDGeometry, SentenceTransformerGeometry
from geometric_coder.apply import aggregate_probabilities, build_apply_proposals
from geometric_coder.focus import recommend_focus_unit
from geometric_coder.geometry.base import Geometry, Matrix, ViewSpec
from geometric_coder.projections import fit_view, transform_view
from geometric_coder.progress import ProgressCallback, emit_progress
from geometric_coder.registry import GeometryRegistry
from geometric_coder.storage import ArtifactStore, ProjectDatabase, SCHEMA_VERSION
from geometric_coder.storage.database import (
    FIXED_COMMITTEE_AGGREGATIONS,
    TRAINABLE_COMMITTEE_AGGREGATIONS,
)

Modality = Literal["text", "image"]


class GeometricCoder:
    """One persistent geometric-coding project.

    The object is intentionally thin: structured state is stored in SQLite,
    and large numerical artifacts are stored in project-local sidecar files.
    """

    DATABASE_FILENAME = "project.sqlite3"
    ARTIFACT_DIRECTORY = "artifacts"

    def __init__(
        self,
        project_dir: Path,
        *,
        external_provider: ExternalDataProvider | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.database = ProjectDatabase(project_dir / self.DATABASE_FILENAME)
        self.artifacts = ArtifactStore(project_dir / self.ARTIFACT_DIRECTORY)
        self.external_provider = external_provider
        self._runtime_geometries: dict[str, Geometry] | None = None

    @classmethod
    def create(
        cls,
        *,
        project_dir: str | Path,
        data: pd.DataFrame,
        keys: Sequence[str],
        text: str,
        modality: Modality = "text",
        geometries: Mapping[str, Any] | None = None,
        metadata: Sequence[str] | None = None,
        overwrite: bool = False,
        progress: ProgressCallback | None = None,
    ) -> GeometricCoder:
        """Create, eagerly compute, and persist a new GeCo project."""
        root = Path(project_dir).expanduser().resolve()
        emit_progress(
            progress,
            phase="project",
            message="Validating input data",
            detail=f"{len(data):,} atomic units",
        )
        key_columns, metadata_columns = cls._validate_input(
            data=data,
            keys=keys,
            text=text,
            modality=modality,
            metadata=metadata,
        )

        registry: GeometryRegistry | None = None
        if geometries:
            registry = cls._prepare_geometry_registry(
                geometry_spec=geometries,
                modality=modality,
                progress=progress,
            )

        emit_progress(
            progress,
            phase="project",
            message="Preparing project directory",
            detail=str(root),
        )
        cls._prepare_directory(root, overwrite=overwrite)
        project: GeometricCoder | None = None
        try:
            project = cls(root)
            emit_progress(
                progress,
                phase="database",
                message="Initializing SQLite project database",
            )
            project.database.initialize(
                modality=modality,
                text_column=text,
                key_columns=key_columns,
                metadata_columns=metadata_columns,
                external_backed=False,
            )
            emit_progress(
                progress,
                phase="database",
                message="Importing texts, keys, and metadata",
                detail=f"{len(data):,} rows",
            )
            project.database.import_dataframe(
                data,
                key_columns=key_columns,
                text_column=text,
                metadata_columns=metadata_columns,
            )
            if registry is not None:
                project._compute_geometries(
                    texts=data[text].astype(str).tolist(),
                    modality=modality,
                    registry=registry,
                    progress=progress,
                )
                project.ensure_default_classifier_specs()
            emit_progress(
                progress,
                phase="complete",
                message="GeCo project initialization complete",
                current=1,
                total=1,
                detail=str(root),
            )
            return project
        except Exception as error:
            # All database operations explicitly close their SQLite connections,
            # but collect once before cleanup to release any traceback-held
            # cursors promptly on Windows. Never let cleanup mask the real error.
            project = None
            gc.collect()
            if root.exists():
                try:
                    cls._remove_directory(root)
                except OSError as cleanup_error:
                    error.add_note(
                        "GeCo could not completely remove the failed project "
                        f"directory {root}: {cleanup_error}"
                    )
            raise

    @classmethod
    def create_external(
        cls,
        *,
        project_dir: str | Path,
        data: pd.DataFrame,
        keys: Sequence[str],
        text: str,
        external_provider: ExternalDataProvider,
        modality: Modality = "text",
        metadata: Sequence[str] | None = None,
        overwrite: bool = False,
        progress: ProgressCallback | None = None,
    ) -> GeometricCoder:
        """Create a TeAL-style project whose numerical resources are provider-backed.

        The document universe is copied into GeCo's normal SQLite store. External
        geometries and views are then attached with ``register_external_geometry``
        and ``register_external_view``. GeCo never interprets provider references.
        """
        if external_provider is None:
            raise ConfigurationError("An external data provider is required.")
        root = Path(project_dir).expanduser().resolve()
        emit_progress(
            progress, phase="project", message="Validating input data",
            detail=f"{len(data):,} atomic units",
        )
        key_columns, metadata_columns = cls._validate_input(
            data=data, keys=keys, text=text, modality=modality, metadata=metadata
        )
        cls._prepare_directory(root, overwrite=overwrite)
        project: GeometricCoder | None = None
        try:
            project = cls(root, external_provider=external_provider)
            project.database.initialize(
                modality=modality, text_column=text, key_columns=key_columns,
                metadata_columns=metadata_columns, external_backed=True,
            )
            project.database.import_dataframe(
                data, key_columns=key_columns, text_column=text,
                metadata_columns=metadata_columns,
            )
            emit_progress(
                progress, phase="complete",
                message="External GeCo project initialization complete",
                current=1, total=1, detail=str(root),
            )
            return project
        except Exception as error:
            project = None
            gc.collect()
            if root.exists():
                try:
                    cls._remove_directory(root)
                except OSError as cleanup_error:
                    error.add_note(
                        "GeCo could not completely remove the failed project "
                        f"directory {root}: {cleanup_error}"
                    )
            raise

    @classmethod
    def _prepare_geometry_registry(
        cls,
        *,
        geometry_spec: Mapping[str, Any],
        modality: str,
        progress: ProgressCallback | None = None,
    ) -> GeometryRegistry:
        """Normalize and preflight geometries before touching project storage."""
        del cls
        emit_progress(
            progress,
            phase="geometry",
            message="Checking geometry requirements",
        )
        registry = GeometryRegistry()
        registry.register_spec(geometry_spec)
        for registration in registry.build_order():
            geometry = registration.geometry
            geometry.validate_modality(modality)
            geometry.validate_runtime()
        return registry

    @classmethod
    def open(
        cls,
        project_dir: str | Path,
        *,
        external_provider: ExternalDataProvider | None = None,
    ) -> GeometricCoder:
        """Open an existing GeCo project, reconnecting a provider when required."""
        root = Path(project_dir).expanduser().resolve()
        database_path = root / cls.DATABASE_FILENAME
        if not database_path.is_file():
            raise ProjectNotFoundError(f"No GeCo project database found at {database_path}")
        project = cls(root, external_provider=external_provider)
        metadata = project.database.project_metadata()
        observed_schema = int(metadata["schema_version"])
        if observed_schema != SCHEMA_VERSION:
            raise RuntimeError(
                f"This GeCo build requires project schema {SCHEMA_VERSION}, but the "
                f"project uses schema {observed_schema}. GeCo does not migrate projects "
                "automatically. Run the matching one-step script in scripts/ manually "
                "or rebuild a disposable project from source."
            )
        if bool(metadata.get("external_backed")) and external_provider is None:
            raise ConfigurationError(
                "This GeCo project uses externally backed numerical resources. "
                "Reopen it with external_provider=... ."
            )
        project.ensure_default_classifier_specs()
        return project

    @classmethod
    def _prepare_directory(cls, root: Path, *, overwrite: bool) -> None:
        if root.exists():
            if not overwrite:
                raise ProjectExistsError(f"Project directory already exists: {root}")
            if root.is_dir():
                cls._remove_directory(root)
            else:
                root.unlink()
        root.mkdir(parents=True)

    @staticmethod
    def _remove_directory(root: Path, *, attempts: int = 5) -> None:
        """Remove a project directory, briefly retrying transient Windows locks."""
        last_error: OSError | None = None
        for attempt in range(attempts):
            try:
                shutil.rmtree(root)
                return
            except PermissionError as error:
                last_error = error
                gc.collect()
                if attempt + 1 < attempts:
                    time.sleep(0.1 * (attempt + 1))
        if last_error is not None:
            raise last_error

    @staticmethod
    def _validate_input(
        *,
        data: pd.DataFrame,
        keys: Sequence[str],
        text: str,
        modality: Modality,
        metadata: Sequence[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        if modality not in {"text", "image"}:
            raise ConfigurationError(f"Unsupported modality: {modality!r}")
        if modality != "text":
            raise ConfigurationError(
                "The schema is modality-ready, but the current MVP implements text projects only."
            )
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if data.empty:
            raise ConfigurationError("data must contain at least one row")
        key_columns = [str(column) for column in keys]
        if not key_columns:
            raise ConfigurationError("At least one key column is required.")
        if len(set(key_columns)) != len(key_columns):
            raise ConfigurationError("Key columns must be unique.")
        required = [*key_columns, text]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ConfigurationError(f"Missing required columns: {missing}")
        if text in key_columns:
            raise ConfigurationError("The text column may not also be a key column.")
        if data[text].isna().any():
            raise ConfigurationError("The text column may not contain missing values.")
        if metadata is None:
            metadata_columns = [
                str(column)
                for column in data.columns
                if column not in {*key_columns, text}
            ]
        else:
            metadata_columns = [str(column) for column in metadata]
            if len(set(metadata_columns)) != len(metadata_columns):
                raise ConfigurationError("Metadata columns must be unique.")
            unavailable = [column for column in metadata_columns if column not in data.columns]
            if unavailable:
                raise ConfigurationError(
                    f"Missing requested metadata columns: {unavailable}"
                )
            reserved = [
                column
                for column in metadata_columns
                if column in {*key_columns, text}
            ]
            if reserved:
                raise ConfigurationError(
                    "Metadata columns may not duplicate key or text columns: "
                    f"{reserved}"
                )
        return key_columns, metadata_columns

    def _compute_geometries(
        self,
        *,
        texts: list[str],
        modality: str,
        registry: GeometryRegistry,
        progress: ProgressCallback | None = None,
    ) -> None:
        build_order = registry.build_order()
        name_by_object: dict[int, str] = {
            id(registration.geometry): registration.name for registration in build_order
        }

        total = len(build_order)
        for index, registration in enumerate(build_order, start=1):
            geometry = registration.geometry
            geometry.validate_modality(modality)
            started = time.perf_counter()
            emit_progress(
                progress,
                phase="geometry",
                message=f"Computing geometry: {registration.name}",
                current=index,
                total=total,
                detail=type(geometry).__name__,
            )
            matrix = geometry.fit_transform(texts)
            elapsed = time.perf_counter() - started
            emit_progress(
                progress,
                phase="geometry",
                message=f"Computed geometry: {registration.name}",
                current=index,
                total=total,
                detail=f"shape={matrix.shape}, {elapsed:.1f}s",
            )
            emit_progress(
                progress,
                phase="storage",
                message=f"Caching geometry: {registration.name}",
                current=index,
                total=total,
            )
            matrix_path, is_sparse = self.artifacts.save_geometry_matrix(
                registration.name,
                matrix,
            )
            state_path = self.artifacts.save_geometry_state(
                registration.name,
                geometry.artifact_state(),
            )
            dependency_names = [
                name_by_object[id(dependency)]
                for dependency in geometry.dependencies
            ]
            geometry_id = self.database.register_geometry(
                name=registration.name,
                class_name=f"{type(geometry).__module__}.{type(geometry).__qualname__}",
                modality=geometry.modality,
                public=registration.public,
                config=geometry.configuration(),
                dependency_names=dependency_names,
                matrix_path=matrix_path,
                matrix_sparse=is_sparse,
                state_path=state_path,
                supports_query=geometry.supports_query,
            )
            if registration.public:
                view_spec = geometry.default_view
                view_started = time.perf_counter()
                emit_progress(
                    progress,
                    phase="projection",
                    message=f"Computing default 2D view: {registration.name}",
                    current=index,
                    total=total,
                    detail=view_spec.method,
                )
                fitted_view = fit_view(matrix, view_spec)
                view_name = f"{registration.name}_default_{view_spec.method}"
                view_path = self.artifacts.save_view(view_name, fitted_view.coordinates)
                view_state_path = self.artifacts.save_view_state(
                    view_name, fitted_view.state
                )
                self.database.register_view(
                    geometry_id=geometry_id,
                    name=view_name,
                    method=view_spec.method,
                    parameters=view_spec.parameters,
                    artifact_path=view_path,
                    state_path=view_state_path,
                    supports_transform=fitted_view.supports_transform,
                )
                emit_progress(
                    progress,
                    phase="projection",
                    message=f"Cached default 2D view: {registration.name}",
                    current=index,
                    total=total,
                    detail=f"{view_spec.method}, {time.perf_counter() - view_started:.1f}s",
                )


    def geometries(self, *, public_only: bool = True) -> list[dict[str, Any]]:
        """Return stable persisted geometry records for public inspection.

        Each record includes the project-local ``geometry_id``, stable registry
        ``name``, ``storage_kind`` (``local`` or ``external``), and
        ``supports_query`` capability flag. Hidden dependency geometries are
        excluded by default.
        """
        return self.database.list_geometries(public_only=public_only)

    def has_geometry(self, name: str) -> bool:
        """Return whether a geometry with ``name`` is registered."""
        clean_name = str(name).strip()
        if not clean_name:
            return False
        try:
            self.database.get_geometry_by_name(clean_name)
        except KeyError:
            return False
        return True

    def views(self, geometry_id: int | None = None) -> list[dict[str, Any]]:
        """Return stable persisted two-dimensional view records."""
        return self.database.list_views(geometry_id=geometry_id)

    def has_view(self, name: str) -> bool:
        """Return whether a two-dimensional view with ``name`` is registered."""
        clean_name = str(name).strip()
        if not clean_name:
            return False
        return any(str(record["name"]) == clean_name for record in self.views())

    def sessions(self) -> list[dict[str, Any]]:
        """Return all named resumable sessions."""
        return self.database.list_sessions()

    def update_session_state(self, session_id: int, state: dict[str, Any]) -> None:
        """Persist the current resumable interface state."""
        self.database.update_session_state(session_id, state)

    def _ordered_user_keys(self) -> list[dict[str, Any]]:
        """Return stable user keys in canonical atomic-row order."""
        return [dict(unit["user_key"]) for unit in self.database.list_units()]

    def _require_external_provider(self) -> ExternalDataProvider:
        provider = self.external_provider
        if provider is None:
            raise ConfigurationError(
                "This numerical resource is externally backed. Reopen the project "
                "with external_provider=... ."
            )
        return provider

    @staticmethod
    def _json_roundtrip_external_ref(external_ref: Any) -> Any:
        """Validate and normalize an opaque reference through JSON round-tripping."""
        try:
            return json.loads(json.dumps(external_ref))
        except (TypeError, ValueError) as error:
            raise ConfigurationError(
                "external_ref must be JSON-serializable and survive JSON round-tripping."
            ) from error

    def _validate_external_geometry_matrix(self, matrix: Any, *, name: str) -> Matrix:
        if sparse.issparse(matrix):
            normalized: Matrix = sparse.csr_matrix(matrix)
        else:
            normalized = np.asarray(matrix)
            if normalized.ndim != 2:
                raise ValueError(
                    f"External geometry {name!r} must be a 2D matrix, got shape "
                    f"{normalized.shape}."
                )
        expected = len(self.database.list_units())
        if normalized.shape[0] != expected:
            raise ValueError(
                f"External geometry {name!r} returned {normalized.shape[0]} rows; "
                f"the GeCo document universe contains {expected}."
            )
        return normalized

    def register_external_geometry(
        self,
        *,
        name: str,
        external_ref: Any,
        supports_query: bool = False,
        public: bool = True,
        if_exists: str = "error",
    ) -> int:
        """Register a provider-backed geometry over the complete document universe.

        ``if_exists="reuse"`` makes restartable notebooks idempotent when an
        existing external geometry has the same opaque reference and declared
        capabilities. Conflicting registrations still fail explicitly.
        """
        if not self.is_external_backed:
            raise ConfigurationError(
                "External geometries may only be registered on a project created "
                "with GeometricCoder.create_external(...)."
            )
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("Geometry name may not be empty.")
        if if_exists not in {"error", "reuse"}:
            raise ValueError("if_exists must be 'error' or 'reuse'")
        ref = self._json_roundtrip_external_ref(external_ref)
        if self.has_geometry(clean_name):
            existing = self.database.get_geometry_by_name(clean_name)
            if if_exists == "reuse":
                compatible = (
                    str(existing.get("storage_kind", "local")) == "external"
                    and existing.get("external_ref") == ref
                    and bool(existing.get("supports_query")) == bool(supports_query)
                    and bool(existing.get("public")) == bool(public)
                )
                if compatible:
                    return int(existing["geometry_id"])
                raise ConfigurationError(
                    f"Geometry {clean_name!r} already exists but does not match the "
                    "requested external registration."
                )
            raise ConfigurationError(f"Geometry {clean_name!r} is already registered.")
        provider = self._require_external_provider()
        derived = self.database.list_observations(kinds=["span", "teaching_example"])
        if derived:
            raise ConfigurationError(
                "Additional external geometries must be registered before creating "
                "spans or teaching examples. GeCo does not backfill geometry "
                "vectors for already-created derived observations."
            )
        if supports_query and not callable(getattr(provider, "transform_query", None)):
            raise ConfigurationError(
                f"External geometry {clean_name!r} was declared queryable, but the "
                "provider does not implement transform_query()."
            )
        matrix = self._validate_external_geometry_matrix(
            provider.geometry_matrix(ref, self._ordered_user_keys()), name=clean_name
        )
        geometry_id = self.database.register_geometry(
            name=clean_name,
            class_name="external",
            modality=str(self.metadata["modality"]),
            public=bool(public),
            config={"type": "external"},
            dependency_names=[],
            matrix_path="",
            matrix_sparse=bool(sparse.issparse(matrix)),
            state_path=None,
            supports_query=bool(supports_query),
            storage_kind="external",
            external_ref=ref,
        )
        self.ensure_default_classifier_specs()
        return geometry_id

    def register_external_view(
        self,
        *,
        geometry_id: int,
        name: str,
        external_ref: Any,
        if_exists: str = "error",
    ) -> int:
        """Register provider-backed 2D coordinates for one parent geometry.

        ``if_exists="reuse"`` returns an existing matching external view.
        """
        if not self.is_external_backed:
            raise ConfigurationError(
                "External views may only be registered on an externally backed project."
            )
        self.database.get_geometry(int(geometry_id))
        clean_name = str(name).strip()
        if not clean_name:
            raise ValueError("View name may not be empty.")
        if if_exists not in {"error", "reuse"}:
            raise ValueError("if_exists must be 'error' or 'reuse'")
        ref = self._json_roundtrip_external_ref(external_ref)
        existing = next(
            (record for record in self.views() if str(record["name"]) == clean_name),
            None,
        )
        if existing is not None:
            if if_exists == "reuse":
                compatible = (
                    str(existing.get("storage_kind", "local")) == "external"
                    and int(existing["geometry_id"]) == int(geometry_id)
                    and existing.get("external_ref") == ref
                )
                if compatible:
                    return int(existing["view_id"])
                raise ConfigurationError(
                    f"View {clean_name!r} already exists but does not match the "
                    "requested external registration."
                )
            raise ConfigurationError(f"View {clean_name!r} is already registered.")
        provider = self._require_external_provider()
        coordinates = np.asarray(
            provider.view_coordinates(ref, self._ordered_user_keys()), dtype=np.float32
        )
        expected = len(self.database.list_units())
        if coordinates.shape != (expected, 2):
            raise ValueError(
                f"External view {clean_name!r} must have shape ({expected}, 2), "
                f"got {coordinates.shape}."
            )
        return self.database.register_view(
            geometry_id=int(geometry_id),
            name=clean_name,
            method="external",
            parameters={},
            artifact_path="",
            state_path=None,
            supports_transform=False,
            storage_kind="external",
            external_ref=ref,
        )

    def geometry_matrix(self, geometry: int | str) -> Matrix:
        """Load one local or provider-backed geometry matrix by ID or name."""
        record = (
            self.database.get_geometry(geometry)
            if isinstance(geometry, int)
            else self.database.get_geometry_by_name(geometry)
        )
        if str(record.get("storage_kind", "local")) == "external":
            provider = self._require_external_provider()
            matrix = provider.geometry_matrix(
                record["external_ref"], self._ordered_user_keys()
            )
            return self._validate_external_geometry_matrix(
                matrix, name=str(record["name"])
            )
        return self.artifacts.load_matrix(
            self.project_dir,
            record["matrix_path"],
            record["matrix_sparse"],
        )

    def ingest(
        self,
        data: pd.DataFrame,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Append already-preprocessed atomic observations to this project.

        Incremental ingestion is intentionally strict. ``data`` must contain
        exactly the same key, text, and metadata columns as the original project.
        GeCo transforms only the new rows through the frozen geometry and view
        pipelines; it never refits the existing representation during ingestion.
        """
        if self.is_external_backed:
            raise ConfigurationError(
                "Incremental ingestion is disabled for externally backed GeCo projects. "
                "Their document universe is owned by the external source."
            )
        metadata = self.metadata
        key_columns = [str(value) for value in metadata["key_columns"]]
        text_column = str(metadata["text_column"])
        metadata_columns = [str(value) for value in metadata["metadata_columns"]]
        expected_columns = [*key_columns, text_column, *metadata_columns]

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if data.columns.duplicated().any():
            duplicates = data.columns[data.columns.duplicated()].tolist()
            raise ConfigurationError(f"Ingestion columns must be unique: {duplicates}")
        observed_columns = [str(value) for value in data.columns]
        if set(observed_columns) != set(expected_columns) or len(observed_columns) != len(
            expected_columns
        ):
            missing = [value for value in expected_columns if value not in observed_columns]
            extra = [value for value in observed_columns if value not in expected_columns]
            raise ConfigurationError(
                "Incremental ingestion requires the project's exact input columns. "
                f"Expected {expected_columns}; missing={missing}; extra={extra}."
            )
        if data.empty:
            raise ConfigurationError("Incremental ingestion requires at least one row.")

        # Reuse creation-time validation for missing text and reserved-column rules,
        # then normalize column order to the project's original schema.
        self._validate_input(
            data=data,
            keys=key_columns,
            text=text_column,
            modality=str(metadata["modality"]),
            metadata=metadata_columns,
        )
        incoming = data.loc[:, expected_columns].reset_index(drop=True)
        texts = incoming[text_column].astype(str).tolist()

        views = self.database.list_views()
        non_transformable = [str(view["name"]) for view in views if not view["supports_transform"]]
        if non_transformable:
            raise ConfigurationError(
                "Incremental ingestion cannot update views that lack a frozen transform: "
                f"{non_transformable}. Recreate or remove those views before ingesting."
            )

        runtime = self.runtime_geometries()
        geometry_records = self.database.list_geometries(public_only=False)
        token = uuid.uuid4().hex[:12]
        new_rows_by_geometry: dict[int, Matrix] = {}
        combined_by_geometry: dict[int, Matrix] = {}
        geometry_artifacts: dict[int, tuple[str, bool]] = {}
        old_artifact_paths: list[str] = []

        total_geometries = len(geometry_records)
        for index, record in enumerate(geometry_records, start=1):
            geometry_id = int(record["geometry_id"])
            name = str(record["name"])
            emit_progress(
                progress,
                phase="ingest",
                message=f"Transforming new observations: {name}",
                current=index,
                total=total_geometries,
                detail=f"{len(incoming):,} new rows",
            )
            new_rows = runtime[name].transform_texts(texts)
            old_matrix = self.geometry_matrix(geometry_id)
            if sparse.issparse(old_matrix) or sparse.issparse(new_rows):
                combined: Matrix = sparse.vstack(
                    [sparse.csr_matrix(old_matrix), sparse.csr_matrix(new_rows)],
                    format="csr",
                )
            else:
                combined = np.vstack([np.asarray(old_matrix), np.asarray(new_rows)])
            new_rows_by_geometry[geometry_id] = new_rows
            combined_by_geometry[geometry_id] = combined
            new_path, is_sparse = self.artifacts.save_geometry_matrix(
                f"{name}_ingest_{token}", combined
            )
            geometry_artifacts[geometry_id] = (new_path, is_sparse)
            old_artifact_paths.append(str(record["matrix_path"]))

        view_artifacts: dict[int, str] = {}
        for index, view in enumerate(views, start=1):
            view_id = int(view["view_id"])
            geometry_id = int(view["geometry_id"])
            emit_progress(
                progress,
                phase="projection",
                message=f"Placing new observations in view: {view['name']}",
                current=index,
                total=len(views),
            )
            state = self.artifacts.load_view_state(self.project_dir, view["state_path"])
            new_coordinates = transform_view(new_rows_by_geometry[geometry_id], state)
            old_coordinates = self.view_coordinates(view_id)
            combined_coordinates = np.vstack([old_coordinates, new_coordinates])
            new_path = self.artifacts.save_view(
                f"{view['name']}_ingest_{token}", combined_coordinates
            )
            view_artifacts[view_id] = new_path
            old_artifact_paths.append(str(view["artifact_path"]))

        emit_progress(
            progress,
            phase="database",
            message="Appending new observations to the project",
            detail=f"{len(incoming):,} rows",
        )
        try:
            appended = self.database.append_dataframe(
                incoming,
                key_columns=key_columns,
                text_column=text_column,
                metadata_columns=metadata_columns,
                geometry_artifacts=geometry_artifacts,
                view_artifacts=view_artifacts,
            )
        except BaseException:
            for relative_path in [
                *[path for path, _ in geometry_artifacts.values()],
                *view_artifacts.values(),
            ]:
                try:
                    (self.project_dir / relative_path).unlink(missing_ok=True)
                except OSError:
                    pass
            raise

        # The database now points at the new complete artifacts. Attach them to
        # the restored runtime pipelines and remove superseded matrix coordinates.
        for record in geometry_records:
            geometry_id = int(record["geometry_id"])
            runtime[str(record["name"])].restore_matrix(combined_by_geometry[geometry_id])
        self._runtime_geometries = runtime
        active_paths = {
            path for path, _ in geometry_artifacts.values()
        } | set(view_artifacts.values())
        for relative_path in old_artifact_paths:
            if relative_path in active_paths:
                continue
            try:
                (self.project_dir / relative_path).unlink(missing_ok=True)
            except OSError:
                # Stale artifacts are harmless; never fail an otherwise successful
                # ingestion merely because the OS retained an old file handle.
                pass

        emit_progress(
            progress,
            phase="complete",
            message="Incremental ingestion complete",
            current=1,
            total=1,
            detail=f"{len(appended):,} atomic observations added",
        )
        return {
            "added": len(appended),
            "unit_ids": [int(row["unit_id"]) for row in appended],
            "observation_ids": [int(row["observation_id"]) for row in appended],
        }

    def view_coordinates(self, view_id: int) -> np.ndarray:
        """Load one local or provider-backed two-dimensional view."""
        view = self.database.get_view(view_id)
        if str(view.get("storage_kind", "local")) == "external":
            provider = self._require_external_provider()
            coordinates = np.asarray(
                provider.view_coordinates(view["external_ref"], self._ordered_user_keys()),
                dtype=np.float32,
            )
            expected = len(self.database.list_units())
            if coordinates.shape != (expected, 2):
                raise ValueError(
                    f"External view {view['name']!r} must have shape ({expected}, 2), "
                    f"got {coordinates.shape}."
                )
            return coordinates
        return self.artifacts.load_view(self.project_dir, view["artifact_path"])

    def create_view(
        self,
        *,
        geometry_id: int,
        name: str,
        method: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> int:
        """Compute, persist, and register a new two-dimensional view."""
        if self.is_external_backed:
            raise ConfigurationError(
                "Local Create 2D view is disabled for externally backed projects. "
                "Create the view in the external system and register it with "
                "register_external_view()."
            )
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("View name may not be empty.")
        self.database.get_geometry(int(geometry_id))
        matrix = self.geometry_matrix(int(geometry_id))
        parameter_dict = dict(parameters or {})
        parameter_dict.pop("n_components", None)
        fitted_view = fit_view(
            matrix,
            ViewSpec(method=method, parameters=parameter_dict),
        )
        artifact_path = self.artifacts.save_view(clean_name, fitted_view.coordinates)
        state_path = self.artifacts.save_view_state(clean_name, fitted_view.state)
        return self.database.register_view(
            geometry_id=int(geometry_id),
            name=clean_name,
            method=method,
            parameters=parameter_dict,
            artifact_path=artifact_path,
            state_path=state_path,
            supports_transform=fitted_view.supports_transform,
        )

    def runtime_geometries(self) -> dict[str, Geometry]:
        """Restore fitted geometry pipelines for query transformation."""
        if self._runtime_geometries is not None:
            return dict(self._runtime_geometries)

        restored: dict[str, Geometry] = {}
        for record in self.database.list_geometries(public_only=False):
            if str(record.get("storage_kind", "local")) == "external":
                continue
            config = dict(record["config"])
            geometry_type = config.pop("type", None)
            dependency_names = list(record["dependency_names"])
            if geometry_type == "count":
                geometry: Geometry = CountGeometry(**config)
            elif geometry_type == "sentence_transformer":
                model = str(config.pop("model"))
                geometry = SentenceTransformerGeometry(model, **config)
            elif geometry_type == "svd":
                if len(dependency_names) != 1:
                    raise ConfigurationError(
                        f"SVD geometry {record['name']!r} must have one dependency."
                    )
                source_name = dependency_names[0]
                try:
                    source = restored[source_name]
                except KeyError:
                    raise ConfigurationError(
                        f"Geometry dependency {source_name!r} was not restored before "
                        f"{record['name']!r}."
                    ) from None
                geometry = SVDGeometry(source=source, **config)
            else:
                raise ConfigurationError(
                    f"Cannot restore unsupported geometry type {geometry_type!r} "
                    f"for {record['name']!r}."
                )

            matrix = self.artifacts.load_matrix(
                self.project_dir,
                record["matrix_path"],
                record["matrix_sparse"],
            )
            state = self.artifacts.load_geometry_state(
                self.project_dir,
                record["state_path"],
            )
            if state:
                geometry.load_artifact_state(state)
            geometry.restore_matrix(matrix)
            restored[record["name"]] = geometry

        self._runtime_geometries = restored
        return dict(restored)

    def transform_query(self, geometry: int | str, query: str) -> Matrix:
        """Transform one semantic query through a registered geometry.

        The caller names the GeCo geometry; GeCo decides whether to use the
        local frozen pipeline or delegate through the runtime external provider.
        Opaque external references never leave this boundary.
        """
        clean_query = str(query).strip()
        if not clean_query:
            raise ValueError("Query may not be empty.")
        record = (
            self.database.get_geometry(geometry)
            if isinstance(geometry, int)
            else self.database.get_geometry_by_name(str(geometry))
        )
        name = str(record["name"])
        if not bool(record.get("supports_query")):
            raise ConfigurationError(
                f"Geometry {name!r} does not support semantic query transformation."
            )
        if str(record.get("storage_kind", "local")) == "external":
            provider = self._require_external_provider()
            transform = getattr(provider, "transform_query", None)
            if not callable(transform):
                raise ConfigurationError(
                    f"External geometry {name!r} supports semantic queries, but the "
                    "connected provider does not implement transform_query()."
                )
            query_vector = transform(record["external_ref"], clean_query)
        else:
            runtime = self.runtime_geometries()
            query_vector = runtime[name].transform_query(clean_query)
        if sparse.issparse(query_vector):
            normalized: Matrix = sparse.csr_matrix(query_vector)
        else:
            normalized = np.asarray(query_vector)
        if normalized.ndim != 2 or normalized.shape[0] != 1:
            raise ValueError(
                f"Query transform for geometry {name!r} must return shape (1, m), "
                f"got {normalized.shape}."
            )
        matrix = self.geometry_matrix(int(record["geometry_id"]))
        if matrix.shape[1] != normalized.shape[1]:
            raise ValueError(
                f"Query transform for geometry {name!r} returned "
                f"{normalized.shape[1]} features, expected {matrix.shape[1]}."
            )
        return normalized

    def semantic_search(
        self,
        query: str,
        *,
        geometry_names: Sequence[str] | None = None,
    ) -> dict[str, list[float]]:
        """Compute cosine similarity in every selected queryable geometry."""
        clean_query = query.strip()
        if not clean_query:
            return {}
        public_records = self.database.list_geometries(public_only=True)
        selected = set(geometry_names) if geometry_names is not None else None
        results: dict[str, list[float]] = {}
        for record in public_records:
            name = str(record["name"])
            if selected is not None and name not in selected:
                continue
            if not record["supports_query"]:
                continue
            matrix = self.geometry_matrix(int(record["geometry_id"]))
            normalized_query = self.transform_query(int(record["geometry_id"]), clean_query)
            scores = cosine_similarity(matrix, normalized_query).reshape(-1)
            results[name] = np.asarray(scores, dtype=float).tolist()
        return results

    def units(self) -> list[dict[str, Any]]:
        """Return all atomic units in canonical order."""
        return self.database.list_units()

    def hierarchy_summary(self) -> list[dict[str, Any]]:
        """Return hierarchy labels and full-prefix group counts."""
        return self.database.hierarchy_level_counts()


    def unit(self, unit_id: int) -> dict[str, Any]:
        """Return one atomic unit."""
        return self.database.get_unit(unit_id)

    def context(self, unit_id: int, *, level: int, window: int = 1) -> list[dict[str, Any]]:
        """Return generic hierarchy-aware context around a focal unit."""
        return self.database.context_units(unit_id=unit_id, level_index=level, window=window)

    def observation(self, observation_id: int) -> dict[str, Any]:
        """Return one generalized codable observation."""
        return self.database.get_observation(observation_id)

    def observation_id_for_unit(self, unit_id: int) -> int:
        """Return the generalized observation ID for an atomic unit."""
        return self.database.observation_for_unit(unit_id)

    def span_observation(self, member_unit_ids: Sequence[int]) -> dict[str, Any] | None:
        """Return a persistent span with an exact ordered signature, if any."""
        return self.database.find_span(member_unit_ids)

    def observation_vector(self, observation_id: int, geometry_id: int) -> Matrix:
        """Load one observation row in a selected geometry."""
        observation = self.database.get_observation(observation_id)
        if observation["kind"] == "atomic":
            unit_id = int(observation["unit_id"])
            row_position = int(self.database.get_unit(unit_id)["row_position"])
            return self.geometry_matrix(geometry_id)[row_position : row_position + 1]
        record = self.database.observation_vector_record(observation_id, geometry_id)
        if record is None:
            raise KeyError(
                f"Observation {observation_id} has no vector for geometry {geometry_id}."
            )
        return self.artifacts.load_matrix(
            self.project_dir, record["artifact_path"], record["matrix_sparse"]
        )

    def observation_view_coordinates(
        self, observation_id: int, view_id: int
    ) -> np.ndarray:
        """Place one atomic or derived observation in a persisted two-dimensional view."""
        observation = self.database.get_observation(observation_id)
        view = self.database.get_view(view_id)
        if observation["kind"] == "atomic":
            row_position = int(observation["row_position"])
            return self.view_coordinates(view_id)[row_position : row_position + 1]
        if not view["supports_transform"]:
            raise ValueError(f"View {view_id} cannot transform derived observations.")
        vector = self.observation_vector(observation_id, int(view["geometry_id"]))
        state = self.artifacts.load_view_state(self.project_dir, view["state_path"])
        return transform_view(vector, state)

    def can_transform_new_observations(self) -> bool:
        """Whether every registered geometry can vectorize newly authored text."""
        if self.is_external_backed:
            if not self.database.list_geometries(public_only=False):
                return False
            provider = self.external_provider
            return provider is not None and callable(
                getattr(provider, "transform_texts", None)
            )
        external = [
            row for row in self.database.list_geometries(public_only=False)
            if str(row.get("storage_kind", "local")) == "external"
        ]
        if not external:
            return True
        provider = self.external_provider
        return provider is not None and callable(getattr(provider, "transform_texts", None))

    def _transform_observation_text(
        self, text: str, *, token: str
    ) -> dict[int, tuple[str, bool]]:
        """Transform and persist a new observation through every compatible geometry."""
        if not self.can_transform_new_observations():
            raise ConfigurationError(
                "This externally backed project cannot create spans or teaching examples "
                "because its provider does not implement transform_texts()."
            )
        runtime = self.runtime_geometries()
        vectors: dict[int, tuple[str, bool]] = {}
        for record in self.database.list_geometries(public_only=False):
            name = str(record["name"])
            if str(record.get("storage_kind", "local")) == "external":
                provider = self._require_external_provider()
                transform = getattr(provider, "transform_texts")
                vector = transform(record["external_ref"], [text])
                if sparse.issparse(vector):
                    normalized: Matrix = sparse.csr_matrix(vector)
                else:
                    normalized = np.asarray(vector)
                corpus = self.geometry_matrix(int(record["geometry_id"]))
                if normalized.ndim != 2 or normalized.shape != (1, corpus.shape[1]):
                    raise ValueError(
                        f"transform_texts for external geometry {name!r} must return "
                        f"shape (1, {corpus.shape[1]}), got {normalized.shape}."
                    )
                vector = normalized
            else:
                geometry = runtime[name]
                vector = geometry.transform_texts([text])
            vectors[int(record["geometry_id"])] = self.artifacts.save_observation_vector(
                token, name, vector
            )
        return vectors

    def annotate_span(
        self,
        member_unit_ids: Sequence[int],
        code_id: int,
        value: str,
        *,
        origin: str = "human_direct",
        classifier_fit_id: int | None = None,
    ) -> tuple[int, int]:
        """Assign a contiguous interval, lazily creating a span when needed."""
        members = [int(value) for value in member_unit_ids]
        if not members:
            raise ValueError("Select at least one atomic observation.")
        if len(members) == 1:
            observation_id = self.database.observation_for_unit(members[0])
            return observation_id, self.database.annotate(
                observation_id=observation_id, code_id=code_id, value=value,
                origin=origin, classifier_fit_id=classifier_fit_id,
            )
        existing = self.database.find_span(members)
        if existing is not None:
            observation_id = int(existing["observation_id"])
            return observation_id, self.database.annotate(
                observation_id=observation_id, code_id=code_id, value=value,
                origin=origin, classifier_fit_id=classifier_fit_id,
            )
        self.database.validate_span_members(members)
        selected = [self.database.get_unit(unit_id) for unit_id in members]
        text = "\n".join(str(unit["text"]) for unit in selected)
        token = f"span_{uuid.uuid4().hex}"
        vectors = self._transform_observation_text(text, token=token)
        return self.database.create_span_with_annotation(
            member_unit_ids=members, text=text, code_id=code_id, value=value,
            origin=origin, geometry_vectors=vectors, classifier_fit_id=classifier_fit_id,
        )

    def create_teaching_example(
        self,
        *,
        code_id: int,
        text: str,
        label: str,
        note: str = "",
        assess_before_training: bool = False,
    ) -> int:
        """Create one active teaching example, optionally probing current fits first."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Teaching-example text may not be empty.")
        if label not in {"positive", "negative"}:
            raise ValueError("Teaching-example label must be positive or negative.")
        token = f"teaching_{uuid.uuid4().hex}"
        vectors = self._transform_observation_text(clean_text, token=token)
        latest_by_spec: dict[int, dict[str, Any]] = {}
        if assess_before_training:
            for spec in self.database.list_classifier_specs(code_id=int(code_id)):
                fit = self.database.latest_classifier_fit(
                    classifier_spec_id=int(spec["classifier_spec_id"]),
                    code_id=int(code_id),
                )
                if fit is not None:
                    latest_by_spec[int(spec["classifier_spec_id"])] = fit
        observation_id = self.database.create_teaching_example(
            code_id=code_id, text=clean_text, label=label, note=note,
            geometry_vectors=vectors,
        )
        for fit in latest_by_spec.values():
            vector = self.observation_vector(
                observation_id, int(fit["geometry_id"])
            )
            estimator = self.artifacts.load_model(
                self.project_dir, str(fit["artifact_path"])
            )
            outputs = score_classifier(
                estimator=estimator,
                algorithm=str(fit["algorithm"]),
                hyperparameters=dict(fit["hyperparameters"]),
                matrix=vector,
            )
            probability = (
                float(outputs.probabilities[0])
                if outputs.probabilities is not None
                else None
            )
            decision_score = (
                float(outputs.decision_scores[0])
                if outputs.decision_scores is not None
                else None
            )
            self.database.register_classifier_evaluation(
                event_type="teaching_example_pretraining",
                code_id=int(code_id),
                classifier_fit_id=int(fit["classifier_fit_id"]),
                observation_id=observation_id,
                human_label=label,
                predicted_label=(
                    "positive" if int(outputs.predicted_labels[0]) == 1 else "negative"
                ),
                probability=probability,
                decision_score=decision_score,
                added_to_training=True,
            )
        return observation_id

    def teaching_examples(
        self, code_id: int, *, include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """List teaching examples for one code."""
        return self.database.list_teaching_examples(
            code_id, include_deleted=include_deleted
        )

    def set_teaching_example_status(self, observation_id: int, status: str) -> None:
        """Activate, deactivate, or soft-delete a teaching example."""
        self.database.set_teaching_example_status(observation_id, status)

    def set_teaching_example_label(self, observation_id: int, label: str) -> int:
        """Switch a teaching example between positive and negative."""
        return self.database.set_teaching_example_label(observation_id, label)

    def create_session(self, title: str, state: dict[str, Any] | None = None) -> int:
        """Create a named resumable navigation session."""
        return self.database.create_session(title, state)

    def record_visit(
        self,
        session_id: int,
        unit_id: int,
        *,
        method: str,
        source_unit_id: int | None = None,
        geometry_id: int | None = None,
        active_code_id: int | None = None,
        classifier_fit_id: int | None = None,
    ) -> int:
        """Record that a unit became focal in a session."""
        return self.database.record_visit(
            session_id=session_id,
            unit_id=unit_id,
            navigation_method=method,
            source_unit_id=source_unit_id,
            geometry_id=geometry_id,
            active_code_id=active_code_id,
            classifier_fit_id=classifier_fit_id,
        )

    def create_code(self, name: str, definition: str = "") -> int:
        """Create a code, then provision its own default classifiers."""
        code_id = self.database.create_code(name, definition)
        self.ensure_default_classifier_specs(code_id=code_id)
        return code_id

    def update_code(self, code_id: int, *, name: str, description: str) -> int:
        """Append a new label/description version for a code."""
        return self.database.update_code(code_id, name=name, description=description)

    def code(self, code_id: int, version_number: int | None = None) -> dict[str, Any]:
        """Return one code version, defaulting to the latest."""
        return self.database.get_code(code_id, version_number)

    def code_versions(self, code_id: int) -> list[dict[str, Any]]:
        """Return every immutable version of one code."""
        return self.database.code_versions(code_id)

    def code_hashtags(self) -> list[str]:
        """Return hashtags from current code descriptions."""
        return self.database.code_hashtags()

    def search_codes(
        self, *, query: str = "", scope: str = "full", hashtags: Sequence[str] = ()
    ) -> list[dict[str, Any]]:
        """Search current code labels/descriptions."""
        return self.database.search_codes(query=query, scope=scope, hashtags=hashtags)

    def positive_observations_for_code(self, code_id: int) -> list[dict[str, Any]]:
        """Return all observations currently labeled positive for one code."""
        return self.database.positive_observations_for_code(code_id)

    def positive_units_for_code(self, code_id: int) -> list[dict[str, Any]]:
        """Return atomic observations currently labeled positive for one code."""
        return self.database.positive_units_for_code(code_id)

    def annotate(
        self,
        observation_id: int,
        code_id: int,
        value: str,
        *,
        origin: str = "human_direct",
        classifier_fit_id: int | None = None,
    ) -> int:
        """Append an annotation event."""
        return self.database.annotate(
            observation_id=observation_id,
            code_id=code_id,
            value=value,
            origin=origin,
            classifier_fit_id=classifier_fit_id,
        )

    def codes(self) -> list[dict[str, Any]]:
        """Return all project codes."""
        return self.database.list_codes()

    def current_annotation(
        self, *, observation_id: int, code_id: int
    ) -> dict[str, Any] | None:
        """Return the current assignment for one observation-code pair."""
        return self.database.current_annotation(
            observation_id=int(observation_id), code_id=int(code_id)
        )

    def current_annotations(self, code_id: int | None = None) -> list[dict[str, Any]]:
        """Return current annotation states."""
        return self.database.current_annotations(code_id)

    def export_codes(self, code: int | str) -> pd.DataFrame:
        """Export current binary human-vetted atomic assignments with original keys."""
        if isinstance(code, int):
            code_id = int(code)
            self.database.get_code(code_id)
        else:
            clean_name = str(code).strip()
            matches = [row for row in self.database.list_codes() if str(row["name"]) == clean_name]
            if not matches:
                raise KeyError(f"Unknown code name: {clean_name!r}")
            if len(matches) != 1:
                raise ValueError(
                    f"Code name {clean_name!r} is ambiguous; export by numeric code ID instead."
                )
            code_id = int(matches[0]["code_id"])
        key_columns = [str(value) for value in self.metadata["key_columns"]]
        rows: list[dict[str, Any]] = []
        for annotation in self.database.current_annotations(code_id):
            value = str(annotation["value"])
            unit_id = annotation["unit_id"]
            if unit_id is None or value not in {"positive", "negative"}:
                continue
            unit = self.database.get_unit(int(unit_id))
            record = {column: unit["user_key"].get(column) for column in key_columns}
            record["label"] = 1 if value == "positive" else 0
            rows.append(record)
        return pd.DataFrame(rows, columns=[*key_columns, "label"])

    def export_classifier(self, name: str) -> Any:
        """Return the current fitted estimator for one uniquely named active classifier."""
        spec = self.database.get_classifier_spec_by_name(str(name).strip())
        fit = self.database.latest_classifier_fit(
            code_id=int(spec["code_id"]),
            classifier_spec_id=int(spec["classifier_spec_id"]),
        )
        if fit is None or not str(fit.get("artifact_path") or ""):
            raise ValueError(f"Classifier {spec['name']!r} has not been trained.")
        return self.artifacts.load_model(self.project_dir, str(fit["artifact_path"]))

    def training_label_counts(self, code_id: int) -> dict[str, int]:
        """Count positive and negative labels currently eligible for training."""
        eligible = self._eligible_training_annotations(
            code_id, self.database.current_annotations(code_id)
        )
        return {
            "positive": sum(row["value"] == "positive" for row in eligible),
            "negative": sum(row["value"] == "negative" for row in eligible),
        }

    def _eligible_training_annotations(
        self,
        code_id: int,
        annotations: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return human labels eligible for one code's classifier training."""
        teaching_status = {
            int(example["observation_id"]): str(example["teaching_status"])
            for example in self.database.list_teaching_examples(
                code_id, include_deleted=True
            )
        }
        return [
            row
            for row in annotations
            if row["value"] in {"positive", "negative"}
            and teaching_status.get(int(row["observation_id"]), "active") == "active"
        ]

    def classifier_specs(
        self,
        code_id: int | None = None,
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """Return code-owned classifiers, optionally filtered to one code."""
        return self.database.list_classifier_specs(
            code_id=code_id, include_deleted=include_deleted
        )

    def create_classifier_spec(
        self,
        *,
        code_id: int,
        name: str,
        geometry_id: int,
        algorithm: str = "logistic_l2",
        hyperparameters: Mapping[str, Any] | None = None,
    ) -> int:
        """Create one classifier owned by exactly one code."""
        self.database.get_code(int(code_id))
        self.database.get_geometry(int(geometry_id))
        parameters = normalize_hyperparameters(algorithm, dict(hyperparameters or {}))
        return self.database.create_classifier_spec(
            code_id=int(code_id),
            name=name,
            geometry_id=int(geometry_id),
            algorithm=algorithm,
            hyperparameters=parameters,
        )

    def update_classifier_spec(
        self,
        classifier_spec_id: int,
        *,
        name: str,
        hyperparameters: Mapping[str, Any],
    ) -> None:
        """Update one code-owned classifier for future explicit training."""
        spec = self.database.get_classifier_spec(int(classifier_spec_id))
        parameters = normalize_hyperparameters(
            str(spec["algorithm"]), dict(hyperparameters)
        )
        self.database.update_classifier_spec(
            int(classifier_spec_id), name=name, hyperparameters=parameters
        )

    def delete_classifier_spec(self, classifier_spec_id: int) -> None:
        """Archive a classifier, free its name, and discard its live fitted state."""
        spec_id = int(classifier_spec_id)
        self.database.delete_classifier_spec(spec_id)
        old_paths = self.database.retire_classifier_fit_states(spec_id)
        for relative_path in old_paths:
            try:
                (self.project_dir / relative_path).unlink(missing_ok=True)
            except OSError:
                pass

    def ensure_default_classifier_specs(
        self, code_id: int | None = None
    ) -> list[dict[str, Any]]:
        """Ensure each code owns one default logistic classifier per public geometry."""
        codes = (
            [self.database.get_code(int(code_id))]
            if code_id is not None
            else self.database.list_codes()
        )
        geometries = self.database.list_geometries(public_only=True)
        for code in codes:
            current_code_id = int(code["code_id"])
            existing = self.database.list_classifier_specs(
                code_id=current_code_id, include_deleted=True
            )
            covered = {int(row["geometry_id"]) for row in existing}
            for geometry in geometries:
                geometry_id = int(geometry["geometry_id"])
                if geometry_id in covered:
                    continue
                base_name = f"{code['name']} · {geometry['name']} logistic"
                active_names = {
                    str(row["name"]) for row in self.database.list_classifier_specs()
                }
                name = (
                    base_name
                    if base_name not in active_names
                    else f"{base_name} [code {current_code_id}]"
                )
                self.create_classifier_spec(
                    code_id=current_code_id,
                    name=name,
                    geometry_id=geometry_id,
                    algorithm="logistic_l2",
                    hyperparameters={"regularization": 1.0},
                )
        return self.database.list_classifier_specs(
            code_id=code_id, include_deleted=False
        )

    def classifier_committees(
        self, code_id: int | None = None
    ) -> list[dict[str, Any]]:
        """Return intentional persisted classifier committees."""
        return self.database.list_classifier_committees(code_id=code_id)

    def create_classifier_committee(
        self,
        *,
        code_id: int,
        name: str,
        classifier_spec_ids: Sequence[int],
        aggregation: str = "mean",
    ) -> int:
        """Create a named committee for one code."""
        self.database.get_code(int(code_id))
        for classifier_spec_id in classifier_spec_ids:
            spec = self.database.get_classifier_spec(int(classifier_spec_id))
            if int(spec["code_id"]) != int(code_id):
                raise ValueError("A committee may contain only classifiers owned by its code.")
        return self.database.create_classifier_committee(
            code_id=int(code_id),
            name=name,
            classifier_spec_ids=classifier_spec_ids,
            aggregation=aggregation,
        )

    def update_classifier_committee(
        self,
        committee_id: int,
        *,
        name: str,
        classifier_spec_ids: Sequence[int],
        aggregation: str,
    ) -> None:
        """Replace one committee definition."""
        committee = self.database.get_classifier_committee(int(committee_id))
        code_id = int(committee["code_id"])
        for classifier_spec_id in classifier_spec_ids:
            spec = self.database.get_classifier_spec(int(classifier_spec_id))
            if int(spec["code_id"]) != code_id:
                raise ValueError("A committee may contain only classifiers owned by its code.")
        self.database.update_classifier_committee(
            int(committee_id),
            name=name,
            classifier_spec_ids=classifier_spec_ids,
            aggregation=aggregation,
        )

    def delete_classifier_committee(self, committee_id: int) -> None:
        """Delete a committee definition without deleting its historical fits."""
        self.database.delete_classifier_committee(int(committee_id))

    def _classifier_training_snapshot(
        self, code_id: int, annotations: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        """Return the reproducible training state for one code."""
        labeled = self._eligible_training_annotations(code_id, annotations)
        return {
            "annotations": [
                {
                    "annotation_event_id": int(row["annotation_event_id"]),
                    "observation_id": int(row["observation_id"]),
                    "value": str(row["value"]),
                }
                for row in labeled
            ],
            "teaching_status": [
                {
                    "observation_id": int(example["observation_id"]),
                    "status": str(example["teaching_status"]),
                }
                for example in self.database.list_teaching_examples(
                    code_id, include_deleted=True
                )
            ],
        }

    def _classifier_scoring_matrix(
        self, geometry_id: int
    ) -> tuple[list[int], Matrix]:
        """Return all current observations and their vectors for one geometry."""
        observations = self.database.list_observations()
        atomic = [row for row in observations if row["kind"] == "atomic"]
        derived = [row for row in observations if row["kind"] != "atomic"]
        observation_ids = [
            *[int(row["observation_id"]) for row in atomic],
            *[int(row["observation_id"]) for row in derived],
        ]
        corpus_matrix = self.geometry_matrix(int(geometry_id))
        if len(atomic) != corpus_matrix.shape[0]:
            raise RuntimeError("Atomic observation order no longer matches geometry rows.")
        if not derived:
            return observation_ids, corpus_matrix
        derived_rows = [
            self.observation_vector(int(row["observation_id"]), int(geometry_id))
            for row in derived
        ]
        if sparse.issparse(corpus_matrix):
            return observation_ids, sparse.vstack(
                [corpus_matrix, *derived_rows], format="csr"
            )
        return observation_ids, np.vstack(
            [np.asarray(corpus_matrix), *[np.asarray(row) for row in derived_rows]]
        )

    def _classifier_training_matrix(
        self, *, code_id: int, geometry_id: int
    ) -> tuple[list[dict[str, Any]], Matrix, np.ndarray, dict[str, Any]]:
        """Return eligible evidence, vectors, labels, and the exact training snapshot."""
        annotations = self.database.current_annotations(int(code_id))
        labeled = self._eligible_training_annotations(int(code_id), annotations)
        if not any(row["value"] == "positive" for row in labeled) or not any(
            row["value"] == "negative" for row in labeled
        ):
            raise ValueError(
                "At least one active positive and one active negative example are required."
            )
        training_rows = [
            self.observation_vector(int(row["observation_id"]), int(geometry_id))
            for row in labeled
        ]
        if sparse.issparse(training_rows[0]):
            training_matrix: Matrix = sparse.vstack(training_rows, format="csr")
        else:
            training_matrix = np.vstack([np.asarray(row) for row in training_rows])
        labels = np.asarray(
            [1 if row["value"] == "positive" else 0 for row in labeled], dtype=int
        )
        snapshot = self._classifier_training_snapshot(int(code_id), annotations)
        return labeled, training_matrix, labels, snapshot

    def _select_classifier_hyperparameters(
        self,
        *,
        code_id: int,
        spec: Mapping[str, Any],
        training_matrix: Matrix,
        labels: np.ndarray,
        folds: int = 5,
        metric: str = "log_loss",
        persist_legacy_l2_run: bool = True,
    ) -> dict[str, Any]:
        """Select family-specific hyperparameters by compact stratified CV when feasible."""
        if metric not in {"log_loss", "brier", "accuracy"}:
            raise ValueError("metric must be log_loss, brier, or accuracy")
        class_counts = np.bincount(labels, minlength=2)
        base = effective_hyperparameters(
            str(spec["algorithm"]),
            dict(spec["hyperparameters"]),
            training_rows=int(labels.size),
        )
        actual_folds = min(int(folds), int(class_counts.min()))
        if actual_folds < 2:
            return {
                "tuned": False,
                "reason": "insufficient_cv_data",
                "folds": 0,
                "metric": metric,
                "selected_hyperparameters": base,
                "results": [],
            }

        splitter = StratifiedKFold(
            n_splits=actual_folds, shuffle=True, random_state=0
        )
        splits = list(splitter.split(training_matrix, labels))
        min_train_rows = min(len(train_positions) for train_positions, _ in splits)
        candidates = tuning_candidates(
            str(spec["algorithm"]),
            base,
            max_neighbors=int(min_train_rows),
        )
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            fold_scores: list[float] = []
            for train_positions, validation_positions in splits:
                fitted = fit_classifier(
                    algorithm=str(spec["algorithm"]),
                    hyperparameters=candidate,
                    training_matrix=training_matrix[train_positions],
                    labels=labels[train_positions],
                    scoring_matrix=training_matrix[validation_positions],
                )
                validation_labels = labels[validation_positions]
                if metric == "log_loss":
                    probabilities = fitted.outputs.probabilities
                    if probabilities is None:
                        raise ValueError(
                            "Automatic CV with log loss requires classifier probabilities."
                        )
                    score = float(
                        log_loss(validation_labels, probabilities, labels=[0, 1])
                    )
                elif metric == "brier":
                    probabilities = fitted.outputs.probabilities
                    if probabilities is None:
                        raise ValueError(
                            "Automatic CV with Brier score requires classifier probabilities."
                        )
                    score = float(brier_score_loss(validation_labels, probabilities))
                else:
                    score = float(
                        accuracy_score(
                            validation_labels, fitted.outputs.predicted_labels
                        )
                    )
                fold_scores.append(score)
            results.append(
                {
                    "hyperparameters": dict(candidate),
                    "score": float(np.mean(fold_scores)),
                    "fold_scores": fold_scores,
                }
            )

        # Candidate order is deliberate and acts as the stable tie-breaker,
        # preferring the earlier/simpler search-grid choice when CV scores tie.
        if metric == "accuracy":
            selected = max(results, key=lambda row: float(row["score"]))
        else:
            selected = min(results, key=lambda row: float(row["score"]))
        selected_parameters = effective_hyperparameters(
            str(spec["algorithm"]),
            dict(selected["hyperparameters"]),
            training_rows=int(labels.size),
        )

        # Preserve the schema-12 lambda-tuning audit trail for the existing L2
        # family. Other families retain their selected parameters in the classifier
        # configuration and fit-time snapshot without introducing a schema change.
        tuning_run_id: int | None = None
        if str(spec["algorithm"]) == "logistic_l2" and persist_legacy_l2_run:
            tuning_run_id = self.database.register_classifier_tuning_run(
                code_id=int(code_id),
                classifier_spec_id=int(spec["classifier_spec_id"]),
                metric=metric,
                folds=actual_folds,
                candidate_lambdas=[
                    float(row["hyperparameters"]["regularization"]) for row in results
                ],
                results=[
                    {
                        "regularization": float(
                            row["hyperparameters"]["regularization"]
                        ),
                        "score": float(row["score"]),
                        "fold_scores": list(row["fold_scores"]),
                    }
                    for row in results
                ],
                selected_lambda=float(selected_parameters["regularization"]),
            )

        return {
            "tuned": True,
            "reason": None,
            "folds": actual_folds,
            "metric": metric,
            "selected_hyperparameters": selected_parameters,
            "results": results,
            "tuning_run_id": tuning_run_id,
        }

    def train_classifier(
        self,
        *,
        code_id: int,
        classifier_spec_id: int,
        tune: bool = True,
        folds: int = 5,
        metric: str = "log_loss",
    ) -> dict[str, Any]:
        """Produce the best current fitted state for one code-owned classifier.

        When enough evidence exists for stratified cross-validation, Train selects
        family-specific hyperparameters and then refits on all eligible current
        evidence. With too little evidence for CV, it fits the family's current/default
        parameters instead. A fit that is already current is reused without rerunning CV.
        """
        spec = self.database.get_classifier_spec(int(classifier_spec_id))
        if int(spec["code_id"]) != int(code_id):
            raise ValueError(
                f"Classifier {spec['name']!r} belongs to code {spec['code_id']}, "
                f"not code {code_id}."
            )

        labeled, training_matrix, labels, snapshot = self._classifier_training_matrix(
            code_id=int(code_id), geometry_id=int(spec["geometry_id"])
        )
        del labeled

        # Fast path: ordinary recommendation requests should not rerun CV when the
        # retained fit already matches both the evidence and current configuration.
        existing = self.database.find_classifier_fit(
            code_id=int(code_id),
            classifier_spec_id=int(classifier_spec_id),
            geometry_id=int(spec["geometry_id"]),
            algorithm=str(spec["algorithm"]),
            hyperparameters=dict(spec["hyperparameters"]),
            training_snapshot=snapshot,
        )
        if existing is not None:
            self._extend_classifier_fit_predictions(existing)
            result = dict(existing)
            result["training_selection"] = {
                "tuned": False, "reason": "current_fit_reused", "folds": 0
            }
            return result

        if tune:
            selection = self._select_classifier_hyperparameters(
                code_id=int(code_id),
                spec=spec,
                training_matrix=training_matrix,
                labels=labels,
                folds=int(folds),
                metric=str(metric),
            )
            selected_parameters = dict(selection["selected_hyperparameters"])
        else:
            selected_parameters = effective_hyperparameters(
                str(spec["algorithm"]),
                dict(spec["hyperparameters"]),
                training_rows=int(labels.size),
            )
            selection = {
                "tuned": False,
                "reason": "tuning_disabled",
                "folds": 0,
                "metric": metric,
                "selected_hyperparameters": selected_parameters,
                "results": [],
            }

        if selected_parameters != dict(spec["hyperparameters"]):
            self.update_classifier_spec(
                int(classifier_spec_id),
                name=str(spec["name"]),
                hyperparameters=selected_parameters,
            )
            spec = self.database.get_classifier_spec(int(classifier_spec_id))

        # A previously retained state may already match the CV-selected parameters.
        existing = self.database.find_classifier_fit(
            code_id=int(code_id),
            classifier_spec_id=int(classifier_spec_id),
            geometry_id=int(spec["geometry_id"]),
            algorithm=str(spec["algorithm"]),
            hyperparameters=selected_parameters,
            training_snapshot=snapshot,
        )
        if existing is not None:
            self._extend_classifier_fit_predictions(existing)
            result = dict(existing)
            result["training_selection"] = selection
            return result

        geometry_id = int(spec["geometry_id"])
        observation_ids, scoring_matrix = self._classifier_scoring_matrix(geometry_id)
        fitted = fit_classifier(
            algorithm=str(spec["algorithm"]),
            hyperparameters=selected_parameters,
            training_matrix=training_matrix,
            labels=labels,
            scoring_matrix=scoring_matrix,
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "code_id": int(code_id),
                    "classifier_spec_id": int(classifier_spec_id),
                    "algorithm": spec["algorithm"],
                    "hyperparameters": selected_parameters,
                    "snapshot": snapshot,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        artifact_path = self.artifacts.save_model(
            f"classifier_{classifier_spec_id}_{fingerprint}",
            fitted.estimator,
        )
        outputs = fitted.outputs
        predictions = [
            {
                "observation_id": observation_id,
                "predicted_label": (
                    "positive"
                    if int(outputs.predicted_labels[position]) == 1
                    else "negative"
                ),
                "probability": (
                    float(outputs.probabilities[position])
                    if outputs.probabilities is not None
                    else None
                ),
                "decision_score": (
                    float(outputs.decision_scores[position])
                    if outputs.decision_scores is not None
                    else None
                ),
                "uncertainty": float(outputs.uncertainty[position]),
            }
            for position, observation_id in enumerate(observation_ids)
        ]
        classifier_fit_id = self.database.register_classifier_fit(
            code_id=int(code_id),
            classifier_spec_id=int(classifier_spec_id),
            classifier_name=str(spec["name"]),
            geometry_id=geometry_id,
            algorithm=str(spec["algorithm"]),
            hyperparameters=selected_parameters,
            training_snapshot=snapshot,
            artifact_path=artifact_path,
            score_kind=outputs.score_kind,
            predictions=predictions,
        )
        superseded_paths = self.database.retire_classifier_fit_states(
            int(classifier_spec_id), keep_classifier_fit_id=classifier_fit_id
        )
        for relative_path in superseded_paths:
            try:
                (self.project_dir / relative_path).unlink(missing_ok=True)
            except OSError:
                pass
        result = self.database.get_classifier_fit(classifier_fit_id)
        result["training_selection"] = selection
        return result

    def _extend_classifier_fit_predictions(self, fit: Mapping[str, Any]) -> None:
        """Score observations added after the classifier's retained fit was created."""
        fit_id = int(fit["classifier_fit_id"])
        existing_ids = {
            int(row["observation_id"])
            for row in self.database.classifier_predictions(fit_id)
        }
        observation_ids, scoring_matrix = self._classifier_scoring_matrix(
            int(fit["geometry_id"])
        )
        missing_positions = [
            position
            for position, observation_id in enumerate(observation_ids)
            if observation_id not in existing_ids
        ]
        if not missing_positions:
            return
        estimator = self.artifacts.load_model(
            self.project_dir, str(fit["artifact_path"])
        )
        missing_matrix = scoring_matrix[missing_positions]
        outputs = score_classifier(
            estimator=estimator,
            algorithm=str(fit["algorithm"]),
            hyperparameters=dict(fit["hyperparameters"]),
            matrix=missing_matrix,
        )
        predictions = []
        for output_position, matrix_position in enumerate(missing_positions):
            predictions.append(
                {
                    "observation_id": observation_ids[matrix_position],
                    "predicted_label": (
                        "positive"
                        if int(outputs.predicted_labels[output_position]) == 1
                        else "negative"
                    ),
                    "probability": (
                        float(outputs.probabilities[output_position])
                        if outputs.probabilities is not None
                        else None
                    ),
                    "decision_score": (
                        float(outputs.decision_scores[output_position])
                        if outputs.decision_scores is not None
                        else None
                    ),
                    "uncertainty": float(outputs.uncertainty[output_position]),
                }
            )
        self.database.append_classifier_predictions(fit_id, predictions)

    def train_classifiers(
        self,
        *,
        code_id: int,
        classifier_spec_ids: Sequence[int],
        tune: bool = True,
        folds: int = 5,
        metric: str = "log_loss",
    ) -> dict[int, dict[str, Any]]:
        """Fit a batch of classifiers, selecting family hyperparameters when feasible."""
        return {
            int(classifier_spec_id): self.train_classifier(
                code_id=int(code_id),
                classifier_spec_id=int(classifier_spec_id),
                tune=bool(tune),
                folds=int(folds),
                metric=str(metric),
            )
            for classifier_spec_id in dict.fromkeys(classifier_spec_ids)
        }

    def classifier_status(
        self, *, code_id: int, classifier_spec_ids: Sequence[int] | None = None
    ) -> dict[str, Any]:
        """Report not-trained, current, or stale status for classifiers."""
        annotations = self.database.current_annotations(int(code_id))
        snapshot = self._classifier_training_snapshot(int(code_id), annotations)
        selected = (
            self.database.list_classifier_specs(code_id=int(code_id))
            if classifier_spec_ids is None
            else [
                self.database.get_classifier_spec(int(classifier_spec_id))
                for classifier_spec_id in classifier_spec_ids
            ]
        )
        rows: list[dict[str, Any]] = []
        for spec in selected:
            if int(spec["code_id"]) != int(code_id):
                raise ValueError("A selected classifier belongs to another code.")
            latest = self.database.latest_classifier_fit(
                code_id=int(code_id),
                classifier_spec_id=int(spec["classifier_spec_id"]),
            )
            prediction_complete = False
            if latest is not None:
                prediction_ids = {
                    int(row["observation_id"])
                    for row in self.database.classifier_predictions(
                        int(latest["classifier_fit_id"])
                    )
                }
                prediction_complete = all(
                    int(row["observation_id"]) in prediction_ids
                    for row in self.database.list_observations()
                )
            rows.append(
                {
                    **spec,
                    "classifier_fit_id": (
                        int(latest["classifier_fit_id"]) if latest is not None else None
                    ),
                    "status": (
                        "not_trained"
                        if latest is None
                        else (
                            "current"
                            if (
                                latest["training_snapshot"] == snapshot
                                and int(latest["geometry_id"]) == int(spec["geometry_id"])
                                and str(latest["algorithm"]) == str(spec["algorithm"])
                                and latest["hyperparameters"] == spec["hyperparameters"]
                                and prediction_complete
                            )
                            else "stale"
                        )
                    ),
                }
            )
        return {"classifiers": rows, "snapshot": snapshot}

    def tune_classifier_regularization(
        self,
        *,
        code_id: int,
        classifier_spec_id: int,
        folds: int = 5,
        metric: str = "log_loss",
        **_: Any,
    ) -> dict[str, Any]:
        """Legacy programmatic L2 selector; ordinary Develop training tunes automatically."""
        spec = self.database.get_classifier_spec(int(classifier_spec_id))
        if int(spec["code_id"]) != int(code_id):
            raise ValueError("The selected classifier belongs to a different code.")
        if str(spec["algorithm"]) != "logistic_l2":
            raise ValueError("This legacy helper supports L2 logistic regression only.")
        _, matrix, labels, _ = self._classifier_training_matrix(
            code_id=int(code_id), geometry_id=int(spec["geometry_id"])
        )
        selection = self._select_classifier_hyperparameters(
            code_id=int(code_id),
            spec=spec,
            training_matrix=matrix,
            labels=labels,
            folds=int(folds),
            metric=str(metric),
        )
        if not selection["tuned"]:
            raise ValueError(
                "Tuning requires at least two active Present and two active Absent examples."
            )
        selected_parameters = dict(selection["selected_hyperparameters"])
        self.update_classifier_spec(
            int(classifier_spec_id),
            name=str(spec["name"]),
            hyperparameters=selected_parameters,
        )
        results = [
            {
                "regularization": float(row["hyperparameters"]["regularization"]),
                "score": float(row["score"]),
                "fold_scores": list(row["fold_scores"]),
            }
            for row in selection["results"]
        ]
        return {
            "tuning_run_id": selection.get("tuning_run_id"),
            "classifier_spec_id": int(classifier_spec_id),
            "metric": str(selection["metric"]),
            "folds": int(selection["folds"]),
            "results": results,
            "selected_lambda": float(selected_parameters["regularization"]),
            "search_method": "training_grid",
        }

    def classifier_tuning_runs(
        self, *, code_id: int, classifier_spec_id: int | None = None
    ) -> list[dict[str, Any]]:
        """Return persisted exploratory tuning runs for one code/classifier."""
        return self.database.list_classifier_tuning_runs(
            code_id=int(code_id),
            classifier_spec_id=(
                int(classifier_spec_id) if classifier_spec_id is not None else None
            ),
        )

    def _committee_training_snapshot(
        self,
        *,
        code_id: int,
        committee: Mapping[str, Any],
        member_fit_ids: Sequence[int],
    ) -> dict[str, Any]:
        annotations = self.database.current_annotations(int(code_id))
        return {
            "training": self._classifier_training_snapshot(int(code_id), annotations),
            "committee_id": int(committee["committee_id"]),
            "committee_updated_at": str(committee["updated_at"]),
            "aggregation": str(committee["aggregation"]),
            "member_fit_ids": [int(value) for value in member_fit_ids],
        }

    def train_classifier_committee(
        self, *, code_id: int, committee_id: int
    ) -> dict[str, Any]:
        """Fit a trainable committee on the latest member-classifier scores."""
        committee = self.database.get_classifier_committee(int(committee_id))
        if int(committee["code_id"]) != int(code_id):
            raise ValueError("The selected committee belongs to another code.")
        aggregation = str(committee["aggregation"])
        if aggregation not in TRAINABLE_COMMITTEE_AGGREGATIONS:
            raise ValueError("This committee aggregation does not require training.")

        member_fits: list[dict[str, Any]] = []
        member_score_maps: list[dict[int, float]] = []
        for member in committee["members"]:
            fit = self.database.latest_classifier_fit(
                code_id=int(code_id),
                classifier_spec_id=int(member["classifier_spec_id"]),
            )
            if fit is None:
                raise ValueError(
                    f"Train {member['classifier_name']} before training this committee."
                )
            scores: dict[int, float] = {}
            for row in self.database.classifier_predictions(int(fit["classifier_fit_id"])):
                score = row["probability"]
                if score is None:
                    score = row["decision_score"]
                if score is not None:
                    scores[int(row["observation_id"])] = float(score)
            member_fits.append(fit)
            member_score_maps.append(scores)

        annotations = self.database.current_annotations(int(code_id))
        labeled = self._eligible_training_annotations(int(code_id), annotations)
        if not any(row["value"] == "positive" for row in labeled) or not any(
            row["value"] == "negative" for row in labeled
        ):
            raise ValueError("Committee training requires Present and Absent examples.")
        training_ids = [int(row["observation_id"]) for row in labeled]
        if any(any(obs_id not in scores for obs_id in training_ids) for scores in member_score_maps):
            raise ValueError("A member classifier is missing scores for committee training examples.")
        training_matrix = np.column_stack(
            [[scores[obs_id] for obs_id in training_ids] for scores in member_score_maps]
        )
        labels = np.asarray([1 if row["value"] == "positive" else 0 for row in labeled], dtype=int)

        observations = self.database.list_observations()
        scoring_ids = [int(row["observation_id"]) for row in observations]
        if any(any(obs_id not in scores for obs_id in scoring_ids) for scores in member_score_maps):
            raise ValueError("A member classifier is missing scores for committee observations.")
        scoring_matrix = np.column_stack(
            [[scores[obs_id] for obs_id in scoring_ids] for scores in member_score_maps]
        )
        member_fit_ids = [int(fit["classifier_fit_id"]) for fit in member_fits]
        snapshot = self._committee_training_snapshot(
            code_id=int(code_id), committee=committee, member_fit_ids=member_fit_ids
        )
        latest = self.database.latest_classifier_committee_fit(
            committee_id=int(committee_id), code_id=int(code_id)
        )
        if latest is not None and latest["training_snapshot"] == snapshot:
            self._extend_classifier_committee_fit_predictions(latest)
            return latest

        fitted = fit_classifier(
            algorithm="logistic_l2",
            hyperparameters={"regularization": 1.0},
            training_matrix=training_matrix,
            labels=labels,
            scoring_matrix=scoring_matrix,
        )
        fingerprint = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        artifact_path = self.artifacts.save_model(
            f"committee_code_{code_id}_committee_{committee_id}_{fingerprint}",
            fitted.estimator,
        )
        probabilities = fitted.outputs.probabilities
        if probabilities is None:
            raise RuntimeError("Logistic committee did not return probabilities.")
        predictions = [
            {
                "observation_id": observation_id,
                "predicted_label": "positive" if probability >= 0.5 else "negative",
                "probability": float(probability),
                "uncertainty": float(1.0 - min(1.0, abs(float(probability) - 0.5) * 2.0)),
            }
            for observation_id, probability in zip(scoring_ids, probabilities, strict=True)
        ]
        committee_fit_id = self.database.register_classifier_committee_fit(
            committee_id=int(committee_id),
            code_id=int(code_id),
            aggregation=aggregation,
            member_fit_ids=member_fit_ids,
            training_snapshot=snapshot,
            artifact_path=artifact_path,
            predictions=predictions,
        )
        return self.database.get_classifier_committee_fit(committee_fit_id)

    def _extend_classifier_committee_fit_predictions(
        self, fit: Mapping[str, Any]
    ) -> None:
        """Score observations added after a trainable committee fit was created."""
        committee_fit_id = int(fit["committee_fit_id"])
        existing_ids = {
            int(row["observation_id"])
            for row in self.database.classifier_committee_predictions(committee_fit_id)
        }
        observations = self.database.list_observations()
        observation_ids = [int(row["observation_id"]) for row in observations]
        missing_ids = [value for value in observation_ids if value not in existing_ids]
        if not missing_ids:
            return

        member_score_maps: list[dict[int, float]] = []
        for member_fit_id in fit["member_fit_ids"]:
            member_fit = self.database.get_classifier_fit(int(member_fit_id))
            self._extend_classifier_fit_predictions(member_fit)
            scores: dict[int, float] = {}
            for row in self.database.classifier_predictions(int(member_fit_id)):
                score = row["probability"]
                if score is None:
                    score = row["decision_score"]
                if score is not None:
                    scores[int(row["observation_id"])] = float(score)
            if any(observation_id not in scores for observation_id in missing_ids):
                raise ValueError(
                    "A member classifier is missing scores for newly ingested observations."
                )
            member_score_maps.append(scores)

        scoring_matrix = np.column_stack(
            [
                [scores[observation_id] for observation_id in missing_ids]
                for scores in member_score_maps
            ]
        )
        estimator = self.artifacts.load_model(
            self.project_dir, str(fit["artifact_path"])
        )
        outputs = score_classifier(
            estimator=estimator,
            algorithm="logistic_l2",
            hyperparameters={"regularization": 1.0},
            matrix=scoring_matrix,
        )
        probabilities = outputs.probabilities
        if probabilities is None:
            raise RuntimeError("Logistic committee did not return probabilities.")
        self.database.append_classifier_committee_predictions(
            committee_fit_id,
            [
                {
                    "observation_id": observation_id,
                    "predicted_label": "positive" if probability >= 0.5 else "negative",
                    "probability": float(probability),
                    "uncertainty": float(
                        1.0 - min(1.0, abs(float(probability) - 0.5) * 2.0)
                    ),
                }
                for observation_id, probability in zip(
                    missing_ids, probabilities, strict=True
                )
            ],
        )

    def classifier_committee_status(
        self, *, code_id: int, committee_id: int
    ) -> dict[str, Any]:
        """Report committee training status and member-classifier freshness."""
        committee = self.database.get_classifier_committee(int(committee_id))
        spec_ids = [int(row["classifier_spec_id"]) for row in committee["members"]]
        member_rows = self.classifier_status(
            code_id=int(code_id), classifier_spec_ids=spec_ids
        )["classifiers"]
        member_counts = {
            status: sum(row["status"] == status for row in member_rows)
            for status in ("current", "stale", "not_trained")
        }
        aggregation = str(committee["aggregation"])
        if aggregation in FIXED_COMMITTEE_AGGREGATIONS:
            status = "no_training_required"
            latest = None
        else:
            latest = self.database.latest_classifier_committee_fit(
                committee_id=int(committee_id), code_id=int(code_id)
            )
            if latest is None or any(row["classifier_fit_id"] is None for row in member_rows):
                status = "not_trained"
            else:
                member_fit_ids = [int(row["classifier_fit_id"]) for row in member_rows]
                snapshot = self._committee_training_snapshot(
                    code_id=int(code_id), committee=committee, member_fit_ids=member_fit_ids
                )
                prediction_complete = (
                    len(
                        self.database.classifier_committee_predictions(
                            int(latest["committee_fit_id"])
                        )
                    )
                    == len(self.database.list_observations())
                )
                status = (
                    "current"
                    if latest["training_snapshot"] == snapshot and prediction_complete
                    else "stale"
                )
        return {
            "committee": committee,
            "status": status,
            "member_counts": member_counts,
            "latest_fit": latest,
        }

    def _latest_trainable_committee_scores(
        self, *, code_id: int, committee_id: int
    ) -> dict[str, Any]:
        status = self.classifier_committee_status(
            code_id=int(code_id), committee_id=int(committee_id)
        )
        fit = status["latest_fit"]
        if fit is None:
            raise ValueError("Train this committee before using its learned aggregation.")
        rows = self.database.classifier_committee_predictions(int(fit["committee_fit_id"]))
        by_unit = {
            int(row["unit_id"]): float(row["probability"])
            for row in rows if row["unit_id"] is not None
        }
        unit_ids = [int(unit["unit_id"]) for unit in self.units()]
        if any(unit_id not in by_unit for unit_id in unit_ids):
            raise ValueError("The committee fit is missing atomic observation scores.")
        return {
            "fit": fit,
            "probabilities": [by_unit[unit_id] for unit_id in unit_ids],
            "stale": status["status"] == "stale",
        }

    def _latest_classifier_scores(
        self, *, code_id: int, classifier_spec_id: int
    ) -> dict[str, Any] | None:
        """Load the newest fit and atomic probability-like scores without refitting."""
        fit = self.database.latest_classifier_fit(
            code_id=int(code_id), classifier_spec_id=int(classifier_spec_id)
        )
        if fit is None:
            return None
        predictions = self.database.classifier_predictions(
            int(fit["classifier_fit_id"]), atomic_only=True
        )
        score_by_unit: dict[int, float] = {}
        for row in predictions:
            if row["unit_id"] is None:
                continue
            score = row["probability"]
            if score is None:
                score = row["decision_score"]
            if score is None:
                raise ValueError(
                    f"Classifier {fit['classifier_name']} exposes no usable score."
                )
            score_by_unit[int(row["unit_id"])] = float(score)
        unit_ids = [int(unit["unit_id"]) for unit in self.units()]
        if any(unit_id not in score_by_unit for unit_id in unit_ids):
            return None
        snapshot = self._classifier_training_snapshot(
            int(code_id), self.database.current_annotations(int(code_id))
        )
        spec = self.database.get_classifier_spec(int(classifier_spec_id))
        return {
            "fit": fit,
            "probabilities": [score_by_unit[unit_id] for unit_id in unit_ids],
            "stale": (
                fit["training_snapshot"] != snapshot
                or int(fit["geometry_id"]) != int(spec["geometry_id"])
                or str(fit["algorithm"]) != str(spec["algorithm"])
                or fit["hyperparameters"] != spec["hyperparameters"]
            ),
        }

    def _resolve_committee_scores(
        self, *, code_id: int, committee_id: int
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Resolve a committee definition to its latest code-specific fits."""
        committee = self.database.get_classifier_committee(int(committee_id))
        if int(committee["code_id"]) != int(code_id):
            raise ValueError("The selected committee belongs to another code.")
        results: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for member in committee["members"]:
            result = self._latest_classifier_scores(
                code_id=int(code_id),
                classifier_spec_id=int(member["classifier_spec_id"]),
            )
            if result is None:
                missing.append(str(member["classifier_name"]))
            else:
                results[str(member["classifier_name"])] = result
        if missing:
            raise ValueError(
                "Train all committee classifiers before using it. Missing: "
                + ", ".join(missing)
                + "."
            )
        return results, committee

    def focus_recommendation(
        self,
        *,
        code_id: int,
        session_id: int,
        strategy: str,
        active_classifier_spec_id: int | None = None,
        committee_id: int | None = None,
        recommendation_source: str = "classifier",
        random_seed: int | None = None,
        auto_retrain: bool = False,
        auto_train_all: bool = True,
    ) -> dict[str, Any] | None:
        """Generate one session-aware recommendation from explicit classifiers."""
        self.database.get_code(int(code_id))
        units = self.units()
        unit_ids = [int(unit["unit_id"]) for unit in units]
        annotations = self.database.current_annotations(int(code_id))
        annotated_ids = {
            int(row["unit_id"])
            for row in annotations
            if row["unit_id"] is not None
        }
        seen_ids = self.database.seen_unit_ids(int(session_id))

        if strategy == "review_unsure":
            unsure_ids = [
                int(row["unit_id"])
                for row in annotations
                if row["value"] == "unsure" and row["unit_id"] is not None
            ]
            if not unsure_ids:
                return None
            candidates = [unit_id for unit_id in unsure_ids if unit_id not in seen_ids]
            if not candidates:
                candidates = unsure_ids
            unit_id = int(
                np.random.default_rng(random_seed).choice(
                    np.asarray(candidates, dtype=int)
                )
            )
            return {
                "unit_id": unit_id,
                "strategy": "review_unsure",
                "score": None,
                "probabilities": {},
                "classifier_fit_ids": {},
                "fallback": False,
            }

        eligible_labels = self._eligible_training_annotations(int(code_id), annotations)
        positive = any(row["value"] == "positive" for row in eligible_labels)
        negative = any(row["value"] == "negative" for row in eligible_labels)
        if strategy == "random" or not positive or not negative:
            recommendation = recommend_focus_unit(
                strategy="random",
                unit_ids=unit_ids,
                seen_unit_ids=seen_ids,
                annotated_unit_ids=annotated_ids,
                random_seed=random_seed,
            )
            if recommendation is None:
                return None
            return {
                "unit_id": recommendation.unit_id,
                "strategy": "random",
                "score": None,
                "probabilities": {},
                "classifier_fit_ids": {},
                "fallback": strategy != "random",
            }

        committee_strategies = {
            "committee_uncertainty",
            "classifier_disagreement",
            "geometry_disagreement",
        }
        use_committee = (
            strategy in committee_strategies or recommendation_source == "committee"
        )
        committee_aggregation = "mean"
        if use_committee:
            if committee_id is None:
                raise ValueError("Select a classifier committee first.")
            committee = self.database.get_classifier_committee(int(committee_id))
            spec_ids = [
                int(member["classifier_spec_id"]) for member in committee["members"]
            ]
            if auto_retrain:
                auto_spec_ids = (
                    [
                        int(row["classifier_spec_id"])
                        for row in self.classifier_specs(code_id=int(code_id))
                    ]
                    if auto_train_all
                    else spec_ids
                )
                self.train_classifiers(
                    code_id=int(code_id), classifier_spec_ids=auto_spec_ids
                )
            results, committee = self._resolve_committee_scores(
                code_id=int(code_id), committee_id=int(committee_id)
            )
            committee_aggregation = str(committee["aggregation"])
            if auto_retrain and committee_aggregation in TRAINABLE_COMMITTEE_AGGREGATIONS:
                self.train_classifier_committee(
                    code_id=int(code_id), committee_id=int(committee_id)
                )
            recommendation_strategy = (
                "classifier_disagreement"
                if strategy in {"classifier_disagreement", "geometry_disagreement"}
                else (
                    "committee_uncertainty"
                    if strategy == "committee_uncertainty"
                    else strategy
                )
            )
            selected_name = None
        else:
            if active_classifier_spec_id is None:
                raise ValueError("Select an active classifier first.")
            if auto_retrain:
                if auto_train_all:
                    self.train_classifiers(
                        code_id=int(code_id),
                        classifier_spec_ids=[
                            int(row["classifier_spec_id"])
                            for row in self.classifier_specs(code_id=int(code_id))
                        ],
                    )
                else:
                    self.train_classifier(
                        code_id=int(code_id),
                        classifier_spec_id=int(active_classifier_spec_id),
                    )
            result = self._latest_classifier_scores(
                code_id=int(code_id),
                classifier_spec_id=int(active_classifier_spec_id),
            )
            if result is None:
                spec = self.database.get_classifier_spec(int(active_classifier_spec_id))
                raise ValueError(
                    f"Train {spec['name']} before using model recommendations."
                )
            name = str(result["fit"]["classifier_name"])
            results = {name: result}
            recommendation_strategy = (
                "classifier_uncertainty"
                if strategy in {"classifier_uncertainty", "geometry_uncertainty"}
                else strategy
            )
            selected_name = name

        recommendation_probabilities = {
            name: result["probabilities"] for name, result in results.items()
        }
        recommendation_aggregation = committee_aggregation
        if (
            use_committee
            and committee_aggregation in TRAINABLE_COMMITTEE_AGGREGATIONS
            and recommendation_strategy != "classifier_disagreement"
        ):
            learned = self._latest_trainable_committee_scores(
                code_id=int(code_id), committee_id=int(committee_id)
            )
            recommendation_probabilities = {"committee": learned["probabilities"]}
            recommendation_aggregation = "mean"

        recommendation = recommend_focus_unit(
            strategy=recommendation_strategy,  # type: ignore[arg-type]
            unit_ids=unit_ids,
            seen_unit_ids=seen_ids,
            annotated_unit_ids=annotated_ids,
            probabilities_by_geometry=recommendation_probabilities,
            selected_geometry=selected_name,
            committee_aggregation=recommendation_aggregation,  # type: ignore[arg-type]
            random_seed=random_seed,
        )
        if recommendation is None:
            return None
        return {
            "unit_id": recommendation.unit_id,
            "strategy": recommendation.strategy,
            "score": recommendation.score,
            "probabilities": recommendation.probabilities,
            "classifier_fit_ids": {
                name: int(result["fit"]["classifier_fit_id"])
                for name, result in results.items()
            },
            "stale_classifiers": [
                name for name, result in results.items() if result.get("stale")
            ],
            "fallback": False,
        }

    def observation_display_key(self, observation_id: int) -> str:
        """Return the compact user-key label used by geometric hover text."""
        observation = self.database.get_observation(int(observation_id))
        kind = str(observation["kind"])
        if kind == "atomic":
            return " · ".join(
                f"{name}: {value}"
                for name, value in observation.get("user_key", {}).items()
            )
        if kind == "teaching_example":
            return "Teaching example"
        member_ids = [int(value) for value in observation.get("member_unit_ids", [])]
        if not member_ids:
            return "Span"
        first = self.unit(member_ids[0])
        last = self.unit(member_ids[-1])
        first_key = list(first["user_key"].items())
        last_key = list(last["user_key"].items())
        if not first_key:
            return "Span"
        parent = " · ".join(f"{name}: {value}" for name, value in first_key[:-1])
        atomic_name, first_value = first_key[-1]
        last_value = dict(last_key).get(atomic_name, first_value)
        interval = (
            f"{atomic_name}: {first_value}"
            if first_value == last_value
            else f"{atomic_name}: {first_value}–{last_value}"
        )
        return f"{parent} · {interval}" if parent else interval

    def classifier_prediction_geometry(
        self, *, code_id: int, classifier_fit_ids: Sequence[int]
    ) -> dict[str, Any]:
        """Project observations through one or more classifier-score dimensions.

        Only observations scored by every selected immutable fit are included. Current
        human assignments are attached for display but do not alter the fitted scores.
        """
        fit_ids = list(dict.fromkeys(int(value) for value in classifier_fit_ids))
        if not fit_ids:
            raise ValueError("Select at least one classifier fit.")
        observations = self.database.list_observations()
        score_maps: list[dict[int, float]] = []
        names: list[str] = []
        for fit_id in fit_ids:
            fit = self.database.get_classifier_fit(fit_id)
            if int(fit["code_id"]) != int(code_id):
                raise ValueError("All selected fits must belong to the active code.")
            rows = self.database.classifier_predictions(fit_id)
            scores: dict[int, float] = {}
            for row in rows:
                raw_score = row["probability"]
                if raw_score is None:
                    raw_score = row["decision_score"]
                if raw_score is None:
                    continue
                scores[int(row["observation_id"])] = float(raw_score)
            score_maps.append(scores)
            names.append(str(fit["classifier_name"]))
        common_observation_ids = {
            int(observation["observation_id"]) for observation in observations
        }
        for scores in score_maps:
            common_observation_ids.intersection_update(scores)
        selected_observations = [
            observation
            for observation in observations
            if int(observation["observation_id"]) in common_observation_ids
        ]
        if not selected_observations:
            raise ValueError("The selected classifier fits share no scored observations.")
        observation_ids = [
            int(observation["observation_id"]) for observation in selected_observations
        ]
        matrix = np.column_stack(
            [np.asarray([scores[value] for value in observation_ids]) for scores in score_maps]
        )
        if matrix.shape[1] == 1:
            x = matrix[:, 0]
            y = np.asarray(
                [
                    ((((observation_id * 2_654_435_761) & 0xFFFFFFFF) % 1000) / 999.0 - 0.5)
                    * 0.08
                    for observation_id in observation_ids
                ],
                dtype=float,
            )
            method = "one_dimension_with_jitter"
            axis_titles = [names[0], "Deterministic jitter"]
        elif matrix.shape[1] == 2:
            x, y = matrix[:, 0], matrix[:, 1]
            method = "two_classifier_scores"
            axis_titles = names
        else:
            if matrix.shape[0] < 2 or np.allclose(matrix, matrix[0]):
                x = np.zeros(matrix.shape[0], dtype=float)
                y = np.zeros(matrix.shape[0], dtype=float)
            else:
                coordinates = PCA(n_components=2, random_state=0).fit_transform(matrix)
                x, y = coordinates[:, 0], coordinates[:, 1]
            method = "pca"
            axis_titles = ["Classifier-score PC1", "Classifier-score PC2"]
        current_labels = {
            int(row["observation_id"]): str(row["value"])
            for row in self.database.current_annotations(int(code_id))
        }
        return {
            "method": method,
            "classifier_fit_ids": fit_ids,
            "classifier_names": names,
            "axis_titles": axis_titles,
            "points": [
                {
                    "unit_id": (
                        int(observation["unit_id"])
                        if observation["unit_id"] is not None
                        else None
                    ),
                    "observation_id": observation_id,
                    "kind": str(observation["kind"]),
                    "text": str(observation["text"]),
                    "display_key": (
                        " · ".join(
                            f"{name}: {value}"
                            for name, value in observation.get("user_key", {}).items()
                        )
                        if observation["kind"] == "atomic"
                        else self.observation_display_key(observation_id)
                    ),
                    "assignment": current_labels.get(observation_id),
                    "x": float(x[position]),
                    "y": float(y[position]),
                    "scores": {
                        names[index]: float(matrix[position, index])
                        for index in range(matrix.shape[1])
                    },
                }
                for position, (observation_id, observation) in enumerate(
                    zip(observation_ids, selected_observations, strict=True)
                )
            ],
        }

    def classifier_testing_summary(self, code_id: int) -> dict[str, Any]:
        """Summarize stored diagnostic assessment events for the Testing Center."""
        events = self.database.list_classifier_evaluations(code_id=int(code_id))
        grouped: dict[int, list[dict[str, Any]]] = {}
        for event in events:
            grouped.setdefault(int(event["classifier_fit_id"]), []).append(event)
        rows = []
        for fit_id, fit_events in grouped.items():
            probabilities = [
                float(event["probability"])
                for event in fit_events
                if event["probability"] is not None
            ]
            labels = [1.0 if event["human_label"] == "positive" else 0.0 for event in fit_events]
            accuracy = float(
                np.mean(
                    [
                        event["predicted_label"] == event["human_label"]
                        for event in fit_events
                    ]
                )
            )
            brier = (
                float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2))
                if len(probabilities) == len(labels) and probabilities
                else None
            )
            rows.append(
                {
                    "classifier_fit_id": fit_id,
                    "classifier_name": str(fit_events[0]["classifier_name"]),
                    "n": len(fit_events),
                    "accuracy": accuracy,
                    "brier": brier,
                }
            )
        return {"events": events, "classifiers": rows}

    def create_apply_run(
        self,
        *,
        code_id: int,
        source_kind: str,
        source_id: int,
        threshold: float = 0.5,
    ) -> int:
        """Generate a reviewable draft from one classifier or persisted committee."""
        if source_kind not in {"classifier", "committee"}:
            raise ValueError("source_kind must be 'classifier' or 'committee'")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must lie in the closed interval [0, 1]")
        self.database.get_code(int(code_id))
        annotations = self.database.current_annotations(int(code_id))
        human_labels = self._eligible_training_annotations(int(code_id), annotations)
        if not any(row["value"] == "positive" for row in human_labels) or not any(
            row["value"] == "negative" for row in human_labels
        ):
            raise ValueError(
                "Apply requires at least one human-reviewed positive and one "
                "human-reviewed negative label."
            )
        if source_kind == "classifier":
            result = self._latest_classifier_scores(
                code_id=int(code_id), classifier_spec_id=int(source_id)
            )
            if result is None:
                spec = self.database.get_classifier_spec(int(source_id))
                raise ValueError(f"Train {spec['name']} before generating proposals.")
            results = {str(result["fit"]["classifier_name"]): result}
            aggregation = "mean"
        else:
            results, committee = self._resolve_committee_scores(
                code_id=int(code_id), committee_id=int(source_id)
            )
            aggregation = str(committee["aggregation"])
        unit_ids = [int(unit["unit_id"]) for unit in self.units()]
        committee_fit_id: int | None = None
        if source_kind == "committee" and aggregation in TRAINABLE_COMMITTEE_AGGREGATIONS:
            learned = self._latest_trainable_committee_scores(
                code_id=int(code_id), committee_id=int(source_id)
            )
            probabilities = np.asarray(learned["probabilities"], dtype=float)
            committee_fit_id = int(learned["fit"]["committee_fit_id"])
        else:
            probabilities = aggregate_probabilities(
                {name: result["probabilities"] for name, result in results.items()},
                aggregation=aggregation,  # type: ignore[arg-type]
            )
        reviewed_ids = {
            int(row["unit_id"])
            for row in annotations
            if row["unit_id"] is not None
        }
        proposals = build_apply_proposals(
            unit_ids=unit_ids,
            probabilities=probabilities,
            eligible_unit_ids=[value for value in unit_ids if value not in reviewed_ids],
            threshold=float(threshold),
        )
        return self.database.register_apply_run(
            code_id=int(code_id),
            aggregation=aggregation,
            source_kind=source_kind,
            source_id=int(source_id),
            threshold=float(threshold),
            classifier_fit_ids=[
                int(result["fit"]["classifier_fit_id"]) for result in results.values()
            ],
            committee_fit_id=committee_fit_id,
            proposals=[
                {
                    "unit_id": proposal.unit_id,
                    "probability": proposal.probability,
                    "proposed_label": proposal.proposed_label,
                }
                for proposal in proposals
            ],
        )

    def apply_runs(
        self,
        *,
        code_id: int | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List persisted Apply proposal runs."""
        return self.database.list_apply_runs(code_id=code_id, status=status)

    def apply_run(self, apply_run_id: int) -> dict[str, Any]:
        """Return one Apply run and its exact component classifier fits."""
        return self.database.get_apply_run(apply_run_id)

    def apply_proposals(
        self,
        apply_run_id: int,
        *,
        proposed_label: str | None = None,
        decision: str | None = None,
        include_unreviewed: bool = True,
        include_inactive: bool = False,
        probability_min: float | None = None,
        probability_max: float | None = None,
        limit: int | None = None,
        offset: int = 0,
        include_component_probabilities: bool = False,
    ) -> list[dict[str, Any]]:
        """Return review rows for an Apply run."""
        proposals = self.database.list_apply_proposals(
            apply_run_id,
            proposed_label=proposed_label,
            decision=decision,
            include_unreviewed=include_unreviewed,
            include_inactive=include_inactive,
            probability_min=probability_min,
            probability_max=probability_max,
            limit=limit,
            offset=offset,
        )
        if not include_component_probabilities or not proposals:
            return proposals
        run = self.database.get_apply_run(apply_run_id)
        components: dict[int, dict[int, float]] = {}
        names: dict[int, str] = {}
        for classifier_fit_id in run["classifier_fit_ids"]:
            fit = self.database.get_classifier_fit(int(classifier_fit_id))
            names[int(classifier_fit_id)] = str(fit["classifier_name"])
            components[int(classifier_fit_id)] = {
                int(row["unit_id"]): float(row["probability"])
                for row in self.database.classifier_predictions(
                    int(classifier_fit_id), atomic_only=True
                )
                if row["unit_id"] is not None and row["probability"] is not None
            }
        for proposal in proposals:
            unit_id = int(proposal["unit_id"])
            proposal["component_probabilities"] = {
                names[classifier_fit_id]: probabilities[unit_id]
                for classifier_fit_id, probabilities in components.items()
            }
        return proposals

    def review_apply_proposal(
        self,
        *,
        apply_run_id: int,
        unit_id: int,
        decision: str,
    ) -> None:
        """Record one individual review decision without changing annotations."""
        self.database.set_apply_decision(
            apply_run_id=apply_run_id,
            unit_id=unit_id,
            decision=decision,
            review_mode="individual",
        )

    def bulk_review_apply_proposals(
        self,
        *,
        apply_run_id: int,
        decision: str,
        proposed_label: str | None = None,
        pending_only: bool = False,
        unit_ids: Sequence[int] | None = None,
    ) -> int:
        """Apply one bulk decision to matching proposal rows."""
        return self.database.bulk_set_apply_decision(
            apply_run_id=apply_run_id,
            decision=decision,
            proposed_label=proposed_label,
            pending_only=pending_only,
            unit_ids=unit_ids,
        )

    def commit_apply_run(
        self,
        apply_run_id: int,
        *,
        unit_ids: Sequence[int] | None = None,
        scope: str = "all_reviewed",
        finalize: bool = True,
    ) -> dict[str, int]:
        """Process non-pending proposals in one explicit commit scope."""
        return self.database.commit_apply_run(
            apply_run_id, unit_ids=unit_ids, scope=scope, finalize=finalize
        )

    def undo_last_apply_commit(self, apply_run_id: int) -> dict[str, int]:
        """Undo the newest Apply commit while preserving an auditable retraction."""
        return self.database.undo_last_apply_commit(int(apply_run_id))

    def can_undo_apply_commit(self, apply_run_id: int) -> bool:
        return self.database.latest_apply_commit_operation(int(apply_run_id)) is not None

    def discard_apply_run(self, apply_run_id: int) -> None:
        """Discard a draft Apply run without changing annotations."""
        self.database.discard_apply_run(apply_run_id)

    def create_memo(
        self,
        *,
        title: str,
        body_markdown: str,
        hashtags: Sequence[str] = (),
        unit_references: Sequence[int] = (),
    ) -> int:
        """Create a standalone memo with optional evidence references."""
        return self.database.create_memo(
            title=title,
            body_markdown=body_markdown,
            hashtags=hashtags,
            unit_references=unit_references,
        )

    def memos(self) -> list[dict[str, Any]]:
        """Return project memos ordered by most recent update."""
        return self.database.list_memos()

    def memo_hashtags(self) -> list[str]:
        """Return all memo hashtags in case-insensitive display order."""
        return self.database.memo_hashtags()

    def search_memos(
        self,
        *,
        query: str = "",
        scope: str = "full",
        hashtags: Sequence[str] = (),
        hashtag_operator: str = "and",
    ) -> list[dict[str, Any]]:
        """Search the memo repository by text and hashtag logic."""
        return self.database.search_memos(
            query=query,
            scope=scope,
            hashtags=hashtags,
            hashtag_operator=hashtag_operator,
        )

    def memo(self, memo_id: int, version_number: int | None = None) -> dict[str, Any]:
        """Return one memo version, defaulting to the latest."""
        return self.database.get_memo(memo_id, version_number)

    def memo_versions(self, memo_id: int) -> list[dict[str, Any]]:
        """Return every immutable version of one memo."""
        return self.database.memo_versions(memo_id)

    def update_memo(
        self,
        memo_id: int,
        *,
        title: str,
        body_markdown: str,
        hashtags: Sequence[str] = (),
        unit_references: Sequence[int] = (),
    ) -> int:
        """Append a new version of an existing memo."""
        return self.database.update_memo(
            memo_id,
            title=title,
            body_markdown=body_markdown,
            hashtags=hashtags,
            unit_references=unit_references,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return project metadata from SQLite."""
        return self.database.project_metadata()

    @property
    def is_external_backed(self) -> bool:
        """Whether this workspace's numerical universe is owned externally."""
        return bool(self.metadata.get("external_backed", False))

    def launch(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8050,
        debug: bool = False,
    ) -> None:
        """Launch the local Dash interface."""
        from geometric_coder.ui import launch_app

        launch_app(self, host=host, port=port, debug=debug)


__all__ = ["GeometricCoder", "Geometry"]
