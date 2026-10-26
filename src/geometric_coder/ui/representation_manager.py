"""Manage named analytic geometries backed by ordinary TeAL artifacts.

GeCo owns registration and UI names; TeAL owns immutable numerical artifacts.
Nothing in this module silently replaces a geometry with its 2D display view.
"""
from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from geometric_coder.exceptions import ConfigurationError
from geometric_coder.ui.topic_creation import _parse_parameter, teal_linked_project

if TYPE_CHECKING:
    from geometric_coder.project import GeometricCoder


# One source geometry and one ordinary TeAL matrix-to-matrix translation per step.
# This is a UI catalog, not a competing scientific pipeline implementation.
GEOMETRY_RECIPES: tuple[tuple[str, str, str], ...] = (
    ("tfidf", "TF-IDF weighting", "TfidfTransformer"),
    ("normalize", "Matrix normalization", "MatrixNormalizer"),
    ("svd", "Truncated SVD", "SVD"),
    ("umap", "UMAP", "UMAP"),
)
PRIMARY_PARAMETERS: dict[str, tuple[str, ...]] = {
    "tfidf": ("norm", "use_idf", "sublinear_tf"),
    "normalize": ("axis", "norm"),
    "svd": ("n_components", "random_state"),
    "umap": ("n_components", "n_neighbors", "min_dist", "metric", "reuse"),
}


@dataclass(frozen=True)
class GeometryCreationResult:
    geometry_id: int
    artifact_id: str
    replayable: bool


def geometry_recipe_options() -> list[dict[str, str]]:
    return [{"label": label, "value": key} for key, label, _ in GEOMETRY_RECIPES]


def _recipe_class(recipe: str) -> Any:
    for key, _, class_name in GEOMETRY_RECIPES:
        if recipe == key:
            from text_analysis_lab import translators
            return getattr(translators, class_name)
    raise ValueError(f"Unknown geometry transformation: {recipe!r}.")


def geometry_parameter_specs(recipe: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read actual TeAL constructor defaults; keep uncommon parameters collapsible."""
    operator = _recipe_class(recipe)
    common = set(PRIMARY_PARAMETERS[recipe])
    primary: list[dict[str, Any]] = []
    advanced: list[dict[str, Any]] = []
    for name, param in signature(operator).parameters.items():
        if name == "operator_id" or param.kind.name in {"VAR_POSITIONAL", "VAR_KEYWORD"}:
            continue
        annotation = str(param.annotation).replace("typing.", "").replace(" ", "")
        default = param.default
        if isinstance(default, bool) or "bool" in annotation:
            kind = "boolean"
        elif "int" in annotation and "str" in annotation:
            kind = "text"
        elif "float" in annotation and "str" in annotation:
            kind = "text"
        elif "int" in annotation:
            kind = "integer"
        elif "float" in annotation:
            kind = "number"
        else:
            kind = "text"
        if isinstance(default, float) and not __import__("math").isfinite(default):
            kind, default = "text", str(default)
        if default is None:
            default = "" if kind == "text" else None
        spec: dict[str, Any] = {
            "name": name, "label": name.replace("_", " ").capitalize(),
            "kind": kind, "default": default, "annotation": annotation,
        }
        if recipe == "umap" and name == "reuse":
            spec.update(kind="choice", options=[
                {"label": label, "value": value} for value, label in (
                    ("none", "Fit-only (no reusable model)"),
                    ("recompute", "Refit from source when needed"),
                    ("stored", "Preserve fitted transformer"),
                )
            ])
        (primary if name in common else advanced).append(spec)
    return primary, advanced


def _operator_for_recipe(recipe: str, parameters: Mapping[str, Any]) -> Any:
    specs = {s["name"]: s for group in geometry_parameter_specs(recipe) for s in group}
    unknown = set(parameters) - set(specs)
    if unknown:
        raise ValueError(f"Unknown TeAL transformation parameters: {sorted(unknown)!r}.")
    kwargs = {
        name: parsed for name, raw in parameters.items()
        if (parsed := _parse_parameter(name, raw, specs[name])) is not None
    }
    return _recipe_class(recipe)(**kwargs)


def linked_geometry_options(coder: GeometricCoder) -> list[dict[str, Any]]:
    """Only TeAL-artifact-backed GeCo geometries can be transformation sources."""
    if teal_linked_project(coder) is None:
        return []
    return [
        {"label": str(g["name"]), "value": int(g["geometry_id"])}
        for g in coder.geometries(public_only=True)
        if isinstance(g.get("external_ref"), Mapping)
        and isinstance(g["external_ref"].get("artifact_id"), str)
    ]


def create_geometry(
    coder: GeometricCoder, *, name: str, source_geometry_id: int,
    recipe: str, parameters: Mapping[str, Any] | None = None,
    navigation_only: bool = False,
) -> GeometryCreationResult:
    """Save one TeAL translation, then register its immutable output in GeCo.

    The standard path delegates capability validation to linked.add_geometry.
    Explicit navigation-only mode registers a full-key matrix without claiming
    it can transform search text or derived observations. In particular, a
    fit-only UMAP may be explored but cannot be used as a text-query geometry.
    """
    clean = str(name).strip()
    if not clean:
        raise ValueError("Enter a geometry name.")
    if coder.has_geometry(clean):
        raise ConfigurationError(f"Geometry {clean!r} is already registered.")
    path = teal_linked_project(coder)
    if path is None:
        raise ConfigurationError("Creating geometries requires a TeAL-linked GeCo workspace.")
    records = [
        g for g in coder.geometries(public_only=True)
        if int(g["geometry_id"]) == int(source_geometry_id)
    ]
    if len(records) != 1:
        raise ValueError("Choose a registered source geometry.")
    ref = records[0].get("external_ref")
    if not isinstance(ref, Mapping) or not isinstance(ref.get("artifact_id"), str):
        raise ConfigurationError("The source geometry is not a TeAL matrix artifact.")

    from text_analysis_lab.core.project import Project

    # Construct/validate the operator before opening any mutable project state.
    operator = _operator_for_recipe(recipe, parameters or {})
    with Project.open(path) as teal:
        links = [
            row for row in teal.geco.list()
            if row.get("mode", "explore") == "explore"
            and Path(str(row["workspace_path"])).resolve() == coder.project_dir.resolve()
        ]
        if len(links) != 1:
            raise ConfigurationError("Could not uniquely match this GeCo workspace to its TeAL link.")
        source = teal.get_artifact(ref["artifact_id"])
        if getattr(source.artifact_type, "value", None) not in {"dense_matrix", "sparse_matrix"}:
            raise ValueError("A source geometry must be a TeAL numerical matrix.")
        output = teal.translate(
            operator, source,
            memo=(f"GeCo Create Geometry {clean!r}: {recipe} from "
                  f"{source.artifact_id!r}. Parameters: {dict(parameters or {})!r}."),
        )["output"]
        if getattr(output.artifact_type, "value", None) not in {"dense_matrix", "sparse_matrix"}:
            raise ValueError("This TeAL transformation did not produce a numerical matrix.")
        with teal.geco.open(str(links[0]["name"])) as linked:
            if navigation_only:
                # TeAL's provider checks actual key correspondence and row coverage.
                # No text_artifact_id or replay claims are stored for fit-only matrices.
                if output.primary_key != linked.documents.primary_key:
                    raise ConfigurationError(
                        "Navigation-only geometry must use the linked document primary keys."
                    )
                geometry_id = coder.register_external_geometry(
                    name=clean, external_ref={"artifact_id": str(output.artifact_id)},
                    supports_query=False, supports_text_transform=False,
                )
            else:
                # This enforces TeAL's existing text-to-geometry Pipeline contract.
                geometry_id = linked.add_geometry(clean, output)
        return GeometryCreationResult(
            geometry_id=int(geometry_id),
            artifact_id=str(output.artifact_id),
            replayable=not navigation_only,
        )


def representation_inventory(coder: GeometricCoder) -> list[dict[str, Any]]:
    """Registry-centered inspection, with optional immutable TeAL provenance.

    An unavailable artifact is reported, not treated as a valid replacement.
    Dimension counting reads only one row when TeAL is linked.
    """
    rows: list[dict[str, Any]] = []
    path = teal_linked_project(coder)
    by_artifact: dict[str, str] = {
        str(ref["artifact_id"]): str(g["name"])
        for g in coder.geometries(public_only=True)
        if isinstance((ref := g.get("external_ref")), Mapping)
        and isinstance(ref.get("artifact_id"), str)
    }
    teal = None
    try:
        if path is not None:
            from text_analysis_lab.core.project import Project
            teal = Project.open(path)
        for g in coder.geometries(public_only=True):
            record = {
                "geometry_id": int(g["geometry_id"]), "name": str(g["name"]),
                "storage_kind": g.get("storage_kind", "local"),
                "queryable": bool(g.get("supports_query")),
                "text_transform": bool(g.get("supports_text_transform")),
                "source": ", ".join(g.get("dependency_names") or []),
                "artifact_id": None, "dimensions": None, "error": None,
            }
            try:
                ref = g.get("external_ref")
                if teal is not None and isinstance(ref, Mapping) and ref.get("artifact_id"):
                    artifact = teal.get_artifact(str(ref["artifact_id"]))
                    record["artifact_id"] = str(artifact.artifact_id)
                    # TeAL lineage is authoritative; do not infer chains from names.
                    from text_analysis_lab.core.lineage import basis_artifact_ids
                    try:
                        bases = basis_artifact_ids(artifact)
                    except Exception:
                        bases = ()
                    record["source"] = ", ".join(
                        by_artifact.get(str(parent), str(parent)) for parent in bases
                    ) or "TeAL artifact"
                    probe = [0] if (artifact.n_rows or 0) > 0 else []
                    record["dimensions"] = int(artifact.get_matrix(positions=probe).shape[1])
                elif g.get("storage_kind") == "local":
                    import numpy as np
                    from scipy import sparse

                    file_path = coder.project_dir / str(g["matrix_path"])
                    if g.get("matrix_sparse"):
                        # Read only the compressed matrix shape metadata.
                        with np.load(file_path, allow_pickle=False) as packet:
                            shape = packet["shape"]
                            record["dimensions"] = int(shape[1])
                    else:
                        record["dimensions"] = int(
                            np.load(file_path, mmap_mode="r", allow_pickle=False).shape[1]
                        )
            except Exception as exc:
                record["error"] = str(exc)
            rows.append(record)
    finally:
        if teal is not None:
            teal.close()
    return rows


def create_view(
    coder: GeometricCoder, *, name: str, geometry_id: int, method: str = "umap",
    parameters: Mapping[str, Any] | None = None,
) -> int:
    """Persist a 2D display projection without changing analytic geometry.

    Identity is available only when the existing geometry is already 2D.
    Projection fits use TeAL translations, keeping transformation provenance
    in TeAL and associating the resulting view with one GeCo geometry ID.
    """
    clean = str(name).strip()
    if not clean:
        raise ValueError("Enter a 2D view name.")
    if coder.has_view(clean):
        raise ConfigurationError(f"2D view {clean!r} already exists.")
    path = teal_linked_project(coder)
    if path is None:
        raise ConfigurationError("Creating 2D views requires a TeAL-linked GeCo workspace.")
    records = [g for g in coder.geometries(public_only=True)
               if int(g["geometry_id"]) == int(geometry_id)]
    if len(records) != 1:
        raise ValueError("Choose a registered parent geometry.")
    ref = records[0].get("external_ref")
    if not isinstance(ref, Mapping) or not isinstance(ref.get("artifact_id"), str):
        raise ConfigurationError("The parent must reference a TeAL numerical artifact.")
    if method not in {"identity", "umap", "svd"}:
        raise ValueError(f"Unsupported 2D projection method {method!r}.")
    if method == "identity" and parameters:
        raise ValueError("Identity view does not accept fit parameters.")
    if method != "identity" and "n_components" in (parameters or {}):
        # A view always has exactly two coordinate dimensions.
        raise ValueError("2D view dimensionality is fixed at two.")

    from text_analysis_lab.core.project import Project
    with Project.open(path) as teal:
        links = [row for row in teal.geco.list()
                 if row.get("mode", "explore") == "explore"
                 and Path(str(row["workspace_path"])).resolve() == coder.project_dir.resolve()]
        if len(links) != 1:
            raise ConfigurationError("Could not uniquely match this GeCo workspace to its TeAL link.")
        source = teal.get_artifact(ref["artifact_id"])
        if getattr(source.artifact_type, "value", None) not in {"dense_matrix", "sparse_matrix"}:
            raise ValueError("A view source must be a numerical TeAL matrix.")
        if method == "identity":
            if int(source.get_matrix(positions=[0]).shape[1]) != 2:
                raise ValueError("Identity view requires exactly two dimensions.")
            output = source
        else:
            operator = _operator_for_recipe(method, {**(parameters or {}), "n_components": 2})
            output = teal.translate(
                operator, source,
                memo=f"GeCo Create 2D View {clean!r}, method={method}, source={source.artifact_id}.",
            )["output"]
        with teal.geco.open(str(links[0]["name"])) as linked:
            return int(linked.add_projection(clean, output, geometry=str(records[0]["name"])))
