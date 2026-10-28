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
        choices = {
            "umap": {
                "metric": ("euclidean", "cosine", "manhattan", "chebyshev", "minkowski"),
                "storage": ("compact", "native"),
            },
            "normalize": {"axis": ("rows", "columns"), "norm": ("l1", "l2")},
            "tfidf": {"norm": ("", "l1", "l2")},
            "svd": {"algorithm": ("randomized", "arpack"),
                    "power_iteration_normalizer": ("auto", "QR", "LU", "none")},
        }.get(recipe, {}).get(name)
        if choices is not None:
            spec["kind"] = "choice"
            spec["options"] = [{"label": option or "None", "value": option}
                               for option in choices]
            if spec["default"] is None:
                spec["default"] = ""
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


def linked_representation_options(coder: GeometricCoder) -> list[dict[str, Any]]:
    """TeAL-backed numerical sources, whether or not available for analysis."""
    if teal_linked_project(coder) is None:
        return []
    return [
        {"label": str(row["name"]), "value": int(row["representation_id"])}
        for row in coder.database.list_representations()
        if isinstance(row.get("external_ref"), Mapping)
        and isinstance(row["external_ref"].get("artifact_id"), str)
    ]


def _source_representation(
    coder: GeometricCoder, *, source_geometry_id: int | None = None,
    source_representation_id: int | None = None,
) -> tuple[int, dict[str, Any]]:
    if source_representation_id is not None:
        record = coder.database.get_representation(int(source_representation_id))
        rep_id = int(record["representation_id"])
    else:
        records = [
            g for g in coder.geometries(public_only=True)
            if int(g["geometry_id"]) == int(source_geometry_id or -1)
        ]
        if len(records) != 1:
            raise ValueError("Choose a registered source geometry.")
        record = dict(records[0])
        rep_id = int(coder.database.representation_for_legacy(
            "geometry", int(source_geometry_id)
        )["representation_id"])
    ref = record.get("external_ref")
    if not isinstance(ref, Mapping) or not isinstance(ref.get("artifact_id"), str):
        raise ConfigurationError("The selected source is not a TeAL numerical matrix.")
    return rep_id, dict(ref)


def create_geometry(
    coder: GeometricCoder, *, name: str,
    recipe: str, source_geometry_id: int | None = None,
    source_representation_id: int | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> GeometryCreationResult:
    """Save one TeAL translation, then register its immutable output in GeCo.

    The standard path delegates capability validation to linked.add_geometry.
    TeAL's recorded pipeline determines whether the output can transform
    new text. A fit-only matrix remains usable for navigation without replay.
    """
    clean = str(name).strip()
    if not clean:
        raise ValueError("Enter a geometry name.")
    if coder.has_geometry(clean):
        raise ConfigurationError(f"Geometry {clean!r} is already registered.")
    path = teal_linked_project(coder)
    if path is None:
        raise ConfigurationError("Creating geometries requires a TeAL-linked GeCo workspace.")
    source_id, ref = _source_representation(
        coder, source_geometry_id=source_geometry_id,
        source_representation_id=source_representation_id,
    )
    # Reject unsupported derived-observation backfill *before* an expensive fit.
    if coder.database.list_observations(kinds=["span", "teaching_example"]):
        raise ConfigurationError(
            "Create Geometry requires a workspace without existing spans or teaching "
            "examples until safe geometry backfill is available."
        )
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
            text_source = teal.get_artifact(str(linked.manifest["text_artifact_id"]))
            from text_analysis_lab.core.errors import PipelineError
            try:
                teal.pipeline(start=text_source, end=output)
            except PipelineError:
                replayable = False
            else:
                replayable = True
            if replayable:
                geometry_id = linked.add_geometry(clean, output)
            else:
                if output.primary_key != linked.documents.primary_key:
                    raise ConfigurationError(
                        "Navigation-only geometry must use linked document primary keys."
                    )
                geometry_id = coder.register_external_geometry(
                    name=clean, external_ref={"artifact_id": str(output.artifact_id)},
                    supports_query=False, supports_text_transform=False,
                )
        # The wizard explicitly identifies one numerical source. Record that
        # relationship independently of TeAL's broader artifact provenance,
        # regardless of whether the transform expands or reduces dimensions.
        child_representation = coder.database.representation_for_legacy(
            "geometry", int(geometry_id)
        )
        coder.database.assign_representation_parent(
            int(child_representation["representation_id"]), source_id
        )
        # A 2D analytic geometry needs a display view under its own stable ID.
        # Do not silently create a projection for higher-dimensional geometries.
        probe = [0] if int(output.n_rows or 0) else []
        if int(output.get_matrix(positions=probe).shape[1]) == 2:
            _ensure_identity_display_view(
                coder, geometry_id=int(geometry_id),
                name=clean, artifact_id=str(output.artifact_id),
            )
        return GeometryCreationResult(
            geometry_id=int(geometry_id),
            artifact_id=str(output.artifact_id),
            replayable=replayable,
        )


def _teal_numeric_ancestors(teal: Any, artifact_id: str) -> tuple[str, ...]:
    """Follow only unique, numerical, one-input operation sources.

    TeAL artifact basis lineage is intentionally excluded: sharing document
    identities does not establish a numerical visualization relationship.
    Multi-input operations terminate the chain rather than inventing a parent.
    """
    seen = {str(artifact_id)}
    current = str(artifact_id)
    ancestry: list[str] = []
    for _ in range(1024):
        operation = teal.catalog.operation_for_artifact(current)
        if not operation:
            return tuple(ancestry)
        sources = teal.catalog.operation_sources(str(operation["operation_id"]))
        if len(sources) != 1:
            return tuple(ancestry)
        upstream = str(sources[0]["source_artifact_id"])
        if upstream in seen:
            raise ConfigurationError("A cycle was found in numerical artifact provenance.")
        source_artifact = teal.get_artifact(upstream)
        if getattr(source_artifact.artifact_type, "value", None) not in {
            "dense_matrix", "sparse_matrix"
        }:
            return tuple(ancestry)
        ancestry.append(upstream)
        seen.add(upstream)
        current = upstream
    raise ConfigurationError("Numerical artifact provenance exceeded the supported depth.")


def reconcile_teal_numeric_parents(coder: GeometricCoder, teal: Any) -> int:
    """Resolve nearest registered numerical parents without import-order dependence.

    This updates only the GeCo-owned parent pointers. It never edits TeAL's
    operation graph, artifacts, aliases, or numerical payloads.
    """
    entries = coder.database.list_representations()
    by_artifact: dict[str, set[int]] = {}
    for row in entries:
        ref = row.get("external_ref")
        if isinstance(ref, Mapping) and isinstance(ref.get("artifact_id"), str):
            by_artifact.setdefault(ref["artifact_id"], set()).add(
                int(row["representation_id"])
            )
    traces: dict[int, tuple[str, ...]] = {}
    for row in entries:
        ref = row.get("external_ref")
        if isinstance(ref, Mapping) and isinstance(ref.get("artifact_id"), str):
            traces[int(row["representation_id"])] = _teal_numeric_ancestors(
                teal, ref["artifact_id"]
            )
    # Establish short/root chains before attempting to refine a descendant's
    # existing parent, so the newer immediate ancestor is itself linked.
    count = 0
    for child_id, ancestry in sorted(traces.items(), key=lambda item: len(item[1])):
        parents = []
        for upstream in ancestry:
            matches = by_artifact.get(upstream, set()) - {child_id}
            if len(matches) > 1:
                break  # ambiguous owner; do not guess
            if len(matches) == 1:
                parents = list(matches)
                break
        if not parents:
            continue
        parent_id = parents[0]
        previous = coder.database.get_representation(child_id)["parent_representation_id"]
        if previous == parent_id:
            continue
        try:
            coder.database.assign_representation_parent(
                child_id, parent_id, allow_refinement=previous is not None
            )
        except ValueError:
            # Retain recorded relationships if they disagree with inferred
            # lineage. The operation remains safe to repeat after correction.
            continue
        count += 1
    return count


def representation_inventory(coder: GeometricCoder) -> list[dict[str, Any]]:
    """List each numerical representation once, regardless of its UI role.

    Source is GeCo's single-input numerical parent. The source label comes
    from the same preferred-name lookup as the representation itself. TeAL
    general basis ancestry must never be used as numerical parenthood.
    """
    registry = coder.database.list_representations()
    path = teal_linked_project(coder)
    teal = None
    if path is not None:
        from text_analysis_lab.core.project import Project
        teal = Project.open(path)
    try:
        if teal is not None:
            reconcile_teal_numeric_parents(coder, teal)
            registry = coder.database.list_representations()

        by_id = {int(row["representation_id"]): row for row in registry}
        public_geometry_ids = {
            int(row["geometry_id"]) for row in coder.geometries(public_only=True)
        }
        names = {}
        artifacts = {}
        for row in registry:
            rid = int(row["representation_id"])
            display_name = str(row["name"])
            ref = row.get("external_ref")
            if teal is not None and isinstance(ref, Mapping) and ref.get("artifact_id"):
                try:
                    artifact = teal.get_artifact(str(ref["artifact_id"]))
                    artifacts[rid] = artifact
                    aliases = artifact.aliases
                    display_name = str(aliases[0] if aliases else artifact.label)
                except Exception:
                    pass
            names[rid] = display_name

        rows = []
        for record in registry:
            rid = int(record["representation_id"])
            kind = str(record["legacy_kind"])
            legacy_id = int(record["legacy_id"])
            # Hidden internal geometries are not researcher-visible entries.
            if kind == "geometry" and legacy_id not in public_geometry_ids:
                continue
            dimensions = record.get("dimensions")
            error = None
            try:
                if dimensions is None:
                    if rid in artifacts:
                        artifact = artifacts[rid]
                        sample = [0] if int(artifact.n_rows or 0) > 0 else []
                        dimensions = int(artifact.get_matrix(positions=sample).shape[1])
                    elif record["storage_kind"] == "local":
                        import numpy as np
                        if kind == "geometry":
                            geom = coder.database.get_geometry(legacy_id)
                            filepath = coder.project_dir / str(geom["matrix_path"])
                            if geom["matrix_sparse"]:
                                with np.load(filepath, allow_pickle=False) as packet:
                                    dimensions = int(packet["shape"][1])
                            else:
                                dimensions = int(np.load(filepath, mmap_mode="r").shape[1])
                        else:
                            view = coder.database.get_view(legacy_id)
                            dimensions = int(
                                np.load(coder.project_dir / str(view["artifact_path"]),
                                        mmap_mode="r").shape[1]
                            )
                    if dimensions is not None:
                        coder.database.record_representation_dimensions(rid, int(dimensions))
            except Exception as exc:
                error = str(exc)
            parent_id = record.get("parent_representation_id")
            source = names.get(int(parent_id), "") if parent_id is not None else ""
            if kind == "geometry":
                geometry = coder.database.get_geometry(legacy_id)
                supports_new_text = bool(
                    geometry["supports_query"] and geometry["supports_text_transform"]
                    and record["analytic_enabled"]
                )
            else:
                supports_new_text = False
            ref = record.get("external_ref")
            rows.append({
                "key": f"representation:{rid}",
                "representation_id": rid,
                "name": names[rid],
                "dimensions": dimensions,
                "source": source,
                "sources": [source] if source else [],
                "new_text": supports_new_text,
                "analytic_enabled": bool(record["analytic_enabled"]),
                "can_view": dimensions in (1, 2),
                "parent_representation_id": parent_id,
                "artifact_id": ref.get("artifact_id") if isinstance(ref, Mapping) else None,
                "storage_kind": record["storage_kind"],
                "error": error,
            })
        return rows
    finally:
        if teal is not None:
            teal.close()


def filter_representations(
    rows: list[dict[str, Any]], *, min_dimensions: int | None = None,
    max_dimensions: int | None = None, source: str | None = None,
) -> list[dict[str, Any]]:
    """Filter registered resources; a blank bound means unbounded."""
    if min_dimensions is not None and min_dimensions < 1:
        raise ValueError("Minimum dimensions must be positive.")
    if max_dimensions is not None and max_dimensions < 1:
        raise ValueError("Maximum dimensions must be positive.")
    if (min_dimensions is not None and max_dimensions is not None
            and min_dimensions > max_dimensions):
        raise ValueError("Minimum dimensions cannot exceed maximum dimensions.")
    return sorted([
        row for row in rows
        if (row.get("dimensions") is not None)
        and (min_dimensions is None or row["dimensions"] >= min_dimensions)
        and (max_dimensions is None or row["dimensions"] <= max_dimensions)
        and (not source or source in row.get("sources", []))
    ], key=lambda row: (str(row["name"]).casefold(), str(row["key"])))


def source_filter_options(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Sorted human-facing source handles, never raw artifact identifiers."""
    return [{"label": name, "value": name}
            for name in sorted(
                {source for row in rows for source in row.get("sources", [])},
                key=str.casefold,
            )]


def _ensure_identity_display_view(
    coder: GeometricCoder, *, geometry_id: int, name: str, artifact_id: str,
) -> int:
    """Give an existing 2D analytic matrix a selectable view under its own geometry.

    An identity view reuses the saved coordinates. It does not fit a second
    transformer, and can be recovered idempotently after interrupted promotion.
    """
    for view in coder.views(geometry_id=int(geometry_id)):
        ref = view.get("external_ref")
        if (isinstance(ref, Mapping)
                and str(ref.get("artifact_id")) == str(artifact_id)):
            return int(view["view_id"])
    base = f"{name} (identity)"
    taken = {str(view["name"]) for view in coder.views()}
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base} {suffix}"
        suffix += 1
    return int(coder.register_external_view(
        geometry_id=int(geometry_id), name=candidate,
        external_ref={"artifact_id": str(artifact_id)},
    ))


def register_view_as_geometry(coder: GeometricCoder, *, view_id: int) -> int:
    """Register an existing 2D view as an analytic geometry, without a fit.

    Reuses its saved provider artifact, conservatively disabling new-text and
    semantic-query transformations until a replayable pipeline is proven.
    """
    view = coder.database.get_view(int(view_id))
    if str(view.get("storage_kind")) != "external":
        raise ConfigurationError(
            "Register as Geometry currently requires a provider-backed 2D view."
        )
    ref = view.get("external_ref")
    if not isinstance(ref, Mapping) or not isinstance(ref.get("artifact_id"), str):
        raise ConfigurationError("The selected view has no registered matrix artifact.")
    artifact_id = str(ref["artifact_id"])
    name = str(view["name"])
    for geometry in coder.geometries(public_only=False):
        existing = geometry.get("external_ref")
        if (isinstance(existing, Mapping)
                and existing.get("artifact_id") == artifact_id):
            if str(geometry["name"]) == name:
                geometry_id = int(geometry["geometry_id"])
                # The original view is already linked to the same canonical
                # numerical ID, so Explorer can use it without a duplicate.
                return geometry_id
            raise ConfigurationError(
                f"This artifact is already registered as geometry {geometry['name']!r}. "
                "Rename the representation before assigning both roles."
            )
    if coder.has_geometry(name):
        raise ConfigurationError(f"Geometry {name!r} is already registered.")
    # The provider validates full keyed coverage. No fitting or new artifact.
    geometry_id = int(coder.register_external_geometry(
        name=name, external_ref={"artifact_id": artifact_id},
        supports_query=False, supports_text_transform=False,
    ))
    return geometry_id


def set_representation_analytic_enabled(
    coder: GeometricCoder, *, representation_id: int, enabled: bool,
) -> bool:
    """Change analytic eligibility without deleting the numerical artifact.

    Provider-backed view-only registrations gain an analytic reference to
    their existing matrix. Demotion refuses roots and active dependent research
    state; legacy geometry rows remain as durable compatibility references.
    """
    record = coder.database.get_representation(int(representation_id))
    if bool(record["analytic_enabled"]) == bool(enabled):
        return bool(enabled)
    if enabled:
        if record["legacy_kind"] == "view":
            register_view_as_geometry(coder, view_id=int(record["legacy_id"]))
        else:
            with coder.database.connect() as connection:
                connection.execute(
                    "UPDATE representations SET analytic_enabled=1 "
                    "WHERE representation_id=?",
                    (int(representation_id),),
                )
        return True

    if record["parent_representation_id"] is None:
        raise ConfigurationError(
            "A root numerical representation cannot be disabled for analysis."
        )
    if record["legacy_kind"] != "geometry":
        raise ConfigurationError("Only analytic geometry registrations may be disabled.")
    geometry_id = int(record["legacy_id"])
    with coder.database.connect() as connection:
        import json
        for session in connection.execute("SELECT state_json FROM sessions"):
            state = json.loads(session["state_json"])
            if state.get("geometry_id") is not None and int(state["geometry_id"]) == geometry_id:
                raise ConfigurationError(
                    "Cannot disable analytic use while a saved session selects this geometry."
                )
        for table in (
            "classifier_specs", "classifier_fits", "navigation_events",
            "observation_geometry_vectors",
        ):
            if connection.execute(
                f"SELECT 1 FROM {table} WHERE geometry_id=? LIMIT 1",
                (geometry_id,),
            ).fetchone() is not None:
                raise ConfigurationError(
                    f"Cannot disable analytic use while {table} references this geometry."
                )
        connection.execute(
            "UPDATE representations SET analytic_enabled=0 "
            "WHERE representation_id=?",
            (int(representation_id),),
        )
    return False


def _representation_record(coder: GeometricCoder, key: str) -> tuple[str, dict[str, Any]]:
    try:
        kind, identifier = str(key).split(":", 1)
        if kind == "representation":
            canonical = coder.database.get_representation(int(identifier))
            kind = str(canonical["legacy_kind"])
            identifier = str(canonical["legacy_id"])
        if kind == "geometry":
            return kind, coder.database.get_geometry(int(identifier))
        if kind == "view":
            return kind, coder.database.get_view(int(identifier))
    except (KeyError, ValueError) as error:
        raise ConfigurationError("Unknown representation.") from error
    raise ConfigurationError("Unknown representation type.")


def rename_representation(coder: GeometricCoder, *, key: str, name: str) -> None:
    """Rename a registration and, when TeAL-linked, rotate its owned alias.

    TeAL artifacts are referenced by immutable IDs; only the selected alias
    is retired, never unrelated aliases belonging to the same artifact.
    """
    import json
    from sqlite3 import IntegrityError

    kind, record = _representation_record(coder, key)
    old = str(record["name"])
    clean = str(name).strip()
    if not clean:
        raise ValueError("Enter a representation name.")
    if clean == old:
        return
    if coder.has_geometry(clean) or coder.has_view(clean):
        raise ConfigurationError(f"A representation named {clean!r} already exists.")
    path = teal_linked_project(coder)
    ref = record.get("external_ref")
    teal = None
    artifact = None
    added = False
    try:
        if path is not None and isinstance(ref, Mapping) and ref.get("artifact_id"):
            from text_analysis_lab.core.project import Project
            teal = Project.open(path)
            artifact = teal.get_artifact(str(ref["artifact_id"]))
            # The TeAL catalog validates both alias syntax and its uniqueness.
            teal.catalog.validate_artifact_alias(clean)
            if clean not in artifact.aliases:
                artifact.add_alias(clean)
                added = True
        table = "geometries" if kind == "geometry" else "views"
        id_column = "geometry_id" if kind == "geometry" else "view_id"
        with coder.database.connect() as con:
            con.execute(
                f"UPDATE {table} SET name = ? WHERE {id_column} = ?",
                (clean, int(record[id_column])),
            )
            con.execute(
                "UPDATE representations SET name = ? WHERE representation_id IN "
                "(SELECT representation_id FROM representation_legacy_refs "
                "WHERE legacy_kind = ? AND legacy_id = ?)",
                (clean, kind, int(record[id_column])),
            )
            # One canonical artifact may have both legacy geometry and view
            # references; keep the transitional names coherent.
            related = con.execute(
                "SELECT l.legacy_kind, l.legacy_id FROM representation_legacy_refs l "
                "JOIN representation_legacy_refs source "
                "ON source.representation_id=l.representation_id "
                "WHERE source.legacy_kind=? AND source.legacy_id=?",
                (kind, int(record[id_column])),
            ).fetchall()
            for alias in related:
                if alias["legacy_kind"] == kind and int(alias["legacy_id"]) == int(record[id_column]):
                    continue
                alias_table = "geometries" if alias["legacy_kind"] == "geometry" else "views"
                alias_column = "geometry_id" if alias["legacy_kind"] == "geometry" else "view_id"
                con.execute(
                    f"UPDATE {alias_table} SET name=? WHERE {alias_column}=?",
                    (clean, int(alias["legacy_id"])),
                )
            # Native dependent geometries store their source by name.
            if kind == "geometry":
                for child in coder.geometries(public_only=False):
                    deps = list(child.get("dependency_names") or [])
                    if old in deps:
                        con.execute(
                            "UPDATE geometries SET dependency_names_json = ? "
                            "WHERE geometry_id = ?",
                            (json.dumps([clean if dep == old else dep for dep in deps]),
                             int(child["geometry_id"])),
                        )
        if artifact is not None and old in artifact.aliases:
            artifact.remove_alias(old)
    except Exception:
        if added and artifact is not None:
            artifact.remove_alias(clean)
        raise
    finally:
        if teal is not None:
            teal.close()


def unregister_representation(coder: GeometricCoder, *, key: str) -> None:
    """Remove only a GeCo registration when no known references remain.

    Never removes TeAL artifacts or aliases. Conservative dependency checks
    guard historical sessions and models, rather than silently orphaning them.
    """
    kind, record = _representation_record(coder, key)
    identifier = int(record["geometry_id" if kind == "geometry" else "view_id"])
    with coder.database.connect() as con:
        sessions = con.execute("SELECT state_json FROM sessions").fetchall()
        for row in sessions:
            import json
            state = json.loads(row["state_json"])
            fields = ("geometry_id",) if kind == "geometry" else ("view_id",)
            if any(str(state.get(field)) == str(identifier) for field in fields):
                raise ConfigurationError(
                    "This representation is used in a saved session. Select "
                    "another representation in that session before removing it."
                )
        associated = con.execute(
            "SELECT representation_id FROM representation_legacy_refs "
            "WHERE legacy_kind = ? AND legacy_id = ?",
            (kind, identifier),
        ).fetchone()
        if associated is not None:
            descendant = con.execute(
                "SELECT 1 FROM representations WHERE parent_representation_id = ? LIMIT 1",
                (int(associated["representation_id"]),),
            ).fetchone()
            if descendant is not None:
                raise ConfigurationError(
                    "Cannot unregister a representation while descendants depend on it."
                )
        if kind == "geometry":
            if len(coder.geometries(public_only=True)) <= 1:
                raise ConfigurationError("Cannot unregister the last analytic geometry.")
            for table in ("views", "classifier_specs", "classifier_fits",
                          "navigation_events", "observation_geometry_vectors"):
                if con.execute(
                    f"SELECT 1 FROM {table} WHERE geometry_id = ? LIMIT 1",
                    (identifier,),
                ).fetchone():
                    raise ConfigurationError(
                        f"Cannot unregister geometry: {table} still references it."
                    )
            for child in coder.geometries(public_only=False):
                if (int(child["geometry_id"]) != identifier
                        and str(record["name"]) in child.get("dependency_names", [])):
                    raise ConfigurationError("Cannot unregister a source of another geometry.")
            con.execute("DELETE FROM geometries WHERE geometry_id = ?", (identifier,))
        else:
            con.execute("DELETE FROM views WHERE view_id = ?", (identifier,))
        if associated is not None:
            representation_id = int(associated["representation_id"])
            con.execute(
                "DELETE FROM representation_legacy_refs "
                "WHERE legacy_kind=? AND legacy_id=?",
                (kind, identifier),
            )
            remaining = con.execute(
                "SELECT legacy_kind, legacy_id FROM representation_legacy_refs "
                "WHERE representation_id=? "
                "ORDER BY CASE legacy_kind WHEN 'geometry' THEN 0 ELSE 1 END, legacy_id",
                (representation_id,),
            ).fetchall()
            if not remaining:
                con.execute(
                    "DELETE FROM representations WHERE representation_id = ?",
                    (representation_id,),
                )
            else:
                first = remaining[0]
                retained_analytic = any(x["legacy_kind"] == "geometry" for x in remaining)
                table_name = "geometries" if first["legacy_kind"] == "geometry" else "views"
                column_name = "geometry_id" if first["legacy_kind"] == "geometry" else "view_id"
                retained = con.execute(
                    f"SELECT name FROM {table_name} WHERE {column_name}=?",
                    (int(first["legacy_id"]),),
                ).fetchone()
                con.execute(
                    "UPDATE representations SET legacy_kind=?, legacy_id=?, "
                    "name=?, analytic_enabled=? WHERE representation_id=?",
                    (first["legacy_kind"], int(first["legacy_id"]), str(retained["name"]),
                     int(retained_analytic), representation_id),
                )


def create_view(
    coder: GeometricCoder, *, name: str, method: str = "umap",
    geometry_id: int | None = None,
    source_representation_id: int | None = None,
    dimensions: int = 2,
    parameters: Mapping[str, Any] | None = None,
) -> int:
    """Persist one visualization-only 1D/2D numerical representation.

    Its identity and source are recorded in the unified representation
    registry. A legacy numerical geometry row is a compatibility backing,
    not a second scientific artifact or an analytic role.
    """
    clean = str(name).strip()
    if not clean:
        raise ValueError("Enter a view representation name.")
    if coder.has_geometry(clean) or coder.has_view(clean):
        raise ConfigurationError(f"Representation {clean!r} is already registered.")
    if dimensions not in (1, 2):
        raise ValueError("Supported view dimensions are 1 and 2.")
    if method not in {"umap", "svd"}:
        raise ValueError(
            "A view of an existing representation is implicit; choose UMAP or SVD to create another."
        )
    if "n_components" in (parameters or {}):
        raise ValueError("View dimensionality is fixed by the dimensions control.")
    path = teal_linked_project(coder)
    if path is None:
        raise ConfigurationError("Creating views requires a TeAL-linked GeCo workspace.")
    source_id, ref = _source_representation(
        coder, source_geometry_id=geometry_id,
        source_representation_id=source_representation_id,
    )
    if coder.database.list_observations(kinds=["span", "teaching_example"]):
        raise ConfigurationError(
            "New numerical representations require derived-observation backfill, "
            "which is not yet implemented."
        )

    from text_analysis_lab.core.project import Project
    operator = _operator_for_recipe(
        method, {**(parameters or {}), "n_components": dimensions}
    )
    with Project.open(path) as teal:
        links = [
            row for row in teal.geco.list()
            if row.get("mode", "explore") == "explore"
            and Path(str(row["workspace_path"])).resolve() == coder.project_dir.resolve()
        ]
        if len(links) != 1:
            raise ConfigurationError(
                "Could not uniquely match this GeCo workspace to its TeAL link."
            )
        source = teal.get_artifact(ref["artifact_id"])
        if getattr(source.artifact_type, "value", None) not in {"dense_matrix", "sparse_matrix"}:
            raise ValueError("A view source must be a numerical TeAL matrix.")
        output = teal.translate(
            operator, source,
            memo=(f"GeCo Create View {clean!r}, {dimensions}D, method={method}, "
                  f"source={source.artifact_id}."),
        )["output"]
        if getattr(output.artifact_type, "value", None) not in {"dense_matrix", "sparse_matrix"}:
            raise ValueError("The view transformation did not produce a numerical matrix.")
        sample = [0] if int(output.n_rows or 0) > 0 else []
        if int(output.get_matrix(positions=sample).shape[1]) != dimensions:
            raise ValueError("The fitted view did not have the requested dimensionality.")
        existing = {
            str(row["external_ref"]["artifact_id"])
            for row in coder.database.list_representations()
            if isinstance(row.get("external_ref"), Mapping)
            and isinstance(row["external_ref"].get("artifact_id"), str)
        }
        if str(output.artifact_id) in existing:
            raise ConfigurationError(
                "This saved artifact is already a GeCo representation; select it instead."
            )
        alias_added = False
        try:
            if clean not in output.aliases:
                output.add_alias(clean)
                alias_added = True
            new_geometry_id = coder.register_external_geometry(
                name=clean, external_ref={"artifact_id": str(output.artifact_id)},
                supports_query=False, supports_text_transform=False,
                analytic_enabled=False,
            )
            result = coder.database.representation_for_legacy(
                "geometry", int(new_geometry_id)
            )
            coder.database.assign_representation_parent(
                int(result["representation_id"]), source_id
            )
        except Exception:
            if alias_added:
                output.remove_alias(clean)
            raise
        return int(result["representation_id"])
