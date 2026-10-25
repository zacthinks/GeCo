"""Curated Topic Model creation through TeAL's public artifact and link APIs.

This module does not implement statistical fitting. It is optional: plain GeCo
projects and non-TeAL providers can still explore previously registered models.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from geometric_coder.exceptions import ConfigurationError

if TYPE_CHECKING:
    from geometric_coder.project import GeometricCoder


@dataclass(frozen=True)
class TopicCreationResult:
    topic_model_id: int
    membership_artifact_id: str
    dominant_artifact_id: str
    description_artifact_id: str | None


def teal_linked_project(coder: GeometricCoder) -> Path | None:
    """Find a TeAL project only through the supported TeAL provider type."""
    if not coder.is_external_backed:
        return None
    try:
        from text_analysis_lab.integrations.geco import TeALGeCoProvider
    except ImportError:
        return None
    provider = coder.external_provider
    if not isinstance(provider, TeALGeCoProvider):
        return None
    path = getattr(provider.project, "path", None)
    return Path(path).resolve() if path is not None else None


def lda_geometry_options(coder: GeometricCoder) -> list[dict[str, Any]]:
    """Only registered TeAL geometry artifacts are candidates for LDA."""
    if teal_linked_project(coder) is None:
        return []
    return [
        {"label": str(row["name"]), "value": int(row["geometry_id"])}
        for row in coder.geometries(public_only=True)
        if isinstance(row.get("external_ref"), Mapping)
        and isinstance(row["external_ref"].get("artifact_id"), str)
        and row["external_ref"]["artifact_id"]
    ]


def create_lda_topic_model(
    coder: GeometricCoder,
    *,
    name: str,
    geometry_id: int,
    n_topics: int,
    max_iter: int = 10,
    random_state: int = 42,
) -> TopicCreationResult:
    """Fit TeAL LDA on an existing count geometry; register via its GeCo link.

    Creation is synchronous: the Dash UI must show a running state, never a
    fabricated progress bar. Failed/unsupported fittings are not registered in
    GeCo; TeAL may retain intermediate lineage artifacts for inspection.
    """
    clean_name = str(name).strip()
    if not clean_name:
        raise ValueError("Enter a name for the Topic Model.")
    if isinstance(n_topics, bool) or not isinstance(n_topics, int) or n_topics < 2:
        raise ValueError("Number of topics must be an integer of at least 2.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("Maximum iterations must be a positive integer.")
    if any(str(row["name"]) == clean_name for row in coder.topic_models()):
        raise ValueError(f"A Topic Model named {clean_name!r} already exists.")

    path = teal_linked_project(coder)
    if path is None:
        raise ConfigurationError(
            "Creating Topic Models is supported only for TeAL-linked GeCo "
            "workspaces. Create artifacts in your provider and register them instead."
        )
    matches = [
        row for row in coder.geometries(public_only=True)
        if int(row["geometry_id"]) == geometry_id
    ]
    if len(matches) != 1:
        raise ValueError("Select an available GeCo representation.")
    ref = matches[0].get("external_ref")
    if not isinstance(ref, Mapping) or not isinstance(ref.get("artifact_id"), str):
        raise ConfigurationError("The selected representation has no TeAL artifact reference.")

    # These are public TeAL APIs. Import only when the creation action executes.
    from text_analysis_lab.core.project import Project
    from text_analysis_lab.translators import ArgMax, LDA

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
        if getattr(source.artifact_type, "value", None) not in {"sparse_matrix", "dense_matrix"}:
            raise ValueError("LDA requires a TeAL count matrix as its input.")
        provenance = (
            f"Requested in GeCo Topic Model manager: {clean_name!r}. "
            f"LDA topics={n_topics}, max_iter={max_iter}, random_state={random_state}. "
            "Source must contain nonnegative integer token counts."
        )
        outputs = teal.translate(
            LDA(n_components=n_topics, max_iter=max_iter, random_state=random_state),
            source,
            memo=provenance,
        )
        membership = outputs["output"]
        description = outputs.get("topics")
        dominant = teal.translate(
            ArgMax(), membership, memo=f"ArgMax for {clean_name!r} LDA membership."
        )["output"]
        with teal.geco.open(str(links[0]["name"])) as linked:
            model_id = linked.add_topic_model(
                clean_name,
                membership=membership,
                dominant=dominant,
                description=description,
            )
        return TopicCreationResult(
            topic_model_id=int(model_id),
            membership_artifact_id=str(membership.artifact_id),
            dominant_artifact_id=str(dominant.artifact_id),
            description_artifact_id=(
                str(description.artifact_id) if description is not None else None
            ),
        )


# TeAL's translator constructors are the authority for parameter names,
# signatures, and defaults. Only the algorithm inventory lives in GeCo.
# "operator_id" is TeAL's internal operation identity, not a fit hyperparameter.
MODEL_CLASSES: tuple[tuple[str, str, str], ...] = (
    ("identity", "Identity (existing geometry)", ""),
    ("lda", "Latent Dirichlet allocation (LDA)", "LDA"),
    ("gaussian_mixture", "Gaussian mixture (GMM)", "GaussianMixture"),
    ("kmeans", "K-means (Euclidean)", "KMeans"),
    ("minibatch_kmeans", "MiniBatch K-means (Euclidean)", "MiniBatchKMeans"),
    ("spherical_kmeans", "Spherical K-means (directional)", "SphericalKMeans"),
    ("kmedoids", "K-medoids", "KMedoids"),
    ("spectral", "Spectral clustering", "SpectralClustering"),
    ("agglomerative", "Agglomerative clustering", "AgglomerativeClustering"),
    ("dbscan", "DBSCAN", "DBSCAN"),
    ("optics", "OPTICS", "OPTICS"),
    ("hdbscan", "HDBSCAN", "HDBSCAN"),
)


def recipe_options() -> list[dict[str, str]]:
    return [{"label": name, "value": key} for key, name, _ in MODEL_CLASSES]


def topic_geometry_options(coder: GeometricCoder) -> list[dict[str, Any]]:
    """List *all* registered GeCo geometries, including standalone projects.

    TeAL fitting availability is a separate capability; hiding geometry names
    misleads researchers into believing their registered resources are missing.
    """
    return [
        {"label": str(row["name"]), "value": int(row["geometry_id"])}
        for row in coder.geometries(public_only=True)
    ]


def lda_geometry_options(coder: GeometricCoder) -> list[dict[str, Any]]:
    """Backward-compatible alias; scientific eligibility is evaluated by TeAL."""
    return topic_geometry_options(coder)


def _recipe_class(recipe: str) -> Any:
    for key, _, class_name in MODEL_CLASSES:
        if key == recipe:
            if key == "identity":
                return None
            from text_analysis_lab import translators
            return getattr(translators, class_name)
    raise ValueError(f"Unknown TeAL Topic Model algorithm {recipe!r}.")


def recipe_fields(recipe: str) -> tuple[str, ...]:
    """Read the full public constructor signature from the installed TeAL."""
    operator = _recipe_class(recipe)
    if operator is None:
        return ()
    from inspect import signature
    return tuple(
        name for name, parameter in signature(operator).parameters.items()
        if name != "operator_id"
        and parameter.kind.name not in {"VAR_POSITIONAL", "VAR_KEYWORD"}
    )


def _annotation_text(parameter: Any) -> str:
    return str(parameter.annotation).replace("typing.", "").replace(" ", "")


def parameter_specs(recipe: str) -> list[dict[str, Any]]:
    """Expose all TeAL constructor hyperparameters with original defaults."""
    operator = _recipe_class(recipe)
    if operator is None:
        return []
    from inspect import signature
    specs: list[dict[str, Any]] = []
    for name, param in signature(operator).parameters.items():
        if name == "operator_id":
            continue
        annotation = _annotation_text(param)
        default = param.default
        if isinstance(default, bool) or "bool" in annotation:
            kind = "boolean"
        elif ("int" in annotation and "str" in annotation):
            kind = "text"  # e.g. sklearn n_init='auto'
        elif "float" in annotation and "str" in annotation:
            kind = "text"  # e.g. eigen_tol='auto'
        elif "int" in annotation:
            kind = "integer"
        elif "float" in annotation:
            kind = "number"
        elif isinstance(default, str):
            kind = "text"
        else:
            kind = "text"
        # Dash/JSON cannot encode +/-inf numeric input values.
        if isinstance(default, float) and not __import__("math").isfinite(default):
            kind = "text"
            default = str(default)
        if default is None:
            default = "" if kind == "text" else None
        specs.append({
            "name": name,
            "label": name.replace("_", " ").capitalize(),
            "kind": kind,
            "default": default,
            "annotation": annotation,
        })
    return specs


def _parse_parameter(name: str, raw: Any, spec: Mapping[str, Any]) -> Any:
    """Only decode input widgets; the translator validates scientific inputs."""
    if raw is None or raw == "":
        return None
    kind = spec["kind"]
    annotation = str(spec["annotation"])
    if kind == "boolean":
        return raw is True or raw == "true"
    if kind == "integer":
        return int(raw)
    if kind == "number":
        return float(raw)
    if "int" in annotation and "str" in annotation and str(raw).lstrip("+-").isdigit():
        return int(raw)
    if "float" in annotation and "str" in annotation:
        try:
            return float(raw)
        except (ValueError, TypeError):
            return str(raw)
    return str(raw)


def _operator_for_recipe(recipe: str, parameters: Mapping[str, Any]) -> Any:
    """Instantiate the actual TeAL translator using its complete public API."""
    operator = _recipe_class(recipe)
    if operator is None:
        raise ValueError("Identity reuses the existing TeAL matrix without fitting.")
    specs = {spec["name"]: spec for spec in parameter_specs(recipe)}
    unknown = set(parameters) - set(specs)
    if unknown:
        raise ValueError(f"Unexpected TeAL hyperparameters: {sorted(unknown)!r}")
    args = {
        name: parsed
        for name, raw in parameters.items()
        if (parsed := _parse_parameter(name, raw, specs[name])) is not None
    }
    return operator(**args)


def create_topic_model(
    coder: GeometricCoder,
    *,
    name: str,
    geometry_id: int,
    recipe: str,
    parameters: Mapping[str, Any] | None = None,
) -> TopicCreationResult:
    """Fit or reuse one TeAL artifact and register one GeCo Topic Model.

    No algorithmic fitting is performed here. Identity preserves the source
    artifact ID exactly. Existing TeAL scientific source validation is retained.
    """
    _recipe_class(recipe)  # Validate selection against the installed TeAL catalog.
    clean_name = str(name).strip()
    if not clean_name:
        raise ValueError("Enter a name for the Topic Model.")
    if any(str(m["name"]) == clean_name for m in coder.topic_models()):
        raise ValueError(f"A Topic Model named {clean_name!r} already exists.")
    path = teal_linked_project(coder)
    if path is None:
        raise ConfigurationError(
            "Creating Topic Models requires a TeAL-linked GeCo workspace."
        )
    matches = [g for g in coder.geometries(public_only=True) if int(g["geometry_id"]) == int(geometry_id)]
    if len(matches) != 1:
        raise ValueError("Choose an existing registered geometry.")
    ref = matches[0].get("external_ref")
    if not isinstance(ref, Mapping) or not isinstance(ref.get("artifact_id"), str):
        raise ConfigurationError("Selected geometry has no linked TeAL artifact.")
    from text_analysis_lab.core.project import Project
    from text_analysis_lab.translators import ArgMax

    with Project.open(path) as teal:
        links = [
            row for row in teal.geco.list()
            if row.get("mode", "explore") == "explore"
            and Path(str(row["workspace_path"])).resolve() == coder.project_dir.resolve()
        ]
        if len(links) != 1:
            raise ConfigurationError("Could not uniquely locate this GeCo workspace's TeAL link.")
        source = teal.get_artifact(str(ref["artifact_id"]))
        if recipe == "identity":
            membership, description = source, None
        else:
            operator = _operator_for_recipe(recipe, parameters or {})
            outputs = teal.translate(
                operator, source,
                memo=f"Requested from GeCo Topic Models: {clean_name!r}, "
                     f"algorithm={recipe}, source={source.artifact_id}.",
            )
            membership = outputs["output"]
            description = outputs.get("topics") if recipe == "lda" else None
        dominant = teal.translate(
            ArgMax(), membership,
            memo=f"Dominant component for Topic Model {clean_name!r}.",
        )["output"]
        with teal.geco.open(str(links[0]["name"])) as linked:
            model_id = linked.add_topic_model(
                clean_name,
                membership=membership,
                dominant=dominant,
                description=description,
            )
        return TopicCreationResult(
            topic_model_id=int(model_id),
            membership_artifact_id=str(membership.artifact_id),
            dominant_artifact_id=str(dominant.artifact_id),
            description_artifact_id=(
                str(description.artifact_id) if description is not None else None
            ),
        )
