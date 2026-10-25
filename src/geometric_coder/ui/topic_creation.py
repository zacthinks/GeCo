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


# An explicit *UI* catalog of TeAL translators, not a duplicated scientific
# eligibility engine. TeAL remains responsible for validating matrix inputs.
# Supported creation fields intentionally expose common parameters rather than
# every low-level estimator knob.
MODEL_RECIPES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("identity", "Identity (existing geometry)", ()),
    ("lda", "Latent Dirichlet allocation (LDA)", ("count", "max_iter")),
    ("gaussian_mixture", "Gaussian mixture (GMM)", ("count", "max_iter", "covariance_type")),
    ("kmeans", "K-means", ("count", "max_iter")),
    ("minibatch_kmeans", "MiniBatch K-means", ("count", "max_iter", "batch_size")),
    ("spherical_kmeans", "Spherical K-means", ("count", "max_iter")),
    ("kmedoids", "K-medoids", ("count", "max_iter", "metric")),
    ("spectral", "Spectral clustering", ("count", "n_neighbors", "affinity")),
    ("agglomerative", "Agglomerative clustering", ("count", "metric", "linkage")),
    ("dbscan", "DBSCAN", ("eps", "min_samples", "metric")),
    ("optics", "OPTICS", ("min_samples", "xi", "metric")),
    ("hdbscan", "HDBSCAN", ("min_cluster_size", "min_samples", "metric")),
)
RECIPE_FIELDS: tuple[tuple[str, str, str, Any], ...] = (
    ("count", "Number of topics / clusters", "int", 10),
    ("max_iter", "Maximum iterations", "int", 100),
    ("batch_size", "Batch size", "int", 1024),
    ("min_cluster_size", "Minimum cluster size", "int", 10),
    ("min_samples", "Minimum samples", "int", 5),
    ("eps", "Neighborhood radius (eps)", "float", 0.5),
    ("xi", "OPTICS steepness (xi)", "float", 0.05),
    ("n_neighbors", "Neighbors", "int", 10),
    ("metric", "Distance metric", "choice", "euclidean"),
    ("covariance_type", "Covariance type", "choice", "full"),
    ("affinity", "Spectral affinity", "choice", "rbf"),
    ("linkage", "Agglomerative linkage", "choice", "ward"),
)
FIELD_CHOICES: dict[str, tuple[str, ...]] = {
    "metric": ("euclidean", "manhattan", "cosine"),
    "covariance_type": ("full", "tied", "diag", "spherical"),
    "affinity": ("rbf", "nearest_neighbors"),
    "linkage": ("ward", "complete", "average", "single"),
}


def recipe_options() -> list[dict[str, str]]:
    return [{"label": title, "value": key} for key, title, _ in MODEL_RECIPES]


def recipe_fields(recipe: str) -> tuple[str, ...]:
    for key, _, fields in MODEL_RECIPES:
        if recipe == key:
            return fields
    raise ValueError(f"Unknown TeAL Topic Model algorithm {recipe!r}.")


def topic_geometry_options(coder: GeometricCoder) -> list[dict[str, Any]]:
    """Offer TeAL-linked registered geometries, without duplicating eligibility."""
    return lda_geometry_options(coder)


def _operator_for_recipe(recipe: str, parameters: Mapping[str, Any]) -> Any:
    """Construct one existing TeAL translator, leaving numerical checks to TeAL."""
    from text_analysis_lab import translators as tr

    fields = recipe_fields(recipe)
    args: dict[str, Any] = {}
    field_types = {field: kind for field, _, kind, _ in RECIPE_FIELDS}
    for key in fields:
        value = parameters.get(key)
        if value is None or value == "":
            continue
        kind = field_types[key]
        if kind == "int":
            if isinstance(value, bool) or int(value) != float(value):
                raise ValueError(f"{key} must be a whole number.")
            value = int(value)
        elif kind == "float":
            value = float(value)
        elif kind == "choice":
            if value not in FIELD_CHOICES[key]:
                raise ValueError(f"Invalid {key}: {value!r}.")
        args[key] = value

    if "count" in args:
        args["n_components" if recipe in {"lda", "gaussian_mixture"} else "n_clusters"] = args.pop("count")

    classes = {
        "lda": tr.LDA,
        "gaussian_mixture": tr.GaussianMixture,
        "kmeans": tr.KMeans,
        "minibatch_kmeans": tr.MiniBatchKMeans,
        "spherical_kmeans": tr.SphericalKMeans,
        "kmedoids": tr.KMedoids,
        "spectral": tr.SpectralClustering,
        "agglomerative": tr.AgglomerativeClustering,
        "dbscan": tr.DBSCAN,
        "optics": tr.OPTICS,
        "hdbscan": tr.HDBSCAN,
    }
    return classes[recipe](**args)


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
    recipe_fields(recipe)  # Validate the algorithm before doing any work.
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
