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


# These are presentation defaults only, not TeAL scientific eligibility rules.
# Every other constructor parameter remains editable under Advanced parameters.
PRIMARY_PARAMETERS: dict[str, tuple[str, ...]] = {
    "identity": (),
    "lda": ("n_components", "learning_method", "max_iter"),
    "gaussian_mixture": ("n_components", "covariance_type", "max_iter"),
    "kmeans": ("n_clusters", "init", "max_iter"),
    "minibatch_kmeans": ("n_clusters", "batch_size", "max_iter"),
    "spherical_kmeans": ("n_clusters", "n_init", "max_iter"),
    "kmedoids": ("n_clusters", "metric", "max_iter"),
    "spectral": ("n_clusters", "affinity", "n_neighbors"),
    "agglomerative": ("n_clusters", "linkage", "metric"),
    "dbscan": ("eps", "min_samples", "metric"),
    "optics": ("min_samples", "metric", "xi"),
    "hdbscan": ("min_cluster_size", "min_samples", "metric"),
}


# User-facing categorical controls. The model constructors remain the authority
# for parameter names and defaults; GeCo supplies human-friendly choices for
# well-known finite settings rather than asking researchers to guess strings.
# Metric menus offer common *feature-vector* distances, not "precomputed".
CATEGORICAL_CHOICES: dict[str, dict[str, tuple[str, ...]]] = {
    "lda": {"learning_method": ("batch", "online")},
    "gaussian_mixture": {
        "covariance_type": ("full", "tied", "diag", "spherical"),
        "init_params": ("kmeans", "k-means++", "random", "random_from_data"),
    },
    "kmeans": {"init": ("k-means++", "random")},
    "minibatch_kmeans": {"init": ("k-means++", "random")},
    "spherical_kmeans": {"init": ("k-means++", "random")},
    "kmedoids": {
        "metric": ("euclidean", "manhattan", "cosine"),
        "init": ("build", "random", "first"),
    },
    "spectral": {
        "affinity": ("rbf", "nearest_neighbors"),
        "assign_labels": ("kmeans", "discretize", "cluster_qr"),
        "eigen_solver": ("", "arpack", "lobpcg"),
    },
    "agglomerative": {
        "metric": ("euclidean", "manhattan", "cosine"),
        "linkage": ("ward", "complete", "average", "single"),
    },
    "dbscan": {
        "metric": ("euclidean", "manhattan", "cosine", "minkowski", "chebyshev"),
        "algorithm": ("auto", "ball_tree", "kd_tree", "brute"),
    },
    "optics": {
        "metric": ("minkowski", "euclidean", "manhattan", "cosine", "chebyshev"),
        "cluster_method": ("xi", "dbscan"),
        "algorithm": ("auto", "ball_tree", "kd_tree", "brute"),
    },
    "hdbscan": {
        "metric": ("euclidean", "manhattan", "cosine", "minkowski", "chebyshev"),
        "cluster_selection_method": ("eom", "leaf"),
        "algorithm": ("auto", "brute", "kd_tree", "ball_tree"),
    },
}

# Some parameters are intentionally numeric or mixed numeric/string ("auto").
# For those, retain the typed/text editor rather than presenting a misleading
# finite list. This catalog is UI guidance, not a numerical eligibility checker.
PARAMETER_LABELS = {"metric": "Distance metric"}


def _choice_options(recipe: str, name: str) -> list[dict[str, str]] | None:
    choices = CATEGORICAL_CHOICES.get(recipe, {}).get(name)
    if choices is None:
        return None
    return [
        {
            "label": (
                "Default" if choice == ""
                else "Auto" if choice == "auto"
                else "Euclidean" if choice == "euclidean"
                else "Manhattan" if choice == "manhattan"
                else "Cosine" if choice == "cosine"
                else "Minkowski" if choice == "minkowski"
                else "Chebyshev" if choice == "chebyshev"
                else choice.replace("_", " ").capitalize()
            ),
            "value": choice,
        }
        for choice in choices
    ]


def _with_control_metadata(recipe: str, spec: dict[str, Any]) -> dict[str, Any]:
    spec["label"] = PARAMETER_LABELS.get(spec["name"], spec["label"])
    options = _choice_options(recipe, spec["name"])
    if options is not None:
        spec["kind"] = "choice"
        spec["options"] = options
        if spec["default"] is None:
            spec["default"] = ""
        elif str(spec["default"]) not in {option["value"] for option in options}:
            raise ValueError(
                f"Categorical choices for {recipe}.{spec['name']} omit "
                f"its constructor default {spec['default']!r}."
            )
    return spec


# Display-only fallback for standalone GeCo: no TeAL fit is offered, but
# researchers can still inspect the wizard's primary controls.
# These defaults are NOT a substitute for TeAL's installed constructor API.
STANDALONE_PRIMARY_DEFAULTS: dict[str, Any] = {
    "n_components": 10, "n_clusters": 10, "learning_method": "batch",
    "max_iter": 100, "covariance_type": "full", "init": "k-means++",
    "batch_size": 1024, "n_init": 10, "metric": "euclidean",
    "affinity": "rbf", "n_neighbors": 10, "linkage": "ward",
    "eps": 0.5, "min_samples": 5, "xi": 0.05,
    "min_cluster_size": 10,
}


def standalone_primary_specs(recipe: str) -> list[dict[str, Any]]:
    """Illustrative fields for an unavailable optional TeAL integration."""
    if recipe not in PRIMARY_PARAMETERS:
        raise ValueError(f"Unknown Topic Model algorithm {recipe!r}.")
    specs = []
    for name in PRIMARY_PARAMETERS[recipe]:
        default = STANDALONE_PRIMARY_DEFAULTS[name]
        kind = "integer" if isinstance(default, int) else (
            "number" if isinstance(default, float) else "text"
        )
        specs.append(_with_control_metadata(recipe, {
            "name": name, "label": name.replace("_", " ").capitalize(),
            "kind": kind, "default": default, "annotation": type(default).__name__,
        }))
    return specs


def split_parameter_specs(recipe: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Divide *all* actual constructor parameters into prominent and advanced."""
    specs = parameter_specs(recipe)
    important = set(PRIMARY_PARAMETERS[recipe])
    return (
        [spec for spec in specs if spec["name"] in important],
        [spec for spec in specs if spec["name"] not in important],
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
    """Legacy TeAL-only geometry selector (retained for older callers)."""
    return topic_geometry_options(coder) if teal_linked_project(coder) is not None else []


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
        if name == "operator_id" or param.kind.name in {"VAR_POSITIONAL", "VAR_KEYWORD"}:
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
        specs.append(_with_control_metadata(recipe, {
            "name": name,
            "label": name.replace("_", " ").capitalize(),
            "kind": kind,
            "default": default,
            "annotation": annotation,
        }))
    return specs


def _parse_parameter(name: str, raw: Any, spec: Mapping[str, Any]) -> Any:
    """Only decode input widgets; the translator validates scientific inputs."""
    if raw is None or raw == "":
        return None
    kind = spec["kind"]
    annotation = str(spec["annotation"])
    if kind == "boolean":
        return raw is True or raw == "true"
    if kind == "choice":
        choices = {option["value"] for option in spec["options"]}
        if raw not in choices:
            raise ValueError(f"Choose a valid value for {spec['label']}.")
        return str(raw)
    if kind == "integer":
        return int(raw)
    if kind == "number":
        return float(raw)
    if "int" in annotation and "str" in annotation and str(raw).lstrip("+-").isdigit():
        return int(raw)
    if "float" in annotation:
        try:
            return float(raw)
        except (ValueError, TypeError):
            if "str" not in annotation:
                raise
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
