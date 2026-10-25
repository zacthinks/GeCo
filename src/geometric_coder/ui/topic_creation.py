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
