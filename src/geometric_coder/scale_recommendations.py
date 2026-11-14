"""Fit-triggered frozen Scale EIG rankings and read-only recommendation retrieval.

Candidate generation and budget accounting are delegated to candidate_sampling.
No numerical fit occurs during recommendation retrieval.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any

import numpy as np

from geometric_coder.scale_acquisition import score_davidson_pairs
from geometric_coder.scale_training import _atomic_positions


def _load_estimator(project: Any, fit: dict[str, Any]) -> Any:
    from geometric_coder.davidson_scaler import LinearDavidsonScaler

    saved = project.artifacts.load_model(project.project_dir, fit["artifact_path"])
    if isinstance(saved, dict) and saved.get("estimator_family") == "linear_davidson":
        return LinearDavidsonScaler.from_state(saved["state"])
    return saved


def _fit_supports_acquisition(state: dict[str, Any]) -> tuple[bool, str | None]:
    if state["status"] != "current" or state["fit"] is None:
        return False, "no_current_fit"
    diagnostics = state["fit"].get("diagnostics", {})
    if (int(diagnostics.get("comparison_count", 0)) < 6
            or int(diagnostics.get("directional_count", 0)) < 3
            or int(diagnostics.get("distinct_observations", 0)) < 4):
        return False, "insufficient_directional_evidence"
    if not diagnostics.get("posterior_available", False):
        return False, "posterior_unavailable"
    return True, None


def refresh_scale_acquisition(
    project: Any, scale_id: int, *,
    budget: int | None = None, seed: int = 0,
) -> dict[str, Any]:
    """Refresh once after an accepted fit, not during Next pair.

    The sampling contract is owned by Phase 3A. Score at most the supplied
    finite candidate budget; if posterior scoring fails, retain the sampled
    pairs in their seeded random order with an inspectable fallback reason.
    """
    from geometric_coder.candidate_sampling import prepare_candidate_run
    from geometric_coder.scale_models import scale_model_status

    state = scale_model_status(project, int(scale_id))
    fit = state.get("fit")
    if state["status"] != "current" or fit is None:
        return {"strategy": "random", "reason": "no_current_fit", "run_id": None}
    fit_id = int(fit["scaler_fit_id"])
    spec_id = int(fit["scaler_spec_id"])
    version_id = int(fit["scale_version_id"])
    run = prepare_candidate_run(
        project, int(scale_id), budget=budget, seed=int(seed),
        source_fit_id=fit_id, source_configuration_id=spec_id,
    )
    supported, reason = _fit_supports_acquisition(state)
    estimator = None
    geometry_id = int(fit["snapshot"]["geometry"]["geometry_id"])
    if supported:
        try:
            estimator = _load_estimator(project, fit)
            if getattr(estimator, "posterior_", None) is None:
                supported, reason = False, "posterior_unavailable"
            elif not bool(getattr(estimator, "success_", True)):
                supported, reason = False, "fit_unsuccessful"
            elif not callable(getattr(estimator, "pair_posterior_moments", None)):
                supported, reason = False, "posterior_moments_unavailable"
        except (ArithmeticError, ValueError, RuntimeError, OSError, KeyError) as error:
            supported, reason = False, "posterior_load_failed"
    # Sparse local .npz reads decompress the full CSR matrix. Load it once
    # per acquisition refresh, then slice candidate rows in memory. Dense
    # local geometries stay page/mmap-backed; external providers remain paged.
    from scipy import sparse
    geometry = project.database.get_geometry(geometry_id)
    cached_sparse = None
    if (supported and geometry["storage_kind"] == "local"
            and geometry["matrix_sparse"]):
        cached_sparse = project.geometry_matrix(geometry_id)
        if not sparse.issparse(cached_sparse):
            raise ValueError("Configured local sparse geometry is not a sparse matrix.")
    accumulated: list[tuple[int, int, float | None]] = []
    for batch in run.batches():
        if not supported:
            accumulated.extend((a, b, None) for a, b in batch.tasks)
            # A seeded random selection is itself the acquisition evaluation.
            # Acknowledging zero would ask the shared sampler to exhaust the
            # entire O(N^2) space attempting to fulfill an EIG-only budget.
            run.acknowledge_evaluated(batch)
            continue
        try:
            ids = sorted({obs for pair in batch.tasks for obs in pair})
            positions = _atomic_positions(project, ids)
            row_positions = [positions[obs] for obs in ids]
            matrix = (cached_sparse[row_positions] if cached_sparse is not None
                      else project.geometry_page_rows(geometry_id, row_positions))
            row_of = {obs: i for i, obs in enumerate(ids)}
            local_pairs = np.asarray(
                [(row_of[a], row_of[b]) for a, b in batch.tasks],
                dtype=np.int64,
            )
            gains = score_davidson_pairs(estimator, matrix, local_pairs)
            if len(gains) != len(batch.tasks) or not np.isfinite(gains).all():
                raise ArithmeticError("invalid EIG candidate scores")
            accumulated.extend(
                (a, b, float(gain))
                for (a, b), gain in zip(batch.tasks, gains, strict=True)
            )
            run.acknowledge_evaluated(batch)
        except (ArithmeticError, ValueError, RuntimeError, OSError, KeyError,
                np.linalg.LinAlgError):
            # A partially evaluated set must not masquerade as a valid
            # global EIG ranking. Preserve sampler order for all candidates.
            supported, reason = False, "eig_calculation_failed"
            accumulated.extend((a, b, None) for a, b in batch.tasks)
            # Remaining tasks have received random acquisition, not EIG.
            # Credit them as acquired so the finite task budget still bounds work.
            run.acknowledge_evaluated(batch)
    if supported:
        accumulated.sort(key=lambda item: (-float(item[2]), item[0], item[1]))
    else:
        accumulated = [(a, b, None) for a, b, _ in accumulated]

    provenance = run.provenance()
    fingerprint = hashlib.sha256(json.dumps({
        "fit": fit_id, "spec": spec_id, "version": version_id,
        "budget": run.requested_budget, "seed": seed,
        "eligibility": run.eligibility_context,
        "sampled": [run.eligibility_fingerprint],
        "method": "joint_laplace_gh_7x7_v1" if supported else "random_v1",
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    # The storage layer invokes this after acquiring BEGIN IMMEDIATE, before
    # replacing any ranking. Separate read connections see the latest
    # committed state, while competing SQLite writers are blocked.
    def check_publication_freshness() -> None:
        run.assert_current()
        latest = scale_model_status(project, int(scale_id))
        if (latest["status"] != "current" or latest.get("fit") is None
                or int(latest["fit"]["scaler_fit_id"]) != fit_id):
            raise RuntimeError("Scaler fit changed during acquisition; discard ranking.")
        current = project.database.scale_acquisition_run(int(scale_id))
        if current is not None and current["scaler_fit_id"] > fit_id:
            raise RuntimeError("A newer fit already published an acquisition ranking.")

    summary = project.database.store_scale_acquisition(
        scale_id=int(scale_id), fit_id=fit_id, scale_version_id=version_id,
        spec_id=spec_id, fingerprint=fingerprint,
        provenance=provenance,
        diagnostics={
            "scoring_method": "joint_laplace_gh_7x7_v1" if supported else None,
            "quadrature_order": 7,
            "evaluated_candidate_count": run.evaluated_count,
            "sampled_candidate_count": len(accumulated),
            "fallback_reason": reason,
        },
        strategy="eig" if supported else "random",
        reason=reason, ranked_pairs=accumulated,
        publication_guard=check_publication_freshness,
    )
    return {**summary, "reason": reason, "provenance": provenance}


def scale_acquisition_status(project: Any, scale_id: int) -> dict[str, Any]:
    """Inspect the frozen ranking without training or re-scoring."""
    from geometric_coder.scale_models import scale_model_status, _geometry_snapshot
    run = project.database.scale_acquisition_run(int(scale_id))
    status = scale_model_status(project, int(scale_id))
    if run is None:
        return {"strategy": "random", "reason": "no_frozen_ranking", "run_id": None}
    fit = status.get("fit")
    if fit is None or int(fit["scaler_fit_id"]) != run["scaler_fit_id"]:
        return {"strategy": "random", "reason": "fit_changed", "run_id": None}
    current_scale = project.scale(int(scale_id))
    config = status.get("configuration")
    snapshot = fit["snapshot"]
    frozen_geometry = snapshot["geometry"]
    if (int(current_scale["scale_version_id"]) != run["scale_version_id"]
            or int(fit["scaler_spec_id"]) != run["scaler_spec_id"]
            or snapshot["eligibility"] != project.scale_eligibility(int(scale_id))
            or config is None
            or int(config["representation_id"]) != int(frozen_geometry["representation_id"])
            or str(config["estimator_family"]) != str(snapshot["estimator_family"])
            or config["hyperparameters"] != snapshot["hyperparameters"]):
        return {"strategy": "random", "reason": "definition_or_configuration_changed", "run_id": None}
    try:
        if frozen_geometry != _geometry_snapshot(
            project, int(frozen_geometry["representation_id"])
        ):
            return {"strategy": "random", "reason": "geometry_changed", "run_id": None}
    except (KeyError, ValueError, OSError):
        return {"strategy": "random", "reason": "geometry_unavailable", "run_id": None}
    if status["status"] not in {"current", "stale"}:
        return {"strategy": "random", "reason": "fit_unavailable", "run_id": None}
    return {
        **run, "fit_status": status["status"], "frozen": status["status"] == "stale",
    }


def next_scale_recommendation(
    project: Any, scale_id: int, *, seed: int | None = None,
) -> dict[str, Any]:
    """Return the top still-eligible, unjudged frozen pair; never fit.

    When no valid ranking remains, use Phase 3A seeded sampling as fallback.
    """
    from geometric_coder.candidate_sampling import prepare_candidate_run

    state = scale_acquisition_status(project, int(scale_id))
    run_id = state.get("run_id")
    if run_id is not None:
        offset = 0
        while True:
            page = project.database.scale_acquisition_page(
                int(run_id), offset=offset, limit=128
            )
            if not page:
                break
            pairs = [
                (int(row["observation_low_id"]), int(row["observation_high_id"]))
                for row in page
            ]
            judged = project.database.scale_judged_candidate_pairs(
                int(scale_id), pairs
            )
            remaining = [pair for pair in pairs if pair not in judged]
            if remaining:
                eligible_ids = project.eligible_scale_observation_ids(
                    int(scale_id), [obs for pair in remaining for obs in pair]
                )
                eligible = set(eligible_ids)
                for low, high in remaining:
                    if low in eligible and high in eligible:
                        return {
                            "pair": (low, high), "strategy": state["strategy"],
                            "fit_status": state["fit_status"],
                            "fallback_reason": state["reason"],
                            "run_id": run_id,
                            "seed": state["provenance"].get("random_seed"),
                        }
            offset += len(page)
    effective_seed = secrets.randbits(63) if seed is None else int(seed)
    run = prepare_candidate_run(
        project, int(scale_id), budget=1, seed=effective_seed
    )
    for batch in run.batches():
        if batch.tasks:
            return {
                "pair": tuple(batch.tasks[0]),
                "strategy": "random",
                "fallback_reason": (
                    "frozen_ranking_exhausted" if run_id is not None else state["reason"]
                ),
                "run_id": None, "seed": effective_seed,
            }
    raise ValueError("No unjudged eligible Scale pair remains in the candidate pool.")
