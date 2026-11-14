"""Persisted Davidson Scale lifecycle on top of GeCo's existing evidence and geometry APIs.

The estimator is an injectable numerical component. No numerical likelihood,
posterior mathematics, acquisition algorithm, or binary classifier code lives here.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from geometric_coder.scale_training import _atomic_positions, build_scale_training_data
from geometric_coder.exceptions import ConfigurationError


def _json_digest(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(document), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def configure_scale_model(
    project: Any, scale_id: int, representation_id: int, *,
    estimator_family: str = "linear_davidson",
    hyperparameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select an explicitly analytic registered geometry and an estimator config."""
    project.scale(int(scale_id))
    representation = project.database.get_representation(int(representation_id))
    if not representation["analytic_enabled"] or representation["legacy_kind"] != "geometry":
        raise ConfigurationError("Scale training requires an analytic geometry, not a display view.")
    if estimator_family != "linear_davidson":
        raise ConfigurationError("Only the initial linear Davidson family is supported.")
    params = dict(hyperparameters or {})
    try:
        json.dumps(params, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Scale hyperparameters must be JSON compatible.") from error
    if not all(isinstance(name, str) for name in params):
        raise ValueError("Scale hyperparameter keys must be strings.")
    return project.database.configure_scaler(
        int(scale_id), int(representation_id), estimator_family, params
    )


def _geometry_snapshot(project: Any, representation_id: int) -> dict[str, Any]:
    representation = project.database.get_representation(int(representation_id))
    if not representation["analytic_enabled"] or representation["legacy_kind"] != "geometry":
        raise ConfigurationError("Selected representation is not an analytic geometry.")
    geometry = project.database.get_geometry(int(representation["legacy_id"]))
    snapshot = {
        "representation_id": int(representation_id),
        "geometry_id": int(geometry["geometry_id"]),
        "representation_dimensions": representation["dimensions"],
        "geometry_created_at": geometry["created_at"],
        "atomic_row_count": project.database.unit_count(),
        "geometry_config": geometry["config"],
        "geometry_dependencies": geometry["dependency_names"],
        "storage_kind": geometry["storage_kind"],
        "external_ref": geometry["external_ref"],
        "matrix_path": geometry["matrix_path"],
        "matrix_sparse": geometry["matrix_sparse"],
        "state_path": geometry["state_path"],
    }
    # Metadata-only local dependency probe; never load the matrix during status checks.
    if geometry["storage_kind"] == "local":
        for item in ("matrix_path", "state_path"):
            relative = geometry[item]
            if relative:
                path = project.project_dir / relative
                try:
                    stat = path.stat()
                    snapshot[item + "_stat"] = [stat.st_size, stat.st_mtime_ns]
                except FileNotFoundError:
                    snapshot[item + "_stat"] = "missing"
    return snapshot


def _training_snapshot(
    project: Any, scale_id: int, spec: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scale = project.scale(int(scale_id))
    eligibility = project.scale_eligibility(int(scale_id))
    events = project.scale_training_comparisons(int(scale_id))
    evidence = [
        [
            int(event["comparison_event_id"]),
            int(event["observation_low_id"]),
            int(event["observation_high_id"]),
            str(event["canonical_outcome"]),
        ]
        for event in events
    ]
    snapshot = {
        "scale_id": int(scale_id),
        "scale_version_id": int(scale["scale_version_id"]),
        "scale_version_number": int(scale["version_number"]),
        "eligibility": eligibility,
        "geometry": _geometry_snapshot(project, int(spec["representation_id"])),
        # The reference sample is deterministic and frozen by these parameters.
        "reference_policy": {"method": "eligible_reservoir_seed_0", "max_rows": 2048},
        "scaler_spec_id": int(spec["scaler_spec_id"]),
        "estimator_family": str(spec["estimator_family"]),
        "hyperparameters": dict(spec["hyperparameters"]),
        "comparison_evidence": evidence,
    }
    return snapshot, events


def scale_model_status(project: Any, scale_id: int) -> dict[str, Any]:
    """Return current, stale, not-trained, or unsupported without loading geometry."""
    project.scale(int(scale_id))
    spec = project.database.get_scaler_spec(int(scale_id))
    last = project.database.latest_scaler_fit(int(scale_id))
    if spec is None:
        return {"status": "not_configured", "configuration": None, "fit": last}
    try:
        snapshot, events = _training_snapshot(project, scale_id, spec)
        fingerprint = _json_digest(snapshot)
        current = project.database.find_scaler_fit(int(scale_id), fingerprint)
        if current is not None and not (project.project_dir / current["artifact_path"]).is_file():
            current = None
        if current is not None:
            return {
                "status": "current", "configuration": spec, "fit": current,
                "comparison_count": len(events), "fingerprint": fingerprint,
            }
        return {
            "status": "stale" if last is not None else "not_trained",
            "configuration": spec, "fit": last,
            "comparison_count": len(events), "fingerprint": fingerprint,
        }
    except (KeyError, ValueError, ConfigurationError, OSError) as error:
        return {
            "status": "unsupported", "configuration": spec, "fit": last,
            "reason": str(error),
        }


def _production_estimator_factory(**parameters: Any) -> Any:
    try:
        from geometric_coder.davidson_scaler import LinearDavidsonScaler
    except ImportError as error:
        raise ConfigurationError(
            "Davidson numerical estimator is not installed yet. "
            "Merge Phase 2A into scaling before training a Scale."
        ) from error
    return LinearDavidsonScaler(**parameters)


def _select_scale_hyperparameters(
    training: Any, reference: Any, current: Mapping[str, Any],
    constructor: Callable[..., Any], *, folds: int = 5,
) -> dict[str, Any]:
    """Grouped pairwise CV using the three-outcome Davidson log score.

    Judgments on the same unordered pair stay in one fold. Unlabeled reference
    geometry is permitted in every fold but held-out outcomes are never fitted.
    """
    pairs = np.asarray(training.pairs, dtype=np.int64)
    outcomes = np.asarray(training.outcomes, dtype=np.int64)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, pair in enumerate(pairs):
        key = tuple(sorted((int(pair[0]), int(pair[1]))))
        groups.setdefault(key, []).append(index)
    keys = sorted(groups)
    actual_folds = min(int(folds), len(keys))
    fallback = {
        "tuned": False, "reason": "insufficient_cv_data", "folds": 0,
        "selected_hyperparameters": dict(current), "results": [],
    }
    if actual_folds < 2 or len(keys) < 3:
        return fallback
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(keys))
    partition = np.array_split(shuffled, actual_folds)
    splits = []
    for holdout in partition:
        held_keys = {keys[int(index)] for index in holdout}
        test = np.asarray([i for key in held_keys for i in groups[key]], dtype=int)
        train = np.asarray(
            [i for key in keys if key not in held_keys for i in groups[key]],
            dtype=int,
        )
        unique_obs = set(pairs[train].ravel().tolist())
        if (len(train) < 2 or len(unique_obs) < 3
                or not np.any(outcomes[train] != 1)):
            return fallback
        splits.append((train, test))
    candidates = [
        {**current, "score_prior_sd": prior, "feature_normalization": normalization}
        for prior in (0.25, 1.0, 4.0)
        for normalization in ("none", "rms")
    ]
    results = []
    for candidate in candidates:
        scores = []
        for train, test in splits:
            try:
                model = constructor(**candidate)
                model.fit(
                    training.features, pairs[train], outcomes[train],
                    reference_X=reference,
                )
                if not bool(getattr(model, "success_", True)):
                    raise RuntimeError("Optimization failed during CV.")
                probabilities = np.asarray(
                    model.predict_comparisons(training.features, pairs[test]),
                    dtype=float,
                )
                if probabilities.shape != (len(test), 3) or not np.isfinite(probabilities).all():
                    raise ValueError("Invalid held-out Davidson probabilities.")
                scores.append(float(-np.log(
                    np.clip(probabilities[np.arange(len(test)), outcomes[test]], 1e-15, 1)
                ).mean()))
            except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError):
                scores = []
                break
        if scores:
            results.append({
                "hyperparameters": candidate,
                "mean_log_loss": float(np.mean(scores)),
                "fold_scores": scores,
            })
    if not results:
        return {**fallback, "reason": "cv_candidates_failed"}
    selected = min(results, key=lambda row: row["mean_log_loss"])
    return {
        "tuned": True, "reason": None, "folds": actual_folds,
        "metric": "three_outcome_log_loss",
        "selected_hyperparameters": selected["hyperparameters"],
        "results": results,
    }


def train_scale_model(
    project: Any, scale_id: int, *,
    estimator_factory: Callable[..., Any] | None = None,
    tune: bool = False, retune_current: bool = False, folds: int = 5,
) -> dict[str, Any]:
    """Train explicitly or at a saved-evidence boundary; reuse an unchanged fit.

    Fitting always starts from the public effective-human-evidence API. A model
    with one directional comparison or exclusively parity evidence is not
    presented as learned; existing fitted state remains preserved.
    """
    if retune_current and not tune:
        raise ValueError("retune_current requires tune=True")
    status = scale_model_status(project, int(scale_id))
    if status["status"] == "current" and not (tune and retune_current):
        return {**status, "reused": True, "training_selection": {
            "tuned": False, "reason": "current_fit_reused", "folds": 0,
        }}
    spec = status["configuration"]
    if spec is None:
        raise ConfigurationError("Select an analytic geometry before training this Scale.")
    if status["status"] == "unsupported":
        raise ConfigurationError(str(status.get("reason", "Unsupported Scale configuration.")))
    snapshot, events = _training_snapshot(project, scale_id, spec)
    distinct = {
        int(obs)
        for event in events
        for obs in (event["observation_low_id"], event["observation_high_id"])
    }
    directional = sum(event["canonical_outcome"] != "parity" for event in events)
    if len(events) < 2 or len(distinct) < 3 or not directional:
        return {
            **status, "status": "insufficient_evidence", "reused": False,
            "reason": (
                "Training requires at least two usable judgments, three distinct "
                "observations, and at least one directional judgment. "
                "Only parity or one comparison cannot establish a meaningful Scale."
            ),
        }

    geometry_id = int(snapshot["geometry"]["geometry_id"])
    training = build_scale_training_data(
        project, int(scale_id), geometry_id, comparisons=events
    )
    constructor = estimator_factory or _production_estimator_factory
    reference_positions, eligible_count = project.database.scale_reference_positions(
        int(scale_id), max_rows=2048
    )
    if not reference_positions:
        raise ConfigurationError("This Scale has no eligible reference observations.")
    reference = project.geometry_page_rows(geometry_id, reference_positions)
    parameters = dict(spec["hyperparameters"])
    selection: dict[str, Any] = {
        "tuned": False, "reason": "tuning_disabled", "folds": 0,
        "selected_hyperparameters": parameters,
    }
    if tune:
        selection = _select_scale_hyperparameters(
            training, reference, parameters, constructor, folds=int(folds)
        )
        parameters = dict(selection["selected_hyperparameters"])
        if parameters != dict(spec["hyperparameters"]):
            # Persist the selected configuration, as classifier training does.
            project.configure_scale_model(
                int(scale_id), int(spec["representation_id"]),
                estimator_family=str(spec["estimator_family"]),
                hyperparameters=parameters,
            )
            spec = project.database.get_scaler_spec(int(scale_id))
            assert spec is not None
            snapshot, _ = _training_snapshot(project, scale_id, spec)
    model = constructor(**parameters)
    model.fit(
        training.features, training.pairs, training.outcomes,
        reference_X=reference,
    )
    if not bool(getattr(model, "success_", True)):
        raise RuntimeError(
            f"Scale optimization failed: {getattr(model, 'message_', 'unknown reason')}"
        )
    posterior_error = None
    if callable(getattr(model, "build_posterior", None)):
        try:
            model.build_posterior()
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
            posterior_error = str(error)

    diagnostics = {
        "comparison_count": training.n_comparisons,
        "directional_count": directional,
        "parity_count": training.n_comparisons - directional,
        "distinct_observations": len(training.observation_ids),
        "geometry_dimensions": int(training.features.shape[1]),
        "max_evidence_rank": min(training.n_comparisons, int(training.features.shape[1])),
        "optimizing_success": True,
        "optimization_message": str(getattr(model, "message_", "")),
        "optimization_iterations": getattr(model, "n_iterations_", None),
        "optimality": getattr(model, "optimality_", None),
        "posterior_available": getattr(model, "posterior_", None) is not None,
        "reference_observation_count": len(reference_positions),
        "reference_population_count": eligible_count,
        "reference_method": "eligible_reservoir_seed_0",
        "training_selection": selection,
        "posterior_error": posterior_error,
        "interpretation": "Exploratory, model-dependent scalar projection; not validated measurement.",
    }
    fingerprint = _json_digest(snapshot)
    path: str | None = None
    try:
        # Unique sidecar path prevents a concurrent/failed fit replacing valid data.
        path = project.artifacts.save_model(
            f"scaler_{scale_id}_{fingerprint[:16]}_{uuid.uuid4().hex}",
            (
                {"estimator_family": "linear_davidson", "state": model.to_state()}
                if callable(getattr(model, "to_state", None)) else model
            ),
        )
        saved = project.database.register_scaler_fit(
            scale_id=int(scale_id), scaler_spec_id=int(spec["scaler_spec_id"]),
            scale_version_id=int(snapshot["scale_version_id"]),
            fingerprint=fingerprint, snapshot=snapshot,
            artifact_path=path, diagnostics=diagnostics,
        )
    except BaseException:
        if path is not None:
            (project.project_dir / path).unlink(missing_ok=True)
        raise
    if saved["artifact_path"] != path:
        # Another publisher won the transaction.
        (project.project_dir / path).unlink(missing_ok=True)
    else:
        for retired in saved.get("_retired_artifact_paths", []):
            (project.project_dir / retired).unlink(missing_ok=True)
    return {
        **scale_model_status(project, int(scale_id)),
        "reused": False, "training_selection": selection,
    }


def scale_model_scores(
    project: Any, scale_id: int, observation_ids: Sequence[int], *,
    allow_stale: bool = False,
    include_uncertainty: bool = False,
) -> list[dict[str, Any]]:
    """Score only selected atomic observations with a retained frozen estimator."""
    state = scale_model_status(project, int(scale_id))
    if state["status"] != "current" and not (allow_stale and state["status"] == "stale"):
        raise ConfigurationError(
            f"Scale model is {state['status']}; train a current model before scoring."
        )
    fit = state["fit"]
    if fit is None or not (project.project_dir / fit["artifact_path"]).is_file():
        raise ConfigurationError("The saved fitted artifact is missing.")
    frozen_geometry = fit["snapshot"]["geometry"]
    if state["status"] == "stale" and frozen_geometry != _geometry_snapshot(
        project, int(frozen_geometry["representation_id"])
    ):
        raise ConfigurationError("A stale fit cannot score against changed geometry.")
    selected = list(dict.fromkeys(int(value) for value in observation_ids))
    if not selected:
        return []
    positions = _atomic_positions(project, selected)
    model = project.artifacts.load_model(project.project_dir, fit["artifact_path"])
    if isinstance(model, dict) and model.get("estimator_family") == "linear_davidson":
        from geometric_coder.davidson_scaler import LinearDavidsonScaler
        model = LinearDavidsonScaler.from_state(model["state"])
    matrix = project.geometry_page_rows(
        int(frozen_geometry["geometry_id"]), [positions[i] for i in selected]
    )
    if matrix.shape[0] != len(selected):
        raise ValueError("Geometry provider returned the wrong number of rows.")
    scores = np.asarray(model.predict_scores(matrix), dtype=float).reshape(-1)
    if scores.size != len(selected) or not np.isfinite(scores).all():
        raise ValueError("Estimator returned invalid Scale scores.")
    uncertainty: np.ndarray | None = None
    if include_uncertainty and callable(getattr(model, "posterior_position_variance", None)):
        try:
            variances = np.asarray(
                model.posterior_position_variance(matrix), dtype=float
            ).reshape(-1)
            if variances.shape == scores.shape and np.isfinite(variances).all():
                uncertainty = np.sqrt(np.maximum(variances, 0.0))
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            uncertainty = None
    return [
        {
            "observation_id": observation_id,
            "score": float(scores[index]),
            "epistemic_sd": (
                None if uncertainty is None else float(uncertainty[index])
            ),
            "scaler_fit_id": int(fit["scaler_fit_id"]),
            "fit_status": state["status"],
        }
        for index, observation_id in enumerate(selected)
    ]
