"""Bounded, key-aligned analytical variables for future chart-role consumers.

This is an additive internal API. Saved Analysis schema v1 and its SQL
aggregations intentionally continue to use analyses.py / analysis_matrix.py.
An analytical variable describes a *source and original value*, never a
chart-specific bin, aggregation, or inferred human Scale label.

Call validate_variable(project, reference) for a canonical descriptor, then
resolve_analytical_variables(project, references, observation_ids) for aligned
columns. Results contain one value and one optional missing reason for each
requested atomic observation ID, *including duplicates and caller order*.
The caller must page IDs; no full corpus or text is loaded into chart state.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from geometric_coder.analyses import _code_join
from geometric_coder.apply import aggregate_probabilities
from geometric_coder.project import FIXED_COMMITTEE_AGGREGATIONS
from geometric_coder.exceptions import ConfigurationError

MAX_RESOLUTION_OBSERVATIONS = 2048
_SQL_BATCH_SIZE = 800


def _positive_id(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def validate_variable(project: Any, reference: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a source/value reference, independently of any chart operation.

    Code: source=labels/predictions/hybrid, value=binary/probability. Human
    labels have no probability representation; hybrid probability uses 1/0
    for human Present/Absent and predictor probabilities for unreviewed units.
    Scale: selected scale_id, value=position only (trained model predictions).
    Metadata: field and value=categorical/numeric, optionally an explicit
    numeric_strings opt-in; strings are never coerced by default.
    """
    if not isinstance(reference, Mapping):
        raise ValueError("Analytical variable must be an object.")
    family = reference.get("type")
    if family == "code":
        code_id = _positive_id(reference.get("code_id"), "code_id")
        try:
            project.code(code_id)
        except KeyError as error:
            raise ValueError(f"Unknown Code ID: {code_id}") from error
        source = reference.get("source", "labels")
        value = reference.get("value", "binary")
        if source not in ("labels", "predictions", "hybrid"):
            raise ValueError("Unknown Code source.")
        if value not in ("binary", "probability"):
            raise ValueError("Unknown Code value representation.")
        if source == "labels" and value != "binary":
            raise ValueError("Human Code judgments are binary, not probabilities.")
        result = {"type": "code", "code_id": code_id, "source": source, "value": value}
        if source != "labels":
            predictor_kind = reference.get("predictor_kind", "classifier")
            if predictor_kind not in ("classifier", "committee"):
                raise ValueError("Unknown Code predictor kind.")
            predictor_id = _positive_id(reference.get("predictor_id"), "predictor_id")
            if predictor_kind == "classifier":
                available = any(
                    int(row["classifier_spec_id"]) == predictor_id
                    for row in project.classifier_specs(code_id)
                )
            else:
                available = any(
                    int(row["committee_id"]) == predictor_id
                    for row in project.classifier_committees(code_id)
                )
            if not available:
                raise ValueError("Selected predictor is unavailable for this Code.")
            result.update(predictor_kind=predictor_kind, predictor_id=predictor_id)
        elif "predictor_kind" in reference or "predictor_id" in reference:
            raise ValueError("Human-only Code labels cannot select a predictor.")
        return result
    if family == "scale":
        scale_id = _positive_id(reference.get("scale_id"), "scale_id")
        try:
            project.scale(scale_id)
        except KeyError as error:
            raise ValueError(f"Unknown Scale ID: {scale_id}") from error
        if reference.get("value", "position") != "position":
            raise ValueError("Scales expose fitted positions only.")
        return {"type": "scale", "scale_id": scale_id, "value": "position"}
    if family == "metadata":
        field = reference.get("field")
        if not isinstance(field, str) or field not in project.metadata.get("metadata_columns", []):
            raise ValueError("Unknown metadata field.")
        value = reference.get("value", "categorical")
        if value not in ("categorical", "numeric"):
            raise ValueError("Metadata must be categorical or numeric.")
        numeric_strings = reference.get("numeric_strings", False)
        if not isinstance(numeric_strings, bool) or (numeric_strings and value != "numeric"):
            raise ValueError("numeric_strings is only valid for numeric metadata.")
        return {"type": "metadata", "field": field, "value": value,
                "numeric_strings": numeric_strings}
    raise ValueError("Unknown analytical variable family.")


def _metadata_value(raw: Any, descriptor: Mapping[str, Any]) -> tuple[Any, str | None]:
    """Interpret a single original JSON metadata value without guessing its type."""
    if raw is None:
        return None, "missing_metadata"
    if descriptor["value"] == "categorical":
        if isinstance(raw, (str, int, float, bool)):
            if isinstance(raw, float) and not math.isfinite(raw):
                return None, "invalid_metadata"
            return raw, None
        return None, "invalid_metadata"
    if isinstance(raw, bool):
        return None, "invalid_numeric"
    if isinstance(raw, (int, float)):
        try:
            number = float(raw)
        except (ValueError, OverflowError):
            return None, "invalid_numeric"
    elif isinstance(raw, str) and descriptor["numeric_strings"]:
        try:
            number = float(raw.strip()) if raw.strip() else math.nan
        except (ValueError, OverflowError):
            return None, "invalid_numeric"
    else:
        return None, "invalid_numeric"
    if not math.isfinite(number):
        return None, "invalid_numeric"
    return number, None


def _committee_join(
    code_id: int, index: int, fit_id: int | None, value: str, source: str,
) -> tuple[str, str, list[Any]]:
    """Read frozen learned-committee predictions; never train or materialize them."""
    a, p = f"a{index}", f"p{index}"
    joins = [
        f"LEFT JOIN current_annotations {a} "
        f"ON {a}.observation_id = u.observation_id AND {a}.code_id = ?",
        f"LEFT JOIN classifier_committee_predictions {p} "
        f"ON {p}.observation_id = u.observation_id AND {p}.committee_fit_id = ?",
    ]
    prediction = (
        f"{p}.probability" if value == "probability" else
        f"CASE {p}.predicted_label WHEN 'positive' THEN 1.0 "
        "WHEN 'negative' THEN 0.0 ELSE NULL END"
    )
    if source == "hybrid":
        human = f"CASE {a}.value WHEN 'positive' THEN 1.0 WHEN 'negative' THEN 0.0 ELSE NULL END"
        prediction = f"CASE WHEN {a}.observation_id IS NOT NULL THEN {human} ELSE {prediction} END"
    return " ".join(joins), prediction, [code_id, -1 if fit_id is None else fit_id]


def _fixed_committee_scores(
    project: Any, observation_ids: list[int], member_fit_ids: list[int],
    aggregation: str,
) -> tuple[dict[int, float], dict[int, str]]:
    """Aggregate only selected frozen member predictions without preparing rankings.

    The standard Apply aggregation function supplies all six fixed-committee
    operations. Missing member coverage must not be replaced by a partial mean.
    """
    if not member_fit_ids:
        return {}, {obs: "not_trained" for obs in observation_ids}
    by_observation: dict[int, dict[int, Any]] = {}
    with project.database.connect() as conn:
        for start in range(0, len(observation_ids), _SQL_BATCH_SIZE):
            obs_batch = observation_ids[start:start + _SQL_BATCH_SIZE]
            for offset in range(0, len(member_fit_ids), 100):
                fit_batch = member_fit_ids[offset:offset + 100]
                rows = conn.execute(
                    "SELECT observation_id, classifier_fit_id, probability "
                    "FROM classifier_predictions WHERE observation_id IN ("
                    + ",".join("?" for _ in obs_batch)
                    + ") AND classifier_fit_id IN ("
                    + ",".join("?" for _ in fit_batch) + ")",
                    [*obs_batch, *fit_batch],
                ).fetchall()
                for row in rows:
                    by_observation.setdefault(int(row["observation_id"]), {})[
                        int(row["classifier_fit_id"])
                    ] = row["probability"]
    valid: list[int] = []
    arrays: dict[str, list[float]] = {str(fit_id): [] for fit_id in member_fit_ids}
    missing: dict[int, str] = {}
    for observation_id in observation_ids:
        entries = by_observation.get(observation_id, {})
        if any(fit_id not in entries for fit_id in member_fit_ids):
            missing[observation_id] = "missing_prediction"
            continue
        if any(entries[fit_id] is None for fit_id in member_fit_ids):
            missing[observation_id] = "missing_probability"
            continue
        try:
            probabilities = [float(entries[fit_id]) for fit_id in member_fit_ids]
        except (ValueError, TypeError, OverflowError):
            missing[observation_id] = "invalid_probability_or_score"
            continue
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            missing[observation_id] = "invalid_probability_or_score"
            continue
        valid.append(observation_id)
        for fit_id, probability in zip(member_fit_ids, probabilities, strict=True):
            arrays[str(fit_id)].append(probability)
    if not valid:
        return {}, missing
    combined = aggregate_probabilities(arrays, aggregation=aggregation)
    return dict(zip(valid, map(float, combined), strict=True)), missing


def _code_missing(row: Any, index: int, descriptor: Mapping[str, Any],
                  fit_status: str) -> str:
    human = row[f"a{index}_value"]
    has_human = row[f"a{index}_id"] is not None
    if descriptor["source"] != "predictions" and has_human:
        return "unsure" if human == "unsure" else "invalid_human_label"
    if descriptor["source"] == "labels":
        return "unreviewed"
    if fit_status == "not_trained":
        return "not_trained"
    if row[f"p{index}_id"] is None:
        return "missing_prediction"
    return ("missing_probability" if descriptor["value"] == "probability"
            else "invalid_prediction")


def _column(descriptor: dict[str, Any], name: str, kind: str, values: list[Any],
            missing_reasons: list[str | None], provenance: dict[str, Any],
            warnings: list[str]) -> dict[str, Any]:
    reasons = dict(Counter(reason for reason in missing_reasons if reason is not None))
    missing = sum(reasons.values())
    return {
        "descriptor": descriptor, "name": name, "kind": kind,
        "values": values, "missing_reasons": missing_reasons,
        "coverage": {"observed": len(values) - missing, "missing": missing,
                     "by_reason": reasons},
        "provenance": provenance, "warnings": warnings,
    }


def resolve_analytical_variables(
    project: Any,
    references: Sequence[Mapping[str, Any]],
    observation_ids: Sequence[int],
    *,
    allow_stale_scales: bool = False,
) -> dict[str, Any]:
    """Resolve a bounded page of *atomic observation IDs* without writes.

    Unknown/non-atomic IDs fail rather than producing an incorrectly aligned
    row. Repeated IDs are retained in the output. Untrained/blocked models
    yield missing values and explicit warnings, never silent fallback to other
    models. The caller can request stale Scale positions explicitly; a stale
    fit with changed geometry remains unusable. Output columns are positional,
    so the same variable may appear in different chart roles later.
    """
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise ValueError("Provide a sequence of analytical variables.")
    if not isinstance(observation_ids, Sequence) or isinstance(observation_ids, (str, bytes)):
        raise ValueError("Provide a sequence of atomic observation IDs.")
    if not isinstance(allow_stale_scales, bool):
        raise ValueError("allow_stale_scales must be a boolean.")
    if not 1 <= len(references) <= 20:
        raise ValueError("Choose between 1 and 20 analytical variables.")
    requested = [_positive_id(value, "observation_id") for value in observation_ids]
    if len(requested) > MAX_RESOLUTION_OBSERVATIONS:
        raise ValueError(
            f"Resolve at most {MAX_RESOLUTION_OBSERVATIONS} observations per call; page the IDs."
        )
    selected = list(dict.fromkeys(requested))
    descriptors = [validate_variable(project, item) for item in references]
    codes = [v for v in descriptors if v["type"] == "code"]
    joins: list[str] = []
    expressions: list[str] = []
    params: list[Any] = []
    code_info: list[dict[str, Any]] = []
    for index, descriptor in enumerate(codes):
        code_id = descriptor["code_id"]
        name = str(project.code(code_id)["name"])
        fit_id = None
        status = "human_only"
        warnings: list[str] = []
        predictor_kind = descriptor.get("predictor_kind", "classifier")
        committee: dict[str, Any] | None = None
        member_fit_ids: list[int | None] = []
        if descriptor["source"] != "labels":
            if predictor_kind == "committee":
                state = project.classifier_committee_status(
                    code_id=code_id, committee_id=descriptor["predictor_id"]
                )
                committee = state["committee"]
                status = state["status"]
                if committee["aggregation"] in FIXED_COMMITTEE_AGGREGATIONS:
                    for member in committee["members"]:
                        latest = project.database.latest_classifier_fit(
                            code_id=code_id,
                            classifier_spec_id=int(member["classifier_spec_id"]),
                        )
                        member_fit_ids.append(
                            None if latest is None else int(latest["classifier_fit_id"])
                        )
                else:
                    fit = state["latest_fit"]
                    fit_id = None if fit is None else int(fit["committee_fit_id"])
                    if fit is not None:
                        member_fit_ids = [int(i) for i in fit["member_fit_ids"]]
            else:
                state = project.classifier_status(
                    code_id=code_id, classifier_spec_ids=[descriptor["predictor_id"]]
                )["classifiers"][0]
                fit_id = state["classifier_fit_id"]
                status = state["status"]
            if status == "not_trained":
                warnings.append(f"{name}: selected predictor is not trained.")
            elif status != "current":
                warnings.append(f"{name}: selected predictor is stale.")
        if committee is not None and committee["aggregation"] in FIXED_COMMITTEE_AGGREGATIONS:
            a = f"a{index}"
            join = (
                f"LEFT JOIN current_annotations {a} "
                f"ON {a}.observation_id = u.observation_id AND {a}.code_id = ?"
            )
            expression, args = "NULL", [code_id]
        elif committee is not None:
            join, expression, args = _committee_join(
                code_id, index, fit_id, descriptor["value"], descriptor["source"]
            )
        else:
            variable = {**descriptor, "fit_id": fit_id if fit_id is not None else -1}
            join, expression, args = _code_join(variable, index)
        joins.append(join)
        expressions.append(f"({expression}) AS c{index}")
        params.extend(args)
        code_info.append({
            "name": name, "status": status, "fit_id": fit_id, "warnings": warnings,
            "predictor_kind": predictor_kind, "committee": committee,
            "member_fit_ids": member_fit_ids,
        })

    # One bounded SQL projection for all requested Code sources and metadata.
    # No source texts, full unit objects, geometry matrices, or complete corpus
    # prediction lists are loaded. The joins use v1's human-first SQL semantics.
    fields = ["u.observation_id", "u.metadata_json", *expressions]
    for index, descriptor in enumerate(codes):
        fields += [f"a{index}.observation_id AS a{index}_id",
                   f"a{index}.value AS a{index}_value"]
        if descriptor["source"] != "labels" and not (
            code_info[index]["committee"] is not None
            and code_info[index]["committee"]["aggregation"] in FIXED_COMMITTEE_AGGREGATIONS
        ):
            fields.append(f"p{index}.observation_id AS p{index}_id")
    rows_by_id: dict[int, Any] = {}
    if selected:
        from_sql = " FROM units u " + " ".join(joins)
        with project.database.connect() as conn:
            for start in range(0, len(selected), _SQL_BATCH_SIZE):
                batch = selected[start:start + _SQL_BATCH_SIZE]
                markers = ",".join("?" for _ in batch)
                rows = conn.execute(
                    "SELECT " + ", ".join(fields) + from_sql
                    + f" WHERE u.observation_id IN ({markers})",
                    [*params, *batch],
                ).fetchall()
                rows_by_id.update((int(row["observation_id"]), row) for row in rows)
        if len(rows_by_id) != len(selected):
            missing = [value for value in selected if value not in rows_by_id]
            raise ValueError(f"Unknown or non-atomic observation IDs: {missing}")

    columns: list[dict[str, Any]] = []
    code_index = 0
    for descriptor in descriptors:
        values: list[Any] = []
        missing_reasons: list[str | None] = []
        warnings: list[str] = []
        provenance: dict[str, Any]
        kind: str
        if descriptor["type"] == "code":
            index = code_index
            code_index += 1
            info = code_info[index]
            kind = "binary" if descriptor["value"] == "binary" else "numeric"
            is_committee = (
                descriptor["source"] != "labels"
                and info["predictor_kind"] == "committee"
            )
            committee = info["committee"]
            fixed = (
                is_committee and committee is not None
                and committee["aggregation"] in FIXED_COMMITTEE_AGGREGATIONS
            )
            provenance = {
                "code_id": descriptor["code_id"], "source": descriptor["source"],
                "representation": descriptor["value"],
                "predictor_kind": info["predictor_kind"] if is_committee else (
                    "classifier" if descriptor["source"] != "labels" else None
                ),
                "classifier_spec_id": (
                    descriptor.get("predictor_id") if not is_committee else None
                ),
                "classifier_fit_id": (
                    info["fit_id"] if not is_committee else None
                ),
                "committee_id": descriptor.get("predictor_id") if is_committee else None,
                "committee_fit_id": (
                    info["fit_id"] if is_committee and not fixed else None
                ),
                "committee_aggregation": committee["aggregation"] if is_committee else None,
                "committee_updated_at": committee["updated_at"] if is_committee else None,
                "member_classifier_fit_ids": info["member_fit_ids"] if is_committee else None,
                "fit_status": info["status"],
            }
            warnings.extend(info["warnings"])
            fixed_values: dict[int, float] = {}
            fixed_missing: dict[int, str] = {}
            if fixed:
                member_fits = info["member_fit_ids"]
                if all(fit is not None for fit in member_fits):
                    fixed_values, fixed_missing = _fixed_committee_scores(
                        project, selected, [int(fit) for fit in member_fits],
                        committee["aggregation"],
                    )
            for observation_id in requested:
                row = rows_by_id[observation_id]
                if fixed:
                    if descriptor["source"] == "hybrid" and row[f"a{index}_id"] is not None:
                        human = row[f"a{index}_value"]
                        if human in ("positive", "negative"):
                            values.append(1.0 if human == "positive" else 0.0)
                            missing_reasons.append(None)
                        else:
                            values.append(None)
                            missing_reasons.append(
                                "unsure" if human == "unsure" else "invalid_human_label"
                            )
                        continue
                    if observation_id in fixed_values:
                        probability = fixed_values[observation_id]
                        values.append(
                            probability if descriptor["value"] == "probability"
                            else float(probability >= 0.5)
                        )
                        missing_reasons.append(None)
                    else:
                        values.append(None)
                        missing_reasons.append(
                            "not_trained" if info["status"] == "not_trained"
                            else fixed_missing.get(observation_id, "missing_prediction")
                        )
                    continue
                raw = row[f"c{index}"]
                if raw is not None:
                    try:
                        numeric = float(raw)
                    except (ValueError, TypeError, OverflowError):
                        numeric = math.nan
                    if (math.isfinite(numeric)
                            and (descriptor["value"] != "probability" or 0 <= numeric <= 1)):
                        values.append(numeric)
                        missing_reasons.append(None)
                        continue
                values.append(None)
                if raw is not None:
                    missing_reasons.append("invalid_probability_or_score")
                else:
                    missing_reasons.append(
                        _code_missing(row, index, descriptor, info["status"])
                    )
            name = info["name"]
        elif descriptor["type"] == "metadata":
            kind = descriptor["value"]
            name = descriptor["field"]
            provenance = {"field": name, "representation": descriptor["value"],
                          "numeric_strings": descriptor["numeric_strings"]}
            for observation_id in requested:
                metadata = json.loads(rows_by_id[observation_id]["metadata_json"])
                raw = metadata.get(name)
                value, reason = _metadata_value(raw, descriptor)
                values.append(value)
                missing_reasons.append(reason)
        else:
            kind = "numeric"
            scale_id = descriptor["scale_id"]
            scale = project.scale(scale_id)
            name = str(scale["name"])
            state = project.scale_model_status(scale_id)
            fit = state.get("fit")
            status = str(state["status"])
            fit_id = None if fit is None else int(fit["scaler_fit_id"])
            provenance = {
                "scale_id": scale_id, "representation": "position",
                "scale_version_id": int(scale["scale_version_id"]),
                "eligibility_revision": int(scale["eligibility_revision"]),
                "scaler_fit_id": fit_id, "fit_status": status,
            }
            eligible = set(project.eligible_scale_observation_ids(scale_id, selected))
            scored: dict[int, float] = {}
            blocked_reason: str | None = None
            if status != "current" and not (allow_stale_scales and status == "stale"):
                blocked_reason = status + "_fit"
                warnings.append(f"{name}: Scale model is {status}; positions unavailable.")
            elif eligible:
                try:
                    score_rows = project.scale_scores(
                        scale_id, [obs for obs in selected if obs in eligible],
                        allow_stale=allow_stale_scales,
                    )
                    for item in score_rows:
                        if (fit_id is None or int(item["scaler_fit_id"]) != fit_id
                                or item["fit_status"] != status):
                            raise ValueError("Scale fit identity changed during resolution.")
                        value = float(item["score"])
                        if not math.isfinite(value):
                            raise ValueError("Scale score is not finite.")
                        scored[int(item["observation_id"])] = value
                except (ConfigurationError, OSError) as error:
                    blocked_reason = "unavailable_fit"
                    warnings.append(f"{name}: selected Scale fit cannot score: {error}")
            if status == "stale" and blocked_reason is None:
                warnings.append(f"{name}: explicitly using a stale Scale fit.")
            for observation_id in requested:
                if observation_id not in eligible:
                    values.append(None)
                    missing_reasons.append("ineligible")
                elif blocked_reason is not None:
                    values.append(None)
                    missing_reasons.append(blocked_reason)
                elif observation_id not in scored:
                    values.append(None)
                    missing_reasons.append("missing_score")
                else:
                    values.append(scored[observation_id])
                    missing_reasons.append(None)
        columns.append(_column(
            descriptor, name, kind, values, missing_reasons, provenance, warnings
        ))
    return {
        "observation_ids": requested, "row_count": len(requested),
        "columns": columns,
        "warnings": [warning for col in columns for warning in col["warnings"]],
    }
