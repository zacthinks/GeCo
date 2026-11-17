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
            predictor_id = _positive_id(reference.get("predictor_id"), "predictor_id")
            if not any(int(row["classifier_spec_id"]) == predictor_id
                       for row in project.classifier_specs(code_id)):
                raise ValueError("Selected classifier is unavailable for this Code.")
            result["predictor_id"] = predictor_id
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
        if descriptor["source"] != "labels":
            state = project.classifier_status(
                code_id=code_id, classifier_spec_ids=[descriptor["predictor_id"]]
            )["classifiers"][0]
            fit_id = state["classifier_fit_id"]
            status = state["status"]
            if fit_id is None:
                warnings.append(f"{name}: selected classifier is not trained.")
            elif status != "current":
                warnings.append(f"{name}: selected classifier is stale.")
        variable = {**descriptor, "fit_id": fit_id if fit_id is not None else -1}
        join, expression, args = _code_join(variable, index)
        joins.append(join)
        expressions.append(f"({expression}) AS c{index}")
        params.extend(args)
        code_info.append({
            "name": name, "status": status, "fit_id": fit_id, "warnings": warnings,
        })

    # One bounded SQL projection for all requested Code sources and metadata.
    # No source texts, full unit objects, geometry matrices, or complete corpus
    # prediction lists are loaded. The joins use v1's human-first SQL semantics.
    fields = ["u.observation_id", "u.metadata_json", *expressions]
    for index, descriptor in enumerate(codes):
        fields += [f"a{index}.observation_id AS a{index}_id",
                   f"a{index}.value AS a{index}_value"]
        if descriptor["source"] != "labels":
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
            provenance = {
                "code_id": descriptor["code_id"], "source": descriptor["source"],
                "representation": descriptor["value"],
                "classifier_spec_id": descriptor.get("predictor_id"),
                "classifier_fit_id": info["fit_id"], "fit_status": info["status"],
            }
            warnings.extend(info["warnings"])
            for observation_id in requested:
                row = rows_by_id[observation_id]
                raw = row[f"c{index}"]
                if raw is not None:
                    numeric = float(raw)
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
