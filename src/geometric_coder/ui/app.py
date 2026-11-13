"""Dash application for GeCo's Explore, Develop, Apply/Review, Codes, and Memos workspaces."""

from __future__ import annotations

from pathlib import Path
from datetime import UTC, date, datetime
from html import escape as html_escape
from hashlib import blake2b
import json
import re
import uuid
from collections import deque
from collections.abc import Iterator, Sequence
from threading import Lock
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, quote, unquote

import numpy as np
import plotly.graph_objects as go

from geometric_coder.classifiers import (
    SUPPORTED_CLASSIFIER_ALGORITHMS,
    classifier_algorithm_label,
    hyperparameter_summary,
)
from geometric_coder.exceptions import ConfigurationError
from geometric_coder.ui.topic_explore import (load_topic_page_colors, normalize_color_mode, topic_legend_content)
from geometric_coder.explore import (
    InvalidRegexError,
    combine_filters,
    deterministic_page,
    deterministic_page_count,
    deterministic_page_rank,
    PageSlice,
    deterministic_page_with_inclusions,
    recommend_page_unit,
    recommend_unit,
)
from geometric_coder.ui.figures import (
    EXPLORE_ELIGIBLE_TRACE_INDEX,
    EXPLORE_FILTERED_TRACE_INDEX,
    EXPLORE_FOCAL_TRACE_INDEX,
    EXPLORE_TRACE_NAMES,
    EXPLORE_VISITED_TRACE_INDEX,
    build_geometry_figure,
)

if TYPE_CHECKING:
    from geometric_coder.project import GeometricCoder


GECO_DASH_THEME: dict[str, str] = {
    "--Dash-Stroke-Strong": "#65766b",
    "--Dash-Stroke-Weak": "#d7e2da",
    "--Dash-Fill-Interactive-Strong": "#2f744a",
    "--Dash-Fill-Interactive-Weak": "#e7f1ea",
    "--Dash-Fill-Inverse-Strong": "#ffffff",
    "--Dash-Text-Primary": "#17251c",
    "--Dash-Text-Strong": "#102a19",
    "--Dash-Text-Weak": "#607067",
    "--Dash-Text-Disabled": "#9aa69e",
    "--Dash-Fill-Primary-Hover": "#eef6f0",
    "--Dash-Fill-Primary-Active": "#dcece1",
    "--Dash-Fill-Disabled": "#eef1ef",
    "--Dash-Shading-Strong": "rgba(16, 42, 25, 0.24)",
    "--Dash-Shading-Weak": "rgba(16, 42, 25, 0.12)",
}

GECO_AG_GRID_THEME: dict[str, str] = {
    "function": """themeQuartz.withParams({
        accentColor: '#2f744a',
        backgroundColor: '#ffffff',
        foregroundColor: '#17251c',
        borderColor: '#d7e2da',
        headerBackgroundColor: '#eef6f0',
        headerTextColor: '#173a27',
        oddRowBackgroundColor: '#fafcfb',
        selectedRowBackgroundColor: 'rgba(47, 116, 74, 0.12)',
        inputBackgroundColor: '#ffffff',
        inputBorder: { color: '#a5c8af', style: 'solid', width: 1 },
        inputTextColor: '#17251c',
        menuBackgroundColor: '#ffffff',
        menuTextColor: '#17251c',
        borderRadius: 6,
        wrapperBorderRadius: 8,
        fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
        fontSize: 13,
        spacing: 6,
        rowVerticalPaddingScale: 0.55,
        headerVerticalPaddingScale: 0.75,
        listItemHeight: 30
    })"""
}

APPLY_DECISION_LABELS: dict[str, str] = {
    "pending": "Pending",
    "accept": "Accept proposal",
    "positive": "Positive",
    "negative": "Negative",
    "unsure": "Unsure",
    "unreviewed": "Leave unreviewed",
}
APPLY_DECISION_VALUES: dict[str, str] = {
    label: value for value, label in APPLY_DECISION_LABELS.items()
}

COMMITTEE_AGGREGATION_LABELS: dict[str, str] = {
    "mean": "Mean probability",
    "median": "Median probability (legacy)",
    "minimum": "Minimum probability",
    "maximum": "Maximum probability",
    "harmonic_mean": "Harmonic mean",
    "geometric_mean": "Geometric mean",
    "logistic_stack": "Logistic stacking",
}


def _is_semantic_modal_click(
    trigger: Any, changes: list[dict[str, Any]] | None,
) -> bool:
    """Only an actual positive click should load or delete a saved query.

    Dash's ALL-pattern inputs can fire when query rows first appear. Those
    newly mounted buttons have n_clicks=0 and must not dismiss the dialog.
    """
    if not isinstance(trigger, dict) or trigger.get("type") not in {
        "semantic-load-query", "semantic-delete-query",
    }:
        return False
    for change in changes or []:
        prop_id = str(change.get("prop_id", ""))
        if not prop_id.endswith(".n_clicks"):
            continue
        try:
            identifier = json.loads(prop_id.removesuffix(".n_clicks"))
        except (TypeError, ValueError):
            continue
        count = change.get("value")
        if identifier == trigger and isinstance(count, int) and count > 0:
            return True
    return False


# Search strings and topic-model artifacts are persistent project resources, but
# color selections, draft queries and similarity sliders belong to the current
# running Explore interaction. Legacy session states may contain these keys.
_TRANSIENT_EXPLORE_KEYS = frozenset({
    "semantic_query", "semantic_active_query", "semantic_range",
    "topic_color_mode", "topic_model_id", "topic_component_id",
})


def _semantic_vector_digest(scores: list[float]) -> str:
    """Identify the exact stored float64 score snapshot across cache eviction."""
    return blake2b(np.asarray(scores, dtype="<f8").tobytes(), digest_size=12).hexdigest()


def _current_semantic_scores(
    payload: dict[str, Any] | None, query: str | None,
    geometry_id: int | None,
) -> dict[str, list[float]]:
    """Only return scores that belong to the current query and geometry."""
    if not isinstance(payload, dict) or not query or geometry_id is None:
        return {}
    if (payload.get("query") != str(query).strip()
            or str(payload.get("geometry_id")) != str(geometry_id)):
        return {}
    return payload.get("scores") or {}


_DEFAULT_UI_SETTINGS = {"cv_folds": 5}


def _read_ui_settings(project_dir: Path) -> dict[str, int]:
    """Read modest project-scoped training defaults, preserving older projects."""
    path = project_dir / "ui_settings.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    try:
        folds = int(raw.get("cv_folds", 5))
    except (TypeError, ValueError):
        folds = 5
    return {"cv_folds": folds if 2 <= folds <= 10 else 5}


def _save_ui_settings(project_dir: Path, *, cv_folds: int) -> None:
    """Save the project CV default atomically; no database/schema migration."""
    folds = int(cv_folds)
    if not 2 <= folds <= 10:
        raise ValueError("Cross-validation folds must be between 2 and 10.")
    path = project_dir / "ui_settings.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"cv_folds": folds}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tooltip_label(html: Any, label: str, description: str) -> Any:
    """Attach low-clutter contextual help to the text itself."""
    return html.Span(
        label, className="geco-tooltip", tabIndex=0,
        **{"data-geco-tooltip": description},
    )


def _resumable_session_state(state: dict[str, Any]) -> dict[str, Any]:
    """Drop legacy transient color/search state without altering navigation."""
    return {key: value for key, value in state.items()
            if key not in _TRANSIENT_EXPLORE_KEYS}


def _selected_focus_construct_id(
    constructs: list[dict[str, Any]], state: dict[str, Any]
) -> int | None:
    """Restore shared selection, falling back to the legacy Code-only key."""
    by_id = {int(row["construct_id"]): str(row["type"]) for row in constructs}
    shared = state.get("focus_construct_id")
    if shared is None:
        legacy = state.get("focus_code_id")
        try:
            value = int(legacy)
        except (TypeError, ValueError):
            return None
        return value if by_id.get(value) == "code" else None
    try:
        value = int(shared)
    except (TypeError, ValueError):
        return None
    return value if value in by_id else None


def _construct_options(constructs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": f"{row['name']} · {'Scale' if row['type'] == 'scale' else 'Code'}",
            "value": int(row["construct_id"]),
        }
        for row in constructs
    ]


def _topic_restore_snapshot(session_id: int, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": int(session_id),
        "model_id": state.get("topic_model_id"),
        "component": state.get("topic_component_id"),
        "display": "intensity" if state.get("topic_color_mode") == "intensity" else "dominant",
    }


def _is_restored_topic_selection(
    model_id: int | None, display: str | None, component: str | None,
    snapshot: dict[str, Any] | None, session_id: int,
) -> bool:
    """Suppress topic activation from restored controls, including in-flight switches."""
    if not isinstance(snapshot, dict):
        return False
    if snapshot.get("session_id") != session_id:
        return True
    return (
        snapshot.get("model_id") == model_id
        and snapshot.get("display") == display
        and snapshot.get("component") == component
    )


def _auto_training_spec_scope(
    project: GeometricCoder, *, code_id: int,
    source: str, classifier_spec_id: int | None,
    committee_id: int | None, train_all: bool,
) -> list[int]:
    """Resolve an explicit scope at the coding-event boundary.

    A single active classifier or the selected committee's members is the
    default. Every classifier in the code is requested *only* when the UI's
    Train All choice was selected. A missing selection trains nothing.
    """
    if train_all:
        return [
            int(spec["classifier_spec_id"])
            for spec in project.database.list_classifier_specs(code_id=int(code_id))
        ]
    if source == "committee":
        if committee_id is None:
            return []
        committee = project.database.get_classifier_committee(int(committee_id))
        if int(committee["code_id"]) != int(code_id):
            raise ValueError("The selected committee belongs to another code.")
        return [
            int(member["classifier_spec_id"]) for member in committee["members"]
        ]
    if classifier_spec_id is None:
        return []
    spec = project.database.get_classifier_spec(int(classifier_spec_id))
    if int(spec["code_id"]) != int(code_id):
        raise ValueError("The selected classifier belongs to another code.")
    return [int(classifier_spec_id)]


def _post_saved_develop_judgment_training_request(
    project: GeometricCoder, *, code_id: int, annotation_event_id: int | None,
    auto_train_values: list[str] | None,
    train_scope_values: list[str] | None, tune_values: list[str] | None,
    recommendation_source: str | None,
    active_classifier_spec_id: int | None, committee_id: int | None,
) -> tuple[dict[str, Any] | None, str]:
    """Compose an optional training request *after* the judgment was persisted.

    Resolving a stale/invalid UI dropdown must never make a successful human
    annotation look like a failed save, or suppress the map/store refresh.
    """
    if "auto" not in set(auto_train_values or []):
        return None, "Judgment saved."
    try:
        selected_ids = _auto_training_spec_scope(
            project, code_id=int(code_id),
            source=str(recommendation_source or "classifier"),
            classifier_spec_id=active_classifier_spec_id,
            committee_id=committee_id,
            train_all="all" in set(train_scope_values or []),
        )
    except (KeyError, ValueError) as error:
        return (
            None,
            "Judgment saved, but automatic training was not requested "
            f"because the classifier selection is invalid: {error}",
        )
    if not selected_ids:
        return (
            None,
            "Judgment saved. Automatic training was not requested because "
            "no classifier or committee members are selected.",
        )
    return (
        {
            "code_id": int(code_id),
            "classifier_spec_ids": selected_ids,
            "tune": "tune" in set(tune_values or []),
            **({"annotation_event_id": int(annotation_event_id)}
               if annotation_event_id is not None else {}),
        },
        "Judgment saved; automatic training requested for the selected classifier(s).",
    )


def _post_saved_external_judgment_training_request(
    project: GeometricCoder, *, code_id: int,
    develop_code_id: int | None, annotation_event_id: int | None,
    auto_train_values: list[str] | None,
    train_scope_values: list[str] | None, tune_values: list[str] | None,
    recommendation_source: str | None,
    active_classifier_spec_id: int | None, committee_id: int | None,
) -> tuple[dict[str, Any] | None, str]:
    """Route an external judgment only to an explicitly selected Develop code.

    Explore/Code Center can assign any code; automatically training unrelated
    models just because a Develop selection exists would violate user scope.
    The saved judgment is acknowledged even if selection is absent or stale.
    """
    if "auto" not in set(auto_train_values or []):
        return None, "Judgment saved."
    if develop_code_id is None or int(code_id) != int(develop_code_id):
        return (
            None,
            "Judgment saved. Automatic training was not requested because "
            "this code is not the selected Develop code.",
        )
    return _post_saved_develop_judgment_training_request(
        project, code_id=int(code_id),
        annotation_event_id=annotation_event_id,
        auto_train_values=auto_train_values,
        train_scope_values=train_scope_values,
        tune_values=tune_values,
        recommendation_source=recommendation_source,
        active_classifier_spec_id=active_classifier_spec_id,
        committee_id=committee_id,
    )


def _post_saved_teaching_example_training_request(
    project: GeometricCoder, *, code_id: int, observation_id: int,
    develop_code_id: int | None, auto_train_values: list[str] | None,
    train_scope_values: list[str] | None, tune_values: list[str] | None,
    recommendation_source: str | None,
    active_classifier_spec_id: int | None, committee_id: int | None,
) -> tuple[dict[str, Any] | None, str]:
    """A saved teaching example is evidence, not an annotation event.

    Only the currently selected Develop code and selected model scope are
    eligible for automatic training. Retain teaching-observation provenance
    without fabricating an annotation-event ID.
    """
    request, status = _post_saved_external_judgment_training_request(
        project, code_id=int(code_id), develop_code_id=develop_code_id,
        annotation_event_id=None, auto_train_values=auto_train_values,
        train_scope_values=train_scope_values, tune_values=tune_values,
        recommendation_source=recommendation_source,
        active_classifier_spec_id=active_classifier_spec_id,
        committee_id=committee_id,
    )
    if request is not None:
        request["origin"] = "teaching_example"
        request["teaching_observation_id"] = int(observation_id)
    return request, status.replace("Judgment saved", "Teaching example saved")


def _teaching_example_for_code(
    project: GeometricCoder, *, code_id: int, observation_id: int,
) -> dict[str, Any] | None:
    """Validate a teaching table row against its owning code before mutation.

    The modal code and selected table row can become desynchronized after a
    code change or a preceding edit. Never write using a stale selection.
    """
    return next((
        row
        for row in project.teaching_examples(int(code_id), include_deleted=True)
        if int(row["observation_id"]) == int(observation_id)
    ), None)


def _post_saved_apply_batch_training_request(
    project: GeometricCoder, *, apply_run_id: int,
    mutation: str, changed_assignment_count: int, commit_operation_id: int,
    develop_code_id: int | None, auto_train_values: list[str] | None,
    train_scope_values: list[str] | None, tune_values: list[str] | None,
    recommendation_source: str | None,
    active_classifier_spec_id: int | None, committee_id: int | None,
) -> tuple[dict[str, Any] | None, str]:
    """Optional training after one *committed* Apply batch or audited undo.

    Proposal edits, bulk draft decisions, conflicts, leave-unreviewed and
    zero-change commits never enqueue fitting. The affected code is read from
    the persisted Apply run, not inferred from the currently selected UI code.
    """
    if mutation not in {"commit", "undo"}:
        raise ValueError("Unknown Apply assignment mutation.")
    if changed_assignment_count <= 0:
        return None, "No assignments changed; no automatic training requested."
    run = project.apply_run(int(apply_run_id))
    request, status = _post_saved_external_judgment_training_request(
        project, code_id=int(run["code_id"]),
        develop_code_id=develop_code_id, annotation_event_id=None,
        auto_train_values=auto_train_values,
        train_scope_values=train_scope_values, tune_values=tune_values,
        recommendation_source=recommendation_source,
        active_classifier_spec_id=active_classifier_spec_id,
        committee_id=committee_id,
    )
    label = "Apply commit saved" if mutation == "commit" else "Apply undo saved"
    if request is not None:
        request["origin"] = "apply"
        request["apply_run_id"] = int(apply_run_id)
        request["apply_mutation"] = mutation
        request["commit_operation_id"] = int(commit_operation_id)
    return request, status.replace("Judgment saved", label)


def _recommendation_matches_context(
    recommendation: dict[str, Any] | None, *,
    unit_id: int | None, code_id: int | None, session_id: int | None,
) -> bool:
    """Match saved recommendation context before showing or attributing scores."""
    if not isinstance(recommendation, dict) or any(
        value is None for value in (unit_id, code_id, session_id)
    ):
        return False
    try:
        return (
            int(recommendation["unit_id"]) == int(unit_id)
            and int(recommendation["code_id"]) == int(code_id)
            and int(recommendation["session_id"]) == int(session_id)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _matching_recommendation_fit_id(
    recommendation: dict[str, Any] | None, *,
    unit_id: int, code_id: int, session_id: int, selected_unit_ids: list[int],
) -> int | None:
    """Never attribute an unrelated recommendation fit to a manual judgment."""
    if (
        selected_unit_ids != [int(unit_id)]
        or not _recommendation_matches_context(
            recommendation, unit_id=unit_id, code_id=code_id, session_id=session_id,
        )
    ):
        return None
    try:
        fit_ids = (recommendation or {}).get("classifier_fit_ids") or {}
        if not isinstance(fit_ids, dict) or len(fit_ids) != 1:
            return None
        return int(next(iter(fit_ids.values())))
    except (TypeError, ValueError):
        return None


def _recommend_without_fitting(
    project: GeometricCoder, *,
    code_id: int, session_id: int, strategy: str,
    active_classifier_spec_id: int | None, committee_id: int | None,
    recommendation_source: str,
) -> dict[str, Any] | None:
    """UI navigation may read frozen scores but must never fit a model.

    Automatic training is an explicit post-evidence event, not a side effect of
    clicking a recommendation button. Programmatic callers of the project API
    retain their existing explicit auto_retrain option for compatibility.
    """
    return project.focus_recommendation(
        code_id=int(code_id),
        session_id=int(session_id),
        strategy=str(strategy),
        active_classifier_spec_id=active_classifier_spec_id,
        committee_id=committee_id,
        recommendation_source=str(recommendation_source),
        auto_retrain=False,
    )


class _ScopedTrainingInbox:
    """Retain each saved-judgment request and serialize synchronous fitting."""

    def __init__(self) -> None:
        self._guard = Lock()
        self._drain_lock = Lock()
        self._pending: deque[dict[str, Any]] = deque()

    def enqueue(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._guard:
            self._pending.append(dict(request))
            return {"nonce": str(uuid.uuid4())}

    def drain(self, runner: Any) -> list[tuple[dict[str, Any], Any, Exception | None]]:
        # Wait for an earlier drain: a second wake-up must not discard pending work.
        self._drain_lock.acquire()
        results: list[tuple[dict[str, Any], Any, Exception | None]] = []
        try:
            while True:
                with self._guard:
                    if not self._pending:
                        break
                    request = self._pending.popleft()
                try:
                    result = runner(request)
                except Exception as error:
                    results.append((request, None, error))
                else:
                    results.append((request, result, None))
        finally:
            self._drain_lock.release()
        return results


class _LazyUnits(Sequence[dict[str, Any]]):
    """Bounded document access for UI callbacks; no eager corpus fetch."""

    def __init__(self, database: Any) -> None:
        self.database = database
        self._count = database.unit_count()

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            positions = range(*index.indices(self._count))
            return self.database.unit_rows_at_positions(positions)
        position = int(index)
        if position < 0:
            position += self._count
        return self.database.unit_rows_at_positions([position])[0]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Stream in batches if a caller intentionally traverses all units.

        Sequence's default __iter__ repeatedly calls __getitem__, which would
        otherwise open a fresh SQLite connection for *every* corpus row.
        """
        for start in range(0, self._count, 900):
            yield from self.database.unit_rows_at_positions(
                range(start, min(self._count, start + 900))
            )

    def at_positions(self, positions: Sequence[int]) -> list[dict[str, Any]]:
        return self.database.unit_rows_at_positions(positions)

    def metadata_rows(self) -> Iterator[dict[str, Any]]:
        return self.database.iter_unit_metadata()


class _LazyTexts(Sequence[str]):
    """Only stream source text for an actual literal-search query."""

    def __init__(self, units: _LazyUnits) -> None:
        self.units = units

    def __len__(self) -> int:
        return len(self.units)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [unit["text"] for unit in self.units[index]]
        return self.units[index]["text"]

    def __iter__(self) -> Iterator[str]:
        return self.units.database.iter_unit_texts()


def create_app(
    project: GeometricCoder,
    *,
    mode: str = "standard",
    focus_session_id: int | None = None,
) -> Any:
    """Create the local GeCo Dash application.

    ``mode="focus"`` renders the finite Focus Coding surface over the same
    GeCo project/session state rather than maintaining a separate coding system.
    """
    if mode == "focus":
        from geometric_coder.ui.focus_coding import create_focus_coding_app

        return create_focus_coding_app(project, session_id=focus_session_id)
    if mode != "standard":
        raise ValueError("mode must be 'standard' or 'focus'")
    try:
        from dash import (
            ALL, Dash, Input, Output, Patch, State, callback_context, dash_table, dcc, html, no_update, set_props,
        )
        import dash_ag_grid as dag
    except ImportError as error:
        raise ConfigurationError(
            "The GeCo web interface is part of the standard installation, but a core "
            "UI dependency could not be imported. Reinstall geometric-coder with its "
            "default dependencies."
        ) from error

    from geometric_coder.ui.figures import one_dimensional_jitter_spread
    from geometric_coder.ui.representation_ui import representation_modal, register_representation_callbacks
    from geometric_coder.ui.topic_modeling import register_topic_details_callbacks
    from geometric_coder.ui.topic_creation import create_topic_model, topic_geometry_options, recipe_options, split_parameter_specs, standalone_primary_specs, teal_linked_project

    metadata = project.metadata
    units = _LazyUnits(project.database)
    # Atomic IDs are allocated sequentially from 1 and never deleted.
    unit_ids = range(1, len(units) + 1)
    texts = _LazyTexts(units)
    metadata_columns = [str(value) for value in metadata.get("metadata_columns", [])]
    # Representative bounded sample: exact corpus profiling can be added
    # separately without blocking a large project's startup.
    sample_positions = np.linspace(
        0, max(0, len(units) - 1),
        num=min(len(units), 2048), dtype=int,
    ).tolist()
    metadata_profiles = _build_metadata_profiles(
        project.database.unit_metadata_at_positions(sample_positions),
        metadata_columns,
    )
    saved_filter_sets_path = project.project_dir / "saved_filter_sets.json"
    geometries = project.geometries(public_only=True)
    geometry_by_id = {int(record["geometry_id"]): record for record in geometries}
    geometry_id_by_name = {str(record["name"]): int(record["geometry_id"]) for record in geometries}
    # The display surfaces share one descendant-aware compatibility path.
    views_by_geometry: dict[int, list[dict[str, Any]]] = {
        int(geometry["geometry_id"]): _display_view_records(
            project, int(geometry["geometry_id"])
        )
        for geometry in geometries
    }
    presentation_views = _presentation_view_records(project, geometries)
    presentation_view_by_id = {
        int(record["view_id"]): record for record in presentation_views
    }
    hierarchy = project.database.hierarchy_levels()
    hierarchy_summary = project.hierarchy_summary()
    codes = project.codes()
    constructs = project.constructs()
    classifier_specs = project.classifier_specs()
    prediction_geometry_cache: dict[str, dict[str, Any]] = {}
    # Only visible point indices are retained for Develop judgment patches.
    # Browser State carries a short token, never the complete Plotly figure.
    prediction_page_indexes: dict[str, tuple[int, str, dict[int, tuple[int, int]]]] = {}
    prediction_page_order: deque[str] = deque()
    prediction_page_lock = Lock()
    # Short-lived, app-local snapshots of the *visible* Explore points only.
    # A revision token matches a plot to its overlay callback, even when
    # callbacks complete out of order. No browser-to-server figure State.
    overlay_frames: dict[str, tuple[int, dict[str, Any]]] = {}
    overlay_frame_order: deque[str] = deque()
    overlay_frame_lock = Lock()

    # Retain similarity vectors on the server; the browser carries only a
    # query/geometry/query-id/token selector.  Bounded retention avoids a
    # permanent corpus-sized copy per past search.
    semantic_vectors: dict[str, tuple[str, int, int, list[float]]] = {}
    semantic_vector_order: deque[str] = deque()
    semantic_vector_lock = Lock()

    def selected_semantic_scores(
        payload: dict[str, Any] | None,
        query: str | None,
        geometry_id: int | None,
    ) -> dict[str, list[float]]:
        if not isinstance(payload, dict) or not query or geometry_id is None:
            return {}
        clean = str(query).strip()
        if payload.get("query") != clean or str(payload.get("geometry_id")) != str(geometry_id):
            return {}
        token = payload.get("token")
        if not isinstance(token, str) or not token:
            return _current_semantic_scores(payload, query, geometry_id)
        try:
            expected_query_id = int(payload["query_id"])
        except (KeyError, TypeError, ValueError):
            return {}
        # Check identity even on a cache hit: a deleted query must never be
        # revived by an older browser tab or a later query with the same text.
        if project.saved_semantic_searches.query_id(clean) != expected_query_id:
            return {}
        with semantic_vector_lock:
            retained = semantic_vectors.get(token)
        if retained is None:
            try:
                restored = project.saved_semantic_searches.find(
                    clean, int(geometry_id), expected_query_id=expected_query_id
                )
            except (ValueError, KeyError, ConfigurationError):
                return {}
            if restored is None:
                return {}
            # A query can be recomputed under the same query ID. An evicted
            # token must never silently resolve to a different score snapshot.
            if not token.endswith("." + _semantic_vector_digest(restored["scores"])):
                return {}
            retained = (clean, int(geometry_id), expected_query_id, restored["scores"])
            with semantic_vector_lock:
                semantic_vectors[token] = retained
                semantic_vector_order.append(token)
                while len(semantic_vector_order) > 8:
                    semantic_vectors.pop(semantic_vector_order.popleft(), None)
        if retained[:3] != (clean, int(geometry_id), expected_query_id):
            return {}
        name = str(geometry_by_id[int(geometry_id)]["name"])
        return {name: retained[3]}
    training_inbox = _ScopedTrainingInbox()

    sessions = project.sessions()
    initial_session = sessions[0]
    initial_state = _resumable_session_state(dict(initial_session["state"]))
    initial_geometry_id = _valid_geometry_id(initial_state.get("geometry_id"), geometries)
    initial_view_id = _valid_view_id(
        initial_state.get("view_id"),
        views_by_geometry.get(initial_geometry_id, []),
    )
    # Project-wide query catalog remains on disk; no search is active at startup.
    initial_semantic_query = ""
    initial_semantic_scores: dict[str, list[float]] = {}
    initial_semantic_status = ""
    initial_history = _normalize_navigation_history(
        initial_state.get("navigation_history"),
        focal_unit_id=initial_state.get("focal_unit_id"),
    )
    initial_explore_code_palette = _normalize_explore_code_palette(
        initial_state.get("explore_code_palette"),
        {int(code["code_id"]) for code in codes},
    )
    initial_seen_count = len(
        project.database.seen_unit_ids(int(initial_session["session_id"]))
    )
    initial_workspace = str(initial_state.get("workspace", "explore"))
    initial_focal_unit_id = initial_state.get("focal_unit_id")

    assets_folder = Path(__file__).parent / "assets"
    app = Dash(
        __name__,
        title="GeCo",
        update_title=None,
        assets_folder=str(assets_folder),
        suppress_callback_exceptions=True,
    )
    app.index_string = _geco_index_string()
    app.clientside_callback(
        """function(clicks) {
            if (!clicks) return window.dash_clientside.no_update;
            fetch('/_geco/close', {method: 'POST', credentials: 'same-origin'})
                .then(response => {
                    if (!response.ok) throw new Error('Close request failed');
                    document.body.innerHTML = '<main style="font:16px system-ui;padding:3rem">'
                      + '<h2>GeCo closed</h2><p>You can close this tab or relaunch GeCo from Python.</p></main>';
                })
                .catch(error => window.alert('Unable to close GeCo: ' + error.message));
            return '';
        }""",
        Output("geco-close-status", "children"),
        Input("geco-close-button", "n_clicks"),
        prevent_initial_call=True,
    )

    from geometric_coder.ui.analyses import analysis_tab

    app.layout = html.Div(
        [
            dcc.Loading(
                id="global-working-indicator",
                delay_show=450,
                delay_hide=150,
                # Do not cover the interface for every lightweight Dash callback.
                # These targets correspond to operations that can legitimately take
                # long enough to need explicit feedback.
                target_components={
                    "semantic-result-candidate": "data",
                    "view-refresh-store": "data",
                    "focus-refresh-store": "data",
                    "explore-code-status": "children",
                    "focus-annotation-status": "children",
                    "focus-model-refresh-store": "data",
                    "focus-auto-training-status": "children",
                    "classifier-spec-refresh-store": "data",
                    "prediction-geometry-refresh-store": "data",
                    "teaching-example-refresh-store": "data",
                    "apply-refresh-store": "data",
                },
                # Dash's fullscreen prop applies only to built-in spinners.  The
                # custom GeCo indicator therefore owns its viewport overlay.
                parent_className="geco-working-loading-root",
                parent_style={"position": "static", "zIndex": "auto"},
                overlay_style={"visibility": "visible"},
                custom_spinner=html.Div(
                    html.Div(
                        [
                            html.Div(
                                html.Div(
                                    "🦎",
                                    className="geco-working-gecko",
                                    **{"aria-hidden": "true"},
                                ),
                                className="geco-working-track",
                                **{"aria-hidden": "true"},
                            ),
                            html.Strong("GeCo is working"),
                            html.Span(
                                "Please sit tight while this operation finishes."
                            ),
                        ],
                        className="geco-working-card",
                    ),
                    className="geco-working-overlay",
                    role="status",
                    **{
                        "aria-live": "polite",
                        "aria-busy": "true",
                        "aria-label": "GeCo is working",
                    },
                ),
                children=[
                html.Div(id="geco-close-status", className="sr-only"),
                dcc.Store(id="geco-tooltip-prefs", storage_type="local", data={"enabled": True}),
                dcc.Store(id="focal-unit-store", data=initial_focal_unit_id),
                dcc.Store(
                    id="explore-focal-unit-store",
                    data=initial_focal_unit_id if initial_workspace == "explore" else None,
                ),
                dcc.Store(
                    id="focus-focal-unit-store",
                    data=initial_focal_unit_id if initial_workspace == "focus" else None,
                ),
                dcc.Store(id="geometry-map-base-revision-store", data=0),
                dcc.Store(id="topic-model-refresh-store", data=0),
                dcc.Store(id="representation-refresh-store", data=0),
                dcc.Store(id="topic-color-mode", data="plain"),
                dcc.Store(id="topic-restore-snapshot", data=_topic_restore_snapshot(int(initial_session["session_id"]), initial_state)),
                dcc.Store(id="session-create-modal-return-focus", data=None),
                dcc.Store(id="semantic-scores-store", data={}),
                dcc.Store(id="semantic-result-candidate", data=None),
                dcc.Store(id="semantic-active-query", data=initial_semantic_query),
                dcc.Store(id="semantic-load-refresh", data=0),
                dcc.Store(id="semantic-range", data=None),
                dcc.Store(
                    id="metadata-filters-store",
                    data=_normalize_metadata_filters(
                        initial_state.get("metadata_filters", []), metadata_profiles
                    ),
                ),
                dcc.Store(id="saved-filter-refresh-store", data=0),
                dcc.Store(id="active-session-store", data=int(initial_session["session_id"])),
                dcc.Store(id="navigation-history-store", data=initial_history),
                dcc.Store(id="desired-view-store", data=initial_view_id),
                dcc.Store(id="view-refresh-store", data=0),
                dcc.Store(
                    id="desired-semantic-range-store",
                    data=None,
                ),
                dcc.Store(id="focus-recommendation-store", data={}),
                dcc.Store(id="focus-refresh-store", data=0),
                dcc.Store(id="focus-model-refresh-store", data=0),
                # A stored code/evidence event separates quick annotation writes
                # from optional model fitting. It is never fired on navigation.
                dcc.Store(id="focus-auto-training-request-store", data=None),
                dcc.Store(id="classifier-spec-refresh-store", data=0),
                dcc.Store(id="committee-refresh-store", data=0),
                dcc.Store(id="prediction-geometry-refresh-store", data=None),
                dcc.Store(id="prediction-geometry-page-token", data=None),
                dcc.Store(id="apply-refresh-store", data=0),
                dcc.Store(id="apply-page-index", data=0),
                dcc.Store(id="apply-page-snapshot", data=None),
                dcc.Store(id="memo-refresh-store", data=0),
                dcc.Store(id="active-memo-store", data=None),
                dcc.Store(id="explore-code-refresh-store", data=0),
                dcc.Store(
                    id="explore-code-palette-store",
                    data=initial_explore_code_palette,
                ),
                dcc.Store(
                    id="explore-code-row-count-store",
                    data=len(initial_explore_code_palette),
                ),
                dcc.Store(id="explore-annotation-refresh-store", data=0),
                dcc.Store(id="explore-map-filter-refresh-store", data=0),
                dcc.Store(
                    id="explore-span-selection-store",
                    data=(
                        [int(initial_state["focal_unit_id"])]
                        if initial_state.get("focal_unit_id") is not None
                        else []
                    ),
                ),
                dcc.Store(
                    id="focus-span-selection-store",
                    data=(
                        [int(initial_state["focal_unit_id"])]
                        if initial_state.get("focal_unit_id") is not None
                        else []
                    ),
                ),
                dcc.Store(id="teaching-example-refresh-store", data=0),
                dcc.Store(id="teaching-example-code-store", data=None),
                dcc.Store(id="code-modal-target-store", data=None),
                dcc.Store(id="memo-center-selected-store", data=None),
                dcc.Store(id="memo-center-version-store", data=None),
                dcc.Store(id="memo-center-edit-store", data=False),
                dcc.Store(id="construct-refresh-store", data=0),
                dcc.Store(id="construct-center-selected-store", data=None),
                dcc.Store(id="code-center-selected-store", data=None),
                dcc.Store(id="code-center-version-store", data=None),
                dcc.Store(id="code-center-edit-store", data=False),
                dcc.Store(id="code-center-focal-store", data=None),
                    html.Header(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "🦎",
                                        className="app-gecko",
                                        role="img",
                                        **{"aria-label": "gecko"},
                                    ),
                                    html.Div(
                                        [
                                            html.H1("GeCo", className="app-title"),
                                            html.P(
                                                "Geometric Coder: Scalable, computationally augmented exploration and coding",
                                                className="app-subtitle",
                                            ),
                                        ]
                                    ),
                                ],
                                className="app-brand",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Button(
                                                "Settings", id="geco-settings-open",
                                                n_clicks=0, className="geco-settings-button",
                                            ),
                                            html.Button(
                                                "Close GeCo", id="geco-close-button",
                                                n_clicks=0, className="geco-close-button",
                                                title="Stop this GeCo application",
                                            ),
                                        ],
                                        className="app-header-actions",
                                    ),
                                    html.Div(
                                        _project_summary_children(
                                            html=html,
                                            modality=str(metadata["modality"]),
                                            key_columns=[str(value) for value in metadata["key_columns"]],
                                            hierarchy_summary=hierarchy_summary,
                                            atomic_count=len(units),
                                        ),
                                        className="project-summary",
                                    ),
                                ],
                                className="app-header-right",
                            ),
                        ],
                        id="app-header",
                        className="app-header",
                    ),
                    dcc.Tabs(
                id="workspace-tabs",
                value="explore",
                children=[
                    dcc.Tab(label="Explore", value="explore", className="workflow-tab", selected_className="workflow-tab-selected", children=_explore_layout(
                        dcc=dcc,
                        html=html,
                        geometries=geometries,
                        views=views_by_geometry.get(initial_geometry_id, []),
                        hierarchy=hierarchy,
                        initial_state=initial_state,
                        initial_geometry_id=initial_geometry_id,
                        initial_view_id=initial_view_id,
                        initial_semantic_scores=initial_semantic_scores,
                        initial_semantic_status=initial_semantic_status,
                        geometry_by_id=geometry_by_id,
                        total_units=len(units),
                        sessions=sessions,
                        initial_session_id=int(initial_session["session_id"]),
                        initial_seen_count=initial_seen_count,
                        metadata_profiles=metadata_profiles,
                        allow_create_view=not project.is_external_backed,
                    )),
                    dcc.Tab(
                        label="Develop",
                        value="focus",
                        className="workflow-tab",
                        selected_className="workflow-tab-selected",
                        children=_focus_layout(
                            dcc=dcc,
                            html=html,
                            codes=codes,
                            constructs=constructs,
                            geometries=geometries,
                            classifier_specs=classifier_specs,
                            hierarchy=hierarchy,
                            initial_state=initial_state,
                        ),
                    ),
                    dcc.Tab(
                        label="Apply and Review",
                        value="apply",
                        className="workflow-tab",
                        selected_className="workflow-tab-selected",
                        children=_apply_layout(
                            dcc=dcc,
                            html=html,
                            dag=dag,
                            codes=codes,
                            classifier_specs=classifier_specs,
                        ),
                    ),
                    dcc.Tab(
                        label="Constructs",
                        value="codes",
                        className="workflow-tab",
                        selected_className="workflow-tab-selected",
                        children=_code_center_layout(
                            dcc=dcc, html=html, dash_table=dash_table,
                            geometries=geometries, views=views_by_geometry.get(initial_geometry_id, []),
                            initial_geometry_id=initial_geometry_id, initial_view_id=initial_view_id,
                            total_units=len(units),
                        ),
                    ),
                    dcc.Tab(
                        label="Analyses",
                        value="analyses",
                        className="workflow-tab",
                        selected_className="workflow-tab-selected",
                        children=analysis_tab(dcc=dcc, html=html, project=project),
                    ),
                    dcc.Tab(
                        label="Memos",
                        value="memos",
                        className="workflow-tab",
                        selected_className="workflow-tab-selected",
                        children=_memo_center_layout(
                            dcc=dcc,
                            html=html,
                            dash_table=dash_table,
                        ),
                    ),
                ],
                    ),
                    dcc.Location(id="memo-navigation-location", refresh=False),
                    dcc.Location(id="memo-presentation-location", refresh=False),
                    _memo_presentation_layout(
                        dcc=dcc,
                        html=html,
                        views=presentation_views,
                        geometry_by_id=geometry_by_id,
                        initial_view_id=initial_view_id,
                        total_units=len(units),
                    ),
                    _code_modal(dcc=dcc, html=html),
                    _teaching_example_modal(
                        dcc=dcc, html=html, dash_table=dash_table
                    ),
                    _classifier_spec_modal(
                        dcc=dcc, html=html, geometries=geometries
                    ),
                    _classifier_manager_modal(
                        dcc=dcc, html=html, dash_table=dash_table
                    ),
                    _new_committee_modal(dcc=dcc, html=html),
                    _committee_modal(
                        dcc=dcc, html=html, dash_table=dash_table
                    ),
                    _testing_center_modal(
                        dcc=dcc, html=html, dash_table=dash_table
                    ),
                    _metadata_filter_modal(
                        dcc=dcc, html=html, metadata_profiles=metadata_profiles
                    ),
                    _save_filter_set_modal(dcc=dcc, html=html),
                    _load_filter_set_modal(dcc=dcc, html=html),
                    dcc.ConfirmDialog(id="saved-filter-delete-confirm", message="Delete this saved filter collection?"),
                    dcc.Store(id="saved-filter-delete-pending", data=None),
                    html.Div([
                        html.Div([
                            html.Div([html.H2("Saved semantic searches"),
                                      html.Button("×", id="semantic-load-close", n_clicks=0, className="modal-close-icon")],
                                     className="modal-header"),
                            html.Div(id="semantic-load-list"),
                        ], className="modal-card"),
                    ], id="semantic-load-modal", className="modal-overlay", style={"display": "none"}),
                    dcc.ConfirmDialog(id="semantic-delete-confirm", message="Delete this saved semantic query and all cached geometry results?"),
                    dcc.Store(id="semantic-delete-pending", data=None),
                    representation_modal(project, html=html, dcc=dcc, dash_table=dash_table),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H2("Manage Topic Models"),
                                html.Button("×", id="topic-manager-close", n_clicks=0, className="modal-close-icon", title="Close", **{"aria-label": "Close"}),
                            ], className="modal-header"),
                            html.P("Rename models and topics without changing their fitted results.", className="modal-help"),
                            html.Label("Registered model"),
                            dcc.Dropdown(id="topic-manager-model", clearable=False),
                            html.Label("Model name"),
                            dcc.Input(id="topic-model-new-name", type="text", className="text-input"),
                            html.Button("Rename model", id="topic-model-rename-save", n_clicks=0),
                            html.H3("Edit topic labels"),
                            html.P("Original topic IDs stay the same when labels change."),
                            dcc.Dropdown(id="topic-manager-component", clearable=False),
                            html.Label("Selected topic label"),
                            dcc.Input(id="topic-component-new-label", type="text", className="text-input"),
                            html.Button("Rename topic", id="topic-label-save", n_clicks=0),
                            html.Div(id="topic-action-status", role="status"),
                            html.Button("Create Topic Model…", id="topic-create-open", n_clicks=0, className="secondary-action"),
                        ], className="modal-card topic-manager-card"),
                    ], id="topic-manager-modal", className="modal-overlay", style={"display": "none"}),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H2("Create Topic Model"),
                                html.Button("×", id="topic-create-close", n_clicks=0, className="modal-close-icon", title="Close", **{"aria-label": "Close"}),
                            ], className="modal-header"),
                            html.Label("Model name"),
                            dcc.Input(id="topic-create-name", type="text", className="text-input", placeholder="Name this model"),
                            html.Label("Geometry"),
                            dcc.Dropdown(id="topic-create-geometry", options=topic_geometry_options(project), clearable=False),
                            html.Label("Algorithm"),
                            dcc.Dropdown(id="topic-create-algorithm", options=recipe_options(), value="identity", clearable=False),
                            html.Div(id="topic-create-parameters", className="topic-create-parameters"),
                            html.Details([
                                html.Summary("Advanced parameters"),
                                html.Div(id="topic-create-advanced", className="topic-create-parameters"),
                            ], id="topic-create-advanced-wrap", className="topic-create-advanced", style={"display": "none"}),
                            html.Div([
                                html.Button("Cancel", id="topic-create-cancel", n_clicks=0),
                                html.Button("Create", id="topic-create-submit", n_clicks=0, disabled=teal_linked_project(project) is None or not bool(topic_geometry_options(project)), className="primary-action"),
                            ], className="modal-actions"),
                            html.Div(id="topic-create-status", role="status", className="control-status"),
                        ], className="modal-card topic-manager-card"),
                    ], id="topic-create-modal", className="modal-overlay", style={"display": "none"}),

                    html.Div([
                        html.Div([
                            html.Div([
                                html.H2("GeCo Settings"),
                                html.Button("×", id="geco-settings-close", n_clicks=0,
                                            className="modal-close-icon", title="Close",
                                            **{"aria-label": "Close Settings"}),
                            ], className="modal-header"),
                            html.H3("Interface"),
                            dcc.Checklist(
                                id="geco-settings-tooltips",
                                options=[{"label": "Show explanatory tooltips", "value": "enabled"}],
                                value=["enabled"],
                            ),
                            html.H3("Classifier training"),
                            html.Label("Cross-validation folds", htmlFor="geco-settings-cv-folds"),
                            dcc.Dropdown(
                                id="geco-settings-cv-folds",
                                options=[{"label": str(n), "value": n} for n in range(2, 11)],
                                value=_read_ui_settings(project.project_dir)["cv_folds"],
                                clearable=False, searchable=False,
                                className="geco-settings-folds",
                            ),
                            html.Small("Maximum folds requested when tuning; fewer are used when labeled evidence is limited."),
                            html.Div(id="geco-settings-status", className="control-status", role="status"),
                        ], className="modal-card geco-settings-card", role="dialog",
                           **{"aria-modal": "true", "aria-label": "GeCo Settings"}),
                    ], id="geco-settings-modal", className="modal-overlay",
                       style={"display": "none"}),
                    html.Div([
                        html.Div([
                            html.Div([html.H2("New session"), html.Button("×", id="new-session-x", n_clicks=0, className="modal-close-icon", title="Close", **{"aria-label": "Close"})], className="modal-header"),
                            html.Label("Session name", htmlFor="new-session-title"),
                            dcc.Input(id="new-session-title", type="text", placeholder="Session name", autoFocus=True),
                            html.Div([
                                html.Button("Cancel", id="new-session-cancel", n_clicks=0),
                                html.Button("Save", id="new-session-save", n_clicks=0, className="primary-action"),
                            ], className="modal-actions"),
                        ], className="modal-card", role="dialog", **{"aria-modal": "true"}),
                    ], id="new-session-modal", className="modal-overlay", style={"display": "none"}),
                ],
            ),
            html.Div(id="session-state-status", className="sr-only"),
        ],
        id="geco-app-shell",
        className="app-shell",
        style=GECO_DASH_THEME,
    )

    @app.callback(
        Output("geco-settings-modal", "style"),
        Input("geco-settings-open", "n_clicks"),
        Input("geco-settings-close", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_geco_settings(_open: int, _close: int) -> dict[str, str]:
        return {"display": "flex"} if callback_context.triggered_id == "geco-settings-open" else {"display": "none"}

    @app.callback(
        Output("geco-tooltip-prefs", "data"),
        Input("geco-settings-tooltips", "value"),
        State("geco-tooltip-prefs", "data"),
        prevent_initial_call=True,
    )
    def save_tooltip_preference(value: list[str] | None, current: dict[str, Any] | None) -> Any:
        enabled = "enabled" in (value or [])
        if isinstance(current, dict) and current.get("enabled") is enabled:
            return no_update
        return {"enabled": enabled}

    @app.callback(
        Output("geco-settings-tooltips", "value"),
        Output("geco-app-shell", "className"),
        Input("geco-tooltip-prefs", "data"),
    )
    def apply_tooltip_preference(prefs: dict[str, Any] | None) -> tuple[list[str], str]:
        enabled = not isinstance(prefs, dict) or prefs.get("enabled", True) is not False
        return (["enabled"] if enabled else []), (
            "app-shell" if enabled else "app-shell geco-tooltip-disabled"
        )

    @app.callback(
        Output("geco-settings-status", "children"),
        Input("geco-settings-cv-folds", "value"),
        prevent_initial_call=True,
    )
    def save_cv_fold_setting(folds: int | None) -> str:
        if folds is None:
            return ""
        try:
            _save_ui_settings(project.project_dir, cv_folds=int(folds))
        except (OSError, ValueError) as error:
            return f"Unable to save CV folds: {error}"
        return f"Using up to {int(folds)} cross-validation folds."

    @app.callback(
        Output("topic-interpretation-details", "style"),
        Input("topic-model-selector", "value"),
    )
    def show_topic_interpretation(model_id: int | None) -> dict[str, str]:
        return {} if model_id is not None else {"display": "none"}

    register_representation_callbacks(
        app, project, html=html, dcc=dcc, Input=Input, Output=Output,
        State=State, ALL=ALL, callback_context=callback_context,
        no_update=no_update, geometries=geometries, geometry_by_id=geometry_by_id,
    )

    @app.callback(
        Output("topic-model-selector", "options"),
        Input("session-dropdown", "value"),
        Input("topic-model-refresh-store", "data"),
    )
    def available_topic_models(_session_id: int, _revision: int) -> list[dict[str, Any]]:
        return [
            {"label": str(model["name"]), "value": int(model["topic_model_id"])}
            for model in project.topic_models()
        ]

    @app.callback(
        Output("topic-color-mode", "data", allow_duplicate=True),
        Output("topic-model-selector", "value", allow_duplicate=True),
        Output("semantic-query", "value", allow_duplicate=True),
        Output("semantic-scores-store", "data", allow_duplicate=True),
        Output("semantic-range", "data", allow_duplicate=True),
        Output("semantic-active-query", "data", allow_duplicate=True),
        Output("desired-semantic-range-store", "data", allow_duplicate=True),
        Output("semantic-search-status", "children", allow_duplicate=True),
        Input("nav-clear-colors", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_all_colors(_clicks: int) -> tuple[str, None, str, dict[str, Any], None, str, None, str]:
        # Clear the query *intent* as well as visible scores: geometry changes
        # must not reactivate a search the researcher explicitly dismissed.
        return "plain", None, "", {}, None, "", None, ""

    @app.callback(
        Output("nav-clear-colors", "style"),
        Input("topic-color-mode", "data"),
    )
    def show_clear_colors(mode: str | None) -> dict[str, str]:
        return {"display": "inline-block"} if normalize_color_mode(mode) != "plain" else {"display": "none"}

    @app.callback(
        Output("topic-color-mode", "data", allow_duplicate=True),
        Input("topic-model-selector", "value"),
        Input("topic-display-mode", "value"),
        Input("topic-component-selector", "value"),
        Input("topic-restore-snapshot", "data"),
        State("active-session-store", "data"),
        State("topic-color-mode", "data"),
        prevent_initial_call=True,
    )
    def activate_topic_colors(
        model_id: int | None, display: str | None, component: str | None,
        restored: dict[str, Any] | None, session_id: int, current_mode: str | None,
    ) -> Any:
        # Session restore also changes topic selectors; those are not user actions.
        if _is_restored_topic_selection(model_id, display, component, restored, session_id):
            return no_update
        triggered = callback_context.triggered_id
        if model_id is None:
            # Manual dropdown clear removes topic coloring, but a programmatic clear
            # after choosing semantic colors must not erase the semantic selection.
            return (
                "plain"
                if triggered == "topic-model-selector"
                and current_mode in {"dominant", "intensity"}
                else no_update
            )
        if triggered == "topic-restore-snapshot":
            return no_update
        if triggered == "topic-component-selector" and display != "intensity":
            return no_update
        if triggered == "topic-component-selector" and component is None:
            return no_update
        return "intensity" if display == "intensity" else "dominant"

    @app.callback(
        Output("semantic-active-query", "data", allow_duplicate=True),
        Output("semantic-query", "value", allow_duplicate=True),
        Output("semantic-scores-store", "data", allow_duplicate=True),
        Input("topic-color-mode", "data"),
        State("topic-model-selector", "value"),
        State("topic-display-mode", "value"),
        State("topic-component-selector", "value"),
        State("topic-restore-snapshot", "data"),
        State("active-session-store", "data"),
        prevent_initial_call=True,
    )
    def clear_semantic_on_topic_color(mode, model, display, component, restored, session_id):
        if mode not in {"dominant", "intensity"}:
            return no_update, no_update, no_update
        if _is_restored_topic_selection(model, display, component, restored, session_id):
            return no_update, no_update, no_update
        return "", "", {}

    @app.callback(
        Output("topic-component-wrap", "style"),
        Input("topic-display-mode", "value"),
    )
    def show_selected_topic(display: str | None) -> dict[str, str]:
        return {"display": "block"} if display == "intensity" else {"display": "none"}

    @app.callback(
        Output("topic-component-selector", "options"),
        Input("topic-model-selector", "value"),
        Input("topic-model-refresh-store", "data"),
    )
    def available_topic_components(model_id: int | None, _revision: int) -> list[dict[str, str]]:
        if model_id is None or not units:
            return []
        try:
            rows = project.topic_model_rows(int(model_id), positions=[0])
        except (ConfigurationError, RuntimeError, ValueError, KeyError):
            return []
        return [
            {"label": rows.topic_labels.get(topic, topic) + (f" ({topic})" if topic in rows.topic_labels else ""), "value": topic}
            for topic in rows.topic_ids
        ]

    @app.callback(
        Output("topic-manager-modal", "style"),
        Input("topic-manager-open", "n_clicks"),
        Input("topic-manager-close", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_topic_manager(_open: int, _close: int) -> dict[str, str]:
        return {"display": "flex"} if callback_context.triggered_id == "topic-manager-open" else {"display": "none"}

    @app.callback(
        Output("topic-manager-model", "options"),
        Input("topic-model-refresh-store", "data"),
    )
    def manager_models(_revision: int) -> list[dict[str, Any]]:
        return [{"label": str(m["name"]), "value": int(m["topic_model_id"])} for m in project.topic_models()]

    @app.callback(
        Output("topic-manager-model", "value"),
        Input("topic-manager-open", "n_clicks"),
        State("topic-model-selector", "value"),
        prevent_initial_call=True,
    )
    def choose_manager_model(_n_clicks: int, selected: int | None) -> int | None:
        models = project.topic_models()
        ids = {int(m["topic_model_id"]) for m in models}
        return selected if selected in ids else (min(ids) if ids else None)

    @app.callback(
        Output("topic-manager-component", "value"),
        Input("topic-manager-component", "options"),
        State("topic-manager-component", "value"),
    )
    def choose_manager_topic(
        options: list[dict[str, str]] | None, selected: str | None,
    ) -> str | None:
        valid = [str(o["value"]) for o in (options or [])]
        return selected if selected in valid else (valid[0] if valid else None)

    @app.callback(
        Output("topic-manager-component", "options"),
        Input("topic-manager-model", "value"),
        Input("topic-model-refresh-store", "data"),
    )
    def manager_components(model_id: int | None, _revision: int) -> list[dict[str, str]]:
        if model_id is None or not units:
            return []
        try:
            rows = project.topic_model_rows(int(model_id), positions=[0])
        except (ConfigurationError, RuntimeError, KeyError, ValueError):
            return []
        return [{"label": rows.topic_labels.get(t, t) + (f" ({t})" if t in rows.topic_labels else ""), "value": t} for t in rows.topic_ids]

    @app.callback(
        Output("topic-create-modal", "style"),
        Input("topic-create-open", "n_clicks"),
        Input("topic-create-close", "n_clicks"),
        Input("topic-create-cancel", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_topic_creation(_open: int, _close: int, _cancel: int) -> dict[str, str]:
        return {"display": "flex"} if callback_context.triggered_id == "topic-create-open" else {"display": "none"}

    @app.callback(
        Output("topic-create-parameters", "children"),
        Output("topic-create-advanced", "children"),
        Output("topic-create-advanced-wrap", "style"),
        Input("topic-create-algorithm", "value"),
    )
    def show_recipe_parameters(algorithm: str | None) -> tuple[list[Any], list[Any], dict[str, str]]:
        try:
            common, advanced = split_parameter_specs(algorithm or "identity")
        except ImportError:
            # TeAL is an optional dependency of standalone GeCo.
            common, advanced = standalone_primary_specs(algorithm or "identity"), []

        def controls_for(specs: list[dict[str, Any]]) -> list[Any]:
            controls = []
            for spec in specs:
                param_id = {"type": "topic-hyperparameter", "name": spec["name"]}
                if spec["kind"] == "boolean":
                    field = dcc.Dropdown(
                        id=param_id,
                        options=[{"label": "True", "value": "true"}, {"label": "False", "value": "false"}],
                        value="true" if spec["default"] else "false", clearable=False,
                    )
                elif spec["kind"] == "choice":
                    field = dcc.Dropdown(
                        id=param_id, options=spec["options"],
                        value=spec["default"], clearable=False,
                    )
                else:
                    field = dcc.Input(
                        id=param_id, value=spec["default"],
                        type="number" if spec["kind"] in {"integer", "number"} else "text",
                        step=1 if spec["kind"] == "integer" else "any",
                    )
                controls.append(html.Div([html.Label(spec["label"]), field], className="topic-parameter-field"))
            return controls

        return controls_for(common), controls_for(advanced), ({"display": "block"} if advanced else {"display": "none"})

    @app.callback(
        Output("topic-model-refresh-store", "data", allow_duplicate=True),
        Output("topic-create-status", "children"),
        Input("topic-create-submit", "n_clicks"),
        State("topic-create-name", "value"),
        State("topic-create-geometry", "value"),
        State("topic-create-algorithm", "value"),
        State({"type": "topic-hyperparameter", "name": ALL}, "value"),
        State({"type": "topic-hyperparameter", "name": ALL}, "id"),
        State("topic-model-refresh-store", "data"),
        prevent_initial_call=True,
        running=[(Output("topic-create-submit", "disabled"), True, False)],
    )
    def create_topic_from_teal(
        clicks: int, name: str | None, geometry: int | None,
        algorithm: str | None, values: list[Any], ids: list[dict[str, str]],
        revision: int | None,
    ) -> tuple[Any, str]:
        del clicks
        if geometry is None:
            return no_update, "Choose a registered geometry."
        params = {item["name"]: value for item, value in zip(ids, values, strict=True)}
        try:
            result = create_topic_model(
                project, name=name or "", geometry_id=int(geometry),
                recipe=algorithm or "identity", parameters=params,
            )
        except Exception as exc:
            return no_update, f"Topic Model creation failed: {exc}"
        return int(revision or 0) + 1, (
            f"Created Topic Model {(name or '').strip()!r} (ID {result.topic_model_id})."
        )

    register_topic_details_callbacks(
        app, project, html=html, Input=Input, Output=Output,
        State=State, callback_context=callback_context,
        no_update=no_update, units=units,
    )

    # Keep tab-specific focal mirrors on the client. Hidden workspaces should not
    # wake their server-side reading/model callbacks every time the global focal
    # unit changes in another workspace. Switching tabs copies the current focal
    # unit into that workspace exactly once.
    app.clientside_callback(
        """
        function(workspace, focalUnitId) {
            const noUpdate = window.dash_clientside.no_update;
            return [
                workspace === "explore" ? focalUnitId : noUpdate,
                workspace === "focus" ? focalUnitId : noUpdate
            ];
        }
        """,
        Output("explore-focal-unit-store", "data"),
        Output("focus-focal-unit-store", "data"),
        Input("workspace-tabs", "value"),
        Input("focal-unit-store", "data"),
    )

    app.clientside_callback(
        """
        function(annotationRefresh, uncodedValues, currentRefresh) {
            const enabled = Array.isArray(uncodedValues) && uncodedValues.includes("uncoded_only");
            if (!enabled) {
                return window.dash_clientside.no_update;
            }
            return (Number(currentRefresh) || 0) + 1;
        }
        """,
        Output("explore-map-filter-refresh-store", "data"),
        Input("explore-annotation-refresh-store", "data"),
        State("explore-uncoded-only", "value"),
        State("explore-map-filter-refresh-store", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("session-dropdown", "options"),
        Output("session-dropdown", "value"),
        Output("new-session-title", "value"),
        Output("new-session-modal", "style"),
        Input("new-session-save", "n_clicks"),
        State("new-session-title", "value"),
        prevent_initial_call=True,
    )
    def create_session(n_clicks: int, title: str | None) -> tuple[list[dict[str, Any]], int, str, dict[str, str]]:
        del n_clicks
        clean_title = (title or "").strip() or f"Session {len(project.sessions()) + 1}"
        session_id = project.create_session(clean_title)
        return _session_options(project.sessions()), session_id, "", {"display": "none"}

    @app.callback(
        Output("new-session-modal", "style", allow_duplicate=True),
        Input("new-session-button", "n_clicks"),
        Input("new-session-cancel", "n_clicks"),
        Input("new-session-x", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_new_session_modal(_open: int, _cancel: int, _x: int) -> dict[str, str]:
        return {"display": "flex"} if callback_context.triggered_id == "new-session-button" else {"display": "none"}

    @app.callback(
        Output("active-session-store", "data"),
        Output("geometry-dropdown", "value"),
        Output("desired-view-store", "data"),
        Output("focal-unit-store", "data", allow_duplicate=True),
        Output("navigation-history-store", "data"),
        Output("literal-query", "value"),
        Output("literal-options", "value"),
        Output("semantic-query", "value"),
        Output("semantic-active-query", "data", allow_duplicate=True),
        Output("semantic-scores-store", "data", allow_duplicate=True),
        Output("desired-semantic-range-store", "data"),
        Output("semantic-range", "data", allow_duplicate=True),
        Output("metadata-filters-store", "data"),
        Output("metadata-filter-operator", "value"),
        Output("explore-uncoded-only", "value"),
        Output("context-level", "value"),
        Output("context-window", "value"),
        Output("page-size", "value"),
        Output("page-number", "value"),
        Output("navigation-new-only", "value"),
        Output("workspace-tabs", "value"),
        Output("explore-code-palette-store", "data"),
        Output("explore-code-row-count-store", "data"),
        Output("focus-construct-dropdown", "value", allow_duplicate=True),
        Output("focus-code-dropdown", "value", allow_duplicate=True),
        Output("focus-active-classifier-dropdown", "value", allow_duplicate=True),
        Output("focus-committee-dropdown", "value", allow_duplicate=True),
        Output("focus-train-scope", "value"),
        Output("focus-auto-retrain", "value"),
        Output("focus-tune-hyperparameters", "value"),
        Output("focus-context-level", "value"),
        Output("focus-context-window", "value"),
        Output("topic-color-mode", "data"),
        Output("topic-model-selector", "value"),
        Output("topic-component-selector", "value"),
        Output("topic-display-mode", "value"),
        Output("semantic-search-status", "children", allow_duplicate=True),
        Output("topic-restore-snapshot", "data"),
        Input("session-dropdown", "value"),
        prevent_initial_call=True,
    )
    def restore_session(session_id: int) -> tuple[Any, ...]:
        session = project.database.get_session(int(session_id))
        state = _resumable_session_state(dict(session["state"]))
        geometry_id = _valid_geometry_id(state.get("geometry_id"), geometries)
        view_id = _valid_view_id(
            state.get("view_id"),
            project.views(geometry_id, include_descendants=True),
        )
        # Each session starts without active semantic or topic coloring.
        query, scores, semantic_status = "", {}, ""
        literal_options = []
        if state.get("case_sensitive"):
            literal_options.append("case")
        if state.get("regex"):
            literal_options.append("regex")
        context_level = int(state.get("context_level", hierarchy[-1]["level_index"]))
        context_window = max(1, int(state.get("context_window", 1)))
        current_code_ids = {int(code["code_id"]) for code in project.codes()}
        focus_construct_id = _selected_focus_construct_id(project.constructs(), state)
        focus_code_id = (
            focus_construct_id if focus_construct_id in current_code_ids else None
        )
        spec_ids = {
            int(record["classifier_spec_id"])
            for record in project.classifier_specs(code_id=focus_code_id)
        } if focus_code_id is not None else set()
        requested_classifier = state.get("focus_classifier_spec_id")
        focus_classifier = (
            int(requested_classifier)
            if requested_classifier is not None and int(requested_classifier) in spec_ids
            else (min(spec_ids) if spec_ids else None)
        )
        committee_ids = {
            int(record["committee_id"])
            for record in project.classifier_committees(focus_code_id)
        } if focus_code_id is not None else set()
        requested_committee = state.get("focus_committee_id")
        focus_committee = (
            int(requested_committee)
            if requested_committee is not None
            and int(requested_committee) in committee_ids
            else (min(committee_ids) if committee_ids else None)
        )
        history = _normalize_navigation_history(
            state.get("navigation_history"),
            focal_unit_id=state.get("focal_unit_id"),
        )
        explore_palette = _normalize_explore_code_palette(
            state.get("explore_code_palette"), current_code_ids
        )
        return (
            int(session_id),
            geometry_id,
            view_id,
            state.get("focal_unit_id"),
            history,
            str(state.get("literal_query", "")),
            literal_options,
            query,
            query,
            scores,
            None,
            None,
            _normalize_metadata_filters(
                state.get("metadata_filters", []), metadata_profiles
            ),
            _normalize_boolean_operator(
                state.get("metadata_filter_operator", "and")
            ),
            (["uncoded_only"] if state.get("explore_uncoded_only", False) else []),
            context_level,
            context_window,
            str(max(1, int(state.get("page_size", 5000)))),
            max(1, int(state.get("page_number", 1))),
            (["new_only"] if state.get("navigation_new_only", True) else []),
            str(state.get("workspace", "explore")),
            explore_palette,
            len(explore_palette),
            focus_construct_id,
            focus_code_id,
            focus_classifier,
            focus_committee,
            (["all"] if state.get("focus_train_all", False) else []),
            (["auto"] if state.get("focus_auto_retrain", True) else []),
            (["tune"] if state.get("focus_tune_hyperparameters", True) else []),
            int(state.get("focus_context_level", hierarchy[-1]["level_index"])),
            max(1, int(state.get("focus_context_window", 1))),
            "plain",
            None,
            None,
            "dominant",
            semantic_status,
            _topic_restore_snapshot(int(session_id), state),
        )

    @app.callback(
        Output("view-dropdown", "options"),
        Output("view-dropdown", "value"),
        Input("geometry-dropdown", "value"),
        Input("view-refresh-store", "data"),
        State("view-dropdown", "value"),
        State("desired-view-store", "data"),
    )
    def update_view_options(
        geometry_id: int | None,
        refresh: int,
        current_view_id: int | None,
        desired_view_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        del refresh
        candidates = (
            project.views(int(geometry_id), include_descendants=True)
            if geometry_id is not None else []
        )
        options = [{"label": view["name"], "value": int(view["view_id"])} for view in candidates]
        valid_ids = {option["value"] for option in options}
        if desired_view_id in valid_ids:
            value = desired_view_id
        elif current_view_id in valid_ids:
            value = current_view_id
        else:
            value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("one-dimensional-jitter-wrap", "style"),
        Input("view-dropdown", "value"),
    )
    def show_one_dimensional_jitter(view_id: int | None) -> dict[str, str]:
        if view_id is None:
            return {"display": "none"}
        if int(view_id) < 0:
            # Unified plot-ready representations declare their dimension in
            # the registry; selecting one need not load its full matrix.
            dimensions = int(project.database.get_representation(-int(view_id))["dimensions"])
        else:
            # Inspect a bounded local row, not an entire 2D view on selection.
            dimensions = int(project.view_coordinate_rows(
                int(view_id), [0] if len(units) else [],
            ).shape[1])
        return {"display": "block"} if dimensions == 1 else {"display": "none"}

    @app.callback(
        Output("projection-modal", "style"),
        Output("view-refresh-store", "data"),
        Output("desired-view-store", "data", allow_duplicate=True),
        Output("projection-create-status", "children"),
        Input("projection-open-button", "n_clicks"),
        Input("projection-cancel-button", "n_clicks"),
        Input("projection-cancel-footer", "n_clicks"),
        Input("projection-create-button", "n_clicks"),
        State("geometry-dropdown", "value"),
        State("projection-name", "value"),
        State("projection-method", "value"),
        State("projection-umap-neighbors", "value"),
        State("projection-umap-min-dist", "value"),
        State("projection-umap-metric", "value"),
        State("projection-umap-spread", "value"),
        State("projection-umap-repulsion", "value"),
        State("projection-umap-random-state", "value"),
        State("projection-pca-solver", "value"),
        State("projection-pca-whiten", "value"),
        State("projection-pca-random-state", "value"),
        State("projection-svd-algorithm", "value"),
        State("projection-svd-n-iter", "value"),
        State("projection-svd-random-state", "value"),
        State("view-refresh-store", "data"),
        prevent_initial_call=True,
        running=[
            (Output("projection-create-button", "disabled"), True, False),
            (
                Output("projection-progress-wrap", "style"),
                {"display": "block"},
                {"display": "none"},
            ),
        ],
    )
    def manage_projection_modal(
        open_clicks: int,
        cancel_clicks: int,
        cancel_footer_clicks: int,
        create_clicks: int,
        geometry_id: int | None,
        name: str | None,
        method: str | None,
        umap_neighbors: int | None,
        umap_min_dist: float | None,
        umap_metric: str | None,
        umap_spread: float | None,
        umap_repulsion: float | None,
        umap_random_state: int | None,
        pca_solver: str | None,
        pca_whiten: list[str] | None,
        pca_random_state: int | None,
        svd_algorithm: str | None,
        svd_n_iter: int | None,
        svd_random_state: int | None,
        refresh: int | None,
    ) -> tuple[dict[str, str], int | Any, int | Any, str | Any]:
        del open_clicks, cancel_clicks, cancel_footer_clicks, create_clicks
        triggered = callback_context.triggered_id
        if triggered == "projection-open-button":
            return {"display": "flex"}, no_update, no_update, ""
        if triggered in {"projection-cancel-button", "projection-cancel-footer"}:
            return {"display": "none"}, no_update, no_update, ""
        if geometry_id is None or not method:
            return {"display": "flex"}, int(refresh or 0), no_update, (
                "Choose a geometry and projection method."
            )
        try:
            parameters = _projection_parameters(
                method=str(method),
                umap_neighbors=umap_neighbors,
                umap_min_dist=umap_min_dist,
                umap_metric=umap_metric,
                umap_spread=umap_spread,
                umap_repulsion=umap_repulsion,
                umap_random_state=umap_random_state,
                pca_solver=pca_solver,
                pca_whiten=pca_whiten,
                pca_random_state=pca_random_state,
                svd_algorithm=svd_algorithm,
                svd_n_iter=svd_n_iter,
                svd_random_state=svd_random_state,
            )
            geometry_name = str(geometry_by_id[int(geometry_id)]["name"])
            clean_name = (name or "").strip() or (
                f"{geometry_name}_{method}_{len(project.views(int(geometry_id))) + 1}"
            )
            view_id = project.create_view(
                geometry_id=int(geometry_id),
                name=clean_name,
                method=str(method),
                parameters=parameters,
            )
        except (KeyError, TypeError, ValueError) as error:
            return {"display": "flex"}, int(refresh or 0), no_update, str(error)
        return (
            {"display": "none"},
            int(refresh or 0) + 1,
            view_id,
            f"Created 2D view {clean_name!r}.",
        )

    @app.callback(
        Output("projection-umap-fields", "style"),
        Output("projection-pca-fields", "style"),
        Output("projection-svd-fields", "style"),
        Input("projection-method", "value"),
    )
    def show_projection_fields(method: str | None) -> tuple[dict[str, str], ...]:
        hidden = {"display": "none"}
        shown = {"display": "grid"}
        return (
            shown if method == "umap" else hidden,
            shown if method == "pca" else hidden,
            shown if method in {"truncated_svd", "svd"} else hidden,
        )

    @app.callback(
        Output("semantic-load-modal", "style"),
        Output("semantic-load-list", "children"),
        Input("semantic-load-open", "n_clicks"),
        Input("semantic-load-close", "n_clicks"),
        Input("semantic-load-refresh", "data"),
        prevent_initial_call=True,
    )
    def semantic_load_dialog(_open, _close, _refresh):
        if callback_context.triggered_id == "semantic-load-close":
            return {"display": "none"}, no_update
        rows = project.saved_semantic_searches.queries()
        children = [
            html.Div([
                html.Span(entry["query"]),
                html.Button("Load", id={"type": "semantic-load-query", "index": entry["query_id"]}, n_clicks=0),
                html.Button("Delete", id={"type": "semantic-delete-query", "index": entry["query_id"]}, n_clicks=0,
                            className="danger-action", title="Delete saved query"),
            ], className="saved-filter-set-row") for entry in rows
        ] or [html.P("No saved semantic searches.", className="filter-empty-state")]
        return {"display": "flex"}, children

    @app.callback(
        Output("semantic-active-query", "data", allow_duplicate=True),
        Output("semantic-query", "value", allow_duplicate=True),
        Output("semantic-load-refresh", "data", allow_duplicate=True),
        Output("semantic-load-modal", "style", allow_duplicate=True),
        Output("topic-model-selector", "value", allow_duplicate=True),
        Output("semantic-delete-pending", "data"),
        Output("semantic-delete-confirm", "displayed"),
        Input({"type": "semantic-load-query", "index": ALL}, "n_clicks"),
        Input({"type": "semantic-delete-query", "index": ALL}, "n_clicks"),
        State("semantic-active-query", "data"),
        State("semantic-load-refresh", "data"),
        prevent_initial_call=True,
    )
    def select_saved_query(_loads, _deletes, active, revision):
        trigger = callback_context.triggered_id
        if not _is_semantic_modal_click(trigger, callback_context.triggered):
            return (no_update,) * 7
        qid = int(trigger["index"])
        if trigger["type"] == "semantic-delete-query":
            return no_update, no_update, no_update, no_update, no_update, qid, True
        query = next((row["query"] for row in project.saved_semantic_searches.queries()
                      if row["query_id"] == qid), None)
        if query is None:
            return (no_update,) * 7
        return query, query, no_update, {"display": "none"}, None, no_update, False

    @app.callback(
        Output("semantic-active-query", "data", allow_duplicate=True),
        Output("semantic-load-refresh", "data", allow_duplicate=True),
        Output("semantic-delete-pending", "data", allow_duplicate=True),
        Input("semantic-delete-confirm", "submit_n_clicks"),
        State("semantic-delete-pending", "data"),
        State("semantic-active-query", "data"),
        State("semantic-load-refresh", "data"),
        prevent_initial_call=True,
    )
    def confirm_saved_query_delete(clicks, qid, active, revision):
        if not clicks or qid is None:
            return no_update, no_update, no_update
        query = next((row["query"] for row in project.saved_semantic_searches.queries()
                      if row["query_id"] == int(qid)), None)
        if query is None:
            return no_update, no_update, None
        project.saved_semantic_searches.delete_query(int(qid))
        # Session state no longer stores active queries; the catalog delete
        # cascades through all per-geometry cached vectors.
        return ("" if query == active else no_update), int(revision or 0) + 1, None

    @app.callback(
        Output("semantic-active-query", "data", allow_duplicate=True),
        Output("topic-model-selector", "value", allow_duplicate=True),
        Input("semantic-search-button", "n_clicks"),
        Input("semantic-query", "n_submit"),
        State("semantic-query", "value"),
        prevent_initial_call=True,
    )
    def submit_semantic_query(_clicks, _enters, query):
        clean = str(query or "").strip()
        if clean:
            project.saved_semantic_searches.register(clean)
        return clean, (None if clean else no_update)

    @app.callback(
        Output("semantic-result-candidate", "data"),
        Input("semantic-active-query", "data"),
        Input("geometry-dropdown", "value"),
        prevent_initial_call=True,
    )
    def display_active_semantic(query, geometry_id):
        """Compute a candidate; the browser alone may commit current results."""
        clean = str(query or "").strip()
        if not clean:
            return {"query": "", "geometry_id": geometry_id, "scores": {}, "status": "", "range": None}
        if geometry_id is None or int(geometry_id) not in geometry_by_id:
            return {"query": clean, "geometry_id": geometry_id, "scores": {},
                    "status": "Select a geometry to search.", "range": None}
        selected = geometry_by_id[int(geometry_id)]
        if not selected.get("supports_query"):
            return {"query": clean, "geometry_id": geometry_id, "scores": {},
                    "status": "This geometry does not support semantic query transformation.",
                    "range": None}
        cache = project.saved_semantic_searches
        qid = cache.query_id(clean)
        if qid is None:
            return {"query": clean, "geometry_id": geometry_id, "scores": {},
                    "status": "This saved query is unavailable.", "range": None}
        try:
            cached = cache.find(clean, int(geometry_id), expected_query_id=qid)
        except ValueError:
            cached = None
        if cached is None:
            try:
                computed = project.semantic_search(clean, geometry_names=[str(selected["name"])])
                scores = computed[str(selected["name"])]
                cache.store(clean, int(geometry_id), scores, expected_query_id=qid)
            except (ValueError, KeyError, ConfigurationError, OSError) as error:
                return {"query": clean, "geometry_id": geometry_id, "scores": {},
                        "status": f"Semantic search unavailable: {error}", "range": None}
            current_range = None
        else:
            scores = cached["scores"]
            current_range = cached["range"]
        # A query deleted after reading the cache must not be displayed.
        if cache.query_id(clean) != qid:
            return {"query": clean, "geometry_id": geometry_id, "scores": {},
                    "status": "This saved query was deleted.", "range": None}
        retained_scores = [float(value) for value in scores]
        token = f"{uuid.uuid4().hex}.{_semantic_vector_digest(retained_scores)}"
        with semantic_vector_lock:
            semantic_vectors[token] = (clean, int(geometry_id), int(qid), retained_scores)
            semantic_vector_order.append(token)
            while len(semantic_vector_order) > 8:
                semantic_vectors.pop(semantic_vector_order.popleft(), None)
        bounds = (
            [float(min(retained_scores)), float(max(retained_scores))]
            if retained_scores else None
        )
        return {"query": clean, "geometry_id": int(geometry_id), "query_id": qid,
                "token": token, "bounds": bounds, "status": "",
                "range": current_range}

    app.clientside_callback(
        """function(candidate, query, geometryId, currentColorMode) {
            const no = window.dash_clientside.no_update;
            if (!candidate || String(candidate.query || "") !== String(query || "").trim()
                || String(candidate.geometry_id) !== String(geometryId)) {
                return [no, no, no, no, no];
            }
            const payload = candidate.token
                ? {query: candidate.query, geometry_id: candidate.geometry_id,
                   query_id: candidate.query_id, token: candidate.token,
                   bounds: candidate.bounds}
                : {};
            // Clearing the semantic query when selecting a Topic Model must not
            // undo topic coloring. Only an active semantic color or an attempted
            // semantic query without results should return to plain.
            const nextColorMode = Object.keys(payload).length
                ? "semantic"
                : (String(query || "").trim() || currentColorMode === "semantic" ? "plain" : no);
            return [payload, candidate.status || "", candidate.range || null,
                    candidate.range || null, nextColorMode];
        }""",
        Output("semantic-scores-store", "data", allow_duplicate=True),
        Output("semantic-search-status", "children", allow_duplicate=True),
        Output("semantic-range", "data", allow_duplicate=True),
        Output("desired-semantic-range-store", "data", allow_duplicate=True),
        Output("topic-color-mode", "data", allow_duplicate=True),
        Input("semantic-result-candidate", "data"),
        State("semantic-active-query", "data"),
        State("geometry-dropdown", "value"),
        State("topic-color-mode", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("semantic-range-slider", "min"),
        Output("semantic-range-slider", "max"),
        Output("semantic-range-slider", "step"),
        Output("semantic-range-slider", "value"),
        Output("semantic-range-slider", "marks"),
        Output("semantic-range-wrap", "style"),
        Output("nav-most-similar", "style"),
        Output("nav-least-similar", "style"),
        Input("semantic-scores-store", "data"),
        Input("geometry-dropdown", "value"),
        Input("semantic-active-query", "data"),
        Input("desired-semantic-range-store", "data"),
    )
    def update_semantic_range(
        all_scores: dict[str, Any] | None,
        geometry_id: int | None,
        active_query: str | None,
        desired_range: list[float] | None,
    ) -> tuple[
        float,
        float,
        float,
        list[float],
        dict[float, str],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        hidden = {"display": "none"}
        if geometry_id is None or not all_scores:
            return 0.0, 1.0, 0.001, [0.0, 1.0], {}, hidden, hidden, hidden
        geometry_name = str(geometry_by_id[int(geometry_id)]["name"])
        scores = selected_semantic_scores(all_scores, active_query, geometry_id).get(geometry_name)
        if not scores:
            return 0.0, 1.0, 0.001, [0.0, 1.0], {}, hidden, hidden, hidden
        values = np.asarray(scores, dtype=float)
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
        if np.isclose(lower, upper):
            upper = lower + 1e-9
        selected = [lower, upper]
        if desired_range and len(desired_range) == 2:
            requested_lower = max(lower, min(upper, float(desired_range[0])))
            requested_upper = max(lower, min(upper, float(desired_range[1])))
            selected = sorted([requested_lower, requested_upper])
        step = max((upper - lower) / 1000, 1e-9)
        marks = {
            lower: f"{lower:.3f}",
            upper: f"{upper:.3f}",
        }
        return (
            lower,
            upper,
            step,
            selected,
            marks,
            {"display": "block"},
            {"display": "inline-block"},
            {"display": "inline-block"},
        )

    @app.callback(
        Output("semantic-range", "data", allow_duplicate=True),
        Input("semantic-range-slider", "value"),
        prevent_initial_call=True,
    )
    def synchronize_semantic_range(value: list[float] | None) -> list[float] | Any:
        if not value or len(value) != 2:
            return no_update
        return sorted([float(value[0]), float(value[1])])


    @app.callback(
        Output("metadata-filter-condition", "options"),
        Output("metadata-filter-condition", "value"),
        Output("metadata-filter-type-help", "children"),
        Output("metadata-filter-value", "type"),
        Output("metadata-filter-value", "placeholder"),
        Output("metadata-filter-value-second", "type"),
        Output("metadata-filter-category-values", "options"),
        Input("metadata-filter-field", "value"),
    )
    def configure_metadata_filter_field(
        field: str | None,
    ) -> tuple[list[dict[str, str]], Any, str, str, str, str, list[dict[str, Any]]]:
        profile = metadata_profiles.get(str(field), {"type": "text", "options": []})
        kind = str(profile["type"])
        options = _metadata_operator_options(kind)
        input_type = "date" if kind == "datetime" else "text"
        placeholder = "Enter a value"
        if kind == "numeric":
            input_type = "number"
            placeholder = "Enter a number"
        elif kind == "datetime":
            placeholder = "YYYY-MM-DD"
        category_options = [
            {"label": str(value), "value": value}
            for value in profile.get("options", [])
        ]
        return (
            options,
            options[0]["value"] if options else None,
            f"Detected as {kind} metadata.",
            input_type,
            placeholder,
            input_type,
            category_options,
        )

    @app.callback(
        Output("metadata-filter-value-wrap", "style"),
        Output("metadata-filter-value-second-wrap", "style"),
        Output("metadata-filter-category-wrap", "style"),
        Input("metadata-filter-field", "value"),
        Input("metadata-filter-condition", "value"),
    )
    def configure_metadata_filter_values(
        field: str | None,
        condition: str | None,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        profile = metadata_profiles.get(str(field), {"type": "text"})
        kind = str(profile["type"])
        if condition in {"missing", "not_missing", "true", "false"}:
            return {"display": "none"}, {"display": "none"}, {"display": "none"}
        if kind == "categorical":
            return {"display": "none"}, {"display": "none"}, {"display": "block"}
        return (
            {"display": "block"},
            {"display": "block" if condition == "between" else "none"},
            {"display": "none"},
        )

    @app.callback(
        Output("metadata-filter-modal", "style"),
        Output("metadata-filters-store", "data", allow_duplicate=True),
        Output("metadata-filter-modal-status", "children"),
        Output("metadata-filter-value", "value"),
        Output("metadata-filter-value-second", "value"),
        Output("metadata-filter-category-values", "value"),
        Input("metadata-filter-add-button", "n_clicks"),
        Input("metadata-filter-modal-close", "n_clicks"),
        Input("metadata-filter-modal-cancel", "n_clicks"),
        Input("metadata-filter-modal-add", "n_clicks"),
        State("metadata-filter-field", "value"),
        State("metadata-filter-condition", "value"),
        State("metadata-filter-value", "value"),
        State("metadata-filter-value-second", "value"),
        State("metadata-filter-category-values", "value"),
        State("metadata-filters-store", "data"),
        prevent_initial_call=True,
    )
    def manage_metadata_filter_modal(
        open_clicks: int,
        close_clicks: int,
        cancel_clicks: int,
        add_clicks: int,
        field: str | None,
        condition: str | None,
        value: Any,
        value2: Any,
        category_values: list[Any] | None,
        current_filters: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, str], Any, str, Any, Any, list[Any]]:
        del open_clicks, close_clicks, cancel_clicks, add_clicks
        triggered = callback_context.triggered_id
        if triggered == "metadata-filter-add-button":
            return {"display": "flex"}, no_update, "", None, None, []
        if triggered in {"metadata-filter-modal-close", "metadata-filter-modal-cancel"}:
            return {"display": "none"}, no_update, "", no_update, no_update, no_update
        profile = metadata_profiles.get(str(field))
        if profile is None or not condition:
            return {"display": "flex"}, no_update, "Choose a field and condition.", no_update, no_update, no_update
        kind = str(profile["type"])
        stored_value: Any = category_values if kind == "categorical" else value
        if condition not in {"missing", "not_missing", "true", "false"}:
            if kind == "categorical" and not category_values:
                return {"display": "flex"}, no_update, "Choose at least one value.", no_update, no_update, no_update
            if kind != "categorical" and (value is None or str(value).strip() == ""):
                return {"display": "flex"}, no_update, "Enter a filter value.", no_update, no_update, no_update
            if condition == "between" and (value2 is None or str(value2).strip() == ""):
                return {"display": "flex"}, no_update, "Enter both ends of the range.", no_update, no_update, no_update
        clause = {
            "id": uuid.uuid4().hex,
            "field": str(field),
            "type": kind,
            "operator": str(condition),
            "value": stored_value,
            "value2": value2,
        }
        candidate = [*(current_filters or []), clause]
        normalized = _normalize_metadata_filters(candidate, metadata_profiles)
        try:
            from geometric_coder.storage.metadata_queries import compile_filter_sql
            compile_filter_sql(normalized, operator="and")
        except (TypeError, ValueError, re.error) as error:
            return {"display": "flex"}, no_update, str(error), no_update, no_update, no_update
        return {"display": "none"}, normalized, "", None, None, []

    @app.callback(
        Output("metadata-filters-store", "data", allow_duplicate=True),
        Input("metadata-filter-clear-button", "n_clicks"),
        Input({"type": "remove-metadata-filter", "index": ALL}, "n_clicks"),
        State("metadata-filters-store", "data"),
        prevent_initial_call=True,
    )
    def remove_metadata_filters(
        clear_clicks: int,
        remove_clicks: list[int],
        current_filters: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | Any:
        del clear_clicks
        triggered = callback_context.triggered_id
        if triggered == "metadata-filter-clear-button":
            return []
        if isinstance(triggered, dict):
            # Pattern-matching callbacks also fire when chip buttons are mounted
            # or remounted. Only remove a clause after an actual click; otherwise
            # changing AND/OR/XOR or rendering a newly added chip can erase state.
            identifier = str(triggered.get("index", ""))
            clauses = list(current_filters or [])
            position = next(
                (
                    index
                    for index, clause in enumerate(clauses)
                    if str(clause.get("id")) == identifier
                ),
                None,
            )
            if (
                position is None
                or position >= len(remove_clicks or [])
                or int((remove_clicks or [])[position] or 0) < 1
            ):
                return no_update
            return [
                clause
                for clause in clauses
                if str(clause.get("id")) != identifier
            ]
        return no_update

    @app.callback(
        Output("active-metadata-filter-chips", "children"),
        Output("metadata-filter-status", "children"),
        Input("metadata-filters-store", "data"),
        Input("metadata-filter-operator", "value"),
    )
    def render_metadata_filters(
        filters: list[dict[str, Any]] | None,
        operator: str | None,
    ) -> tuple[list[Any], str]:
        clauses = _normalize_metadata_filters(filters or [], metadata_profiles)
        if not clauses:
            message = (
                "No metadata columns were supplied."
                if not metadata_profiles
                else "No active metadata filters."
            )
            return [html.Span("No active filters", className="filter-empty-state")], message
        try:
            match_count = project.database.filtered_unit_count(
                clauses, operator=operator,
            )
            errors: list[str] = []
        except (TypeError, ValueError, re.error) as error:
            match_count = 0
            errors = [str(error)]
        chips = []
        for clause in clauses:
            label = _filter_clause_label(clause)
            chips.append(
                html.Span(
                    [
                        html.Span(label, className="metadata-filter-chip-label"),
                        html.Button(
                            "×",
                            id={
                                "type": "remove-metadata-filter",
                                "index": str(clause["id"]),
                            },
                            n_clicks=0,
                            title="Remove this filter",
                        ),
                    ],
                    className="metadata-filter-chip",
                    title=label,
                )
            )
        if errors:
            return chips, "; ".join(errors)
        mode = _normalize_boolean_operator(operator).upper()
        return chips, f"{match_count:,} of {len(units):,} units match · {mode}"

    @app.callback(
        Output("save-filter-set-modal", "style"),
        Output("save-filter-set-name", "value"),
        Output("save-filter-set-status", "children"),
        Output("saved-filter-refresh-store", "data", allow_duplicate=True),
        Input("metadata-filter-save-open-button", "n_clicks"),
        Input("save-filter-set-close", "n_clicks"),
        Input("save-filter-set-cancel", "n_clicks"),
        Input("save-filter-set-confirm", "n_clicks"),
        State("save-filter-set-name", "value"),
        State("metadata-filters-store", "data"),
        State("metadata-filter-operator", "value"),
        State("saved-filter-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def manage_save_filter_set(
        open_clicks: int,
        close_clicks: int,
        cancel_clicks: int,
        confirm_clicks: int,
        name: str | None,
        filters: list[dict[str, Any]] | None,
        operator: str | None,
        refresh: int | None,
    ) -> tuple[dict[str, str], Any, str, int]:
        del open_clicks, close_clicks, cancel_clicks, confirm_clicks
        triggered = callback_context.triggered_id
        current_refresh = int(refresh or 0)
        if triggered == "metadata-filter-save-open-button":
            return {"display": "flex"}, "", "", current_refresh
        if triggered in {"save-filter-set-close", "save-filter-set-cancel"}:
            return {"display": "none"}, "", "", current_refresh
        clean_name = (name or "").strip()
        clauses = _normalize_metadata_filters(filters or [], metadata_profiles)
        if not clean_name:
            return {"display": "flex"}, no_update, "Enter a name.", current_refresh
        if not clauses:
            return {"display": "flex"}, no_update, "Add at least one filter first.", current_refresh
        records = _read_saved_filter_sets(saved_filter_sets_path)
        records.append(
            {
                "id": uuid.uuid4().hex,
                "name": clean_name,
                "operator": _normalize_boolean_operator(operator),
                "filters": clauses,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        _write_saved_filter_sets(saved_filter_sets_path, records)
        return {"display": "none"}, "", "", current_refresh + 1

    @app.callback(
        Output("saved-filter-set-list", "children"),
        Input("saved-filter-refresh-store", "data"),
    )
    def render_saved_filter_sets(refresh: int | None) -> list[Any]:
        del refresh
        records = _read_saved_filter_sets(saved_filter_sets_path)
        if not records:
            return [html.P("No filter collections have been saved.", className="filter-empty-state")]
        return [
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(str(record.get("name", "Untitled"))),
                            html.Small(
                                f"{len(record.get('filters', []))} filters · "
                                f"{_normalize_boolean_operator(record.get('operator')).upper()}"
                            ),
                        ],
                        className="saved-filter-set-description",
                    ),
                    html.Button(
                        "Load",
                        id={"type": "load-filter-set", "index": str(record.get("id"))},
                        n_clicks=0,
                    ),
                    html.Button(
                        "×",
                        id={"type": "delete-filter-set", "index": str(record.get("id"))},
                        n_clicks=0,
                        className="danger-icon-action",
                        title="Delete saved filter collection",
                    ),
                ],
                className="saved-filter-set-row",
            )
            for record in records
        ]

    @app.callback(
        Output("load-filter-set-modal", "style"),
        Output("metadata-filters-store", "data", allow_duplicate=True),
        Output("metadata-filter-operator", "value", allow_duplicate=True),
        Output("saved-filter-refresh-store", "data", allow_duplicate=True),
        Output("load-filter-set-status", "children"),
        Output("saved-filter-delete-pending", "data"),
        Output("saved-filter-delete-confirm", "displayed"),
        Input("metadata-filter-load-open-button", "n_clicks"),
        Input("load-filter-set-close", "n_clicks"),
        Input("load-filter-set-done", "n_clicks"),
        Input({"type": "load-filter-set", "index": ALL}, "n_clicks"),
        Input({"type": "delete-filter-set", "index": ALL}, "n_clicks"),
        State("saved-filter-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def manage_load_filter_set(
        open_clicks: int,
        close_clicks: int,
        done_clicks: int,
        load_clicks: list[int],
        delete_clicks: list[int],
        refresh: int | None,
    ) -> tuple[dict[str, str], Any, Any, int, str]:
        del open_clicks, close_clicks, done_clicks
        triggered = callback_context.triggered_id
        current_refresh = int(refresh or 0)
        if triggered == "metadata-filter-load-open-button":
            return {"display": "flex"}, no_update, no_update, current_refresh, "", no_update, False
        if isinstance(triggered, str) and triggered in {
            "load-filter-set-close",
            "load-filter-set-done",
        }:
            return {"display": "none"}, no_update, no_update, current_refresh, "", no_update, False
        if not isinstance(triggered, dict):
            return no_update, no_update, no_update, current_refresh, "", no_update, False
        identifier = str(triggered.get("index", ""))
        records = _read_saved_filter_sets(saved_filter_sets_path)
        selected_position = next(
            (
                index
                for index, record in enumerate(records)
                if str(record.get("id")) == identifier
            ),
            None,
        )
        if selected_position is None:
            return {"display": "flex"}, no_update, no_update, current_refresh, "Saved collection not found.", no_update, False
        clicked_values = (
            delete_clicks
            if triggered.get("type") == "delete-filter-set"
            else load_clicks
        )
        if (
            selected_position >= len(clicked_values or [])
            or int((clicked_values or [])[selected_position] or 0) < 1
        ):
            # Saved-set rows use pattern IDs. Mounting those buttons is not a
            # user action and must not load or delete anything.
            return no_update, no_update, no_update, current_refresh, "", no_update, False
        selected = records[selected_position]
        if triggered.get("type") == "delete-filter-set":
            return {"display": "flex"}, no_update, no_update, current_refresh, "", identifier, True
        clauses = _normalize_metadata_filters(
            selected.get("filters", []), metadata_profiles
        )
        return (
            {"display": "none"},
            clauses,
            _normalize_boolean_operator(selected.get("operator")),
            current_refresh,
            "",
            no_update,
            False,
        )

    @app.callback(
        Output("saved-filter-refresh-store", "data", allow_duplicate=True),
        Output("saved-filter-delete-pending", "data", allow_duplicate=True),
        Input("saved-filter-delete-confirm", "submit_n_clicks"),
        State("saved-filter-delete-pending", "data"),
        State("saved-filter-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def confirm_filter_set_delete(clicks, identifier, revision):
        if not clicks or not identifier:
            return no_update, no_update
        records = _read_saved_filter_sets(saved_filter_sets_path)
        _write_saved_filter_sets(
            saved_filter_sets_path,
            [record for record in records if str(record.get("id")) != identifier],
        )
        return int(revision or 0) + 1, None

    @app.callback(
        Output("geometry-map", "figure"),
        Output("literal-search-status", "children"),
        Output("page-status", "children"),
        Output("topic-color-legend", "children"),
        Output("geometry-map-base-revision-store", "data"),
        Input("view-dropdown", "value"),
        Input("geometry-dropdown", "value"),
        Input("one-dimensional-jitter", "value"),
        Input("literal-query", "value"),
        Input("literal-options", "value"),
        Input("semantic-scores-store", "data"),
        Input("semantic-active-query", "data"),
        Input("semantic-range", "data"),
        Input("metadata-filters-store", "data"),
        Input("metadata-filter-operator", "value"),
        Input("explore-uncoded-only", "value"),
        Input("explore-map-filter-refresh-store", "data"),
        Input("session-dropdown", "value"),
        Input("page-size", "value"),
        Input("page-number", "value"),
        Input("topic-color-mode", "data"),
        Input("topic-model-selector", "value"),
        Input("topic-component-selector", "value"),
        Input("topic-model-refresh-store", "data"),
        State("explore-focal-unit-store", "data"),
        State("geometry-map-base-revision-store", "data"),
    )
    def update_map(
        view_id: int | None,
        geometry_id: int | None,
        jitter_strength: float | None,
        literal_query: str | None,
        literal_options: list[str] | None,
        all_semantic_scores: dict[str, Any] | None,
        active_query: str | None,
        semantic_range: list[float] | None,
        metadata_filters: list[dict[str, Any]] | None,
        metadata_filter_operator: str | None,
        uncoded_only_values: list[str] | None,
        map_filter_refresh: int | None,
        session_id: int,
        page_size: Any,
        page_number: Any,
        color_mode: str | None,
        topic_model_id: int | None,
        topic_component_id: str | None,
        topic_revision: int | None,
        focal_unit_id: int | None,
        base_revision: int | None,
    ) -> tuple[Any, str, str, Any, str]:
        del map_filter_refresh, topic_revision, base_revision
        next_revision = uuid.uuid4().hex
        if view_id is None or geometry_id is None:
            return _empty_figure("No computed geometry view is available."), "", "", "", next_revision
        options = set(literal_options or [])
        geometry_record = geometry_by_id[int(geometry_id)]
        semantic_scores = selected_semantic_scores(all_semantic_scores, active_query, geometry_id).get(str(geometry_record["name"]))
        try:
            filters = combine_filters(
                texts,
                literal_query=literal_query or "",
                case_sensitive="case" in options,
                regex="regex" in options,
                semantic_scores=semantic_scores,
                semantic_range=(
                    tuple(semantic_range)
                    if semantic_scores is not None and semantic_range
                    else None
                ),
            )
            literal_status = (
                f"{int(filters.literal_matches.sum()):,} literal matches"
                if literal_query
                else ""
            )
        except InvalidRegexError as error:
            filters = combine_filters(texts)
            literal_status = f"Invalid regular expression: {error}"

        size = _coerce_positive_int(page_size, default=5000)
        number = _coerce_positive_int(page_number, default=1)
        positions, page, filter_match_count = _sql_metadata_page(
            project,
            total_units=len(units),
            filters=metadata_filters or [],
            operator=metadata_filter_operator,
            uncoded_only="uncoded_only" in set(uncoded_only_values or []),
            page_size=size,
            page_number=number,
        )
        # Validate the selected descendant view *before* asking a provider to
        # load its entire coordinate universe.
        view = _display_view_record(
            project, geometry_id=int(geometry_id), view_id=int(view_id)
        )
        if view is None:
            return (
                _empty_figure("The selected view is unavailable for this geometry."),
                literal_status, "", "", next_revision,
            )
        coordinates = project.view_coordinate_rows(int(view_id), positions.tolist())
        page_units = units.at_positions(positions.tolist())
        page_eligible = filters.eligible[positions]
        page_semantic = (
            np.asarray(semantic_scores, dtype=float)[positions]
            if semantic_scores is not None
            else None
        )
        selected_color_mode = normalize_color_mode(color_mode)
        page_topics = None
        color_status = ""
        if selected_color_mode in {"dominant", "intensity"}:
            page_topics, color_status = load_topic_page_colors(
                project, model_id=topic_model_id, positions=positions,
                mode=selected_color_mode, selected_topic=topic_component_id,
            )
        seen = project.database.seen_unit_ids(int(session_id))
        figure = build_geometry_figure(
            coordinates=coordinates,
            units=page_units,
            eligible=page_eligible,
            seen_unit_ids=seen,
            focal_unit_id=int(focal_unit_id) if focal_unit_id is not None else None,
            semantic_scores=page_semantic if selected_color_mode == "semantic" else None,
            topic_colors=page_topics.colors if page_topics is not None else None,
            geometry_name=str(geometry_record["name"]),
            view_name=str(view["name"]),
            page_label=f"Page {page.page_index + 1} of {page.page_count}",
            uirevision=f"view:{int(view_id)}:page:{page.page_index}",
            jitter_strength=float(jitter_strength if jitter_strength is not None else 1.0),
            jitter_spread=(
                one_dimensional_jitter_spread(
                    project.view_coordinate_rows(
                        int(view_id),
                        np.linspace(
                            0, max(0, len(units) - 1),
                            num=min(len(units), 2048), dtype=int,
                        ).tolist(),
                    )[:, 0]
                ) if coordinates.shape[1] == 1 else None
            ),
        )
        # Plotly's view-scoped uirevision preserves deliberate same-view zooms.
        # Never replay stale relayoutData from another representation; changing
        # the view identity asks Plotly to autoscale the new coordinates.
        if callback_context.triggered_id in {"view-dropdown", "geometry-dropdown"}:
            figure.update_xaxes(autorange=True)
            figure.update_yaxes(autorange=True)
        page_status = (
            f"Page {page.page_index + 1} of {page.page_count} · "
            f"{len(positions):,} points shown · "
            f"{filter_match_count:,} points match filters"
        )
        if selected_color_mode in {"dominant", "intensity"}:
            page_status += f" · {color_status}"
        legend = topic_legend_content(html, page_topics) if page_topics is not None else ""
        overlay_frame = _geometry_overlay_frame(figure)
        if overlay_frame is not None:
            with overlay_frame_lock:
                overlay_frames[next_revision] = (int(session_id), overlay_frame)
                overlay_frame_order.append(next_revision)
                while len(overlay_frame_order) > 4:
                    overlay_frames.pop(overlay_frame_order.popleft(), None)
        return figure, literal_status, page_status, legend, next_revision

    @app.callback(
        Output("geometry-map", "figure", allow_duplicate=True),
        Output("session-progress", "children", allow_duplicate=True),
        Input("explore-focal-unit-store", "data"),
        Input("geometry-map-base-revision-store", "data"),
        Input("session-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_map_navigation_overlays(
        focal_unit_id: int | None,
        base_revision: str | int | None,
        session_id: int,
    ) -> tuple[Any, str]:
        seen = project.database.seen_unit_ids(int(session_id))
        progress = f"Visited {len(seen):,} of {len(units):,}"
        if callback_context.triggered_id == "session-dropdown":
            # The base-map callback will rebuild for the new session and then
            # advance geometry-map-base-revision-store. Patch only after that.
            return no_update, progress
        with overlay_frame_lock:
            retained = overlay_frames.get(str(base_revision))
        if retained is None or retained[0] != int(session_id):
            return no_update, progress
        payload = _geometry_overlay_payload(
            retained[1],
            seen_unit_ids=seen,
            focal_unit_id=focal_unit_id,
        )
        if payload is None:
            return no_update, progress
        patch = Patch()
        for trace_index, values in payload.items():
            for property_name, property_value in values.items():
                patch["data"][int(trace_index)][property_name] = property_value
        return patch, progress

    @app.callback(
        Output("page-number", "max"),
        Output("page-number", "value", allow_duplicate=True),
        Input("page-previous", "n_clicks"),
        Input("page-next", "n_clicks"),
        Input("page-size", "value"),
        Input("metadata-filters-store", "data"),
        Input("metadata-filter-operator", "value"),
        Input("explore-uncoded-only", "value"),
        Input("explore-map-filter-refresh-store", "data"),
        State("page-number", "value"),
        prevent_initial_call=True,
    )
    def change_page(
        previous_clicks: int,
        next_clicks: int,
        page_size: Any,
        metadata_filters: list[dict[str, Any]] | None,
        metadata_filter_operator: str | None,
        uncoded_only_values: list[str] | None,
        map_filter_refresh: int | None,
        page_number: Any,
    ) -> tuple[int, int]:
        del previous_clicks, next_clicks, map_filter_refresh
        size = _coerce_positive_int(page_size, default=5000)
        matching_count = project.database.filtered_unit_count(
            metadata_filters or [],
            operator=metadata_filter_operator,
            uncoded_only="uncoded_only" in set(uncoded_only_values or []),
        )
        page_count = deterministic_page_count(matching_count, page_size=size)
        current = min(max(1, _coerce_positive_int(page_number, default=1)), page_count)
        triggered = callback_context.triggered_id
        if triggered in {
            "page-size",
            "metadata-filters-store",
            "metadata-filter-operator",
            "explore-uncoded-only",
            "explore-map-filter-refresh-store",
        }:
            current = 1
        elif triggered == "page-previous":
            current = max(1, current - 1)
        elif triggered == "page-next":
            current = min(page_count, current + 1)
        return page_count, current

    navigation_inputs = [
        Input("geometry-map", "clickData"),
        Input("nav-nearest", "n_clicks"),
        Input("nav-farthest", "n_clicks"),
        Input("nav-most-similar", "n_clicks"),
        Input("nav-least-similar", "n_clicks"),
        Input("nav-random", "n_clicks"),
    ]

    @app.callback(
        Output("focal-unit-store", "data", allow_duplicate=True),
        Output("navigation-history-store", "data", allow_duplicate=True),
        Output("navigation-status", "children"),
        *navigation_inputs,
        State("navigation-new-only", "value"),
        State("session-dropdown", "value"),
        State("geometry-dropdown", "value"),
        State("focal-unit-store", "data"),
        State("literal-query", "value"),
        State("literal-options", "value"),
        State("semantic-scores-store", "data"),
        State("semantic-active-query", "data"),
        State("semantic-range", "data"),
        State("metadata-filters-store", "data"),
        State("metadata-filter-operator", "value"),
        State("explore-uncoded-only", "value"),
        State("page-size", "value"),
        State("page-number", "value"),
        State("navigation-history-store", "data"),
        prevent_initial_call=True,
    )
    def navigate(
        click_data: dict[str, Any] | None,
        nearest_clicks: int,
        farthest_clicks: int,
        most_similar_clicks: int,
        least_similar_clicks: int,
        random_clicks: int,
        new_only_values: list[str] | None,
        session_id: int,
        geometry_id: int | None,
        focal_unit_id: int | None,
        literal_query: str | None,
        literal_options: list[str] | None,
        all_semantic_scores: dict[str, Any] | None,
        active_query: str | None,
        semantic_range: list[float] | None,
        metadata_filters: list[dict[str, Any]] | None,
        metadata_filter_operator: str | None,
        uncoded_only_values: list[str] | None,
        page_size: Any,
        page_number: Any,
        history: dict[str, Any] | None,
    ) -> tuple[int | None, dict[str, Any] | Any, str]:
        del (
            nearest_clicks,
            farthest_clicks,
            most_similar_clicks,
            least_similar_clicks,
            random_clicks,
        )
        triggered = callback_context.triggered_id
        if triggered == "geometry-map":
            unit_id = _clicked_unit_id(click_data)
            if unit_id is None:
                return no_update, no_update, no_update
            method = "plot_click"
        else:
            strategy_by_button = {
                "nav-nearest": "nearest",
                "nav-farthest": "farthest",
                "nav-most-similar": "most_similar",
                "nav-least-similar": "least_similar",
                "nav-random": "random",
            }
            strategy = strategy_by_button[str(triggered)]
            options = set(literal_options or [])
            geometry_record = (
                geometry_by_id[int(geometry_id)] if geometry_id is not None else None
            )
            semantic_scores = (
                selected_semantic_scores(all_semantic_scores, active_query, geometry_id).get(str(geometry_record["name"]))
                if geometry_record is not None
                else None
            )
            try:
                filters = combine_filters(
                    texts,
                    literal_query=literal_query or "",
                    case_sensitive="case" in options,
                    regex="regex" in options,
                    semantic_scores=semantic_scores,
                    semantic_range=(
                        tuple(semantic_range)
                        if semantic_scores is not None and semantic_range
                        else None
                    ),
                )
            except InvalidRegexError as error:
                return (
                    no_update,
                    no_update,
                    f"Fix the regular expression before navigating: {error}",
                )

            positions, _page, _match_count = _sql_metadata_page(
                project,
                total_units=len(units),
                filters=metadata_filters or [],
                operator=metadata_filter_operator,
                uncoded_only="uncoded_only" in set(uncoded_only_values or []),
                page_size=_coerce_positive_int(page_size, default=5000),
                page_number=_coerce_positive_int(page_number, default=1),
            )
            page_ids = np.asarray([unit_ids[int(position)] for position in positions], dtype=int)
            avoid_seen = "new_only" in (new_only_values or [])
            seen = project.database.seen_unit_ids(int(session_id)) if avoid_seen else set()
            if strategy in {"most_similar", "least_similar"}:
                if semantic_scores is None:
                    return no_update, no_update, "Run a semantic search in the selected geometry first."
                page_scores = np.asarray(semantic_scores, dtype=float)[positions]
                page_eligible = np.asarray(filters.eligible, dtype=bool)[positions]
                candidates = [
                    index
                    for index, unit_id_value in enumerate(page_ids.tolist())
                    if bool(page_eligible[index]) and int(unit_id_value) not in seen
                ]
                if not candidates:
                    qualifier = "new " if avoid_seen else ""
                    return (
                        no_update,
                        no_update,
                        f"No {qualifier}unit satisfies the current page and filters.",
                    )
                selected_index = (
                    max(candidates, key=lambda index: float(page_scores[index]))
                    if strategy == "most_similar"
                    else min(candidates, key=lambda index: float(page_scores[index]))
                )
                unit_id = int(page_ids[selected_index])
                method = str(strategy)
            else:
                matrix = None
                if strategy in {"nearest", "farthest"}:
                    if geometry_id is None:
                        return no_update, no_update, "Choose a geometry before using nearest or farthest."
                    if focal_unit_id is None:
                        return no_update, no_update, "Select a point on the current page first."
                    if int(focal_unit_id) not in set(page_ids.tolist()):
                        return (
                            no_update,
                            no_update,
                            "The focal point is not on the current page. "
                            "Select a point on this page first.",
                        )
                    matrix = project.geometry_page_rows(int(geometry_id), positions)

                try:
                    unit_id = recommend_unit(
                        strategy=strategy,  # type: ignore[arg-type]
                        unit_ids=page_ids,
                        seen_unit_ids=seen,
                        eligible=np.asarray(filters.eligible, dtype=bool)[positions],
                        matrix=matrix,
                        focal_unit_id=(
                            int(focal_unit_id)
                            if focal_unit_id is not None
                            and int(focal_unit_id) in set(page_ids.tolist())
                            else None
                        ),
                    )
                except (KeyError, ValueError) as error:
                    return no_update, no_update, str(error)
                if unit_id is None:
                    qualifier = "new " if avoid_seen else ""
                    return (
                        no_update,
                        no_update,
                        f"No {qualifier}unit satisfies the current page and filters.",
                    )
                method = str(strategy)

        project.record_visit(
            int(session_id),
            unit_id,
            method=method,
            source_unit_id=int(focal_unit_id) if focal_unit_id is not None else None,
            geometry_id=int(geometry_id) if geometry_id is not None else None,
        )
        updated_history = _append_navigation_history(history, int(unit_id))
        return unit_id, updated_history, ""

    @app.callback(
        Output("focal-unit-store", "data", allow_duplicate=True),
        Output("navigation-history-store", "data", allow_duplicate=True),
        Input("nav-back", "n_clicks"),
        Input("nav-forward", "n_clicks"),
        State("navigation-history-store", "data"),
        State("session-dropdown", "value"),
        State("geometry-dropdown", "value"),
        prevent_initial_call=True,
    )
    def navigate_history(
        back_clicks: int,
        forward_clicks: int,
        history: dict[str, Any] | None,
        session_id: int,
        geometry_id: int | None,
    ) -> tuple[int | Any, dict[str, Any] | Any]:
        del back_clicks, forward_clicks
        direction = -1 if callback_context.triggered_id == "nav-back" else 1
        moved = _move_navigation_history(history, direction)
        if moved is None:
            return no_update, no_update
        unit_id, updated = moved
        project.record_visit(
            int(session_id),
            int(unit_id),
            method="history_back" if direction < 0 else "history_forward",
            geometry_id=int(geometry_id) if geometry_id is not None else None,
        )
        return int(unit_id), updated

    @app.callback(
        Output("nav-back", "disabled"),
        Output("nav-forward", "disabled"),
        Input("navigation-history-store", "data"),
    )
    def set_history_button_state(
        history: dict[str, Any] | None,
    ) -> tuple[bool, bool]:
        normalized = _normalize_navigation_history(history)
        index = int(normalized["index"])
        items = list(normalized["items"])
        return index <= 0, index < 0 or index >= len(items) - 1

    @app.callback(
        Output("focal-key", "children"),
        Output("context-content", "children"),
        Output("metadata-content", "children"),
        Input("explore-focal-unit-store", "data"),
        Input("context-level", "value"),
        Input("context-window", "value"),
    )
    def update_reading_panel(
        focal_unit_id: int | None,
        level_index: int,
        window: int,
    ) -> tuple[Any, Any, Any]:
        if focal_unit_id is None:
            placeholder = "Select a point or use a navigation command to begin reading."
            return "No focal unit", placeholder, ""
        focal = project.unit(int(focal_unit_id))
        context = project.context(int(focal_unit_id), level=int(level_index), window=int(window))
        key_label = " › ".join(f"{name}: {value}" for name, value in focal["user_key"].items())
        context_children = _render_context_units(
            html,
            dcc,
            context,
            focal_unit_id=int(focal_unit_id),
            key_columns=list(metadata["key_columns"]),
            selected_unit_ids=[int(focal_unit_id)],
            checkbox_type="explore-span-checkbox",
            span_selection_enabled=True,
        )
        metadata_children = _metadata_table(html, focal["metadata"])
        return key_label, context_children, metadata_children

    @app.callback(
        Output("explore-span-selection-store", "data"),
        Input("explore-focal-unit-store", "data"),
        Input("context-level", "value"),
        Input("context-window", "value"),
    )
    def reset_explore_span_selection(
        focal_unit_id: int | None, level_index: int, window: int
    ) -> list[int]:
        del level_index, window
        return [int(focal_unit_id)] if focal_unit_id is not None else []

    @app.callback(
        Output("explore-span-selection-store", "data", allow_duplicate=True),
        Input({"type": "explore-span-checkbox", "unit_id": ALL}, "value"),
        State({"type": "explore-span-checkbox", "unit_id": ALL}, "id"),
        State("focal-unit-store", "data"),
        State("context-level", "value"),
        State("context-window", "value"),
        State("explore-span-selection-store", "data"),
        prevent_initial_call=True,
    )
    def update_explore_span_selection(
        values: list[list[int]], ids: list[dict[str, Any]],
        focal_unit_id: int | None, level_index: int, window: int,
        current: list[int] | None,
    ) -> Any:
        triggered = callback_context.triggered_id
        if focal_unit_id is None or not isinstance(triggered, dict):
            return no_update
        clicked = int(triggered.get("unit_id"))
        value_by_id = {
            int(identifier["unit_id"]): value
            for identifier, value in zip(ids or [], values or [], strict=False)
        }
        checked = bool(value_by_id.get(clicked))
        if not checked and clicked not in set(current or []):
            return no_update
        context = project.context(
            int(focal_unit_id), level=int(level_index), window=int(window)
        )
        updated = _update_span_selection(
            context, focal_unit_id=int(focal_unit_id), current_unit_ids=current,
            clicked_unit_id=clicked, checked=checked,
        )
        return updated if updated != list(current or []) else no_update

    @app.callback(
        Output("context-window", "value", allow_duplicate=True),
        Input("context-level", "value"),
        Input("context-window-minus", "n_clicks"),
        Input("context-window-plus", "n_clicks"),
        State("context-window", "value"),
        prevent_initial_call=True,
    )
    def update_context_window(
        level_index: int,
        minus_clicks: int,
        plus_clicks: int,
        current_window: int | None,
    ) -> int:
        del level_index, minus_clicks, plus_clicks
        triggered = callback_context.triggered_id
        if triggered == "context-level":
            return 1
        current = max(1, int(current_window or 1))
        if triggered == "context-window-minus":
            return max(1, current - 1)
        if triggered == "context-window-plus":
            return current + 1
        return current

    @app.callback(
        Output("memo-dropdown", "options"),
        Input("memo-refresh-store", "data"),
    )
    def refresh_memo_options(refresh: int | None) -> list[dict[str, Any]]:
        del refresh
        return _memo_options(project.memos())

    @app.callback(
        Output("active-memo-store", "data", allow_duplicate=True),
        Output("memo-title", "value", allow_duplicate=True),
        Output("memo-body", "value", allow_duplicate=True),
        Output("memo-status", "children", allow_duplicate=True),
        Input("memo-dropdown", "value"),
        prevent_initial_call=True,
    )
    def load_memo(memo_id: int | None) -> tuple[Any, str, str, str]:
        if memo_id is None:
            return None, "", "", ""
        memo = project.memo(int(memo_id))
        return int(memo_id), str(memo["title"]), str(memo["body_markdown"]), ""

    @app.callback(
        Output("memo-dropdown", "value", allow_duplicate=True),
        Output("active-memo-store", "data", allow_duplicate=True),
        Output("memo-title", "value", allow_duplicate=True),
        Output("memo-body", "value", allow_duplicate=True),
        Output("memo-status", "children", allow_duplicate=True),
        Input("memo-new-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def new_memo(n_clicks: int) -> tuple[None, None, str, str, str]:
        del n_clicks
        return None, None, "", "", "New memo ready."

    @app.callback(
        Output("memo-dropdown", "options", allow_duplicate=True),
        Output("memo-dropdown", "value", allow_duplicate=True),
        Output("active-memo-store", "data", allow_duplicate=True),
        Output("memo-refresh-store", "data", allow_duplicate=True),
        Output("memo-status", "children", allow_duplicate=True),
        Output("memo-title", "value", allow_duplicate=True),
        Output("memo-body", "value", allow_duplicate=True),
        Input("memo-save-button", "n_clicks"),
        State("active-memo-store", "data"),
        State("memo-title", "value"),
        State("memo-body", "value"),
        State("memo-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def save_memo(
        n_clicks: int,
        memo_id: int | None,
        title: str | None,
        body: str | None,
        refresh: int | None,
    ) -> tuple[Any, int | None, int | None, int, str, Any, Any]:
        del n_clicks
        clean_title = (title or "").strip()
        if not clean_title:
            return (
                no_update, no_update, no_update, int(refresh or 0),
                "Enter a memo title.", no_update, no_update,
            )
        clean_body = body or ""
        hashtags, references = _extract_memo_metadata(clean_body)
        if memo_id is None:
            saved_id = project.create_memo(
                title=clean_title,
                body_markdown=clean_body,
                hashtags=hashtags,
                unit_references=references,
            )
            message = f"Created memo {clean_title!r}."
        else:
            saved_id = int(memo_id)
            version_id = project.update_memo(
                saved_id,
                title=clean_title,
                body_markdown=clean_body,
                hashtags=hashtags,
                unit_references=references,
            )
            del version_id
            message = f"Saved a new version of {clean_title!r}."
        options = _memo_options(project.memos())
        return (
            options, saved_id, saved_id, int(refresh or 0) + 1,
            message, clean_title, clean_body,
        )

    app.clientside_callback(
        """
        function(nClicks, body, focalUnitId) {
            if (!nClicks || focalUnitId === null || focalUnitId === undefined) {
                return window.dash_clientside.no_update;
            }
            const textarea = document.getElementById("memo-body");
            const current = body || "";
            const start = textarea && Number.isInteger(textarea.selectionStart)
                ? textarea.selectionStart : current.length;
            const end = textarea && Number.isInteger(textarea.selectionEnd)
                ? textarea.selectionEnd : start;
            const token = `[[unit:${focalUnitId}]]`;
            const before = current.slice(0, start);
            const after = current.slice(end);
            const prefix = before && !/[\\s]$/.test(before) ? " " : "";
            const suffix = after && !/^[\\s]/.test(after) ? " " : "";
            const insertion = prefix + token + suffix;
            const next = before + insertion + after;
            const cursor = before.length + insertion.length;
            window.setTimeout(function() {
                const el = document.getElementById("memo-body");
                if (el) {
                    el.focus();
                    el.setSelectionRange(cursor, cursor);
                }
            }, 0);
            return next;
        }
        """,
        Output("memo-body", "value", allow_duplicate=True),
        Input("memo-insert-reference-button", "n_clicks"),
        State("memo-body", "value"),
        State("focal-unit-store", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("memo-preview", "children"),
        Input("memo-body", "value"),
    )
    def preview_memo(body: str | None) -> Any:
        return _render_memo_preview(dcc, html, project, body or "")

    @app.callback(
        Output("memo-center-tags", "options"),
        Input("memo-refresh-store", "data"),
    )
    def refresh_memo_center_hashtags(refresh: int | None) -> list[dict[str, str]]:
        del refresh
        return [
            {"label": f"#{tag}", "value": tag}
            for tag in project.memo_hashtags()
        ]

    @app.callback(
        Output("memo-center-table", "data"),
        Output("memo-center-table", "selected_rows"),
        Output("memo-center-status", "children"),
        Input("memo-center-query", "value"),
        Input("memo-center-scope", "value"),
        Input("memo-center-tags", "value"),
        Input("memo-center-tag-operator", "value"),
        Input("memo-refresh-store", "data"),
        State("memo-center-selected-store", "data"),
    )
    def filter_memo_center(
        query: str | None,
        scope: str | None,
        hashtags: list[str] | None,
        operator: str | None,
        refresh: int | None,
        selected_memo_id: int | None,
    ) -> tuple[list[dict[str, Any]], list[int], str]:
        del refresh
        matches = project.search_memos(
            query=query or "",
            scope=str(scope or "full"),
            hashtags=hashtags or [],
            hashtag_operator=str(operator or "and"),
        )
        rows = [
            {
                "memo_id": int(memo["memo_id"]),
                "title": str(memo["title"]),
                "hashtags": " ".join(f"#{tag}" for tag in memo["hashtags"]),
                "updated_at": str(memo["updated_at"]),
            }
            for memo in matches
        ]
        selected_rows = [
            index for index, row in enumerate(rows)
            if selected_memo_id is not None and int(row["memo_id"]) == int(selected_memo_id)
        ]
        return rows, selected_rows[:1], f"{len(rows):,} memo{'s' if len(rows) != 1 else ''} found"

    @app.callback(
        Output("memo-center-selected-store", "data"),
        Output("memo-center-version-store", "data"),
        Output("memo-center-title", "children"),
        Output("memo-center-preview", "children"),
        Output("memo-version-label", "children"),
        Output("memo-version-previous", "disabled"),
        Output("memo-version-next", "disabled"),
        Output("memo-center-edit-button", "disabled"),
        Output("memo-center-edit-title", "value"),
        Output("memo-center-edit-body", "value"),
        Input("memo-center-table", "selected_rows"),
        State("memo-center-table", "data"),
        Input("memo-refresh-store", "data"),
    )
    def select_memo_center(
        selected_rows: list[int] | None, rows: list[dict[str, Any]] | None, refresh: int | None
    ) -> tuple[Any, ...]:
        del refresh
        if not selected_rows or not rows:
            empty = html.P("Choose a memo from the results to read it here.", className="memo-preview-empty")
            return None, None, "Select a memo", empty, "No version selected", True, True, True, "", ""
        index = int(selected_rows[0])
        if index < 0 or index >= len(rows):
            return None, None, "Select a memo", "", "No version selected", True, True, True, "", ""
        memo_id = int(rows[index]["memo_id"])
        versions = project.memo_versions(memo_id)
        memo = versions[-1]
        number = int(memo["version_number"])
        return (
            memo_id, number, str(memo["title"]),
            _render_memo_preview(dcc, html, project, str(memo["body_markdown"])),
            f"Version {number} of {len(versions)}", number <= 1, number >= len(versions), False,
            str(memo["title"]), str(memo["body_markdown"]),
        )

    @app.callback(
        Output("memo-center-version-store", "data", allow_duplicate=True),
        Output("memo-center-title", "children", allow_duplicate=True),
        Output("memo-center-preview", "children", allow_duplicate=True),
        Output("memo-version-label", "children", allow_duplicate=True),
        Output("memo-version-previous", "disabled", allow_duplicate=True),
        Output("memo-version-next", "disabled", allow_duplicate=True),
        Output("memo-center-edit-title", "value", allow_duplicate=True),
        Output("memo-center-edit-body", "value", allow_duplicate=True),
        Input("memo-version-previous", "n_clicks"),
        Input("memo-version-next", "n_clicks"),
        State("memo-center-selected-store", "data"),
        State("memo-center-version-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_memo_versions(
        previous: int, next_clicks: int, memo_id: int | None, version_number: int | None
    ) -> tuple[Any, ...]:
        del previous, next_clicks
        if memo_id is None:
            return (no_update,) * 8
        versions = project.memo_versions(int(memo_id))
        current = int(version_number or len(versions))
        if callback_context.triggered_id == "memo-version-previous":
            target = max(1, current - 1)
        else:
            target = min(len(versions), current + 1)
        memo = project.memo(int(memo_id), target)
        return (
            target, str(memo["title"]),
            _render_memo_preview(dcc, html, project, str(memo["body_markdown"])),
            f"Version {target} of {len(versions)}", target <= 1, target >= len(versions),
            str(memo["title"]), str(memo["body_markdown"]),
        )

    @app.callback(
        Output("memo-center-edit-store", "data"),
        Output("memo-center-editor-wrap", "style"),
        Output("memo-center-preview", "style", allow_duplicate=True),
        Input("memo-center-edit-button", "n_clicks"),
        Input("memo-center-cancel-button", "n_clicks"),
        State("memo-center-selected-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_memo_center_edit(
        edit_clicks: int, cancel_clicks: int, memo_id: int | None
    ) -> tuple[bool, dict[str, str], dict[str, str]]:
        del edit_clicks, cancel_clicks
        editing = callback_context.triggered_id == "memo-center-edit-button" and memo_id is not None
        return (
            editing,
            ({"display": "block"} if editing else {"display": "none"}),
            ({"display": "none"} if editing else {}),
        )

    @app.callback(
        Output("memo-center-edit-preview", "children"),
        Input("memo-center-edit-body", "value"),
    )
    def preview_memo_center_edit(body: str | None) -> Any:
        return _render_memo_preview(dcc, html, project, body or "")

    @app.callback(
        Output("memo-refresh-store", "data", allow_duplicate=True),
        Output("memo-center-version-store", "data", allow_duplicate=True),
        Output("memo-center-title", "children", allow_duplicate=True),
        Output("memo-center-preview", "children", allow_duplicate=True),
        Output("memo-version-label", "children", allow_duplicate=True),
        Output("memo-version-previous", "disabled", allow_duplicate=True),
        Output("memo-version-next", "disabled", allow_duplicate=True),
        Output("memo-center-edit-store", "data", allow_duplicate=True),
        Output("memo-center-editor-wrap", "style", allow_duplicate=True),
        Output("memo-center-preview", "style", allow_duplicate=True),
        Output("memo-center-edit-status", "children"),
        Input("memo-center-save-button", "n_clicks"),
        State("memo-center-selected-store", "data"),
        State("memo-center-edit-title", "value"),
        State("memo-center-edit-body", "value"),
        State("memo-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def save_memo_center_version(
        clicks: int, memo_id: int | None, title: str | None, body: str | None, refresh: int | None
    ) -> tuple[Any, ...]:
        del clicks
        if memo_id is None:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, True, {"display": "none"}, {}, "Select a memo first."
        clean_title = (title or "").strip()
        if not clean_title:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, True, {"display": "block"}, {"display": "none"}, "Enter a memo title."
        clean_body = body or ""
        hashtags, references = _extract_memo_metadata(clean_body)
        project.update_memo(int(memo_id), title=clean_title, body_markdown=clean_body, hashtags=hashtags, unit_references=references)
        versions = project.memo_versions(int(memo_id))
        number = len(versions)
        memo = versions[-1]
        return (
            int(refresh or 0) + 1, number, clean_title,
            _render_memo_preview(dcc, html, project, clean_body),
            f"Version {number} of {number}", number <= 1, True, False, {"display": "none"}, {},
            f"Saved version {number}.",
        )


    @app.callback(
        Output("memo-presentation-link", "href"),
        Output("memo-presentation-link", "className"),
        Input("memo-center-selected-store", "data"),
        Input("memo-center-version-store", "data"),
    )
    def update_memo_presentation_link(
        memo_id: int | None, version_number: int | None
    ) -> tuple[str | None, str]:
        base_class = "memo-presentation-link"
        if memo_id is None or version_number is None:
            return None, f"{base_class} disabled"
        return (
            f"/?presentation=memo&memo_id={int(memo_id)}&version={int(version_number)}",
            base_class,
        )

    @app.callback(
        Output("memo-presentation-shell", "className"),
        Input("memo-presentation-location", "search"),
    )
    def show_memo_presentation(search: str | None) -> str:
        values = parse_qs((search or "").lstrip("?"))
        if values.get("presentation", [None])[0] == "memo":
            return "memo-presentation-shell memo-presentation-active"
        return "memo-presentation-shell"

    @app.callback(
        Output("memo-presentation-saved-filter", "options"),
        Input("memo-presentation-location", "pathname"),
        Input("saved-filter-refresh-store", "data"),
    )
    def refresh_presentation_saved_filters(
        pathname: str | None, refresh: int | None
    ) -> list[dict[str, str]]:
        del pathname, refresh
        options = [{"label": "All observations", "value": "__all__"}]
        options.extend(
            {
                "label": str(record.get("name", "Untitled filter")),
                "value": str(record.get("id")),
            }
            for record in _read_saved_filter_sets(saved_filter_sets_path)
        )
        return options

    @app.callback(
        Output("memo-presentation-title", "children"),
        Output("memo-presentation-version", "children"),
        Output("memo-presentation-body", "children"),
        Input("memo-presentation-location", "search"),
    )
    def load_memo_presentation(search: str | None) -> tuple[str, str, Any]:
        memo_id, requested_version = _parse_memo_presentation_request(search)
        if memo_id is None:
            return (
                "Memo presentation",
                "No memo selected",
                html.P(
                    "Open Presentation mode from a memo in Memo Center.",
                    className="memo-preview-empty",
                ),
            )
        try:
            versions = project.memo_versions(int(memo_id))
        except KeyError:
            versions = []
        if not versions:
            return (
                "Memo presentation",
                "Memo not found",
                html.P("This memo is unavailable.", className="memo-preview-empty"),
            )
        memo = _memo_version_by_number(versions, requested_version)
        number = int(memo["version_number"])
        return (
            str(memo["title"]),
            f"Version {number} of {len(versions)}",
            _render_memo_preview(
                dcc,
                html,
                project,
                str(memo["body_markdown"]),
                unit_hash_prefix="geco-presentation-unit-",
                link_hashtags=False,
            ),
        )

    @app.callback(
        Output("memo-presentation-page-number", "max"),
        Output("memo-presentation-page-number", "value"),
        Input("memo-presentation-page-previous", "n_clicks"),
        Input("memo-presentation-page-next", "n_clicks"),
        Input("memo-presentation-page-size", "value"),
        Input("memo-presentation-saved-filter", "value"),
        Input("memo-presentation-location", "hash"),
        Input("memo-presentation-location", "search"),
        State("memo-presentation-page-number", "value"),
    )
    def update_memo_presentation_page(
        previous_clicks: int,
        next_clicks: int,
        page_size: Any,
        saved_filter_id: str | None,
        location_hash: str | None,
        location_search: str | None,
        current_page: Any,
    ) -> tuple[Any, Any]:
        del previous_clicks, next_clicks
        if not _memo_presentation_is_active(location_search):
            return no_update, no_update
        size = _coerce_positive_int(page_size, default=5000)
        filters, operator = _presentation_filter_definition(
            saved_filter_sets_path,
            saved_filter_id,
            metadata_profiles,
        )
        mask, _errors = _evaluate_metadata_filters(units, filters, operator=operator)
        page_count = deterministic_page_count(int(mask.sum()), page_size=size)
        current = min(max(1, _coerce_positive_int(current_page, default=1)), page_count)
        triggered = callback_context.triggered_id
        if triggered in {"memo-presentation-page-size", "memo-presentation-saved-filter"}:
            current = 1
        elif triggered == "memo-presentation-page-previous":
            current = max(1, current - 1)
        elif triggered == "memo-presentation-page-next":
            current = min(page_count, current + 1)
        elif triggered == "memo-presentation-location":
            focal_unit_id = _presentation_unit_id(location_hash)
            focal_position = (
                project.database.unit_position_for_id(focal_unit_id)
                if focal_unit_id is not None else None
            )
            if focal_position is not None:
                target = _deterministic_page_number_for_position(
                    mask,
                    corpus_position=focal_position,
                    page_size=size,
                )
                if target is not None:
                    if target == current:
                        # Avoid emitting a redundant page-value update. Otherwise
                        # the map can redraw a second time with page-number as the
                        # apparent trigger and Plotly may discard the zoom camera.
                        return page_count, no_update
                    current = target
                else:
                    # Out-of-filter references are overlaid on the current page.
                    # Keep the page input untouched so focal selection is the only
                    # redraw trigger and the current camera can be preserved.
                    return page_count, no_update
        return page_count, current

    @app.callback(
        Output("memo-presentation-view", "options"),
        Output("memo-presentation-view", "value"),
        Input("representation-refresh-store", "data"),
        Input("view-refresh-store", "data"),
        Input("memo-presentation-location", "search"),
        State("memo-presentation-view", "value"),
    )
    def refresh_presentation_views(
        _representation_revision: int | None,
        _view_revision: int | None,
        location_search: str | None,
        current_view_id: int | None,
    ) -> tuple[Any, Any]:
        if not _memo_presentation_is_active(location_search):
            return no_update, no_update
        current_geometries = project.geometries(public_only=True)
        records = _presentation_view_records(project, current_geometries)
        presentation_view_by_id.clear()
        presentation_view_by_id.update({
            int(row["view_id"]): row for row in records
        })
        options = _presentation_view_options(records, current_geometries)
        valid_ids = {int(option["value"]) for option in options}
        selected = (
            int(current_view_id)
            if current_view_id is not None and int(current_view_id) in valid_ids
            else (int(options[0]["value"]) if options else None)
        )
        return options, selected

    @app.callback(
        Output("memo-presentation-map", "figure"),
        Output("memo-presentation-page-status", "children"),
        Input("memo-presentation-view", "value"),
        Input("representation-refresh-store", "data"),
        Input("view-refresh-store", "data"),
        Input("memo-presentation-saved-filter", "value"),
        Input("memo-presentation-page-size", "value"),
        Input("memo-presentation-page-number", "value"),
        Input("memo-presentation-location", "hash"),
        Input("memo-presentation-location", "search"),
        State("memo-presentation-map", "relayoutData"),
    )
    def update_memo_presentation_map(
        view_id: int | None,
        _representation_revision: int | None,
        _view_revision: int | None,
        saved_filter_id: str | None,
        page_size: Any,
        page_number: Any,
        location_hash: str | None,
        location_search: str | None,
        relayout_data: dict[str, Any] | None,
    ) -> tuple[Any, Any]:
        if not _memo_presentation_is_active(location_search):
            return no_update, no_update
        if view_id is None or int(view_id) not in presentation_view_by_id:
            return _empty_figure("No compatible display representation is available."), ""
        view_record = presentation_view_by_id[int(view_id)]
        geometry_id = int(view_record["geometry_id"])
        geometry_record = next(
            (record for record in project.geometries(public_only=True)
             if int(record["geometry_id"]) == geometry_id),
            None,
        )
        if geometry_record is None:
            return _empty_figure("This view does not belong to a public geometry."), ""
        size = _coerce_positive_int(page_size, default=5000)
        number = _coerce_positive_int(page_number, default=1)
        filters, operator = _presentation_filter_definition(
            saved_filter_sets_path,
            saved_filter_id,
            metadata_profiles,
        )
        positions, page, mask, _errors = _metadata_filtered_page(
            units,
            filters,
            operator=operator,
            page_size=size,
            page_number=number,
        )
        focal_unit_id = _presentation_unit_id(location_hash)
        display_positions = positions.tolist()
        focal_position = (
            project.database.unit_position_for_id(focal_unit_id)
            if focal_unit_id is not None else None
        )
        if focal_position is not None:
            if focal_position not in display_positions:
                display_positions.append(focal_position)
        display = np.asarray(display_positions, dtype=int)
        coordinates = project.view_coordinate_rows(int(view_id), display.tolist())
        page_units = units.at_positions(display.tolist())
        eligible = mask[display] if display.size else np.asarray([], dtype=bool)
        figure = build_geometry_figure(
            coordinates=coordinates,
            units=page_units,
            eligible=eligible,
            seen_unit_ids=(),
            focal_unit_id=focal_unit_id,
            geometry_name=str(geometry_record["name"]),
            view_name=str(view_record["name"]),
            page_label=f"Page {page.page_index + 1} of {page.page_count}",
            uirevision=f"memo-presentation:{int(view_id)}:page:{page.page_index}",
        )
        # Focal selection rebuilds the figure so an out-of-filter memo reference
        # can be overlaid. Preserve the user's current camera when only the focal
        # observation changes; deliberate view/filter/page changes should reset
        # the camera to the newly requested surface.
        if callback_context.triggered_id not in {
            "memo-presentation-view",
            "memo-presentation-saved-filter",
            "memo-presentation-page-size",
            "memo-presentation-page-number",
        }:
            _apply_relayout_ranges(figure, relayout_data)
        status = (
            f"Page {page.page_index + 1} of {page.page_count} · "
            f"{len(positions):,} filter points · {int(mask.sum()):,} total matches"
        )
        return figure, status

    @app.callback(
        Output("memo-presentation-location", "hash", allow_duplicate=True),
        Input("memo-presentation-map", "clickData"),
        prevent_initial_call=True,
    )
    def select_memo_presentation_map_point(
        click_data: dict[str, Any] | None,
    ) -> Any:
        unit_id = _clicked_unit_id(click_data)
        if unit_id is None:
            return no_update
        return f"#geco-presentation-unit-{int(unit_id)}"

    @app.callback(
        Output("memo-presentation-focal-key", "children"),
        Output("memo-presentation-text", "children"),
        Output("memo-presentation-metadata", "children"),
        Input("memo-presentation-location", "hash"),
    )
    def update_memo_presentation_reader(
        location_hash: str | None,
    ) -> tuple[Any, Any, Any]:
        unit_id = _presentation_unit_id(location_hash)
        if unit_id is None:
            return (
                "No focal observation",
                "Select a linked observation in the memo or click a point on the map.",
                "",
            )
        try:
            focal = project.unit(int(unit_id))
        except KeyError:
            return "Observation unavailable", "The referenced observation could not be loaded.", ""
        key_label = " › ".join(
            f"{name}: {value}" for name, value in focal["user_key"].items()
        )
        context_level = int(hierarchy[-1]["level_index"])
        context = project.context(int(unit_id), level=context_level, window=5)
        return (
            key_label,
            _render_readonly_context_units(
                html,
                context,
                focal_unit_id=int(unit_id),
                key_columns=list(metadata["key_columns"]),
            ),
            _metadata_table(html, focal["metadata"]),
        )

    @app.callback(
        Output("focal-unit-store", "data", allow_duplicate=True),
        Output("navigation-history-store", "data", allow_duplicate=True),
        Output("workspace-tabs", "value", allow_duplicate=True),
        Output("memo-center-tags", "value", allow_duplicate=True),
        Output("memo-navigation-location", "hash"),
        Input("memo-navigation-location", "hash"),
        State("session-dropdown", "value"),
        State("geometry-dropdown", "value"),
        State("focal-unit-store", "data"),
        State("navigation-history-store", "data"),
        prevent_initial_call=True,
    )
    def follow_memo_link(
        location_hash: str | None,
        session_id: int,
        geometry_id: int | None,
        focal_unit_id: int | None,
        history: dict[str, Any] | None,
    ) -> tuple[Any, Any, Any, Any, Any]:
        unit_match = re.fullmatch(r"#geco-unit-(\d+)", location_hash or "")
        if unit_match is not None:
            unit_id = int(unit_match.group(1))
            project.record_visit(
                int(session_id),
                unit_id,
                method="memo_reference",
                source_unit_id=int(focal_unit_id) if focal_unit_id is not None else None,
                geometry_id=int(geometry_id) if geometry_id is not None else None,
            )
            return (
                unit_id,
                _append_navigation_history(history, unit_id),
                "explore",
                no_update,
                "",
            )

        tag_match = re.fullmatch(r"#geco-tag-(.+)", location_hash or "")
        if tag_match is not None:
            return (
                no_update,
                no_update,
                "memos",
                [unquote(tag_match.group(1))],
                "",
            )
        return no_update, no_update, no_update, no_update, no_update

    @app.callback(
        Output("focus-construct-dropdown", "options"),
        Input("construct-refresh-store", "data"),
        Input("explore-code-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
    )
    def refresh_construct_options(*_refresh: Any) -> list[dict[str, Any]]:
        return _construct_options(project.constructs())

    @app.callback(
        Output("focus-code-dropdown", "value", allow_duplicate=True),
        Output("focus-teaching-select", "style"),
        Output("focus-scale-state", "style"),
        Output("focus-scale-heading", "children"),
        Output("focus-scale-description", "children"),
        Output("focus-scale-eligibility", "children"),
        Output("focus-code-summary", "style"),
        Output("focus-new-code-button", "style"),
        Output("focus-active-area", "style"),
        Input("focus-construct-dropdown", "value"),
        prevent_initial_call=True,
    )
    def route_focus_construct(construct_id: int | None) -> tuple[Any, ...]:
        try:
            construct = project.construct(int(construct_id)) if construct_id is not None else None
        except (KeyError, ValueError, TypeError):
            construct = None
        if construct is not None and construct["type"] == "code":
            return int(construct["construct_id"]), {}, {"display": "none"}, "", "", "", {}, {}, {}
        if construct is None:
            return None, {"display": "none"}, {"display": "none"}, "", "", "", {"display": "none"}, {}, {}
        scale = project.scale(int(construct["construct_id"]))
        eligibility = scale.get("eligibility", {"kind": "all"})
        return (
            None, {"display": "none"}, {}, f"Scale: {scale['name']}",
            scale["description"] or "No description has been written yet.",
            f"Definition version {scale['version_number']} · Eligibility: {json.dumps(eligibility, sort_keys=True)}",
            {"display": "none"}, {"display": "none"}, {"display": "none"},
        )

    @app.callback(
        Output("code-center-tags", "options"),
        Input("explore-code-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
    )
    def refresh_code_center_hashtags(
        explore_refresh: int | None, focus_refresh: int | None
    ) -> list[dict[str, str]]:
        del explore_refresh, focus_refresh
        return [{"label": f"#{tag}", "value": tag} for tag in project.code_hashtags()]

    @app.callback(
        Output("code-center-table", "data"),
        Output("code-center-table", "selected_rows"),
        Output("code-center-status", "children"),
        Input("code-center-query", "value"),
        Input("code-center-scope", "value"),
        Input("code-center-tags", "value"),
        Input("explore-code-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("construct-refresh-store", "data"),
        State("construct-center-selected-store", "data"),
    )
    def filter_code_center(
        query: str | None, scope: str | None, hashtags: list[str] | None,
        explore_refresh: int | None, focus_refresh: int | None,
        construct_refresh: int | None, selected_construct_id: int | None,
    ) -> tuple[list[dict[str, Any]], list[int], str]:
        del explore_refresh, focus_refresh, construct_refresh
        matches = project.search_codes(query=query or "", scope=str(scope or "full"), hashtags=hashtags or [])
        code_tags = {int(row["code_id"]): row.get("hashtags", []) for row in matches}
        code_ids = set(code_tags)
        term = (query or "").casefold().strip()
        rows = []
        for construct in project.constructs():
            kind = str(construct["type"])
            if kind == "code" and int(construct["construct_id"]) not in code_ids:
                continue
            if kind == "scale":
                if hashtags:
                    continue  # Hashtag filter retains its existing Code-specific semantics.
                searchable = str(construct["name"])
                if str(scope or "full") != "label":
                    searchable += " " + str(construct.get("description", ""))
                if term and term not in searchable.casefold():
                    continue
            rows.append({
                "construct_id": int(construct["construct_id"]),
                "code_id": (int(construct["construct_id"]) if kind == "code" else None),
                "type": "Scale" if kind == "scale" else "Code",
                "name": str(construct["name"]),
                "hashtags": " ".join(
                    f"#{tag}" for tag in code_tags.get(int(construct["construct_id"]), [])
                ),
            })
        selected_rows = [
            index for index, row in enumerate(rows)
            if selected_construct_id is not None
            and row["construct_id"] == int(selected_construct_id)
        ]
        return rows, selected_rows[:1], f"{len(rows):,} construct{'s' if len(rows) != 1 else ''} found"

    @app.callback(
        Output("construct-center-selected-store", "data"),
        Input("code-center-table", "selected_rows"),
        State("code-center-table", "data"),
    )
    def selected_center_construct(
        selected_rows: list[int] | None, rows: list[dict[str, Any]] | None
    ) -> int | None:
        if not selected_rows or not rows or int(selected_rows[0]) >= len(rows):
            return None
        return int(rows[int(selected_rows[0])]["construct_id"])

    @app.callback(
        Output("code-center-type-label", "children"),
        Output("code-center-eligibility", "children"),
        Output("code-center-edit-name-label", "children"),
        Output("code-center-binary-content", "style"),
        Input("construct-center-selected-store", "data"),
    )
    def show_center_construct_type(construct_id: int | None) -> tuple[Any, Any, str, dict[str, str]]:
        if construct_id is None:
            return "", "", "Construct name", {"display": "none"}
        construct = project.construct(int(construct_id))
        if construct["type"] == "code":
            return "Binary Code", "", "Code label", {}
        scale = project.scale(int(construct_id))
        return (
            "Continuous Scale",
            "Eligibility: " + json.dumps(scale.get("eligibility", {"kind": "all"}), sort_keys=True),
            "Scale name",
            {"display": "none"},
        )

    @app.callback(
        Output("code-center-selected-store", "data"),
        Output("code-center-version-store", "data"),
        Output("code-center-title", "children"),
        Output("code-center-description", "children"),
        Output("code-version-label", "children"),
        Output("code-version-previous", "disabled"),
        Output("code-version-next", "disabled"),
        Output("code-center-edit-button", "disabled"),
        Output("code-center-edit-name", "value"),
        Output("code-center-edit-description", "value"),
        Input("code-center-table", "selected_rows"),
        State("code-center-table", "data"),
        Input("explore-code-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
    )
    def select_code_center(
        selected_rows: list[int] | None, rows: list[dict[str, Any]] | None,
        explore_refresh: int | None, focus_refresh: int | None,
    ) -> tuple[Any, ...]:
        del explore_refresh, focus_refresh
        if not selected_rows or not rows:
            empty = html.P("Choose a Construct to inspect its definition.", className="memo-preview-empty")
            return None, None, "Select a Construct", empty, "No version selected", True, True, True, "", ""
        index = int(selected_rows[0])
        if index < 0 or index >= len(rows):
            return None, None, "Select a Construct", "", "No version selected", True, True, True, "", ""
        construct_id = int(rows[index]["construct_id"])
        construct = project.construct(construct_id)
        is_code = construct["type"] == "code"
        versions = (
            project.code_versions(construct_id) if is_code else project.scale_versions(construct_id)
        )
        number = int(construct["version_number"])
        description = str(construct["description"])
        return (
            (construct_id if is_code else None), number, str(construct["name"]),
            (_render_memo_preview(dcc, html, project, description) if description.strip() else None),
            f"Version {number} of {len(versions)}", number <= 1, number >= len(versions), False,
            str(construct["name"]), description,
        )

    @app.callback(
        Output("code-center-version-store", "data", allow_duplicate=True),
        Output("code-center-title", "children", allow_duplicate=True),
        Output("code-center-description", "children", allow_duplicate=True),
        Output("code-version-label", "children", allow_duplicate=True),
        Output("code-version-previous", "disabled", allow_duplicate=True),
        Output("code-version-next", "disabled", allow_duplicate=True),
        Output("code-center-edit-name", "value", allow_duplicate=True),
        Output("code-center-edit-description", "value", allow_duplicate=True),
        Input("code-version-previous", "n_clicks"), Input("code-version-next", "n_clicks"),
        State("construct-center-selected-store", "data"), State("code-center-version-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_code_versions(
        previous: int, next_clicks: int, code_id: int | None, version_number: int | None
    ) -> tuple[Any, ...]:
        del previous, next_clicks
        if code_id is None:
            return (no_update,) * 8
        construct = project.construct(int(code_id))
        versions = (
            project.code_versions(int(code_id))
            if construct["type"] == "code" else project.scale_versions(int(code_id))
        )
        current = int(version_number or len(versions))
        target = max(1, current - 1) if callback_context.triggered_id == "code-version-previous" else min(len(versions), current + 1)
        revision = project.construct(int(code_id), target)
        description = str(revision["description"])
        return (
            target, str(revision["name"]),
            (_render_memo_preview(dcc, html, project, description) if description.strip() else None),
            f"Version {target} of {len(versions)}", target <= 1, target >= len(versions),
            str(revision["name"]), description,
        )

    @app.callback(
        Output("code-center-edit-store", "data"),
        Output("code-center-editor-wrap", "style"),
        Output("code-center-description", "style", allow_duplicate=True),
        Input("code-center-edit-button", "n_clicks"), Input("code-center-cancel-button", "n_clicks"),
        State("construct-center-selected-store", "data"), prevent_initial_call=True,
    )
    def toggle_code_center_edit(
        edit_clicks: int, cancel_clicks: int, code_id: int | None
    ) -> tuple[bool, dict[str, str], dict[str, str]]:
        del edit_clicks, cancel_clicks
        editing = callback_context.triggered_id == "code-center-edit-button" and code_id is not None
        return (
            editing,
            ({"display": "block"} if editing else {"display": "none"}),
            ({"display": "none"} if editing else {}),
        )

    @app.callback(Output("code-center-edit-preview", "children"), Input("code-center-edit-description", "value"))
    def preview_code_center_edit(description: str | None) -> Any:
        return _render_memo_preview(dcc, html, project, description or "")

    @app.callback(
        Output("explore-code-refresh-store", "data", allow_duplicate=True),
        Output("focus-refresh-store", "data", allow_duplicate=True),
        Output("code-center-version-store", "data", allow_duplicate=True),
        Output("code-center-title", "children", allow_duplicate=True),
        Output("code-center-description", "children", allow_duplicate=True),
        Output("code-version-label", "children", allow_duplicate=True),
        Output("code-version-previous", "disabled", allow_duplicate=True),
        Output("code-version-next", "disabled", allow_duplicate=True),
        Output("code-center-edit-store", "data", allow_duplicate=True),
        Output("code-center-editor-wrap", "style", allow_duplicate=True),
        Output("code-center-description", "style", allow_duplicate=True),
        Output("code-center-edit-status", "children"),
        Input("code-center-save-button", "n_clicks"),
        State("construct-center-selected-store", "data"), State("code-center-edit-name", "value"),
        State("code-center-edit-description", "value"), State("explore-code-refresh-store", "data"),
        State("focus-refresh-store", "data"), prevent_initial_call=True,
    )
    def save_code_center_version(
        clicks: int, code_id: int | None, name: str | None, description: str | None,
        explore_refresh: int | None, focus_refresh: int | None,
    ) -> tuple[Any, ...]:
        del clicks
        if code_id is None:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, True, {"display": "none"}, {}, "Select a Construct first."
        clean_name = (name or "").strip()
        if not clean_name:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, True, {"display": "block"}, {"display": "none"}, "Enter a Construct name."
        clean_description = description or ""
        kind = project.construct(int(code_id))["type"]
        if kind == "code":
            project.update_code(int(code_id), name=clean_name, description=clean_description)
            versions = project.code_versions(int(code_id))
        else:
            project.update_scale(int(code_id), clean_name, clean_description)
            versions = project.scale_versions(int(code_id))
        number = len(versions)
        return (
            int(explore_refresh or 0) + 1, int(focus_refresh or 0) + 1, number, clean_name,
            (_render_memo_preview(dcc, html, project, clean_description) if clean_description.strip() else None),
            f"Version {number} of {number}",
            number <= 1, True, False, {"display": "none"}, {}, f"Saved version {number}.",
        )

    @app.callback(
        Output("construct-refresh-store", "data", allow_duplicate=True),
        Output("code-center-create-scale-status", "children"),
        Output("code-center-new-scale-name", "value"),
        Output("code-center-new-scale-description", "value"),
        Input("code-center-create-scale", "n_clicks"),
        State("code-center-new-scale-name", "value"),
        State("code-center-new-scale-description", "value"),
        State("construct-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def create_center_scale(
        clicks: int | None, name: str | None, description: str | None, refresh: int | None
    ) -> tuple[Any, str, str, str]:
        if not clicks:
            return no_update, "", no_update, no_update
        if not (name or "").strip():
            return no_update, "Enter a Scale name.", no_update, no_update
        try:
            scale_id = project.create_scale((name or "").strip(), description or "")
        except ValueError as error:
            return no_update, str(error), no_update, no_update
        return int(refresh or 0) + 1, f"Created Scale {scale_id}.", "", ""

    @app.callback(
        Output("code-center-teaching-button", "disabled"),
        Output("focus-teaching-button", "disabled"),
        Input("code-center-selected-store", "data"),
        Input("focus-code-dropdown", "value"),
    )
    def enable_teaching_example_manager(
        code_center_id: int | None, focus_code_id: int | None
    ) -> tuple[bool, bool]:
        # Authoring a teaching example never requires an eager transform.
        can_create = True
        return code_center_id is None or not can_create, focus_code_id is None or not can_create

    @app.callback(
        Output("focus-teaching-counts", "children"),
        Input("focus-code-dropdown", "value"),
        Input("teaching-example-refresh-store", "data"),
    )
    def update_focus_teaching_counts(
        code_id: int | None, refresh: int | None
    ) -> Any:
        del refresh
        if code_id is None:
            return "Select a code to manage examples."
        examples = project.teaching_examples(int(code_id))
        active = [
            example
            for example in examples
            if str(example["teaching_status"]) == "active"
        ]
        present = sum(
            str(example["teaching_label"]) == "positive" for example in active
        )
        absent = sum(
            str(example["teaching_label"]) == "negative" for example in active
        )
        inactive = sum(
            str(example["teaching_status"]) == "inactive" for example in examples
        )
        children: list[Any] = [html.Div(f"Present {present:,} · Absent {absent:,}")]
        if inactive:
            children.append(html.Small(f"{inactive:,} inactive example{'s' if inactive != 1 else ''}"))
        return children

    @app.callback(
        Output("teaching-example-modal", "style"),
        Output("teaching-example-status", "children"),
        Output("teaching-example-refresh-store", "data", allow_duplicate=True),
        Output("teaching-example-text", "value"),
        Output("teaching-example-note", "value"),
        Output("teaching-example-code-store", "data"),
        Output("focus-auto-training-request-store", "data", allow_duplicate=True),
        Input("code-center-teaching-button", "n_clicks"),
        Input("focus-teaching-button", "n_clicks"),
        Input("teaching-example-close", "n_clicks"),
        Input("teaching-example-save", "n_clicks"),
        State("code-center-selected-store", "data"),
        State("focus-code-dropdown", "value"),
        State("teaching-example-code-store", "data"),
        State("teaching-example-text", "value"),
        State("teaching-example-label", "value"),
        State("teaching-example-note", "value"),
        State("teaching-example-assess-before-training", "value"),
        State("teaching-example-refresh-store", "data"),
        State("focus-code-dropdown", "value"),
        State("focus-auto-retrain", "value"),
        State("focus-train-scope", "value"),
        State("focus-tune-hyperparameters", "value"),
        State("focus-recommendation-source", "value"),
        State("focus-active-classifier-dropdown", "value"),
        State("focus-committee-dropdown", "value"),
        prevent_initial_call=True,
    )
    def manage_teaching_example_modal(
        code_center_open_clicks: int,
        focus_open_clicks: int,
        close_clicks: int,
        save_clicks: int,
        code_center_id: int | None,
        focus_code_id: int | None,
        modal_code_id: int | None,
        text: str | None,
        label: str | None,
        note: str | None,
        assess_values: list[str] | None,
        refresh: int | None,
        develop_code_id: int | None,
        auto_train_values: list[str] | None,
        train_scope_values: list[str] | None,
        tune_values: list[str] | None,
        recommendation_source: str | None,
        active_classifier_spec_id: int | None,
        committee_id: int | None,
    ) -> tuple[dict[str, str], str, int, Any, Any, Any, Any]:
        del code_center_open_clicks, focus_open_clicks, close_clicks, save_clicks
        triggered = callback_context.triggered_id
        if triggered == "teaching-example-close":
            return (
                {"display": "none"}, "", int(refresh or 0),
                no_update, no_update, modal_code_id, no_update,
            )
        if triggered == "code-center-teaching-button":
            selected_code_id = code_center_id
        elif triggered == "focus-teaching-button":
            selected_code_id = focus_code_id
        else:
            selected_code_id = modal_code_id
        if selected_code_id is None:
            return (
                {"display": "none"}, "Select a code first.", int(refresh or 0),
                no_update, no_update, no_update, no_update,
            )
        if triggered in {"code-center-teaching-button", "focus-teaching-button"}:
            return (
                {"display": "flex"}, "", int(refresh or 0),
                no_update, no_update, int(selected_code_id), no_update,
            )
        clean_text = (text or "").strip()
        if not clean_text:
            return (
                {"display": "flex"}, "Enter example text.", int(refresh or 0),
                no_update, no_update, int(selected_code_id), no_update,
            )
        try:
            observation_id = project.create_teaching_example(
                code_id=int(selected_code_id), text=clean_text,
                label=str(label or "positive"), note=(note or "").strip(),
                assess_before_training="assess" in set(assess_values or []),
            )
        except Exception as error:
            return (
                {"display": "flex"}, str(error), int(refresh or 0),
                no_update, no_update, int(selected_code_id), no_update,
            )
        request, note = _post_saved_teaching_example_training_request(
            project, code_id=int(selected_code_id), observation_id=int(observation_id),
            develop_code_id=develop_code_id, auto_train_values=auto_train_values,
            train_scope_values=train_scope_values, tune_values=tune_values,
            recommendation_source=recommendation_source,
            active_classifier_spec_id=active_classifier_spec_id, committee_id=committee_id,
        )
        return (
            {"display": "flex"}, f"Added an active teaching example. {note}",
            int(refresh or 0) + 1, "", "", int(selected_code_id),
            training_inbox.enqueue(request) if request is not None else no_update,
        )

    @app.callback(
        Output("teaching-example-table", "data"),
        Output("teaching-example-table", "selected_rows"),
        Input("teaching-example-code-store", "data"),
        Input("teaching-example-refresh-store", "data"),
    )
    def load_teaching_examples(
        code_id: int | None, refresh: int | None
    ) -> tuple[list[dict[str, Any]], list[int]]:
        del refresh
        if code_id is None:
            return [], []
        rows = [
            {
                "observation_id": int(example["observation_id"]),
                "text": str(example["text"]),
                "label": "Present" if example["teaching_label"] == "positive" else "Absent",
                "status": str(example["teaching_status"]).title(),
                "created": str(example["created_at"]),
            }
            for example in project.teaching_examples(int(code_id))
        ]
        return rows, []

    @app.callback(
        Output("teaching-example-status", "children", allow_duplicate=True),
        Output("teaching-example-refresh-store", "data", allow_duplicate=True),
        Output("focus-auto-training-request-store", "data", allow_duplicate=True),
        Input("teaching-example-apply-status", "n_clicks"),
        State("teaching-example-table", "selected_rows"),
        State("teaching-example-table", "data"),
        State("teaching-example-status-action", "value"),
        State("teaching-example-refresh-store", "data"),
        State("teaching-example-code-store", "data"),
        State("focus-code-dropdown", "value"),
        State("focus-auto-retrain", "value"),
        State("focus-train-scope", "value"),
        State("focus-tune-hyperparameters", "value"),
        State("focus-recommendation-source", "value"),
        State("focus-active-classifier-dropdown", "value"),
        State("focus-committee-dropdown", "value"),
        prevent_initial_call=True,
    )
    def change_teaching_example_status(
        clicks: int, selected_rows: list[int] | None, rows: list[dict[str, Any]] | None,
        status: str | None, refresh: int | None,
        selected_code_id: int | None,
        develop_code_id: int | None,
        auto_train_values: list[str] | None,
        train_scope_values: list[str] | None,
        tune_values: list[str] | None,
        recommendation_source: str | None,
        active_classifier_spec_id: int | None,
        committee_id: int | None,
    ) -> tuple[str, int, Any]:
        del clicks
        if selected_code_id is None:
            return "Select a teaching-example code first.", int(refresh or 0), no_update
        if not selected_rows or not rows:
            return "Select an example first.", int(refresh or 0), no_update
        index = int(selected_rows[0])
        if index < 0 or index >= len(rows):
            return "Select an example first.", int(refresh or 0), no_update
        observation_id = int(rows[index]["observation_id"])
        existing = _teaching_example_for_code(
            project, code_id=int(selected_code_id), observation_id=observation_id
        )
        if existing is None:
            return (
                "This teaching example is no longer in the selected code. Refresh the list.",
                int(refresh or 0), no_update,
            )
        selected_status = str(status or "inactive")
        if str(existing["teaching_status"]) == selected_status:
            return "Teaching example status is already current.", int(refresh or 0), no_update
        project.set_teaching_example_status(observation_id, selected_status)
        action = "Deleted" if selected_status == "deleted" else selected_status.title()
        request, note = _post_saved_teaching_example_training_request(
            project, code_id=int(selected_code_id), observation_id=observation_id,
            develop_code_id=develop_code_id, auto_train_values=auto_train_values,
            train_scope_values=train_scope_values, tune_values=tune_values,
            recommendation_source=recommendation_source,
            active_classifier_spec_id=active_classifier_spec_id, committee_id=committee_id,
        )
        return f"{action} the selected teaching example. {note}", int(refresh or 0) + 1, (training_inbox.enqueue(request) if request is not None else no_update)

    @app.callback(
        Output("teaching-example-status", "children", allow_duplicate=True),
        Output("teaching-example-refresh-store", "data", allow_duplicate=True),
        Output("focus-auto-training-request-store", "data", allow_duplicate=True),
        Input("teaching-example-apply-label", "n_clicks"),
        State("teaching-example-table", "selected_rows"),
        State("teaching-example-table", "data"),
        State("teaching-example-label-action", "value"),
        State("teaching-example-refresh-store", "data"),
        State("teaching-example-code-store", "data"),
        State("focus-code-dropdown", "value"),
        State("focus-auto-retrain", "value"),
        State("focus-train-scope", "value"),
        State("focus-tune-hyperparameters", "value"),
        State("focus-recommendation-source", "value"),
        State("focus-active-classifier-dropdown", "value"),
        State("focus-committee-dropdown", "value"),
        prevent_initial_call=True,
    )
    def change_teaching_example_label(
        clicks: int, selected_rows: list[int] | None, rows: list[dict[str, Any]] | None,
        label: str | None, refresh: int | None,
        selected_code_id: int | None,
        develop_code_id: int | None,
        auto_train_values: list[str] | None,
        train_scope_values: list[str] | None,
        tune_values: list[str] | None,
        recommendation_source: str | None,
        active_classifier_spec_id: int | None,
        committee_id: int | None,
    ) -> tuple[str, int, Any]:
        del clicks
        if selected_code_id is None:
            return "Select a teaching-example code first.", int(refresh or 0), no_update
        if not selected_rows or not rows:
            return "Select an example first.", int(refresh or 0), no_update
        index = int(selected_rows[0])
        if index < 0 or index >= len(rows):
            return "Select an example first.", int(refresh or 0), no_update
        observation_id = int(rows[index]["observation_id"])
        existing = _teaching_example_for_code(
            project, code_id=int(selected_code_id), observation_id=observation_id
        )
        if existing is None:
            return (
                "This teaching example is no longer in the selected code. Refresh the list.",
                int(refresh or 0), no_update,
            )
        selected_label = str(label or "positive")
        if str(existing["teaching_label"]) == selected_label:
            return "Teaching example label is already current.", int(refresh or 0), no_update
        project.set_teaching_example_label(observation_id, selected_label)
        display = "Present" if selected_label == "positive" else "Absent"
        request, note = _post_saved_teaching_example_training_request(
            project, code_id=int(selected_code_id), observation_id=int(observation_id),
            develop_code_id=develop_code_id, auto_train_values=auto_train_values,
            train_scope_values=train_scope_values, tune_values=tune_values,
            recommendation_source=recommendation_source,
            active_classifier_spec_id=active_classifier_spec_id, committee_id=committee_id,
        )
        return f"Changed the selected teaching example to {display}. {note}", int(refresh or 0) + 1, (training_inbox.enqueue(request) if request is not None else no_update)

    @app.callback(
        Output("code-center-units-table", "data"),
        Input("code-center-selected-store", "data"),
        Input("workspace-tabs", "value"),
        Input("code-center-view-tabs", "value"),
        Input("explore-annotation-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
    )
    def load_code_center_units(
        code_id: int | None,
        workspace: str | None,
        content_tab: str | None,
        explore_refresh: int | None,
        focus_refresh: int | None,
        teaching_refresh: int | None,
    ) -> Any:
        del explore_refresh, focus_refresh, teaching_refresh
        if workspace != "codes" or content_tab != "list":
            return no_update
        if code_id is None:
            return []
        return [
            {
                "observation_id": int(observation["observation_id"]),
                "kind": str(observation["kind"]).replace("_", " ").title(),
                "key": _observation_display_key(project, observation),
                "text": str(observation["text"]),
            }
            for observation in project.positive_observations_for_code(int(code_id))
        ]

    @app.callback(
        Output("code-center-view", "options"), Output("code-center-view", "value"),
        Input("code-center-geometry", "value"),
        Input("representation-refresh-store", "data"),
        Input("view-refresh-store", "data"),
        State("code-center-view", "value"),
    )
    def update_code_center_views(
        geometry_id: int | None, _revision: int | None,
        _view_revision: int | None, current_view_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        if geometry_id is None:
            return [], None
        records = _display_view_records(project, int(geometry_id))
        options = [{"label": row["name"], "value": int(row["view_id"])} for row in records]
        valid_ids = {int(row["view_id"]) for row in records}
        selected = (
            int(current_view_id)
            if current_view_id is not None and int(current_view_id) in valid_ids
            else (int(records[0]["view_id"]) if records else None)
        )
        return options, selected

    @app.callback(
        Output("code-center-map", "figure"),
        Output("code-center-page-status", "children"),
        Input("code-center-selected-store", "data"),
        Input("code-center-geometry", "value"),
        Input("code-center-view", "value"),
        Input("code-center-page-size", "value"),
        Input("code-center-page-number", "value"),
        Input("code-center-include-positives", "value"),
        Input("workspace-tabs", "value"),
        Input("code-center-view-tabs", "value"),
        Input("explore-annotation-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
        State("code-center-focal-store", "data"),
        State("code-center-map", "relayoutData"),
    )
    def update_code_center_map(
        code_id: int | None,
        geometry_id: int | None,
        view_id: int | None,
        page_size: Any,
        page_number: Any,
        include_positive_values: list[str] | None,
        workspace: str | None,
        content_tab: str | None,
        explore_refresh: int | None,
        focus_refresh: int | None,
        teaching_refresh: int | None,
        focal_unit_id: int | None,
        relayout_data: dict[str, Any] | None,
    ) -> tuple[Any, str]:
        del explore_refresh, focus_refresh, teaching_refresh
        if workspace != "codes" or content_tab != "geometry":
            return no_update, no_update
        if geometry_id is None or view_id is None:
            return {}, ""
        positive_observations = (
            project.positive_observations_for_code(int(code_id))
            if code_id is not None else []
        )
        positive = {
            int(observation["unit_id"])
            for observation in positive_observations
            if observation["kind"] == "atomic" and observation["unit_id"] is not None
        }
        positive_positions = set(project.database.unit_positions_for_ids(positive).values())
        size = _coerce_positive_int(page_size, default=5000)
        number = _coerce_positive_int(page_number, default=1)
        include_all_positives = "include" in set(include_positive_values or [])
        page = deterministic_page_with_inclusions(
            len(units),
            page_size=size,
            page_index=number - 1,
            seed=0,
            include_positions=(positive_positions if include_all_positives else ()),
        )
        positions = page.positions
        page_units = units.at_positions(positions.tolist())
        page_eligible = [int(unit["unit_id"]) in positive for unit in page_units]
        geometry_record = geometry_by_id[int(geometry_id)]
        view_record = _display_view_record(
            project, geometry_id=int(geometry_id), view_id=int(view_id)
        )
        if view_record is None:
            return _empty_figure("This view is no longer available."), ""
        figure = build_geometry_figure(
            coordinates=project.view_coordinate_rows(int(view_id), positions),
            units=page_units,
            eligible=page_eligible,
            focal_unit_id=None,
            geometry_name=str(geometry_record["name"]),
            view_name=str(view_record["name"]),
            page_label=f"Page {page.page_index + 1} of {page.page_count}",
            uirevision=(
                f"code-center:{geometry_id}:{view_id}:{code_id}:"
                f"{page.page_index}:{int(include_all_positives)}"
            ),
        )
        for trace in figure.data:
            if trace.name == "Eligible":
                trace.name = "Positive atomic observations"
                trace.showlegend = True
            elif trace.name == "Filtered":
                trace.name = "Other atomic observations"
                trace.showlegend = True
        derived_positive = [
            observation for observation in positive_observations
            if observation["kind"] != "atomic"
        ]
        omitted_derived = 0
        for kind, symbol, label in (
            ("span", "triangle-up", "Positive spans"),
            ("teaching_example", "star", "Positive teaching examples"),
        ):
            observations_of_kind = [
                observation for observation in derived_positive
                if observation["kind"] == kind
            ]
            coordinates: list[list[float]] = []
            observation_ids: list[int] = []
            hover: list[str] = []
            for observation in observations_of_kind:
                try:
                    point = project.observation_view_coordinates(
                        int(observation["observation_id"]), int(view_id)
                    )[0]
                except (KeyError, ValueError):
                    omitted_derived += 1
                    continue
                coordinates.append([float(point[0]), float(point[1])])
                observation_ids.append(int(observation["observation_id"]))
                hover.append(_observation_display_key(project, observation))
            if coordinates:
                array = np.asarray(coordinates, dtype=float)
                marker_line = ["#111111" for _value in observation_ids]
                marker_width = [0.8 for _value in observation_ids]
                figure.add_trace(
                    go.Scattergl(
                        x=array[:, 0].tolist(), y=array[:, 1].tolist(),
                        mode="markers", name=label, customdata=observation_ids,
                        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
                        marker={
                            "size": 10,
                            "symbol": symbol,
                            "color": "#111111",
                            "opacity": 0.96,
                            "line": {"color": marker_line, "width": marker_width},
                        },
                    )
                )

        focal_x: list[float] = []
        focal_y: list[float] = []
        focal_customdata: list[int] = []
        focal_hover: list[str] = []
        if focal_unit_id is not None:
            try:
                focal_observation = project.observation(int(focal_unit_id))
                focal_point = project.observation_view_coordinates(
                    int(focal_unit_id), int(view_id)
                )[0]
            except (KeyError, ValueError):
                pass
            else:
                focal_x = [float(focal_point[0])]
                focal_y = [float(focal_point[1])]
                focal_customdata = [int(focal_unit_id)]
                focal_hover = [_observation_display_key(project, focal_observation)]
        figure.add_trace(
            go.Scattergl(
                x=focal_x,
                y=focal_y,
                mode="markers",
                name="Focal observation",
                customdata=focal_customdata,
                hovertext=focal_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                marker={
                    "size": 13,
                    "color": "rgba(0, 0, 0, 0)",
                    "line": {"color": "#178547", "width": 2.6},
                },
                showlegend=False,
            )
        )

        figure.update_layout(
            showlegend=True,
            legend={"orientation": "h", "y": -0.08},
        )
        # Keep the lightweight focal ring at a stable trace index so selecting
        # a point can patch only this trace instead of rebuilding the map.
        figure.data = (figure.data[-1], *figure.data[:-1])

        if callback_context.triggered_id not in {
            "code-center-selected-store",
            "code-center-geometry",
            "code-center-view",
            "code-center-page-size",
            "code-center-page-number",
            "code-center-include-positives",
        }:
            _apply_relayout_ranges(figure, relayout_data)
        extras = max(0, len(positions) - min(size, len(units) - page.page_index * size))
        status = (
            f"Page {page.page_index + 1} of {page.page_count} · "
            f"{len(positions):,} points shown"
        )
        if include_all_positives:
            status += f" · all {len(positive):,} positive units included"
            if extras:
                status += f" ({extras:,} added beyond the page sample)"
        if derived_positive:
            shown_derived = len(derived_positive) - omitted_derived
            status += f" · {shown_derived:,} derived positive observations shown"
            if omitted_derived:
                status += f" · {omitted_derived:,} unavailable in this view"
        return figure, status

    @app.callback(
        Output("code-center-map", "figure", allow_duplicate=True),
        Input("code-center-focal-store", "data"),
        State("code-center-view", "value"),
        State("code-center-page-status", "children"),
        prevent_initial_call=True,
    )
    def update_code_center_focal_marker(
        observation_id: int | None,
        view_id: int | None,
        page_status: str | None,
    ) -> Any:
        # Page status and the map are produced in the same callback.  The map
        # has a stable focal trace at data[0] whenever it is valid, so a short
        # readiness status replaces sending the entire Plotly figure as State.
        if view_id is None or not page_status:
            return no_update
        patch = Patch()
        if observation_id is None:
            patch["data"][0]["x"] = []
            patch["data"][0]["y"] = []
            patch["data"][0]["customdata"] = []
            patch["data"][0]["hovertext"] = []
            return patch
        try:
            observation = project.observation(int(observation_id))
            point = project.observation_view_coordinates(
                int(observation_id), int(view_id)
            )[0]
        except (KeyError, ValueError):
            return no_update
        patch["data"][0]["x"] = [float(point[0])]
        patch["data"][0]["y"] = [float(point[1])]
        patch["data"][0]["customdata"] = [int(observation_id)]
        patch["data"][0]["hovertext"] = [
            _observation_display_key(project, observation)
        ]
        return patch

    @app.callback(
        Output("code-center-page-number", "max"),
        Output("code-center-page-number", "value", allow_duplicate=True),
        Input("code-center-page-previous", "n_clicks"),
        Input("code-center-page-next", "n_clicks"),
        Input("code-center-page-size", "value"),
        State("code-center-page-number", "value"),
        prevent_initial_call=True,
    )
    def change_code_center_page(
        previous_clicks: int,
        next_clicks: int,
        page_size: Any,
        page_number: Any,
    ) -> tuple[int, int]:
        del previous_clicks, next_clicks
        size = _coerce_positive_int(page_size, default=5000)
        page_count = deterministic_page_count(len(units), page_size=size)
        current = min(max(1, _coerce_positive_int(page_number, default=1)), page_count)
        triggered = callback_context.triggered_id
        if triggered == "code-center-page-size":
            current = 1
        elif triggered == "code-center-page-previous":
            current = max(1, current - 1)
        elif triggered == "code-center-page-next":
            current = min(page_count, current + 1)
        return page_count, current

    @app.callback(
        Output("code-center-focal-store", "data"),
        Input("code-center-map", "clickData"),
        Input("code-center-units-table", "active_cell"),
        State("code-center-units-table", "data"),
        prevent_initial_call=True,
    )
    def select_code_center_unit(
        click_data: dict[str, Any] | None,
        active_cell: dict[str, Any] | None,
        rows: list[dict[str, Any]] | None,
    ) -> Any:
        if callback_context.triggered_id == "code-center-map":
            clicked = _clicked_unit_id(click_data)
            if clicked is None:
                return no_update
            return project.observation_id_for_unit(int(clicked))
        if active_cell and rows:
            row_index = int(active_cell.get("row", -1))
            if 0 <= row_index < len(rows):
                row = rows[row_index]
                return int(row.get("observation_id", row.get("unit_id")))
        return no_update

    @app.callback(Output("code-center-unit-preview", "children"), Input("code-center-focal-store", "data"))
    def preview_code_center_unit(observation_id: int | None) -> Any:
        if observation_id is None:
            return html.P(
                "Click a positive observation or a point to inspect it.",
                className="memo-preview-empty",
            )
        observation = project.observation(int(observation_id))
        return [
            html.Strong(_observation_display_key(project, observation)),
            html.Small(str(observation["kind"]).replace("_", " ").title()),
            html.P(str(observation["text"])),
        ]

    @app.callback(
        Output("code-center-assignment-positive", "disabled"),
        Output("code-center-assignment-negative", "disabled"),
        Output("code-center-assignment-unsure", "disabled"),
        Output("code-center-assignment-positive", "className"),
        Output("code-center-assignment-negative", "className"),
        Output("code-center-assignment-unsure", "className"),
        Input("code-center-selected-store", "data"),
        Input("code-center-focal-store", "data"),
        Input("workspace-tabs", "value"),
        Input("explore-annotation-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
    )
    def style_code_center_assignment_buttons(
        code_id: int | None,
        observation_id: int | None,
        workspace: str | None,
        explore_refresh: int | None,
        focus_refresh: int | None,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        del explore_refresh, focus_refresh
        if workspace != "codes":
            return (no_update,) * 6
        current = None
        disabled = code_id is None or observation_id is None
        if not disabled:
            observation = project.observation(int(observation_id))
            # Teaching examples have their own label/status manager. Revising
            # them here would desynchronize the teaching-example metadata.
            disabled = str(observation["kind"]) == "teaching_example"
            if not disabled:
                current = _current_annotation_value(
                    project, int(observation_id), int(code_id)
                )
        classes = _judgment_button_classes(
            current, prefix="code-center-assignment"
        )
        return disabled, disabled, disabled, *classes

    @app.callback(
        Output("code-center-assignment-status", "children"),
        Output("explore-annotation-refresh-store", "data", allow_duplicate=True),
        Output("focus-refresh-store", "data", allow_duplicate=True),
        Output("focus-auto-training-request-store", "data", allow_duplicate=True),
        Input("code-center-assignment-positive", "n_clicks"),
        Input("code-center-assignment-negative", "n_clicks"),
        Input("code-center-assignment-unsure", "n_clicks"),
        State("code-center-selected-store", "data"),
        State("code-center-focal-store", "data"),
        State("explore-annotation-refresh-store", "data"),
        State("focus-refresh-store", "data"),
        State("focus-code-dropdown", "value"),
        State("focus-auto-retrain", "value"),
        State("focus-train-scope", "value"),
        State("focus-tune-hyperparameters", "value"),
        State("focus-recommendation-source", "value"),
        State("focus-active-classifier-dropdown", "value"),
        State("focus-committee-dropdown", "value"),
        prevent_initial_call=True,
    )
    def revise_code_center_assignment(
        positive_clicks: int,
        negative_clicks: int,
        unsure_clicks: int,
        code_id: int | None,
        observation_id: int | None,
        explore_refresh: int | None,
        focus_refresh: int | None,
        develop_code_id: int | None,
        auto_train_values: list[str] | None,
        train_scope_values: list[str] | None,
        tune_values: list[str] | None,
        recommendation_source: str | None,
        active_classifier_spec_id: int | None,
        committee_id: int | None,
    ) -> tuple[str, int, int, Any]:
        del positive_clicks, negative_clicks, unsure_clicks
        if code_id is None or observation_id is None:
            return (
                "Select a code and observation first.",
                int(explore_refresh or 0),
                int(focus_refresh or 0),
                no_update,
            )
        observation = project.observation(int(observation_id))
        if str(observation["kind"]) == "teaching_example":
            return (
                "Revise teaching examples through the teaching-example manager.",
                int(explore_refresh or 0),
                int(focus_refresh or 0),
                no_update,
            )
        value_by_button = {
            "code-center-assignment-positive": "positive",
            "code-center-assignment-negative": "negative",
            "code-center-assignment-unsure": "unsure",
        }
        value = value_by_button[str(callback_context.triggered_id)]
        current = _current_annotation_value(
            project, int(observation_id), int(code_id)
        )
        if current == value:
            return "", int(explore_refresh or 0), int(focus_refresh or 0), no_update
        event_id = project.annotate(
            int(observation_id),
            int(code_id),
            value,
            origin="human_code_center",
        )
        request, status = _post_saved_external_judgment_training_request(
            project, code_id=int(code_id), develop_code_id=develop_code_id,
            annotation_event_id=int(event_id),
            auto_train_values=auto_train_values,
            train_scope_values=train_scope_values,
            tune_values=tune_values,
            recommendation_source=recommendation_source,
            active_classifier_spec_id=active_classifier_spec_id,
            committee_id=committee_id,
        )
        if request is not None:
            request["origin"] = "code_center"
        return (
            f"Revised the assignment to {value}. {status}",
            int(explore_refresh or 0) + 1,
            int(focus_refresh or 0) + 1,
            training_inbox.enqueue(request) if request is not None else no_update,
        )

    @app.callback(
        Output("focus-code-dropdown", "options"),
        Input("explore-code-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
    )
    def refresh_code_options(
        explore_refresh: int | None, focus_refresh: int | None
    ) -> list[dict[str, Any]]:
        del explore_refresh, focus_refresh
        return _code_options(project.codes())

    @app.callback(
        Output("code-modal", "style"),
        Output("code-modal-target-store", "data"),
        Output("explore-code-refresh-store", "data", allow_duplicate=True),
        Output("focus-refresh-store", "data", allow_duplicate=True),
        Output("focus-code-dropdown", "options", allow_duplicate=True),
        Output("focus-code-dropdown", "value", allow_duplicate=True),
        Output("focus-construct-dropdown", "value", allow_duplicate=True),
        Output("explore-code-palette-store", "data", allow_duplicate=True),
        Output("explore-code-row-count-store", "data", allow_duplicate=True),
        Output("code-modal-name", "value"),
        Output("code-modal-definition", "value"),
        Output("code-modal-status", "children"),
        Input("explore-new-code-button", "n_clicks"),
        Input("focus-new-code-button", "n_clicks"),
        Input("code-center-new-code-button", "n_clicks"),
        Input("code-modal-cancel-button", "n_clicks"),
        Input("code-modal-cancel-footer", "n_clicks"),
        Input("code-modal-create-button", "n_clicks"),
        State("code-modal-target-store", "data"),
        State("code-modal-name", "value"),
        State("code-modal-definition", "value"),
        State("explore-code-refresh-store", "data"),
        State("focus-refresh-store", "data"),
        State("explore-code-palette-store", "data"),
        prevent_initial_call=True,
    )
    def manage_code_modal(
        explore_open: int,
        focus_open: int,
        center_open: int,
        cancel_header: int,
        cancel_footer: int,
        create_clicks: int,
        target: str | None,
        name: str | None,
        definition: str | None,
        explore_refresh: int | None,
        focus_refresh: int | None,
        explore_palette: list[int | None] | None,
    ) -> tuple[Any, ...]:
        del explore_open, focus_open, center_open, cancel_header, cancel_footer, create_clicks
        triggered = callback_context.triggered_id
        if triggered == "explore-new-code-button":
            return (
                {"display": "flex"}, "explore", no_update, no_update,
                no_update, no_update, no_update, no_update, no_update, "", "", "",
            )
        if triggered == "focus-new-code-button":
            return (
                {"display": "flex"}, "focus", no_update, no_update,
                no_update, no_update, no_update, no_update, no_update, "", "", "",
            )
        if triggered == "code-center-new-code-button":
            return (
                {"display": "flex"}, "center", no_update, no_update,
                no_update, no_update, no_update, no_update, no_update, "", "", "",
            )
        if triggered in {"code-modal-cancel-button", "code-modal-cancel-footer"}:
            return (
                {"display": "none"}, None, no_update, no_update,
                no_update, no_update, no_update, no_update, no_update, "", "", "",
            )

        clean_name = (name or "").strip()
        if not clean_name:
            return (
                {"display": "flex"}, target, no_update, no_update,
                no_update, no_update, no_update, no_update, no_update,
                name or "", definition or "", "Enter a code label.",
            )
        existing = {
            str(code["name"]).casefold(): int(code["code_id"])
            for code in project.codes()
        }
        if clean_name.casefold() in existing:
            code_id = existing[clean_name.casefold()]
        else:
            code_id = project.create_code(clean_name, (definition or "").strip())
        current_codes = project.codes()
        options = _code_options(current_codes)
        focus_value = code_id if target == "focus" else no_update
        palette_value: Any = no_update
        row_count_value: Any = no_update
        if target == "explore":
            palette_value = _place_code_in_explore_palette(
                explore_palette,
                code_id,
                {int(code["code_id"]) for code in current_codes},
            )
            row_count_value = len(palette_value)
        return (
            {"display": "none"},
            None,
            int(explore_refresh or 0) + 1,
            int(focus_refresh or 0) + 1,
            options,
            focus_value,
            focus_value,
            palette_value,
            row_count_value,
            "",
            "",
            "",
        )

    @app.callback(
        Output("explore-code-palette-store", "data", allow_duplicate=True),
        Output("explore-code-row-count-store", "data", allow_duplicate=True),
        Input("explore-add-code-row", "n_clicks"),
        Input({"type": "explore-code-remove", "index": ALL}, "n_clicks"),
        State("explore-code-palette-store", "data"),
        prevent_initial_call=True,
    )
    def change_explore_code_palette_rows(
        add_clicks: int | None,
        remove_clicks: list[int] | None,
        palette: list[int | None] | None,
    ) -> tuple[Any, Any]:
        del remove_clicks
        valid_ids = {int(code["code_id"]) for code in project.codes()}
        current = _normalize_explore_code_palette(palette, valid_ids)
        triggered = callback_context.triggered_id
        triggered_value = (
            callback_context.triggered[0].get("value")
            if callback_context.triggered
            else None
        )
        if triggered == "explore-add-code-row":
            if int(add_clicks or 0) < 1:
                return no_update, no_update
            updated = [*current, None]
            return updated, len(updated)
        if isinstance(triggered, dict) and triggered.get("type") == "explore-code-remove":
            # Pattern-matched buttons fire when Dash mounts them. A newly mounted
            # remove button has n_clicks=0 and is not a user request to remove a row.
            if int(triggered_value or 0) < 1:
                return no_update, no_update
            index = int(triggered.get("index", -1))
            if index <= 0 or index >= len(current):
                return no_update, no_update
            updated = [value for position, value in enumerate(current) if position != index]
            return updated, len(updated)
        return no_update, no_update

    @app.callback(
        Output("explore-code-palette-store", "data", allow_duplicate=True),
        Input({"type": "explore-code-dropdown", "index": ALL}, "value"),
        State({"type": "explore-code-dropdown", "index": ALL}, "id"),
        State("explore-code-palette-store", "data"),
        prevent_initial_call=True,
    )
    def select_explore_palette_code(
        values: list[int | None],
        ids: list[dict[str, Any]],
        palette: list[int | None] | None,
    ) -> Any:
        del values, ids
        triggered = callback_context.triggered_id
        if not isinstance(triggered, dict):
            return no_update
        row_index = int(triggered.get("index", -1))
        triggered_value = (
            callback_context.triggered[0].get("value")
            if callback_context.triggered
            else None
        )
        valid_ids = {int(code["code_id"]) for code in project.codes()}
        current = _normalize_explore_code_palette(palette, valid_ids)
        if not 0 <= row_index < len(current):
            return no_update
        new_value = int(triggered_value) if triggered_value is not None else None
        if new_value is not None:
            selected_elsewhere = {
                int(value)
                for index, value in enumerate(current)
                if index != row_index and value is not None
            }
            if new_value in selected_elsewhere:
                # Keep one code per palette row without rerendering the whole
                # palette. Reset only the dropdown the user just changed.
                set_props(triggered, {"value": current[row_index]})
                return no_update
        if current[row_index] == new_value:
            return no_update
        updated = list(current)
        updated[row_index] = new_value
        return updated

    @app.callback(
        Output("explore-code-palette", "children"),
        Input("explore-code-row-count-store", "data"),
        Input("explore-code-refresh-store", "data"),
        State("explore-code-palette-store", "data"),
    )
    def render_explore_code_palette(
        row_count: int | None,
        code_refresh: int | None,
        palette: list[int | None] | None,
    ) -> list[Any]:
        del code_refresh
        current_codes = project.codes()
        valid_ids = {int(code["code_id"]) for code in current_codes}
        normalized = _explore_palette_for_row_count(palette, valid_ids, row_count)
        selected_codes = {int(value) for value in normalized if value is not None}
        rows: list[Any] = []
        for index, code_id in enumerate(normalized):
            options = [
                {"label": code["name"], "value": int(code["code_id"])}
                for code in current_codes
                if int(code["code_id"]) == code_id
                or int(code["code_id"]) not in selected_codes
            ]
            selector_children: list[Any] = [
                dcc.Dropdown(
                    id={"type": "explore-code-dropdown", "index": index},
                    options=options,
                    value=code_id,
                    placeholder="Select a code",
                    clearable=True,
                )
            ]
            if index > 0:
                selector_children.append(
                    html.Button(
                        "×",
                        id={"type": "explore-code-remove", "index": index},
                        n_clicks=0,
                        className="explore-code-remove",
                        title="Remove this coding row",
                        **{"aria-label": "Remove this coding row"},
                    )
                )
            rows.append(
                html.Div(
                    [
                        html.Div(selector_children, className="explore-code-selector-row"),
                        html.Div(
                            [
                                html.Button(
                                    "Present",
                                    id={"type": "explore-code-positive", "index": index},
                                    n_clicks=0,
                                    className="judgment-button positive-button",
                                    disabled=True,
                                ),
                                html.Button(
                                    "Absent",
                                    id={"type": "explore-code-negative", "index": index},
                                    n_clicks=0,
                                    className="judgment-button negative-button",
                                    disabled=True,
                                ),
                                html.Button(
                                    "Unsure",
                                    id={"type": "explore-code-unsure", "index": index},
                                    n_clicks=0,
                                    className="judgment-button unsure-button",
                                    disabled=True,
                                ),
                            ],
                            className="explore-code-buttons",
                        ),
                    ],
                    className="explore-code-entry",
                )
            )
        return rows

    @app.callback(
        Output({"type": "explore-code-dropdown", "index": ALL}, "options"),
        Input("explore-code-palette-store", "data"),
        Input("explore-code-refresh-store", "data"),
        Input({"type": "explore-code-dropdown", "index": ALL}, "id"),
    )
    def sync_explore_palette_dropdown_options(
        palette: list[int | None] | None,
        code_refresh: int | None,
        ids: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        del code_refresh
        current_codes = project.codes()
        valid_ids = {int(code["code_id"]) for code in current_codes}
        normalized = _normalize_explore_code_palette(palette, valid_ids)
        selected = {int(value) for value in normalized if value is not None}
        options_by_row: list[list[dict[str, Any]]] = []
        for component_id in ids or []:
            index = int(component_id.get("index", -1))
            value = normalized[index] if 0 <= index < len(normalized) else None
            options_by_row.append(
                [
                    {"label": code["name"], "value": int(code["code_id"])}
                    for code in current_codes
                    if int(code["code_id"]) == value
                    or int(code["code_id"]) not in selected
                ]
            )
        return options_by_row

    @app.callback(
        Output({"type": "explore-code-positive", "index": ALL}, "disabled"),
        Output({"type": "explore-code-negative", "index": ALL}, "disabled"),
        Output({"type": "explore-code-unsure", "index": ALL}, "disabled"),
        Output({"type": "explore-code-positive", "index": ALL}, "className"),
        Output({"type": "explore-code-negative", "index": ALL}, "className"),
        Output({"type": "explore-code-unsure", "index": ALL}, "className"),
        Input("explore-span-selection-store", "data"),
        Input("explore-annotation-refresh-store", "data"),
        Input("explore-code-palette-store", "data"),
        Input({"type": "explore-code-positive", "index": ALL}, "id"),
        State("explore-focal-unit-store", "data"),
    )
    def style_explore_palette_buttons(
        selected_unit_ids: list[int] | None,
        annotation_refresh: int | None,
        palette: list[int | None] | None,
        ids: list[dict[str, Any]],
        focal_unit_id: int | None,
    ) -> tuple[list[bool], list[bool], list[bool], list[str], list[str], list[str]]:
        del annotation_refresh
        valid_ids = {int(code["code_id"]) for code in project.codes()}
        normalized = _normalize_explore_code_palette(palette, valid_ids)
        values_by_code: dict[int, str] = {}
        if focal_unit_id is not None:
            observation_id = _selection_observation_id(
                project, int(focal_unit_id), selected_unit_ids
            )
            if observation_id is not None:
                values_by_code = {
                    int(row["code_id"]): str(row["value"])
                    for row in project.current_annotations_for_observation(observation_id)
                }
        disabled_values: list[bool] = []
        positive_classes: list[str] = []
        negative_classes: list[str] = []
        unsure_classes: list[str] = []
        for component_id in ids or []:
            index = int(component_id.get("index", -1))
            code_id = normalized[index] if 0 <= index < len(normalized) else None
            disabled = focal_unit_id is None or code_id is None
            disabled_values.append(disabled)
            current = values_by_code.get(int(code_id)) if code_id is not None else None
            positive, negative, unsure = _judgment_button_classes(
                current, prefix="explore-code"
            )
            positive_classes.append(positive)
            negative_classes.append(negative)
            unsure_classes.append(unsure)
        return (
            disabled_values,
            list(disabled_values),
            list(disabled_values),
            positive_classes,
            negative_classes,
            unsure_classes,
        )

    @app.callback(
        Output("explore-annotation-refresh-store", "data", allow_duplicate=True),
        Output("explore-code-status", "children", allow_duplicate=True),
        Output("focus-auto-training-request-store", "data", allow_duplicate=True),
        Input({"type": "explore-code-positive", "index": ALL}, "n_clicks"),
        Input({"type": "explore-code-negative", "index": ALL}, "n_clicks"),
        Input({"type": "explore-code-unsure", "index": ALL}, "n_clicks"),
        State("explore-focal-unit-store", "data"),
        State("explore-code-palette-store", "data"),
        State("explore-span-selection-store", "data"),
        State("explore-annotation-refresh-store", "data"),
        State("focus-code-dropdown", "value"),
        State("focus-auto-retrain", "value"),
        State("focus-train-scope", "value"),
        State("focus-tune-hyperparameters", "value"),
        State("focus-recommendation-source", "value"),
        State("focus-active-classifier-dropdown", "value"),
        State("focus-committee-dropdown", "value"),
        prevent_initial_call=True,
    )
    def annotate_from_explore_palette(
        positive_clicks: list[int] | None,
        negative_clicks: list[int] | None,
        unsure_clicks: list[int] | None,
        focal_unit_id: int | None,
        palette: list[int | None] | None,
        selected_unit_ids: list[int] | None,
        refresh: int | None,
        develop_code_id: int | None,
        auto_train_values: list[str] | None,
        train_scope_values: list[str] | None,
        tune_values: list[str] | None,
        recommendation_source: str | None,
        active_classifier_spec_id: int | None,
        committee_id: int | None,
    ) -> tuple[Any, Any, Any]:
        del positive_clicks, negative_clicks, unsure_clicks
        triggered = callback_context.triggered_id
        triggered_value = (
            callback_context.triggered[0].get("value")
            if callback_context.triggered
            else None
        )
        # Pattern-matched buttons can fire when a dynamic row is mounted. Only a
        # positive click count is an actual coding judgment.
        if not isinstance(triggered, dict) or int(triggered_value or 0) < 1:
            return no_update, no_update, no_update
        if focal_unit_id is None:
            return no_update, "Select a focal unit first.", no_update
        valid_ids = {int(code["code_id"]) for code in project.codes()}
        normalized = _normalize_explore_code_palette(palette, valid_ids)
        row_index = int(triggered.get("index", -1))
        code_id = normalized[row_index] if 0 <= row_index < len(normalized) else None
        if code_id is None:
            return no_update, "Select or create a code first.", no_update
        values = {
            "explore-code-positive": "positive",
            "explore-code-negative": "negative",
            "explore-code-unsure": "unsure",
        }
        value = values.get(str(triggered.get("type")))
        if value is None:
            return no_update, no_update, no_update
        selected = [int(item) for item in (selected_unit_ids or [focal_unit_id])]
        observation_id = _selection_observation_id(project, int(focal_unit_id), selected)
        current = _current_annotation_value(project, observation_id, int(code_id))
        if current == value:
            return no_update, no_update, no_update
        _observation_id, event_id = project.annotate_span(
            selected, int(code_id), value, origin="human_direct"
        )
        request, message = _post_saved_external_judgment_training_request(
            project, code_id=int(code_id), develop_code_id=develop_code_id,
            annotation_event_id=int(event_id),
            auto_train_values=auto_train_values,
            train_scope_values=train_scope_values,
            tune_values=tune_values,
            recommendation_source=recommendation_source,
            active_classifier_spec_id=active_classifier_spec_id,
            committee_id=committee_id,
        )
        if request is not None:
            request["origin"] = "explore"
        return (
            int(refresh or 0) + 1, message,
            training_inbox.enqueue(request) if request is not None else no_update,
        )

    @app.callback(
        Output("focus-active-area", "className"),
        Input("focus-code-dropdown", "value"),
    )
    def set_focus_active_state(code_id: int | None) -> str:
        return "focus-active-area" if code_id is not None else "focus-active-area focus-disabled"


    @app.callback(
        Output("classifier-spec-modal", "style"),
        Output("classifier-spec-status", "children"),
        Input("focus-new-classifier-button", "n_clicks"),
        Input("classifier-spec-close", "n_clicks"),
        Input("classifier-spec-save", "n_clicks"),
        State("classifier-spec-name", "value"),
        State("classifier-spec-geometry", "value"),
        State("classifier-spec-algorithm", "value"),
        State("classifier-spec-scaling", "value"),
        State("focus-code-dropdown", "value"),
        prevent_initial_call=True,
    )
    def manage_classifier_spec_modal(
        open_clicks: int,
        close_clicks: int,
        save_clicks: int,
        name: str | None,
        geometry_id: int | None,
        algorithm: str | None,
        feature_scaling: str | None,
        code_id: int | None,
    ) -> tuple[dict[str, str], str]:
        del open_clicks, close_clicks, save_clicks
        triggered = str(callback_context.triggered_id)
        if triggered == "focus-new-classifier-button":
            return {"display": "flex"}, ""
        if triggered == "classifier-spec-close":
            return {"display": "none"}, ""
        if code_id is None:
            return {"display": "flex"}, "Select a code first."
        if geometry_id is None:
            return {"display": "flex"}, "Select a parent geometry."
        try:
            selected_algorithm = str(algorithm or "logistic_l2")
            clean_name = (name or "").strip()
            if not clean_name:
                geometry = geometry_by_id[int(geometry_id)]
                code = project.code(int(code_id))
                clean_name = (
                    f"{code['name']} · {geometry['name']} · "
                    f"{classifier_algorithm_label(selected_algorithm)}"
                )
            project.create_classifier_spec(
                code_id=int(code_id),
                name=clean_name,
                geometry_id=int(geometry_id),
                algorithm=selected_algorithm,
                hyperparameters={"feature_scaling": str(feature_scaling or "off")},
            )
        except (KeyError, ValueError) as error:
            return {"display": "flex"}, str(error)
        return {"display": "none"}, ""

    @app.callback(
        Output("classifier-spec-refresh-store", "data"),
        Input("classifier-spec-save", "n_clicks"),
        State("classifier-spec-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def refresh_after_classifier_spec_save(
        clicks: int, refresh: int | None
    ) -> int:
        del clicks
        return int(refresh or 0) + 1

    @app.callback(
        Output("classifier-manager-modal", "style"),
        Output("classifier-manager-table", "data"),
        Output("classifier-manager-status", "children"),
        Output("classifier-spec-refresh-store", "data", allow_duplicate=True),
        Output("committee-refresh-store", "data", allow_duplicate=True),
        Input("focus-manage-classifiers", "n_clicks"),
        Input("classifier-manager-close", "n_clicks"),
        Input("classifier-manager-rename", "n_clicks"),
        Input("classifier-manager-scaling-save", "n_clicks"),
        Input("classifier-manager-delete", "n_clicks"),
        State("classifier-manager-table", "selected_rows"),
        State("classifier-manager-table", "data"),
        State("classifier-manager-rename-name", "value"),
        State("classifier-manager-scaling", "value"),
        State("focus-code-dropdown", "value"),
        State("classifier-spec-refresh-store", "data"),
        State("committee-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def manage_classifier_manager_modal(
        open_clicks: int,
        close_clicks: int,
        rename_clicks: int,
        scaling_clicks: int,
        delete_clicks: int,
        selected_rows: list[int] | None,
        table_data: list[dict[str, Any]] | None,
        rename_name: str | None,
        scaling_mode: str | None,
        code_id: int | None,
        spec_refresh: int | None,
        committee_refresh: int | None,
    ) -> tuple[dict[str, str], list[dict[str, Any]], str, int, int]:
        del open_clicks, close_clicks, rename_clicks, scaling_clicks, delete_clicks
        triggered = str(callback_context.triggered_id)
        next_spec_refresh = int(spec_refresh or 0)
        next_committee_refresh = int(committee_refresh or 0)
        if triggered == "classifier-manager-close":
            return {"display": "none"}, [], "", next_spec_refresh, next_committee_refresh
        message = ""
        if triggered == "classifier-manager-rename":
            rows = table_data or []
            selection = selected_rows or []
            if not selection:
                message = "Select a classifier to rename."
            else:
                try:
                    row = rows[int(selection[0])]
                    project.rename_classifier_spec(
                        int(row["classifier_spec_id"]), name=str(rename_name or "")
                    )
                except (IndexError, KeyError, ValueError) as error:
                    message = str(error)
                else:
                    next_spec_refresh += 1
                    next_committee_refresh += 1
                    message = f"Classifier renamed to {str(rename_name).strip()!r}."
        if triggered == "classifier-manager-scaling-save":
            rows = table_data or []
            selection = selected_rows or []
            if not selection:
                message = "Select a classifier to change its scaling."
            else:
                try:
                    selected = rows[int(selection[0])]
                    spec = project.database.get_classifier_spec(int(selected["classifier_spec_id"]))
                    parameters = dict(spec["hyperparameters"])
                    parameters["feature_scaling"] = str(scaling_mode or "off")
                    parameters.pop("feature_scale_gamma", None)
                    project.update_classifier_spec(
                        int(spec["classifier_spec_id"]),
                        name=str(spec["name"]),
                        hyperparameters=parameters,
                    )
                except (IndexError, KeyError, ValueError) as error:
                    message = str(error)
                else:
                    next_spec_refresh += 1
                    next_committee_refresh += 1
                    message = "Classifier scaling updated. Retrain to refresh its fitted model."
        if triggered == "classifier-manager-delete":
            rows = table_data or []
            selection = selected_rows or []
            if not selection:
                message = "Select a classifier to delete."
            else:
                try:
                    project.delete_classifier_spec(
                        int(rows[int(selection[0])]["classifier_spec_id"])
                    )
                except (KeyError, ValueError) as error:
                    message = str(error)
                else:
                    next_spec_refresh += 1
                    next_committee_refresh += 1
                    message = "Classifier deleted from active use."
        rows = [
            {
                "classifier_spec_id": int(row["classifier_spec_id"]),
                "name": str(row["name"]),
                "geometry": str(row["geometry_name"]),
                "algorithm": classifier_algorithm_label(str(row["algorithm"])),
                "settings": hyperparameter_summary(
                    str(row["algorithm"]), dict(row["hyperparameters"])
                ),
                "feature_scaling": dict(row["hyperparameters"]).get("feature_scaling", "off"),
            }
            for row in (
                project.classifier_specs(code_id=int(code_id))
                if code_id is not None else []
            )
        ]
        return {"display": "flex"}, rows, message, next_spec_refresh, next_committee_refresh

    @app.callback(
        Output("classifier-manager-rename-name", "value"),
        Input("classifier-manager-table", "selected_rows"),
        Input("classifier-manager-table", "data"),
        prevent_initial_call=True,
    )
    def populate_classifier_rename_name(
        selected_rows: list[int] | None,
        table_data: list[dict[str, Any]] | None,
    ) -> str:
        rows = table_data or []
        selection = selected_rows or []
        if not selection:
            return ""
        try:
            return str(rows[int(selection[0])]["name"])
        except (IndexError, KeyError, TypeError, ValueError):
            return ""

    @app.callback(
        Output("classifier-manager-scaling", "value"),
        Input("classifier-manager-table", "selected_rows"),
        Input("classifier-manager-table", "data"),
        prevent_initial_call=True,
    )
    def populate_classifier_scaling(
        selected_rows: list[int] | None,
        table_data: list[dict[str, Any]] | None,
    ) -> str:
        rows = table_data or []
        selection = selected_rows or []
        if not selection:
            return "off"
        try:
            return str(rows[int(selection[0])].get("feature_scaling", "off"))
        except (IndexError, KeyError, TypeError, ValueError):
            return "off"

    @app.callback(
        Output("committee-modal", "style"),
        Output("committee-create-modal", "style"),
        Output("committee-members", "options"),
        Output("committee-members", "value"),
        Output("committee-table", "data"),
        Output("committee-table", "selected_rows"),
        Output("committee-status", "children"),
        Output("committee-refresh-store", "data", allow_duplicate=True),
        Input("focus-manage-committees", "n_clicks"),
        Input("focus-new-committee", "n_clicks"),
        Input("committee-create-close", "n_clicks"),
        Input("committee-close", "n_clicks"),
        Input("committee-save", "n_clicks"),
        Input("committee-rename", "n_clicks"),
        Input("committee-delete", "n_clicks"),
        State("focus-code-dropdown", "value"),
        State("committee-name", "value"),
        State("committee-members", "value"),
        State("committee-aggregation", "value"),
        State("committee-table", "selected_rows"),
        State("committee-table", "data"),
        State("committee-manager-rename-name", "value"),
        State("committee-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def manage_committee_modal(
        open_clicks: int,
        new_clicks: int,
        create_close_clicks: int,
        close_clicks: int,
        save_clicks: int,
        rename_clicks: int,
        delete_clicks: int,
        code_id: int | None,
        name: str | None,
        member_ids: list[int] | None,
        aggregation: str | None,
        selected_rows: list[int] | None,
        table_data: list[dict[str, Any]] | None,
        rename_name: str | None,
        refresh: int | None,
    ) -> tuple[
        Any,
        Any,
        list[dict[str, Any]],
        list[int],
        list[dict[str, Any]],
        list[int],
        str,
        int,
    ]:
        del open_clicks, new_clicks, create_close_clicks, close_clicks
        del save_clicks, rename_clicks, delete_clicks
        triggered = str(callback_context.triggered_id)
        if triggered == "committee-close":
            return {"display": "none"}, no_update, [], [], [], [], "", int(refresh or 0)
        if triggered == "committee-create-close":
            return no_update, {"display": "none"}, [], [], [], [], "", int(refresh or 0)
        if code_id is None:
            return (
                {"display": "none"},
                {"display": "none"},
                [],
                [],
                [],
                [],
                "Select a code first.",
                int(refresh or 0),
            )
        specs = project.classifier_specs(code_id=int(code_id))
        valid_member_ids = {int(row["classifier_spec_id"]) for row in specs}
        visible_member_ids = [
            int(value)
            for value in (member_ids or [])
            if int(value) in valid_member_ids
        ]
        # Opening the manager always starts with a clean selection. In particular,
        # dcc.Checklist retains its value when only its options change, which can
        # otherwise leave an invisible classifier from the previously active code
        # selected and submit it with the new code's visible classifiers.
        selected_member_ids = (
            [] if triggered in {"focus-manage-committees", "focus-new-committee"}
            else visible_member_ids
        )
        selected_committee_rows = list(selected_rows or [])
        message = ""
        next_refresh = int(refresh or 0)
        created = False
        try:
            if triggered == "committee-save":
                clean_name = (name or "").strip()
                if not clean_name:
                    raise ValueError("Enter a committee name.")
                members = visible_member_ids
                if not members:
                    raise ValueError("Select at least one classifier.")
                project.create_classifier_committee(
                    code_id=int(code_id),
                    name=clean_name,
                    classifier_spec_ids=members,
                    aggregation=str(aggregation or "mean"),
                )
                selected_member_ids = []
                selected_committee_rows = []
                next_refresh += 1
                message = "Committee saved."
                created = True
            elif triggered == "committee-rename":
                rows = table_data or []
                selection = selected_rows or []
                if not selection:
                    raise ValueError("Select a committee to rename.")
                project.rename_classifier_committee(
                    int(rows[int(selection[0])]["committee_id"]),
                    name=str(rename_name or ""),
                )
                next_refresh += 1
                message = f"Committee renamed to {str(rename_name or '').strip()!r}."
            elif triggered == "committee-delete":
                rows = table_data or []
                selection = selected_rows or []
                if not selection:
                    raise ValueError("Select a committee to delete.")
                project.delete_classifier_committee(
                    int(rows[int(selection[0])]["committee_id"])
                )
                next_refresh += 1
                message = "Committee deleted."
        except (KeyError, ValueError) as error:
            message = str(error)
        member_options = [
            {
                "label": f"{row['name']} · {row['geometry_name']}",
                "value": int(row["classifier_spec_id"]),
            }
            for row in specs
        ]
        committees = project.classifier_committees(int(code_id))
        rows = [
            {
                "committee_id": int(row["committee_id"]),
                "name": str(row["name"]),
                "aggregation": COMMITTEE_AGGREGATION_LABELS.get(
                    str(row["aggregation"]), str(row["aggregation"])
                ),
                "members": ", ".join(
                    str(member["classifier_name"]) for member in row["members"]
                ),
            }
            for row in committees
        ]
        return (
            {"display": "flex"}
            if triggered in {"focus-manage-committees", "committee-rename", "committee-delete"}
            else no_update,
            {"display": "none"} if created
            else {"display": "flex"} if triggered == "focus-new-committee"
            else no_update,
            member_options,
            selected_member_ids,
            rows,
            selected_committee_rows,
            message,
            next_refresh,
        )

    @app.callback(
        Output("committee-manager-rename-name", "value"),
        Input("committee-table", "selected_rows"),
        Input("committee-table", "data"),
        prevent_initial_call=True,
    )
    def populate_committee_rename_name(
        selected_rows: list[int] | None,
        table_data: list[dict[str, Any]] | None,
    ) -> str:
        rows = table_data or []
        selection = selected_rows or []
        if not selection:
            return ""
        try:
            return str(rows[int(selection[0])]["name"])
        except (IndexError, KeyError, TypeError, ValueError):
            return ""

    @app.callback(
        Output("testing-center-modal", "style"),
        Output("testing-center-summary", "children"),
        Output("testing-center-table", "data"),
        Output("testing-center-events", "data"),
        Input("focus-testing-center-button", "n_clicks"),
        Input("testing-center-close", "n_clicks"),
        Input("focus-code-dropdown", "value"),
        Input("teaching-example-refresh-store", "data"),
        Input("focus-model-refresh-store", "data"),
        State("testing-center-modal", "style"),
        prevent_initial_call=True,
    )
    def manage_testing_center(
        open_clicks: int,
        close_clicks: int,
        code_id: int | None,
        teaching_refresh: int | None,
        model_refresh: int | None,
        current_style: dict[str, str] | None,
    ) -> tuple[dict[str, str], Any, list[dict[str, Any]], list[dict[str, Any]]]:
        del open_clicks, close_clicks, teaching_refresh, model_refresh
        triggered = str(callback_context.triggered_id)
        if triggered == "testing-center-close":
            return {"display": "none"}, "", [], []
        if triggered != "focus-testing-center-button" and (current_style or {}).get("display") == "none":
            return {"display": "none"}, "", [], []
        if code_id is None:
            return {"display": "flex"}, "Select a code first.", [], []
        summary = project.classifier_testing_summary(int(code_id))
        classifier_rows = [
            {
                "classifier": row["classifier_name"],
                "n": int(row["n"]),
                "accuracy": f"{float(row['accuracy']):.3f}",
                "brier": (
                    f"{float(row['brier']):.3f}" if row["brier"] is not None else "—"
                ),
            }
            for row in summary["classifiers"]
        ]
        event_rows = [
            {
                "event_type": row["event_type"],
                "classifier": row["classifier_name"],
                "human": row["human_label"],
                "predicted": row["predicted_label"],
                "probability": (
                    f"{float(row['probability']):.3f}"
                    if row["probability"] is not None
                    else "—"
                ),
            }
            for row in summary["events"]
        ]
        return (
            {"display": "flex"},
            f"{len(event_rows):,} stored diagnostic assessment events.",
            classifier_rows,
            event_rows,
        )

    @app.callback(
        Output("focus-active-classifier-dropdown", "options"),
        Output("focus-active-classifier-dropdown", "value"),
        Input("focus-code-dropdown", "value"),
        Input("classifier-spec-refresh-store", "data"),
        State("focus-active-classifier-dropdown", "value"),
    )
    def refresh_focus_classifier_specs(
        code_id: int | None,
        refresh: int | None,
        current_spec_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        del refresh
        if code_id is None:
            return [], None
        rows = project.classifier_specs(code_id=int(code_id))
        options = [
            {
                "label": f"{row['name']} · {row['geometry_name']}",
                "value": int(row["classifier_spec_id"]),
            }
            for row in rows
        ]
        valid = {int(option["value"]) for option in options}
        value = (
            int(current_spec_id)
            if current_spec_id is not None and int(current_spec_id) in valid
            else (int(options[0]["value"]) if options else None)
        )
        return options, value

    @app.callback(
        Output("focus-committee-dropdown", "options"),
        Output("focus-committee-dropdown", "value"),
        Input("focus-code-dropdown", "value"),
        Input("committee-refresh-store", "data"),
        State("focus-committee-dropdown", "value"),
    )
    def refresh_focus_committees(
        code_id: int | None,
        refresh: int | None,
        current_committee_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        del refresh
        rows = project.classifier_committees(int(code_id)) if code_id is not None else []
        options = [
            {
                "label": f"{row['name']} ({len(row['members'])} classifiers)",
                "value": int(row["committee_id"]),
            }
            for row in rows
        ]
        valid = {int(option["value"]) for option in options}
        value = (
            int(current_committee_id)
            if current_committee_id is not None and int(current_committee_id) in valid
            else (int(options[0]["value"]) if options else None)
        )
        return options, value

    @app.callback(
        Output("focus-active-classifier-status", "children"),
        Output("focus-active-classifier-status", "className"),
        Output("focus-active-classifier-detail", "children"),
        Input("focus-code-dropdown", "value"),
        Input("focus-active-classifier-dropdown", "value"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
        Input("focus-model-refresh-store", "data"),
        Input("classifier-spec-refresh-store", "data"),
        Input("apply-refresh-store", "data"),
        Input("workspace-tabs", "value"),
    )
    def update_focus_active_classifier_status(
        code_id: int | None,
        classifier_spec_id: int | None,
        focus_refresh: int | None,
        teaching_refresh: int | None,
        model_refresh: int | None,
        spec_refresh: int | None,
        apply_refresh: int | None,
        workspace: str | None,
    ) -> tuple[Any, Any, Any]:
        del focus_refresh, teaching_refresh, model_refresh, spec_refresh, apply_refresh
        if workspace != "focus":
            return no_update, no_update, no_update
        if code_id is None:
            return "No code", "classifier-status-badge status-not-trained", ""
        if classifier_spec_id is None:
            return "No classifier", "classifier-status-badge status-not-trained", ""
        row = project.classifier_status(
            code_id=int(code_id), classifier_spec_ids=[int(classifier_spec_id)]
        )["classifiers"][0]
        status = str(row["status"])
        label = {
            "current": "Current",
            "stale": "Stale",
            "not_trained": "Not trained",
        }[status]
        algorithm = str(row["algorithm"])
        parameter_detail = hyperparameter_summary(algorithm, dict(row["hyperparameters"]))
        detail = f"{row['geometry_name']} · {classifier_algorithm_label(algorithm)}"
        if parameter_detail:
            detail += f" · {parameter_detail}"
        return label, f"classifier-status-badge status-{status.replace('_', '-')}", detail

    @app.callback(
        Output("focus-committee-status", "children"),
        Output("focus-committee-badge", "children"),
        Output("focus-committee-badge", "className"),
        Input("focus-code-dropdown", "value"),
        Input("focus-committee-dropdown", "value"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
        Input("focus-model-refresh-store", "data"),
        Input("committee-refresh-store", "data"),
        Input("classifier-spec-refresh-store", "data"),
        Input("apply-refresh-store", "data"),
        Input("workspace-tabs", "value"),
    )
    def update_focus_committee_status(
        code_id: int | None,
        committee_id: int | None,
        focus_refresh: int | None,
        teaching_refresh: int | None,
        model_refresh: int | None,
        committee_refresh: int | None,
        spec_refresh: int | None,
        apply_refresh: int | None,
        workspace: str | None,
    ) -> tuple[Any, Any, Any]:
        del focus_refresh, teaching_refresh, model_refresh, committee_refresh, spec_refresh, apply_refresh
        if workspace != "focus":
            return no_update, no_update, no_update
        if code_id is None:
            return "Select a code first.", "No code", "classifier-status-badge status-not-trained"
        if committee_id is None:
            return "No committee selected.", "No committee", "classifier-status-badge status-not-trained"
        result = project.classifier_committee_status(
            code_id=int(code_id), committee_id=int(committee_id)
        )
        committee = result["committee"]
        counts = result["member_counts"]
        status = str(result["status"])
        badge = {
            "current": "Current",
            "stale": "Stale",
            "not_trained": "Not trained",
        }[status]
        aggregation = str(committee["aggregation"])
        aggregation_label = COMMITTEE_AGGREGATION_LABELS.get(aggregation, aggregation)
        member_summary = (
            f"Members: {counts['current']} current · {counts['stale']} stale · "
            f"{counts['not_trained']} not trained"
        )
        detail = (
            " · No separate committee training needed"
            if aggregation != "logistic_stack"
            else " · Learned stacking"
        )
        return (
            f"{aggregation_label}{detail} · {member_summary}",
            badge,
            f"classifier-status-badge status-{status.replace('_', '-')}",
        )

    @app.callback(
        Output("focus-train-action-status", "children"),
        Output("focus-model-refresh-store", "data", allow_duplicate=True),
        Input("focus-train-active", "n_clicks"),
        State("focus-code-dropdown", "value"),
        State("focus-active-classifier-dropdown", "value"),
        State("focus-train-scope", "value"),
        State("focus-tune-hyperparameters", "value"),
        State("focus-model-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def train_active_classifier(
        clicks: int,
        code_id: int | None,
        classifier_spec_id: int | None,
        train_scope: list[str] | None,
        tune_values: list[str] | None,
        refresh: int | None,
    ) -> tuple[str, int]:
        del clicks
        if code_id is None or classifier_spec_id is None:
            return "Select a code and classifier first.", int(refresh or 0)
        train_all = "all" in set(train_scope or [])
        tune_hyperparameters = "tune" in set(tune_values or [])
        spec_ids = (
            [
                int(row["classifier_spec_id"])
                for row in project.classifier_specs(code_id=int(code_id))
            ]
            if train_all
            else [int(classifier_spec_id)]
        )
        try:
            fits = project.train_classifiers(
                code_id=int(code_id),
                classifier_spec_ids=spec_ids,
                tune=tune_hyperparameters,
                retune_current=tune_hyperparameters,
                folds=_read_ui_settings(project.project_dir)["cv_folds"],
            )
        except (KeyError, ValueError) as error:
            return str(error), int(refresh or 0)
        selections = [dict(fit.get("training_selection", {})) for fit in fits.values()]
        tuned_count = sum(bool(selection.get("tuned")) for selection in selections)
        reused_count = sum(
            selection.get("reason") == "current_fit_reused" for selection in selections
        )
        fast_fit_count = sum(
            selection.get("reason") == "tuning_disabled" for selection in selections
        )
        insufficient_count = sum(
            selection.get("reason") == "insufficient_cv_data" for selection in selections
        )
        if train_all:
            parts = [f"Updated {len(fits):,} classifiers"]
            if tuned_count:
                parts.append(f"CV-tuned {tuned_count:,}")
            if fast_fit_count:
                parts.append(f"fit {fast_fit_count:,} with current hyperparameters")
            if insufficient_count:
                parts.append(
                    f"could not CV-tune {insufficient_count:,} with too little labeled data"
                )
            if reused_count:
                parts.append(f"reused {reused_count:,} current fits")
            message = ". ".join(parts) + "."
        else:
            fit = next(iter(fits.values()))
            selection = dict(fit.get("training_selection", {}))
            name = str(fit["classifier_name"])
            if selection.get("tuned"):
                message = (
                    f"CV-tuned {name} with {int(selection['folds'])}-fold "
                    f"cross-validation."
                )
                if selection.get("fit_reused"):
                    message += " The selected configuration was already fitted."
            elif selection.get("reason") == "insufficient_cv_data":
                message = (
                    f"CV tuning requested for {name}, but more labeled examples are "
                    "needed; using the current hyperparameters."
                )
            elif selection.get("reason") == "current_fit_reused":
                message = f"{name} is already current."
            elif selection.get("reason") == "tuning_disabled":
                message = f"Trained {name} with current hyperparameters."
            else:
                message = f"Trained {name}."
        return message, int(refresh or 0) + 1

    @app.callback(
        Output("focus-train-committee", "style"),
        Output("focus-train-committee", "disabled"),
        Input("focus-committee-dropdown", "value"),
        Input("committee-refresh-store", "data"),
    )
    def configure_committee_train_button(
        committee_id: int | None, refresh: int | None
    ) -> tuple[dict[str, str], bool]:
        del refresh
        if committee_id is None:
            return {"display": "none"}, True
        committee = project.database.get_classifier_committee(int(committee_id))
        trainable = str(committee["aggregation"]) == "logistic_stack"
        return ({} if trainable else {"display": "none"}), not trainable

    @app.callback(
        Output("focus-committee-train-status", "children"),
        Output("focus-model-refresh-store", "data", allow_duplicate=True),
        Input("focus-train-committee", "n_clicks"),
        State("focus-code-dropdown", "value"),
        State("focus-committee-dropdown", "value"),
        State("focus-model-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def train_selected_committee(
        clicks: int,
        code_id: int | None,
        committee_id: int | None,
        refresh: int | None,
    ) -> tuple[str, int]:
        del clicks
        if code_id is None or committee_id is None:
            return "Select a code and committee first.", int(refresh or 0)
        try:
            fit = project.train_classifier_committee(
                code_id=int(code_id), committee_id=int(committee_id)
            )
        except (KeyError, ValueError) as error:
            return str(error), int(refresh or 0)
        return (
            f"Trained learned committee aggregation #{int(fit['committee_fit_id'])}.",
            int(refresh or 0) + 1,
        )

    @app.callback(
        Output("focus-disagreement", "style"),
        Input("focus-recommendation-source", "value"),
    )
    def show_committee_disagreement(source: str | None) -> dict[str, str]:
        return {} if source == "committee" else {"display": "none"}

    focus_recommendation_inputs = [
        Input("focus-most-likely", "n_clicks"),
        Input("focus-least-likely", "n_clicks"),
        Input("focus-most-uncertain", "n_clicks"),
        Input("focus-disagreement", "n_clicks"),
        Input("focus-review-unsure", "n_clicks"),
        Input("focus-random", "n_clicks"),
    ]

    @app.callback(
        Output("focal-unit-store", "data", allow_duplicate=True),
        Output("navigation-history-store", "data", allow_duplicate=True),
        Output("focus-recommendation-status", "children"),
        Output("focus-recommendation-store", "data"),
        *focus_recommendation_inputs,
        State("focus-recommendation-source", "value"),
        State("focus-code-dropdown", "value"),
        State("active-session-store", "data"),
        State("focus-active-classifier-dropdown", "value"),
        State("focus-committee-dropdown", "value"),
        State("focal-unit-store", "data"),
        State("navigation-history-store", "data"),
        prevent_initial_call=True,
    )
    def focus_recommend(
        most_likely_clicks: int,
        least_likely_clicks: int,
        most_uncertain_clicks: int,
        disagreement_clicks: int,
        review_unsure_clicks: int,
        random_clicks: int,
        recommendation_source: str | None,
        code_id: int | None,
        session_id: int,
        active_classifier_spec_id: int | None,
        committee_id: int | None,
        focal_unit_id: int | None,
        history: dict[str, Any] | None,
    ) -> tuple[Any, Any, str, dict[str, Any]]:
        del most_likely_clicks, least_likely_clicks, most_uncertain_clicks
        del disagreement_clicks, review_unsure_clicks, random_clicks
        if code_id is None:
            return no_update, no_update, "Create or select a code first.", {}
        source = str(recommendation_source or "classifier")
        triggered = str(callback_context.triggered_id)
        strategy_by_button = {
            "focus-most-likely": "likely_positive",
            "focus-least-likely": "likely_negative",
            "focus-most-uncertain": (
                "committee_uncertainty"
                if source == "committee"
                else "classifier_uncertainty"
            ),
            "focus-disagreement": "classifier_disagreement",
            "focus-review-unsure": "review_unsure",
            "focus-random": "random",
        }
        strategy = strategy_by_button[triggered]
        if source == "committee" and committee_id is None and strategy not in {
            "random",
            "review_unsure",
        }:
            return no_update, no_update, "Select a classifier committee first.", {}
        try:
            result = _recommend_without_fitting(
                project,
                code_id=int(code_id),
                session_id=int(session_id),
                strategy=strategy,
                active_classifier_spec_id=(
                    int(active_classifier_spec_id)
                    if active_classifier_spec_id is not None
                    else None
                ),
                committee_id=(int(committee_id) if committee_id is not None else None),
                recommendation_source=source,
            )
        except (KeyError, ValueError) as error:
            return no_update, no_update, str(error), {}
        if result is None:
            if strategy == "review_unsure":
                return no_update, no_update, "This code has no unsure units to revisit.", {}
            return no_update, no_update, "No unseen, unreviewed units remain in this session.", {}

        fit_ids = list(result.get("classifier_fit_ids", {}).values())
        classifier_fit_id = int(fit_ids[0]) if len(fit_ids) == 1 else None
        project.record_visit(
            int(session_id),
            int(result["unit_id"]),
            method=f"focus_{result['strategy']}",
            source_unit_id=int(focal_unit_id) if focal_unit_id is not None else None,
            active_code_id=int(code_id),
            classifier_fit_id=classifier_fit_id,
        )
        if result["fallback"]:
            status = (
                "Both positive and negative labels are required. "
                "GeCo supplied a random unseen unit instead."
            )
        else:
            status = (
                f"Opened unit {int(result['unit_id']):,} via "
                f"{result['strategy'].replace('_', ' ')}."
            )
            if list(result.get("stale_classifiers", [])):
                status += (
                    " Using classifier scores from before the latest labels; "
                    "train when the current labeling batch is ready."
                )
        unit_id = int(result["unit_id"])
        # The recommendation remains in a browser Store after navigation or code
        # changes; retain its exact context to validate later label provenance.
        return (
            unit_id,
            _append_navigation_history(history, unit_id),
            status,
            {**result, "code_id": int(code_id), "session_id": int(session_id)},
        )

    @app.callback(
        Output("focus-auto-training-status", "children"),
        Output("focus-model-refresh-store", "data", allow_duplicate=True),
        Output("explore-code-status", "children", allow_duplicate=True),
        Output("code-center-assignment-status", "children", allow_duplicate=True),
        Output("teaching-example-status", "children", allow_duplicate=True),
        Output("apply-training-status", "children"),
        Input("focus-auto-training-request-store", "data"),
        State("focus-model-refresh-store", "data"),
        prevent_initial_call=True,
        running=[
            (
                Output("focus-auto-training-progress", "children"),
                "Training selected classifier(s) after saved coding judgment…",
                "",
            ),
        ],
    )
    def train_after_develop_judgment(
        request: dict[str, Any] | None, refresh: int | None
    ) -> tuple[str, Any, Any, Any, Any, Any]:
        if not request:
            return "", no_update, no_update, no_update, no_update, no_update

        def fit(next_request: dict[str, Any]) -> dict[str, Any]:
            return project.train_classifiers_after_evidence_change(
                code_id=int(next_request["code_id"]),
                classifier_spec_ids=[int(i) for i in next_request["classifier_spec_ids"]],
                tune=bool(next_request["tune"]),
                folds=_read_ui_settings(project.project_dir)["cv_folds"],
            )

        completed = training_inbox.drain(fit)
        if not completed:
            return no_update, no_update, no_update, no_update, no_update, no_update
        messages: dict[str, str] = {}
        any_trained = False
        for task, result, error in completed:
            origin = str(task.get("origin", "develop"))
            saved_label = (
                "Teaching example saved" if origin == "teaching_example"
                else ("Apply batch saved" if origin == "apply" else "Saved judgment")
            )
            if error is not None:
                message = f"{saved_label}. Automatic training failed: {error}"
            elif str(result["status"]) == "trained":
                any_trained = True
                message = (
                    f"{saved_label}; trained {len(result['trained_fit_ids'])} "
                    "selected classifier(s)."
                )
            elif str(result["status"]) == "awaiting_two_classes":
                message = (
                    f"{saved_label}; training awaits positive and negative "
                    "evidence within each selected classifier's geometry."
                )
            else:
                message = f"{saved_label}; selected classifier(s) are current."
            awaiting = list((result or {}).get("awaiting_classifier_spec_ids", []))
            if awaiting and str(result["status"]) == "trained":
                message += (
                    f" {len(awaiting)} other classifier(s) lack both training "
                    "classes in their own geometry."
                )
            messages[origin] = message
        return (
            messages.get("develop", no_update),
            int(refresh or 0) + 1 if any_trained else no_update,
            messages.get("explore", no_update),
            messages.get("code_center", no_update),
            messages.get("teaching_example", no_update),
            messages.get("apply", no_update),
        )

    @app.callback(
        Output("focus-annotation-status", "children"),
        Output("focus-refresh-store", "data", allow_duplicate=True),
        Output("focus-prediction-map", "figure", allow_duplicate=True),
        Output("focus-auto-training-request-store", "data", allow_duplicate=True),
        Input("focus-label-positive", "n_clicks"),
        Input("focus-label-negative", "n_clicks"),
        Input("focus-label-unsure", "n_clicks"),
        State("focus-code-dropdown", "value"),
        State("focal-unit-store", "data"),
        State("active-session-store", "data"),
        State("focus-recommendation-store", "data"),
        State("focus-span-selection-store", "data"),
        State("focus-refresh-store", "data"),
        State("prediction-geometry-page-token", "data"),
        State("prediction-geometry-refresh-store", "data"),
        State("focus-auto-retrain", "value"),
        State("focus-train-scope", "value"),
        State("focus-tune-hyperparameters", "value"),
        State("focus-recommendation-source", "value"),
        State("focus-active-classifier-dropdown", "value"),
        State("focus-committee-dropdown", "value"),
        prevent_initial_call=True,
    )
    def annotate_focus_unit(
        positive_clicks: int,
        negative_clicks: int,
        unsure_clicks: int,
        code_id: int | None,
        focal_unit_id: int | None,
        session_id: int,
        recommendation: dict[str, Any] | None,
        selected_unit_ids: list[int] | None,
        refresh: int | None,
        prediction_page_token: str | None,
        prediction_refresh: dict[str, Any] | None,
        auto_train_values: list[str] | None,
        train_scope_values: list[str] | None,
        tune_values: list[str] | None,
        recommendation_source: str | None,
        active_classifier_spec_id: int | None,
        committee_id: int | None,
    ) -> tuple[str, int, Any, Any]:
        del positive_clicks, negative_clicks, unsure_clicks
        if code_id is None or focal_unit_id is None:
            return (
                "Select a code and focal unit before labeling.",
                int(refresh or 0),
                no_update,
                no_update,
            )
        value_by_button = {
            "focus-label-positive": "positive",
            "focus-label-negative": "negative",
            "focus-label-unsure": "unsure",
        }
        value = value_by_button[str(callback_context.triggered_id)]
        selected = [int(item) for item in (selected_unit_ids or [focal_unit_id])]
        observation_id = _selection_observation_id(
            project, int(focal_unit_id), selected
        )
        current = _current_annotation_value(project, observation_id, int(code_id))
        if current == value:
            return "", int(refresh or 0), no_update, no_update
        classifier_fit_id = _matching_recommendation_fit_id(
            recommendation,
            unit_id=int(focal_unit_id),
            code_id=int(code_id),
            session_id=int(session_id),
            selected_unit_ids=selected,
        )
        observation_id, _event_id = project.annotate_span(
            selected,
            int(code_id),
            value,
            origin="human_active_learning",
            classifier_fit_id=classifier_fit_id,
        )
        for cached in prediction_geometry_cache.values():
            if int(cached.get("code_id", -1)) != int(code_id):
                continue
            for point in cached.get("points", []):
                if int(point["observation_id"]) == int(observation_id):
                    point["assignment"] = value

        color = {
            "positive": "#3f8d5d",
            "negative": "#a83f3f",
            "unsure": "#d39a22",
        }[value]
        patch: Any = no_update
        if prediction_page_token:
            with prediction_page_lock:
                page = prediction_page_indexes.get(str(prediction_page_token))
            location = _matching_prediction_page_point(
                page, code_id=code_id, refresh_state=prediction_refresh,
                observation_id=int(observation_id),
            )
            if location is not None:
                trace_index, position = location
                patch = Patch()
                patch["data"][trace_index]["marker"]["color"][position] = color
                patch["data"][trace_index]["marker"]["line"]["color"][position] = color
        # The persisted human judgment always wins over a possibly stale UI
        # classifier selection. Report scope errors without discarding the
        # normal annotation-store/figure refresh.
        request, message = _post_saved_develop_judgment_training_request(
            project, code_id=int(code_id), annotation_event_id=int(_event_id),
            auto_train_values=auto_train_values,
            train_scope_values=train_scope_values,
            tune_values=tune_values,
            recommendation_source=recommendation_source,
            active_classifier_spec_id=active_classifier_spec_id,
            committee_id=committee_id,
        )
        return message, int(refresh or 0) + 1, patch, (
            training_inbox.enqueue(request) if request is not None else no_update
        )

    @app.callback(
        Output("focus-code-definition", "children"),
        Output("focus-label-counts", "children"),
        Output("focus-current-label", "children"),
        Input("focus-code-dropdown", "value"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
        Input("workspace-tabs", "value"),
    )
    def update_focus_code_summary(
        code_id: int | None,
        refresh: int,
        teaching_refresh: int | None,
        workspace: str | None,
    ) -> tuple[Any, Any, Any]:
        # Return to Develop is the reconciliation point for evidence saved in
        # Explore or Apply. Neither draft refreshes nor focal navigation should
        # repeatedly scan all annotations. Hidden tabs need no summary work.
        del refresh, teaching_refresh
        if workspace != "focus":
            return no_update, no_update, no_update
        if code_id is None:
            return "No code selected.", "", ""
        code = project.database.get_code(int(code_id))
        counts = project.database.annotation_counts(int(code_id))
        atomic_reviewed = sum(
            1
            for row in project.current_annotations(int(code_id))
            if row["unit_id"] is not None
        )
        unreviewed = max(0, len(units) - atomic_reviewed)
        description = code["description"] or "No description has been written yet."
        count_text = (
            f"Positive {counts['positive']:,} · Negative {counts['negative']:,} · "
            f"Unsure {counts['unsure']:,} · Unreviewed {unreviewed:,}"
        )
        training_counts = project.training_label_counts(int(code_id))
        model_note = (
            "Model-ready"
            if training_counts["positive"] > 0 and training_counts["negative"] > 0
            else "Random sampling until both active positive and negative examples exist"
        )
        return description, [html.Div(count_text), html.Small(model_note)], ""

    @app.callback(
        Output("focus-label-positive", "disabled"),
        Output("focus-label-negative", "disabled"),
        Output("focus-label-unsure", "disabled"),
        Output("focus-label-positive", "className"),
        Output("focus-label-negative", "className"),
        Output("focus-label-unsure", "className"),
        Input("focus-code-dropdown", "value"),
        Input("focus-focal-unit-store", "data"),
        Input("focus-span-selection-store", "data"),
        Input("focus-refresh-store", "data"),
    )
    def style_focus_judgment_buttons(
        code_id: int | None,
        focal_unit_id: int | None,
        selected_unit_ids: list[int] | None,
        refresh: int | None,
    ) -> tuple[bool, bool, bool, str, str, str]:
        del refresh
        disabled = code_id is None or focal_unit_id is None
        current = None
        if not disabled:
            observation_id = _selection_observation_id(
                project, int(focal_unit_id), selected_unit_ids
            )
            current = _current_annotation_value(
                project, observation_id, int(code_id)
            )
        classes = _judgment_button_classes(current, prefix="focus-label")
        return disabled, disabled, disabled, *classes

    @app.callback(
        Output("focus-focal-key", "children"),
        Output("focus-context-content", "children"),
        Output("focus-metadata-content", "children"),
        Input("focus-focal-unit-store", "data"),
        Input("focus-context-level", "value"),
        Input("focus-context-window", "value"),
    )
    def update_focus_reading_panel(
        focal_unit_id: int | None,
        level_index: int,
        window: int,
    ) -> tuple[Any, Any, Any]:
        if focal_unit_id is None:
            return (
                "No focal unit",
                "Choose a recommendation to begin developing this code.",
                "",
            )
        focal = project.unit(int(focal_unit_id))
        context = project.context(int(focal_unit_id), level=int(level_index), window=int(window))
        key_label = " › ".join(f"{name}: {value}" for name, value in focal["user_key"].items())
        children = _render_context_units(
            html,
            dcc,
            context,
            focal_unit_id=int(focal_unit_id),
            key_columns=list(metadata["key_columns"]),
            selected_unit_ids=[int(focal_unit_id)],
            checkbox_type="focus-span-checkbox",
            span_selection_enabled=True,
        )
        return key_label, children, _metadata_table(html, focal["metadata"])

    @app.callback(
        Output("focus-span-selection-store", "data"),
        Input("focus-focal-unit-store", "data"),
        Input("focus-context-level", "value"),
        Input("focus-context-window", "value"),
    )
    def reset_focus_span_selection(
        focal_unit_id: int | None, level_index: int, window: int
    ) -> list[int]:
        del level_index, window
        return [int(focal_unit_id)] if focal_unit_id is not None else []

    @app.callback(
        Output("focus-span-selection-store", "data", allow_duplicate=True),
        Input({"type": "focus-span-checkbox", "unit_id": ALL}, "value"),
        State({"type": "focus-span-checkbox", "unit_id": ALL}, "id"),
        State("focal-unit-store", "data"),
        State("focus-context-level", "value"),
        State("focus-context-window", "value"),
        State("focus-span-selection-store", "data"),
        prevent_initial_call=True,
    )
    def update_focus_span_selection(
        values: list[list[int]], ids: list[dict[str, Any]],
        focal_unit_id: int | None, level_index: int, window: int,
        current: list[int] | None,
    ) -> Any:
        triggered = callback_context.triggered_id
        if focal_unit_id is None or not isinstance(triggered, dict):
            return no_update
        clicked = int(triggered.get("unit_id"))
        value_by_id = {
            int(identifier["unit_id"]): value
            for identifier, value in zip(ids or [], values or [], strict=False)
        }
        checked = bool(value_by_id.get(clicked))
        if not checked and clicked not in set(current or []):
            return no_update
        context = project.context(
            int(focal_unit_id), level=int(level_index), window=int(window)
        )
        updated = _update_span_selection(
            context, focal_unit_id=int(focal_unit_id), current_unit_ids=current,
            clicked_unit_id=clicked, checked=checked,
        )
        return updated if updated != list(current or []) else no_update

    @app.callback(
        Output("focus-context-window", "value", allow_duplicate=True),
        Input("focus-context-level", "value"),
        Input("focus-context-window-minus", "n_clicks"),
        Input("focus-context-window-plus", "n_clicks"),
        State("focus-context-window", "value"),
        prevent_initial_call=True,
    )
    def update_focus_context_window(
        level_index: int,
        minus_clicks: int,
        plus_clicks: int,
        current_window: int | None,
    ) -> int:
        del level_index, minus_clicks, plus_clicks
        triggered = callback_context.triggered_id
        if triggered == "focus-context-level":
            return 1
        current = max(1, int(current_window or 1))
        if triggered == "focus-context-window-minus":
            return max(1, current - 1)
        if triggered == "focus-context-window-plus":
            return current + 1
        return current

    @app.callback(
        Output("focus-probabilities", "children"),
        Input("focus-recommendation-store", "data"),
        Input("focus-focal-unit-store", "data"),
        Input("focus-code-dropdown", "value"),
        Input("active-session-store", "data"),
    )
    def update_focus_probabilities(
        recommendation: dict[str, Any] | None,
        focal_unit_id: int | None,
        code_id: int | None,
        session_id: int | None,
    ) -> Any:
        data = recommendation or {}
        if not _recommendation_matches_context(
            recommendation, unit_id=focal_unit_id, code_id=code_id,
            session_id=session_id,
        ):
            return ""
        probabilities = dict(data.get("probabilities", {}))
        committee_probability = data.get("committee_probability")
        if not probabilities and committee_probability is None:
            return html.P("No model probabilities were used for this recommendation.")
        content: list[Any] = []
        if probabilities:
            rows = [
                html.Tr([html.Th(name), html.Td(f"{float(probability):.3f}")])
                for name, probability in probabilities.items()
            ]
            content.extend(
                [html.H4("Model probabilities"), html.Table(html.Tbody(rows))]
            )
        if committee_probability is not None:
            content.extend(
                [
                    html.H4("Committee probability"),
                    html.P(f"{float(committee_probability):.3f}"),
                ]
            )
        return content

    @app.callback(
        Output("focus-prediction-fits", "options"),
        Output("focus-prediction-fits", "value"),
        Input("focus-code-dropdown", "value"),
        Input("focus-model-refresh-store", "data"),
        Input("classifier-spec-refresh-store", "data"),
        State("focus-prediction-fits", "value"),
    )
    def refresh_prediction_fit_options(
        code_id: int | None,
        model_refresh: int | None,
        spec_refresh: int | None,
        current_values: list[int] | None,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        del model_refresh, spec_refresh
        if code_id is None:
            return [], []
        status_rows = project.classifier_status(code_id=int(code_id))["classifiers"]
        options = [
            {
                "label": (
                    f"{row['name']} · {row['geometry_name']} · "
                    f"{str(row['status']).replace('_', ' ').title()}"
                ),
                # Store classifier specification IDs so an explicit retrain does
                # not silently replace the user's prediction-geometry selection.
                "value": int(row["classifier_spec_id"]),
                "disabled": row["classifier_fit_id"] is None,
            }
            for row in status_rows
        ]
        valid = {
            int(option["value"])
            for option in options
            if not bool(option.get("disabled"))
        }
        selected = [
            int(value) for value in (current_values or []) if int(value) in valid
        ]
        if not selected and valid:
            selected = [
                next(
                    int(option["value"])
                    for option in options
                    if int(option["value"]) in valid
                )
            ]
        return options, selected

    @app.callback(
        Output("prediction-geometry-refresh-store", "data"),
        Output("focus-prediction-status", "children"),
        Input("focus-prediction-refresh", "n_clicks"),
        Input("focus-prediction-score-type", "value"),
        State("focus-code-dropdown", "value"),
        State("focus-prediction-fits", "value"),
        prevent_initial_call=True,
    )
    def refresh_prediction_geometry(
        clicks: int,
        score_type: str | None,
        code_id: int | None,
        classifier_spec_ids: list[int] | None,
    ) -> tuple[dict[str, Any] | Any, str]:
        del clicks
        if code_id is None:
            return no_update, "Select a code first."
        spec_ids = [int(value) for value in (classifier_spec_ids or [])]
        if not spec_ids:
            return no_update, "Select at least one trained classifier."
        status_rows = project.classifier_status(
            code_id=int(code_id), classifier_spec_ids=spec_ids
        )["classifiers"]
        missing = [
            str(row["name"])
            for row in status_rows
            if row["classifier_fit_id"] is None
        ]
        if missing:
            return no_update, "Train these classifiers first: " + ", ".join(missing) + "."
        fit_ids = [int(row["classifier_fit_id"]) for row in status_rows]
        try:
            geometry = _compact_prediction_geometry(
                project.classifier_prediction_geometry(
                    code_id=int(code_id), classifier_fit_ids=fit_ids,
                    score_type=str(score_type or "probability"),
                )
            )
        except (KeyError, ValueError) as error:
            return no_update, str(error)
        cache_key = uuid.uuid4().hex
        geometry["code_id"] = int(code_id)
        prediction_geometry_cache[cache_key] = geometry
        while len(prediction_geometry_cache) > 4:
            prediction_geometry_cache.pop(next(iter(prediction_geometry_cache)))
        method_labels = {
            "one_dimension_with_jitter": "one classifier with deterministic jitter",
            "two_classifier_scores": "two direct classifier-score axes",
            "pca": "PCA of classifier scores",
        }
        return (
            {
                "cache_key": cache_key,
                "code_id": int(code_id),
                "classifier_spec_ids": spec_ids,
                "classifier_fit_ids": fit_ids,
                "score_type": geometry["score_type"],
            },
            (
                f"Refreshed {len(geometry['points']):,} scored observations using "
                f"{method_labels[geometry['method']]} "
                f"({ 'native decision scores' if geometry['score_type'] == 'decision_score' else 'estimated probabilities' })."
            ),
        )

    @app.callback(
        Output("focus-prediction-page-number", "max"),
        Output("focus-prediction-page-number", "value", allow_duplicate=True),
        Input("focus-prediction-page-previous", "n_clicks"),
        Input("focus-prediction-page-next", "n_clicks"),
        Input("focus-prediction-page-size", "value"),
        Input("focus-prediction-display-options", "value"),
        Input("prediction-geometry-refresh-store", "data"),
        State("focus-prediction-page-number", "value"),
        prevent_initial_call=True,
    )
    def change_prediction_geometry_page(
        previous_clicks: int,
        next_clicks: int,
        page_size: Any,
        display_options: list[str] | None,
        refresh_state: dict[str, Any] | None,
        page_number: Any,
    ) -> tuple[int, int]:
        del previous_clicks, next_clicks
        geometry = _prediction_geometry_from_cache(
            prediction_geometry_cache, refresh_state
        )
        visible = _prediction_geometry_visible_points(geometry, display_options)
        size = _coerce_positive_int(page_size, default=5000)
        page_count = deterministic_page_count(len(visible), page_size=size)
        current = min(max(1, _coerce_positive_int(page_number, default=1)), page_count)
        triggered = callback_context.triggered_id
        if triggered in {
            "focus-prediction-page-size",
            "focus-prediction-display-options",
            "prediction-geometry-refresh-store",
        }:
            current = 1
        elif triggered == "focus-prediction-page-previous":
            current = max(1, current - 1)
        elif triggered == "focus-prediction-page-next":
            current = min(page_count, current + 1)
        return page_count, current

    @app.callback(
        Output("focus-prediction-map", "figure"),
        Output("focus-prediction-page-status", "children"),
        Output("prediction-geometry-page-token", "data"),
        Input("prediction-geometry-refresh-store", "data"),
        Input("focus-prediction-page-size", "value"),
        Input("focus-prediction-page-number", "value"),
        Input("focus-prediction-display-options", "value"),
        Input("focus-code-dropdown", "value"),
    )
    def render_prediction_geometry(
        refresh_state: dict[str, Any] | None,
        page_size: Any,
        page_number: Any,
        display_options: list[str] | None,
        code_id: int | None,
    ) -> tuple[Any, str, str | None]:
        if not _prediction_geometry_matches_code(refresh_state, code_id):
            return go.Figure(), "Code changed. Refresh the score view for this code.", None
        geometry = _prediction_geometry_from_cache(
            prediction_geometry_cache, refresh_state
        )
        if not geometry:
            return go.Figure(), "Refresh the score view to display observations.", None
        visible = _prediction_geometry_visible_points(geometry, display_options)
        if not visible:
            return go.Figure(), "No observations match the selected display options.", None
        size = _coerce_positive_int(page_size, default=5000)
        number = _coerce_positive_int(page_number, default=1)
        include_coded = "coded" in set(display_options or [])
        coded_positions = {
            index
            for index, point in enumerate(visible)
            if point.get("assignment") in {"positive", "negative", "unsure"}
        }
        page = deterministic_page_with_inclusions(
            len(visible),
            page_size=size,
            page_index=number - 1,
            seed=0,
            include_positions=(coded_positions if include_coded else ()),
        )
        points = [visible[int(position)] for position in page.positions]
        figure = _build_prediction_geometry_figure(
            geometry=geometry,
            points=points,
            page_index=page.page_index,
        )
        extras = max(
            0,
            len(page.positions)
            - min(size, len(visible) - page.page_index * size),
        )
        status = (
            f"Page {page.page_index + 1} of {page.page_count} · "
            f"{len(points):,} points shown"
        )
        if include_coded:
            status += f" · all {len(coded_positions):,} coded observations included"
            if extras:
                status += f" ({extras:,} added beyond the page sample)"
        token = uuid.uuid4().hex
        index = _prediction_geometry_page_index(figure)
        with prediction_page_lock:
            prediction_page_indexes[token] = (
                int(code_id), str(refresh_state["cache_key"]), index
            )
            prediction_page_order.append(token)
            while len(prediction_page_order) > 8:
                prediction_page_indexes.pop(prediction_page_order.popleft(), None)
        return figure, status, token

    @app.callback(
        Output("focus-prediction-status", "children", allow_duplicate=True),
        Input("focus-code-dropdown", "value"),
        Input("focus-model-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
        State("focus-prediction-fits", "value"),
        prevent_initial_call=True,
    )
    def mark_prediction_geometry_stale(
        code_id: int | None,
        model_refresh: int | None,
        focus_refresh: int | None,
        teaching_refresh: int | None,
        classifier_spec_ids: list[int] | None,
    ) -> Any:
        del model_refresh, focus_refresh, teaching_refresh
        if code_id is None or not classifier_spec_ids:
            return no_update
        status_rows = project.classifier_status(
            code_id=int(code_id),
            classifier_spec_ids=[int(value) for value in classifier_spec_ids],
        )["classifiers"]
        if any(row["classifier_fit_id"] is None for row in status_rows):
            return "One or more selected classifiers have not been trained."
        if any(str(row["status"]) == "stale" for row in status_rows):
            return (
                "Selected classifiers are stale because labels or settings changed. "
                "The plotted scores remain the last trained snapshot until you train and refresh."
            )
        return no_update

    @app.callback(
        Output("focal-unit-store", "data", allow_duplicate=True),
        Output("navigation-history-store", "data", allow_duplicate=True),
        Output("focus-span-selection-store", "data", allow_duplicate=True),
        Input("focus-prediction-map", "clickData"),
        State("active-session-store", "data"),
        State("focus-code-dropdown", "value"),
        State("prediction-geometry-refresh-store", "data"),
        State("focal-unit-store", "data"),
        State("navigation-history-store", "data"),
        prevent_initial_call=True,
    )
    def open_prediction_geometry_point(
        click_data: dict[str, Any] | None,
        session_id: int,
        code_id: int | None,
        prediction_refresh: dict[str, Any] | None,
        focal_unit_id: int | None,
        history: dict[str, Any] | None,
    ) -> tuple[Any, Any, Any]:
        if (
            not _prediction_geometry_matches_code(prediction_refresh, code_id)
            or not _prediction_geometry_from_cache(
                prediction_geometry_cache, prediction_refresh
            )
            or not click_data
            or not click_data.get("points")
        ):
            return no_update, no_update, no_update
        customdata = click_data["points"][0].get("customdata")
        if not isinstance(customdata, (list, tuple)) or len(customdata) < 3:
            return no_update, no_update, no_update
        observation_id = int(customdata[0])
        kind = str(customdata[2])
        if kind == "teaching_example":
            return no_update, no_update, no_update
        if kind == "span":
            observation = project.observation(observation_id)
            member_unit_ids = [int(value) for value in observation["member_unit_ids"]]
            if not member_unit_ids:
                return no_update, no_update, no_update
            unit_id = member_unit_ids[len(member_unit_ids) // 2]
            selection = member_unit_ids
        else:
            if customdata[1] is None:
                return no_update, no_update, no_update
            unit_id = int(customdata[1])
            selection = [unit_id]
        fit_ids = [
            int(value)
            for value in (prediction_refresh or {}).get("classifier_fit_ids", [])
        ]
        project.record_visit(
            int(session_id),
            unit_id,
            method="focus_prediction_geometry",
            source_unit_id=int(focal_unit_id) if focal_unit_id is not None else None,
            active_code_id=int(code_id) if code_id is not None else None,
            classifier_fit_id=(fit_ids[0] if len(fit_ids) == 1 else None),
        )
        return unit_id, _append_navigation_history(history, unit_id), selection

    @app.callback(
        Output("apply-code-dropdown", "options"),
        Output("apply-code-dropdown", "value"),
        Input("focus-refresh-store", "data"),
        State("apply-code-dropdown", "value"),
    )
    def refresh_apply_codes(
        focus_refresh: int,
        current_code_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        del focus_refresh
        code_options = _code_options(project.codes())
        valid_codes = {int(option["value"]) for option in code_options}
        code_id = (
            int(current_code_id)
            if current_code_id is not None and int(current_code_id) in valid_codes
            else (code_options[0]["value"] if code_options else None)
        )
        return code_options, code_id

    @app.callback(
        Output("apply-run-dropdown", "options"),
        Output("apply-run-dropdown", "value"),
        Input("apply-code-dropdown", "value"),
        Input("apply-refresh-store", "data"),
        State("apply-run-dropdown", "value"),
    )
    def refresh_apply_runs(
        code_id: int | None,
        apply_refresh: int,
        current_run_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        del apply_refresh
        runs = project.apply_runs(code_id=code_id) if code_id is not None else []
        reviewable_runs = [
            row
            for row in runs
            if str(row["status"]) == "draft"
            or project.can_undo_apply_commit(int(row["apply_run_id"]))
        ]
        run_options = _apply_run_options(reviewable_runs)
        valid_runs = {int(option["value"]) for option in run_options}
        run_id = (
            int(current_run_id)
            if current_run_id is not None and int(current_run_id) in valid_runs
            else (run_options[0]["value"] if run_options else None)
        )
        return run_options, run_id

    @app.callback(
        Output("apply-source-dropdown", "options"),
        Output("apply-source-dropdown", "value"),
        Input("apply-code-dropdown", "value"),
        Input("focus-model-refresh-store", "data"),
        Input("classifier-spec-refresh-store", "data"),
        Input("committee-refresh-store", "data"),
        Input("focus-refresh-store", "data"),
        Input("teaching-example-refresh-store", "data"),
        Input("workspace-tabs", "value"),
        State("apply-source-dropdown", "value"),
    )
    def refresh_apply_sources(
        code_id: int | None,
        model_refresh: int | None,
        spec_refresh: int | None,
        committee_refresh: int | None,
        focus_refresh: int | None,
        teaching_refresh: int | None,
        workspace: str | None,
        current_value: str | None,
    ) -> tuple[Any, Any]:
        del model_refresh, spec_refresh, committee_refresh, focus_refresh, teaching_refresh
        if workspace != "apply":
            return no_update, no_update
        if code_id is None:
            return [], None
        status_rows = project.classifier_status(code_id=int(code_id))["classifiers"]
        options = [
            {
                "label": (
                    f"Classifier: {row['name']} · "
                    f"{str(row['status']).replace('_', ' ')}"
                ),
                "value": f"classifier:{int(row['classifier_spec_id'])}",
            }
            for row in status_rows
            if row["classifier_fit_id"] is not None
        ]
        for committee in project.classifier_committees(int(code_id)):
            committee_status = project.classifier_committee_status(
                code_id=int(code_id), committee_id=int(committee["committee_id"])
            )
            member_counts = committee_status["member_counts"]
            members_ready = int(member_counts["not_trained"]) == 0
            aggregation = str(committee["aggregation"])
            trainable = aggregation == "logistic_stack"
            committee_ready = (
                committee_status["latest_fit"] is not None if trainable else members_ready
            )
            if committee_ready:
                stale = (
                    committee_status["status"] == "stale"
                    if trainable
                    else int(member_counts["stale"]) > 0
                )
                options.append(
                    {
                        "label": (
                            f"Committee: {committee['name']}"
                            + (" · stale" if stale else "")
                        ),
                        "value": f"committee:{int(committee['committee_id'])}",
                    }
                )
        valid = {str(option["value"]) for option in options}
        value = (
            str(current_value)
            if current_value is not None and str(current_value) in valid
            else (str(options[0]["value"]) if options else None)
        )
        return options, value

    @app.callback(
        Output("apply-status", "children", allow_duplicate=True),
        Output("apply-refresh-store", "data", allow_duplicate=True),
        Output("apply-run-dropdown", "options", allow_duplicate=True),
        Output("apply-run-dropdown", "value", allow_duplicate=True),
        Input("apply-generate", "n_clicks"),
        State("apply-code-dropdown", "value"),
        State("apply-source-dropdown", "value"),
        State("apply-threshold", "value"),
        State("apply-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def generate_apply_run(
        n_clicks: int,
        code_id: int | None,
        source_value: str | None,
        threshold: float | None,
        refresh: int | None,
    ) -> tuple[str, int, Any, Any]:
        del n_clicks
        if code_id is None:
            return "Create or select a code first.", int(refresh or 0), no_update, no_update
        if not source_value or ":" not in str(source_value):
            return "Train and select a classifier or committee first.", int(refresh or 0), no_update, no_update
        source_kind, source_id_text = str(source_value).split(":", 1)
        try:
            draft_id = project.create_apply_run(
                code_id=int(code_id),
                source_kind=source_kind,
                source_id=int(source_id_text),
                threshold=float(threshold if threshold is not None else 0.5),
            )
        except (KeyError, ValueError) as error:
            return str(error), int(refresh or 0), no_update, no_update
        draft_options = _apply_run_options(
            [
                row
                for row in project.apply_runs(code_id=int(code_id))
                if str(row["status"]) == "draft"
                or project.can_undo_apply_commit(int(row["apply_run_id"]))
            ]
        )
        return (
            f"Created draft {draft_id}.",
            int(refresh or 0) + 1,
            draft_options,
            int(draft_id),
        )

    @app.callback(
        Output("apply-page-index", "data"),
        Input("apply-page-previous", "n_clicks"),
        Input("apply-page-next", "n_clicks"),
        Input("apply-run-dropdown", "value"),
        Input("apply-proposed-filter", "value"),
        Input("apply-decision-filter", "value"),
        Input("apply-probability-range", "value"),
        Input("apply-page-size", "value"),
        Input("apply-table", "columnState"),
        State("apply-page-snapshot", "data"),
        prevent_initial_call=True,
    )
    def navigate_apply_page(
        previous: int, next_clicks: int, draft_id: int | None,
        proposed: str | None, decision: str | None, probability: list[float] | None,
        size: int | None, columns: list[dict[str, Any]] | None,
        snapshot: dict[str, Any] | None,
    ) -> int:
        del previous, next_clicks, proposed, decision, probability, size, columns
        current = dict(snapshot or {})
        if current.get("draft_id") != draft_id:
            return 0
        if callback_context.triggered_id == "apply-page-previous":
            return max(0, int(current.get("page_index", 0)) - 1)
        if callback_context.triggered_id == "apply-page-next":
            return min(int(current.get("page_count", 1)) - 1,
                       int(current.get("page_index", 0)) + 1)
        # Sorting, filtering and page-size changes always start from page one.
        return 0

    @app.callback(
        Output("apply-table", "rowData"),
        Output("apply-run-summary", "children"),
        Output("apply-page-status", "children"),
        Output("apply-page-previous", "disabled"),
        Output("apply-page-next", "disabled"),
        Output("apply-page-snapshot", "data"),
        Input("apply-run-dropdown", "value"),
        Input("apply-proposed-filter", "value"),
        Input("apply-decision-filter", "value"),
        Input("apply-probability-range", "value"),
        Input("apply-page-index", "data"),
        Input("apply-page-size", "value"),
        Input("apply-table", "columnState"),
        Input("apply-refresh-store", "data"),
    )
    def load_apply_table(
        draft_id: int | None,
        proposed_filter: str | None,
        decision_filter: str | None,
        probability_range: list[float] | None,
        page_index: int | None,
        page_size: int | None,
        column_state: list[dict[str, Any]] | None,
        refresh: int | None,
    ) -> tuple[list[dict[str, Any]], Any, str, bool, bool, dict[str, Any] | None]:
        if draft_id is None:
            return [], "No proposal draft has been generated for this code.", "", True, True, None
        lower, upper = _normalized_probability_range(probability_range)
        chosen_size = int(page_size or 25)
        if chosen_size not in {25, 50, 100}:
            chosen_size = 25
        ordering = _apply_sort_spec(column_state)
        count = project.count_apply_proposals(
            int(draft_id), proposed_label=proposed_filter or None,
            decision=decision_filter or None,
            probability_min=lower, probability_max=upper,
        )
        pages = max(1, (count + chosen_size - 1) // chosen_size)
        index = min(max(0, int(page_index or 0)), pages - 1)
        draft = project.apply_run(int(draft_id))
        proposals = project.apply_proposals(
            int(draft_id),
            proposed_label=proposed_filter or None,
            decision=decision_filter or None,
            probability_min=lower,
            probability_max=upper,
            limit=chosen_size, offset=index * chosen_size, sort_by=ordering,
        )
        data = [
            {
                "unit_id": int(row["unit_id"]),
                "apply_run_id": int(draft_id),
                "key": " › ".join(
                    f"{name}: {value}" for name, value in row["user_key"].items()
                ),
                "text": str(row["text"]),
                "probability": round(float(row["probability"]), 6),
                "proposed_label": str(row["proposed_label"]),
                "decision": APPLY_DECISION_LABELS.get(
                    str(row["decision"]), str(row["decision"])
                ),
            }
            for row in proposals
        ]
        summary = (
            f"Draft {draft['apply_run_id']} · "
            f"{draft['reviewed_count']:,} of {draft['proposal_count']:,} decided · "
            f"threshold {draft['threshold']:.3f} · {draft['aggregation']} aggregation"
        )
        if int(draft.get("unreviewed_count", 0)):
            summary += f" · {int(draft['unreviewed_count']):,} marked Leave unreviewed"
        page_status = (
            f"Page {index + 1:,} of {pages:,} · {count:,} matching proposals"
        )
        snapshot = {
            "draft_id": int(draft_id),
            "page_index": index, "page_count": pages,
            "page_size": chosen_size, "count": count,
            "unit_ids": [int(row["unit_id"]) for row in proposals],
            "proposed": proposed_filter or "", "decision": decision_filter or "",
            "range": [lower, upper],
            "sort": [list(pair) for pair in ordering],
            "refresh": int(refresh or 0),
        }
        return data, summary, page_status, index <= 0, index >= pages - 1, snapshot

    @app.callback(
        Output("apply-status", "children", allow_duplicate=True),
        Output("apply-refresh-store", "data", allow_duplicate=True),
        Input("apply-table", "cellValueChanged"),
        State("apply-run-dropdown", "value"),
        State("apply-refresh-store", "data"),
        State("apply-page-snapshot", "data"),
        State("apply-proposed-filter", "value"),
        State("apply-decision-filter", "value"),
        State("apply-probability-range", "value"),
        State("apply-page-size", "value"),
        State("apply-table", "columnState"),
        prevent_initial_call=True,
    )
    def persist_apply_table_edits(
        changed: dict[str, Any] | list[dict[str, Any]] | None,
        draft_id: int | None,
        refresh: int | None,
        snapshot: dict[str, Any] | None,
        proposed_filter: str | None,
        decision_filter: str | None,
        probability_range: list[float] | None,
        page_size: int | None,
        column_state: list[dict[str, Any]] | None,
    ) -> tuple[Any, int]:
        if not changed or draft_id is None:
            return no_update, int(refresh or 0)
        events = changed if isinstance(changed, list) else [changed]
        count = 0
        try:
            valid_ids = set(_verified_apply_page(
                project, draft_id=int(draft_id), snapshot=snapshot,
                proposed_filter=proposed_filter, decision_filter=decision_filter,
                probability_range=probability_range, page_size=page_size,
                column_state=column_state, refresh=refresh,
            ))
            for event in events:
                if str(event.get("colId", "")) != "decision":
                    continue
                row = dict(event.get("data") or {})
                if (
                    not _apply_rows_match_draft([row], int(draft_id))
                    or int(row["unit_id"]) not in valid_ids
                ):
                    return (
                        "The edited proposal is not on the current verified page.",
                        int(refresh or 0),
                    )
                decision = _normalize_apply_decision(row.get("decision"))
                project.review_apply_proposal(
                    apply_run_id=int(draft_id),
                    unit_id=int(row["unit_id"]),
                    decision=decision,
                )
                count += 1
        except (KeyError, ValueError) as error:
            return str(error), int(refresh or 0)
        if count == 0:
            return no_update, int(refresh or 0)
        return (
            f"Saved {count:,} review decision{'s' if count != 1 else ''}.",
            int(refresh or 0) + 1,
        )

    apply_action_inputs = [
        Input("apply-bulk-apply", "n_clicks"),
        Input("apply-clear-decisions", "n_clicks"),
        Input("apply-clear-visible", "n_clicks"),
        Input("apply-commit", "n_clicks"),
        Input("apply-commit-visible", "n_clicks"),
        Input("apply-undo-last-commit", "n_clicks"),
    ]

    @app.callback(
        Output("apply-status", "children", allow_duplicate=True),
        Output("apply-refresh-store", "data", allow_duplicate=True),
        Output("apply-table", "selectedRows"),
        Output("focus-auto-training-request-store", "data", allow_duplicate=True),
        *apply_action_inputs,
        State("apply-run-dropdown", "value"),
        State("apply-page-snapshot", "data"),
        State("apply-table", "selectedRows"),
        State("apply-proposed-filter", "value"),
        State("apply-decision-filter", "value"),
        State("apply-probability-range", "value"),
        State("apply-page-size", "value"),
        State("apply-table", "columnState"),
        State("apply-bulk-decision", "value"),
        State("apply-bulk-scope", "value"),
        State("apply-refresh-store", "data"),
        State("focus-code-dropdown", "value"),
        State("focus-auto-retrain", "value"),
        State("focus-train-scope", "value"),
        State("focus-tune-hyperparameters", "value"),
        State("focus-recommendation-source", "value"),
        State("focus-active-classifier-dropdown", "value"),
        State("focus-committee-dropdown", "value"),
        prevent_initial_call=True,
    )
    def apply_action(
        bulk_apply_clicks: int,
        clear_clicks: int,
        clear_visible_clicks: int,
        commit_clicks: int,
        commit_visible_clicks: int,
        undo_clicks: int,
        draft_id: int | None,
        snapshot: dict[str, Any] | None,
        selected_rows: list[dict[str, Any]] | None,
        proposed_filter: str | None,
        decision_filter: str | None,
        probability_range: list[float] | None,
        page_size: int | None,
        column_state: list[dict[str, Any]] | None,
        bulk_decision: str | None,
        bulk_scope: str | None,
        refresh: int | None,
        develop_code_id: int | None,
        auto_train_values: list[str] | None,
        train_scope_values: list[str] | None,
        tune_values: list[str] | None,
        recommendation_source: str | None,
        active_classifier_spec_id: int | None,
        committee_id: int | None,
    ) -> tuple[Any, int, Any, Any]:
        del (
            bulk_apply_clicks,
            clear_clicks,
            clear_visible_clicks,
            commit_clicks,
            commit_visible_clicks,
            undo_clicks,
        )
        if draft_id is None:
            return "Select a draft first.", int(refresh or 0), no_update, no_update
        triggered = str(callback_context.triggered_id)
        clear_selection: Any = no_update
        training_request: Any = no_update
        try:
            current_page_ids = _verified_apply_page(
                project, draft_id=int(draft_id), snapshot=snapshot,
                proposed_filter=proposed_filter, decision_filter=decision_filter,
                probability_range=probability_range, page_size=page_size,
                column_state=column_state, refresh=refresh,
            )
            if triggered == "apply-bulk-apply" and bulk_scope == "selected":
                if not _apply_rows_match_draft(selected_rows, int(draft_id)):
                    raise ValueError("Selection belongs to another Apply draft.")
                page_ids = set(current_page_ids)
                if any(
                    int(row["unit_id"]) not in page_ids
                    for row in (selected_rows or [])
                ):
                    raise ValueError(
                        "Selection changed pages. Select proposals on the displayed page."
                    )
            if triggered == "apply-bulk-apply":
                decision = _normalize_apply_decision(bulk_decision)
                unit_ids = _apply_bulk_scope_unit_ids(
                    scope=str(bulk_scope or "selected"),
                    current_page_rows=[
                        {"unit_id": unit_id} for unit_id in current_page_ids
                    ],
                    selected_rows=selected_rows or [],
                )
                if bulk_scope == "selected" and not unit_ids:
                    return (
                        "Select at least one proposal before applying a selected-row action.",
                        int(refresh or 0),
                        no_update, no_update,
                    )
                count = project.bulk_review_apply_proposals(
                    apply_run_id=int(draft_id),
                    decision=decision,
                    unit_ids=unit_ids,
                )
                scope_label = {
                    "all": "all proposals",
                    "current_page": "proposals on the current page",
                    "selected": "selected proposals",
                }.get(str(bulk_scope), "proposals")
                decision_label = APPLY_DECISION_LABELS.get(decision, decision)
                message = f"Applied {decision_label.lower()} to {count:,} {scope_label}."
                if bulk_scope == "selected":
                    clear_selection = []
            elif triggered == "apply-clear-decisions":
                count = project.bulk_review_apply_proposals(
                    apply_run_id=int(draft_id), decision="pending"
                )
                message = f"Cleared all {count:,} review decisions."
            elif triggered == "apply-clear-visible":
                count = project.bulk_review_apply_proposals(
                    apply_run_id=int(draft_id),
                    decision="pending",
                    unit_ids=current_page_ids,
                )
                message = f"Cleared {count:,} review decisions on the current page."
            elif triggered in {"apply-commit", "apply-commit-visible"}:
                current_page_only = triggered == "apply-commit-visible"
                result = project.commit_apply_run(
                    int(draft_id),
                    unit_ids=current_page_ids if current_page_only else None,
                    scope="current_page" if current_page_only else "all_reviewed",
                )
                scope_label = "current-page reviewed" if current_page_only else "reviewed"
                message = (
                    f"Processed {result['processed']:,} {scope_label} proposals: "
                    f"created {result['committed']:,} assignments"
                )
                if result["removed_unreviewed"]:
                    message += f", left {result['removed_unreviewed']:,} unreviewed"
                if result["skipped_conflicts"]:
                    message += (
                        f", skipped {result['skipped_conflicts']:,} existing assignments"
                    )
                message += f". {result['left_pending']:,} pending proposals remain."
                request, note = _post_saved_apply_batch_training_request(
                    project, apply_run_id=int(draft_id), mutation="commit",
                    changed_assignment_count=int(result["committed"]),
                    commit_operation_id=int(result["commit_operation_id"]),
                    develop_code_id=develop_code_id, auto_train_values=auto_train_values,
                    train_scope_values=train_scope_values, tune_values=tune_values,
                    recommendation_source=recommendation_source,
                    active_classifier_spec_id=active_classifier_spec_id, committee_id=committee_id,
                )
                message += f" {note}"
                if request is not None:
                    training_request = training_inbox.enqueue(request)
            elif triggered == "apply-undo-last-commit":
                result = project.undo_last_apply_commit(int(draft_id))
                message = (
                    f"Undid the last commit: restored {result['restored_proposals']:,} "
                    f"proposals and retracted {result['retracted_assignments']:,} "
                    "assignments created by that commit."
                )
                request, note = _post_saved_apply_batch_training_request(
                    project, apply_run_id=int(draft_id), mutation="undo",
                    changed_assignment_count=int(result["retracted_assignments"]),
                    commit_operation_id=int(result["commit_operation_id"]),
                    develop_code_id=develop_code_id, auto_train_values=auto_train_values,
                    train_scope_values=train_scope_values, tune_values=tune_values,
                    recommendation_source=recommendation_source,
                    active_classifier_spec_id=active_classifier_spec_id, committee_id=committee_id,
                )
                message += f" {note}"
                if request is not None:
                    training_request = training_inbox.enqueue(request)
            else:
                return no_update, int(refresh or 0), no_update, no_update
        except (KeyError, ValueError) as error:
            return str(error), int(refresh or 0), no_update, no_update
        return message, int(refresh or 0) + 1, clear_selection, training_request

    @app.callback(
        Output("apply-undo-last-commit", "disabled"),
        Input("apply-run-dropdown", "value"),
        Input("apply-refresh-store", "data"),
    )
    def update_apply_undo_availability(
        draft_id: int | None, refresh: int | None
    ) -> bool:
        del refresh
        return draft_id is None or not project.can_undo_apply_commit(int(draft_id))

    @app.callback(
        Output("apply-discard-confirm", "displayed"),
        Input("apply-discard", "n_clicks"),
        prevent_initial_call=True,
    )
    def confirm_discard_draft(clicks: int) -> bool:
        return bool(clicks)

    @app.callback(
        Output("apply-status", "children", allow_duplicate=True),
        Output("apply-refresh-store", "data", allow_duplicate=True),
        Input("apply-discard-confirm", "submit_n_clicks"),
        State("apply-run-dropdown", "value"),
        State("apply-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def discard_draft(
        submit_clicks: int | None,
        draft_id: int | None,
        refresh: int | None,
    ) -> tuple[Any, int]:
        if not submit_clicks or draft_id is None:
            return no_update, int(refresh or 0)
        try:
            project.discard_apply_run(int(draft_id))
        except (KeyError, ValueError) as error:
            return str(error), int(refresh or 0)
        return f"Discarded draft {int(draft_id)} without changing annotations.", int(refresh or 0) + 1

    @app.callback(
        Output("session-state-status", "children", allow_duplicate=True),
        Input("explore-code-palette-store", "data"),
        State("active-session-store", "data"),
        State("memo-presentation-location", "search"),
        prevent_initial_call=True,
    )
    def persist_explore_code_palette(
        palette: list[int | None] | None,
        session_id: int,
        presentation_search: str | None,
    ) -> Any:
        presentation_values = parse_qs((presentation_search or "").lstrip("?"))
        if presentation_values.get("presentation", [None])[0] == "memo":
            return no_update
        project.patch_session_state(
            int(session_id),
            {
                "explore_code_palette": _normalize_explore_code_palette(
                    palette, {int(code["code_id"]) for code in project.codes()}
                )
            },
        )
        return "saved"

    @app.callback(
        Output("session-state-status", "children"),
        Input("active-session-store", "data"),
        Input("geometry-dropdown", "value"),
        Input("view-dropdown", "value"),
        Input("focal-unit-store", "data"),
        Input("navigation-history-store", "data"),
        Input("literal-query", "value"),
        Input("literal-options", "value"),
        Input("metadata-filters-store", "data"),
        Input("metadata-filter-operator", "value"),
        Input("explore-uncoded-only", "value"),
        Input("context-level", "value"),
        Input("context-window", "value"),
        Input("page-size", "value"),
        Input("page-number", "value"),
        Input("navigation-new-only", "value"),
        Input("workspace-tabs", "value"),
        Input("focus-construct-dropdown", "value"),
        Input("focus-code-dropdown", "value"),
        Input("focus-active-classifier-dropdown", "value"),
        Input("focus-committee-dropdown", "value"),
        Input("focus-train-scope", "value"),
        Input("focus-auto-retrain", "value"),
        Input("focus-tune-hyperparameters", "value"),
        Input("focus-context-level", "value"),
        Input("focus-context-window", "value"),
        State("explore-code-palette-store", "data"),
        State("memo-presentation-location", "search"),
    )
    def persist_session_state(
        session_id: int,
        geometry_id: int | None,
        view_id: int | None,
        focal_unit_id: int | None,
        navigation_history: dict[str, Any] | None,
        literal_query: str | None,
        literal_options: list[str] | None,
        metadata_filters: list[dict[str, Any]] | None,
        metadata_filter_operator: str | None,
        explore_uncoded_values: list[str] | None,
        context_level: int,
        context_window: int,
        page_size: Any,
        page_number: Any,
        navigation_new_only: list[str] | None,
        workspace: str,
        focus_construct_id: int | None,
        focus_code_id: int | None,
        focus_classifier_spec_id: int | None,
        focus_committee_id: int | None,
        focus_train_scope_values: list[str] | None,
        focus_auto_retrain_values: list[str] | None,
        focus_tune_hyperparameter_values: list[str] | None,
        focus_context_level: int,
        focus_context_window: int,
        explore_code_palette: list[int | None] | None,
        presentation_search: str | None,
    ) -> str:
        presentation_values = parse_qs((presentation_search or "").lstrip("?"))
        if presentation_values.get("presentation", [None])[0] == "memo":
            return "Presentation mode leaves Explore session state unchanged."
        options = set(literal_options or [])
        # Persist a Code ID only when it is the selected Construct's actual Code ID.
        valid_construct = _selected_focus_construct_id(
            project.constructs(), {"focus_construct_id": focus_construct_id}
        )
        selected_code = (
            int(focus_code_id)
            if valid_construct is not None
            and focus_code_id is not None
            and int(focus_code_id) == valid_construct
            and project.construct(valid_construct)["type"] == "code"
            else None
        )
        # Preserve session keys independently owned by Focus Coding and other APIs.
        project.patch_session_state(
            int(session_id),
            {
                "geometry_id": geometry_id,
                "view_id": view_id,
                "focal_unit_id": focal_unit_id,
                "navigation_history": _normalize_navigation_history(
                    navigation_history,
                    focal_unit_id=focal_unit_id,
                ),
                "literal_query": literal_query or "",
                "case_sensitive": "case" in options,
                "regex": "regex" in options,
                "metadata_filters": _normalize_metadata_filters(
                    metadata_filters or [], metadata_profiles
                ),
                "metadata_filter_operator": _normalize_boolean_operator(
                    metadata_filter_operator
                ),
                "explore_uncoded_only": (
                    "uncoded_only" in set(explore_uncoded_values or [])
                ),
                "explore_code_palette": _normalize_explore_code_palette(
                    explore_code_palette,
                    {int(code["code_id"]) for code in project.codes()},
                ),
                "context_level": int(context_level),
                "context_window": max(1, int(context_window or 1)),
                "page_size": _coerce_positive_int(page_size, default=5000),
                "page_number": _coerce_positive_int(page_number, default=1),
                "navigation_new_only": "new_only" in (navigation_new_only or []),
                "workspace": workspace,
                "focus_construct_id": valid_construct,
                "focus_code_id": selected_code,
                "focus_classifier_spec_id": (
                    focus_classifier_spec_id if selected_code is not None else None
                ),
                "focus_committee_id": (
                    focus_committee_id if selected_code is not None else None
                ),
                "focus_train_all": "all" in set(focus_train_scope_values or []),
                "focus_auto_retrain": "auto" in set(focus_auto_retrain_values or []),
                "focus_tune_hyperparameters": (
                    "tune" in set(focus_tune_hyperparameter_values or [])
                ),
                "focus_context_level": int(focus_context_level),
                "focus_context_window": max(1, int(focus_context_window or 1)),
            },
        )
        return "saved"

    from geometric_coder.ui.analyses import register_analysis_callbacks
    register_analysis_callbacks(app, project)
    return app


def _explore_layout(
    *,
    dcc: Any,
    html: Any,
    geometries: list[dict[str, Any]],
    views: list[dict[str, Any]],
    hierarchy: list[dict[str, Any]],
    initial_state: dict[str, Any],
    initial_geometry_id: int | None,
    initial_view_id: int | None,
    initial_semantic_scores: dict[str, list[float]],
    initial_semantic_status: str,
    geometry_by_id: dict[int, dict[str, Any]],
    total_units: int,
    sessions: list[dict[str, Any]],
    initial_session_id: int,
    initial_seen_count: int,
    metadata_profiles: dict[str, dict[str, Any]],
    allow_create_view: bool,
) -> Any:
    literal_options = []
    if initial_state.get("case_sensitive"):
        literal_options.append("case")
    if initial_state.get("regex"):
        literal_options.append("regex")
    geometry_name = (
        str(geometry_by_id[initial_geometry_id]["name"])
        if initial_geometry_id is not None
        else ""
    )
    initial_scores = initial_semantic_scores.get(geometry_name)
    if initial_scores:
        semantic_min = float(np.nanmin(initial_scores))
        semantic_max = float(np.nanmax(initial_scores))
        semantic_value = initial_state.get("semantic_range") or [semantic_min, semantic_max]
        semantic_style = {"display": "block"}
    else:
        semantic_min, semantic_max = 0.0, 1.0
        semantic_value = [0.0, 1.0]
        semantic_style = {"display": "none"}
    initial_page_size = max(1, int(initial_state.get("page_size", 5000)))
    initial_page_count = max(1, (total_units + initial_page_size - 1) // initial_page_size)
    initial_page_number = min(
        initial_page_count, max(1, int(initial_state.get("page_number", 1)))
    )
    new_only = ["new_only"] if initial_state.get("navigation_new_only", True) else []
    uncoded_only = (
        ["uncoded_only"] if initial_state.get("explore_uncoded_only", False) else []
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(_tooltip_label(html, "Session", "A session records which units you've visited and the order in which you visited them."), className="control-block-title"),
                            html.Div([
                                dcc.Dropdown(
                                    id="session-dropdown",
                                    options=_session_options(sessions),
                                    value=initial_session_id,
                                    clearable=False,
                                    searchable=False,
                                ),
                                html.Button(
                                    "New session", id="new-session-button", n_clicks=0,
                                    className="secondary-action",
                                ),
                            ], className="session-select-row"),
                            html.Div(
                                f"Visited {int(initial_seen_count):,} of {int(total_units):,}",
                                id="session-progress",
                                className="session-progress",
                            ),
                        ],
                        className="control-block session-control-block",
                    ),
                    html.Div(
                        [
                            html.H3(_tooltip_label(html, "Literal Search", "Find units containing a specified substring or matching a regular expression."), className="control-block-title"),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="literal-query",
                                        type="text",
                                        value=initial_state.get("literal_query", ""),
                                        placeholder="Substring or regular expression",
                                        debounce=True,
                                    ),
                                    dcc.Checklist(
                                        id="literal-options",
                                        options=[
                                            {"label": "Case sensitive", "value": "case"},
                                            {"label": "Regex", "value": "regex"},
                                        ],
                                        value=literal_options,
                                        inline=True,
                                    ),
                                    html.Div(
                                        id="literal-search-status",
                                        className="control-status",
                                    ),
                                ],
                                className="search-section",
                            ),
                        ],
                        className="control-block keyword-search-control-block",
                    ),
                    html.Div(
                        [
                            html.H3(_tooltip_label(html, "Semantic search", "Calculate similarity between search text and other units using the selected geometry. Requires support for new translations."), className="control-block-title"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dcc.Input(
                                                id="semantic-query",
                                                type="text",
                                                debounce=False,
                                                value="",
                                                placeholder="Describe what you want to find",
                                            ),
                                            html.Button("Search", id="semantic-search-button", n_clicks=0),
                                            html.Button("Load", id="semantic-load-open", n_clicks=0),
                                        ],
                                        className="inline-controls semantic-query-row",
                                    ),
                                    dcc.Loading(
                                        id="semantic-search-loading",
                                        type="circle",
                                        children=html.Div(
                                            [
                                                html.Div(
                                                    id="semantic-search-status",
                                                    children=initial_semantic_status,
                                                    className="control-status",
                                                ),
                                                html.Div(
                                                    dcc.RangeSlider(
                                                        id="semantic-range-slider",
                                                        min=semantic_min,
                                                        max=semantic_max,
                                                        step=max(
                                                            (semantic_max - semantic_min)
                                                            / 1000,
                                                            1e-9,
                                                        ),
                                                        value=semantic_value,
                                                        marks=(
                                                            {
                                                                semantic_min: f"{semantic_min:.3f}",
                                                                semantic_max: f"{semantic_max:.3f}",
                                                            }
                                                            if semantic_style.get("display")
                                                            != "none"
                                                            else {}
                                                        ),
                                                        allowCross=False,
                                                        allow_direct_input=False,
                                                    ),
                                                    id="semantic-range-wrap",
                                                    className="semantic-range-wrap",
                                                    style=semantic_style,
                                                ),

                                            ]
                                        ),
                                    ),

                                ],
                                className="search-section semantic-search-section",
                            ),
                        ],
                        className="control-block semantic-search-control-block",
                    ),
                    html.Div(
                        [
                            html.H3("Geometry and View", className="control-block-title"),
                            html.Label(_tooltip_label(html, "Geometry", "A geometry represents units of analysis in a numerical space, allowing their relationships to be measured.")),
                            dcc.Dropdown(
                                id="geometry-dropdown",
                                options=[
                                    {
                                        "label": record["name"],
                                        "value": int(record["geometry_id"]),
                                    }
                                    for record in geometries
                                ],
                                value=initial_geometry_id,
                                clearable=False,
                            ),
                            html.Label(_tooltip_label(html, "View", "A view displays a geometry in one or two dimensions without changing the underlying geometry used for analysis."), className="stacked-control-label"),
                            dcc.Dropdown(
                                id="view-dropdown",
                                options=[
                                    {
                                        "label": view["name"],
                                        "value": int(view["view_id"]),
                                    }
                                    for view in views
                                ],
                                value=initial_view_id,
                                clearable=False,
                                searchable=False,
                            ),
                            html.Button(
                                "Create 2D view",
                                id="projection-open-button",
                                n_clicks=0,
                                className="secondary-action",
                                disabled=not allow_create_view,
                                style={} if allow_create_view else {"display": "none"},
                                title=(
                                    None
                                    if allow_create_view
                                    else "Use Manage Representations for TeAL-backed 2D views."
                                ),
                            ),
                            html.Div([
                                html.Label("1D jitter strength"),
                                dcc.Slider(
                                    id="one-dimensional-jitter", min=0, max=3, step=0.25,
                                    value=1, marks={0: "Off", 1: "1×", 2: "2×", 3: "3×"},
                                    updatemode="mouseup",
                                ),
                            ], id="one-dimensional-jitter-wrap",
                               style={"display": "none"},
                               className="one-dimensional-jitter-control"),
                            html.Button("Manage Representations", id="representation-manager-open", n_clicks=0, className="secondary-action"),
                        ],
                        className="control-block geometry-view-control-block",
                    ),
                    html.Div(
                        [
                            html.H3(_tooltip_label(html, "Filters", "Choose which units appear in Explore using metadata, optionally restricting the display to uncoded units."), className="control-block-title"),
                            html.Div(
                                id="active-metadata-filter-chips",
                                className="metadata-filter-chip-box",
                            ),
                            dcc.RadioItems(
                                id="metadata-filter-operator",
                                options=[
                                    {"label": "AND", "value": "and"},
                                    {"label": "OR", "value": "or"},
                                    {"label": "XOR", "value": "xor"},
                                ],
                                value=str(
                                    initial_state.get("metadata_filter_operator", "and")
                                ),
                                inline=True,
                                className="metadata-filter-operator",
                            ),
                            dcc.Checklist(
                                id="explore-uncoded-only",
                                options=[
                                    {
                                        "label": "Show only uncoded points",
                                        "value": "uncoded_only",
                                    }
                                ],
                                value=uncoded_only,
                                className="explore-uncoded-filter",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Add",
                                        id="metadata-filter-add-button",
                                        n_clicks=0,
                                        disabled=not bool(metadata_profiles),
                                    ),
                                    html.Button(
                                        "Clear",
                                        id="metadata-filter-clear-button",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Save",
                                        id="metadata-filter-save-open-button",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Load",
                                        id="metadata-filter-load-open-button",
                                        n_clicks=0,
                                    ),
                                ],
                                className="metadata-filter-actions",
                            ),
                            html.Div(id="metadata-filter-status", className="control-status"),
                        ],
                        className="control-block metadata-filter-block",
                    ),
                    html.Div(
                        [
                            html.H3(_tooltip_label(html, "Topic Models", "Explore topics represented by a topic model. Color units by their dominant topic or by their association with a selected topic."), className="control-block-title"),
                            dcc.Dropdown(id="topic-model-selector", clearable=True, placeholder="Select a Topic Model"),
                            html.Div([html.Label("Selected topic"), dcc.Dropdown(id="topic-component-selector", clearable=False)], id="topic-component-wrap"),
                            dcc.RadioItems(id="topic-display-mode", options=[
                                {"label": "Dominant topic", "value": "dominant"},
                                {"label": "Selected topic weight", "value": "intensity"},
                            ], value="intensity" if initial_state.get("topic_color_mode") == "intensity" else "dominant"),
                            html.Button("Manage Topic Models", id="topic-manager-open", n_clicks=0, className="secondary-action"),

                        ],
                        className="control-block",
                    ),
                ],
                className="explore-controls",
            ),
            html.Div(
                [
                    html.Button("← Back", id="nav-back", n_clicks=0, disabled=True),
                    html.Button("Forward →", id="nav-forward", n_clicks=0, disabled=True),
                    html.Span(className="navigation-divider"),
                    html.Button("Nearest", id="nav-nearest", n_clicks=0),
                    html.Button("Farthest", id="nav-farthest", n_clicks=0),
                    html.Button(
                        "Most similar",
                        id="nav-most-similar",
                        n_clicks=0,
                        className="semantic-navigation-button",
                        style=(
                            {"display": "inline-block"}
                            if initial_semantic_scores
                            else {"display": "none"}
                        ),
                    ),
                    html.Button(
                        "Least similar",
                        id="nav-least-similar",
                        n_clicks=0,
                        className="semantic-navigation-button",
                        style=(
                            {"display": "inline-block"}
                            if initial_semantic_scores
                            else {"display": "none"}
                        ),
                    ),
                    html.Button("Random", id="nav-random", n_clicks=0),
                    dcc.Checklist(
                        id="navigation-new-only",
                        options=[{"label": _tooltip_label(html, "New points only", "Exclude units already visited in this session when using the navigation buttons."), "value": "new_only"}],
                        value=new_only,
                        inline=True,
                        className="navigation-repeat-control",
                    ),
                    html.Button("Clear colors", id="nav-clear-colors", n_clicks=0, style={"display": "none"}),
                    html.Div(id="navigation-status", className="sr-only"),
                ],
                className="navigation-bar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                id="geometry-map",
                                figure=_empty_figure("Select or compute a geometry view."),
                                config={
                                    "displaylogo": False,
                                    "scrollZoom": True,
                                    "doubleClick": "reset",
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                },
                                clear_on_unhover=True,
                                className="geometry-map",
                            ),
                            html.Div(id="topic-color-legend", className="map-encoding-key"),
                            html.Div(
                                [
                                    html.Span(className="visited-outline-swatch"),
                                    html.Span("Visited in this session"),
                                    html.Span(className="focal-outline-swatch"),
                                    html.Span("Current focal unit"),
                                ],
                                className="map-encoding-key",
                            ),
                            html.Div(
                                [
                                    html.Label("Points per page", htmlFor="page-size"),
                                    dcc.Input(
                                        id="page-size",
                                        type="text",
                                        inputMode="numeric",
                                        debounce=True,
                                        value=str(initial_page_size),
                                        className="page-size-input",
                                    ),
                                    html.Button("Previous", id="page-previous", n_clicks=0),
                                    html.Label("Page", htmlFor="page-number"),
                                    dcc.Input(
                                        id="page-number",
                                        type="number",
                                        min=1,
                                        step=1,
                                        debounce=True,
                                        value=initial_page_number,
                                        max=initial_page_count,
                                        className="page-number-input",
                                    ),
                                    html.Button("Next", id="page-next", n_clicks=0),
                                    html.Div(id="page-status", className="page-status"),
                                ],
                                className="pagination-bar",
                            ),
                        ],
                        className="map-pane",
                    ),
                    html.Aside(
                        [
                            html.Div(id="focal-key", className="focal-key inspection-key"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Context level"),
                                            dcc.Dropdown(
                                                id="context-level",
                                                options=[
                                                    {
                                                        "label": level["column_name"],
                                                        "value": int(level["level_index"]),
                                                    }
                                                    for level in hierarchy
                                                ],
                                                value=int(
                                                    initial_state.get(
                                                        "context_level",
                                                        hierarchy[-1]["level_index"],
                                                    )
                                                ),
                                                clearable=False,
                                                searchable=False,
                                            ),
                                        ],
                                        className="context-control context-level-control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Window"),
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "−",
                                                        id="context-window-minus",
                                                        n_clicks=0,
                                                    ),
                                                    dcc.Input(
                                                        id="context-window",
                                                        type="text",
                                                        inputMode="numeric",
                                                        value=str(
                                                            max(
                                                                1,
                                                                int(
                                                                    initial_state.get(
                                                                        "context_window", 1
                                                                    )
                                                                ),
                                                            )
                                                        ),
                                                    ),
                                                    html.Button(
                                                        "+",
                                                        id="context-window-plus",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                                className="window-stepper",
                                            ),
                                        ],
                                        className="context-control context-window-control",
                                    ),
                                ],
                                className="context-controls",
                            ),
                            html.Div(
                                [
                                    html.Div(id="context-content", className="context-content"),
                                    html.Details(
                                        [
                                            html.Summary("Metadata"),
                                            html.Div(
                                                id="metadata-content",
                                                className="metadata-content",
                                            ),
                                        ],
                                        open=False,
                                    ),
                                    html.Details(
                                        [
                                            html.Summary("Topic Model interpretation"),
                                            html.Div(id="topic-focal-key"),
                                            html.Div(id="topic-memberships"),
                                            html.Div(id="topic-descriptors"),
                                        ],
                                        id="topic-interpretation-details",
                                        style={"display": "none"},
                                    ),
                                ],
                                className="inspection-scroll",
                            ),
                        ],
                        className="reading-pane inspection-pane",
                    ),
                ],
                className="explore-main",
            ),
            html.Div(
                [
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.H3("Memos"),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="memo-dropdown",
                                                options=[],
                                                value=None,
                                                placeholder="Select a memo",
                                                clearable=True,
                                            ),
                                            html.Button(
                                                "New memo", id="memo-new-button", n_clicks=0
                                            ),
                                        ],
                                        className="memo-toolbar",
                                    ),
                                ],
                                className="panel-heading",
                            ),
                            dcc.Input(
                                id="memo-title",
                                type="text",
                                placeholder="Memo title",
                                className="text-input",
                            ),
                            dcc.Textarea(
                                id="memo-body",
                                placeholder=(
                                    "Write in Markdown. Use #hashtags or insert a "
                                    "reference to the focal unit."
                                ),
                                className="memo-editor",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Insert reference to focal unit",
                                        id="memo-insert-reference-button",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Save memo",
                                        id="memo-save-button",
                                        n_clicks=0,
                                        className="primary-action",
                                    ),
                                    html.Div(id="memo-status", className="control-status"),
                                ],
                                className="memo-actions",
                            ),
                            html.Div(
                                [
                                    html.Div("Preview", className="memo-preview-label"),
                                    html.Div(id="memo-preview", className="memo-preview"),
                                ]
                            ),
                        ],
                        className="analysis-panel memo-panel",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.H3("Coding"),
                                    html.Button(
                                        "New Code",
                                        id="explore-new-code-button",
                                        n_clicks=0,
                                    ),
                                ],
                                className="coding-panel-heading",
                            ),
                            html.Div(
                                id="explore-code-palette",
                                className="explore-code-palette",
                            ),
                            html.Button(
                                "+ Add code",
                                id="explore-add-code-row",
                                n_clicks=0,
                                className="explore-add-code-row",
                            ),
                            html.Div(id="explore-code-status", className="sr-only"),
                        ],
                        className="analysis-panel coding-panel",
                    ),
                ],
                className="explore-analysis-row",
            ),
            _projection_modal(dcc=dcc, html=html),
        ],
        className="explore-workspace",
    )


def _memo_center_layout(*, dcc: Any, html: Any, dash_table: Any) -> Any:
    """Return the versioned project-wide memo workspace."""
    return html.Div(
        [
            html.Aside(
                [
                    html.H2("Memo Center"),
                    html.Label("Search memos"),
                    dcc.Input(
                        id="memo-center-query", type="text",
                        placeholder="Search memo titles or contents", debounce=True,
                        className="text-input",
                    ),
                    dcc.RadioItems(
                        id="memo-center-scope",
                        options=[
                            {"label": "Full memo", "value": "full"},
                            {"label": "Titles only", "value": "title"},
                        ],
                        value="full", inline=True, className="memo-center-scope",
                    ),
                    html.Label("Hashtags", className="memo-center-filter-label"),
                    dcc.Dropdown(
                        id="memo-center-tags", options=[], value=[], multi=True,
                        placeholder="Filter by hashtag",
                    ),
                    dcc.RadioItems(
                        id="memo-center-tag-operator",
                        options=[
                            {"label": "AND", "value": "and"},
                            {"label": "OR", "value": "or"},
                            {"label": "XOR", "value": "xor"},
                        ],
                        value="and", inline=True, className="memo-center-operator",
                    ),
                    html.Div(id="memo-center-status", className="control-status"),
                    dash_table.DataTable(
                        id="memo-center-table",
                        columns=[
                            {"name": "Title", "id": "title"},
                            {"name": "Hashtags", "id": "hashtags"},
                            {"name": "Updated", "id": "updated_at"},
                        ],
                        data=[], row_selectable="single", selected_rows=[], page_size=20,
                        style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
                        style_table={"overflowX": "auto"},
                    ),
                ],
                className="memo-center-sidebar",
            ),
            html.Main(
                [
                    html.Div(
                        [
                            html.H2(id="memo-center-title", children="Select a memo"),
                            html.Div(
                                [
                                    html.A(
                                        "Presentation mode",
                                        id="memo-presentation-link",
                                        href=None,
                                        target="_blank",
                                        className="memo-presentation-link disabled",
                                        title="Open this memo with a minimal interactive map",
                                    ),
                                    html.Button("Edit", id="memo-center-edit-button", n_clicks=0, disabled=True),
                                ],
                                className="memo-center-heading-actions",
                            ),
                        ],
                        className="center-heading-row",
                    ),
                    html.Div(
                        [
                            html.Button("←", id="memo-version-previous", n_clicks=0, disabled=True),
                            html.Span("No version selected", id="memo-version-label"),
                            html.Button("→", id="memo-version-next", n_clicks=0, disabled=True),
                        ],
                        className="version-navigation",
                    ),
                    html.Div(
                        id="memo-center-preview",
                        className="memo-center-preview memo-preview",
                    ),
                    html.Div(
                        [
                            html.Label("Title"),
                            dcc.Input(id="memo-center-edit-title", type="text", className="text-input"),
                            html.Label("Markdown", className="memo-center-editor-label"),
                            dcc.Textarea(id="memo-center-edit-body", className="memo-editor memo-center-editor"),
                            html.Div(
                                [
                                    html.Button("Cancel", id="memo-center-cancel-button", n_clicks=0),
                                    html.Button("Save new version", id="memo-center-save-button", n_clicks=0, className="primary-action"),
                                ],
                                className="memo-actions",
                            ),
                            html.Div(id="memo-center-edit-status", className="control-status"),
                            html.Div(id="memo-center-edit-preview", className="memo-preview"),
                        ],
                        id="memo-center-editor-wrap",
                        className="memo-center-editor-wrap",
                        style={"display": "none"},
                    ),
                ],
                className="memo-center-main",
            ),
        ],
        className="memo-center-workspace",
    )


def _memo_presentation_layout(
    *,
    dcc: Any,
    html: Any,
    views: list[dict[str, Any]],
    geometry_by_id: dict[int, dict[str, Any]],
    initial_view_id: int | None,
    total_units: int,
) -> Any:
    """Return the minimal memo-and-map presentation surface."""
    public_views = [
        view for view in views if int(view["geometry_id"]) in geometry_by_id
    ]
    view_options = _presentation_view_options(
        public_views, list(geometry_by_id.values())
    )
    selected_view = (
        int(initial_view_id)
        if initial_view_id is not None
        and any(int(view["view_id"]) == int(initial_view_id) for view in public_views)
        else (int(public_views[0]["view_id"]) if public_views else None)
    )
    initial_page_count = deterministic_page_count(total_units, page_size=5000)
    return html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.Span("🦎", className="memo-presentation-gecko", **{"aria-hidden": "true"}),
                            html.Div(
                                [
                                    html.H1(id="memo-presentation-title", children="Memo presentation"),
                                    html.Div(id="memo-presentation-version", className="memo-presentation-version"),
                                ]
                            ),
                        ],
                        className="memo-presentation-brand",
                    ),
                    html.Div(
                        "Presentation mode · exploratory evidence view",
                        className="memo-presentation-mode-label",
                    ),
                ],
                className="memo-presentation-header",
            ),
            html.Div(
                [
                    html.Section(
                        html.Div(
                            id="memo-presentation-body",
                            className="memo-preview memo-presentation-body",
                        ),
                        className="memo-presentation-memo-pane",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("View", htmlFor="memo-presentation-view"),
                                            dcc.Dropdown(
                                                id="memo-presentation-view",
                                                options=view_options,
                                                value=selected_view,
                                                clearable=False,
                                                searchable=False,
                                            ),
                                        ],
                                        className="memo-presentation-control",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Saved filter", htmlFor="memo-presentation-saved-filter"),
                                            dcc.Dropdown(
                                                id="memo-presentation-saved-filter",
                                                options=[{"label": "All observations", "value": "__all__"}],
                                                value="__all__",
                                                clearable=False,
                                            ),
                                        ],
                                        className="memo-presentation-control",
                                    ),
                                ],
                                className="memo-presentation-controls",
                            ),
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="memo-presentation-map",
                                        figure=_empty_figure("Choose a memo and view."),
                                        config={
                                            "displaylogo": False,
                                            "scrollZoom": True,
                                            "doubleClick": "reset",
                                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                        },
                                        clear_on_unhover=True,
                                        className="memo-presentation-map",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Points per page", htmlFor="memo-presentation-page-size"),
                                            dcc.Input(
                                                id="memo-presentation-page-size",
                                                type="text",
                                                inputMode="numeric",
                                                debounce=True,
                                                value="5000",
                                                className="page-size-input",
                                            ),
                                            html.Button("Previous", id="memo-presentation-page-previous", n_clicks=0),
                                            html.Label("Page", htmlFor="memo-presentation-page-number"),
                                            dcc.Input(
                                                id="memo-presentation-page-number",
                                                type="number",
                                                min=1,
                                                step=1,
                                                debounce=True,
                                                value=1,
                                                max=initial_page_count,
                                                className="page-number-input",
                                            ),
                                            html.Button("Next", id="memo-presentation-page-next", n_clicks=0),
                                            html.Div(id="memo-presentation-page-status", className="page-status"),
                                        ],
                                        className="pagination-bar memo-presentation-pagination",
                                    ),
                                ],
                                className="memo-presentation-map-wrap",
                            ),
                            html.Aside(
                                [
                                    html.Div(
                                        id="memo-presentation-focal-key",
                                        className="focal-key inspection-key",
                                        children="No focal observation",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                id="memo-presentation-text",
                                                className="context-content memo-presentation-text",
                                                children="Select a linked observation in the memo or click a point on the map.",
                                            ),
                                            html.Details(
                                                [
                                                    html.Summary("Metadata"),
                                                    html.Div(
                                                        id="memo-presentation-metadata",
                                                        className="metadata-content",
                                                    ),
                                                ],
                                                open=False,
                                            ),
                                        ],
                                        className="inspection-scroll",
                                    ),
                                ],
                                className="reading-pane inspection-pane memo-presentation-reader",
                            ),
                        ],
                        className="memo-presentation-evidence-pane",
                    ),
                ],
                className="memo-presentation-grid",
            ),
        ],
        id="memo-presentation-shell",
        className="memo-presentation-shell",
    )



def _code_center_layout(
    *, dcc: Any, html: Any, dash_table: Any, geometries: list[dict[str, Any]],
    views: list[dict[str, Any]], initial_geometry_id: int | None, initial_view_id: int | None,
    total_units: int,
) -> Any:
    """Return the versioned Code Center with list and geometric views."""
    initial_page_count = deterministic_page_count(total_units, page_size=5000)
    return html.Div(
        [
            html.Aside(
                [
                    html.H2("Constructs"),
                    html.Button("New Code", id="code-center-new-code-button", n_clicks=0),
                    html.Details([
                        html.Summary("New Scale"),
                        html.Label("Scale name"),
                        dcc.Input(id="code-center-new-scale-name", type="text", className="text-input"),
                        html.Label("Description"),
                        dcc.Textarea(id="code-center-new-scale-description", className="memo-editor code-center-editor"),
                        html.Button("Create Scale", id="code-center-create-scale", n_clicks=0, className="primary-action"),
                        html.Div(id="code-center-create-scale-status", className="control-status"),
                    ]),
                    html.Label("Search Constructs"),
                    dcc.Input(id="code-center-query", type="text", debounce=True, className="text-input", placeholder="Search labels or descriptions"),
                    dcc.RadioItems(
                        id="code-center-scope",
                        options=[{"label": "Label and description", "value": "full"}, {"label": "Labels only", "value": "label"}],
                        value="full", inline=True, className="memo-center-scope",
                    ),
                    html.Label("Hashtags", className="memo-center-filter-label"),
                    dcc.Dropdown(id="code-center-tags", options=[], value=[], multi=True, placeholder="Filter by hashtag"),
                    html.Div(id="code-center-status", className="control-status"),
                    dash_table.DataTable(
                        id="code-center-table",
                        columns=[{"name": "Construct", "id": "name"}, {"name": "Type", "id": "type"}, {"name": "Hashtags", "id": "hashtags"}],
                        data=[], row_selectable="single", selected_rows=[], page_size=20,
                        style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
                        style_table={"overflowX": "auto"},
                    ),
                ],
                className="code-center-sidebar",
            ),
            html.Main(
                [
                    html.Div(
                        [
                            html.Div([
                                html.H2(id="code-center-title", children="Select a Construct"),
                                html.Div(id="code-center-type-label", className="focus-counts"),
                            ]),
                            html.Div(
                                [
                                    html.Button(
                                        "Teaching Examples",
                                        id="code-center-teaching-button",
                                        n_clicks=0,
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "Edit",
                                        id="code-center-edit-button",
                                        n_clicks=0,
                                        disabled=True,
                                    ),
                                ],
                                className="center-heading-actions",
                            ),
                        ],
                        className="center-heading-row",
                    ),
                    html.Div(
                        [
                            html.Button("←", id="code-version-previous", n_clicks=0, disabled=True),
                            html.Span("No version selected", id="code-version-label"),
                            html.Button("→", id="code-version-next", n_clicks=0, disabled=True),
                        ],
                        className="version-navigation",
                    ),
                    html.Div(id="code-center-description", className="memo-preview code-description-preview"),
                    html.Div(id="code-center-eligibility", className="focus-counts"),
                    html.Div(
                        [
                            html.Label("Construct name", id="code-center-edit-name-label"),
                            dcc.Input(id="code-center-edit-name", type="text", className="text-input code-name-input"),
                            html.Label("Description", className="memo-center-editor-label"),
                            dcc.Textarea(id="code-center-edit-description", className="memo-editor code-center-editor"),
                            html.Div(
                                [
                                    html.Button("Cancel", id="code-center-cancel-button", n_clicks=0),
                                    html.Button("Save new version", id="code-center-save-button", n_clicks=0, className="primary-action"),
                                ], className="memo-actions",
                            ),
                            html.Div(id="code-center-edit-status", className="control-status"),
                            html.Div(id="code-center-edit-preview", className="memo-preview"),
                        ],
                        id="code-center-editor-wrap", style={"display": "none"},
                    ),
                    html.Div(
                        [
                        dcc.Tabs(
                        id="code-center-view-tabs", value="list", className="code-center-content-tabs",
                        children=[
                            dcc.Tab(
                                label="Positive units", value="list",
                                children=dash_table.DataTable(
                                    id="code-center-units-table",
                                    columns=[
                                        {"name": "Type", "id": "kind"},
                                        {"name": "Key", "id": "key"},
                                        {"name": "Text", "id": "text"},
                                    ],
                                    data=[], page_size=25,
                                    style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
                                    style_table={"overflowX": "auto"},
                                ),
                            ),
                            dcc.Tab(
                                label="Geometric view", value="geometry",
                                children=html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div([html.Label("Geometry"), dcc.Dropdown(
                                                    id="code-center-geometry", options=[{"label": g["name"], "value": int(g["geometry_id"])} for g in geometries],
                                                    value=initial_geometry_id, clearable=False,
                                                )]),
                                                html.Div([html.Label("2D view"), dcc.Dropdown(
                                                    id="code-center-view", options=[{"label": v["name"], "value": int(v["view_id"])} for v in views],
                                                    value=initial_view_id, clearable=False,
                                                )]),
                                                html.Div(
                                                    dcc.Checklist(
                                                        id="code-center-include-positives",
                                                        options=[{"label": "Always show all positive units", "value": "include"}],
                                                        value=["include"],
                                                        className="code-center-positive-toggle",
                                                    )
                                                ),
                                            ], className="code-center-geometry-controls",
                                        ),
                                        dcc.Graph(
                                            id="code-center-map",
                                            figure=_code_center_placeholder_figure(),
                                            config={
                                                "displaylogo": False,
                                                "scrollZoom": True,
                                                "doubleClick": "reset",
                                                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                            },
                                            className="code-center-map",
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Points per page", htmlFor="code-center-page-size"),
                                                dcc.Input(
                                                    id="code-center-page-size",
                                                    type="text",
                                                    inputMode="numeric",
                                                    debounce=True,
                                                    value="5000",
                                                    className="page-size-input",
                                                ),
                                                html.Button("Previous", id="code-center-page-previous", n_clicks=0),
                                                html.Label("Page", htmlFor="code-center-page-number"),
                                                dcc.Input(
                                                    id="code-center-page-number",
                                                    type="number",
                                                    min=1,
                                                    step=1,
                                                    debounce=True,
                                                    value=1,
                                                    max=initial_page_count,
                                                    className="page-number-input",
                                                ),
                                                html.Button("Next", id="code-center-page-next", n_clicks=0),
                                                html.Div(id="code-center-page-status", className="page-status"),
                                            ],
                                            className="pagination-bar code-center-pagination",
                                        ),
                                    ], className="code-center-geometry-pane",
                                ),
                            ),
                        ],
                    ),
                    html.Section(
                        [
                            html.H3("Selected observation"),
                            html.Div(
                                id="code-center-unit-preview",
                                className="memo-preview code-center-unit-preview",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Present",
                                        id="code-center-assignment-positive",
                                        n_clicks=0,
                                        className="judgment-button positive-button",
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "Absent",
                                        id="code-center-assignment-negative",
                                        n_clicks=0,
                                        className="judgment-button negative-button",
                                        disabled=True,
                                    ),
                                    html.Button(
                                        "Unsure",
                                        id="code-center-assignment-unsure",
                                        n_clicks=0,
                                        className="judgment-button unsure-button",
                                        disabled=True,
                                    ),
                                ],
                                className="code-center-assignment-buttons",
                            ),
                            html.Div(
                                id="code-center-assignment-status",
                                className="control-status",
                            ),
                        ],
                        className="code-center-assignment-panel",
                    ),
                        ],
                        id="code-center-binary-content",
                    ),
                ],
                className="code-center-main",
            ),
        ],
        className="code-center-workspace",
    )

def _teaching_example_modal(*, dcc: Any, html: Any, dash_table: Any) -> Any:
    """Return the teaching-example manager for the selected code."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Teaching Examples"),
                            html.Button(
                                "×", id="teaching-example-close", className="modal-close-icon", title="Close", **{"aria-label": "Close"}, n_clicks=0
                            ),
                        ],
                        className="modal-heading-row",
                    ),
                    html.P(
                        "Teaching examples are researcher-authored positive or negative "
                        "examples. Active examples receive the same classifier weight as "
                        "coded corpus observations.",
                        className="focus-help",
                    ),
                    html.Label("Example text"),
                    dcc.Textarea(
                        id="teaching-example-text",
                        className="memo-editor teaching-example-text",
                        placeholder="Write an example that resembles the empirical material.",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Label"),
                                    dcc.RadioItems(
                                        id="teaching-example-label",
                                        options=[
                                            {"label": "Present", "value": "positive"},
                                            {"label": "Absent", "value": "negative"},
                                        ],
                                        value="positive",
                                        inline=True,
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Optional note"),
                                    dcc.Input(
                                        id="teaching-example-note",
                                        type="text",
                                        className="text-input",
                                    ),
                                ]
                            ),
                        ],
                        className="teaching-example-fields",
                    ),
                    html.Button(
                        "Add active example",
                        id="teaching-example-save",
                        n_clicks=0,
                        className="primary-action",
                    ),
                    dcc.Checklist(
                        id="teaching-example-assess-before-training",
                        options=[
                            {
                                "label": "Assess current classifiers before adding this example to training",
                                "value": "assess",
                            }
                        ],
                        value=[],
                    ),
                    html.Div(id="teaching-example-status", className="control-status"),
                    html.H4("Existing examples"),
                    dash_table.DataTable(
                        id="teaching-example-table",
                        columns=[
                            {"name": "Text", "id": "text"},
                            {"name": "Label", "id": "label"},
                            {"name": "Status", "id": "status"},
                            {"name": "Created", "id": "created"},
                        ],
                        data=[],
                        row_selectable="single",
                        selected_rows=[],
                        page_size=10,
                        style_cell={
                            "whiteSpace": "normal",
                            "height": "auto",
                            "textAlign": "left",
                        },
                        style_table={"overflowX": "auto"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Change status"),
                                    dcc.Dropdown(
                                        id="teaching-example-status-action",
                                        options=[
                                            {"label": "Active", "value": "active"},
                                            {"label": "Inactive", "value": "inactive"},
                                            {"label": "Delete", "value": "deleted"},
                                        ],
                                        value="inactive",
                                        clearable=False,
                                    ),
                                    html.Button(
                                        "Apply status",
                                        id="teaching-example-apply-status",
                                        n_clicks=0,
                                    ),
                                ],
                                className="teaching-example-action-group",
                            ),
                            html.Div(
                                [
                                    html.Label("Change label"),
                                    dcc.Dropdown(
                                        id="teaching-example-label-action",
                                        options=[
                                            {"label": "Present", "value": "positive"},
                                            {"label": "Absent", "value": "negative"},
                                        ],
                                        value="positive",
                                        clearable=False,
                                    ),
                                    html.Button(
                                        "Apply label",
                                        id="teaching-example-apply-label",
                                        n_clicks=0,
                                    ),
                                ],
                                className="teaching-example-action-group",
                            ),
                        ],
                        className="teaching-example-actions",
                    ),
                ],
                className="modal-card teaching-example-modal-card",
            )
        ],
        id="teaching-example-modal",
        className="modal-overlay",
        style={"display": "none"},
    )



def _classifier_spec_modal(*, dcc: Any, html: Any, geometries: list[dict[str, Any]]) -> Any:
    """Return the classifier-specification creation modal."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("New classifier"),
                            html.Button(
                                "×",
                                id="classifier-spec-close",
                                n_clicks=0,
                                className="modal-close-icon",
                                title="Close",
                                **{"aria-label": "Close"},
                            ),
                        ],
                        className="modal-heading-row",
                    ),
                    html.Label("Classifier name"),
                    dcc.Input(
                        id="classifier-spec-name",
                        type="text",
                        className="text-input",
                        placeholder="Optional; GeCo will generate a descriptive name",
                    ),
                    html.Label("Parent geometry"),
                    dcc.Dropdown(
                        id="classifier-spec-geometry",
                        options=[
                            {
                                "label": str(row["name"]),
                                "value": int(row["geometry_id"]),
                            }
                            for row in geometries
                        ],
                        value=(int(geometries[0]["geometry_id"]) if geometries else None),
                        clearable=False,
                    ),
                    html.Label("Algorithm"),
                    dcc.Dropdown(
                        id="classifier-spec-algorithm",
                        options=[
                            {
                                "label": classifier_algorithm_label(algorithm),
                                "value": algorithm,
                            }
                            for algorithm in SUPPORTED_CLASSIFIER_ALGORITHMS
                        ],
                        value="logistic_l2",
                        clearable=False,
                    ),
                    html.Label("Feature-column normalization"),
                    dcc.Dropdown(
                        id="classifier-spec-scaling",
                        options=[
                            {"label": "Off (original features)", "value": "off"},
                            {"label": "Fixed (full RMS, γ=1)", "value": "fixed"},
                            {"label": "Tune γ with cross-validation", "value": "tune"},
                        ],
                        value="off",
                        clearable=False,
                    ),
                    html.Small(
                        "Scaling is supported for regularized logistic and linear SVM models. "
                        "Tune compares γ=0, 0.5, 1 alongside the existing regularization search.",
                        className="focus-help",
                    ),
                    html.Small(
                        "New classifiers start with family defaults. In Develop, optionally "
                        "enable Tune hyperparameters with cross-validation before Train when "
                        "you want a slower hyperparameter search.",
                        className="focus-help",
                    ),
                    html.Div(id="classifier-spec-status", className="control-status"),
                    html.Button(
                        "Create classifier",
                        id="classifier-spec-save",
                        n_clicks=0,
                        className="primary-action",
                    ),
                ],
                className="modal-card classifier-modal-card",
            )
        ],
        id="classifier-spec-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _classifier_manager_modal(*, dcc: Any, html: Any, dash_table: Any) -> Any:
    """Return the active-classifier manager."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Manage classifiers"),
                            html.Button(
                                "×",
                                id="classifier-manager-close",
                                n_clicks=0,
                                className="modal-close-icon",
                                title="Close",
                                **{"aria-label": "Close"},
                            ),
                        ],
                        className="modal-heading-row",
                    ),
                    html.P(
                        "Rename a classifier without changing its fitted state, or delete it from active use. Classifier and committee names share one project-wide active predictor namespace. Historical fitted models and drafts retain the name recorded when they were created.",
                        className="focus-help",
                    ),
                    dash_table.DataTable(
                        id="classifier-manager-table",
                        columns=[
                            {"name": "Name", "id": "name"},
                            {"name": "Geometry", "id": "geometry"},
                            {"name": "Algorithm", "id": "algorithm"},
                            {"name": "Selected settings", "id": "settings"},
                        ],
                        data=[],
                        row_selectable="single",
                        selected_rows=[],
                        page_size=12,
                        style_cell={
                            "whiteSpace": "normal",
                            "height": "auto",
                            "textAlign": "left",
                        },
                        style_table={"overflowX": "auto"},
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="classifier-manager-rename-name",
                                type="text",
                                className="text-input",
                                placeholder="Classifier name",
                            ),
                            html.Button(
                                "Rename selected classifier",
                                id="classifier-manager-rename",
                                n_clicks=0,
                            ),
                        ],
                        className="classifier-manager-rename-row",
                    ),
                    html.Div(
                        [
                            html.Label("Selected classifier feature-column normalization"),
                            dcc.Dropdown(
                                id="classifier-manager-scaling",
                                options=[
                                    {"label": "Off", "value": "off"},
                                    {"label": "Fixed (full RMS, γ=1)", "value": "fixed"},
                                    {"label": "Tune γ (0, 0.5, 1)", "value": "tune"},
                                ],
                                value="off",
                                clearable=False,
                            ),
                            html.Button(
                                "Update selected classifier scaling",
                                id="classifier-manager-scaling-save",
                                n_clicks=0,
                            ),
                        ],
                        className="classifier-manager-rename-row",
                    ),
                    html.Div(
                        [
                            html.Div(
                                id="classifier-manager-status",
                                className="control-status",
                            ),
                            html.Button(
                                "Delete selected classifier",
                                id="classifier-manager-delete",
                                n_clicks=0,
                                className="danger-action",
                            ),
                        ],
                        className="modal-actions classifier-manager-actions",
                    ),
                ],
                className="modal-card classifier-manager-modal-card",
            )
        ],
        id="classifier-manager-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _new_committee_modal(*, dcc: Any, html: Any) -> Any:
    """Dedicated creation flow, parallel to New classifier."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("New committee"),
                            html.Button(
                                "×", id="committee-create-close", n_clicks=0,
                                className="modal-close-icon", title="Close",
                                **{"aria-label": "Close"},
                            ),
                        ],
                        className="modal-heading-row",
                    ),
                    html.P(
                        "Combine trained classifiers into a fixed or learned aggregation. "
                        "Fixed committees need no separate training or preparation.",
                        className="focus-help",
                    ),
                    html.Label("Committee name"),
                    dcc.Input(id="committee-name", type="text", className="text-input"),
                    html.Label("Classifier members"),
                    dcc.Checklist(
                        id="committee-members",
                        options=[],
                        value=[],
                        className="committee-member-list",
                    ),
                    html.Label("Aggregation"),
                    dcc.Dropdown(
                        id="committee-aggregation",
                        options=[
                            {"label": "Mean probability", "value": "mean"},
                            {"label": "Minimum probability (strict AND)", "value": "minimum"},
                            {"label": "Maximum probability (permissive OR)", "value": "maximum"},
                            {"label": "Harmonic mean", "value": "harmonic_mean"},
                            {"label": "Geometric mean", "value": "geometric_mean"},
                            {
                                "label": "Logistic stacking (learned weights)",
                                "value": "logistic_stack",
                            },
                        ],
                        value="mean",
                        clearable=False,
                    ),
                    html.Button(
                        "Save new committee",
                        id="committee-save",
                        n_clicks=0,
                        className="primary-action",
                    ),
                    html.Div(id="committee-status", className="control-status"),
                ],
                className="modal-card committee-modal-card",
            )
        ],
        id="committee-create-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _committee_modal(*, dcc: Any, html: Any, dash_table: Any) -> Any:
    """Return the code-specific classifier-committee manager."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Manage committees"),
                            html.Button(
                                "×",
                                id="committee-close",
                                n_clicks=0,
                                className="modal-close-icon",
                                title="Close",
                                **{"aria-label": "Close"},
                            ),
                        ],
                        className="modal-heading-row",
                    ),
                    html.P(
                        "A committee is an intentional named set of classifier specifications for the active code. Rename an existing committee without changing its members, aggregation, or learned fitted state; classifier and committee names share one project-wide active predictor namespace.",
                        className="focus-help",
                    ),
                    html.H4("Existing committees"),
                    dash_table.DataTable(
                        id="committee-table",
                        columns=[
                            {"name": "Name", "id": "name"},
                            {"name": "Aggregation", "id": "aggregation"},
                            {"name": "Members", "id": "members"},
                        ],
                        data=[],
                        row_selectable="single",
                        selected_rows=[],
                        page_size=10,
                        style_cell={
                            "whiteSpace": "normal",
                            "height": "auto",
                            "textAlign": "left",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="committee-manager-rename-name",
                                type="text",
                                className="text-input",
                                placeholder="Committee name",
                            ),
                            html.Button(
                                "Rename selected committee",
                                id="committee-rename",
                                n_clicks=0,
                            ),
                        ],
                        className="classifier-manager-rename-row",
                    ),
                    html.Button(
                        "Delete selected committee",
                        id="committee-delete",
                        n_clicks=0,
                        className="danger-action",
                    ),
                ],
                className="modal-card committee-modal-card",
            )
        ],
        id="committee-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _testing_center_modal(*, dcc: Any, html: Any, dash_table: Any) -> Any:
    """Return the classifier Testing Center foundation."""
    del dcc
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Classifier Testing Center"),
                            html.Button("×", id="testing-center-close", className="modal-close-icon", title="Close", **{"aria-label": "Close"}, n_clicks=0),
                        ],
                        className="modal-heading-row",
                    ),
                    html.P(
                        "Diagnostic teaching-example probes are scored before they enter training. These are diagnostics, not formal validation estimates.",
                        className="focus-help",
                    ),
                    html.Div(id="testing-center-summary", className="focus-model-status"),
                    dash_table.DataTable(
                        id="testing-center-table",
                        columns=[
                            {"name": "Classifier", "id": "classifier"},
                            {"name": "Events", "id": "n"},
                            {"name": "Accuracy", "id": "accuracy"},
                            {"name": "Brier", "id": "brier"},
                        ],
                        data=[],
                        page_size=12,
                        style_cell={"textAlign": "left"},
                    ),
                    html.H4("Assessment events"),
                    dash_table.DataTable(
                        id="testing-center-events",
                        columns=[
                            {"name": "Type", "id": "event_type"},
                            {"name": "Classifier", "id": "classifier"},
                            {"name": "Human", "id": "human"},
                            {"name": "Predicted", "id": "predicted"},
                            {"name": "Probability", "id": "probability"},
                        ],
                        data=[],
                        page_size=12,
                        style_cell={"textAlign": "left"},
                    ),
                ],
                className="modal-card testing-center-modal-card",
            )
        ],
        id="testing-center-modal",
        className="modal-overlay",
        style={"display": "none"},
    )

def _code_modal(*, dcc: Any, html: Any) -> Any:
    """Return the shared code-creation modal used by Explore and Develop."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("New Code"),
                            html.Button("×", id="code-modal-cancel-button", n_clicks=0),
                        ],
                        className="modal-header",
                    ),
                    html.Div(
                        [
                            html.Label("Code label"),
                            dcc.Input(
                                id="code-modal-name",
                                type="text",
                                placeholder="Enter a concise code label",
                                className="text-input code-name-input",
                            ),
                        ],
                        className="code-modal-field",
                    ),
                    html.Div(
                        [
                            html.Label("Description (optional)"),
                            dcc.Textarea(
                                id="code-modal-definition",
                                placeholder="Describe what the code captures. Markdown and #hashtags are supported.",
                                className="code-definition-editor code-modal-definition",
                            ),
                        ],
                        className="code-modal-field",
                    ),
                    html.Div(id="code-modal-status", className="control-status modal-status"),
                    html.Div(
                        [
                            html.Button("Cancel", id="code-modal-cancel-footer", n_clicks=0),
                            html.Button(
                                "Create Code",
                                id="code-modal-create-button",
                                n_clicks=0,
                                className="primary-action",
                            ),
                        ],
                        className="modal-actions",
                    ),
                ],
                className="modal-card code-modal-card",
            )
        ],
        id="code-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _projection_modal(*, dcc: Any, html: Any) -> Any:
    """Return a native Dash modal for creating a persistent 2D view."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Create 2D view"),
                            html.Button("×", id="projection-cancel-button", n_clicks=0),
                        ],
                        className="modal-header",
                    ),
                    html.P(
                        "The new projection is computed from the selected geometry, "
                        "cached, and added to the 2D-view menu.",
                        className="modal-help",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("View name"),
                                    dcc.Input(
                                        id="projection-name",
                                        type="text",
                                        placeholder="Optional; generated automatically",
                                        className="text-input",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Projection type"),
                                    dcc.Dropdown(
                                        id="projection-method",
                                        options=[
                                            {"label": "UMAP", "value": "umap"},
                                            {"label": "PCA", "value": "pca"},
                                            {
                                                "label": "Truncated SVD",
                                                "value": "truncated_svd",
                                            },
                                        ],
                                        value="umap",
                                        clearable=False,
                                        searchable=False,
                                    ),
                                ]
                            ),
                        ],
                        className="projection-base-fields",
                    ),
                    html.Div(
                        [
                            _number_field(
                                dcc=dcc, html=html, label="Neighbors",
                                component_id="projection-umap-neighbors",
                                value=15,
                                minimum=2,
                                step=1,
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Minimum distance",
                                component_id="projection-umap-min-dist",
                                value=0.05,
                                minimum=0.0,
                                step=0.01,
                            ),
                            html.Div(
                                [
                                    html.Label("Metric"),
                                    dcc.Dropdown(
                                        id="projection-umap-metric",
                                        options=[
                                            {"label": value, "value": value}
                                            for value in [
                                                "cosine", "euclidean", "manhattan",
                                                "correlation", "chebyshev",
                                            ]
                                        ],
                                        value="cosine",
                                        clearable=False,
                                        searchable=False,
                                    ),
                                ]
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Spread",
                                component_id="projection-umap-spread",
                                value=2.5,
                                minimum=0.01,
                                step=0.1,
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Repulsion strength",
                                component_id="projection-umap-repulsion",
                                value=1.5,
                                minimum=0.0,
                                step=0.1,
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Random state",
                                component_id="projection-umap-random-state", value=0, step=1,
                            ),
                        ],
                        id="projection-umap-fields",
                        className="projection-parameter-grid",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("SVD solver"),
                                    dcc.Dropdown(
                                        id="projection-pca-solver",
                                        options=[
                                            {"label": value, "value": value}
                                            for value in [
                                                "auto",
                                                "full",
                                                "covariance_eigh",
                                                "arpack",
                                                "randomized",
                                            ]
                                        ],
                                        value="auto",
                                        clearable=False,
                                        searchable=False,
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Whiten"),
                                    dcc.Checklist(
                                        id="projection-pca-whiten",
                                        options=[{"label": "Scale components", "value": "whiten"}],
                                        value=[],
                                    ),
                                ]
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Random state",
                                component_id="projection-pca-random-state", value=0, step=1,
                            ),
                        ],
                        id="projection-pca-fields",
                        className="projection-parameter-grid",
                        style={"display": "none"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Algorithm"),
                                    dcc.Dropdown(
                                        id="projection-svd-algorithm",
                                        options=[
                                            {"label": "Randomized", "value": "randomized"},
                                            {"label": "ARPACK", "value": "arpack"},
                                        ],
                                        value="randomized",
                                        clearable=False,
                                        searchable=False,
                                    ),
                                ]
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Power iterations",
                                component_id="projection-svd-n-iter", value=5, minimum=1, step=1,
                            ),
                            _number_field(
                                dcc=dcc, html=html, label="Random state",
                                component_id="projection-svd-random-state", value=0, step=1,
                            ),
                        ],
                        id="projection-svd-fields",
                        className="projection-parameter-grid",
                        style={"display": "none"},
                    ),
                    html.Div(
                        html.Div(className="projection-progress-bar"),
                        id="projection-progress-wrap",
                        className="projection-progress-track",
                        style={"display": "none"},
                    ),
                    html.Div(
                        id="projection-create-status",
                        className="control-status modal-status",
                    ),
                    html.Div(
                        [
                            html.Button("Cancel", id="projection-cancel-footer", n_clicks=0),
                            html.Button(
                                "Create view",
                                id="projection-create-button",
                                n_clicks=0,
                                className="primary-action",
                            ),
                        ],
                        className="modal-actions",
                    ),
                ],
                className="modal-card",
            )
        ],
        id="projection-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _number_field(
    *,
    dcc: Any,
    html: Any,
    label: str,
    component_id: str,
    value: int | float,
    minimum: int | float | None = None,
    step: int | float = 1,
) -> Any:
    """Render a plain numeric text field without browser spinner controls.

    Validation happens only when the projection is submitted, which avoids the
    inconsistent native-number validation users encountered across browsers.
    """
    del minimum, step
    return html.Div(
        [
            html.Label(label),
            dcc.Input(
                id=component_id,
                type="text",
                value=str(value),
                className="projection-number-input",
            ),
        ]
    )



def _focus_layout(
    *,
    dcc: Any,
    html: Any,
    codes: list[dict[str, Any]],
    constructs: list[dict[str, Any]],
    geometries: list[dict[str, Any]],
    classifier_specs: list[dict[str, Any]],
    hierarchy: list[dict[str, Any]],
    initial_state: dict[str, Any],
) -> Any:
    """Return a shared Construct selector with the existing binary Develop workspace."""
    del geometries
    code_ids = {int(code["code_id"]) for code in codes}
    initial_construct_id = _selected_focus_construct_id(constructs, initial_state)
    initial_code_id = (
        initial_construct_id if initial_construct_id in code_ids else None
    )

    classifier_specs = [
        row
        for row in classifier_specs
        if initial_code_id is not None and int(row["code_id"]) == initial_code_id
    ]
    spec_ids = {int(row["classifier_spec_id"]) for row in classifier_specs}
    try:
        requested_spec_id = int(initial_state.get("focus_classifier_spec_id"))
    except (TypeError, ValueError):
        requested_spec_id = -1
    initial_spec_id = (
        requested_spec_id
        if requested_spec_id in spec_ids
        else (min(spec_ids) if spec_ids else None)
    )
    active_class = "focus-active-area" + (
        " focus-disabled" if initial_code_id is None else ""
    )
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Construct"),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="focus-construct-dropdown",
                                        options=_construct_options(constructs),
                                        value=initial_construct_id,
                                        clearable=True,
                                        placeholder="Select a Code or Scale",
                                    ),
                                    # Existing binary callbacks remain bound to this
                                    # Code-only value; Scale IDs never enter it.
                                    dcc.Dropdown(
                                        id="focus-code-dropdown",
                                        options=_code_options(codes),
                                        value=initial_code_id,
                                        clearable=True,
                                        style={"display": "none"},
                                    ),
                                    html.Button(
                                        "New Code", id="focus-new-code-button", n_clicks=0,
                                        style=({"display": "none"} if initial_construct_id is not None and initial_code_id is None else {}),
                                    ),
                                ],
                                className="code-selector-row",
                            ),
                            html.Div(
                                [
                                    html.Div(id="focus-code-definition", className="code-definition"),
                                    html.Div(id="focus-label-counts", className="focus-counts"),
                                    html.Div(id="focus-code-create-status", className="control-status"),
                                ],
                                id="focus-code-summary",
                                style=({"display": "none"} if initial_code_id is None else {}),
                            ),
                        ],
                        className="focus-code-select",
                    ),
                    html.Div(
                        [
                            html.Label(_tooltip_label(html, "Teaching examples", "Add your own labeled text examples that can help train classifiers alongside coded units.")),
                            html.Div(
                                "Select a code to manage examples.",
                                id="focus-teaching-counts",
                                className="focus-counts",
                            ),
                            html.Button(
                                "Manage teaching examples",
                                id="focus-teaching-button",
                                n_clicks=0,
                                disabled=initial_code_id is None,
                            ),
                        ],
                        id="focus-teaching-select",
                        style=({"display": "none"} if initial_code_id is None else {}),
                        className="focus-teaching-select",
                    ),
                ],
                className="focus-topbar",
            ),
            html.Section(
                [
                    html.H3(
                        id="focus-scale-heading",
                        children=(f"Scale: {next(row['name'] for row in constructs if row['construct_id'] == initial_construct_id)}"
                                  if initial_construct_id is not None and initial_code_id is None else ""),
                    ),
                    html.Div(
                        id="focus-scale-description",
                        children=(next(row.get("description", "") for row in constructs if row['construct_id'] == initial_construct_id)
                                  if initial_construct_id is not None and initial_code_id is None else ""),
                        className="code-definition",
                    ),
                    html.Div(
                        id="focus-scale-eligibility",
                        children=(
                            f"Definition version {next(row['version_number'] for row in constructs if row['construct_id'] == initial_construct_id)} · "
                            f"Eligibility: {json.dumps(next(row['eligibility'] for row in constructs if row['construct_id'] == initial_construct_id), sort_keys=True)}"
                            if initial_construct_id is not None and initial_code_id is None else ""
                        ),
                        className="focus-counts",
                    ),
                    html.P(
                        "Comparative development for this Scale will be available in a subsequent phase.",
                        className="focus-help",
                    ),
                ],
                id="focus-scale-state",
                style=({} if initial_construct_id is not None and initial_code_id is None else {"display": "none"}),
                className="focus-disabled-message",
            ),
            html.Div(
                [
                    html.Div(
                        "Select a code to begin developing it.",
                        id="focus-disabled-message",
                        className="focus-disabled-message",
                    ),
                    html.Div(
                        [
                            html.Aside(
                                [
                                    html.H3("Classifier controls"),
                                    html.Div(
                                        [
                                            html.Label("Active classifier"),
                                            html.Span(
                                                "Not trained",
                                                id="focus-active-classifier-status",
                                                className="classifier-status-badge status-not-trained",
                                            ),
                                        ],
                                        className="active-classifier-heading",
                                    ),
                                    dcc.Dropdown(
                                        id="focus-active-classifier-dropdown",
                                        options=[
                                            {
                                                "label": f"{row['name']} · {row['geometry_name']}",
                                                "value": int(row["classifier_spec_id"]),
                                            }
                                            for row in classifier_specs
                                        ],
                                        value=initial_spec_id,
                                        clearable=False,
                                        placeholder="Create a classifier",
                                    ),
                                    html.Button(
                                        "Train",
                                        id="focus-train-active",
                                        n_clicks=0,
                                        className="primary-action classifier-train-button",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="focus-train-scope",
                                                options=[
                                                    {
                                                        "label": "Train all classifiers",
                                                        "value": "all",
                                                    }
                                                ],
                                                value=(
                                                    ["all"]
                                                    if bool(
                                                        initial_state.get(
                                                            "focus_train_all", False
                                                        )
                                                    )
                                                    else []
                                                ),
                                                className="classifier-train-scope",
                                            ),
                                            dcc.Checklist(
                                                id="focus-tune-hyperparameters",
                                                options=[
                                                    {
                                                        "label": _tooltip_label(html, "Tune hyperparameters with cross-validation", "Compare hyperparameter settings using cross-validation, then fit the selected configuration. This takes longer than ordinary training."),
                                                        "value": "tune",
                                                    }
                                                ],
                                                value=(
                                                    ["tune"]
                                                    if bool(
                                                        initial_state.get(
                                                            "focus_tune_hyperparameters", True
                                                        )
                                                    )
                                                    else []
                                                ),
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Checklist(
                                                        id="focus-auto-retrain",
                                                        options=[
                                                            {"label": _tooltip_label(html, "Automatically train", "Automatically request training when saved coding changes affect the code selected in Develop."), "value": "auto"}
                                                        ],
                                                        value=(
                                                            ["auto"]
                                                            if bool(
                                                                initial_state.get(
                                                                    "focus_auto_retrain", True
                                                                )
                                                            )
                                                            else []
                                                        ),
                                                    ),
                                                ],
                                                className="classifier-auto-train-row",
                                            ),
                                        ],
                                        className="classifier-training-options",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "New classifier",
                                                id="focus-new-classifier-button",
                                                n_clicks=0,
                                            ),
                                            html.Button(
                                                "Manage classifiers",
                                                id="focus-manage-classifiers",
                                                n_clicks=0,
                                            ),
                                        ],
                                        className="classifier-management-row",
                                    ),
                                    html.Div(
                                        id="focus-active-classifier-detail",
                                        className="focus-model-detail",
                                    ),
                                    html.Div(
                                        id="focus-train-action-status",
                                        className="control-status",
                                    ),
                                    html.Div(
                                        id="focus-auto-training-progress",
                                        className="control-status",
                                        role="status",
                                        **{"aria-live": "polite"},
                                    ),
                                    html.Div(
                                        id="focus-auto-training-status",
                                        className="control-status",
                                        role="status",
                                        **{"aria-live": "polite"},
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Active committee"),
                                            html.Span(
                                                "Not trained",
                                                id="focus-committee-badge",
                                                className="classifier-status-badge status-not-trained",
                                            ),
                                        ],
                                        className="active-classifier-heading active-committee-heading",
                                    ),
                                    dcc.Dropdown(
                                        id="focus-committee-dropdown",
                                        options=[],
                                        value=None,
                                        clearable=True,
                                        placeholder="Select a saved committee",
                                    ),
                                    html.Button(
                                        "Train committee",
                                        id="focus-train-committee",
                                        n_clicks=0,
                                        className="primary-action classifier-train-button",
                                        style={"display": "none"},
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "New committee",
                                                id="focus-new-committee",
                                                n_clicks=0,
                                            ),
                                            html.Button(
                                                "Manage committees",
                                                id="focus-manage-committees",
                                                n_clicks=0,
                                            ),
                                        ],
                                        className="classifier-management-row",
                                    ),
                                    html.Div(
                                        id="focus-committee-status",
                                        className="focus-model-detail",
                                    ),
                                    html.Div(
                                        id="focus-committee-train-status",
                                        className="control-status",
                                    ),
                                    html.Button(
                                        "Testing Center (beta)",
                                        id="focus-testing-center-button",
                                        n_clicks=0,
                                        style={"width": "100%"},
                                    ),
                                ],
                                className="focus-sidebar",
                            ),
                            html.Main(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label("Recommendation source"),
                                                    dcc.Dropdown(
                                                        id="focus-recommendation-source",
                                                        options=[
                                                            {
                                                                "label": "Active classifier",
                                                                "value": "classifier",
                                                            },
                                                            {
                                                                "label": "Selected committee",
                                                                "value": "committee",
                                                            },
                                                        ],
                                                        value="classifier",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="recommendation-source-control",
                                            ),
                                            html.Div(
                                                [
                                                    html.Button(
                                                        "Most likely",
                                                        id="focus-most-likely",
                                                        n_clicks=0,
                                                    ),
                                                    html.Button(
                                                        "Least likely",
                                                        id="focus-least-likely",
                                                        n_clicks=0,
                                                    ),
                                                    html.Button(
                                                        "Most uncertain",
                                                        id="focus-most-uncertain",
                                                        n_clicks=0,
                                                    ),
                                                    html.Button(
                                                        "Greatest disagreement",
                                                        id="focus-disagreement",
                                                        n_clicks=0,
                                                        style={"display": "none"},
                                                    ),
                                                    html.Button(
                                                        _tooltip_label(html, "Review unsure", "Revisit units you previously marked as unsure."),
                                                        id="focus-review-unsure",
                                                        n_clicks=0,
                                                    ),
                                                    html.Button(
                                                        "Random",
                                                        id="focus-random",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                                className="focus-recommendation-buttons",
                                            ),
                                        ],
                                        className="focus-recommendation-panel",
                                    ),
                                    html.Div(
                                        id="focus-recommendation-status",
                                        className="focus-status",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        id="focus-focal-key",
                                                        className="focal-key",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Label(
                                                                        "Context level"
                                                                    ),
                                                                    dcc.Dropdown(
                                                                        id="focus-context-level",
                                                                        options=[
                                                                            {
                                                                                "label": level[
                                                                                    "column_name"
                                                                                ],
                                                                                "value": int(
                                                                                    level[
                                                                                        "level_index"
                                                                                    ]
                                                                                ),
                                                                            }
                                                                            for level in hierarchy
                                                                        ],
                                                                        value=int(
                                                                            initial_state.get(
                                                                                "focus_context_level",
                                                                                hierarchy[-1][
                                                                                    "level_index"
                                                                                ],
                                                                            )
                                                                        ),
                                                                        clearable=False,
                                                                        searchable=False,
                                                                    ),
                                                                ],
                                                                className="context-control",
                                                            ),
                                                            html.Div(
                                                                [
                                                                    html.Label("Window"),
                                                                    html.Div(
                                                                        [
                                                                            html.Button(
                                                                                "−",
                                                                                id="focus-context-window-minus",
                                                                                n_clicks=0,
                                                                            ),
                                                                            dcc.Input(
                                                                                id="focus-context-window",
                                                                                type="text",
                                                                                inputMode="numeric",
                                                                                value=str(
                                                                                    max(
                                                                                        1,
                                                                                        int(
                                                                                            initial_state.get(
                                                                                                "focus_context_window",
                                                                                                1,
                                                                                            )
                                                                                        ),
                                                                                    )
                                                                                ),
                                                                            ),
                                                                            html.Button(
                                                                                "+",
                                                                                id="focus-context-window-plus",
                                                                                n_clicks=0,
                                                                            ),
                                                                        ],
                                                                        className="window-stepper",
                                                                    ),
                                                                ],
                                                                className="context-control",
                                                            ),
                                                        ],
                                                        className="context-controls",
                                                    ),
                                                    html.Div(
                                                        id="focus-context-content",
                                                        className="context-content focus-context-content",
                                                    ),
                                                    html.Details(
                                                        [
                                                            html.Summary("Metadata"),
                                                            html.Div(
                                                                id="focus-metadata-content"
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                className="focus-reading",
                                            ),
                                            html.Aside(
                                                [
                                                    html.H3("Code this observation"),
                                                    html.Div(
                                                        [
                                                            html.Button(
                                                                "Present",
                                                                id="focus-label-positive",
                                                                n_clicks=0,
                                                                className="judgment-button positive-button",
                                                            ),
                                                            html.Button(
                                                                "Absent",
                                                                id="focus-label-negative",
                                                                n_clicks=0,
                                                                className="judgment-button negative-button",
                                                            ),
                                                            html.Button(
                                                                "Unsure",
                                                                id="focus-label-unsure",
                                                                n_clicks=0,
                                                                className="judgment-button unsure-button",
                                                            ),
                                                        ],
                                                        className="focus-label-buttons",
                                                    ),
                                                    html.Div(
                                                        id="focus-current-label",
                                                        className="sr-only",
                                                    ),
                                                    html.Div(
                                                        id="focus-annotation-status",
                                                        className="control-status",
                                                    ),
                                                    html.Div(
                                                        id="focus-probabilities",
                                                        className="focus-probabilities",
                                                    ),
                                                ],
                                                className="focus-coding-panel",
                                            ),
                                        ],
                                        className="focus-main",
                                    ),
                                    html.Section(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.H3(
                                                                "Prediction geometry"
                                                            ),
                                                            html.P(
                                                                "Choose fitted classifiers. One score is shown as a jittered line, two as direct axes, and three or more through PCA.",
                                                                className="focus-help",
                                                            ),
                                                        ]
                                                    ),
                                                    html.Button(
                                                        "Refresh score view",
                                                        id="focus-prediction-refresh",
                                                        n_clicks=0,
                                                    ),
                                                ],
                                                className="center-heading-row",
                                            ),
                                            html.Details(
                                                [
                                                    html.Summary(
                                                        "Select classifiers for prediction geometry"
                                                    ),
                                                    dcc.Checklist(
                                                        id="focus-prediction-fits",
                                                        options=[],
                                                        value=[],
                                                        className="prediction-fit-checklist",
                                                    ),
                                                ],
                                                className="prediction-classifier-picker",
                                            ),
                                            html.Div(
                                                [
                                                    dcc.RadioItems(
                                                        id="focus-prediction-score-type",
                                                        options=[
                                                            {"label": "Estimated probabilities", "value": "probability"},
                                                            {"label": "Native decision scores", "value": "decision_score"},
                                                        ],
                                                        value="probability",
                                                        inline=True,
                                                        className="prediction-display-options",
                                                    ),
                                                ],
                                                className="prediction-geometry-options",
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Checklist(
                                                        id="focus-prediction-display-options",
                                                        options=[
                                                            {
                                                                "label": "Show all coded units",
                                                                "value": "coded",
                                                            },
                                                            {
                                                                "label": "Show spans",
                                                                "value": "spans",
                                                            },
                                                            {
                                                                "label": "Show teaching examples",
                                                                "value": "teaching",
                                                            },
                                                        ],
                                                        value=["coded"],
                                                        inline=True,
                                                        className="prediction-display-options",
                                                    ),
                                                ],
                                                className="prediction-geometry-options",
                                            ),
                                            html.Div(
                                                id="focus-prediction-status",
                                                className="control-status",
                                            ),
                                            dcc.Graph(
                                                id="focus-prediction-map",
                                                figure=go.Figure(),
                                                config={
                                                    "displaylogo": False,
                                                    "scrollZoom": True,
                                                    "doubleClick": "reset",
                                                    "modeBarButtonsToRemove": [
                                                        "lasso2d",
                                                        "select2d",
                                                    ],
                                                },
                                                className="focus-prediction-map",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Points per page",
                                                        htmlFor="focus-prediction-page-size",
                                                    ),
                                                    dcc.Input(
                                                        id="focus-prediction-page-size",
                                                        type="text",
                                                        inputMode="numeric",
                                                        debounce=True,
                                                        value="5000",
                                                        className="page-size-input",
                                                    ),
                                                    html.Button(
                                                        "Previous",
                                                        id="focus-prediction-page-previous",
                                                        n_clicks=0,
                                                    ),
                                                    html.Label(
                                                        "Page",
                                                        htmlFor="focus-prediction-page-number",
                                                    ),
                                                    dcc.Input(
                                                        id="focus-prediction-page-number",
                                                        type="number",
                                                        min=1,
                                                        step=1,
                                                        debounce=True,
                                                        value=1,
                                                        max=1,
                                                        className="page-number-input",
                                                    ),
                                                    html.Button(
                                                        "Next",
                                                        id="focus-prediction-page-next",
                                                        n_clicks=0,
                                                    ),
                                                    html.Div(
                                                        id="focus-prediction-page-status",
                                                        className="page-status",
                                                    ),
                                                ],
                                                className="pagination-bar prediction-pagination",
                                            ),
                                        ],
                                        className="prediction-geometry-panel",
                                    ),
                                ],
                                className="focus-content",
                            ),
                        ],
                        className="focus-workspace-grid",
                    ),
                ],
                id="focus-active-area",
                className=active_class,
                style=({"display": "none"} if initial_construct_id is not None and initial_code_id is None else {}),
            ),
        ],
        className="focus-workspace",
    )


def _apply_layout(
    *, dcc: Any, html: Any, dag: Any,
    codes: list[dict[str, Any]], classifier_specs: list[dict[str, Any]],
) -> Any:
    del classifier_specs
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Apply and Review"),
                    html.P("Generate an inspectable proposal draft. Nothing becomes an annotation until you commit reviewed decisions."),
                    html.Label("Code"),
                    dcc.Dropdown(id="apply-code-dropdown", options=_code_options(codes), value=(int(codes[0]["code_id"]) if codes else None), clearable=False),
                    html.Label("Classifier or committee"),
                    dcc.Dropdown(
                        id="apply-source-dropdown",
                        options=[],
                        value=None,
                        clearable=False,
                        placeholder="Train a classifier in Develop first",
                    ),
                    html.Div(
                        "Apply uses the newest explicitly trained fit. Stale fits remain usable and are identified in Develop.",
                        className="focus-help",
                    ),
                    html.Div(
                        [
                            html.Label("Decision threshold"),
                            dcc.Input(id="apply-threshold", type="number", min=0.0, max=1.0, step=0.01, value=0.5),
                            html.Button("Generate proposal draft", id="apply-generate", n_clicks=0, className="primary-action"),
                        ], className="apply-generation-controls",
                    ),
                ], className="apply-sidebar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div([html.Label("Draft"), dcc.Dropdown(id="apply-run-dropdown", options=[], value=None)], className="apply-draft-select"),
                            html.Button("Discard draft", id="apply-discard", n_clicks=0, className="danger-action"),
                            dcc.ConfirmDialog(id="apply-discard-confirm", message="Discard this proposal draft? No annotations will be changed."),
                        ], className="apply-draft-header",
                    ),
                    html.Div(id="apply-run-summary", className="focus-status"),
                    html.Div(
                        [
                            dcc.Dropdown(id="apply-proposed-filter", options=[{"label": "All proposals", "value": ""}, {"label": "Positive proposals", "value": "positive"}, {"label": "Negative proposals", "value": "negative"}], value="", clearable=False),
                            dcc.Dropdown(id="apply-decision-filter", options=[{"label": "All decisions", "value": ""}, {"label": "Pending", "value": "pending"}, {"label": "Accepted", "value": "accept"}, {"label": "Positive override", "value": "positive"}, {"label": "Negative override", "value": "negative"}, {"label": "Unsure", "value": "unsure"}, {"label": "Leave unreviewed", "value": "unreviewed"}], value="", clearable=False),
                            html.Div(
                                [html.Label("Probability range"), dcc.RangeSlider(
                                    id="apply-probability-range",
                                    min=0.0,
                                    max=1.0,
                                    step=0.000001,
                                    value=[0.0, 1.0],
                                    marks={},
                                    allowCross=False,
                                    allow_direct_input=True,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                        "transform": "formatProbability",
                                        "style": {
                                            "minWidth": "92px",
                                            "textAlign": "center",
                                            "fontVariantNumeric": "tabular-nums",
                                        },
                                    },
                                )],
                                className="apply-probability-filter",
                            ),
                        ], className="apply-filters",
                    ),
                    dag.AgGrid(
                        id="apply-table",
                        columnDefs=[
                            {
                                "headerName": "Unit",
                                "field": "unit_id",
                                "width": 92,
                                "editable": False,
                                "pinned": "left",
                            },
                            {
                                "headerName": "Key",
                                "field": "key",
                                "minWidth": 190,
                                "flex": 0.9,
                                "wrapText": True,
                                "autoHeight": True,
                                "cellStyle": {"lineHeight": "1.3"},
                                "editable": False,
                            },
                            {
                                "headerName": "Text",
                                "field": "text",
                                "minWidth": 360,
                                "flex": 2.2,
                                "wrapText": True,
                                "autoHeight": True,
                                "cellStyle": {"lineHeight": "1.3"},
                                "editable": False,
                            },
                            {
                                "headerName": "Probability",
                                "field": "probability",
                                "width": 128,
                                "editable": False,
                                "valueFormatter": {
                                    "function": "params.value == null ? '' : Number(params.value).toFixed(6)"
                                },
                            },
                            {
                                "headerName": "Proposal",
                                "field": "proposed_label",
                                "width": 118,
                                "editable": False,
                            },
                            {
                                "headerName": "Decision",
                                "field": "decision",
                                "width": 180,
                                "editable": True,
                                "cellEditor": "agSelectCellEditor",
                                "cellEditorParams": {
                                    "values": list(APPLY_DECISION_VALUES),
                                },
                            },
                        ],
                        rowData=[],
                        defaultColDef={
                            "sortable": True,
                            "resizable": True,
                            "suppressHeaderMenuButton": True,
                        },
                        columnSize="responsiveSizeToFit",
                        dashGridOptions={
                            "theme": GECO_AG_GRID_THEME,
                            "animateRows": False,
                            "pagination": False,
                            "singleClickEdit": True,
                            "stopEditingWhenCellsLoseFocus": True,
                            "rowSelection": {
                                "mode": "multiRow",
                                "checkboxes": True,
                                "headerCheckbox": True,
                                # AG Grid only holds one server page at a time.
                                # "all" selects that loaded page, not the draft.
                                "selectAll": "all",
                            },
                            "selectionColumnDef": {
                                "width": 44,
                                "resizable": False,
                                "sortable": False,
                                "pinned": "left",
                                "lockPosition": "left",
                                "suppressMovable": True,
                                "suppressHeaderMenuButton": True,
                            },
                            "getRowId": {"function": "params.data.unit_id"},
                        },
                        className="apply-review-grid",
                        style={"height": "690px", "width": "100%"},
                    ),
                    html.Div(
                        [
                            html.Label("Rows per page", htmlFor="apply-page-size"),
                            dcc.Dropdown(
                                id="apply-page-size",
                                options=[
                                    {"label": str(size), "value": size}
                                    for size in (25, 50, 100)
                                ],
                                value=25,
                                clearable=False,
                                searchable=False,
                                className="apply-page-size",
                            ),
                            html.Button("Previous", id="apply-page-previous", n_clicks=0),
                            html.Span(id="apply-page-status", className="page-status"),
                            html.Button("Next", id="apply-page-next", n_clicks=0),
                        ],
                        className="pagination-bar apply-pagination",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4("Bulk proposal actions"),
                                    html.Div(
                                        [
                                            html.Label("Decision", htmlFor="apply-bulk-decision"),
                                            dcc.Dropdown(
                                                id="apply-bulk-decision",
                                                options=[
                                                    {"label": "Accept proposal", "value": "accept"},
                                                    {"label": "Assign positive", "value": "positive"},
                                                    {"label": "Assign negative", "value": "negative"},
                                                    {"label": "Assign unsure", "value": "unsure"},
                                                    {"label": "Leave unreviewed", "value": "unreviewed"},
                                                ],
                                                value="accept",
                                                clearable=False,
                                            ),
                                        ],
                                        className="apply-bulk-field",
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Scope", htmlFor="apply-bulk-scope"),
                                            dcc.Dropdown(
                                                id="apply-bulk-scope",
                                                options=[
                                                    {"label": "All proposals", "value": "all"},
                                                    {"label": "Current page", "value": "current_page"},
                                                    {"label": "Selected proposals (this page)", "value": "selected"},
                                                ],
                                                value="selected",
                                                clearable=False,
                                            ),
                                        ],
                                        className="apply-bulk-field",
                                    ),
                                    html.Button(
                                        "Apply",
                                        id="apply-bulk-apply",
                                        n_clicks=0,
                                        className="primary-action apply-bulk-button",
                                    ),
                                ],
                                className="apply-action-group apply-bulk-action-group",
                            ),
                            html.Div([html.H4("Clear decisions"), html.Button("Clear all", id="apply-clear-decisions", n_clicks=0), html.Button("Clear current page", id="apply-clear-visible", n_clicks=0)], className="apply-action-group"),
                            html.Div(
                                [
                                    html.H4("Commit"),
                                    html.Button(
                                        "Commit all reviewed",
                                        id="apply-commit",
                                        n_clicks=0,
                                        className="primary-action",
                                    ),
                                    html.Button(
                                        "Commit current page reviewed",
                                        id="apply-commit-visible",
                                        n_clicks=0,
                                    ),
                                    html.Button(
                                        "Undo last commit",
                                        id="apply-undo-last-commit",
                                        n_clicks=0,
                                        className="danger-action",
                                        disabled=True,
                                    ),
                                ],
                                className="apply-action-group",
                            ),
                        ], className="apply-actions",
                    ),
                    html.Div(id="apply-status", className="control-status"),
                    html.Div(id="apply-training-status", className="control-status", role="status"),
                ], className="apply-main",
            ),
        ], className="apply-workspace",
    )

def _project_summary_children(
    *,
    html: Any,
    modality: str,
    key_columns: list[str],
    hierarchy_summary: list[dict[str, Any]],
    atomic_count: int,
) -> list[Any]:
    """Build the compact modality and hierarchy summary shown in the header."""
    atomic_key = key_columns[-1]
    children: list[Any] = [
        html.Span([html.Strong("Modality:"), " ", modality]),
        html.Span("·", className="project-summary-separator"),
        html.Span(
            [html.Strong("Atomic key:"), " ", f"{atomic_key} (n={atomic_count:,})"]
        ),
    ]
    parents = hierarchy_summary[:-1]
    if parents:
        label = "Parent key:" if len(parents) == 1 else "Parent keys:"
        parent_children: list[Any] = [html.Strong(label), " "]
        for index, parent in enumerate(parents):
            if index:
                parent_children.append(", ")
            parent_children.append(
                f"{parent['column_name']} (n={int(parent['count']):,})"
            )
        children.extend(
            [
                html.Span("·", className="project-summary-separator"),
                html.Span(parent_children),
            ]
        )
    return children


def _normalized_probability_range(value: list[float] | None) -> tuple[float, float]:
    """Return a validated closed probability interval."""
    if not value or len(value) != 2:
        return 0.0, 1.0
    lower = min(1.0, max(0.0, float(value[0])))
    upper = min(1.0, max(0.0, float(value[1])))
    return (lower, upper) if lower <= upper else (upper, lower)


def _normalize_apply_decision(value: Any) -> str:
    """Normalize an AG Grid display label or stored value to a decision value."""
    rendered = str(value if value is not None else "pending").strip()
    if rendered in APPLY_DECISION_LABELS:
        return rendered
    if rendered in APPLY_DECISION_VALUES:
        return APPLY_DECISION_VALUES[rendered]
    if rendered == "Exclude":
        # Compatibility for a stale pre-0.5.1 browser row. Projects are rebuilt.
        return "unreviewed"
    raise ValueError(f"Unknown review decision {rendered!r}.")


def _apply_sort_spec(
    column_state: list[dict[str, Any]] | None,
) -> list[tuple[str, str]]:
    """Canonical server ordering matching AG Grid's sortable column state."""
    sortable = {"unit_id", "key", "text", "probability", "proposed_label", "decision"}
    entries = [
        (index, str(column["colId"]), str(column["sort"]))
        for index, column in enumerate(column_state or [])
        if column.get("sort") in {"asc", "desc"}
        and str(column.get("colId")) in sortable
    ]
    entries.sort(key=lambda item: (
        int((column_state or [])[item[0]].get("sortIndex") or 0),
        item[0],
    ))
    return [(field, direction) for _, field, direction in entries[:3]]


def _verified_apply_page(
    project: GeometricCoder, *,
    draft_id: int, snapshot: dict[str, Any] | None,
    proposed_filter: str | None, decision_filter: str | None,
    probability_range: list[float] | None, page_size: int | None,
    column_state: list[dict[str, Any]] | None,
    refresh: int | None,
) -> list[int]:
    """Refuse a stale page identity instead of targeting a different row set."""
    page = dict(snapshot or {})
    lower, upper = _normalized_probability_range(probability_range)
    ordering = _apply_sort_spec(column_state)
    if (
        page.get("draft_id") != int(draft_id)
        or page.get("refresh") != int(refresh or 0)
        or page.get("proposed") != (proposed_filter or "")
        or page.get("decision") != (decision_filter or "")
        or page.get("range") != [lower, upper]
        or page.get("sort") != [list(pair) for pair in ordering]
        or page.get("page_size") != int(page_size or 25)
    ):
        raise ValueError("Apply page changed. Let the table refresh before reviewing.")
    count = project.count_apply_proposals(
        int(draft_id), proposed_label=proposed_filter or None,
        decision=decision_filter or None,
        probability_min=lower, probability_max=upper,
    )
    if count != int(page.get("count", -1)):
        raise ValueError("Apply proposals changed. Refresh the current page.")
    ids = [
        int(row["unit_id"])
        for row in project.apply_proposals(
            int(draft_id), proposed_label=proposed_filter or None,
            decision=decision_filter or None,
            probability_min=lower, probability_max=upper,
            limit=int(page["page_size"]),
            offset=int(page["page_index"]) * int(page["page_size"]),
            sort_by=ordering,
        )
    ]
    if ids != page.get("unit_ids"):
        raise ValueError("Apply page changed. Refresh before reviewing these proposals.")
    return ids


def _apply_current_page_rows(
    rows: list[dict[str, Any]], pagination_info: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Return exactly the rows displayed on the current client-side AG Grid page."""
    info = pagination_info or {}
    current_page = max(0, int(info.get("currentPage", 0) or 0))
    page_size = max(1, int(info.get("pageSize", 25) or 25))
    start = current_page * page_size
    return list(rows[start : start + page_size])


def _apply_bulk_scope_unit_ids(
    *,
    scope: str,
    current_page_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> list[int] | None:
    """Resolve the proposal IDs targeted by one bulk review action."""
    if scope == "all":
        return None
    if scope == "current_page":
        return [int(row["unit_id"]) for row in current_page_rows]
    if scope == "selected":
        return [int(row["unit_id"]) for row in selected_rows]
    raise ValueError(f"Unknown bulk proposal scope {scope!r}.")


def _code_options(codes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"label": code["name"], "value": int(code["code_id"])}
        for code in codes
    ]

def _session_options(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"label": session["title"], "value": int(session["session_id"])}
        for session in sessions
    ]


def _valid_geometry_id(value: Any, geometries: list[dict[str, Any]]) -> int | None:
    valid = {int(record["geometry_id"]) for record in geometries}
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        candidate = -1
    return candidate if candidate in valid else (min(valid) if valid else None)


def _valid_view_id(value: Any, views: list[dict[str, Any]]) -> int | None:
    """Restore a saved view, preferring legacy views for an unset selection.

    Canonical-only representation views use negative compatibility IDs; an
    unset/invalid saved view must not suddenly prefer one merely because its
    negative ID sorts before the original registered view.
    """
    valid = {int(record["view_id"]) for record in views}
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        candidate = None
    if candidate is not None and candidate in valid:
        return candidate
    legacy = [view_id for view_id in valid if view_id > 0]
    if legacy:
        return min(legacy)
    return int(views[0]["view_id"]) if views else None


def _display_view_records(
    project: GeometricCoder, geometry_id: int,
) -> list[dict[str, Any]]:
    """Include numerical descendants, not just owned legacy 2D views."""
    return project.views(int(geometry_id), include_descendants=True)


def _memo_presentation_is_active(search: str | None) -> bool:
    """Only a presentation URL may trigger expensive map/filter work."""
    return parse_qs((search or "").lstrip("?")).get("presentation") == ["memo"]


def _presentation_view_records(
    project: GeometricCoder, geometries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """List distinct plot-ready view IDs under currently analytic geometries.

    A descendant may visualize several ancestors; presentation offers one
    selection per numerical view, deterministically choosing a public analytic
    source rather than showing duplicate dropdown values.
    """
    result: dict[int, dict[str, Any]] = {}
    for geometry in geometries:
        geometry_id = int(geometry["geometry_id"])
        for view in _display_view_records(project, geometry_id):
            view_id = int(view["view_id"])
            if view_id not in result:
                result[view_id] = {
                    "view_id": view_id,
                    "geometry_id": geometry_id,
                    "name": str(view["name"]),
                }
    return sorted(result.values(), key=lambda row: (
        str(row["name"]).casefold(), int(row["view_id"]),
    ))


def _presentation_view_options(
    views: list[dict[str, Any]], geometries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Provide labels with analytic source context and stable display IDs."""
    names = {int(row["geometry_id"]): str(row["name"]) for row in geometries}
    return [
        {
            "label": f"{names[int(view['geometry_id'])]} — {view['name']}",
            "value": int(view["view_id"]),
        }
        for view in views
        if int(view["geometry_id"]) in names
    ]


def _display_view_record(
    project: GeometricCoder, *, geometry_id: int, view_id: int,
) -> dict[str, Any] | None:
    """Resolve a selected legacy or canonical-only view within its geometry."""
    return next(
        (row for row in _display_view_records(project, geometry_id)
         if int(row["view_id"]) == int(view_id)),
        None,
    )


def _coerce_positive_int(value: Any, *, default: int) -> int:
    """Parse a user-entered positive integer without browser step constraints."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed >= 1 else int(default)


def _clicked_unit_id(click_data: dict[str, Any] | None) -> int | None:
    """Extract one unit ID from a Plotly click event."""
    if not click_data or not click_data.get("points"):
        return None
    customdata = click_data["points"][0].get("customdata")
    if isinstance(customdata, (list, tuple, np.ndarray)):
        if len(customdata) == 0:
            return None
        customdata = customdata[0]
    try:
        return int(customdata)
    except (TypeError, ValueError):
        return None


def _projection_parameters(
    *,
    method: str,
    umap_neighbors: int | None,
    umap_min_dist: float | None,
    umap_metric: str | None,
    umap_spread: float | None,
    umap_repulsion: float | None,
    umap_random_state: int | None,
    pca_solver: str | None,
    pca_whiten: list[str] | None,
    pca_random_state: int | None,
    svd_algorithm: str | None,
    svd_n_iter: int | None,
    svd_random_state: int | None,
) -> dict[str, Any]:
    """Validate projection-form fields and return reducer parameters."""
    if method == "umap":
        neighbors = int(umap_neighbors or 15)
        min_dist = float(0.05 if umap_min_dist is None else umap_min_dist)
        spread = float(2.5 if umap_spread is None else umap_spread)
        repulsion = float(1.5 if umap_repulsion is None else umap_repulsion)
        if neighbors < 2:
            raise ValueError("UMAP neighbors must be at least 2.")
        if min_dist < 0:
            raise ValueError("UMAP minimum distance cannot be negative.")
        if spread <= 0:
            raise ValueError("UMAP spread must be positive.")
        if min_dist > spread:
            raise ValueError("UMAP minimum distance cannot exceed spread.")
        if repulsion < 0:
            raise ValueError("UMAP repulsion strength cannot be negative.")
        return {
            "n_neighbors": neighbors,
            "min_dist": min_dist,
            "metric": str(umap_metric or "cosine"),
            "spread": spread,
            "repulsion_strength": repulsion,
            "random_state": int(umap_random_state or 0),
        }
    if method == "pca":
        return {
            "svd_solver": str(pca_solver or "auto"),
            "whiten": "whiten" in (pca_whiten or []),
            "random_state": int(pca_random_state or 0),
        }
    if method in {"truncated_svd", "svd"}:
        n_iter = int(svd_n_iter or 5)
        if n_iter < 1:
            raise ValueError("Truncated-SVD power iterations must be at least 1.")
        return {
            "algorithm": str(svd_algorithm or "randomized"),
            "n_iter": n_iter,
            "random_state": int(svd_random_state or 0),
        }
    raise ValueError(f"Unknown projection method: {method!r}")


_MEMO_REFERENCE_PATTERN = re.compile(r"\[\[unit:(\d+)\]\]")
_MEMO_HASHTAG_PATTERN = re.compile(r"(?<!\w)#([A-Za-z0-9_-]+)")


def _memo_options(memos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"label": str(memo["title"]), "value": int(memo["memo_id"])}
        for memo in memos
    ]


def _extract_memo_metadata(body: str) -> tuple[list[str], list[int]]:
    hashtags = list(dict.fromkeys(_MEMO_HASHTAG_PATTERN.findall(body)))
    references = list(
        dict.fromkeys(int(value) for value in _MEMO_REFERENCE_PATTERN.findall(body))
    )
    return hashtags, references


def _apply_run_options(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": (
                f"Draft {run['apply_run_id']} · {run['status']} · "
                f"{run['reviewed_count']}/{run['proposal_count']} decided"
            ),
            "value": int(run["apply_run_id"]),
        }
        for run in runs
    ]



def _geco_index_string() -> str:
    """Return Dash's HTML shell with the GeCo gecko favicon."""
    return """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        <link rel="icon" type="image/svg+xml" href="/assets/geco-favicon.svg">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


def _build_metadata_profiles(
    units: list[dict[str, Any]], columns: list[str]
) -> dict[str, dict[str, Any]]:
    """Infer practical UI types for explicitly supplied metadata columns."""
    profiles: dict[str, dict[str, Any]] = {}
    for column in columns:
        values = [unit.get("metadata", {}).get(column) for unit in units]
        nonmissing = [value for value in values if not _is_missing(value)]
        kind = "text"
        options: list[Any] = []
        if nonmissing and all(isinstance(value, bool) for value in nonmissing):
            kind = "boolean"
        elif nonmissing and all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            for value in nonmissing
        ):
            kind = "numeric"
        elif nonmissing and all(_coerce_datetime(value) is not None for value in nonmissing):
            kind = "datetime"
        else:
            unique = sorted({str(value) for value in nonmissing}, key=str.casefold)
            unique_ratio = len(unique) / max(1, len(nonmissing))
            max_length = max((len(value) for value in unique), default=0)
            if len(unique) <= 100 and (unique_ratio <= 0.35 or max_length <= 80):
                kind = "categorical"
                options = unique
        profiles[column] = {
            "field": column,
            "type": kind,
            "options": options,
            "nonmissing_count": len(nonmissing),
        }
    return profiles


def _normalize_boolean_operator(value: Any) -> str:
    operator = str(value or "and").lower()
    return operator if operator in {"and", "or", "xor"} else "and"


def _normalize_metadata_filters(
    filters: Any,
    profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop malformed filter clauses while preserving stable clause IDs."""
    normalized: list[dict[str, Any]] = []
    if not isinstance(filters, list):
        return normalized
    for raw in filters:
        if not isinstance(raw, dict):
            continue
        field = str(raw.get("field", ""))
        if field not in profiles:
            continue
        profile = profiles[field]
        normalized.append(
            {
                "id": str(raw.get("id") or uuid.uuid4().hex),
                "field": field,
                "type": str(profile["type"]),
                "operator": str(raw.get("operator", "")),
                "value": raw.get("value"),
                "value2": raw.get("value2"),
            }
        )
    return normalized



def _explore_uncoded_mask(
    project: GeometricCoder,
    units: list[dict[str, Any]],
    *,
    enabled: bool,
) -> np.ndarray:
    """Return atomic-unit eligibility for corpus-level coding coverage.

    An atomic point counts as coded only when it currently has at least one
    positive/Present assignment under any code. Negative/Absent and Unsure
    judgments do not satisfy coverage because no substantive code has been
    applied to the point.
    """
    if not enabled:
        return np.ones(len(units), dtype=bool)
    coded_unit_ids = {
        int(annotation["unit_id"])
        for annotation in project.database.current_annotations()
        if annotation.get("unit_id") is not None
        and str(annotation.get("value")) == "positive"
    }
    return np.asarray(
        [
            int(unit_id) not in coded_unit_ids
            for unit_id in (
                range(1, len(units) + 1)
                if isinstance(units, _LazyUnits)
                else (unit["unit_id"] for unit in units)
            )
        ],
        dtype=bool,
    )

def _sql_metadata_page(
    project: GeometricCoder,
    *,
    total_units: int,
    filters: list[dict[str, Any]],
    operator: str | None,
    uncoded_only: bool,
    page_size: int,
    page_number: int,
) -> tuple[np.ndarray, PageSlice, int]:
    """Page filtered units in SQLite; preserve bounded random unfiltered pages."""
    if not filters and not uncoded_only:
        page = deterministic_page(
            total_units, page_size=page_size,
            page_index=max(0, int(page_number) - 1), seed=0,
        )
        return page.positions, page, total_units
    positions, count, selected = project.database.filtered_unit_page(
        filters, operator=operator, uncoded_only=uncoded_only,
        page_size=page_size, page_number=page_number,
    )
    result = np.asarray(positions, dtype=int)
    page = PageSlice(
        positions=result, page_index=selected - 1,
        page_count=deterministic_page_count(count, page_size=page_size),
        page_size=page_size,
    )
    return result, page, count


def _metadata_filtered_page(
    units: list[dict[str, Any]],
    filters: list[dict[str, Any]],
    *,
    operator: str | None,
    page_size: int,
    page_number: int,
    additional_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, Any, np.ndarray, list[str]]:
    """Return one deterministic page drawn only from currently matching units."""
    mask, errors = _evaluate_metadata_filters(units, filters, operator=operator)
    if additional_mask is not None:
        extra = np.asarray(additional_mask, dtype=bool)
        if extra.shape != mask.shape:
            raise ValueError("additional_mask must align with units")
        mask = mask & extra
    # Unfiltered browsing must not build a corpus-sized positions array.
    if not filters and (additional_mask is None or bool(np.all(additional_mask))):
        page = deterministic_page(
            len(units), page_size=page_size,
            page_index=max(0, int(page_number) - 1), seed=0,
        )
        return page.positions, page, mask, errors
    matching_positions = np.flatnonzero(mask)
    page = deterministic_page(
        len(matching_positions),
        page_size=page_size,
        page_index=max(0, int(page_number) - 1),
        seed=0,
    )
    positions = matching_positions[page.positions]
    return positions, page, mask, errors


def _metadata_operator_options(kind: str) -> list[dict[str, str]]:
    labels: dict[str, list[tuple[str, str]]] = {
        "numeric": [
            ("Less than", "lt"),
            ("Less than or equal", "lte"),
            ("Greater than", "gt"),
            ("Greater than or equal", "gte"),
            ("Equal to", "eq"),
            ("Not equal to", "neq"),
            ("Between", "between"),
            ("Is missing", "missing"),
            ("Is not missing", "not_missing"),
        ],
        "text": [
            ("Contains", "contains"),
            ("Does not contain", "excludes"),
            ("Equals", "eq"),
            ("Does not equal", "neq"),
            ("Starts with", "starts"),
            ("Ends with", "ends"),
            ("Regular expression", "regex"),
            ("Is missing", "missing"),
            ("Is not missing", "not_missing"),
        ],
        "categorical": [
            ("Is one of", "in"),
            ("Is not one of", "not_in"),
            ("Is missing", "missing"),
            ("Is not missing", "not_missing"),
        ],
        "boolean": [
            ("Is true", "true"),
            ("Is false", "false"),
            ("Is missing", "missing"),
            ("Is not missing", "not_missing"),
        ],
        "datetime": [
            ("Before", "lt"),
            ("On or before", "lte"),
            ("After", "gt"),
            ("On or after", "gte"),
            ("On", "eq"),
            ("Between", "between"),
            ("Is missing", "missing"),
            ("Is not missing", "not_missing"),
        ],
    }
    return [
        {"label": label, "value": value}
        for label, value in labels.get(kind, labels["text"])
    ]


def _metadata_filter_modal(
    *, dcc: Any, html: Any, metadata_profiles: dict[str, dict[str, Any]]
) -> Any:
    fields = [
        {
            "label": f"{field} ({profile['type']})",
            "value": field,
        }
        for field, profile in metadata_profiles.items()
    ]
    first_field = fields[0]["value"] if fields else None
    first_type = (
        str(metadata_profiles[str(first_field)]["type"])
        if first_field is not None
        else "text"
    )
    operators = _metadata_operator_options(first_type)
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Add metadata filter"),
                            html.Button("×", id="metadata-filter-modal-close", n_clicks=0),
                        ],
                        className="modal-header",
                    ),
                    html.P(
                        "Choose a metadata field and a condition. Search filters and "
                        "metadata filters are applied together.",
                        className="modal-help",
                    ),
                    html.Label("Metadata field"),
                    dcc.Dropdown(
                        id="metadata-filter-field",
                        options=fields,
                        value=first_field,
                        clearable=False,
                    ),
                    html.Div(id="metadata-filter-type-help", className="control-status"),
                    html.Label("Condition", className="modal-field-label"),
                    dcc.Dropdown(
                        id="metadata-filter-condition",
                        options=operators,
                        value=(operators[0]["value"] if operators else None),
                        clearable=False,
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Value"),
                                    dcc.Input(
                                        id="metadata-filter-value",
                                        type="text",
                                        className="text-input",
                                    ),
                                ],
                                id="metadata-filter-value-wrap",
                            ),
                            html.Div(
                                [
                                    html.Label("Upper value"),
                                    dcc.Input(
                                        id="metadata-filter-value-second",
                                        type="text",
                                        className="text-input",
                                    ),
                                ],
                                id="metadata-filter-value-second-wrap",
                                style={"display": "none"},
                            ),
                            html.Div(
                                [
                                    html.Label("Values"),
                                    dcc.Dropdown(
                                        id="metadata-filter-category-values",
                                        options=[],
                                        value=[],
                                        multi=True,
                                    ),
                                ],
                                id="metadata-filter-category-wrap",
                                style={"display": "none"},
                            ),
                        ],
                        className="metadata-filter-value-grid",
                    ),
                    html.Div(id="metadata-filter-modal-status", className="modal-status"),
                    html.Div(
                        [
                            html.Button(
                                "Cancel", id="metadata-filter-modal-cancel", n_clicks=0
                            ),
                            html.Button(
                                "Add filter",
                                id="metadata-filter-modal-add",
                                n_clicks=0,
                                className="primary-action",
                            ),
                        ],
                        className="modal-actions",
                    ),
                ],
                className="modal-card filter-modal-card",
            )
        ],
        id="metadata-filter-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _save_filter_set_modal(*, dcc: Any, html: Any) -> Any:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Save filter collection"),
                            html.Button("×", id="save-filter-set-close", n_clicks=0),
                        ],
                        className="modal-header",
                    ),
                    html.Label("Name"),
                    dcc.Input(
                        id="save-filter-set-name",
                        type="text",
                        placeholder="e.g., Public institutions over age 30",
                        className="text-input",
                    ),
                    html.Div(id="save-filter-set-status", className="modal-status"),
                    html.Div(
                        [
                            html.Button("Cancel", id="save-filter-set-cancel", n_clicks=0),
                            html.Button(
                                "Save",
                                id="save-filter-set-confirm",
                                n_clicks=0,
                                className="primary-action",
                            ),
                        ],
                        className="modal-actions",
                    ),
                ],
                className="modal-card filter-modal-card",
            )
        ],
        id="save-filter-set-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _load_filter_set_modal(*, dcc: Any, html: Any) -> Any:
    del dcc
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Saved filter collections"),
                            html.Button("×", id="load-filter-set-close", n_clicks=0),
                        ],
                        className="modal-header",
                    ),
                    html.Div(id="saved-filter-set-list", className="saved-filter-set-list"),
                    html.Div(id="load-filter-set-status", className="modal-status"),
                    html.Div(
                        [html.Button("Close", id="load-filter-set-done", n_clicks=0)],
                        className="modal-actions",
                    ),
                ],
                className="modal-card filter-modal-card",
            )
        ],
        id="load-filter-set-modal",
        className="modal-overlay",
        style={"display": "none"},
    )


def _filter_clause_label(clause: dict[str, Any]) -> str:
    field = str(clause.get("field", ""))
    operator = str(clause.get("operator", ""))
    value = clause.get("value")
    value2 = clause.get("value2")
    symbols = {
        "lt": "<",
        "lte": "≤",
        "gt": ">",
        "gte": "≥",
        "eq": "=",
        "neq": "≠",
        "contains": "contains",
        "excludes": "excludes",
        "starts": "starts with",
        "ends": "ends with",
        "regex": "matches",
        "in": "is one of",
        "not_in": "is not one of",
        "true": "is true",
        "false": "is false",
        "missing": "is missing",
        "not_missing": "is not missing",
    }
    if operator == "between":
        return f"{field}: {value} to {value2}"
    if operator in {"missing", "not_missing", "true", "false"}:
        return f"{field}: {symbols[operator]}"
    if isinstance(value, list):
        rendered = ", ".join(str(item) for item in value)
    else:
        rendered = str(value)
    return f"{field}: {symbols.get(operator, operator)} {rendered}".strip()


def _evaluate_metadata_filters(
    units: list[dict[str, Any]],
    filters: list[dict[str, Any]],
    *,
    operator: str | None,
) -> tuple[np.ndarray, list[str]]:
    """Evaluate metadata clauses and combine them with AND/OR/exactly-one XOR."""
    if not filters:
        return np.ones(len(units), dtype=bool), []
    masks: list[np.ndarray] = []
    errors: list[str] = []
    for clause in filters:
        field = str(clause.get("field", ""))
        kind = str(clause.get("type", "text"))
        condition = str(clause.get("operator", ""))
        value = clause.get("value")
        value2 = clause.get("value2")
        try:
            matches = _prepared_metadata_matcher(
                kind=kind, condition=condition, value=value, value2=value2,
            )
            mask = np.asarray(
                [
                    matches(unit.get("metadata", {}).get(field))
                    for unit in (
                        units.metadata_rows()
                        if isinstance(units, _LazyUnits) else units
                    )
                ],
                dtype=bool,
            )
        except (TypeError, ValueError, re.error) as error:
            errors.append(f"{field}: {error}")
            mask = np.zeros(len(units), dtype=bool)
        masks.append(mask)
    stacked = np.vstack(masks)
    normalized_operator = _normalize_boolean_operator(operator)
    if normalized_operator == "or":
        combined = np.any(stacked, axis=0)
    elif normalized_operator == "xor":
        combined = np.sum(stacked, axis=0) == 1
    else:
        combined = np.all(stacked, axis=0)
    return combined, errors


def _prepared_metadata_matcher(
    *, kind: str, condition: str, value: Any, value2: Any,
) -> Any:
    """Prepare one clause once rather than repeat setup for every unit.

    Defer parsing/regex validation until the first nonmissing value so invalid
    clauses retain the original all-missing behavior and error semantics.
    """
    if condition == "missing":
        return _is_missing
    if condition == "not_missing":
        return lambda raw: not _is_missing(raw)
    if kind == "boolean":
        return lambda raw: (
            False if _is_missing(raw)
            else (bool(raw) if condition == "true" else not bool(raw))
        )
    if kind == "categorical":
        selected = value if isinstance(value, list) else [value]
        choices = {str(item).casefold() for item in selected}
        return lambda raw: (
            False if _is_missing(raw)
            else ((str(raw).casefold() in choices) != (condition == "not_in"))
        )
    if kind == "text":
        target = str(value or "")
        folded = target.casefold()
        expression: re.Pattern[str] | None = None

        def match_text(raw: Any) -> bool:
            nonlocal expression
            if _is_missing(raw):
                return False
            text = str(raw)
            lowered = text.casefold()
            if condition == "contains":
                return folded in lowered
            if condition == "excludes":
                return folded not in lowered
            if condition == "eq":
                return lowered == folded
            if condition == "neq":
                return lowered != folded
            if condition == "starts":
                return lowered.startswith(folded)
            if condition == "ends":
                return lowered.endswith(folded)
            if condition == "regex":
                if expression is None:
                    expression = re.compile(target, flags=re.IGNORECASE)
                return expression.search(text) is not None
            raise ValueError(f"unsupported condition {condition!r}")

        return match_text
    if kind in {"numeric", "datetime"}:
        prepared = False
        right: Any = None
        upper: Any = None

        def match_ordered(raw: Any) -> bool:
            nonlocal prepared, right, upper
            if _is_missing(raw):
                return False
            left = float(raw) if kind == "numeric" else _coerce_datetime(raw)
            if not prepared:
                if kind == "numeric":
                    right = float(value)
                    upper = float(value2) if condition == "between" else None
                else:
                    right = _coerce_datetime(value)
                    upper = _coerce_datetime(value2) if condition == "between" else None
                prepared = True
            if kind == "datetime" and (left is None or right is None):
                raise ValueError("enter a valid date")
            return _ordered_comparison(left, condition, right, upper)

        return match_ordered
    return lambda raw: _metadata_value_matches(
        raw, kind=kind, condition=condition, value=value, value2=value2,
    )


def _metadata_value_matches(
    raw: Any,
    *,
    kind: str,
    condition: str,
    value: Any,
    value2: Any,
) -> bool:
    missing = _is_missing(raw)
    if condition == "missing":
        return missing
    if condition == "not_missing":
        return not missing
    if missing:
        return False
    if kind == "boolean":
        return bool(raw) if condition == "true" else not bool(raw)
    if kind == "numeric":
        left = float(raw)
        right = float(value)
        upper = float(value2) if condition == "between" else None
        return _ordered_comparison(left, condition, right, upper)
    if kind == "datetime":
        left_date = _coerce_datetime(raw)
        right_date = _coerce_datetime(value)
        upper_date = _coerce_datetime(value2) if condition == "between" else None
        if left_date is None or right_date is None:
            raise ValueError("enter a valid date")
        return _ordered_comparison(left_date, condition, right_date, upper_date)
    text = str(raw)
    lowered = text.casefold()
    if kind == "categorical":
        selected = value if isinstance(value, list) else [value]
        selected_values = {str(item).casefold() for item in selected}
        matched = lowered in selected_values
        return not matched if condition == "not_in" else matched
    target = str(value or "")
    target_lower = target.casefold()
    if condition == "contains":
        return target_lower in lowered
    if condition == "excludes":
        return target_lower not in lowered
    if condition == "eq":
        return lowered == target_lower
    if condition == "neq":
        return lowered != target_lower
    if condition == "starts":
        return lowered.startswith(target_lower)
    if condition == "ends":
        return lowered.endswith(target_lower)
    if condition == "regex":
        return re.search(target, text, flags=re.IGNORECASE) is not None
    raise ValueError(f"unsupported condition {condition!r}")


def _ordered_comparison(left: Any, condition: str, right: Any, upper: Any) -> bool:
    if condition == "lt":
        return left < right
    if condition == "lte":
        return left <= right
    if condition == "gt":
        return left > right
    if condition == "gte":
        return left >= right
    if condition == "eq":
        return left == right
    if condition == "neq":
        return left != right
    if condition == "between":
        if upper is None:
            raise ValueError("enter both ends of the range")
        lower_bound, upper_bound = sorted((right, upper))
        return lower_bound <= left <= upper_bound
    raise ValueError(f"unsupported condition {condition!r}")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _coerce_datetime(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or not re.match(r"^\d{4}-\d{1,2}-\d{1,2}", text):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _read_saved_filter_sets(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _write_saved_filter_sets(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_memo_presentation_request(
    search: str | None,
) -> tuple[int | None, int | None]:
    """Parse memo/version identifiers from the presentation URL query string."""
    values = parse_qs((search or "").lstrip("?"))
    try:
        memo_id = int(values.get("memo_id", [""])[0])
    except (TypeError, ValueError):
        memo_id = None
    try:
        version = int(values.get("version", [""])[0])
    except (TypeError, ValueError):
        version = None
    return memo_id, version


def _memo_version_by_number(
    versions: list[dict[str, Any]], requested_version: int | None
) -> dict[str, Any]:
    """Return the requested memo version, falling back to the latest."""
    if not versions:
        raise ValueError("versions must not be empty")
    if requested_version is not None:
        for version in versions:
            if int(version["version_number"]) == int(requested_version):
                return version
    return versions[-1]


def _presentation_unit_id(location_hash: str | None) -> int | None:
    match = re.fullmatch(r"#geco-presentation-unit-(\d+)", location_hash or "")
    return int(match.group(1)) if match is not None else None


def _presentation_filter_definition(
    path: Path,
    saved_filter_id: str | None,
    metadata_profiles: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Resolve one saved metadata filter collection for presentation mode."""
    if not saved_filter_id or saved_filter_id == "__all__":
        return [], "and"
    for record in _read_saved_filter_sets(path):
        if str(record.get("id")) == str(saved_filter_id):
            return (
                _normalize_metadata_filters(
                    record.get("filters", []), metadata_profiles
                ),
                _normalize_boolean_operator(record.get("operator")),
            )
    return [], "and"


def _deterministic_page_number_for_position(
    mask: np.ndarray,
    *,
    corpus_position: int,
    page_size: int,
    seed: int = 0,
) -> int | None:
    """Return the 1-based deterministic page containing one eligible corpus row."""
    eligible_positions = np.flatnonzero(np.asarray(mask, dtype=bool))
    local = np.flatnonzero(eligible_positions == int(corpus_position))
    if local.size != 1:
        return None
    rank = deterministic_page_rank(
        len(eligible_positions), int(local[0]), seed=seed,
    )
    return rank // max(1, int(page_size)) + 1


def _render_memo_preview(
    dcc: Any,
    html: Any,
    project: GeometricCoder,
    body: str,
    *,
    unit_hash_prefix: str = "geco-unit-",
    link_hashtags: bool = True,
) -> Any:
    """Render Markdown while resolving GeCo hashtags and unit references."""
    if not body:
        return html.P("Memo preview will appear here.", className="memo-preview-empty")

    placeholders: dict[str, str] = {}
    reference_ids = [int(match.group(1)) for match in _MEMO_REFERENCE_PATTERN.finditer(body)]
    reference_keys = project.database.unit_keys_for_ids(reference_ids) if reference_ids else {}

    def replace_reference(match: re.Match[str]) -> str:
        unit_id = int(match.group(1))
        user_key = reference_keys.get(unit_id)
        label = (
            " › ".join(f"{name}: {value}" for name, value in user_key.items())
            if user_key is not None
            else f"unit {unit_id}"
        )
        token = f"GECOREFERENCE{len(placeholders)}TOKEN"
        placeholders[token] = (
            f"[{_escape_markdown_label(label)}](#{unit_hash_prefix}{unit_id})"
        )
        return token

    rendered = _MEMO_REFERENCE_PATTERN.sub(replace_reference, body)

    def replace_hashtag(match: re.Match[str]) -> str:
        tag = match.group(1)
        token = f"GECOHASHTAG{len(placeholders)}TOKEN"
        placeholders[token] = (
            f"[#{tag}](#geco-tag-{quote(tag, safe='')})" if link_hashtags else f"#{tag}"
        )
        return token

    rendered = _MEMO_HASHTAG_PATTERN.sub(replace_hashtag, rendered)
    for token, replacement in placeholders.items():
        rendered = rendered.replace(token, replacement)
    return dcc.Markdown(
        rendered,
        dangerously_allow_html=False,
        link_target="_self",
        className="memo-markdown",
    )


def _escape_markdown_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _normalize_navigation_history(
    history: dict[str, Any] | None,
    *,
    focal_unit_id: int | None = None,
) -> dict[str, Any]:
    data = history or {}
    items: list[int] = []
    for value in data.get("items", []):
        try:
            items.append(int(value))
        except (TypeError, ValueError):
            continue
    if not items and focal_unit_id is not None:
        items = [int(focal_unit_id)]
    try:
        index = int(data.get("index", len(items) - 1))
    except (TypeError, ValueError):
        index = len(items) - 1
    index = min(max(index, -1), len(items) - 1)
    return {"items": items, "index": index}


def _append_navigation_history(
    history: dict[str, Any] | None,
    unit_id: int,
) -> dict[str, Any]:
    normalized = _normalize_navigation_history(history)
    items = list(normalized["items"])
    index = int(normalized["index"])
    if index >= 0 and index < len(items) and items[index] == int(unit_id):
        return normalized
    items = items[: index + 1]
    items.append(int(unit_id))
    return {"items": items, "index": len(items) - 1}


def _move_navigation_history(
    history: dict[str, Any] | None,
    direction: int,
) -> tuple[int, dict[str, Any]] | None:
    normalized = _normalize_navigation_history(history)
    items = list(normalized["items"])
    target = int(normalized["index"]) + int(direction)
    if target < 0 or target >= len(items):
        return None
    updated = {"items": items, "index": target}
    return int(items[target]), updated


def _normalize_explore_code_palette(
    raw: Any,
    valid_code_ids: set[int],
) -> list[int | None]:
    """Return a duplicate-free Explore coding palette with one permanent row."""
    values = raw if isinstance(raw, list) else []
    normalized: list[int | None] = []
    seen: set[int] = set()
    for value in values:
        if value is None:
            normalized.append(None)
            continue
        try:
            code_id = int(value)
        except (TypeError, ValueError):
            normalized.append(None)
            continue
        if code_id not in valid_code_ids or code_id in seen:
            normalized.append(None)
            continue
        normalized.append(code_id)
        seen.add(code_id)
    return normalized or [None]


def _explore_palette_for_row_count(
    raw: Any,
    valid_code_ids: set[int],
    row_count: int | None,
) -> list[int | None]:
    """Return normalized palette values padded/truncated to the visible row count."""
    palette = _normalize_explore_code_palette(raw, valid_code_ids)
    target = max(1, int(row_count or len(palette) or 1))
    if len(palette) < target:
        return [*palette, *([None] * (target - len(palette)))]
    return palette[:target]


def _place_code_in_explore_palette(
    raw: Any,
    code_id: int,
    valid_code_ids: set[int],
) -> list[int | None]:
    """Place a code into the first empty Explore row, appending when needed."""
    palette = _normalize_explore_code_palette(raw, valid_code_ids)
    selected = {int(value) for value in palette if value is not None}
    code_id = int(code_id)
    if code_id in selected:
        return palette
    for index, value in enumerate(palette):
        if value is None:
            updated = list(palette)
            updated[index] = code_id
            return updated
    return [*palette, code_id]


def _judgment_button_classes(
    current: str | None,
    *,
    prefix: str,
) -> tuple[str, str, str]:
    del prefix
    base = "judgment-button"
    return (
        f"{base} positive-button" + (" judgment-selected" if current == "positive" else ""),
        f"{base} negative-button" + (" judgment-selected" if current == "negative" else ""),
        f"{base} unsure-button" + (" judgment-selected" if current == "unsure" else ""),
    )



def _render_readonly_context_units(
    html: Any,
    units: list[dict[str, Any]],
    *,
    focal_unit_id: int,
    key_columns: list[str],
) -> list[Any]:
    """Render hierarchy-aware context without coding/span controls."""
    if not units:
        return []
    parent_columns = key_columns[:-1]
    children: list[Any] = []
    open_values: list[Any] = []
    for unit in units:
        path = [unit["user_key"].get(column) for column in parent_columns]
        shared = 0
        while (
            shared < len(open_values)
            and shared < len(path)
            and open_values[shared] == path[shared]
        ):
            shared += 1
        for level in range(len(open_values) - 1, shared - 1, -1):
            children.append(
                _context_boundary(
                    html,
                    parent_columns[level],
                    open_values[level],
                    boundary="end",
                    level=level,
                )
            )
        open_values = open_values[:shared]
        for level in range(shared, len(path)):
            open_values.append(path[level])
            children.append(
                _context_boundary(
                    html,
                    parent_columns[level],
                    path[level],
                    boundary="start",
                    level=level,
                )
            )
        unit_id = int(unit["unit_id"])
        is_focal = unit_id == int(focal_unit_id)
        children.append(
            html.P(
                [
                    html.Strong(unit["text"], className="focal-text")
                    if is_focal
                    else unit["text"]
                ],
                className=(
                    "context-unit focal-context-unit" if is_focal else "context-unit"
                ),
            )
        )
    for level in range(len(open_values) - 1, -1, -1):
        children.append(
            _context_boundary(
                html,
                parent_columns[level],
                open_values[level],
                boundary="end",
                level=level,
            )
        )
    return children


def _render_context_units(
    html: Any,
    dcc: Any,
    units: list[dict[str, Any]],
    *,
    focal_unit_id: int,
    key_columns: list[str],
    selected_unit_ids: list[int] | None = None,
    checkbox_type: str,
    span_selection_enabled: bool = True,
) -> list[Any]:
    """Render context with a focal-anchored contiguous span selector."""
    if not units:
        return []
    parent_columns = key_columns[:-1]
    selected = set(_normalized_span_selection(units, focal_unit_id, selected_unit_ids))
    children: list[Any] = []
    open_values: list[Any] = []

    for unit in units:
        path = [unit["user_key"].get(column) for column in parent_columns]
        shared = 0
        while (
            shared < len(open_values)
            and shared < len(path)
            and open_values[shared] == path[shared]
        ):
            shared += 1
        for level in range(len(open_values) - 1, shared - 1, -1):
            children.append(
                _context_boundary(
                    html, parent_columns[level], open_values[level],
                    boundary="end", level=level,
                )
            )
        open_values = open_values[:shared]
        for level in range(shared, len(path)):
            open_values.append(path[level])
            children.append(
                _context_boundary(
                    html, parent_columns[level], path[level],
                    boundary="start", level=level,
                )
            )

        unit_id = int(unit["unit_id"])
        is_focal = unit_id == int(focal_unit_id)
        text_child: Any = (
            html.Strong(unit["text"], className="focal-text")
            if is_focal else unit["text"]
        )
        checkbox = dcc.Checklist(
            id={"type": checkbox_type, "unit_id": unit_id},
            options=[
                {
                    "label": "",
                    "value": unit_id,
                    "disabled": is_focal or not span_selection_enabled,
                }
            ],
            value=[unit_id] if unit_id in selected else [],
            className="span-unit-checkbox",
        )
        children.append(
            html.Div(
                [
                    checkbox,
                    html.P(
                        [text_child],
                        className=(
                            "context-unit focal-context-unit"
                            if is_focal else "context-unit"
                        ),
                    ),
                ],
                className=(
                    "context-unit-row span-selected"
                    if unit_id in selected else "context-unit-row"
                ),
            )
        )

    for level in range(len(open_values) - 1, -1, -1):
        children.append(
            _context_boundary(
                html, parent_columns[level], open_values[level],
                boundary="end", level=level,
            )
        )
    return children


def _normalized_span_selection(
    context_units: list[dict[str, Any]],
    focal_unit_id: int,
    selected_unit_ids: list[int] | None,
) -> list[int]:
    """Return one visible contiguous interval anchored on the focal unit."""
    order = [int(unit["unit_id"]) for unit in context_units]
    if int(focal_unit_id) not in order:
        return [int(focal_unit_id)]
    focal_index = order.index(int(focal_unit_id))
    selected = {int(value) for value in (selected_unit_ids or []) if int(value) in order}
    selected.add(int(focal_unit_id))
    indices = [order.index(value) for value in selected]
    start = min(min(indices), focal_index)
    end = max(max(indices), focal_index)
    return order[start : end + 1]


def _update_span_selection(
    context_units: list[dict[str, Any]],
    *,
    focal_unit_id: int,
    current_unit_ids: list[int] | None,
    clicked_unit_id: int,
    checked: bool,
) -> list[int]:
    """Move one endpoint of the focal-anchored discrete interval."""
    order = [int(unit["unit_id"]) for unit in context_units]
    focal = int(focal_unit_id)
    clicked = int(clicked_unit_id)
    current = _normalized_span_selection(context_units, focal, current_unit_ids)
    if clicked not in order or clicked == focal:
        return current
    focal_index = order.index(focal)
    clicked_index = order.index(clicked)
    current_indices = [order.index(value) for value in current]
    start, end = min(current_indices), max(current_indices)
    if checked:
        if clicked_index < focal_index:
            start = clicked_index
        else:
            end = clicked_index
    elif clicked in current:
        if clicked_index < focal_index:
            start = clicked_index + 1
        else:
            end = clicked_index - 1
    return order[start : end + 1]


def _selection_observation_id(
    project: GeometricCoder,
    focal_unit_id: int,
    selected_unit_ids: list[int] | None,
) -> int | None:
    """Resolve the active atomic observation or an already-persisted span."""
    selected = [int(value) for value in (selected_unit_ids or [focal_unit_id])]
    if len(selected) <= 1:
        return project.observation_id_for_unit(int(focal_unit_id))
    span = project.span_observation(selected)
    return int(span["observation_id"]) if span is not None else None


def _context_boundary(
    html: Any,
    column: str,
    value: Any,
    *,
    boundary: str,
    level: int,
) -> Any:
    return html.Div(
        f"— {column}: {value} {boundary} —",
        className=f"context-boundary context-boundary-{boundary}",
        style={"--context-level": str(level)},
    )


def _current_annotation_value(
    project: GeometricCoder, observation_id: int | None, code_id: int
) -> str | None:
    if observation_id is None:
        return None
    annotation = project.current_annotation(
        observation_id=int(observation_id), code_id=int(code_id)
    )
    return str(annotation["value"]) if annotation is not None else None


def _geometry_overlay_frame(figure: go.Figure) -> dict[str, Any] | None:
    """Retain only plotted point identities and coordinates for overlay patches.

    The complete Plotly figure never needs to travel back from the browser just
    to update visited and focal outlines.  Capture the two base traces while
    they are built, not their colors, styling, or layout/camera.
    """
    if len(figure.data) <= EXPLORE_FOCAL_TRACE_INDEX:
        return None
    names = tuple(str(trace.name) for trace in figure.data[:4])
    if names != EXPLORE_TRACE_NAMES:
        return None
    data: list[dict[str, Any]] = []
    for index, trace in enumerate(figure.data[:4]):
        row: dict[str, Any] = {"name": trace.name}
        if index in {EXPLORE_FILTERED_TRACE_INDEX, EXPLORE_ELIGIBLE_TRACE_INDEX}:
            for field in ("x", "y", "customdata", "hovertext"):
                value = getattr(trace, field)
                row[field] = list(value) if value is not None else []
        data.append(row)
    return {"data": data}


def _geometry_overlay_payload(
    figure: dict[str, Any] | None,
    *,
    seen_unit_ids: set[int],
    focal_unit_id: int | None,
) -> dict[int, dict[str, list[Any]]] | None:
    """Build lightweight visited/focal overlay arrays from the current base map."""
    if not figure or not isinstance(figure.get("data"), list):
        return None
    data = figure["data"]
    if len(data) <= EXPLORE_FOCAL_TRACE_INDEX:
        return None
    names = tuple(str(data[index].get("name", "")) for index in range(4))
    if names != EXPLORE_TRACE_NAMES:
        return None

    visible: list[tuple[int, Any, Any, Any]] = []
    for trace_index in (EXPLORE_FILTERED_TRACE_INDEX, EXPLORE_ELIGIBLE_TRACE_INDEX):
        trace = data[trace_index]
        xs = list(trace.get("x") or [])
        ys = list(trace.get("y") or [])
        ids = list(trace.get("customdata") or [])
        hover = list(trace.get("hovertext") or [])
        for position, raw_unit_id in enumerate(ids):
            try:
                unit_id = int(raw_unit_id)
            except (TypeError, ValueError):
                continue
            if position >= len(xs) or position >= len(ys):
                continue
            hover_value = hover[position] if position < len(hover) else ""
            visible.append((unit_id, xs[position], ys[position], hover_value))

    seen = {int(value) for value in seen_unit_ids}
    visited_points = [point for point in visible if point[0] in seen]
    focal_points = (
        [point for point in visible if point[0] == int(focal_unit_id)]
        if focal_unit_id is not None
        else []
    )

    def columns(points: list[tuple[int, Any, Any, Any]]) -> dict[str, list[Any]]:
        return {
            "x": [point[1] for point in points],
            "y": [point[2] for point in points],
            "customdata": [point[0] for point in points],
            "hovertext": [point[3] for point in points],
        }

    return {
        EXPLORE_VISITED_TRACE_INDEX: columns(visited_points),
        EXPLORE_FOCAL_TRACE_INDEX: columns(focal_points),
    }


def _apply_relayout_ranges(
    figure: Any, relayout_data: dict[str, Any] | None
) -> None:
    """Carry the current Plotly camera through point-selection redraws."""
    if not relayout_data:
        return
    x_range = relayout_data.get("xaxis.range")
    y_range = relayout_data.get("yaxis.range")
    x0 = relayout_data.get("xaxis.range[0]")
    x1 = relayout_data.get("xaxis.range[1]")
    y0 = relayout_data.get("yaxis.range[0]")
    y1 = relayout_data.get("yaxis.range[1]")
    if isinstance(x_range, (list, tuple)) and len(x_range) == 2:
        x0, x1 = x_range
    if isinstance(y_range, (list, tuple)) and len(y_range) == 2:
        y0, y1 = y_range
    if x0 is not None and x1 is not None:
        figure.update_xaxes(range=[float(x0), float(x1)], autorange=False)
    if y0 is not None and y1 is not None:
        figure.update_yaxes(range=[float(y0), float(y1)], autorange=False)


def _observation_display_key(
    project: GeometricCoder, observation: dict[str, Any]
) -> str:
    kind = str(observation.get("kind", "atomic"))
    if kind == "atomic":
        return " › ".join(
            f"{name}: {value}"
            for name, value in observation.get("user_key", {}).items()
        )
    if kind == "teaching_example":
        return f"Teaching example {int(observation['observation_id']):,}"
    members = observation.get("member_unit_ids")
    if members is None:
        full = project.observation(int(observation["observation_id"]))
        members = full.get("member_unit_ids", [])
    member_ids = [int(value) for value in members]
    if not member_ids:
        return f"Span {int(observation['observation_id']):,}"
    first = project.unit(member_ids[0])
    last = project.unit(member_ids[-1])
    parent_key = list(first["user_key"].items())[:-1]
    parent = " › ".join(f"{name}: {value}" for name, value in parent_key)
    atomic_name = next(reversed(first["user_key"]))
    first_value = first["user_key"][atomic_name]
    last_value = last["user_key"][atomic_name]
    interval = (
        f"{atomic_name}: {first_value}"
        if first_value == last_value
        else f"{atomic_name}: {first_value}–{last_value}"
    )
    return f"{parent} › {interval}" if parent else f"Span {interval}"


def _apply_rows_match_draft(
    rows: list[dict[str, Any]] | None, draft_id: int,
) -> bool:
    """Reject AG Grid rows left behind when a different Apply draft is selected."""
    try:
        return all(int(row["apply_run_id"]) == int(draft_id) for row in (rows or []))
    except (KeyError, TypeError, ValueError):
        return False


def _prediction_geometry_matches_code(
    refresh_state: dict[str, Any] | None, code_id: int | None,
) -> bool:
    """Reject a prediction snapshot rendered for a different Develop code."""
    if not isinstance(refresh_state, dict) or code_id is None:
        return False
    try:
        return int(refresh_state["code_id"]) == int(code_id)
    except (KeyError, TypeError, ValueError):
        return False


def _compact_prediction_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    """Keep only the already computed display data in the short-lived plot cache.

    Projection and PCA are computed against every selected immutable fit *before*
    compaction. Individual point texts and score dictionaries are not used by
    the prediction plot, so retaining them four times wastes memory.
    """
    for point in geometry.get("points", []):
        point.pop("text", None)
        point.pop("scores", None)
    return geometry


def _prediction_geometry_from_cache(
    cache: dict[str, dict[str, Any]],
    refresh_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve one server-side prediction-geometry cache entry."""
    if not refresh_state:
        return {}
    key = str(refresh_state.get("cache_key", ""))
    return cache.get(key, {})


def _prediction_geometry_visible_points(
    geometry: dict[str, Any],
    display_options: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter cached prediction points by user-selected observation kinds."""
    options = set(display_options or [])
    allowed = {"atomic"}
    if "spans" in options:
        allowed.add("span")
    if "teaching" in options:
        allowed.add("teaching_example")
    return [
        point
        for point in geometry.get("points", [])
        if str(point.get("kind")) in allowed
    ]


def _prediction_geometry_page_index(figure: go.Figure) -> dict[int, tuple[int, int]]:
    """Locate plotted observation IDs for small, identity-checked point patches."""
    points: dict[int, tuple[int, int]] = {}
    for trace_index, trace in enumerate(figure.data):
        custom = getattr(trace, "customdata", None)
        if custom is None:
            continue
        for position, value in enumerate(custom):
            if isinstance(value, (list, tuple, np.ndarray)) and len(value):
                points[int(value[0])] = (trace_index, position)
    return points


def _matching_prediction_page_point(
    page: tuple[int, str, dict[int, tuple[int, int]]] | None,
    *, code_id: int | None, refresh_state: dict[str, Any] | None,
    observation_id: int,
) -> tuple[int, int] | None:
    """Resolve a plotted point only for the active frozen score snapshot."""
    if page is None or not _prediction_geometry_matches_code(refresh_state, code_id):
        return None
    if page[0] != int(code_id) or page[1] != str(refresh_state.get("cache_key")):
        return None
    return page[2].get(int(observation_id))


def _build_prediction_geometry_figure(
    *,
    geometry: dict[str, Any],
    points: list[dict[str, Any]],
    page_index: int,
) -> Any:
    """Build the paged classifier-score map with stable type and label encodings."""
    color_by_assignment = {
        None: "#171b18",
        "positive": "#3f8d5d",
        "negative": "#a83f3f",
        "unsure": "#d39a22",
    }
    label_by_assignment = {
        None: "Unreviewed",
        "positive": "Present",
        "negative": "Absent",
        "unsure": "Unsure",
    }
    symbol_by_kind = {
        "atomic": "circle",
        "span": "triangle-up",
        "teaching_example": "star",
    }
    kind_label = {
        "atomic": "Atomic",
        "span": "Span",
        "teaching_example": "Teaching example",
    }
    figure = go.Figure()

    # Assignment-color legend. The real points are grouped by observation kind so
    # changing a judgment does not create Plotly's persistent selected-point fade.
    for assignment in (None, "positive", "negative", "unsure"):
        figure.add_trace(
            go.Scattergl(
                x=[None],
                y=[None],
                mode="markers",
                name=label_by_assignment[assignment],
                hoverinfo="skip",
                marker={
                    "size": 8,
                    "symbol": "circle",
                    "color": color_by_assignment[assignment],
                },
                showlegend=True,
            )
        )

    for kind in ("atomic", "span", "teaching_example"):
        selected = [point for point in points if str(point.get("kind")) == kind]
        if not selected:
            continue
        colors = [color_by_assignment[point.get("assignment")] for point in selected]
        figure.add_trace(
            go.Scattergl(
                x=[float(point["x"]) for point in selected],
                y=[float(point["y"]) for point in selected],
                mode="markers",
                name=kind_label[kind],
                customdata=[
                    [
                        int(point["observation_id"]),
                        (
                            int(point["unit_id"])
                            if point.get("unit_id") is not None
                            else None
                        ),
                        kind,
                    ]
                    for point in selected
                ],
                hovertext=[
                    str(point.get("display_key") or kind_label[kind])
                    for point in selected
                ],
                hovertemplate="%{hovertext}<extra></extra>",
                marker={
                    "size": 8 if kind == "atomic" else 10,
                    "symbol": symbol_by_kind[kind],
                    "color": colors,
                    "opacity": 0.9,
                    "line": {"color": colors, "width": 0.7},
                },
                showlegend=True,
            )
        )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 55, "r": 20, "t": 30, "b": 75},
        hovermode="closest",
        hoverdistance=8,
        clickmode="event",
        dragmode="zoom",
        uirevision=(
            "prediction-geometry:"
            + ",".join(str(value) for value in geometry.get("classifier_fit_ids", []))
            + f":page:{page_index}"
        ),
        xaxis_title=str(geometry["axis_titles"][0]),
        yaxis_title=str(geometry["axis_titles"][1]),
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def _metadata_table(html: Any, metadata: dict[str, Any]) -> Any:
    if not metadata:
        return html.P("No metadata columns were supplied.")
    rows = []
    for key, value in metadata.items():
        rows.append(html.Tr([html.Th(str(key)), html.Td(str(value))]))
    return html.Table([html.Tbody(rows)], className="metadata-table")


def _code_center_placeholder_figure() -> Any:
    """Return a Patch-safe initial Code Center figure.

    The focal-marker callback partially updates ``data[0]``. Dash 4.x raises
    ``Cannot patch undefined`` if either ``figure`` or that trace is absent
    when the partial update arrives, so the layout must provide the exact
    nested structure that callback patches.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=[],
            y=[],
            mode="markers",
            name="Focal observation",
            customdata=[],
            hovertext=[],
            hovertemplate="%{hovertext}<extra></extra>",
            marker={
                "size": 13,
                "color": "rgba(0, 0, 0, 0)",
                "line": {"color": "#178547", "width": 2.6},
            },
            showlegend=False,
        )
    )
    figure.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return figure


def _empty_figure(message: str) -> Any:
    import plotly.graph_objects as go

    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, showarrow=False)
    figure.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return figure


def _local_browser_url(host: str, port: int) -> str:
    """Return a browser-usable local URL for a Dash bind host and port."""
    browser_host = str(host).strip() or "127.0.0.1"
    if browser_host in {"0.0.0.0", "::", "[::]"}:
        browser_host = "127.0.0.1"
    elif ":" in browser_host and not browser_host.startswith("["):
        browser_host = f"[{browser_host}]"
    return f"http://{browser_host}:{int(port)}/"


def _display_local_browser_url(host: str, port: int) -> str:
    """Display the local browser URL without changing Dash notebook embedding."""
    url = _local_browser_url(host, port)
    try:
        from IPython import get_ipython

        shell = get_ipython()
    except Exception:
        shell = None

    if shell is not None:
        try:
            from IPython.display import HTML, display

            safe_url = html_escape(url, quote=True)
            display(
                HTML(
                    "<div style='margin:0.25rem 0 0.5rem'>"
                    f"<div><strong>GeCo is running at</strong> "
                    f"<a href='{safe_url}' target='_blank' rel='noopener noreferrer'>{safe_url}</a> "
                    "<span style='color:#6b756e'>(local browser URL)</span></div>"
                    f"<div><a href='{safe_url}' target='_blank' rel='noopener noreferrer'>"
                    "Open in browser</a></div>"
                    "</div>"
                )
            )
            return url
        except Exception:
            # URL visibility should never prevent the application from launching.
            pass

    print(f"GeCo is running at {url} (local browser URL)")
    return url


def _should_warm_semantic_runtime(project: GeometricCoder, *, mode: str) -> bool:
    """Only prewarm for standard workspaces with external queryable geometry."""
    if mode != "standard" or not getattr(project, "is_external_backed", False):
        return False
    return any(
        str(record.get("storage_kind")) == "external"
        and bool(record.get("supports_query"))
        for record in project.geometries(public_only=True)
    )


def _warm_semantic_runtime_libraries() -> bool:
    """Pay process-wide transformer import and first CPU-kernel costs at launch.

    This does not reconstruct a frozen TeAL pipeline, load a particular model,
    encode real text, or save a semantic query. Actual per-model initialization
    still occurs on first use, preserving the provider's existing thread-local
    project and model contracts.
    """
    import importlib
    from time import perf_counter

    start = perf_counter()
    print("GeCo: preparing semantic-search libraries...", flush=True)
    try:
        torch = importlib.import_module("torch")
        importlib.import_module("sentence_transformers")
        # A minimal disposable similarity computation initializes the shared
        # PyTorch CPU backend without consuming RNG state or using a model.
        with torch.inference_mode():
            query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
            _ = torch.matmul(query, query.T)
    except Exception as error:
        # Optional transformer resources must never prevent unrelated GeCo
        # capabilities (or non-transformer semantic search) from launching.
        # Backend initialization can raise errors beyond ImportError/OSError.
        print(f"GeCo: semantic-library warm-up skipped ({error}).", flush=True)
        return False
    print(f"GeCo: semantic-search libraries ready ({perf_counter() - start:.1f}s).", flush=True)
    return True


def launch_app(
    project: GeometricCoder,
    *,
    host: str,
    port: int,
    debug: bool,
    use_reloader: bool = False,
    mode: str = "standard",
    focus_session_id: int | None = None,
) -> Any:
    """Start one closeable Dash server and return its notebook-friendly handle."""
    if use_reloader:
        raise ValueError("The closeable GeCo server does not support use_reloader=True.")
    from geometric_coder.ui.server import GeCoServer

    app = create_app(project, mode=mode, focus_session_id=focus_session_id)
    if _should_warm_semantic_runtime(project, mode=mode):
        _warm_semantic_runtime_libraries()
    server = GeCoServer(app, host=host, port=port)
    server.start()
    _display_local_browser_url(host, server.port)
    return server
