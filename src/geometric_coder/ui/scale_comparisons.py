"""Manual Scale comparisons inside the existing Develop workspace.

The browser only requests one small page of atomic units. The persisted task
stores stable observation IDs, not text, row positions, or candidate pairs.
"""
from __future__ import annotations

import json
import secrets
import uuid
from typing import Any


_PAGE_SIZE = 20
_HISTORY_SIZE = 60
_OUTCOMES = {
    "scale-judge-first": "first_higher",
    "scale-judge-parity": "parity",
    "scale-judge-second": "second_higher",
    "scale-judge-unsure": "unsure",
}
_OUTCOME_LABELS = {
    "first_higher": "A is higher",
    "parity": "Approximately equal",
    "second_higher": "B is higher",
    "unsure": "Unsure / Skip",
}


def _scale_task_snapshot(project: Any, scale_id: int) -> dict[str, Any]:
    """Freeze both definition and eligibility dependency revisions for one task."""
    scale = project.scale(scale_id)
    eligibility = project.scale_eligibility(scale_id)
    return {
        "version_number": int(scale["version_number"]),
        "eligibility": json.loads(json.dumps(eligibility, sort_keys=True)),
    }


def _restore_scale_task(
    project: Any, scale_id: int, session_id: int, stored: Any,
) -> dict[str, Any]:
    """Reject malformed or cross-session state, retaining valid stale tasks for review."""
    empty = {
        "session_id": int(session_id), "scale_id": int(scale_id),
        "picked_a": None, "picked_b": None, "status": "selecting",
    }
    if not isinstance(stored, dict):
        return empty
    if (stored.get("session_id") != session_id
            or stored.get("scale_id") != scale_id
            or stored.get("status") not in {"selecting", "ready", "saved"}):
        return empty
    try:
        picked = [
            int(stored[key]) for key in ("picked_a", "picked_b")
            if stored.get(key) is not None
        ]
        presented = [
            int(stored[key]) for key in ("first", "second")
            if stored.get(key) is not None
        ]
        candidates = set(picked + presented)
        if len(candidates) > 2:
            return empty
        if any(project.observation(item)["kind"] != "atomic" for item in candidates):
            return empty
        if stored["status"] in {"ready", "saved"}:
            if (len(presented) != 2 or presented[0] == presented[1]
                    or not isinstance(stored.get("submission_key"), str)
                    or not stored.get("snapshot")):
                return empty
    except (TypeError, ValueError, KeyError):
        return empty
    return stored


def _prepare_scale_task(
    project: Any, task: dict[str, Any], reviewer: str | None,
) -> dict[str, Any]:
    scale_id = int(task["scale_id"])
    first = task.get("picked_a")
    second = task.get("picked_b")
    if first is None or second is None:
        raise ValueError("Choose two observations before preparing a comparison.")
    first, second = int(first), int(second)
    if first == second:
        raise ValueError("A Scale comparison requires two different observations.")
    eligible = project.eligible_scale_observation_ids(scale_id, [first, second])
    if set(eligible) != {first, second}:
        raise ValueError("Both observations must satisfy this Scale's current eligibility.")
    if secrets.randbelow(2):
        first, second = second, first
    return {
        **task,
        "first": first, "second": second,
        "snapshot": _scale_task_snapshot(project, scale_id),
        "submission_key": uuid.uuid4().hex,
        "reviewer_key": (reviewer or "").strip() or None,
        "mode": "independent",
        "supersedes_event_id": None,
        "status": "ready",
    }


def _random_scale_task(
    project: Any, scale_id: int, session_id: int, total_units: int,
) -> dict[str, Any]:
    """Prepare a random eligible pair without loading the corpus or fitting models.

    Draw bounded batches of immutable row positions, resolve to observation IDs,
    and reject previously judged unordered pairs. This is a temporary discovery
    strategy until model-informed acquisition is implemented.
    """
    if total_units < 2:
        raise ValueError("At least two observations are needed for a comparison.")
    rng = secrets.SystemRandom()
    for _ in range(8):
        positions = rng.sample(range(total_units), min(total_units, 64))
        units = project.database.unit_rows_at_positions(positions)
        ids = [int(row["observation_id"]) for row in units]
        eligible = project.eligible_scale_observation_ids(scale_id, ids)
        if len(eligible) < 2:
            continue
        previously_judged = project.scale_existing_pairs(scale_id, eligible)
        rng.shuffle(eligible)
        for first in eligible:
            for second in eligible:
                if first == second:
                    continue
                if tuple(sorted((first, second))) in previously_judged:
                    continue
                return _prepare_scale_task(project, {
                    "scale_id": int(scale_id), "session_id": int(session_id),
                    "picked_a": first, "picked_b": second,
                    "status": "selecting",
                }, None)
    raise ValueError(
        "No unjudged eligible pair was found in the bounded random sample. "
        "Try the manual observation selector under More options."
    )


def comparison_layout(*, dcc: Any, html: Any, dash_table: Any, auto_train: bool = False) -> Any:
    """A single pair-reading task, with optional manual controls and history."""
    return html.Div([
        html.Div([
            html.H4("Scale model"),
            html.Label("Analytic geometry"),
            dcc.Dropdown(
                id="scale-model-geometry", options=[], value=None,
                clearable=False, placeholder="Choose an analytic geometry",
            ),
            html.Div([
                html.Button(
                    "Train", id="scale-model-train", n_clicks=0,
                    className="primary-action",
                ),
                dcc.Checklist(
                    id="scale-auto-train",
                    options=[{
                        "label": "Automatically train after saved comparisons",
                        "value": "auto",
                    }],
                    value=["auto"] if auto_train else [],
                ),
            ], className="classifier-training-options"),
            html.Div(id="scale-model-preference-status", style={"display": "none"}),
            html.Div(id="scale-model-summary", className="focus-model-detail"),
            html.Div(id="scale-model-train-status", className="control-status"),
            html.Div(id="scale-model-auto-status", className="control-status"),
        ], className="focus-coding-panel"),
        html.H3("Compare two observations"),
        html.P(
            "Read both observations and choose which expresses more of the selected "
            "Scale, or whether they are approximately equal. Pairs are sampled "
            "randomly for now; informative pair recommendations will come later.",
            className="focus-help",
        ),
        html.Div(id="scale-task-warning", className="control-status"),
        html.Div([
            html.Div(id="scale-reading-first", className="reading-pane"),
            html.Div(id="scale-reading-second", className="reading-pane"),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(290px, 1fr))",
            "gap": "16px", "alignItems": "start",
        }),
        html.Div([
            html.Button(
                "A is higher", id="scale-judge-first", n_clicks=0,
                className="judgment-button positive-button", disabled=True,
            ),
            html.Button(
                "Approximately equal", id="scale-judge-parity", n_clicks=0,
                className="judgment-button", disabled=True,
            ),
            html.Button(
                "B is higher", id="scale-judge-second", n_clicks=0,
                className="judgment-button negative-button", disabled=True,
            ),
            html.Button(
                "Unsure / Skip", id="scale-judge-unsure", n_clicks=0,
                className="judgment-button unsure-button", disabled=True,
            ),
        ], className="code-center-assignment-buttons"),
        html.Div(id="scale-action-status", className="control-status"),
        html.Button(
            "Next pair", id="scale-next-pair", n_clicks=0,
            className="primary-action",
        ),
        html.Details([
            html.Summary("More options · choose a pair manually"),
            html.P(
                "Optional: choose two specific observations instead of a randomly "
                "presented pair. Selection uses a small page, never the entire corpus.",
                className="focus-help",
            ),
            html.Label("Observation browser page"),
            dcc.Input(
                id="scale-browse-page", type="number", min=1, step=1,
                value=1, className="page-number-input",
            ),
            html.Div(id="scale-browse-status", className="control-status"),
            dash_table.DataTable(
                id="scale-browser",
                columns=[
                    {"name": "Unit", "id": "unit_id"},
                    {"name": "Source key", "id": "key"},
                    {"name": "Preview", "id": "preview"},
                    {"name": "Eligibility", "id": "eligibility"},
                ],
                data=[], row_selectable="single", selected_rows=[],
                page_size=_PAGE_SIZE,
                style_table={"overflowX": "auto"},
                style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
            ),
            html.Div([
                html.Button("Choose as first", id="scale-select-first", n_clicks=0),
                html.Button("Choose as second", id="scale-select-second", n_clicks=0),
                html.Button(
                    "Use Explore focal as first", id="scale-use-explore-focal",
                    n_clicks=0,
                ),
            ], className="memo-actions"),
            html.Div(id="scale-selection-summary", className="control-status"),
            html.Button(
                "Show selected pair", id="scale-prepare", n_clicks=0,
                className="primary-action",
            ),
            html.Button("Clear selection", id="scale-clear", n_clicks=0),
            dcc.Input(
                id="scale-reviewer-key", type="text", value="",
                style={"display": "none"},
            ),
            html.H4("Repeat an existing comparison"),
            html.Div(id="scale-pair-history-note", className="control-status"),
            dcc.Checklist(
                id="scale-repeat-ack",
                options=[{
                    "label": "Make an independent second judgment of this pair",
                    "value": "repeat",
                }],
                value=[],
            ),
            html.Button(
                "Compare this pair again", id="scale-new-repeat", n_clicks=0,
            ),
        ], className="focus-coding-panel"),
        html.Details([
            html.Summary("Comparison history and corrections"),
            html.P(
                "Past judgments remain available, including earlier Scale "
                "definitions, corrected events, and imported evidence.",
                className="focus-help",
            ),
            html.Label("History page (latest first)"),
            dcc.Input(
                id="scale-history-page", type="number", min=1, step=1, value=1,
                className="page-number-input",
            ),
            dash_table.DataTable(
                id="scale-history", data=[], row_selectable="single", selected_rows=[],
                columns=[
                    {"name": "Event", "id": "comparison_event_id"},
                    {"name": "Definition", "id": "version"},
                    {"name": "Observations (A / B)", "id": "pair"},
                    {"name": "Judgment", "id": "judgment"},
                    {"name": "Provenance", "id": "origin"},
                    {"name": "Status", "id": "state"},
                    {"name": "Corrects", "id": "corrects"},
                ],
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={"whiteSpace": "normal", "height": "auto", "textAlign": "left"},
            ),
            html.Div(id="scale-history-status", className="control-status"),
            html.Button(
                "Correct selected judgment", id="scale-correct", n_clicks=0,
            ),
            html.P(
                "Corrections preserve the original definition and its recorded "
                "history.",
                className="focus-help",
            ),
        ], className="focus-coding-panel"),
    ], className="focus-coding-panel")

def register_comparison_callbacks(
    app: Any, project: Any, *, total_units: int, hierarchy: list[dict[str, Any]],
    html: Any, dcc: Any,
) -> None:
    from dash import Input, Output, State, callback_context, no_update
    from geometric_coder.ui.app import _metadata_table, _render_context_units
    del hierarchy

    def active_scale_id(construct_id: int | None) -> int | None:
        if construct_id is None:
            return None
        try:
            row = project.construct(int(construct_id))
        except (KeyError, TypeError, ValueError):
            return None
        return int(construct_id) if row["type"] == "scale" else None

    @app.callback(
        Output("scale-model-preference-status", "children"),
        Input("scale-auto-train", "value"),
        State("active-session-store", "data"),
        prevent_initial_call=True,
    )
    def persist_scale_auto_training(
        values: list[str] | None, session_id: int | None,
    ) -> str:
        if session_id is not None:
            project.patch_session_state(
                int(session_id), {"scale_auto_train": "auto" in (values or [])}
            )
        return ""

    @app.callback(
        Output("scale-model-geometry", "options"),
        Output("scale-model-geometry", "value"),
        Input("focus-construct-dropdown", "value"),
    )
    def scale_model_geometry_options(
        construct_id: int | None,
    ) -> tuple[list[dict[str, Any]], int | None]:
        scale_id = active_scale_id(construct_id)
        if scale_id is None:
            return [], None
        reps = [
            item for item in project.representations()
            if item["analytic_enabled"] and item["legacy_kind"] == "geometry"
        ]
        options = [
            {"label": str(item["name"]), "value": int(item["representation_id"])}
            for item in reps
        ]
        spec = project.database.get_scaler_spec(scale_id)
        active = spec["representation_id"] if spec is not None else None
        return options, active if any(option["value"] == active for option in options) else None

    @app.callback(
        Output("scale-model-train-status", "children"),
        Input("scale-model-train", "n_clicks"),
        State("focus-construct-dropdown", "value"),
        State("scale-model-geometry", "value"),
        prevent_initial_call=True,
    )
    def train_selected_scale(
        clicks: int | None, construct_id: int | None,
        representation_id: int | None,
    ) -> str:
        if not clicks:
            return ""
        scale_id = active_scale_id(construct_id)
        if scale_id is None:
            return "Select a Scale before training."
        if representation_id is None:
            return "Choose an analytic geometry before training."
        try:
            project.configure_scale_model(scale_id, int(representation_id))
            result = project.train_scale(scale_id)
            if result["status"] == "insufficient_evidence":
                return str(result["reason"])
            return (
                "Current Scale model reused."
                if result.get("reused")
                else f"Scale fit {result['fit']['scaler_fit_id']} saved."
            )
        except (ValueError, KeyError, RuntimeError) as error:
            return f"Scale training unavailable: {error}"

    @app.callback(
        Output("scale-model-auto-status", "children"),
        Input("scale-action-status", "children"),
        State("scale-task-store", "data"),
        State("focus-construct-dropdown", "value"),
        State("scale-auto-train", "value"),
        State("scale-model-geometry", "value"),
        prevent_initial_call=True,
    )
    def train_after_saved_scale_comparison(
        message: str | None, task: dict[str, Any] | None,
        construct_id: int | None, auto_values: list[str] | None,
        representation_id: int | None,
    ) -> str:
        # A saved-evidence signal only; browsing or presenting a pair never fits.
        if not isinstance(message, str) or not message.startswith("Saved event "):
            return ""
        if not task or task.get("status") != "saved":
            return ""
        event_id = task.get("event_id")
        if event_id is None or not message.startswith(f"Saved event {event_id}:"):
            return ""
        if "auto" not in (auto_values or []):
            return "Comparison saved; Scale training is manual."
        scale_id = active_scale_id(construct_id)
        if scale_id is None or int(task.get("scale_id", -1)) != scale_id:
            return ""
        if representation_id is None:
            return "Comparison saved; select an analytic geometry to train."
        try:
            project.configure_scale_model(scale_id, int(representation_id))
            result = project.train_scale(scale_id)
            if result["status"] == "insufficient_evidence":
                return f"Comparison saved; {result['reason']}"
            return (
                "Comparison saved; Scale fit already current."
                if result.get("reused")
                else f"Comparison saved; Scale fit {result['fit']['scaler_fit_id']} updated."
            )
        except (KeyError, ValueError, RuntimeError) as error:
            return f"Comparison saved; automatic Scale training unavailable: {error}"

    @app.callback(
        Output("scale-model-summary", "children"),
        Input("focus-construct-dropdown", "value"),
        Input("scale-model-train-status", "children"),
        Input("scale-model-auto-status", "children"),
        Input("scale-task-store", "data"),
    )
    def refresh_scale_model_summary(
        construct_id: int | None, _manual: str | None,
        _automatic: str | None, _task: dict[str, Any] | None,
    ) -> str:
        scale_id = active_scale_id(construct_id)
        if scale_id is None:
            return ""
        status = project.scale_model_status(scale_id)
        descriptor = status["status"].replace("_", " ").capitalize()
        count = status.get("comparison_count")
        fit = status.get("fit")
        details = f"{count} usable comparisons" if count is not None else "No fit"
        if fit is not None:
            diagnostics = fit.get("diagnostics", {})
            details += (
                f"; saved fit {fit['scaler_fit_id']} "
                f"({diagnostics.get('geometry_dimensions', '?')} features)"
            )
        return f"{descriptor} — {details}. Scalar scores are exploratory, not validated measurements."

    def key_label(observation_id: int) -> str:
        unit = project.observation(int(observation_id))
        return " › ".join(f"{k}: {v}" for k, v in unit["user_key"].items()) or (
            f"Unit {unit['unit_id']}"
        )

    @app.callback(
        Output("scale-browser", "data"),
        Output("scale-browser", "selected_rows"),
        Output("scale-browse-status", "children"),
        Input("focus-construct-dropdown", "value"),
        Input("scale-browse-page", "value"),
        Input("workspace-tabs", "value"),
    )
    def browse_scale_units(
        construct_id: int | None, page_number: int | None, workspace: str | None,
    ) -> tuple[Any, list[int], str]:
        scale_id = active_scale_id(construct_id)
        if scale_id is None or workspace != "focus":
            return [], [], ""
        page_count = max(1, (total_units + _PAGE_SIZE - 1) // _PAGE_SIZE)
        try:
            page = max(1, min(int(page_number or 1), page_count))
        except (ValueError, TypeError):
            page = 1
        start = (page - 1) * _PAGE_SIZE
        units = project.database.unit_rows_at_positions(
            list(range(start, min(start + _PAGE_SIZE, total_units)))
        )
        allowed = set(project.eligible_scale_observation_ids(
            scale_id, [int(row["observation_id"]) for row in units],
        ))
        rows = [{
            "unit_id": int(unit["unit_id"]),
            "observation_id": int(unit["observation_id"]),
            "key": " › ".join(f"{k}: {v}" for k, v in unit["user_key"].items()),
            "preview": " ".join(str(unit["text"]).split())[:180],
            "eligibility": (
                "Eligible" if unit["observation_id"] in allowed else "Excluded"
            ),
        } for unit in units]
        return rows, [], (
            f"Page {page:,} of {page_count:,}. {len(allowed):,} eligible of "
            f"{len(rows):,} displayed. Browse another page to find more."
        )

    @app.callback(
        Output("scale-task-store", "data"),
        Output("scale-action-status", "children"),
        Output("scale-repeat-ack", "value"),
        Input("focus-construct-dropdown", "value"),
        Input("active-session-store", "data"),
        Input("scale-select-first", "n_clicks"),
        Input("scale-select-second", "n_clicks"),
        Input("scale-use-explore-focal", "n_clicks"),
        Input("scale-prepare", "n_clicks"),
        Input("scale-new-repeat", "n_clicks"),
        Input("scale-clear", "n_clicks"),
        Input("scale-correct", "n_clicks"),
        Input("scale-judge-first", "n_clicks"),
        Input("scale-judge-parity", "n_clicks"),
        Input("scale-judge-second", "n_clicks"),
        Input("scale-judge-unsure", "n_clicks"),
        State("scale-browser", "data"),
        State("scale-browser", "selected_rows"),
        State("scale-task-store", "data"),
        State("scale-reviewer-key", "value"),
        State("scale-repeat-ack", "value"),
        State("scale-history", "data"),
        State("scale-history", "selected_rows"),
        State("focal-unit-store", "data"),
    )
    def manage_scale_comparison(
        construct_id: int | None, session_id: int | None,
        select_first: int, select_second: int, use_focal: int,
        prepare: int, repeat: int, clear: int, correct: int,
        judge_first: int, judge_parity: int, judge_second: int, judge_unsure: int,
        browser_rows: list[dict[str, Any]] | None, browser_selection: list[int] | None,
        task_state: dict[str, Any] | None, reviewer: str | None,
        repeat_ack: list[str] | None, history_rows: list[dict[str, Any]] | None,
        history_selection: list[int] | None, explore_focal: int | None,
    ) -> tuple[Any, Any, Any]:
        del (select_first, select_second, use_focal, prepare, repeat, clear,
             correct, judge_first, judge_parity, judge_second, judge_unsure)
        scale_id = active_scale_id(construct_id)
        if scale_id is None or session_id is None:
            return None, "", []
        session_id = int(session_id)
        trigger = callback_context.triggered_id
        if trigger in {None, "focus-construct-dropdown", "active-session-store"}:
            stored = project.database.get_session(session_id)["state"].get(
                "scale_comparison_task"
            )
            current = _restore_scale_task(project, scale_id, session_id, stored)
            if current["status"] == "selecting" and current.get("picked_a") is None and current.get("picked_b") is None:
                try:
                    current = _random_scale_task(project, scale_id, session_id, total_units)
                except ValueError as error:
                    return current, str(error), []
                project.patch_session_state(session_id, {"scale_comparison_task": current})
            return current, "", []
        task = _restore_scale_task(project, scale_id, session_id, task_state)
        if trigger == "scale-clear":
            task = _restore_scale_task(project, scale_id, session_id, None)
            project.patch_session_state(session_id, {"scale_comparison_task": task})
            return task, "Selection cleared.", []
        try:
            if trigger in {
                "scale-select-first", "scale-select-second", "scale-use-explore-focal",
            }:
                if trigger == "scale-use-explore-focal":
                    if explore_focal is None:
                        raise ValueError("Choose a focal observation in Explore first.")
                    observation_id = project.observation_id_for_unit(int(explore_focal))
                else:
                    if not browser_selection or not browser_rows:
                        raise ValueError("Select an observation in the browser first.")
                    index = int(browser_selection[0])
                    if not 0 <= index < len(browser_rows):
                        raise ValueError("Select an observation from the current page.")
                    observation_id = int(browser_rows[index]["observation_id"])
                if observation_id not in project.eligible_scale_observation_ids(
                    scale_id, [observation_id],
                ):
                    raise ValueError("This observation is not eligible for the selected Scale.")
                task = {
                    "scale_id": scale_id, "session_id": session_id,
                    "picked_a": (
                        observation_id if trigger != "scale-select-second"
                        else task.get("picked_a")
                    ),
                    "picked_b": (
                        observation_id if trigger == "scale-select-second"
                        else task.get("picked_b")
                    ),
                    "status": "selecting",
                }
                project.patch_session_state(session_id, {"scale_comparison_task": task})
                return task, "Observation selected. Prepare when both are chosen.", []
            if trigger in {"scale-prepare", "scale-new-repeat"}:
                if trigger == "scale-new-repeat":
                    if not task.get("first") or not task.get("second"):
                        raise ValueError("Prepare a pair before requesting another judgment.")
                    task["picked_a"] = task["first"]
                    task["picked_b"] = task["second"]
                task = _prepare_scale_task(project, task, reviewer)
                project.patch_session_state(session_id, {"scale_comparison_task": task})
                return task, (
                    "New comparison ready. A/B order is fixed for this task."
                ), []
            if trigger == "scale-correct":
                if not history_selection or not history_rows:
                    raise ValueError("Select a history row to correct.")
                index = int(history_selection[0])
                if not 0 <= index < len(history_rows):
                    raise ValueError("Select a valid history row.")
                event_id = int(history_rows[index]["comparison_event_id"])
                previous = project.scale_comparison_event(scale_id, event_id)
                if previous is None:
                    raise ValueError("The selected comparison is no longer available.")
                if previous["is_superseded"]:
                    raise ValueError("This judgment was already corrected. Select its latest revision.")
                if previous["version_number"] != project.scale(scale_id)["version_number"]:
                    raise ValueError("An earlier Scale definition cannot be corrected under the current definition.")
                if not str(previous["origin"]).startswith("human_"):
                    raise ValueError("A nonhuman event cannot be edited as a human judgment.")
                # Corrections carry forward the event's recorded identity.
                # Single-researcher mode does not ask for a reviewer key.
                first, second = (
                    int(previous["presented_first_id"]),
                    int(previous["presented_second_id"]),
                )
                if set(project.eligible_scale_observation_ids(scale_id, [first, second])) != {first, second}:
                    raise ValueError("This pair is no longer eligible; its history is still available.")
                task = {
                    "scale_id": scale_id, "session_id": session_id,
                    "picked_a": first, "picked_b": second,
                    "first": first, "second": second,
                    "snapshot": _scale_task_snapshot(project, scale_id),
                    "submission_key": uuid.uuid4().hex,
                    "reviewer_key": previous["reviewer_key"],
                    "mode": "correction", "supersedes_event_id": event_id,
                    "status": "ready",
                }
                project.patch_session_state(session_id, {"scale_comparison_task": task})
                return task, (
                    f"Correcting event {event_id}; original presentation order is retained."
                ), []
            if trigger in _OUTCOMES:
                if task.get("status") == "saved":
                    return task, f"Already saved as event {task.get('event_id')}.", no_update
                if task.get("status") != "ready":
                    raise ValueError("Prepare a comparison before recording a judgment.")
                if task.get("snapshot") != _scale_task_snapshot(project, scale_id):
                    raise ValueError(
                        "The Scale definition or eligibility changed. Prepare a new task before saving."
                    )
                first, second = int(task["first"]), int(task["second"])
                if set(project.eligible_scale_observation_ids(scale_id, [first, second])) != {first, second}:
                    raise ValueError("The pair is no longer eligible. Prepare a new task.")
                previously_judged = project.scale_pair_history_count(
                    scale_id, first, second,
                ) > 0
                if (task.get("mode") != "correction" and previously_judged
                        and "repeat" not in (repeat_ack or [])):
                    raise ValueError(
                        "This pair already has evidence. Confirm an intentional independent repeat."
                    )
                outcome = _OUTCOMES[trigger]
                provenance = {
                    "strategy": "manual", "workspace": "develop",
                    "mode": task.get("mode", "independent"),
                    "task_nonce": task["submission_key"],
                    "definition_version": task["snapshot"]["version_number"],
                    "eligibility_revision": task["snapshot"]["eligibility"]["eligibility_revision"],
                }
                event_id = project.record_scale_comparison(
                    scale_id, first, second, outcome,
                    origin="human_direct",
                    reviewer_key=task.get("reviewer_key"),
                    session_id=session_id,
                    selection_provenance=provenance,
                    supersedes_event_id=task.get("supersedes_event_id"),
                    submission_key=task["submission_key"],
                )
                task = {**task, "status": "saved", "event_id": event_id, "outcome": outcome}
                project.patch_session_state(session_id, {"scale_comparison_task": task})
                return task, f"Saved event {event_id}: {_OUTCOME_LABELS[outcome]}.", no_update
        except (KeyError, ValueError, TypeError) as error:
            return no_update, str(error), no_update
        return no_update, no_update, no_update

    @app.callback(
        Output("scale-task-store", "data", allow_duplicate=True),
        Output("scale-action-status", "children", allow_duplicate=True),
        Output("scale-repeat-ack", "value", allow_duplicate=True),
        Input("scale-next-pair", "n_clicks"),
        State("focus-construct-dropdown", "value"),
        State("active-session-store", "data"),
        prevent_initial_call=True,
    )
    def next_scale_pair(
        clicks: int | None, construct_id: int | None, session_id: int | None,
    ) -> tuple[Any, str, Any]:
        if not clicks:
            return no_update, "", no_update
        scale_id = active_scale_id(construct_id)
        if scale_id is None or session_id is None:
            return no_update, "", no_update
        try:
            task = _random_scale_task(project, scale_id, int(session_id), total_units)
        except ValueError as error:
            return no_update, str(error), no_update
        project.patch_session_state(
            int(session_id), {"scale_comparison_task": task},
        )
        return task, "New pair ready.", []

    @app.callback(
        Output("scale-selection-summary", "children"),
        Output("scale-reading-first", "children"),
        Output("scale-reading-second", "children"),
        Output("scale-task-warning", "children"),
        Output("scale-pair-history-note", "children"),
        Output("scale-judge-first", "disabled"),
        Output("scale-judge-parity", "disabled"),
        Output("scale-judge-second", "disabled"),
        Output("scale-judge-unsure", "disabled"),
        Input("scale-task-store", "data"),
        Input("scale-repeat-ack", "value"),
        Input("focus-construct-dropdown", "value"),
        Input("focus-refresh-store", "data"),
    )
    def display_scale_task(
        task: dict[str, Any] | None, repeat_ack: list[str] | None,
        construct_id: int | None, refresh: int | None,
    ) -> tuple[Any, ...]:
        del refresh
        scale_id = active_scale_id(construct_id)
        if scale_id is None or not isinstance(task, dict) or task.get("scale_id") != scale_id:
            return "", "", "", "", "", True, True, True, True
        summary = []
        for field, name in (("picked_a", "First selection"), ("picked_b", "Second selection")):
            value = task.get(field)
            summary.append(
                f"{name}: {key_label(value) if value is not None else 'Choose an observation'}"
            )
        if task.get("status") not in {"ready", "saved"}:
            return " · ".join(summary), "", "", "", "", True, True, True, True
        first, second = int(task["first"]), int(task["second"])
        snapshot_ok = task.get("snapshot") == _scale_task_snapshot(project, scale_id)
        eligible = set(project.eligible_scale_observation_ids(scale_id, [first, second])) == {first, second}
        warning = "" if (snapshot_ok and eligible) else (
            "This comparison was prepared under an older definition or eligibility "
            "state. Prepare a new task; the existing task cannot be saved."
        )
        def reading(observation_id: int, letter: str) -> Any:
            item = project.observation(observation_id)
            unit_id = int(item["unit_id"])
            # A bounded local context avoids loading an entire long document.
            # Adjacent units must belong to the same parent source hierarchy.
            position = int(item["row_position"])
            positions = range(max(0, position - 3), min(total_units, position + 4))
            nearby = project.database.unit_rows_at_positions(list(positions))
            parent_keys = list(project.metadata["key_columns"])[:-1]
            context = [
                unit for unit in nearby
                if (int(unit["unit_id"]) == unit_id
                    or (parent_keys and all(
                        unit["user_key"].get(key) == item["user_key"].get(key)
                        for key in parent_keys
                    )))
            ]
            children = _render_context_units(
                html, dcc, context, focal_unit_id=unit_id,
                key_columns=list(project.metadata["key_columns"]),
                selected_unit_ids=[unit_id], checkbox_type=f"scale-context-{letter}",
                span_selection_enabled=False,
            )
            return html.Div([
                html.H4(f"Observation {letter}"),
                html.P(key_label(observation_id), className="focal-key"),
                html.P(item["text"], className="focal-text"),
                html.Details([
                    html.Summary("Metadata"),
                    _metadata_table(html, item["metadata"]),
                ]),
                html.Details([
                    html.Summary("Surrounding context"),
                    html.Div(children, className="context-content"),
                ]),
            ], className="focus-coding-panel")
        match_count = project.scale_pair_history_count(scale_id, first, second)
        correction = task.get("mode") == "correction"
        prior_message = (
            f"Correcting event {task['supersedes_event_id']} (append-only)."
            if correction else (
                f"This pair has {match_count} previous event(s), including earlier definitions "
                "and corrections. Confirm an independent repeat to record another judgment."
                if match_count else "No previous judgment for this pair."
            )
        )
        enabled = (
            task["status"] == "ready" and snapshot_ok and eligible
            and (correction or not match_count or "repeat" in (repeat_ack or []))
        )
        if task["status"] == "saved":
            warning = f"Saved event {task['event_id']}. Prepare a new task to judge again."
        return (
            " · ".join(summary),
            reading(first, "A"), reading(second, "B"),
            warning, prior_message, *([not enabled] * 4),
        )

    @app.callback(
        Output("scale-history", "data"),
        Output("scale-history", "selected_rows"),
        Output("scale-history-status", "children"),
        Input("focus-construct-dropdown", "value"),
        Input("scale-task-store", "data"),
        Input("workspace-tabs", "value"),
        Input("scale-history-page", "value"),
    )
    def refresh_scale_history(
        construct_id: int | None, task: Any, workspace: str | None,
        page: int | None = 1,
    ) -> tuple[Any, list[int], str]:
        del task
        scale_id = active_scale_id(construct_id)
        if scale_id is None or workspace != "focus":
            return [], [], ""
        history, total, current_page = project.scale_comparison_history_page(
            scale_id, page=max(1, int(page or 1)), page_size=_HISTORY_SIZE,
        )
        current_version = int(project.scale(scale_id)["version_number"])
        labels: dict[int, str] = {}
        def label(obs: int) -> str:
            if obs not in labels:
                labels[obs] = key_label(obs)
            return labels[obs]
        records = []
        for row in history:
            event_id = int(row["comparison_event_id"])
            is_human = str(row["origin"]).startswith("human_")
            records.append({
                "comparison_event_id": event_id,
                "version": row["version_number"],
                "pair": (
                    f"{label(row['presented_first_id'])} / "
                    f"{label(row['presented_second_id'])}"
                ),
                "judgment": _OUTCOME_LABELS[row["outcome"]],
                "origin": (
                    f"Human · {row['origin']}" if is_human
                    else f"Nonhuman · {row['origin']}"
                ),
                "reviewer": row["reviewer_key"] or "Not specified",
                "state": (
                    ("Superseded" if row["is_superseded"] else "Effective")
                    + (" · earlier definition" if int(row["version_number"]) != current_version else "")
                ),
                "corrects": row["supersedes_event_id"] or "",
            })
        return records, [], (
            f"{total:,} recorded event(s), page {current_page:,} of "
            f"{max(1, (total + _HISTORY_SIZE - 1) // _HISTORY_SIZE):,} "
            f"({len(records):,} visible). Older events can be opened by changing pages."
        )
