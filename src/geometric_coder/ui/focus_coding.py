"""Focused finite-task coding mode for blind audit and review workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from geometric_coder.exceptions import ConfigurationError

if TYPE_CHECKING:
    from geometric_coder.project import GeometricCoder


def create_focus_coding_app(
    project: GeometricCoder,
    *,
    session_id: int | None = None,
) -> Any:
    """Create the dedicated Focus Coding surface over one configured task.

    This is a specialized mode of the same GeCo project/session state rather than
    an independent coding subsystem.  It writes ordinary annotation events and
    relies on ``GeometricCoder.focus_*`` methods for task/progress semantics.
    """
    try:
        from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
    except ImportError as error:
        raise ConfigurationError(
            "The GeCo web interface is part of the standard installation, but a core "
            "UI dependency could not be imported. Reinstall geometric-coder with its "
            "default dependencies."
        ) from error

    task = project.focus_task(session_id)
    session_id = int(task["session_id"])
    required_unit_ids = [int(value) for value in task["required_unit_ids"]]
    first_unit_id = _initial_focus_unit_id(project, task)

    assets_folder = Path(__file__).parent / "assets"
    app = Dash(
        __name__,
        title="GeCo · Focus Coding",
        update_title=None,
        assets_folder=str(assets_folder),
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div(
        [
            dcc.Store(id="focus-coding-session-store", data=session_id),
            dcc.Store(id="focus-coding-current-unit-store", data=first_unit_id),
            dcc.Store(id="focus-coding-refresh-store", data=0),
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
                                        "Geometric Coder — A geometric coding platform for qualitative inquiry",
                                        className="app-subtitle",
                                    ),
                                ]
                            ),
                        ],
                        className="app-brand",
                    ),
                    html.Div(id="focus-coding-progress-summary", className="focus-coding-progress-summary"),
                ],
                className="app-header",
            ),
            html.Div("Focus Coding Session", className="focus-coding-mode-heading"),
            html.Div(
                [
                    html.Progress(
                        id="focus-coding-progress-bar",
                        value=0,
                        max=max(1, len(required_unit_ids)),
                        className="focus-coding-progress-bar",
                    ),
                    html.Div(id="focus-coding-judgment-summary", className="focus-coding-judgment-summary"),
                ],
                className="focus-coding-progress-wrap",
            ),
            html.Div(id="focus-coding-notice", className="focus-coding-notice"),
            html.Div(
                [
                    html.Section(
                        [
                            html.Div(id="focus-coding-position", className="focus-coding-position"),
                            html.Div(id="focus-coding-key", className="focus-coding-key"),
                            html.Div(id="focus-coding-metadata", className="focus-coding-metadata"),
                            html.Article(id="focus-coding-text", className="focus-coding-text"),
                        ],
                        className="focus-coding-document-pane",
                    ),
                    html.Aside(
                        [
                            html.H2("Coding", className="focus-coding-codes-heading"),
                            html.Div(id="focus-coding-code-cards", className="focus-coding-code-cards"),
                        ],
                        className="focus-coding-codes-pane",
                    ),
                ],
                id="focus-coding-workspace",
                className="focus-coding-workspace",
            ),
            html.Footer(
                [
                    html.Div(
                        [
                            html.Button("Previous", id="focus-coding-previous", n_clicks=0),
                            html.Button("Next", id="focus-coding-next", n_clicks=0),
                            html.Button(
                                "Next Incomplete",
                                id="focus-coding-next-incomplete",
                                n_clicks=0,
                                className="secondary-action",
                            ),
                            dcc.Checklist(
                                id="focus-coding-auto-advance",
                                options=[{"label": "Auto-advance", "value": "enabled"}],
                                value=[],
                                persistence=True,
                                persistence_type="session",
                                className="focus-coding-auto-advance",
                            ),
                        ],
                        className="focus-coding-nav-buttons",
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Done",
                                id="focus-coding-done",
                                n_clicks=0,
                                className="primary-action focus-coding-done",
                            ),
                            html.Button(
                                "Reopen for editing",
                                id="focus-coding-reopen",
                                n_clicks=0,
                                className="secondary-action",
                                style={"display": "none"},
                            ),
                        ],
                        className="focus-coding-completion-actions",
                    ),
                ],
                className="focus-coding-footer",
            ),
            html.Div(id="focus-coding-complete-panel", className="focus-coding-complete-panel"),
        ],
        className="focus-coding-shell",
    )

    @app.callback(
        Output("focus-coding-progress-summary", "children"),
        Output("focus-coding-progress-bar", "value"),
        Output("focus-coding-progress-bar", "max"),
        Output("focus-coding-judgment-summary", "children"),
        Output("focus-coding-position", "children"),
        Output("focus-coding-key", "children"),
        Output("focus-coding-metadata", "children"),
        Output("focus-coding-text", "children"),
        Output("focus-coding-code-cards", "children"),
        Output("focus-coding-workspace", "style"),
        Output("focus-coding-previous", "disabled"),
        Output("focus-coding-next", "disabled"),
        Output("focus-coding-next-incomplete", "disabled"),
        Output("focus-coding-done", "style"),
        Output("focus-coding-reopen", "style"),
        Output("focus-coding-complete-panel", "children"),
        Input("focus-coding-current-unit-store", "data"),
        Input("focus-coding-refresh-store", "data"),
        State("focus-coding-session-store", "data"),
    )
    def render_focus_coding(
        current_unit_id: int | None,
        refresh: int | None,
        active_session_id: int,
    ) -> tuple[Any, ...]:
        del refresh
        task_now = project.focus_task(int(active_session_id))
        progress = project.focus_progress(int(active_session_id))
        # A completed Focus task is only closed while its completion invariant
        # still holds. An assignment may be edited later through another GeCo
        # surface or process; reopening Focus must never display a stale
        # completion screen over newly unresolved work.
        if str(task_now.get("status", "open")) == "complete" and not bool(progress["complete"]):
            project.reopen_focus(int(active_session_id))
            task_now = project.focus_task(int(active_session_id))
            progress = project.focus_progress(int(active_session_id))
        required = [int(value) for value in task_now["required_unit_ids"]]
        status = str(task_now.get("status", "open"))
        if current_unit_id is None or int(current_unit_id) not in set(required):
            current_unit_id = _initial_focus_unit_id(project, task_now)
        current_unit_id = int(current_unit_id)
        position = required.index(current_unit_id)

        progress_summary = (
            f"{progress['complete_units']} / {progress['total_units']} documents complete"
        )
        judgment_summary = (
            f"{progress['resolved_judgments']} / {progress['total_judgments']} required judgments resolved"
        )

        if status == "complete":
            complete_panel = html.Div(
                [
                    html.Div("✓", className="focus-coding-complete-check"),
                    html.H2("Focus coding complete"),
                    html.P(
                        f"All {progress['total_judgments']} required judgments are resolved."
                    ),
                    html.P(
                        "This focus session is closed. You can close this view or reopen it explicitly for editing.",
                        className="focus-coding-complete-help",
                    ),
                ]
            )
            return (
                progress_summary,
                progress["complete_units"],
                max(1, progress["total_units"]),
                judgment_summary,
                "",
                "",
                "",
                "",
                [],
                {"display": "none"},
                True,
                True,
                True,
                {"display": "none"},
                {"display": "inline-flex"},
                complete_panel,
            )

        unit = project.unit(current_unit_id)
        frozen_codes = [
            project.code(int(spec["code_id"]), int(spec["version_number"]))
            for spec in task_now["required_codes"]
        ]
        current = {
            int(row["code_id"]): str(row["value"])
            for row in project.current_annotations_for_observation(int(unit["observation_id"]))
        }
        cards = [
            _focus_code_card(
                dcc=dcc,
                html=html,
                code=code,
                value=current.get(int(code["code_id"])),
                allow_unsure=bool(task_now["allow_unsure"]),
            )
            for code in frozen_codes
        ]
        key_children = _key_children(html, dict(unit.get("user_key") or {}))
        metadata_children = _metadata_children(html, dict(unit.get("metadata") or {}))
        unresolved_unit_ids = _unresolved_unit_ids(progress)

        return (
            progress_summary,
            progress["complete_units"],
            max(1, progress["total_units"]),
            judgment_summary,
            f"Document {position + 1} of {len(required)}",
            key_children,
            metadata_children,
            str(unit["text"]),
            cards,
            {},
            position == 0,
            position == len(required) - 1,
            not unresolved_unit_ids,
            {"display": "inline-flex"},
            {"display": "none"},
            [],
        )

    @app.callback(
        Output("focus-coding-current-unit-store", "data", allow_duplicate=True),
        Output("focus-coding-refresh-store", "data", allow_duplicate=True),
        Input({"type": "focus-coding-judgment", "code_id": ALL, "value": ALL}, "n_clicks"),
        State("focus-coding-current-unit-store", "data"),
        State("focus-coding-session-store", "data"),
        State("focus-coding-refresh-store", "data"),
        State("focus-coding-auto-advance", "value"),
        prevent_initial_call=True,
    )
    def record_focus_judgment(
        clicks: list[int | None],
        current_unit_id: int | None,
        active_session_id: int,
        refresh: int | None,
        auto_advance: list[str] | None,
    ) -> tuple[Any, int | Any]:
        del clicks
        triggered = callback_context.triggered_id
        if not isinstance(triggered, dict):
            return no_update, no_update
        code_id = triggered.get("code_id")
        value = triggered.get("value")
        if code_id is None or value not in {"positive", "negative", "unsure"}:
            return no_update, no_update
        prop_value = callback_context.triggered[0].get("value") if callback_context.triggered else None
        if not prop_value or int(prop_value) <= 0:
            return no_update, no_update
        if current_unit_id is None:
            return no_update, no_update

        task_now = project.focus_task(int(active_session_id))
        if str(task_now.get("status", "open")) == "complete":
            return no_update, no_update
        required_code_ids = {int(spec["code_id"]) for spec in task_now["required_codes"]}
        if int(code_id) not in required_code_ids:
            return no_update, no_update
        if value == "unsure" and not bool(task_now["allow_unsure"]):
            return no_update, no_update
        before_progress = project.focus_progress(int(active_session_id))
        unit = project.unit(int(current_unit_id))
        project.annotate(
            int(unit["observation_id"]),
            int(code_id),
            str(value),
            origin="human_focus_coder",
        )
        next_unit: Any = no_update
        if "enabled" in set(auto_advance or []):
            after_progress = project.focus_progress(int(active_session_id))
            target = _auto_advance_target(
                [int(value) for value in task_now["required_unit_ids"]],
                before_progress,
                after_progress,
                int(current_unit_id),
            )
            if target is not None:
                _persist_focus_position(project, int(active_session_id), target)
                next_unit = target
        return next_unit, int(refresh or 0) + 1

    @app.callback(
        Output("focus-coding-current-unit-store", "data", allow_duplicate=True),
        Output("focus-coding-refresh-store", "data", allow_duplicate=True),
        Output("focus-coding-notice", "children"),
        Output("focus-coding-notice", "className"),
        Input("focus-coding-previous", "n_clicks"),
        Input("focus-coding-next", "n_clicks"),
        Input("focus-coding-next-incomplete", "n_clicks"),
        Input("focus-coding-done", "n_clicks"),
        Input("focus-coding-reopen", "n_clicks"),
        State("focus-coding-current-unit-store", "data"),
        State("focus-coding-session-store", "data"),
        State("focus-coding-refresh-store", "data"),
        prevent_initial_call=True,
    )
    def navigate_focus_coding(
        previous_clicks: int,
        next_clicks: int,
        incomplete_clicks: int,
        done_clicks: int,
        reopen_clicks: int,
        current_unit_id: int | None,
        active_session_id: int,
        refresh: int | None,
    ) -> tuple[Any, int, str, str]:
        del previous_clicks, next_clicks, incomplete_clicks, done_clicks, reopen_clicks
        triggered = callback_context.triggered_id
        task_now = project.focus_task(int(active_session_id))
        required = [int(value) for value in task_now["required_unit_ids"]]
        current = int(current_unit_id) if current_unit_id in required else required[0]

        if triggered == "focus-coding-reopen":
            project.reopen_focus(int(active_session_id))
            return current, int(refresh or 0) + 1, "Focus session reopened for editing.", "focus-coding-notice"

        if triggered == "focus-coding-done":
            progress = project.complete_focus(int(active_session_id))
            if progress["complete"]:
                return current, int(refresh or 0) + 1, "Focus coding complete.", "focus-coding-notice focus-coding-notice-success"
            first_unresolved = int(progress["current_unit_id"])
            unsure_text = ""
            if not bool(progress["allow_unsure"]) and int(progress["unsure_judgments"]):
                unsure_text = f" {progress['unsure_judgments']} Unsure judgments must be resolved."
            message = (
                f"{progress['unresolved_count']} judgments still need attention."
                f"{unsure_text} Review mode now moves among unresolved documents."
            )
            return first_unresolved, int(refresh or 0) + 1, message, "focus-coding-notice focus-coding-notice-warning"

        if triggered == "focus-coding-next-incomplete":
            progress = project.focus_progress(int(active_session_id))
            unresolved_ids = _unresolved_unit_ids(progress)
            if not unresolved_ids:
                return current, int(refresh or 0), "No unresolved documents remain.", "focus-coding-notice focus-coding-notice-success"
            next_unit = _next_from_subset(required, unresolved_ids, current)
            _persist_focus_position(project, int(active_session_id), next_unit)
            return next_unit, int(refresh or 0), "", "focus-coding-notice"

        if bool(task_now.get("review_unresolved", False)) and triggered in {
            "focus-coding-previous",
            "focus-coding-next",
        }:
            progress = project.focus_progress(int(active_session_id))
            unresolved_ids = _unresolved_unit_ids(progress)
            if not unresolved_ids:
                return (
                    current,
                    int(refresh or 0),
                    "No unresolved judgments remain. Click Done to complete the focus session.",
                    "focus-coding-notice focus-coding-notice-success",
                )
            if triggered == "focus-coding-previous":
                current = _previous_from_subset(required, unresolved_ids, current)
            else:
                current = _next_from_subset(required, unresolved_ids, current)
            _persist_focus_position(project, int(active_session_id), current)
            return current, int(refresh or 0), "", "focus-coding-notice"

        index = required.index(current)
        if triggered == "focus-coding-previous" and index > 0:
            current = required[index - 1]
        elif triggered == "focus-coding-next" and index + 1 < len(required):
            current = required[index + 1]
        _persist_focus_position(project, int(active_session_id), current)
        return current, int(refresh or 0), "", "focus-coding-notice"

    return app


def _initial_focus_unit_id(project: GeometricCoder, task: dict[str, Any]) -> int:
    required = [int(value) for value in task["required_unit_ids"]]
    requested = task.get("current_unit_id")
    try:
        requested_id = int(requested)
    except (TypeError, ValueError):
        requested_id = -1
    if requested_id in set(required):
        return requested_id
    progress = project.focus_progress(int(task["session_id"]))
    unresolved = _unresolved_unit_ids(progress)
    return unresolved[0] if unresolved else required[0]


def _focus_code_card(*, dcc: Any, html: Any, code: dict[str, Any], value: str | None, allow_unsure: bool) -> Any:
    code_id = int(code["code_id"])
    buttons: list[Any] = [
        _judgment_button(html, code_id, "positive", "Present", value),
        _judgment_button(html, code_id, "negative", "Absent", value),
    ]
    if allow_unsure:
        buttons.append(_judgment_button(html, code_id, "unsure", "Unsure", value))
    description = str(code.get("description", "") or "")
    return html.Div(
        [
            html.Div(
                [
                    html.H3(str(code["name"]), className="focus-coding-code-name"),
                    html.Span(
                        f"v{int(code['version_number'])}",
                        className="focus-coding-code-version",
                    ),
                ],
                className="focus-coding-code-heading",
            ),
            dcc.Markdown(description, className="focus-coding-code-description"),
            html.Div(buttons, className="focus-coding-judgment-buttons"),
        ],
        className="focus-coding-code-card",
    )


def _judgment_button(html: Any, code_id: int, value: str, label: str, current: str | None) -> Any:
    semantic_class = {
        "positive": "positive-button",
        "negative": "negative-button",
        "unsure": "unsure-button",
    }[value]
    classes = ["focus-coding-judgment", "judgment-button", semantic_class]
    if current == value:
        classes.append("judgment-selected")
    return html.Button(
        label,
        id={"type": "focus-coding-judgment", "code_id": int(code_id), "value": value},
        n_clicks=0,
        className=" ".join(classes),
    )


def _key_children(html: Any, user_key: dict[str, Any]) -> Any:
    if not user_key:
        return ""
    return html.Div(
        [
            html.Span(
                [
                    html.Strong(f"{key}: "),
                    html.Span(str(value)),
                ],
                className="focus-coding-key-field",
            )
            for key, value in user_key.items()
        ]
    )


def _metadata_children(html: Any, metadata: dict[str, Any]) -> Any:
    if not metadata:
        return html.Div("No metadata supplied for this focus task.", className="focus-coding-metadata-empty")
    return html.Div(
        [
            html.Div(
                [
                    html.Span(str(key), className="focus-coding-metadata-label"),
                    html.Span(str(value), className="focus-coding-metadata-value"),
                ],
                className="focus-coding-metadata-row",
            )
            for key, value in metadata.items()
        ]
    )


def _unresolved_unit_ids(progress: dict[str, Any]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for row in progress.get("unresolved", []):
        unit_id = int(row["unit_id"])
        if unit_id not in seen:
            seen.add(unit_id)
            ordered.append(unit_id)
    return ordered



def _auto_advance_target(
    required: list[int],
    before_progress: dict[str, Any],
    after_progress: dict[str, Any],
    current: int,
) -> int | None:
    """Return the next incomplete unit after this unit just became complete."""
    before_unresolved = set(_unresolved_unit_ids(before_progress))
    after_unresolved = _unresolved_unit_ids(after_progress)
    if current not in before_unresolved or current in set(after_unresolved):
        return None
    if not after_unresolved:
        return None
    return _next_from_subset(required, after_unresolved, current)

def _previous_from_subset(required: list[int], subset: list[int], current: int) -> int:
    subset_set = set(subset)
    if not subset:
        return current
    current_index = required.index(current)
    for offset in range(1, len(required) + 1):
        candidate = required[(current_index - offset) % len(required)]
        if candidate in subset_set:
            return candidate
    return subset[-1]


def _next_from_subset(required: list[int], subset: list[int], current: int) -> int:
    subset_set = set(subset)
    if not subset:
        return current
    current_index = required.index(current)
    for offset in range(1, len(required) + 1):
        candidate = required[(current_index + offset) % len(required)]
        if candidate in subset_set:
            return candidate
    return subset[0]


def _persist_focus_position(project: GeometricCoder, session_id: int, unit_id: int) -> None:
    task = project.focus_task(session_id)
    task["current_unit_id"] = int(unit_id)
    project.patch_session_state(session_id, {"focus_task": task})
