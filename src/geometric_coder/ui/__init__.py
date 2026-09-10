"""GeCo local web interfaces."""

from __future__ import annotations

from typing import Any

from geometric_coder.ui.app import create_app, launch_app


def launch_focus_coder(
    project: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    use_reloader: bool = False,
    session_id: int | None = None,
) -> None:
    """Launch the dedicated Focus Coding mode over an existing GeCo session."""
    launch_app(
        project,
        host=host,
        port=port,
        debug=debug,
        use_reloader=use_reloader,
        mode="focus",
        focus_session_id=session_id,
    )


__all__ = ["create_app", "launch_app", "launch_focus_coder"]
