"""Progress reporting helpers for long-running GeCo startup operations."""

from __future__ import annotations

import html
import json
import threading
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One human-readable progress update."""

    phase: str
    message: str
    current: int | None = None
    total: int | None = None
    detail: str | None = None


ProgressCallback = Callable[[ProgressEvent], None]


def emit_progress(
    callback: ProgressCallback | None,
    *,
    phase: str,
    message: str,
    current: int | None = None,
    total: int | None = None,
    detail: str | None = None,
) -> None:
    """Emit an event when a callback was supplied."""
    if callback is None:
        return
    callback(
        ProgressEvent(
            phase=phase,
            message=message,
            current=current,
            total=total,
            detail=detail,
        )
    )


class ConsoleProgress:
    """Print flushed progress updates suitable for scripts and terminals."""

    def __init__(self, *, prefix: str = "GeCo") -> None:
        self.prefix = prefix

    def __call__(self, event: ProgressEvent) -> None:
        counter = ""
        if event.current is not None and event.total is not None:
            counter = f" [{event.current}/{event.total}]"
        detail = f" — {event.detail}" if event.detail else ""
        print(
            f"[{self.prefix}] {event.message}{counter}{detail}",
            flush=True,
        )


class ProgressGroup:
    """Send each progress event to several callbacks."""

    def __init__(self, *callbacks: ProgressCallback | None) -> None:
        self.callbacks = tuple(callback for callback in callbacks if callback is not None)

    def __call__(self, event: ProgressEvent) -> None:
        for callback in self.callbacks:
            callback(event)


class BrowserProgressPage:
    """Serve a temporary local splash page that follows startup progress.

    The page uses only the Python standard library. Once marked ready, it polls
    the target GeCo URL and redirects as soon as the Dash server responds.
    """

    def __init__(
        self,
        *,
        target_url: str,
        title: str = "Geometric Coder is starting",
        host: str = "127.0.0.1",
    ) -> None:
        self.target_url = target_url
        self.title = title
        self.host = host
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "phase": "startup",
            "message": "Preparing GeCo…",
            "detail": "",
            "current": None,
            "total": None,
            "ready": False,
            "failed": False,
            "target_url": target_url,
        }
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("Browser progress page has not been started.")
        return f"http://{self.host}:{self._server.server_port}"

    def start(self, *, open_browser: bool = True) -> str:
        """Start the temporary status server and optionally open a browser."""
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/status":
                    with owner._lock:
                        payload = json.dumps(owner._state).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                body = owner._render_page().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self._server = ThreadingHTTPServer((self.host, 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="geco-startup-page",
            daemon=True,
        )
        self._thread.start()
        if open_browser:
            webbrowser.open(self.url, new=2)
        return self.url

    def __call__(self, event: ProgressEvent) -> None:
        with self._lock:
            self._state.update(
                {
                    "phase": event.phase,
                    "message": event.message,
                    "detail": event.detail or "",
                    "current": event.current,
                    "total": event.total,
                }
            )

    def ready(self, *, message: str = "Project ready. Starting GeCo…") -> None:
        """Tell the splash page to wait for and then open the Dash application."""
        with self._lock:
            self._state.update(
                {
                    "message": message,
                    "detail": self.target_url,
                    "current": 1,
                    "total": 1,
                    "ready": True,
                    "failed": False,
                }
            )

    def fail(self, error: BaseException | str) -> None:
        """Leave a readable failure message on the splash page."""
        with self._lock:
            self._state.update(
                {
                    "message": "GeCo startup failed",
                    "detail": str(error),
                    "ready": False,
                    "failed": True,
                }
            )

    def close(self) -> None:
        """Stop the temporary server."""
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        self._server = None

    def _render_page(self) -> str:
        title = html.escape(self.title)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; min-height: 100vh; display: grid; place-items: center;
            background: #f4f6f3; color: #172019; }}
    main {{ width: min(680px, calc(100vw - 48px)); background: white; border-radius: 20px;
            padding: 42px; box-shadow: 0 18px 60px rgba(25, 50, 31, .14); }}
    .gecko {{ font-size: 42px; margin-bottom: 8px; }}
    h1 {{ margin: 0 0 10px; font-size: 30px; }}
    #message {{ font-size: 18px; font-weight: 650; margin-top: 28px; }}
    #detail {{ min-height: 24px; color: #59645c; margin-top: 8px; overflow-wrap: anywhere; }}
    .track {{ height: 12px; background: #e4e9e4; border-radius: 999px; overflow: hidden;
              margin-top: 24px; }}
    #bar {{ height: 100%; width: 8%; background: #338451; border-radius: inherit;
            transition: width .25s ease; }}
    #counter {{ color: #59645c; margin-top: 9px; font-variant-numeric: tabular-nums; }}
    .failed #bar {{ background: #b42318; }}
    .failed #message {{ color: #b42318; }}
    footer {{ margin-top: 30px; color: #748078; font-size: 14px; }}
  </style>
</head>
<body>
  <main id="card">
    <div class="gecko">🦎</div>
    <h1>{title}</h1>
    <div id="message">Preparing GeCo…</div>
    <div id="detail"></div>
    <div class="track"><div id="bar"></div></div>
    <div id="counter"></div>
    <footer>This page will open the application automatically when it is ready.</footer>
  </main>
<script>
  let waitingForApp = false;

  async function openWhenAvailable(target) {{
    if (waitingForApp) return;
    waitingForApp = true;
    while (true) {{
      try {{
        await fetch(target, {{mode: "no-cors", cache: "no-store"}});
        window.location.replace(target);
        return;
      }} catch (error) {{
        await new Promise(resolve => setTimeout(resolve, 500));
      }}
    }}
  }}

  async function update() {{
    try {{
      const response = await fetch("/status", {{cache: "no-store"}});
      const state = await response.json();
      document.getElementById("message").textContent = state.message || "Working…";
      document.getElementById("detail").textContent = state.detail || "";
      const card = document.getElementById("card");
      card.classList.toggle("failed", Boolean(state.failed));

      let percent = 8;
      let counter = "";
      if (state.current !== null && state.total !== null && state.total > 0) {{
        percent = Math.max(4, Math.min(100, 100 * state.current / state.total));
        counter = `${{state.current}} of ${{state.total}}`;
      }}
      document.getElementById("bar").style.width = `${{percent}}%`;
      document.getElementById("counter").textContent = counter;

      if (state.ready && state.target_url) {{
        document.getElementById("bar").style.width = "100%";
        openWhenAvailable(state.target_url);
      }}
    }} catch (error) {{
      document.getElementById("detail").textContent = "Waiting for startup status…";
    }}
  }}

  update();
  setInterval(update, 400);
</script>
</body>
</html>"""


__all__ = [
    "BrowserProgressPage",
    "ConsoleProgress",
    "ProgressCallback",
    "ProgressEvent",
    "ProgressGroup",
    "emit_progress",
]
