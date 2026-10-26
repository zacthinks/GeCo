"""Explicit lifecycle for GeCo's local Dash HTTP server.

The server is independently closeable: closing the browser UI does not terminate
the notebook kernel or the project object that launched it.
"""
from __future__ import annotations

import threading
from typing import Any

from werkzeug.serving import make_server


class GeCoServer:
    """One bound GeCo HTTP listener, controllable from UI or Python."""

    def __init__(self, app: Any, *, host: str, port: int) -> None:
        try:
            self._http = make_server(host, port, app.server, threaded=True)
        except SystemExit as exc:
            # Werkzeug may raise SystemExit on EADDRINUSE. Do not kill a
            # caller's Jupyter kernel because the requested port is occupied.
            raise OSError(
                f"Could not start GeCo at {host}:{port}. "
                "The port may already be in use."
            ) from exc
        self._host = host
        self._thread = threading.Thread(
            target=self._http.serve_forever,
            name=f"geco-server-{self._http.server_port}",
            daemon=True,
        )
        self._close_lock = threading.Lock()
        self._closed = False
        self._closed_event = threading.Event()

    @property
    def port(self) -> int:
        """Actual bound port; also works with explicit port=0."""
        return int(self._http.server_port)

    @property
    def url(self) -> str:
        from geometric_coder.ui.app import _local_browser_url

        return _local_browser_url(self._host, self.port)

    @property
    def closed(self) -> bool:
        return self._closed

    def start(self) -> GeCoServer:
        if not self._closed and not self._thread.is_alive():
            self._thread.start()
        return self

    def close(self) -> None:
        """Stop the server without stopping Python or closing its data project."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._http.shutdown()
            self._http.server_close()
            if self._thread.is_alive() and threading.current_thread() is not self._thread:
                self._thread.join(timeout=5.0)
        finally:
            self._closed_event.set()

    def request_close(self) -> None:
        """Let the browser receive its response before terminating its server."""
        timer = threading.Timer(0.6, self.close)
        timer.name = f"geco-close-{self.port}"
        timer.daemon = True
        timer.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the server closes, or until timeout if provided."""
        return self._closed_event.wait(timeout)

    def __enter__(self) -> GeCoServer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
