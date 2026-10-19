"""One-slot reverse proxy for disposable GeCo Dash sessions in a Docker Space.

Each visitor is served by a private Dash subprocess bound to localhost. Only
the holder of a random HttpOnly cookie can reach that subprocess. No server-side
project state survives End Session, timeout, or container restart.
"""

from __future__ import annotations

import atexit
import http.client
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, make_response, redirect, render_template_string, request

COOKIE_NAME = "geco_demo_lease"
HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
LANDING = """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Try GeCo</title>
<style>body{font:16px/1.6 system-ui;background:#f4f6f3;color:#17251c;
margin:0;display:grid;min-height:100vh;place-items:center}
main{background:white;max-width:590px;margin:24px;padding:32px;
border:1px solid #d7e2da;border-radius:14px}
h1{margin:0 0 10px}button{background:#2f744a;color:white;
border:0;border-radius:6px;padding:12px 20px;font-size:16px;cursor:pointer}
p{color:#45554a}</style></head><body><main>
<h1>Explore GeCo</h1>
{% if busy %}
<p>The online sandbox is in use. It supports one visitor at a time.
Please come back after the current session ends.</p>
{% else %}
<p>Try the real GeCo interface with a preloaded corpus, geometries,
codes, and classifiers. You can make judgments, write memos, and retrain.</p>
<p><strong>All changes are temporary.</strong> Use <em>End Session</em>
when finished. Inactive sessions expire after {{ minutes }} minutes.</p>
<form action="/__demo/start" method="post"><button type="submit">
Start demo session</button></form>
{% endif %}
</main></body></html>
"""


@dataclass
class Lease:
    token: str
    process: subprocess.Popen[bytes]
    directory: Path
    last_access: float
    in_flight: int = 0


class SlotManager:
    """Exactly one independent project at a time. No shared GeCo instance."""

    def __init__(
        self, template: Path, *, port: int = 7861,
        idle_seconds: int = 1800, startup_seconds: int = 90,
    ):
        self.template = template.resolve()
        if not (self.template / "project.sqlite3").is_file():
            raise FileNotFoundError(f"Build the starter GeCo project first: {self.template}")
        self.port = port
        self.idle_seconds = idle_seconds
        self.startup_seconds = startup_seconds
        self._lock = threading.RLock()
        self._lease: Lease | None = None

    def _retire_locked(self) -> None:
        lease, self._lease = self._lease, None
        if lease is None:
            return
        if lease.process.poll() is None:
            # Kill the isolated process group, including any training children.
            try:
                os.killpg(lease.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                lease.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(lease.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                lease.process.wait(timeout=5)
        shutil.rmtree(lease.directory, ignore_errors=True)

    def _check_locked(self) -> None:
        lease = self._lease
        if lease is None:
            return
        dead = lease.process.poll() is not None
        idle = lease.in_flight == 0 and time.monotonic() - lease.last_access > self.idle_seconds
        if dead or idle:
            self._retire_locked()

    def active(self) -> bool:
        with self._lock:
            self._check_locked()
            return self._lease is not None

    def claim(self) -> str | None:
        with self._lock:
            self._check_locked()
            if self._lease is not None:
                return None
            directory = Path(tempfile.mkdtemp(prefix="geco-demo-"))
            process = None
            try:
                project_dir = directory / "project.geco"
                shutil.copytree(self.template, project_dir)
                process = subprocess.Popen(
                    [sys.executable, "-m", "demo.worker",
                     "--project", str(project_dir), "--port", str(self.port)],
                    cwd=Path(__file__).resolve().parent.parent,
                    start_new_session=True,
                )
                deadline = time.monotonic() + self.startup_seconds
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("GeCo Dash process exited during startup")
                    try:
                        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                        try:
                            conn.request("GET", "/_dash-layout")
                            response = conn.getresponse()
                            response.read()
                            if response.status == 200:
                                token = secrets.token_urlsafe(32)
                                self._lease = Lease(token, process, directory, time.monotonic())
                                return token
                        finally:
                            conn.close()
                    except (OSError, http.client.HTTPException):
                        pass
                    time.sleep(0.25)
                raise TimeoutError("GeCo did not become ready")
            except Exception:
                if process is not None and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                shutil.rmtree(directory, ignore_errors=True)
                raise

    def borrow(self, token: str | None) -> int | None:
        with self._lock:
            self._check_locked()
            if self._lease is None or not token or not secrets.compare_digest(
                token, self._lease.token
            ):
                return None
            self._lease.in_flight += 1
            self._lease.last_access = time.monotonic()
            return self.port

    def release(self, token: str | None) -> None:
        with self._lock:
            if self._lease is not None and token == self._lease.token:
                self._lease.in_flight = max(0, self._lease.in_flight - 1)
                self._lease.last_access = time.monotonic()

    def end(self, token: str | None) -> bool:
        with self._lock:
            if self._lease is None or not token or not secrets.compare_digest(
                token, self._lease.token
            ):
                return False
            self._retire_locked()
            return True

    def expire(self) -> None:
        with self._lock:
            self._check_locked()

    def close(self) -> None:
        with self._lock:
            self._retire_locked()


def _same_origin_post() -> bool:
    # A malicious website must not be able to claim the only slot or end it.
    if request.headers.get("Sec-Fetch-Site") not in (None, "none", "same-origin"):
        return False
    origin = request.headers.get("Origin")
    return not origin or urlsplit(origin).netloc == request.host


def create_frontend(manager: SlotManager) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
    secure_cookie = os.environ.get("GECO_DEMO_SECURE_COOKIE", "1") != "0"

    @app.after_request
    def avoid_caching(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/__demo/healthz")
    def healthz() -> Response:
        return Response("ok\n", mimetype="text/plain")

    @app.post("/__demo/start")
    def start():
        if not _same_origin_post():
            return Response("Forbidden", status=403)
        if request.cookies.get(COOKIE_NAME) and manager.borrow(
            request.cookies.get(COOKIE_NAME)
        ) is not None:
            manager.release(request.cookies.get(COOKIE_NAME))
            return redirect("/")
        try:
            token = manager.claim()
        except Exception:
            app.logger.exception("Cannot start demo session")
            return Response("Demo startup failed. Please retry later.", status=503)
        if token is None:
            return Response(
                render_template_string(
                    LANDING, busy=True, minutes=manager.idle_seconds // 60
                ), status=503, mimetype="text/html"
            )
        response = make_response(redirect("/"))
        response.set_cookie(
            COOKIE_NAME, token, httponly=True, secure=secure_cookie,
            samesite="Lax", max_age=12 * 3600, path="/",
        )
        return response

    @app.post("/__demo/end")
    def end():
        if not _same_origin_post():
            return Response("Forbidden", status=403)
        manager.end(request.cookies.get(COOKIE_NAME))
        response = make_response(redirect("/"))
        response.delete_cookie(
            COOKIE_NAME, path="/", secure=secure_cookie, httponly=True, samesite="Lax"
        )
        return response

    @app.route("/", defaults={"path": ""}, methods=["GET", "POST", "HEAD"])
    @app.route("/<path:path>", methods=["GET", "POST", "HEAD"])
    def serve(path: str):
        token = request.cookies.get(COOKIE_NAME)
        port = manager.borrow(token)
        if port is None:
            if path:
                return Response("No active demo session", status=403)
            return Response(
                render_template_string(
                    LANDING, busy=manager.active(), minutes=manager.idle_seconds // 60
                ), mimetype="text/html"
            )

        try:
            upstream_path = "/" + path
            if request.query_string:
                upstream_path += "?" + request.query_string.decode("latin-1")
            exclude = HOP_HEADERS | {
                "host", "cookie", "content-length", "accept-encoding",
                "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
            }
            headers = {
                name: value for name, value in request.headers.items()
                if name.lower() not in exclude
            }
            headers["Host"] = f"127.0.0.1:{port}"
            headers["Accept-Encoding"] = "identity"
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
            try:
                conn.request(request.method, upstream_path, body=request.get_data(), headers=headers)
                upstream = conn.getresponse()
                body = upstream.read()
                response_headers = [
                    (key, value) for key, value in upstream.getheaders()
                    if key.lower() not in HOP_HEADERS | {
                        "content-length", "set-cookie", "cache-control"
                    }
                ]
                return Response(body, status=upstream.status, headers=response_headers)
            finally:
                conn.close()
        except (OSError, http.client.HTTPException, TimeoutError):
            app.logger.exception("GeCo worker request failed")
            return Response("Demo worker unavailable. Refresh to restart.", status=502)
        finally:
            manager.release(token)

    return app


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    manager = SlotManager(
        Path(os.getenv("GECO_DEMO_TEMPLATE", str(root / "demo_data" / "starter.geco"))),
        idle_seconds=int(os.getenv("GECO_DEMO_IDLE_SECONDS", "1800")),
    )
    atexit.register(manager.close)

    def janitor() -> None:
        while True:
            time.sleep(15)
            manager.expire()

    threading.Thread(target=janitor, daemon=True).start()
    app = create_frontend(manager)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")),
            debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
