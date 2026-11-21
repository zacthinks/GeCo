"""Automatic local port binding across public GeCo launch surfaces."""

from __future__ import annotations

import inspect
import socket
from types import SimpleNamespace
from urllib.request import urlopen

import pytest
from flask import Flask

from geometric_coder import GeometricCoder
import geometric_coder.cli as cli
import geometric_coder.ui as ui
import geometric_coder.ui.app as app_module


def _lightweight_app():
    flask_app = Flask(f"geco-port-test-{id(object())}")

    @flask_app.get("/")
    def index():
        return "geco-port-ready"

    return SimpleNamespace(server=flask_app)


def test_two_automatic_servers_bind_distinct_reachable_ports(monkeypatch):
    """The OS assigns ports atomically; neither a port probe nor 8050 is needed."""
    shown = []
    monkeypatch.setattr(app_module, "create_app", lambda *args, **kwargs: _lightweight_app())
    monkeypatch.setattr(
        app_module, "_display_local_browser_url",
        lambda host, port: shown.append((host, port)),
    )
    servers = []
    try:
        for _ in range(2):
            servers.append(
                app_module.launch_app(
                    object(), host="127.0.0.1", port=0,
                    debug=False, warmup=False,
                )
            )
        assert servers[0].port > 0
        assert servers[1].port > 0
        assert servers[0].port != servers[1].port
        assert shown == [("127.0.0.1", server.port) for server in servers]
        for server in servers:
            assert server.url == f"http://127.0.0.1:{server.port}/"
            with urlopen(server.url, timeout=5) as response:
                assert response.status == 200
                assert response.read() == b"geco-port-ready"
    finally:
        for server in reversed(servers):
            server.close()


def test_occupied_explicit_port_fails_without_success_url(monkeypatch):
    shown = []
    monkeypatch.setattr(app_module, "create_app", lambda *args, **kwargs: _lightweight_app())
    monkeypatch.setattr(
        app_module, "_display_local_browser_url",
        lambda *args: shown.append(args),
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        occupied = listener.getsockname()[1]
        with pytest.raises(OSError, match="port may already be in use"):
            app_module.launch_app(
                object(), host="127.0.0.1", port=occupied,
                debug=False, warmup=False,
            )
    assert shown == []


def test_start_failure_does_not_advertise_success(monkeypatch):
    shown = []
    monkeypatch.setattr(app_module, "create_app", lambda *args, **kwargs: object())
    monkeypatch.setattr(app_module, "_display_local_browser_url", lambda *args: shown.append(args))

    class FailedServer:
        def __init__(self, app, *, host, port):
            assert port == 0

        def start(self):
            raise RuntimeError("startup failed")

    monkeypatch.setattr("geometric_coder.ui.server.GeCoServer", FailedServer)
    with pytest.raises(RuntimeError, match="startup failed"):
        app_module.launch_app(object(), host="127.0.0.1", port=0, debug=False, warmup=False)
    assert shown == []


@pytest.mark.parametrize("requested,expected", [(None, 0), (0, 0), (8124, 8124)])
def test_python_launcher_forwards_auto_or_strict_port(monkeypatch, requested, expected):
    received = []

    def fake_launch(project, **kwargs):
        received.append(kwargs)
        return object()

    monkeypatch.setattr(ui, "launch_app", fake_launch)
    GeometricCoder.launch(object(), port=requested, warmup=False)
    assert received[0]["port"] == expected
    assert received[0]["host"] == "127.0.0.1"
    assert received[0]["use_reloader"] is False
    assert received[0]["warmup"] is False


@pytest.mark.parametrize("requested,expected", [(None, 0), (0, 0), (8124, 8124)])
def test_direct_focus_launcher_forwards_auto_or_strict_port(monkeypatch, requested, expected):
    received = []
    monkeypatch.setattr(ui, "launch_app", lambda project, **kwargs: received.append(kwargs))
    ui.launch_focus_coder(object(), port=requested)
    assert received[0]["port"] == expected
    assert received[0]["mode"] == "focus"
    assert received[0]["use_reloader"] is False


def test_python_and_direct_focus_signatures_default_to_auto():
    assert inspect.signature(GeometricCoder.launch).parameters["port"].default is None
    assert inspect.signature(GeometricCoder.launch_focus_coder).parameters["port"].default is None
    assert inspect.signature(ui.launch_focus_coder).parameters["port"].default is None


@pytest.mark.parametrize("argv,expected", [
    (["launch", "project.geco"], None),
    (["launch", "project.geco", "--port", "8124"], 8124),
    (["launch", "project.geco", "--port", "0"], 0),
])
def test_cli_port_omission_and_explicit_override(monkeypatch, argv, expected):
    requests = []

    class Handle:
        def wait(self):
            requests.append("wait")

        def close(self):
            requests.append("close")

    class Project:
        def launch(self, **kwargs):
            requests.append(kwargs)
            return Handle()

    monkeypatch.setattr(cli.GeometricCoder, "open", lambda path: Project())
    assert cli.main(argv) == 0
    assert requests[0]["port"] == expected
    assert requests[0]["host"] == "127.0.0.1"
    assert requests[0]["debug"] is False
    assert requests[0]["warmup"] is True
    assert requests[1:] == ["wait", "close"]
