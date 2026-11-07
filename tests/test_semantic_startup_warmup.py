"""Launch-time semantic-library warm-up is bounded and model-independent."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

from geometric_coder.ui import app as app_module


class _Project:
    def __init__(self, *, external=True, queryable=True):
        self.is_external_backed = external
        self._queryable = queryable

    def geometries(self, *, public_only):
        assert public_only
        return [
            {
                "storage_kind": "external",
                "supports_query": self._queryable,
            }
        ]


def test_warmup_selection_only_for_queryable_external_standard_app():
    external = _Project()
    assert app_module._should_warm_semantic_runtime(external, mode="standard")
    assert not app_module._should_warm_semantic_runtime(external, mode="focus")
    assert not app_module._should_warm_semantic_runtime(
        _Project(queryable=False), mode="standard"
    )
    assert not app_module._should_warm_semantic_runtime(
        _Project(external=False), mode="standard"
    )


def test_semantic_warmup_imports_libraries_and_runs_disposable_cpu_kernel(monkeypatch):
    imported = []
    computations = []

    class Tensor:
        T = None

    tensor = Tensor()
    tensor.T = tensor

    def make_tensor(values, *, dtype):
        assert values == [[1.0, 0.0]]
        assert dtype == "float32"
        computations.append("tensor")
        return tensor

    torch = SimpleNamespace(
        inference_mode=nullcontext,
        tensor=make_tensor,
        float32="float32",
        matmul=lambda left, right: computations.append("matmul"),
    )

    def fake_import(name):
        imported.append(name)
        if name == "torch":
            return torch
        if name == "sentence_transformers":
            return object()
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert app_module._warm_semantic_runtime_libraries() is True
    assert imported == ["torch", "sentence_transformers"]
    assert computations == ["tensor", "matmul"]


def test_semantic_warmup_does_not_block_launch_if_optional_dependency_missing(monkeypatch):
    def fake_import(name):
        raise ImportError(f"{name} not installed")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    assert app_module._warm_semantic_runtime_libraries() is False

    def broken_backend(name):
        raise RuntimeError(f"{name}: broken optional backend")

    monkeypatch.setattr(importlib, "import_module", broken_backend)
    assert app_module._warm_semantic_runtime_libraries() is False


def test_launch_warms_before_server_start_only_when_relevant(monkeypatch):
    events = []

    class FakeServer:
        port = 8050

        def __init__(self, app, *, host, port):
            assert app == "app"
            events.append("server init")

        def start(self):
            events.append("server start")

    monkeypatch.setattr(
        "geometric_coder.ui.server.GeCoServer", FakeServer
    )
    monkeypatch.setattr(
        app_module, "create_app", lambda *args, **kwargs: "app"
    )
    monkeypatch.setattr(
        app_module, "_warm_semantic_runtime_libraries",
        lambda: events.append("warm") or True,
    )
    monkeypatch.setattr(app_module, "_display_local_browser_url", lambda *args: None)
    app_module.launch_app(_Project(), host="127.0.0.1", port=8050, debug=False)
    assert events == ["warm", "server init", "server start"]
    events.clear()
    app_module.launch_app(
        _Project(external=False), host="127.0.0.1", port=8050, debug=False
    )
    assert events == ["server init", "server start"]
