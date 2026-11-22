"""Focused launch-time execution-device and compatibility regressions.

These are mocked device-routing tests, not real CUDA/MPS inference.
"""
from __future__ import annotations

from contextlib import nullcontext
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from geometric_coder import GeometricCoder
from geometric_coder.exceptions import ConfigurationError
from geometric_coder.runtime_device import (
    acceleration_status, current_device, device_scope, provider_transform,
    frozen_teal_device_conflict, resolve_device,
)


def _torch(cuda=0, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda > 0,
            device_count=lambda: cuda,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.mark.parametrize("cuda,mps,requested,expected", [
    (0, False, None, "cpu"), (0, False, "auto", "cpu"),
    (2, True, None, "cuda:0"), (2, True, "CUDA:1", "cuda:1"),
    (0, True, None, "mps"), (0, True, "mps", "mps"),
    (2, True, "cpu", "cpu"), (2, False, "cuda", "cuda:0"),
])
def test_selection_uses_compatible_backend(monkeypatch, cuda, mps, requested, expected):
    real = importlib.import_module
    monkeypatch.setattr(
        importlib, "import_module",
        lambda name: _torch(cuda, mps) if name == "torch" else real(name),
    )
    assert resolve_device(requested) == expected


@pytest.mark.parametrize("requested", ["cuda", "cuda:0", "mps"])
def test_explicit_unavailable_accelerator_raises(monkeypatch, requested):
    monkeypatch.setattr(
        importlib, "import_module", lambda name: _torch() if name == "torch" else None,
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_device(requested)


def test_bad_cuda_index_and_invalid_syntax(monkeypatch):
    monkeypatch.setattr(
        importlib, "import_module", lambda name: _torch(cuda=1),
    )
    with pytest.raises(RuntimeError, match="detected 1"):
        resolve_device("cuda:1")
    for value in ("cuda:-1", "gpu", "cuda:all"):
        with pytest.raises(ValueError, match="device must be"):
            resolve_device(value)


def test_torch_absent_is_cpu_only_for_auto_and_not_required_for_cpu(monkeypatch):
    imported = []
    def missing(name):
        imported.append(name)
        raise ImportError("no torch")
    monkeypatch.setattr(importlib, "import_module", missing)
    assert resolve_device() == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="PyTorch"):
        resolve_device("mps")
    assert imported == ["torch", "torch"]


def test_device_scope_restores_previous_session_and_thread_context():
    assert current_device() is None
    with device_scope("cpu"):
        assert current_device() == "cpu"
        with device_scope("cuda:1"):
            assert current_device() == "cuda:1"
        assert current_device() == "cpu"
    assert current_device() is None


def test_legacy_external_adapter_and_device_forwarding():
    calls = []
    class Legacy:
        def transform_query(self, ref, text):
            calls.append(("legacy", ref, text))
            return "ok"
    class Current:
        def transform_query(self, ref, text, *, device=None):
            calls.append(("current", device))
            return "ok"
    with device_scope("cpu"):
        assert provider_transform(Legacy(), "transform_query", "r", "q") == "ok"
        assert provider_transform(Current(), "transform_query", "r", "q") == "ok"
    assert calls == [("legacy", "r", "q"), ("current", "cpu")]


def test_device_controlled_legacy_provider_fails_without_hiding_type_errors():
    class Old:
        requires_device_control = True
        def transform_texts(self, ref, values):
            return values
    class Broken:
        def transform_texts(self, ref, values, *, device=None):
            raise TypeError("inference error")
    with device_scope("cpu"):
        with pytest.raises(ConfigurationError, match="cannot enforce"):
            provider_transform(Old(), "transform_texts", "r", ["x"])
        with pytest.raises(TypeError, match="inference error"):
            provider_transform(Broken(), "transform_texts", "r", ["x"])


def _project(tmp_path):
    from geometric_coder import CountGeometry
    return GeometricCoder.create(
        project_dir=tmp_path / "device.geco",
        data=pd.DataFrame({"row_id": [1, 2, 3], "text": ["alpha", "beta", "gamma"]}),
        keys=["row_id"], text="text",
        geometries={"lex": CountGeometry(lemmatize=False)},
    )


def test_launch_resolves_before_app_and_remains_fixed(monkeypatch, tmp_path):
    from geometric_coder.ui import app as app_module
    project = _project(tmp_path)
    events = []
    monkeypatch.setattr(app_module, "resolve_device", lambda request: "cuda:1")
    class FakeApp:
        _geco_execution_device = "cuda:1"
        server = None
    def build(*args, **kwargs):
        events.append(("build", kwargs["device"], current_device()))
        return FakeApp()
    class Server:
        port = 1234
        def __init__(self, app, *, host, port):
            self.execution_device = app._geco_execution_device
            events.append(("server", current_device()))
        def start(self):
            events.append(("start", current_device()))
    monkeypatch.setattr(app_module, "create_app", build)
    monkeypatch.setattr("geometric_coder.ui.server.GeCoServer", Server)
    monkeypatch.setattr(app_module, "_display_local_browser_url", lambda *a: None)
    server = app_module.launch_app(
        project, host="127.0.0.1", port=0, debug=False,
        device="cuda", warmup=False,
    )
    assert server.execution_device == "cuda:1"
    assert events == [("build", "cuda:1", "cuda:1"), ("server", None), ("start", None)]
    assert current_device() is None


def test_ui_status_is_read_only_and_never_persisted(tmp_path):
    pytest.importorskip("dash")
    from geometric_coder.ui.app import create_app
    project = _project(tmp_path)

    def find(component, name):
        if getattr(component, "id", None) == name:
            return component
        children = getattr(component, "children", None)
        for child in (children if isinstance(children, (list, tuple)) else [children]):
            if child is not None and not isinstance(child, (str, int, float, bool)):
                result = find(child, name)
                if result is not None:
                    return result
        return None

    cpu = create_app(project, device="cpu")
    gpu = create_app(project, device="cuda:0")
    assert find(cpu.layout, "geco-settings-acceleration-status").children == (
        "GPU acceleration disabled — CPU"
    )
    assert find(gpu.layout, "geco-settings-acceleration-status").children == (
        "GPU acceleration enabled — CUDA:0"
    )
    assert find(gpu.layout, "geco-settings-acceleration-status").role == "status"
    assert not (project.project_dir / "ui_settings.json").exists()
    assert acceleration_status("mps") == "GPU acceleration enabled — MPS"
    assert not any("geco-settings-acceleration-status" in key for key in cpu.callback_map)


def test_flask_request_binding_is_isolated_between_apps(tmp_path):
    pytest.importorskip("dash")
    from geometric_coder.ui.app import create_app
    project = _project(tmp_path)
    cpu = create_app(project, device="cpu")
    gpu = create_app(project, device="mps")
    @cpu.server.get("/_test_execution_device")
    def cpu_device():
        return current_device() or "missing"
    @gpu.server.get("/_test_execution_device")
    def gpu_device():
        return current_device() or "missing"
    assert cpu.server.test_client().get("/_test_execution_device").text == "cpu"
    assert gpu.server.test_client().get("/_test_execution_device").text == "mps"
    assert cpu.server.test_client().get("/_test_execution_device").text == "cpu"
    assert current_device() is None


def test_native_geometry_cache_is_device_separated_and_legacy_conflict(tmp_path):
    from threading import RLock
    import geometric_coder.geometry.embedding as embeddings
    # Use a restored, never-downloaded native encoder and immutable saved config.
    config = {"type": "sentence_transformer", "model": "fake/model",
              "device": None, "normalize_embeddings": True}
    rows = [{"name": "sentence", "storage_kind": "local",
             "config": config, "dependency_names": [],
             "matrix_path": "matrix.npy", "matrix_sparse": False,
             "state_path": "state.joblib"}]
    coder = GeometricCoder.__new__(GeometricCoder)
    coder.project_dir = tmp_path
    coder.database = SimpleNamespace(list_geometries=lambda **kwargs: rows)
    coder.artifacts = SimpleNamespace(
        load_matrix=lambda *args: np.zeros((3, 2), dtype=np.float32),
        load_geometry_state=lambda *args: {},
    )
    from threading import RLock
    coder._runtime_geometries = None
    coder._session_runtime_geometries = {}
    coder._runtime_geometry_lock = RLock()

    with device_scope("cpu"):
        cpu = coder.runtime_geometries()["sentence"]
        assert cpu.options.device == "cpu"
        assert coder.runtime_geometries()["sentence"] is cpu
    with device_scope("cuda:0"):
        gpu = coder.runtime_geometries()["sentence"]
        assert gpu.options.device == "cuda:0"
        assert gpu is not cpu
    with device_scope("cpu"):
        assert coder.runtime_geometries()["sentence"] is cpu
    assert config["device"] is None  # Never changed saved config.

    # A GPU-preferred session may use a historical CPU-only native model.
    config["device"] = "cpu"
    coder._session_runtime_geometries.clear()
    with device_scope("cuda:0"):
        cpu_fallback = coder.runtime_geometries()["sentence"]
        assert cpu_fallback.options.device == "cpu"
    assert config["device"] == "cpu"

    # Conversely, an explicit CPU launch must never run an archived CUDA model.
    config["device"] = "cuda:1"
    coder._session_runtime_geometries.clear()
    with device_scope("cpu"):
        blocked = coder.runtime_geometries()["sentence"]
        assert blocked.options.device == "cuda:1"
        with pytest.raises(ConfigurationError, match="incompatible"):
            blocked.transform_query("test")
        with pytest.raises(ConfigurationError, match="incompatible"):
            blocked.transform_texts(["test"])
    assert config["device"] == "cuda:1"


def test_native_encoder_loads_selected_session_device_without_reusing_other_model(monkeypatch):
    import sys
    from geometric_coder.geometry import SentenceTransformerGeometry
    created = []
    class FakeModel:
        def __init__(self, name, *, device):
            created.append((name, device))
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 2), dtype=np.float32)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeModel),
    )
    cpu = SentenceTransformerGeometry("fake/model", device="cpu")
    gpu = SentenceTransformerGeometry("fake/model", device="cuda:1")
    assert cpu.transform_texts(["a"]).shape == (1, 2)
    assert gpu.transform_texts(["b"]).shape == (1, 2)
    assert cpu.transform_query("c").shape == (1, 2)
    assert created == [("fake/model", "cpu"), ("fake/model", "cuda:1")]


def test_real_project_query_forwards_cpu_to_device_capable_provider(tmp_path):
    """GeCo's query API, not just its helper, reaches the device-aware provider."""
    calls = []
    class Provider:
        def geometry_matrix(self, ref, keys):
            return np.asarray([[1., 0.], [0., 1.]], dtype=np.float32)
        def view_coordinates(self, ref, keys):
            return np.asarray([[0., 0.], [1., 1.]], dtype=np.float32)
        def transform_query(self, ref, text, *, device=None):
            calls.append((ref, text, device))
            return np.asarray([[1., 0.]], dtype=np.float32)
    provider = Provider()
    project = GeometricCoder.create_external(
        project_dir=tmp_path / "provider.geco",
        data=pd.DataFrame({"id": [1, 2], "text": ["one", "two"]}),
        keys=["id"], text="text", external_provider=provider,
    )
    project.register_external_geometry(
        name="ext", external_ref={"id": "x"}, supports_query=True,
    )
    with device_scope("cpu"):
        assert project.transform_query("ext", "hello").shape == (1, 2)
        assert len(project.semantic_search("hello", geometry_names=["ext"])["ext"]) == 2
    assert calls == [
        ({"id": "x"}, "hello", "cpu"),
        ({"id": "x"}, "hello", "cpu"),
    ]


def test_resolver_has_no_persistent_project_side_effects(tmp_path, monkeypatch):
    project = _project(tmp_path)
    before = sorted(str(p.relative_to(project.project_dir))
                    for p in project.project_dir.rglob("*") if p.is_file())
    with device_scope("cpu"):
        assert current_device() == "cpu"
        assert project.runtime_geometries()
    after = sorted(str(p.relative_to(project.project_dir))
                   for p in project.project_dir.rglob("*") if p.is_file())
    assert before == after


def test_teal_frozen_device_conflict_is_unwrapped_without_hiding_other_failures():
    """Recognize a genuine TeAL operator lock, not every provider RuntimeError."""
    errors = pytest.importorskip("text_analysis_lab.core.errors")
    from geometric_coder.runtime_device import frozen_teal_device_conflict

    def wrapped(reason):
        try:
            raise errors.OperatorError(reason)
        except errors.OperatorError as original:
            try:
                raise RuntimeError("TeAL could not transform query") from original
            except RuntimeError as provider_error:
                return provider_error

    cpu_lock = frozen_teal_device_conflict(
        wrapped("Frozen device lock requires 'cpu', not 'cuda:0'.")
    )
    assert cpu_lock is not None
    assert "requires 'cpu', not 'cuda:0'" in cpu_lock
    assert "relaunch GeCo" in cpu_lock

    gpu_lock = frozen_teal_device_conflict(
        wrapped("Frozen device lock requires 'cuda', not 'cpu'.")
    )
    assert gpu_lock is not None
    assert "requires 'cuda', not 'cpu'" in gpu_lock

    legacy = frozen_teal_device_conflict(
        wrapped("Legacy frozen operator requests 'cuda:0' (currently resolves to "
                "'cuda:0'); an explicit 'cpu' override would change its saved "
                "device behavior.")
    )
    assert legacy is not None
    assert "Legacy frozen operator" in legacy

    assert frozen_teal_device_conflict(RuntimeError("ordinary provider failure")) is None
    assert frozen_teal_device_conflict(wrapped("unrelated operator error")) is None



def test_cpu_lexical_search_ignores_unrelated_cuda_native_geometry(tmp_path, monkeypatch):
    """A mixed persisted project never initializes an incompatible encoder for lexical search."""
    import sys
    project = _project(tmp_path)
    lexical = project.database.get_geometry_by_name("lex")
    # Register a historical CUDA-only native geometry with an existing matrix
    # sidecar. No neural model is downloaded or loaded during the test.
    project.database.register_geometry(
        name="embeddings",
        class_name="SentenceTransformerGeometry",
        modality="text",
        public=True,
        config={"type": "sentence_transformer", "model": "fake/legacy",
                "device": "cuda:0"},
        dependency_names=[],
        matrix_path=lexical["matrix_path"],
        matrix_sparse=lexical["matrix_sparse"],
        state_path=None,
        supports_query=True,
        supports_text_transform=True,
    )
    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("Incompatible CUDA encoder must not be initialized")
    monkeypatch.setitem(
        sys.modules, "sentence_transformers",
        SimpleNamespace(SentenceTransformer=forbidden_constructor),
    )

    with device_scope("cpu"):
        restored = project.runtime_geometries()
        assert set(restored) == {"lex", "embeddings"}
        assert restored["embeddings"].options.device == "cuda:0"
        assert restored["embeddings"]._model is None
        # This goes through the actual semantic-search / transform_query path.
        scores = project.semantic_search("alpha", geometry_names=["lex"])
        assert len(scores["lex"]) == 3
        assert len(project.transform_query("lex", "beta").shape) == 2
        with pytest.raises(ConfigurationError, match="Native geometry 'embeddings'.*incompatible"):
            project.semantic_search("alpha", geometry_names=["embeddings"])
        with pytest.raises(ConfigurationError, match="Native geometry 'embeddings'.*incompatible"):
            project.runtime_geometries()["embeddings"].transform_texts(["new text"])
        # Failure of embeddings must not poison the lexical cache.
        assert len(project.semantic_search("gamma", geometry_names=["lex"])["lex"]) == 3
    assert project.database.get_geometry_by_name("embeddings")["config"]["device"] == "cuda:0"
