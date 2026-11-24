"""Per-launch execution placement for GeCo-controlled accelerator work.

A Flask request binds its own server's immutable choice. Outside a launched
GeCo request, direct Python use keeps the historical geometry/provider defaults.
No selection is stored in a research project or installed as a Torch global.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
import importlib
import inspect
import re
from typing import Any, Iterator

from geometric_coder.exceptions import ConfigurationError

_device: ContextVar[str | None] = ContextVar("geco_execution_device", default=None)


def current_device() -> str | None:
    """Chosen device within this launch's request, or None outside a session."""
    return _device.get()


def bind_device(device: str) -> Token[str | None]:
    return _device.set(device)


def unbind_device(token: Token[str | None]) -> None:
    _device.reset(token)


@contextmanager
def device_scope(device: str) -> Iterator[None]:
    token = bind_device(device)
    try:
        yield
    finally:
        unbind_device(token)


def resolve_device(requested: str | None = None) -> str:
    """Resolve a real Torch-compatible backend once without importing models.

    CPU never requires Torch; an absent/broken optional Torch installation is a
    normal CPU fallback for 'auto', but explicit unavailable devices are errors.
    """
    name = "auto" if requested is None else str(requested).strip().lower()
    name = name or "auto"
    if name not in {"auto", "cpu", "cuda", "mps"} and not re.fullmatch(r"cuda:\d+", name):
        raise ValueError("device must be 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'.")
    if name == "cpu":
        return "cpu"
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError, RuntimeError) as exc:
        if name == "auto":
            return "cpu"
        raise RuntimeError(
            f"GeCo device {name!r} is unavailable: PyTorch could not be loaded ({exc})."
        ) from exc

    def cuda_count() -> int:
        try:
            return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        except (AttributeError, OSError, RuntimeError, AssertionError):
            return 0

    def mps_available() -> bool:
        try:
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            return bool(mps is not None and mps.is_available())
        except (AttributeError, OSError, RuntimeError, AssertionError):
            return False

    if name in {"auto", "cuda"} or name.startswith("cuda:"):
        count = cuda_count()
        if count:
            if name in {"auto", "cuda"}:
                return "cuda:0"
            index = int(name.split(":", 1)[1])
            if index < count:
                return f"cuda:{index}"
        if name != "auto":
            raise RuntimeError(
                f"GeCo device {name!r} is unavailable: detected {count} usable CUDA device(s)."
            )
    if name in {"auto", "mps"}:
        if mps_available():
            return "mps"
        if name == "mps":
            raise RuntimeError("GeCo device 'mps' is unavailable in this PyTorch runtime.")
    return "cpu"


def acceleration_status(device: str) -> str:
    """Concise Settings label for the session's resolved execution device."""
    if device == "cpu":
        return "CPU"
    return f"GPU · {device.upper()}"


def frozen_teal_device_conflict(error: BaseException) -> str | None:
    """Expose TeAL's frozen-device conflict through a wrapped provider failure.

    TeAL is optional. Import its error class only on this failure path, and
    inspect only an explicitly recognized frozen-device rejection. Other
    provider runtime failures must not be reported as harmless UI statuses.
    """
    try:
        from text_analysis_lab.core.errors import OperatorError
    except ImportError:
        return None

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OperatorError):
            detail = str(current)
            if detail.startswith((
                "Frozen device lock requires ",
                "Legacy frozen operator requests ",
            )):
                return (
                    "This geometry's frozen TeAL model has an incompatible "
                    f"execution-device requirement. {detail} "
                    "Close and relaunch GeCo with a compatible device, or "
                    "create a device-portable frozen model."
                )
        current = current.__cause__
    return None


def _accepts_device(fn: Any) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    explicit = parameters.get("device")
    return (
        explicit is not None
        and explicit.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ) or any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())


def provider_transform(provider: Any, method: str, ref: Any, value: Any) -> Any:
    """Forward to device-aware providers without breaking legacy CPU adapters.

    A known TeAL or explicitly declared device-controlled provider must accept
    the new API; other opaque two-argument providers retain their old contract.
    """
    transform = getattr(provider, method)
    selected = current_device()
    if selected is None:
        return transform(ref, value)
    if _accepts_device(transform):
        return transform(ref, value, device=selected)
    controlled = (
        type(provider).__module__.startswith("text_analysis_lab.")
        or bool(getattr(provider, "requires_device_control", False))
    )
    if controlled:
        raise ConfigurationError(
            f"Provider {type(provider).__name__} cannot enforce this GeCo session's "
            f"device {selected!r}: {method}() lacks the runtime device argument. "
            "Use a TeAL build with execution-device forwarding or relaunch with "
            "a compatible provider."
        )
    return transform(ref, value)
