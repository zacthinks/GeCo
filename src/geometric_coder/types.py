"""Shared public typing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from geometric_coder.geometry.base import Geometry

GeometrySpec: TypeAlias = Geometry | Mapping[str, "GeometrySpec"]
