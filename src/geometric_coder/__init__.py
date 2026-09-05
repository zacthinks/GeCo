"""Geometric Coder (GeCo)."""

from geometric_coder.geometry import (
    CountGeometry,
    Geometry,
    SVDGeometry,
    SentenceTransformerGeometry,
    ViewSpec,
)
from geometric_coder.external import ExternalDataProvider
from geometric_coder.project import GeometricCoder

__all__ = [
    "CountGeometry",
    "ExternalDataProvider",
    "GeometricCoder",
    "Geometry",
    "SVDGeometry",
    "SentenceTransformerGeometry",
    "ViewSpec",
]

__version__ = "0.8.1"
