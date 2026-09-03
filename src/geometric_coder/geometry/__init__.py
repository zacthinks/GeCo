"""Public geometry classes."""

from geometric_coder.geometry.base import Geometry, Matrix, ViewSpec
from geometric_coder.geometry.count import CountGeometry, CountOptions
from geometric_coder.geometry.derived import SVDGeometry, SVDOptions
from geometric_coder.geometry.embedding import EmbeddingOptions, SentenceTransformerGeometry

__all__ = [
    "CountGeometry",
    "CountOptions",
    "EmbeddingOptions",
    "Geometry",
    "Matrix",
    "SVDGeometry",
    "SVDOptions",
    "SentenceTransformerGeometry",
    "ViewSpec",
]
