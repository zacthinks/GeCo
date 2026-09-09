"""SentenceTransformer-based embedding geometries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from geometric_coder.exceptions import MissingOptionalDependencyError
from geometric_coder.geometry.base import Geometry, Matrix, ViewSpec


@dataclass(frozen=True, slots=True)
class EmbeddingOptions:
    model: str
    normalize_embeddings: bool = True
    batch_size: int = 32
    device: str | None = None
    show_progress_bar: bool = True


class SentenceTransformerGeometry(Geometry):
    """Dense sentence or short-document embeddings."""

    default_view = ViewSpec(
        "umap",
        {
            "n_components": 2,
            "metric": "cosine",
            "n_neighbors": 15,
            "min_dist": 0.05,
            "spread": 2.5,
            "repulsion_strength": 1.5,
            "random_state": 0,
        },
    )

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__()
        self.options = EmbeddingOptions(model=model, **kwargs)
        self._model: Any | None = None

    @property
    def supports_query(self) -> bool:
        return self._model is not None

    @property
    def supports_text_transform(self) -> bool:
        return self._model is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise MissingOptionalDependencyError(
                "SentenceTransformerGeometry requires the 'embeddings' extra: "
                "uv sync --extra embeddings"
            ) from None
        self._model = SentenceTransformer(self.options.model, device=self.options.device)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        encoded = model.encode(
            texts,
            batch_size=self.options.batch_size,
            show_progress_bar=self.options.show_progress_bar,
            normalize_embeddings=self.options.normalize_embeddings,
            convert_to_numpy=True,
        )
        return np.asarray(encoded, dtype=np.float32)

    def _fit_transform(self, texts: list[str]) -> Matrix:
        return self._encode(texts)

    def transform_texts(self, texts: list[str]) -> Matrix:
        return self._encode(texts)

    def configuration(self) -> dict[str, Any]:
        return {"type": "sentence_transformer", **asdict(self.options)}

    def artifact_state(self) -> dict[str, Any]:
        # The model identifier is enough to reconstruct the encoder. Avoid
        # serializing the full neural model into every project.
        return {}
