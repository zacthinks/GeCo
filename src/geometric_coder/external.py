"""Runtime protocol for externally backed numerical GeCo resources."""

from __future__ import annotations

from typing import Any, Protocol

from geometric_coder.geometry.base import Matrix


class ExternalDataProvider(Protocol):
    """Resolve opaque external geometry/view references at runtime.

    GeCo persists ``external_ref`` values as JSON and never interprets them. The
    provider is responsible for resolving those references, aligning rows by the
    supplied stable user keys, and returning rows in the same order.

    ``transform_query`` and ``transform_texts`` are optional capabilities. GeCo
    checks for them with ``getattr`` before exposing semantic search or creating
    derived observations that require new numerical vectors.
    """

    def geometry_matrix(
        self, external_ref: Any, ordered_user_keys: list[dict[str, Any]]
    ) -> Matrix: ...

    def view_coordinates(
        self, external_ref: Any, ordered_user_keys: list[dict[str, Any]]
    ) -> Any: ...
