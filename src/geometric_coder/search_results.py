"""Durable, identity-checked GeCo semantic-search results.

Search scores are immutable snapshots until an explicit new Search. The database
stores compressed float64 arrays so rows, metadata, and vectors commit atomically.
"""
from __future__ import annotations

import hashlib
import json
import zlib
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from geometric_coder.exceptions import ConfigurationError

if TYPE_CHECKING:
    from geometric_coder.project import GeometricCoder


class SavedSemanticSearches:
    """Project-local semantic-search results; never mutate provider artifacts."""

    def __init__(self, project: GeometricCoder) -> None:
        self.project = project

    def _signature(self, geometry_id: int) -> tuple[str, int, str]:
        """Fingerprint immutable resource identity and ordered atomic source keys.

        External resources must use stable immutable references. This cannot
        detect provider-side mutation behind an unchanged opaque reference.
        """
        db = self.project.database
        record = db.get_geometry(int(geometry_id))
        if not bool(record["public"]):
            raise ConfigurationError("Saved search requires a public geometry.")
        source = {
            "geometry_id": int(geometry_id),
            "class_name": record["class_name"],
            "storage_kind": record["storage_kind"],
            "config": record["config"],
            "dependencies": record["dependency_names"],
            "matrix_path": record["matrix_path"],
            "state_path": record["state_path"],
            "created_at": record["created_at"],
            "external_ref": record["external_ref"],
        }
        # Renames are intentionally not part of the signature.
        if record["storage_kind"] == "local":
            for field in ("matrix_path", "state_path"):
                relative = record[field]
                if relative:
                    path = self.project.project_dir / relative
                    try:
                        stat = path.stat()
                    except FileNotFoundError:
                        source[field + "_file"] = None
                    else:
                        source[field + "_file"] = [stat.st_size, stat.st_mtime_ns]
        hasher = hashlib.sha256(
            json.dumps(source, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        )
        count = 0
        # Hash the ordered universe in batches, without loading all source texts.
        with db.connect() as connection:
            cursor = connection.execute(
                "SELECT row_position, unit_id, user_key_json FROM units ORDER BY row_position"
            )
            for row in cursor:
                hasher.update(json.dumps(
                    [int(row["row_position"]), int(row["unit_id"]), json.loads(row["user_key_json"])],
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8"))
                hasher.update(b"\\n")
                count += 1
        return hasher.hexdigest(), count, str(record["name"])

    def save(
        self,
        *,
        geometry_id: int,
        query: str,
        scores: Sequence[float],
        name: str | None = None,
        session_id: int | None = None,
        range_values: Sequence[float] | None = None,
    ) -> int:
        """Save a named result or replace the latest result for session+geometry.

        Exactly one of name and session_id must be given. Named search titles
        are unique across the project, case-insensitively.
        """
        title = (name or "").strip()
        clean_query = str(query).strip()
        if not clean_query:
            raise ValueError("A saved search must have a nonempty query.")
        if bool(title) == (session_id is not None):
            raise ValueError("Provide either a name or a session ID, not both.")
        signature, count, _ = self._signature(int(geometry_id))
        vector = np.asarray(scores, dtype=np.float64)
        if vector.ndim != 1 or vector.shape[0] != count or not np.isfinite(vector).all():
            raise ValueError("Saved similarity scores must be finite and match the atomic universe.")
        selected_range: list[float] | None = None
        if range_values is not None:
            selected_range = [float(x) for x in range_values]
            if len(selected_range) != 2 or not all(np.isfinite(selected_range)):
                raise ValueError("Saved similarity range must have two finite bounds.")
        payload = zlib.compress(vector.astype("<f8", copy=False).tobytes())
        kind = "named" if title else "auto"
        with self.project.database.connect() as connection:
            if kind == "auto":
                session = connection.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?", (int(session_id),)
                ).fetchone()
                if session is None:
                    raise KeyError(f"Unknown session ID: {session_id}")
                connection.execute(
                    "DELETE FROM semantic_search_results "
                    "WHERE kind = 'auto' AND session_id = ? AND geometry_id = ?",
                    (int(session_id), int(geometry_id)),
                )
            try:
                cursor = connection.execute(
                    """INSERT INTO semantic_search_results
                       (kind, name, session_id, geometry_id, query, signature,
                        unit_count, scores_blob, range_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
                    (kind, title if title else None, int(session_id) if session_id is not None else None,
                     int(geometry_id), clean_query, signature, count,
                     payload, json.dumps(selected_range)),
                )
            except Exception as exc:
                if "UNIQUE constraint failed" in str(exc):
                    raise ValueError(f"Saved search name {title!r} already exists.") from exc
                raise
            return int(cursor.lastrowid)

    def list(self, *, session_id: int | None = None) -> list[dict[str, Any]]:
        """Return saved-search metadata without reading compressed score blobs."""
        with self.project.database.connect() as connection:
            if session_id is None:
                rows = connection.execute(
                    """SELECT result_id, kind, name, session_id, geometry_id,
                              query, created_at, unit_count FROM semantic_search_results
                       WHERE kind = 'named' ORDER BY name COLLATE NOCASE"""
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT result_id, kind, name, session_id, geometry_id,
                              query, created_at, unit_count FROM semantic_search_results
                       WHERE kind = 'named' OR (kind = 'auto' AND session_id = ?)
                       ORDER BY kind, created_at DESC""", (int(session_id),)
                ).fetchall()
        return [dict(row) for row in rows]

    def load(self, result_id: int) -> dict[str, Any]:
        """Read scores after checking representation and source-universe identity."""
        with self.project.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM semantic_search_results WHERE result_id = ?", (int(result_id),)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown saved search: {result_id}")
            record = dict(row)
        try:
            signature, count, geometry_name = self._signature(int(record["geometry_id"]))
        except (KeyError, ConfigurationError) as exc:
            raise ValueError("Saved search is stale: its geometry is unavailable.") from exc
        if signature != record["signature"] or count != int(record["unit_count"]):
            raise ValueError("Saved search is stale: geometry or ordered source universe changed.")
        try:
            scores = np.frombuffer(zlib.decompress(record["scores_blob"]), dtype="<f8")
        except (zlib.error, ValueError) as exc:
            raise ValueError("Saved search scores are unreadable.") from exc
        if scores.size != count or not np.isfinite(scores).all():
            raise ValueError("Saved search scores are corrupt or misaligned.")
        return {
            "result_id": int(record["result_id"]),
            "kind": record["kind"],
            "name": record["name"],
            "session_id": record["session_id"],
            "geometry_id": int(record["geometry_id"]),
            "geometry_name": geometry_name,
            "query": record["query"],
            "scores": scores.astype(float).tolist(),
            "range": json.loads(record["range_json"]),
            "created_at": record["created_at"],
        }

    def rename(self, result_id: int, name: str) -> None:
        clean = str(name).strip()
        if not clean:
            raise ValueError("Saved search name cannot be empty.")
        with self.project.database.connect() as connection:
            row = connection.execute(
                "UPDATE semantic_search_results SET name = ? "
                "WHERE result_id = ? AND kind = 'named'",
                (clean, int(result_id)),
            )
            if row.rowcount != 1:
                raise KeyError(f"Unknown named saved search: {result_id}")

    def delete(self, result_id: int) -> None:
        with self.project.database.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM semantic_search_results WHERE result_id = ?", (int(result_id),)
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown saved search: {result_id}")

    def latest(self, session_id: int, geometry_id: int) -> dict[str, Any] | None:
        with self.project.database.connect() as connection:
            row = connection.execute(
                """SELECT result_id FROM semantic_search_results
                   WHERE kind = 'auto' AND session_id = ? AND geometry_id = ?""",
                (int(session_id), int(geometry_id)),
            ).fetchone()
        return self.load(int(row[0])) if row is not None else None
