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

    def register(self, query: str) -> int:
        """Register a query once, independently of geometry and session."""
        clean = str(query).strip()
        if not clean:
            raise ValueError("Search query must not be empty.")
        with self.project.database.connect() as con:
            con.execute(
                "INSERT OR IGNORE INTO semantic_queries(query, created_at) "
                "VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))", (clean,),
            )
            return int(con.execute(
                "SELECT query_id FROM semantic_queries WHERE query = ?", (clean,),
            ).fetchone()[0])

    def query_id(self, query: str) -> int | None:
        """Return an existing query identity without registering or reviving it."""
        with self.project.database.connect() as con:
            row = con.execute(
                "SELECT query_id FROM semantic_queries WHERE query = ?",
                (str(query).strip(),),
            ).fetchone()
        return int(row[0]) if row is not None else None

    def queries(self) -> list[dict[str, Any]]:
        with self.project.database.connect() as con:
            return [dict(x) for x in con.execute(
                "SELECT query_id, query, created_at FROM semantic_queries "
                "ORDER BY created_at DESC, query_id DESC"
            )]

    def delete_query(self, query_id: int) -> None:
        with self.project.database.connect() as con:
            result = con.execute(
                "DELETE FROM semantic_queries WHERE query_id = ?", (int(query_id),),
            )
            if result.rowcount != 1:
                raise KeyError(f"Unknown saved query: {query_id}")

    def find(self, query: str, geometry_id: int, *,
             expected_query_id: int | None = None) -> dict[str, Any] | None:
        clean = str(query).strip()
        with self.project.database.connect() as con:
            row = con.execute(
                """SELECT r.*, q.query FROM semantic_search_results r
                   JOIN semantic_queries q ON q.query_id=r.query_id
                   WHERE q.query=? AND r.geometry_id=?
                   AND (? IS NULL OR q.query_id=?)""",
                (clean, int(geometry_id), expected_query_id, expected_query_id),
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        try:
            signature, count, name = self._signature(int(geometry_id))
        except (KeyError, ConfigurationError) as error:
            raise ValueError("Saved search is stale: geometry unavailable.") from error
        if signature != record["signature"] or count != int(record["unit_count"]):
            raise ValueError("Saved search is stale: geometry or ordered keys changed.")
        try:
            values = np.frombuffer(zlib.decompress(record["scores_blob"]), dtype="<f8")
        except (zlib.error, ValueError) as error:
            raise ValueError("Saved search scores are unreadable.") from error
        if values.size != count or not np.isfinite(values).all():
            raise ValueError("Saved search scores are corrupt or misaligned.")
        return {"query": clean, "geometry_id": int(geometry_id), "geometry_name": name,
                "scores": values.astype(float).tolist(),
                "range": json.loads(record["range_json"])}

    def store(self, query: str, geometry_id: int, scores: Sequence[float],
              range_values: Sequence[float] | None = None, *,
              expected_query_id: int | None = None) -> int:
        signature, count, _ = self._signature(int(geometry_id))
        vector = np.asarray(scores, dtype=np.float64)
        if vector.ndim != 1 or vector.size != count or not np.isfinite(vector).all():
            raise ValueError("Saved similarity scores must match the atomic universe.")
        clean = str(query).strip()
        if not clean:
            raise ValueError("Search query must not be empty.")
        chosen = None
        if range_values is not None:
            chosen = [float(x) for x in range_values]
            if len(chosen) != 2 or not all(np.isfinite(chosen)):
                raise ValueError("Invalid saved similarity range.")
        blob = zlib.compress(vector.astype("<f8", copy=False).tobytes())
        with self.project.database.connect() as con:
            if expected_query_id is None:
                # The programmatic API may create an automatically saved query.
                con.execute(
                    "INSERT OR IGNORE INTO semantic_queries(query, created_at) "
                    "VALUES (?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
                    (clean,),
                )
                qid = int(con.execute(
                    "SELECT query_id FROM semantic_queries WHERE query=?", (clean,),
                ).fetchone()[0])
            else:
                # UI requests capture query identity *before* computation.
                # If deletion won the race, never register a replacement row.
                row = con.execute(
                    "SELECT query_id FROM semantic_queries WHERE query_id=? AND query=?",
                    (int(expected_query_id), clean),
                ).fetchone()
                if row is None:
                    raise KeyError("Semantic search was deleted while calculating scores.")
                qid = int(row[0])
            con.execute(
                """INSERT INTO semantic_search_results
                   (query_id, geometry_id, signature, unit_count, scores_blob, range_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                   ON CONFLICT(query_id,geometry_id) DO UPDATE SET
                   signature=excluded.signature,unit_count=excluded.unit_count,
                   scores_blob=excluded.scores_blob,range_json=excluded.range_json,
                   created_at=excluded.created_at""",
                (qid, int(geometry_id), signature, count, blob, json.dumps(chosen)),
            )
            return qid
