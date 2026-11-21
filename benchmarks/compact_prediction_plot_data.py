"""Reproducible synthetic SQLite read-workload comparison for Develop plot data.

Run: python benchmarks/compact_prediction_plot_data.py
This measures observation enumeration only, not classifier computation or UI latency.
"""
from __future__ import annotations

import json
import sqlite3
import time
import tracemalloc


def run(rows: int = 5000, text_bytes: int = 4096, metadata_bytes: int = 1024) -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE observations (
            observation_id INTEGER PRIMARY KEY, kind TEXT, text TEXT, created_at TEXT
        );
        CREATE TABLE units (
            unit_id INTEGER PRIMARY KEY, observation_id INTEGER,
            row_position INTEGER, user_key_json TEXT, metadata_json TEXT
        );
        CREATE TABLE teaching_examples (observation_id INTEGER, status TEXT);
    """)
    connection.executemany(
        "INSERT INTO observations VALUES (?, ?, ?, ?)",
        ((i, "atomic", "T" * text_bytes, "2026-11-21") for i in range(1, rows + 1)),
    )
    connection.executemany(
        "INSERT INTO units VALUES (?, ?, ?, ?, ?)",
        (
            (
                i, i, i, json.dumps({"doc": i // 100, "row": i}),
                json.dumps({"note": "M" * metadata_bytes}),
            )
            for i in range(1, rows + 1)
        ),
    )
    base = """
        FROM observations AS o
        LEFT JOIN units AS u ON u.observation_id = o.observation_id
        LEFT JOIN teaching_examples AS te ON te.observation_id = o.observation_id
        WHERE o.kind != 'teaching_example' OR te.status != 'deleted'
        ORDER BY CASE o.kind WHEN 'atomic' THEN 0 WHEN 'span' THEN 1 ELSE 2 END,
                 COALESCE(u.row_position, o.observation_id), o.observation_id
    """
    projections = {
        "baseline": """
            SELECT o.observation_id, o.kind, o.text, o.created_at,
                   u.unit_id, u.row_position, u.user_key_json, u.metadata_json,
                   te.observation_id AS teaching_code_id,
                   te.status AS teaching_status
        """ + base,
        "compact": """
            SELECT o.observation_id, o.kind, u.unit_id, u.user_key_json
        """ + base,
    }
    for label, query in projections.items():
        tracemalloc.start()
        start = time.perf_counter()
        selected = connection.execute(query).fetchall()
        decoded = []
        for row in selected:
            item = dict(row)
            item["user_key"] = json.loads(item["user_key_json"])
            if label == "baseline":
                item["metadata"] = json.loads(item["metadata_json"])
            decoded.append(item)
        elapsed = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        payload_bytes = sum(
            len(value.encode("utf-8"))
            for row in selected for value in row if isinstance(value, str)
        )
        print(
            f"{label}: rows={len(selected)} columns={len(selected[0])} "
            f"fetched_utf8_bytes={payload_bytes:,} peak_tracemalloc_bytes={peak:,} "
            f"seconds={elapsed:.3f} "
            f"json_decodes={len(selected) * (2 if label == 'baseline' else 1)}"
        )
    connection.close()


if __name__ == "__main__":
    run()
