"""Optional synthetic 50k-row context workload (stdlib SQLite only).

Run: python benchmarks/bounded_context_50k.py --units 50000 --cap 200
Compares the current unrestricted SELECT and bounded position-probe query shape.
This is a query-layout benchmark, not an end-to-end Dash browser benchmark.
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=50000)
    parser.add_argument("--cap", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.units < 1 or args.cap < 1 or args.repeats < 1:
        parser.error("units, cap and repeats must be positive")
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE units(
            unit_id INTEGER PRIMARY KEY, row_position INTEGER NOT NULL UNIQUE,
            text TEXT NOT NULL, user_key_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE unit_groups(
            unit_id INTEGER NOT NULL, level_index INTEGER NOT NULL,
            group_id INTEGER NOT NULL, PRIMARY KEY(unit_id, level_index)
        );
    """)
    con.executemany(
        "INSERT INTO units VALUES (?,?,?,?,?)",
        ((i + 1, i, "text " * 30,
          '{"doc":"same","sentence":%d}' % i, '{"category":"A"}')
         for i in range(args.units)),
    )
    con.executemany(
        "INSERT INTO unit_groups VALUES (?,0,1)",
        ((i + 1,) for i in range(args.units)),
    )
    focal = args.units // 2

    def unlimited():
        return con.execute("""
            SELECT DISTINCT u.* FROM units u
            JOIN unit_groups ug ON ug.unit_id = u.unit_id
            WHERE ug.level_index = ? AND ug.group_id IN (?)
            ORDER BY u.row_position
        """, (0, 1)).fetchall()

    def bounded():
        positions = []
        for comparison, direction in (("<=", "DESC"), (">", "ASC")):
            nearby = con.execute(f"""
                SELECT u.row_position FROM units AS u
                WHERE u.row_position {comparison} ?
                  AND EXISTS (
                      SELECT 1 FROM unit_groups AS ug
                      WHERE ug.unit_id = u.unit_id AND ug.level_index = ?
                        AND ug.group_id IN (?)
                  )
                ORDER BY u.row_position {direction} LIMIT ?
            """, (focal, 0, 1, args.cap)).fetchall()
            positions.extend(int(row[0]) for row in nearby)
        positions = sorted(sorted(
            positions, key=lambda pos: (abs(pos - focal), pos),
        )[:args.cap])
        rows = []
        for start in range(0, len(positions), 900):
            chunk = positions[start:start + 900]
            marks = ",".join("?" for _ in chunk)
            rows.extend(con.execute(
                f"SELECT * FROM units WHERE row_position IN ({marks}) "
                "ORDER BY row_position", chunk,
            ).fetchall())
        return rows

    for name, operation in (("unlimited", unlimited), ("bounded", bounded)):
        operation()  # warm the SQLite page cache
        elapsed = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            rows = operation()
            elapsed.append((time.perf_counter() - started) * 1000)
        byte_count = sum(
            len(str(value).encode()) for row in rows for value in row
        )
        print(
            f"{name}: rows={len(rows)} materialized_bytes={byte_count} "
            f"median_ms={statistics.median(elapsed):.3f} "
            f"min_ms={min(elapsed):.3f} max_ms={max(elapsed):.3f}"
        )


if __name__ == "__main__":
    main()
