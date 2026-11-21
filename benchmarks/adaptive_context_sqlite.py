"""Opt-in SQLite query-plan and adaptive context microbenchmark.

Run: python benchmarks/adaptive_context_sqlite.py --units 300000 --samples 20
SQLite-only (not browser or full GeCo API latency); outputs JSON for reproducibility.
No timing thresholds or assertions based on wall time.
"""
from __future__ import annotations

import argparse
import json
import resource
import sqlite3
import statistics
import time
import tracemalloc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=300000)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--cap", type=int, default=200)
    parser.add_argument("--output", type=str, default="-")
    args = parser.parse_args()
    if args.units < 60000 or args.samples < 20 or args.cap < 1:
        parser.error("units >= 60000, samples >= 20 and cap >= 1 are required")

    connection = sqlite3.connect(":memory:")
    connection.executescript("""
        CREATE TABLE units (
            unit_id INTEGER PRIMARY KEY, row_position INTEGER NOT NULL UNIQUE,
            text TEXT, user_key_json TEXT, metadata_json TEXT
        );
        CREATE TABLE unit_groups (
            unit_id INTEGER NOT NULL, level_index INTEGER NOT NULL,
            group_id INTEGER NOT NULL, PRIMARY KEY (unit_id, level_index)
        );
    """)
    sizes = [8, 16, 64, 200, 201, 1000, 50000]
    groups: dict[str, tuple[int, int]] = {}
    positions: list[tuple[int, int, str, str, str]] = []
    members: list[tuple[int, int, int]] = []
    next_group = 1
    position = 0
    # Varied groups first, then 16-member documents for the rest of the corpus.
    while position < args.units:
        size = min(sizes.pop(0) if sizes else 16, args.units - position)
        if size in {8, 16, 64, 200, 201, 1000, 50000} and str(size) not in groups:
            groups[str(size)] = (next_group, position + size // 2)
        for offset in range(size):
            pos = position + offset
            positions.append((
                pos + 1, pos, f"Sentence {pos} " + "text " * 10,
                json.dumps({"document": next_group, "sentence": offset}),
                '{"category":"A"}',
            ))
            members.append((pos + 1, 0, next_group))
        if len(positions) >= 10000:
            connection.executemany("INSERT INTO units VALUES (?,?,?,?,?)", positions)
            connection.executemany("INSERT INTO unit_groups VALUES (?,?,?)", members)
            positions.clear()
            members.clear()
        position += size
        next_group += 1
    if positions:
        connection.executemany("INSERT INTO units VALUES (?,?,?,?,?)", positions)
        connection.executemany("INSERT INTO unit_groups VALUES (?,?,?)", members)
    connection.execute(
        "CREATE INDEX unit_groups_group_unit_idx "
        "ON unit_groups(level_index, group_id, unit_id)"
    )

    def nearest(group_id: int, focal: int, *, adaptive: bool) -> list[int]:
        use_group = False
        if adaptive:
            threshold = 8 * args.cap
            probe = connection.execute(
                "SELECT 1 FROM unit_groups WHERE level_index=0 AND group_id=? LIMIT ?",
                (group_id, threshold + 1),
            ).fetchall()
            use_group = len(probe) <= threshold
        candidates: list[int] = []
        for comparison, direction in (("<=", "DESC"), (">", "ASC")):
            if use_group:
                sql = (
                    "SELECT u.row_position FROM unit_groups ug "
                    "JOIN units u ON u.unit_id=ug.unit_id "
                    f"WHERE ug.level_index=0 AND ug.group_id=? AND u.row_position {comparison} ? "
                    f"ORDER BY u.row_position {direction} LIMIT ?"
                )
                params = (group_id, focal, args.cap)
            else:
                sql = (
                    "SELECT u.row_position FROM units u "
                    f"WHERE u.row_position {comparison} ? AND EXISTS ("
                    "SELECT 1 FROM unit_groups ug WHERE ug.unit_id=u.unit_id "
                    "AND ug.level_index=0 AND ug.group_id=?) "
                    f"ORDER BY u.row_position {direction} LIMIT ?"
                )
                params = (focal, group_id, args.cap)
            candidates.extend(int(row[0]) for row in connection.execute(sql, params))
        return sorted(
            sorted(candidates, key=lambda p: (abs(p - focal), p))[:args.cap]
        )

    def measurement(operation: object) -> dict[str, float]:
        assert callable(operation)
        operation()
        elapsed = []
        for _ in range(args.samples):
            begin = time.perf_counter_ns()
            operation()
            elapsed.append((time.perf_counter_ns() - begin) / 1e6)
        elapsed.sort()
        return {
            "p50_ms": statistics.median(elapsed),
            "p95_ms": elapsed[round((len(elapsed) - 1) * 0.95)],
        }

    results: dict[str, object] = {
        "units": args.units,
        "samples": args.samples,
        "cap": args.cap,
        "sqlite_version": sqlite3.sqlite_version,
        "cases": {},
    }
    cases: dict[str, object] = {}
    for size, (group_id, focal) in groups.items():
        expected = nearest(group_id, focal, adaptive=False)
        assert nearest(group_id, focal, adaptive=True) == expected
        cases[size] = {
            "same_positions": True,
            "old_position_probe": measurement(
                lambda: nearest(group_id, focal, adaptive=False)
            ),
            "adaptive": measurement(lambda: nearest(group_id, focal, adaptive=True)),
            "adaptive_control": measurement(
                lambda: nearest(group_id, focal, adaptive=True)
            ),
        }
    results["cases"] = cases

    statements = {
        "membership_probe": (
            "SELECT 1 FROM unit_groups ug WHERE ug.level_index=0 "
            "AND ug.group_id=? LIMIT ?", (groups["16"][0], 1601),
        ),
        "position_probe": (
            "SELECT u.row_position FROM units u WHERE u.row_position<=? "
            "AND EXISTS (SELECT 1 FROM unit_groups ug WHERE ug.unit_id=u.unit_id "
            "AND ug.level_index=0 AND ug.group_id=?) "
            "ORDER BY u.row_position DESC LIMIT ?", (
                groups["16"][1], groups["16"][0], args.cap,
            ),
        ),
        "group_neighbor": (
            "SELECT u.row_position FROM unit_groups ug JOIN units u "
            "ON u.unit_id=ug.unit_id WHERE ug.level_index=0 "
            "AND ug.group_id=? AND u.row_position<=? "
            "ORDER BY u.row_position DESC LIMIT ?", (
                groups["16"][0], groups["16"][1], args.cap,
            ),
        ),
    }
    results["query_plans"] = {
        label: [list(row) for row in connection.execute(
            "EXPLAIN QUERY PLAN " + sql, params
        )]
        for label, (sql, params) in statements.items()
    }
    tracemalloc.start()
    nearest(groups["16"][0], groups["16"][1], adaptive=True)
    results["python_tracemalloc_peak_bytes"] = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # Peak RSS is process-wide; never interpret as the context query's memory use.
    results["process_peak_rss_bytes_linux"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    content = json.dumps(results, indent=2) + "\n"
    if args.output == "-":
        print(content, end="")
    else:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(content)


if __name__ == "__main__":
    main()
