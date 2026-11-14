"""Practical 300k-observation synthetic Scale candidate benchmark.

Usage: uv run python benchmarks/candidate_sampling_300k.py --units 300000 --budget 1000
The synthetic project provides atomic IDs and an actual SQLite indexed judgment
table; it does not construct full text or geometry. Reports sampler runtime,
tracemalloc peak, and number of bounded pair-history SQL queries.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
import time
import tracemalloc
from array import array
from pathlib import Path

from geometric_coder.candidate_sampling import prepare_candidate_run


class SyntheticDatabase:
    def __init__(self, size: int, path: Path):
        self.size = size
        self.path = path
        self.pair_queries = 0
        with sqlite3.connect(path) as db:
            db.executescript("""
                CREATE TABLE scale_comparison_events(
                    scale_id INTEGER NOT NULL,
                    observation_low_id INTEGER NOT NULL,
                    observation_high_id INTEGER NOT NULL
                );
                CREATE INDEX comparisons_pair
                    ON scale_comparison_events(scale_id, observation_low_id, observation_high_id);
            """)
            # A sparse history, including pairs from an earlier definition.
            db.executemany(
                "INSERT INTO scale_comparison_events VALUES (1, ?, ?)",
                ((i, i + 1) for i in range(1, min(1001, size))),
            )

    def acquisition_eligible_atomic_ids(self, *args, **kwargs):
        return array("q", range(1, self.size + 1))

    def scale_judgment_revision(self, scale_id):
        return min(1000, self.size - 1), min(1000, self.size - 1)

    def scale_judged_candidate_pairs(self, scale_id, pairs):
        assert len(pairs) <= 256
        self.pair_queries += 1
        values = ", ".join("(?, ?)" for _ in pairs)
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                f"""WITH candidates(low, high) AS (VALUES {values})
                    SELECT c.low, c.high FROM candidates AS c
                    WHERE EXISTS (
                      SELECT 1 FROM scale_comparison_events AS e
                      WHERE e.scale_id = ? AND e.observation_low_id = c.low
                        AND e.observation_high_id = c.high
                    )""",
                (*[value for pair in pairs for value in pair], scale_id),
            ).fetchall()
        return {tuple(row) for row in rows}


class SyntheticProject:
    def __init__(self, database: SyntheticDatabase, directory: Path):
        self.database = database
        self.project_dir = directory

    def construct(self, construct_id):
        return {"type": "scale", "version_id": 1, "version_number": 1}

    def scale_eligibility(self, scale_id):
        return {"eligibility": {"kind": "all"}, "eligibility_revision": 1, "code_revisions": {}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=300_000)
    parser.add_argument("--budget", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.units < 0 or args.budget <= 0:
        parser.error("units must be nonnegative and benchmark budget must be positive")
    with tempfile.TemporaryDirectory() as tmp:
        project = SyntheticProject(
            SyntheticDatabase(args.units, Path(tmp) / "comparisons.sqlite"),
            Path(tmp),
        )
        tracemalloc.start()
        started = time.perf_counter()
        run = prepare_candidate_run(
            project, 1, budget=args.budget, seed=args.seed, batch_size=128,
        )
        for batch in run.batches():
            run.acknowledge_evaluated(batch)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(json.dumps({
            "observations": args.units,
            "possible_pairs": run.possible_task_count,
            "requested_budget": args.budget,
            "evaluated_candidates": run.evaluated_count,
            "seconds": round(elapsed, 4),
            "peak_tracemalloc_mib": round(peak / (1024 * 1024), 3),
            "pair_membership_sql_queries": project.database.pair_queries,
            "pairs_inspected": run.inspected_task_count,
            "exhausted": run.exhausted,
            "strategy": run.sampling_strategy,
        }, indent=2))


if __name__ == "__main__":
    main()
