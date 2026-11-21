# Tiered backend performance benchmarks

This is a measurement baseline, not a performance optimization or browser benchmark. The harness creates disposable deterministic projects in synthetic mode. Medium and stress workloads are manual.

## Tiers and commands

| Tier | Units | Invocation |
| --- | ---: | --- |
| A routine | 1,000; optional 10,000 | Fast pytest smoke in ordinary CI; manual timing |
| B diagnostic | ~50,000 | Manual before/after performance PRs |
| C stress | ~300,000 | Deliberate release or integration checks, never ordinary CI |

Run from the repository root in an installed environment:

- A: uv run python -m benchmarks.tiered_linked --tier a --repeats 20 --output reports/tier-a.json
- A 10k: uv run python -m benchmarks.tiered_linked --tier a --units 10000 --repeats 20 --output reports/tier-a-10k.json
- B: uv run python -m benchmarks.tiered_linked --tier b --repeats 20 --output reports/tier-b.json
- C: uv run python -m benchmarks.tiered_linked --tier c --repeats 20 --output reports/tier-c.json

Small local smoke: uv run pytest -q tests/test_tiered_benchmark.py. Override the fixture size with --units 40 --repeats 1 for rapid development. There is no CI latency threshold. JSON is printed to stdout and optionally saved with --output.

## Covered operations and correctness

The shared suite measures indexed SQLite user-key/text pages, a bounded context group, a categorical filtered SQL page, 2D view coordinates, and external topic-model page/one-row selector calls. The synthetic fixture uses composite document/sentence keys with documents capped at 16 units. A seeded page contains up to 64 distinct positions by default (custom --page-size 1–5000). Validation occurs outside timing and checks stable order, document identities, category counts, page sizes, view coordinates, and topic membership/ArgMax alignment.

Missing features receive explicit unavailable records. A real context request lacking a max_units API is skipped rather than risking a 300k-unit ancestor read.

## Report contract

The machine-readable report contains schema_version; environment (git SHA/ref where available, Python/platform/dependency versions); fixture (dataset identity, seed, units, fixture setup time excluded); execution (tier and operations); cold_process; and limitations. Each available operation records:

- first_call_proxy: first measured invocation after fixture setup, in the same process; NOT an isolated cold start.
- warm: individual samples, p50 wall milliseconds, p95 only with 20 or more warm samples (otherwise null and insufficient_samples), repetition count.
- python_allocation_peak_bytes: per-call tracemalloc peak above the start baseline. Tracing is reset after fixture setup and before validation; tracing overhead affects timings.
- process RSS, SQLite/provider disk I/O, network bytes, browser paint: explicitly not measured. Local serialized JSON result size, where available, is not actual Dash transport size. Topic provider packets have no comparable JSON payload here.

To compare revisions, match commit, Python packages, machine, CPU/thread limits, OS, tier, fixture version, seed, page size, repetitions, and dataset identity. Use multiple fresh processes/interleaved runs to check cold startup separately. A fresh process is necessary for cold start but does not itself make the first measured call cold, because imports and setup precede this measurement. Synthetic math providers do NOT benchmark TeAL DuckDB/artifact I/O.

## Illustrative report fragments (not actual timing measurements)

A: fixture.mode=synthetic_external_provider_no_teal_io, execution.tier=a, fixture.actual_units=1000, execution.page_size=64, execution.warm_repeats=5, execution.operations.sqlite_key_page.result.row_count=64, execution.operations.sqlite_key_page.warm.p95_wall_ms=null.

B: fixture.mode=synthetic_external_provider_no_teal_io, execution.tier=b, fixture.actual_units=50000, execution.warm_repeats=20. p50/p95 would be measured after running; no sample values are asserted here.

C: fixture.mode=synthetic_external_provider_no_teal_io, execution.tier=c, fixture.actual_units=300000, execution.warm_repeats=20. Not executed in default CI. All sample values must come from an actual run.

These fragments illustrate the expected tier identities; they are NOT benchmark results or invented performance numbers.

## Optional real TeAL-linked reads

Use only on a disposable snapshot/copy. The runner does not invoke explicit GeCo or TeAL writes, but a provider or TeAL Project.open may perform implicit version-specific migration/cache writes, so this is not filesystem-enforced read-only. Never use the only copy of research data.

uv run python -m benchmarks.tiered_linked --tier b --geco-path /path/to/snapshot.geco --teal-path /path/to/teal-snapshot --dataset-id snapshot-2026-11 --filter-field category --filter-value A --repeats 20 --output reports/teal-linked.json

TeAL imports are optional and only needed for this mode. When no view, topic model or safe context path is registered, unavailable is reported. With real links, provider calls may include DuckDB/artifact I/O, but disk bytes/SQL query counts are not directly instrumented.

## Existing specialized benchmarks

- Recommendation: benchmarks/phase1e_recommendations.py covers trained classifier/committee first/warm, visited and stale paths at 1k/10k; training and committee preparation are separate.
- Scale: benchmarks/candidate_sampling_300k.py covers bounded pair sampling and SQLite history; benchmarks/scale_eig.py and benchmarks/davidson_scaler.py cover scoring.
- Analyses: benchmarks/analyses_300k.py checks continuous charts, exact aggregate correctness, result bytes, and Plotly JSON payload limits.
- UI pilot: benchmarks/phase2a_runtime_pilot.py tests Dash HTTP/setup but not browser paint.

Future independent measurement lanes: isolated-process cold startup; real process RSS sampling; provider/SQLite I/O; Dash transport payloads; browser click-to-visible acceptance. Backend medians are not end-user UI latencies.
