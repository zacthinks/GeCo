# Bayesian Scale pair acquisition (Phase 3B draft)

This feature acquires one eligible, unjudged atomic-observation pair at a
time for the existing Develop Scale comparison interface. It is exploratory
research assistance, not a guarantee of measurement validity or an independent
probability-based audit sample.

## Mathematical acquisition

Given a trained linear Davidson scaler, each pair yields a joint Gaussian
Laplace approximation for (utility difference `delta`, tie parameter `eta`).
The three comparison logits are `[delta / 2, eta, -delta / 2]`.
For outcome `Y`, expected information gain in nats is

`H(E[p(Y | delta, eta)]) - E[H(p(Y | delta, eta))]`.

`scale_acquisition.three_outcome_information_gain` evaluates the expectation
with deterministic tensor 7×7 Gauss–Hermite quadrature (49 outcome probability
evaluations per candidate pair). Both dimensions and their covariance are
retained, and singular, highly correlated and near-zero covariance are handled
without inventing a full-dimensional random posterior sampler. EIG is not
predictive entropy, geometric distance, or proximity of estimated scores.

## Candidate sampling and ranking

The shared candidate-evaluation budget and pair sampler are owned by
`candidate_sampling.prepare_candidate_run` (Phase 3A). It supplies unique
canonical *stable observation IDs* in seeded bounded batches. Phase 3B maps
only those candidate IDs to geometry rows, evaluates the fitted scaler's
`pair_posterior_moments`, computes EIG, and saves a fit-specific ordered
ranking. Equal acquisition scores break ties by stable (low-ID, high-ID) pair.

A normal Scale candidate budget of 1,000 means at most 1,000 distinct sampled
eligible pair tasks evaluated by the acquisition algorithm, not 1,000
observations or 1,000 researcher responses. Exhaustive-mode limits are
enforced by Phase 3A before pair enumeration. No full O(N²) pair table is
persisted. The numerical core processes bounded chunks and never loads full
source text.

Schema 21 retains one lightweight current acquisition run and bounded ranked
pairs per Scale. The run records the source fit, definition version,
configuration, candidate provenance, evaluation count, scoring method, seed,
budget, and fallback diagnostics. A new accepted fit replaces the current
acquisition run transactionally.

Before meaningful evidence, with an unsupported geometry, bad posterior,
invalid configuration or numerical failure, Develop falls back to the seeded
shared random sampler and exposes its reason. The initial operational gate for
EIG is six usable comparisons, three directional comparisons and four distinct
observations, plus successful posterior construction. No parity judgments are
required; this is a conservative heuristic, not a model-identifiability proof.

## Training and frozen recommendations

- New **saved** human evidence may trigger the existing auto-training
  mechanism. On a newly accepted fit the acquisition ranking is rebuilt.
- With automatic training **off**, new judgments make the fit stale but the
  previous queue remains frozen. Retrieval skips already-judged or ineligible
  pairs without recomputing scores.
- Explicit Train accepts accumulated evidence and refreshes the queue through
  the same successful-fit hook.
- **Next pair** advances through the persisted frozen order using a
  session-local cursor, even if the currently displayed comparison is
  unanswered. This does not write a judgment, retrain, or re-rank the model.
  Reopening that session resumes its cursor; a new ranked run resets it.
  Judged or newly ineligible pairs are skipped. If no alternative pair remains,
  the UI says so rather than re-showing the same unanswered pair.
- Opening context, switching tabs, and viewing/correcting the current browser
  selection never train or re-rank the model.

The UI keeps a single comparison, four actions (A higher, approximately equal,
B higher, unsure/skip), and Next pair. Manual selection, comparison history and
correction remain available.

## Researcher acceptance walkthrough (after Phase 3A merge)

1. Open an existing copy of a real `.geco` project on the integrated
   `scaling` development branch and select a continuous Scale in Develop.
   Use an analytic geometry supported by the trained linear Davidson scaler.
2. With no saved Scale comparisons, click **Next pair** and confirm that one
   eligible, previously unjudged pair appears with random acquisition.
   Click Next pair twice without saving and confirm that each click
   advances to another pair while comparison history remains unchanged.
   Try all four judgment buttons and confirm the history records actual
   comparisons; skips must not become parity evidence.
3. Record at least six effective comparisons involving four observations and
   at least three directional outcomes; select **Train** (with optional
   hyperparameter tuning). Inspect acquisition status. If EIG is available,
   confirm Next pair chooses a high-ranked unjudged pair. If unavailable,
   inspect the reported fallback reason.
4. Turn **Automatically train** OFF; record another comparison. Confirm the
   fit becomes stale and the previously saved ranking remains unchanged.
   Next pair must skip the judged pair without fitting or re-ranking.
5. Reopen the project; confirm the same saved ranking is reused and normal
   comparison-history exclusion still operates. Turn automatic training ON
   again or explicitly **Train** and verify that a new successful fit refreshes
   the ranking.
6. Change the Scale definition or selected geometry and verify that the prior
   ranking is rejected rather than silently reused. Switch to a binary Code
   and confirm its original Develop recommendations are unaffected.
7. Repeat with a larger external-provider-backed project and compare dense
   and CSR feature sources. Confirm bounded source reads, identical identity
   alignment, and that Next pair remains responsive.

Do not use the benchmark results as real-project latency guarantees.

## Regression and benchmark commands

```bash
uv run pytest -q tests/test_scale_acquisition.py tests/test_scale_fit_integration.py tests/test_schema_policy.py
OPENBLAS_NUM_THREADS=2 uv run python benchmarks/scale_eig.py
```

The benchmark currently isolates synthetic candidate-array creation from
fit-independent EIG scoring and Python-traced peak memory for 256D and 768D,
dense and CSR cases. Full candidate sampler, SQLite persistence/retrieval,
provider I/O and real-project timing must be benchmarked after Phase 3A
integration. A complete exact-head CI pass is required before merge.

**Phase 3B is a draft until Phase 3A integration and real-project acceptance
are verified; do not merge independently.**
