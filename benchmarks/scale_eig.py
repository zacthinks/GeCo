"""Reproducible Phase 3B EIG timing: generation, scoring and memory.

Run: OPENBLAS_NUM_THREADS=2 python benchmarks/scale_eig.py
Timing excludes model fitting/posterior construction but includes an independent
Phase 3A sampler only after its interface is merged. This initial benchmark
times the supplied candidate arrays and acquisition scoring.
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np
from scipy import sparse

from geometric_coder.davidson_scaler import LinearDavidsonScaler
from geometric_coder.scale_acquisition import score_davidson_pairs


def benchmark(d: int, *, csr: bool, budget: int = 1000) -> None:
    rng = np.random.default_rng(141)
    n = 2500
    X = rng.normal(size=(n, d))
    if csr:
        X[rng.random((n, d)) > 0.04] = 0
        X = sparse.csr_matrix(X)
    train_pairs = np.stack((rng.integers(0, n, 96), rng.integers(0, n - 1, 96)), axis=1)
    train_pairs[:, 1] += (train_pairs[:, 1] >= train_pairs[:, 0])
    outcomes = rng.integers(0, 3, len(train_pairs))
    model = LinearDavidsonScaler(posterior_method="dual").fit(
        X, train_pairs, outcomes, reference_X=X
    ).build_posterior()
    t0 = time.perf_counter()
    # Synthetic candidate generation; NOT a benchmark of the Phase 3A sampler.
    candidates = rng.integers(0, n, size=(budget, 2))
    candidates[:, 1] = (candidates[:, 0] + 1 + candidates[:, 1]) % n
    candidate_generation = time.perf_counter() - t0

    tracemalloc.start()
    t0 = time.perf_counter()
    gains = score_davidson_pairs(model, X, candidates)
    scoring = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert np.isfinite(gains).all() and np.all(gains >= 0)
    print({
        "dimensions": d,
        "storage": "CSR" if csr else "dense",
        "candidates": budget,
        "candidate_generation_s_synthetic_only": round(candidate_generation, 4),
        "EIG_scoring_s": round(scoring, 4),
        "Python_tracemalloc_peak_MiB": round(peak / 2**20, 2),
        "database_persistence_retrieval_s": "not_benchmarked",
    })


if __name__ == "__main__":
    for dimension in (256, 768):
        for as_csr in (False, True):
            benchmark(dimension, csr=as_csr)
