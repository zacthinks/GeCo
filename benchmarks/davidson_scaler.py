"""Opt-in bounded Davidson numerical benchmark (not a CI timing gate).

Run: python benchmarks/davidson_scaler.py
"""
from __future__ import annotations

import time
from itertools import product

import numpy as np
from scipy import sparse

from geometric_coder.davidson_scaler import LinearDavidsonScaler, davidson_probabilities


def run(*, seed=19):
    rng = np.random.default_rng(seed)
    for dimensions, sparse_input in product((256, 768, 1536), (False, True)):
        n = 320
        m = 120
        X = rng.normal(size=(n, dimensions))
        if sparse_input:
            X[rng.uniform(size=X.shape) > .10] = 0
            matrix = sparse.csr_matrix(X)
        else:
            matrix = X
        pairs = rng.integers(0, n, size=(m, 2))
        pairs[:, 1] = (pairs[:, 0]+1+rng.integers(0,n-1,size=m))%n
        beta = rng.normal(size=dimensions) / np.sqrt(dimensions)
        prob = davidson_probabilities((X[pairs[:,0]]-X[pairs[:,1]]) @ beta, -.7)
        outcomes = np.array([rng.choice(3,p=p) for p in prob])
        model = LinearDavidsonScaler(posterior_method="auto")

        start = time.perf_counter()
        model.fit(matrix, pairs, outcomes)
        fit_s = time.perf_counter()-start

        start = time.perf_counter()
        model.build_posterior()
        posterior_s = time.perf_counter()-start

        start = time.perf_counter()
        scores = model.predict_scores(matrix)
        candidate_mean, candidate_covariance = model.pair_posterior_moments(
            matrix, pairs[:24]
        )
        position_var = model.posterior_position_variance(matrix)
        predict_s = time.perf_counter()-start

        assert (np.isfinite(scores).all() and np.isfinite(candidate_mean).all()
                and np.isfinite(candidate_covariance).all() and np.isfinite(position_var).all())
        print(
            f"d={dimensions} sparse={sparse_input} m={m} "
            f"mode={model.posterior_.mode} fitted={model.success_} "
            f"fit={fit_s:.3f}s posterior={posterior_s:.3f}s "
            f"predict={predict_s:.3f}s"
        )


if __name__ == "__main__":
    run()
