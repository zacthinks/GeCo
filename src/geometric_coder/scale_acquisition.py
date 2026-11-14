"""Three-outcome Bayesian information gain for fitted Davidson Scales.

This module only scores supplied row-index pairs. Candidate sampling, budgets,
persistence and Develop navigation belong to the application layer.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp, xlogy


def three_outcome_information_gain(
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    quadrature_order: int = 7,
    chunk_size: int = 256,
    return_predictive_entropy: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """I(Y; delta, eta) under joint Gaussian Laplace pair moments, in nats.

    ``means`` is (n, 2) for (score difference, log tie propensity) and
    ``covariances`` is (n, 2, 2), INCLUDING their posterior covariance.
    Integrates the three Davidson logits [delta/2, eta, -delta/2] with
    deterministic tensor Gauss-Hermite quadrature; never uses predictive
    entropy alone as an acquisition proxy.
    """
    mu = np.asarray(means, dtype=np.float64)
    cov = np.asarray(covariances, dtype=np.float64)
    if mu.ndim != 2 or mu.shape[1] != 2:
        raise ValueError("means must have shape (n, 2)")
    if cov.shape != (len(mu), 2, 2):
        raise ValueError("covariances must have shape (n, 2, 2)")
    if not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ArithmeticError("non-finite Laplace pair moments")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not isinstance(quadrature_order, int) or not 2 <= quadrature_order <= 32:
        raise ValueError("quadrature_order must be an integer from 2 to 32")

    nodes, weights_1d = hermgauss(quadrature_order)
    grid_a, grid_b = np.meshgrid(nodes, nodes, indexing="ij")
    standard_nodes = np.sqrt(2.0) * np.column_stack((grid_a.ravel(), grid_b.ravel()))
    weights = np.outer(weights_1d, weights_1d).ravel() / np.pi
    weights /= weights.sum()

    eig = np.empty(len(mu), dtype=float)
    predictive = np.empty(len(mu), dtype=float)
    for start in range(0, len(mu), chunk_size):
        stop = min(start + chunk_size, len(mu))
        block_cov = cov[start:stop]
        # Cholesky needs a strictly positive-definite matrix. Symmetric
        # eigendecomposition also handles rank-one and near-zero posteriors.
        asymmetry = np.max(np.abs(block_cov - block_cov.swapaxes(-1, -2)), axis=(1, 2))
        scale = np.maximum(1.0, np.max(np.abs(block_cov), axis=(1, 2)))
        if np.any(asymmetry > 1e-10 * scale):
            raise ArithmeticError("Laplace pair covariance is not symmetric")
        eigenvalues, eigenvectors = np.linalg.eigh(
            (block_cov + block_cov.swapaxes(-1, -2)) / 2.0
        )
        if np.any(eigenvalues[:, 0] < -1e-10 * scale):
            raise ArithmeticError("Laplace pair covariance is not positive semidefinite")
        transform = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[:, None, :]
        sampled = mu[start:stop, None, :] + np.einsum(
            "bij,kj->bki", transform, standard_nodes, optimize=True
        )
        if not np.isfinite(sampled).all():
            raise ArithmeticError("non-finite Davidson quadrature nodes")
        logits = np.stack(
            (0.5 * sampled[..., 0], sampled[..., 1], -0.5 * sampled[..., 0]),
            axis=-1,
        )
        log_probs = logits - logsumexp(logits, axis=-1, keepdims=True)
        probabilities = np.exp(log_probs)
        average = np.einsum("bkc,k->bc", probabilities, weights, optimize=True)
        pred_entropy = -np.sum(xlogy(average, average), axis=-1)
        conditional_entropy = -np.einsum(
            "bk,k->b",
            np.sum(xlogy(probabilities, probabilities), axis=-1),
            weights,
            optimize=True,
        )
        gain = pred_entropy - conditional_entropy
        if not np.isfinite(gain).all() or np.any(gain < -1e-10):
            raise ArithmeticError("invalid negative or non-finite expected information gain")
        eig[start:stop] = np.maximum(gain, 0.0)
        predictive[start:stop] = pred_entropy
    if return_predictive_entropy:
        return eig, predictive
    return eig


def score_davidson_pairs(
    model: object,
    features: np.ndarray,
    candidate_pairs: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Batch production-estimator pair moments for a supplied candidate pool.

    ``candidate_pairs`` indexes rows of ``features`` (not observation IDs).
    Reuses the fitted, joint Davidson Laplace posterior without refitting.
    """
    pairs = np.asarray(candidate_pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("candidate_pairs must be an integer (n, 2) array")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    gains = np.empty(len(pairs), dtype=float)
    for start in range(0, len(pairs), chunk_size):
        stop = min(start + chunk_size, len(pairs))
        # Restrict geometry to the distinct rows used by this chunk. The
        # production estimator scales every row of its X argument, so passing
        # a full 300k-row matrix here would repeat unnecessary transformations.
        indices, local_pairs = np.unique(
            pairs[start:stop].ravel(), return_inverse=True
        )
        if (np.any(indices < 0) or np.any(indices >= features.shape[0])
                or np.any(pairs[start:stop, 0] == pairs[start:stop, 1])):
            raise ValueError("candidate pairs contain invalid row indices")
        local_features = features[indices]
        means, covariance = model.pair_posterior_moments(
            local_features, local_pairs.reshape(-1, 2)
        )
        gains[start:stop] = three_outcome_information_gain(
            means, covariance, chunk_size=chunk_size
        )
    return gains
