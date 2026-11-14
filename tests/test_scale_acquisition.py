"""Bayesian Scale acquisition tests independent of database candidate selection."""
import numpy as np
import pytest
from geometric_coder.scale_acquisition import score_davidson_pairs, three_outcome_information_gain

def test_entropy_is_not_information_gain():
    mu = np.zeros((2, 2))
    covariance = np.array([np.zeros((2, 2)), np.diag([8., 1.])])
    gains, entropies = three_outcome_information_gain(
        mu, covariance, return_predictive_entropy=True
    )
    assert entropies[0] > 0.9
    assert gains[0] < 1e-12
    assert gains[1] > 0.1
    assert gains[1] < entropies[1]

def test_joint_covariance_and_swapping():
    mu = np.array([[0.7, -0.8]])
    base = np.array([[[1.5, 0.65], [0.65, 0.8]]])
    uncoupled = base.copy()
    uncoupled[:, 0, 1] = uncoupled[:, 1, 0] = 0
    gain = three_outcome_information_gain(mu, base)[0]
    assert abs(gain - three_outcome_information_gain(mu, uncoupled)[0]) > 1e-4
    swapped = mu * np.array([-1., 1.])
    cross_swapped = base.copy()
    cross_swapped[:, 0, 1] *= -1
    cross_swapped[:, 1, 0] *= -1
    assert np.isclose(gain, three_outcome_information_gain(swapped, cross_swapped)[0])

def test_near_singular_and_chunk_independence():
    mu = np.array([[1.0, 2.0], [0., -2.0], [500., -700.]])
    covariance = np.array([
        [[1., 1.], [1., 1.]],
        [[1e-18, 0.], [0., 1e-18]],
        [[2., 0.], [0., 2.]],
    ])
    big = three_outcome_information_gain(mu, covariance, chunk_size=20)
    small = three_outcome_information_gain(mu, covariance, chunk_size=1)
    assert np.allclose(big, small, atol=1e-12)
    assert np.all(big >= 0)
    assert np.isfinite(big).all()
    assert big[1] < 1e-10

def test_bad_posterior_rejected():
    with pytest.raises(ArithmeticError, match="positive semidefinite"):
        three_outcome_information_gain(
            np.zeros((1, 2)), np.array([[[-1., 0.], [0., 1.]]])
        )
    with pytest.raises(ArithmeticError, match="non-finite"):
        three_outcome_information_gain(
            np.array([[np.nan, 0.]]), np.zeros((1, 2, 2))
        )

def test_score_adapter_uses_actual_joint_pair_api():
    class Model:
        def __init__(self):
            self.calls = 0
        def pair_posterior_moments(self, X, pairs):
            self.calls += 1
            differences = X[pairs[:, 0], 0] - X[pairs[:, 1], 0]
            return np.column_stack((differences, np.zeros(len(pairs)))), np.repeat(
                np.diag([1., .5])[None, :, :], len(pairs), axis=0
            )
    model = Model()
    X = np.array([[0.], [1.], [2.], [3.]])
    pairs = np.array([[0, 1], [0, 2], [1, 3], [0, 3]])
    result = score_davidson_pairs(model, X, pairs, chunk_size=2)
    assert len(result) == 4 and np.isfinite(result).all()
    assert model.calls == 2

def test_actual_davidson_estimator_repeated_noisy_labels():
    from geometric_coder.davidson_scaler import LinearDavidsonScaler
    X = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    pair = np.array([[0, 1]])
    repeated = np.repeat(pair, 80, axis=0)
    noisy = np.array([0, 2] * 40)
    early = LinearDavidsonScaler().fit(
        X, repeated[:4], noisy[:4], reference_X=X
    ).build_posterior()
    late = LinearDavidsonScaler().fit(
        X, repeated, noisy, reference_X=X
    ).build_posterior()
    gain_early = score_davidson_pairs(early, X, pair)[0]
    gain_late = score_davidson_pairs(late, X, pair)[0]
    assert gain_late < gain_early
    assert late.predict_comparisons(X, pair)[0, 0] == pytest.approx(
        late.predict_comparisons(X, pair)[0, 2], abs=1e-8
    )
    assert gain_late >= 0

def test_actual_davidson_estimator_transitive_and_orthogonal():
    from geometric_coder.davidson_scaler import LinearDavidsonScaler
    # A, B, C vary along x; D introduces a previously unobserved y direction.
    X = np.array([[2., 0.], [1., 0.], [0., 0.], [0., 2.]])
    pairs = np.repeat(np.array([[0, 1], [1, 2]]), 20, axis=0)
    y = np.zeros(len(pairs), dtype=int)
    model = LinearDavidsonScaler().fit(
        X, pairs, y, reference_X=X
    ).build_posterior()
    # Evidence about A>B and B>C informs A>C without a direct A-C label.
    informed = model.pair_posterior_moments(X, [[0, 2]])[1][0, 0, 0]
    unexplored = model.pair_posterior_moments(X, [[2, 3]])[1][0, 0, 0]
    assert informed < unexplored
    gains = score_davidson_pairs(model, X, np.array([[0, 2], [2, 3]]))
    assert gains[1] > gains[0]

def test_dense_sparse_eig_agreement():
    from scipy import sparse
    from geometric_coder.davidson_scaler import LinearDavidsonScaler
    rng = np.random.default_rng(13)
    X = rng.normal(size=(25, 6))
    X[np.abs(X) < 0.8] = 0
    pairs = np.stack((rng.integers(0, 24, 40), rng.integers(1, 25, 40)), axis=1)
    pairs[:, 1] = (pairs[:, 0] + 1 + pairs[:, 1]) % 25
    outcome = rng.integers(0, 3, len(pairs))
    a = LinearDavidsonScaler().fit(X, pairs, outcome, reference_X=X).build_posterior()
    b = LinearDavidsonScaler().fit(
        sparse.csr_matrix(X), pairs, outcome, reference_X=sparse.csr_matrix(X)
    ).build_posterior()
    candidates = pairs[:10]
    assert np.allclose(
        score_davidson_pairs(a, X, candidates),
        score_davidson_pairs(b, sparse.csr_matrix(X), candidates),
        atol=1e-6
    )
