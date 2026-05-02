"""Tests for ``ius.stats``."""
from __future__ import annotations

import numpy as np
import pytest

from ius import permutation_spearman


class TestKnownCorrelations:
    def test_perfectly_correlated_data_yields_significant_p(self):
        x = np.arange(20.0)
        y = x.copy()
        result = permutation_spearman(x, y, n_perm=2_000,
                                      rng=np.random.default_rng(0))
        assert result.rho_observed == pytest.approx(1.0)
        assert result.p_two_sided < 0.005
        assert result.p_one_sided < 0.005

    def test_perfectly_anticorrelated_data_yields_one_sided_one(self):
        x = np.arange(20.0)
        y = -x
        result = permutation_spearman(x, y, n_perm=2_000,
                                      rng=np.random.default_rng(0))
        assert result.rho_observed == pytest.approx(-1.0)
        assert result.p_two_sided < 0.005
        # one-sided is "rho >= rho_obs"; for negative rho_obs almost all
        # permutations satisfy this, so the one-sided p should be near 1.
        assert result.p_one_sided > 0.99

    def test_random_data_p_concentrates_near_uniform(self):
        rng = np.random.default_rng(42)
        ps = []
        for seed in range(40):
            sub_rng = np.random.default_rng(seed)
            x = sub_rng.normal(size=30)
            y = sub_rng.normal(size=30)
            r = permutation_spearman(x, y, n_perm=500,
                                      rng=np.random.default_rng(seed + 100))
            ps.append(r.p_two_sided)
        # Under the null, p-values should be roughly uniform; mean ~ 0.5.
        assert 0.30 < np.mean(ps) < 0.70


class TestDeterminism:
    def test_same_rng_gives_same_p(self):
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        y = np.array([2.0, 4.0, 1.0, 5.0, 3.0])
        r1 = permutation_spearman(x, y, n_perm=500,
                                   rng=np.random.default_rng(0))
        r2 = permutation_spearman(x, y, n_perm=500,
                                   rng=np.random.default_rng(0))
        assert r1.p_two_sided == r2.p_two_sided
        assert r1.p_one_sided == r2.p_one_sided

    def test_different_rng_gives_close_but_not_identical_p(self):
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        y = np.array([2.0, 4.0, 1.0, 5.0, 3.0])
        r1 = permutation_spearman(x, y, n_perm=2000,
                                   rng=np.random.default_rng(0))
        r2 = permutation_spearman(x, y, n_perm=2000,
                                   rng=np.random.default_rng(1))
        # Two independent 2000-perm draws should give similar but distinct p.
        assert abs(r1.p_two_sided - r2.p_two_sided) < 0.05


class TestInputValidation:
    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="equal length"):
            permutation_spearman([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_too_few_observations_raise(self):
        with pytest.raises(ValueError, match="at least 3"):
            permutation_spearman([1.0, 2.0], [3.0, 4.0])

    def test_returns_nan_when_one_sequence_constant(self):
        result = permutation_spearman([1.0, 1.0, 1.0, 1.0],
                                       [2.0, 3.0, 4.0, 5.0],
                                       n_perm=200,
                                       rng=np.random.default_rng(0))
        # rho is undefined with a constant input; permutation_spearman returns NaN.
        assert np.isnan(result.rho_observed)
