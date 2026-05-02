"""Tests for ``ius.leadership``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ius import channel_names, granger_f_ratio, simulate_market


class TestKnownDgpRecovery:
    def test_strong_leader_yields_high_L(self, K, n_obs):
        rng = np.random.default_rng(0)
        df = simulate_market(alpha=+0.9, n=n_obs, K=K, rng=rng, beta=0.7)
        result = granger_f_ratio(df, "prob", channel_names(K), lag=1)
        assert result.L > 0.7, f"strong leader DGP gave L={result.L}"

    def test_strong_follower_yields_low_L(self, K, n_obs):
        rng = np.random.default_rng(0)
        df = simulate_market(alpha=-0.9, n=n_obs, K=K, rng=rng, beta=0.7)
        result = granger_f_ratio(df, "prob", channel_names(K), lag=1)
        assert result.L < 0.3, f"strong follower DGP gave L={result.L}"

    def test_independent_dgp_yields_intermediate_L(self, K, n_obs):
        rng = np.random.default_rng(0)
        df = simulate_market(alpha=0.0, n=n_obs, K=K, rng=rng)
        result = granger_f_ratio(df, "prob", channel_names(K), lag=1)
        # No strong tendency in either direction.
        assert 0.20 <= result.L <= 0.80, f"alpha=0 gave L={result.L}"


class TestEdgeCases:
    def test_returns_nan_when_below_min_n(self, rng, K):
        df = simulate_market(alpha=0.0, n=30, K=K, rng=rng)
        result = granger_f_ratio(df, "prob", channel_names(K), lag=1, min_n=50)
        assert np.isnan(result.L)
        assert result.n_after_lags < 50

    def test_constant_panel_returns_neutral_L(self, K):
        # All-constant input -> no variance -> denominator near zero -> L=0.5.
        n = 200
        df = pd.DataFrame({
            "prob": np.ones(n),
            **{c: np.zeros(n) for c in channel_names(K)},
        })
        result = granger_f_ratio(df, "prob", channel_names(K), lag=1, min_n=50)
        assert result.L == 0.5

    def test_min_n_threshold_can_be_relaxed(self, rng, K):
        # Same panel that fails at min_n=50 should succeed at min_n=20.
        df = simulate_market(alpha=0.5, n=40, K=K, rng=rng)
        strict = granger_f_ratio(df, "prob", channel_names(K), lag=1, min_n=50)
        relaxed = granger_f_ratio(df, "prob", channel_names(K), lag=1, min_n=20)
        assert np.isnan(strict.L)
        assert not np.isnan(relaxed.L)


class TestLagSensitivity:
    def test_higher_lag_runs_and_returns_finite_L(self, K, n_obs):
        rng = np.random.default_rng(0)
        df = simulate_market(alpha=0.5, n=n_obs, K=K, rng=rng)
        for lag in (1, 2, 3):
            result = granger_f_ratio(df, "prob", channel_names(K),
                                      lag=lag, min_n=50)
            assert 0.0 <= result.L <= 1.0
