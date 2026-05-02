"""Tests for ``ius.simulate``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ius import channel_names, simulate_market


class TestSimulateShape:
    def test_returns_dataframe_with_expected_columns(self, rng, K, n_obs):
        df = simulate_market(alpha=0.5, n=n_obs, K=K, rng=rng)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (n_obs, K + 2)
        expected = ["prob", "dprob", *channel_names(K)]
        assert list(df.columns) == expected

    def test_dprob_is_first_difference_of_prob(self, rng, K, n_obs):
        df = simulate_market(alpha=0.0, n=n_obs, K=K, rng=rng)
        # dprob.iloc[0] is NaN (first diff of length-n series)
        np.testing.assert_array_equal(
            df["dprob"].iloc[1:].values,
            df["prob"].diff().iloc[1:].values,
        )

    def test_seed_determinism(self, K, n_obs):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        df1 = simulate_market(alpha=0.7, n=n_obs, K=K, rng=rng1)
        df2 = simulate_market(alpha=0.7, n=n_obs, K=K, rng=rng2)
        pd.testing.assert_frame_equal(df1, df2)


class TestSimulateDgpDirection:
    """The DGP must inject the cross-coupling we claim: alpha>0 -> PM leads."""

    def test_leader_dgp_F_correlates_with_lagged_P(self, rng, K, n_obs):
        df = simulate_market(alpha=0.9, n=n_obs, K=K, rng=rng, beta=0.6)
        # When PM leads, F[t] should respond to P[t-1].
        rho_lead, _ = stats.pearsonr(df["F0_logret"].iloc[1:],
                                     df["prob"].shift(1).iloc[1:])
        rho_lag, _ = stats.pearsonr(df["F0_logret"].shift(1).iloc[2:],
                                    df["prob"].iloc[2:])
        # Forward correlation (P[t-1] -> F[t]) must exceed reverse (F[t-1] -> P[t]).
        assert abs(rho_lead) > abs(rho_lag), (
            f"leader DGP failed direction: |corr(P[-1]->F)|={abs(rho_lead):.3f} "
            f"<= |corr(F[-1]->P)|={abs(rho_lag):.3f}"
        )

    def test_follower_dgp_P_correlates_with_lagged_F(self, rng, K, n_obs):
        df = simulate_market(alpha=-0.9, n=n_obs, K=K, rng=rng, beta=0.6)
        f_avg = df[channel_names(K)].mean(axis=1)
        rho_lead, _ = stats.pearsonr(df["prob"].iloc[1:],
                                     f_avg.shift(1).iloc[1:])
        rho_lag, _ = stats.pearsonr(df["prob"].shift(1).iloc[2:],
                                    f_avg.iloc[2:])
        assert abs(rho_lead) > abs(rho_lag)

    def test_independent_dgp_has_no_cross_coupling(self, rng, K):
        # Larger n to tighten the null bound on the cross-correlation.
        df = simulate_market(alpha=0.0, n=4000, K=K, rng=rng, beta=0.6)
        rho, _ = stats.pearsonr(df["F0_logret"].iloc[1:],
                                df["prob"].shift(1).iloc[1:])
        assert abs(rho) < 0.10, f"alpha=0 produced cross-correlation {rho:.3f}"


class TestDgpVariants:
    def test_heavy_tail_runs_and_has_heavier_tails_than_gaussian(self, rng, K, n_obs):
        df_g = simulate_market(alpha=0.0, n=4000, K=K,
                               rng=np.random.default_rng(0), dgp="gaussian")
        df_t = simulate_market(alpha=0.0, n=4000, K=K,
                               rng=np.random.default_rng(0), dgp="heavy_tail",
                               t_df=3)
        # Excess kurtosis of t_3 innovations is theoretically infinite; sample
        # kurtosis is large but finite. Compare against Gaussian baseline.
        kurt_g = float(stats.kurtosis(df_g["F0_logret"]))
        kurt_t = float(stats.kurtosis(df_t["F0_logret"]))
        assert kurt_t > kurt_g

    def test_regime_switch_produces_sign_change_in_cross_coupling(self, K, n_obs):
        df = simulate_market(alpha=0.7, n=2000, K=K,
                             rng=np.random.default_rng(0),
                             dgp="regime_switch", beta=0.8,
                             regime_flip_at=0.5)
        first_half = df.iloc[:1000]
        second_half = df.iloc[1000:].reset_index(drop=True)
        rho1, _ = stats.pearsonr(first_half["F0_logret"].iloc[1:],
                                 first_half["prob"].shift(1).iloc[1:])
        rho2, _ = stats.pearsonr(second_half["F0_logret"].iloc[1:],
                                 second_half["prob"].shift(1).iloc[1:])
        # The two halves should have opposite-sign cross-correlations.
        assert rho1 * rho2 < 0, f"regime-switch failed: rho1={rho1:.3f}, rho2={rho2:.3f}"

    def test_latent_confounder_increases_cross_correlation_at_alpha_zero(self, K):
        df_clean = simulate_market(alpha=0.0, n=4000, K=K,
                                   rng=np.random.default_rng(0))
        df_conf = simulate_market(alpha=0.0, n=4000, K=K,
                                  rng=np.random.default_rng(0),
                                  dgp="latent_conf",
                                  confounder_strength=1.5)
        rho_clean, _ = stats.pearsonr(df_clean["F0_logret"], df_clean["prob"])
        rho_conf, _ = stats.pearsonr(df_conf["F0_logret"], df_conf["prob"])
        assert abs(rho_conf) > abs(rho_clean)

    def test_unknown_dgp_raises(self, rng, K, n_obs):
        with pytest.raises(ValueError, match="unknown dgp"):
            simulate_market(alpha=0.0, n=n_obs, K=K, rng=rng, dgp="nonsense")
