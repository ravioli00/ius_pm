"""End-to-end pipeline tests.

Run the synthetic-MC validation on a small panel and check that
:math:`\\rho(\\mathrm{ILS}, L)` recovers a strong positive value when the DGP
has known cross-coupling.
"""
from __future__ import annotations

import numpy as np
import pytest

from ius import (channel_names, estimate_ils, granger_f_ratio,
                 permutation_spearman, simulate_market)


def _run_recovery(n_markets: int, n_obs: int, K: int, beta: float,
                  base_seed: int, n_perm: int = 1_000):
    """Replicate the core synthetic validation loop."""
    rng = np.random.default_rng(base_seed)
    alphas = rng.uniform(-1.0, 1.0, n_markets)
    chs = channel_names(K)
    ils_vals: list[float] = []
    L_vals: list[float] = []
    for alpha in alphas:
        df = simulate_market(alpha=alpha, n=n_obs, K=K, rng=rng, beta=beta)
        ils = estimate_ils(df, "prob", "dprob", primary=chs)
        gr = granger_f_ratio(df, "prob", chs, lag=1)
        if ils.ils is not None and not np.isnan(gr.L):
            ils_vals.append(ils.ils)
            L_vals.append(gr.L)
    perm = permutation_spearman(np.array(ils_vals), np.array(L_vals),
                                 n_perm=n_perm,
                                 rng=np.random.default_rng(base_seed + 1))
    return perm, len(ils_vals)


def test_pipeline_recovers_positive_rho_at_realistic_n():
    perm, n_successful = _run_recovery(n_markets=20, n_obs=600, K=5,
                                        beta=0.6, base_seed=0)
    assert n_successful >= 18
    assert perm.rho_observed > 0.6, (
        f"recovery rho = {perm.rho_observed:.3f} below 0.6 at n_markets=20")
    assert perm.p_two_sided < 0.01


def test_pipeline_recovers_positive_rho_at_empirical_panel_size():
    perm, n_successful = _run_recovery(n_markets=16, n_obs=600, K=5,
                                        beta=0.6, base_seed=20260502)
    assert n_successful >= 14
    assert perm.rho_observed > 0.5, (
        f"recovery rho = {perm.rho_observed:.3f} below 0.5 at n_markets=16")
    assert perm.p_two_sided < 0.05


def test_pipeline_recovers_correct_sign_for_pure_leader_vs_follower_panels():
    """Sign recovery: a panel mixing leaders (alpha=+0.7) and followers (-0.7)
    should yield ILS values higher for leaders than followers.

    This is a stronger null check than the beta=0 test (which was conceptually
    flawed: ILS and L are correlated estimators on the same panels and
    co-fluctuate under sampling noise). Here we partition the panel into known
    leaders and known followers and check that the methodology recovers the
    rank distinction in both directions.
    """
    rng = np.random.default_rng(20260601)
    chs = channel_names(5)
    leader_ils, follower_ils = [], []
    leader_L, follower_L = [], []
    for _ in range(15):
        df_lead = simulate_market(alpha=+0.7, n=600, K=5, rng=rng, beta=0.6)
        df_follow = simulate_market(alpha=-0.7, n=600, K=5, rng=rng, beta=0.6)
        leader_ils.append(estimate_ils(df_lead, "prob", "dprob", primary=chs).ils)
        follower_ils.append(estimate_ils(df_follow, "prob", "dprob", primary=chs).ils)
        leader_L.append(granger_f_ratio(df_lead, "prob", chs, lag=1).L)
        follower_L.append(granger_f_ratio(df_follow, "prob", chs, lag=1).L)
    leader_ils = [v for v in leader_ils if v is not None]
    follower_ils = [v for v in follower_ils if v is not None]
    assert np.mean(leader_ils) > np.mean(follower_ils), (
        f"leader mean ILS {np.mean(leader_ils):.3f} should exceed "
        f"follower mean ILS {np.mean(follower_ils):.3f}"
    )
    assert np.mean(leader_L) > np.mean(follower_L), (
        f"leader mean L {np.mean(leader_L):.3f} should exceed "
        f"follower mean L {np.mean(follower_L):.3f}"
    )
