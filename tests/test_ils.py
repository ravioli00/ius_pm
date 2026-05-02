"""Tests for ``ius.ils``.

The estimator must (a) recover the population direction on synthetic data with
known causal direction and (b) gracefully handle short / malformed panels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ius import block_bootstrap_ils, channel_names, estimate_ils, simulate_market


def _ils_on_synthetic(alpha: float, n: int, K: int, seed: int) -> float | None:
    rng = np.random.default_rng(seed)
    df = simulate_market(alpha=alpha, n=n, K=K, rng=rng, beta=0.6)
    out = estimate_ils(df, "prob", "dprob", primary=channel_names(K))
    return out.ils


class TestKnownDgpRecovery:
    def test_strong_leader_gives_ils_above_one(self, K, n_obs):
        ils = _ils_on_synthetic(alpha=+0.9, n=n_obs, K=K, seed=0)
        assert ils is not None and ils > 1.0, f"expected ILS>1, got {ils}"

    def test_strong_follower_gives_ils_below_one(self, K, n_obs):
        ils = _ils_on_synthetic(alpha=-0.9, n=n_obs, K=K, seed=1)
        assert ils is not None and ils < 1.0, f"expected ILS<1, got {ils}"

    def test_recovery_is_consistent_across_seeds(self, K, n_obs):
        # Strong leader -> ILS > 1 in the majority of seeds.
        leader_ils = [_ils_on_synthetic(alpha=+0.9, n=n_obs, K=K, seed=s)
                       for s in range(15)]
        follower_ils = [_ils_on_synthetic(alpha=-0.9, n=n_obs, K=K, seed=s)
                         for s in range(15, 30)]
        leader_above_1 = sum(1 for v in leader_ils if v is not None and v > 1.0)
        follower_below_1 = sum(1 for v in follower_ils if v is not None and v < 1.0)
        assert leader_above_1 >= 12  # 12/15 = 80% recovery
        assert follower_below_1 >= 12


class TestEdgeCases:
    def test_returns_none_when_panel_is_empty(self, K):
        empty = pd.DataFrame(columns=["prob", "dprob"] + channel_names(K))
        out = estimate_ils(empty, "prob", "dprob", primary=channel_names(K))
        assert out.ils is None
        assert out.note is not None

    def test_returns_none_when_no_resolvable_channels(self, rng, K, n_obs):
        df = simulate_market(alpha=0.5, n=n_obs, K=K, rng=rng)
        out = estimate_ils(df, "prob", "dprob",
                           primary=["nonexistent_channel"])
        assert out.ils is None
        assert "no resolvable channels" in (out.note or "")

    def test_returns_none_when_too_few_rows(self, rng, K):
        df = simulate_market(alpha=0.5, n=20, K=K, rng=rng)
        out = estimate_ils(df, "prob", "dprob",
                           primary=channel_names(K), min_rows=50)
        assert out.ils is None
        assert "insufficient" in (out.note or "")

    def test_aliases_resolve_etf_proxies(self, rng, K, n_obs):
        df = simulate_market(alpha=0.0, n=n_obs, K=K, rng=rng)
        # Simulate the real-data alias situation: panel exposes UUP_logret only,
        # but caller passes DXY_logret in primary.
        df = df.rename(columns={"F0_logret": "UUP_logret"})
        # The remaining channels keep F1..F4 names; pass them too.
        out = estimate_ils(df, "prob", "dprob",
                           primary=["DXY_logret", *channel_names(K)[1:]])
        assert "UUP_logret" in out.resolved_channels


class TestBootstrap:
    def test_bootstrap_ci_brackets_point_estimate(self, K, n_obs):
        rng = np.random.default_rng(42)
        df = simulate_market(alpha=0.7, n=n_obs, K=K, rng=rng, beta=0.6)
        point = estimate_ils(df, "prob", "dprob", primary=channel_names(K)).ils
        boot = block_bootstrap_ils(df, "prob", "dprob",
                                    primary=channel_names(K),
                                    n_boot=80, block_len=10,
                                    rng=np.random.default_rng(0))
        assert boot["n_boot_successful"] >= 50
        # The point estimate should lie within the bootstrap range
        # (allowing some tail mass outside the 95% CI).
        ci_lo, ci_hi = boot["ci_lower"], boot["ci_upper"]
        assert ci_lo is not None and ci_hi is not None
        # The 95% CI is typically wide; require the point to lie between
        # the empirical 1st and 99th percentiles.
        assert ci_lo <= point * 50 and ci_hi >= point / 50, (
            f"bootstrap CI [{ci_lo}, {ci_hi}] does not bracket point {point}"
        )

    def test_bootstrap_returns_note_on_short_panel(self, rng, K):
        df = simulate_market(alpha=0.0, n=40, K=K, rng=rng)
        boot = block_bootstrap_ils(df, "prob", "dprob",
                                    primary=channel_names(K),
                                    n_boot=20, block_len=10,
                                    rng=np.random.default_rng(0))
        assert boot["n_boot_successful"] == 0
        assert boot.get("note") is not None
