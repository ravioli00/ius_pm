"""Synthetic data-generating processes for IUS validation.

A market is a vector autoregression with one prediction-market series ``P`` and
``K`` financial channels ``F_1, ..., F_K``. The direction parameter
:math:`\\alpha \\in [-1, 1]` controls the population causal regime:

* :math:`\\alpha > 0`: PM is a leader. ``F[j, t] = phi * F[j, t-1] + beta * alpha * P[t-1] + eps``
* :math:`\\alpha < 0`: PM is a follower. ``P[t]    = phi * P[t-1] + beta * |alpha| * F_avg[t-1] + eps``
* :math:`\\alpha = 0`: ``P`` and ``F`` evolve independently.

Three optional DGP modifications (used in the violated-DGP robustness study):

``"heavy_tail"``
    Student-t innovations (default ``df=3``).
``"regime_switch"``
    Cross-coupling sign flips at ``regime_flip_at`` fraction of the time series.
``"latent_conf"``
    A hidden AR(1) process drives both ``P`` and ``F`` with strength
    ``confounder_strength``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_market(
    alpha: float,
    n: int,
    K: int,
    rng: np.random.Generator,
    *,
    phi_p: float = 0.7,
    phi_f: float = 0.5,
    beta: float = 0.6,
    sigma_p: float = 1.0,
    sigma_f: float = 1.0,
    dgp: str = "gaussian",
    t_df: int = 3,
    regime_flip_at: float = 0.5,
    confounder_strength: float = 0.0,
) -> pd.DataFrame:
    """Generate one synthetic market.

    Returns a DataFrame with columns ``prob`` (the PM level), ``dprob`` (its
    first difference), and ``F0_logret``, ..., ``F{K-1}_logret`` (the financial
    channels).
    """
    if dgp not in {"gaussian", "heavy_tail", "regime_switch", "latent_conf"}:
        raise ValueError(f"unknown dgp={dgp!r}")

    P = np.zeros(n)
    F = np.zeros((n, K))
    flip_t = int(n * regime_flip_at)

    def draw_eps_p() -> float:
        if dgp == "heavy_tail":
            scale = sigma_p / np.sqrt(t_df / (t_df - 2))
            return float(rng.standard_t(t_df) * scale)
        return float(rng.normal(0.0, sigma_p))

    def draw_eps_f() -> np.ndarray:
        if dgp == "heavy_tail":
            scale = sigma_f / np.sqrt(t_df / (t_df - 2))
            return rng.standard_t(t_df, size=K) * scale
        return rng.normal(0.0, sigma_f, size=K)

    S = np.zeros(n)
    if dgp == "latent_conf":
        for t in range(1, n):
            S[t] = 0.6 * S[t - 1] + rng.normal()

    for t in range(1, n):
        b = -beta if (dgp == "regime_switch" and t >= flip_t) else beta
        eps_p = draw_eps_p()
        eps_f = draw_eps_f()
        s_term = confounder_strength * S[t] if dgp == "latent_conf" else 0.0
        if alpha >= 0:
            P[t] = phi_p * P[t - 1] + s_term + eps_p
            F[t] = phi_f * F[t - 1] + b * alpha * P[t - 1] + s_term + eps_f
        else:
            f_avg_lag = F[t - 1].mean()
            P[t] = phi_p * P[t - 1] + b * abs(alpha) * f_avg_lag + s_term + eps_p
            F[t] = phi_f * F[t - 1] + s_term + eps_f

    df = pd.DataFrame({"prob": P})
    df["dprob"] = df["prob"].diff()
    for j in range(K):
        df[f"F{j}_logret"] = F[:, j]
    return df


def channel_names(K: int) -> list[str]:
    """Default channel names used by the synthetic DGP."""
    return [f"F{j}_logret" for j in range(K)]
