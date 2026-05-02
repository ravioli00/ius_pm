"""Granger F-ratio leadership score.

The leadership score :math:`L \\in [0, 1]` summarises whether a
prediction-market series tends to lead its financial channels (``L > 0.5``)
or follow them (``L < 0.5``):

.. math::
    L = \\frac{\\mathrm{SS}_{P \\to F}}{\\mathrm{SS}_{P \\to F} + \\mathrm{SS}_{F \\to P}},

where :math:`\\mathrm{SS}_{X \\to Y}` is the incremental sum-of-squares from
adding the source variable to a lag-only baseline regression on :math:`Y`.

Under linear-Gaussian dependence, :math:`L` is a monotone transform of the
Granger F-ratio that the bootstrap-PCMCI+ leadership score also targets, with
the practical advantage of running in milliseconds on small windows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GrangerResult:
    """Output of :func:`granger_f_ratio`."""

    L: float
    ssr_p_to_f: float | None
    ssr_f_to_p: float | None
    n_after_lags: int


def granger_f_ratio(
    df: pd.DataFrame,
    prob_col: str,
    channels: list[str],
    *,
    lag: int = 1,
    min_n: int = 50,
) -> GrangerResult:
    """Compute the Granger F-ratio leadership score on a panel."""
    sub = df[[prob_col] + channels].dropna().copy()
    for lg in range(1, lag + 1):
        sub[f"{prob_col}_lag{lg}"] = sub[prob_col].shift(lg)
        for c in channels:
            sub[f"{c}_lag{lg}"] = sub[c].shift(lg)
    sub = sub.dropna()
    if len(sub) < min_n:
        return GrangerResult(L=float("nan"), ssr_p_to_f=None,
                             ssr_f_to_p=None, n_after_lags=int(len(sub)))

    p_lags = [f"{prob_col}_lag{lg}" for lg in range(1, lag + 1)]

    ssr_p_to_f = 0.0
    for c in channels:
        c_lags = [f"{c}_lag{lg}" for lg in range(1, lag + 1)]
        other_lags = [f"{cc}_lag{lg}" for cc in channels for lg in range(1, lag + 1)
                      if cc != c]
        baseline_cols = c_lags + other_lags
        full_cols = baseline_cols + p_lags
        y = sub[c].values
        X_b = sub[baseline_cols].values
        X_f = sub[full_cols].values
        coef_b, *_ = np.linalg.lstsq(X_b, y, rcond=None)
        coef_f, *_ = np.linalg.lstsq(X_f, y, rcond=None)
        ssr_b = float(((y - X_b @ coef_b) ** 2).sum())
        ssr_f = float(((y - X_f @ coef_f) ** 2).sum())
        ssr_p_to_f += max(ssr_b - ssr_f, 0.0)

    f_lags_all = [f"{c}_lag{lg}" for c in channels for lg in range(1, lag + 1)]
    y = sub[prob_col].values
    X_b = sub[p_lags].values
    X_f = sub[p_lags + f_lags_all].values
    coef_b, *_ = np.linalg.lstsq(X_b, y, rcond=None)
    coef_f, *_ = np.linalg.lstsq(X_f, y, rcond=None)
    ssr_b = float(((y - X_b @ coef_b) ** 2).sum())
    ssr_f = float(((y - X_f @ coef_f) ** 2).sum())
    ssr_f_to_p = max(ssr_b - ssr_f, 0.0)

    denom = ssr_p_to_f + ssr_f_to_p
    if denom < 1e-12:
        return GrangerResult(L=0.5, ssr_p_to_f=ssr_p_to_f,
                             ssr_f_to_p=ssr_f_to_p, n_after_lags=int(len(sub)))
    return GrangerResult(
        L=float(ssr_p_to_f / denom),
        ssr_p_to_f=float(ssr_p_to_f),
        ssr_f_to_p=float(ssr_f_to_p),
        n_after_lags=int(len(sub)),
    )
