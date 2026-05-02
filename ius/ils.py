"""Information Leadership Score (ILS) estimator.

The ILS is the linear-Gaussian operational proxy for the Information Uniqueness
Score of Sayahpour (2026). For a prediction-market series :math:`P_t` and
financial channels :math:`\\mathbf{F}_t`,

.. math::
    \\mathrm{ILS}(P, \\mathbf{F}) = \\frac{\\overline{\\Delta R^2_{P \\to F^j}}}
                                          {\\Delta R^2_{\\mathbf{F} \\to P}}.

The numerator averages the incremental :math:`R^2` from adding :math:`P_t` to a
regression of :math:`\\Delta F^j_{t+h}` on :math:`\\mathbf{F}^{-j}_t`. The
denominator is the incremental :math:`R^2` from adding :math:`\\mathbf{F}_t` to
a regression of :math:`\\Delta P_{t+h}` on :math:`P_t`.

ILS > 1 marks a net information leader; ILS < 1 a net follower.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# Channel-name aliases used when raw panels expose ETF proxies for macro variables.
DEFAULT_ALIASES: dict[str, str] = {
    "DXY_logret": "UUP_logret",
    "Gold_logret": "GLD_logret",
    "VIX_logchg": "UVXY_logchg",
    "Oil_logret": "USO_logret",
}


@dataclass
class IlsEstimate:
    """Output of :func:`estimate_ils`."""

    ils: float | None
    delta_r2_f_to_p: float | None
    avg_delta_r2_p_to_f: float | None
    n: int
    resolved_channels: list[str]
    note: str | None = None


def _resolve_channel(name: str, panel: pd.DataFrame,
                     aliases: dict[str, str]) -> str | None:
    if name in panel.columns:
        return name
    alt = aliases.get(name)
    if alt and alt in panel.columns:
        return alt
    return None


def estimate_ils(
    panel: pd.DataFrame,
    prob_col: str,
    dprob_col: str,
    primary: list[str],
    *,
    secondary: list[str] | None = None,
    aliases: dict[str, str] | None = None,
    min_rows: int = 50,
    floor: float = 1e-8,
) -> IlsEstimate:
    """Estimate the ILS on one panel.

    Parameters
    ----------
    panel
        Hourly panel with ``prob_col``, ``dprob_col``, and the financial-channel
        columns. The function does not mutate the input.
    prob_col, dprob_col
        Names of the prediction-market level and first-difference columns.
    primary
        Required financial channels. Panel rows missing these after the
        regression's lag-1 setup are dropped. Aliases (see ``DEFAULT_ALIASES``)
        are applied to handle ETF proxies.
    secondary
        Optional fallback channels appended after primary.
    min_rows
        Minimum row count after dropna for the estimator to return a value.
    floor
        Lower bound on the denominator to avoid divide-by-zero blowups.

    Returns
    -------
    IlsEstimate
        ``ils=None`` when the estimator cannot be computed (insufficient rows or
        no resolvable channels); ``note`` records the reason.
    """
    aliases = aliases if aliases is not None else DEFAULT_ALIASES
    resolved: list[str] = []
    for c in primary:
        rc = _resolve_channel(c, panel, aliases)
        if rc and rc not in resolved:
            resolved.append(rc)
    for c in secondary or []:
        rc = _resolve_channel(c, panel, aliases)
        if rc and rc not in resolved:
            resolved.append(rc)
    if not resolved:
        return IlsEstimate(None, None, None, 0, [], note="no resolvable channels")

    sub = panel[[prob_col, dprob_col] + resolved].dropna().copy()
    if len(sub) < min_rows:
        return IlsEstimate(None, None, None, len(sub), resolved,
                           note=f"insufficient rows after dropna (got {len(sub)}, need {min_rows})")

    sub["prob_lag1"] = sub[prob_col].shift(1)
    sub = sub.dropna()
    if len(sub) < min_rows:
        return IlsEstimate(None, None, None, len(sub), resolved,
                           note="insufficient rows after lag")

    y_p = sub[dprob_col].values
    X_base_p = sub[["prob_lag1"]].values
    X_full_p = np.hstack([X_base_p, sub[resolved].values])

    r2_base_p = LinearRegression().fit(X_base_p, y_p).score(X_base_p, y_p)
    r2_full_p = LinearRegression().fit(X_full_p, y_p).score(X_full_p, y_p)
    delta_r2_f_to_p = max(r2_full_p - r2_base_p, 0.0)

    delta_p_to_fs: list[float] = []
    for c in resolved:
        others = [other for other in resolved if other != c]
        sub_j = sub[[c, prob_col] + others].copy()
        sub_j["prob_lag1"] = sub_j[prob_col].shift(1)
        if others:
            for other in others:
                sub_j[f"{other}_lag1"] = sub_j[other].shift(1)
            baseline_cols = [f"{other}_lag1" for other in others]
        else:
            sub_j[f"{c}_lag1"] = sub_j[c].shift(1)
            baseline_cols = [f"{c}_lag1"]
        sub_j = sub_j.dropna()
        if len(sub_j) < 30:
            continue
        y_f = sub_j[c].values
        X_base_f = sub_j[baseline_cols].values
        X_full_f = np.hstack([X_base_f, sub_j[["prob_lag1"]].values])
        r2_b = LinearRegression().fit(X_base_f, y_f).score(X_base_f, y_f)
        r2_f = LinearRegression().fit(X_full_f, y_f).score(X_full_f, y_f)
        delta_p_to_fs.append(max(r2_f - r2_b, 0.0))

    if not delta_p_to_fs:
        return IlsEstimate(None, float(delta_r2_f_to_p), None, len(sub), resolved,
                           note="no valid P->F regressions")

    avg_delta_r2_p_to_f = float(np.mean(delta_p_to_fs))
    denom = max(delta_r2_f_to_p, floor)
    ils = avg_delta_r2_p_to_f / denom
    return IlsEstimate(
        ils=float(ils),
        delta_r2_f_to_p=float(delta_r2_f_to_p),
        avg_delta_r2_p_to_f=avg_delta_r2_p_to_f,
        n=len(sub),
        resolved_channels=resolved,
    )


def block_bootstrap_ils(
    panel: pd.DataFrame,
    prob_col: str,
    dprob_col: str,
    primary: list[str],
    *,
    n_boot: int = 500,
    block_len: int = 10,
    rng: np.random.Generator | None = None,
    secondary: list[str] | None = None,
    min_rows: int = 50,
) -> dict:
    """Block bootstrap the ILS to obtain a 95% CI."""
    rng = rng if rng is not None else np.random.default_rng(0)
    sub_cols = [prob_col, dprob_col] + primary + (secondary or [])
    sub = panel[[c for c in sub_cols if c in panel.columns]].dropna().reset_index(drop=True)
    if len(sub) < max(block_len * 5, min_rows):
        return {"ils_mean": None, "ci_lower": None, "ci_upper": None,
                "n_boot_successful": 0,
                "note": "insufficient rows for block bootstrap"}
    n_blocks = (len(sub) + block_len - 1) // block_len
    boot_estimates: list[float] = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(len(sub) - block_len + 1, 1), size=n_blocks)
        idx_blocks = [np.arange(s, min(s + block_len, len(sub))) for s in starts]
        idx = np.concatenate(idx_blocks)[: len(sub)]
        out = estimate_ils(sub.iloc[idx].reset_index(drop=True), prob_col, dprob_col,
                           primary=primary, secondary=secondary, min_rows=min_rows)
        if out.ils is not None:
            boot_estimates.append(out.ils)
    if not boot_estimates:
        return {"ils_mean": None, "ci_lower": None, "ci_upper": None,
                "n_boot_successful": 0,
                "note": "no successful bootstrap replicates"}
    arr = np.array(boot_estimates)
    return {
        "ils_mean": float(arr.mean()),
        "ci_lower": float(np.quantile(arr, 0.025)),
        "ci_upper": float(np.quantile(arr, 0.975)),
        "n_boot_successful": len(boot_estimates),
    }
