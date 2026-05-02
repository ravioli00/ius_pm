"""Permutation tests used in the IUS panel analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats


@dataclass
class PermutationResult:
    """Output of :func:`permutation_spearman`."""

    rho_observed: float
    p_two_sided: float
    p_one_sided: float
    n: int
    n_perm: int


def permutation_spearman(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    n_perm: int = 10_000,
    rng: np.random.Generator | None = None,
) -> PermutationResult:
    """Two-sided and one-sided permutation tests for Spearman rho.

    The one-sided p-value is the fraction of permutations with rho >= rho_observed,
    aligned with the directional alternative ``rho > 0``.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if len(x) < 3:
        raise ValueError("need at least 3 observations for Spearman rho")

    rho_obs, _ = sp_stats.spearmanr(x, y)
    if np.isnan(rho_obs):
        return PermutationResult(float("nan"), float("nan"), float("nan"),
                                 n=len(x), n_perm=n_perm)

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        null[i] = sp_stats.spearmanr(x, rng.permutation(y))[0]
    null = null[~np.isnan(null)]
    p_two = float((np.abs(null) >= abs(rho_obs)).mean())
    p_one = float((null >= rho_obs).mean())
    return PermutationResult(
        rho_observed=float(rho_obs),
        p_two_sided=p_two,
        p_one_sided=p_one,
        n=int(len(x)),
        n_perm=int(len(null)),
    )
