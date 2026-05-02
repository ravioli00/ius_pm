"""Reproduces the synthetic Monte Carlo validation in Appendix D of the paper.

Stage 1: linear-Gaussian baseline at three panel sizes (n=16, 30, 60).
Stage 2: violated-DGP robustness (heavy-tail, regime-switch, latent confounder).
Stage 3: multi-seed effect-size sweep at the empirical panel size.
Stage 4: multi-seed panel-size sweep at fixed beta=0.30.

Output: outputs/ius_synthetic_validation.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ius import (channel_names, estimate_ils, granger_f_ratio,
                 permutation_spearman, simulate_market)

REPO = Path(__file__).resolve().parents[1]
OUT_PATH = REPO / "outputs/ius_synthetic_validation.json"


def run_one_config(n_markets: int, n_obs: int, K: int, *, seed: int,
                   beta: float = 0.6, dgp: str = "gaussian",
                   n_perm: int = 10_000, **dgp_kwargs) -> dict:
    rng = np.random.default_rng(seed)
    alphas = rng.uniform(-1.0, 1.0, n_markets)
    chs = channel_names(K)
    rows = []
    for i, alpha in enumerate(alphas):
        df = simulate_market(alpha=alpha, n=n_obs, K=K, rng=rng,
                             beta=beta, dgp=dgp, **dgp_kwargs)
        ils_out = estimate_ils(df, "prob", "dprob", primary=chs)
        gr_out = granger_f_ratio(df, "prob", chs, lag=1)
        rows.append({
            "market": i, "alpha": float(alpha),
            "ils": ils_out.ils, "L": gr_out.L,
            "n": ils_out.n,
        })
    valid = [r for r in rows if r["ils"] is not None and not np.isnan(r["L"])]
    if len(valid) < 5:
        return {"error": "too few successful estimates",
                "n_successful": len(valid)}
    ils_arr = np.array([r["ils"] for r in valid])
    L_arr = np.array([r["L"] for r in valid])
    alpha_arr = np.array([r["alpha"] for r in valid])
    perm = permutation_spearman(ils_arr, L_arr, n_perm=n_perm,
                                 rng=np.random.default_rng(seed + 1))
    from scipy import stats
    rho_ils_a = float(stats.spearmanr(ils_arr, alpha_arr).statistic)
    rho_L_a = float(stats.spearmanr(L_arr, alpha_arr).statistic)
    return {
        "n_markets_simulated": n_markets,
        "n_markets_successful": len(valid),
        "n_obs": n_obs, "K": K, "beta": beta, "dgp": dgp,
        "rho_ils_L": perm.rho_observed,
        "permutation_p_two_sided": perm.p_two_sided,
        "permutation_p_one_sided": perm.p_one_sided,
        "rho_ils_alpha": rho_ils_a,
        "rho_L_alpha": rho_L_a,
        "per_market": rows,
    }


def multi_seed_sweep(*, betas=None, n_markets=None, n_obs=1000, K=5,
                     n_seeds=30, base_seed=20260601):
    """Multi-seed average for either beta or n_markets sweep.

    Pass exactly one of ``betas`` (list) or ``n_markets`` (list); the other
    parameter must be a scalar.
    """
    if betas is None and isinstance(n_markets, list):
        return _sweep(over="n_markets", values=n_markets, fixed_beta=0.30,
                      fixed_n=None, n_obs=n_obs, K=K, n_seeds=n_seeds,
                      base_seed=base_seed)
    return _sweep(over="beta", values=betas, fixed_beta=None,
                  fixed_n=n_markets, n_obs=n_obs, K=K, n_seeds=n_seeds,
                  base_seed=base_seed)


def _sweep(*, over, values, fixed_beta, fixed_n, n_obs, K, n_seeds, base_seed):
    chs = channel_names(K)
    rows = []
    for value in values:
        rhos_ils_L, rhos_ils_a, rhos_L_a = [], [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + int(value * 1000) * 1000 + s)
            beta = value if over == "beta" else fixed_beta
            n_m = value if over == "n_markets" else fixed_n
            alphas = rng.uniform(-1.0, 1.0, n_m)
            ils_vals, L_vals, alpha_vals = [], [], []
            for alpha in alphas:
                df = simulate_market(alpha=alpha, n=n_obs, K=K, rng=rng, beta=beta)
                ils = estimate_ils(df, "prob", "dprob", primary=chs).ils
                L = granger_f_ratio(df, "prob", chs, lag=1).L
                if ils is not None and not np.isnan(L):
                    ils_vals.append(ils)
                    L_vals.append(L)
                    alpha_vals.append(alpha)
            if len(ils_vals) >= 5:
                from scipy import stats
                rhos_ils_L.append(stats.spearmanr(ils_vals, L_vals).statistic)
                rhos_ils_a.append(stats.spearmanr(ils_vals, alpha_vals).statistic)
                rhos_L_a.append(stats.spearmanr(L_vals, alpha_vals).statistic)
        if rhos_ils_L:
            arr = np.array(rhos_ils_L)
            row = {
                over: float(value),
                "n_seeds": len(rhos_ils_L),
                "mean_rho_ils_L": float(arr.mean()),
                "std_rho_ils_L": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "ci_lo_rho_ils_L": float(np.quantile(arr, 0.025)) if len(arr) >= 4 else None,
                "ci_hi_rho_ils_L": float(np.quantile(arr, 0.975)) if len(arr) >= 4 else None,
                "mean_rho_ils_alpha": float(np.mean(rhos_ils_a)),
                "mean_rho_L_alpha": float(np.mean(rhos_L_a)),
            }
            rows.append(row)
            print(f"  {over}={value}  rho={row['mean_rho_ils_L']:+.3f} "
                  f"+/- {row['std_rho_ils_L']:.3f} "
                  f"CI=[{row['ci_lo_rho_ils_L']:+.3f}, {row['ci_hi_rho_ils_L']:+.3f}]",
                  flush=True)
    return {"sweep_over": over, "rows": rows}


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    print("--- Stage 1: linear-Gaussian baseline ---")
    for label, params in [
        ("baseline_n16", dict(n_markets=16, n_obs=1000, K=5, seed=20260502)),
        ("baseline_n30", dict(n_markets=30, n_obs=1000, K=5, seed=20260503)),
        ("baseline_n60", dict(n_markets=60, n_obs=2000, K=5, seed=20260504)),
    ]:
        r = run_one_config(**params)
        results[label] = r
        print(f"  {label}: rho={r['rho_ils_L']:+.4f}, perm p={r['permutation_p_two_sided']:.4f}")

    print("\n--- Stage 2: DGP-violation robustness ---")
    for label, params in [
        ("violated_heavy_tail", dict(n_markets=30, n_obs=1000, K=5,
                                     seed=20260505, dgp="heavy_tail")),
        ("violated_regime_switch", dict(n_markets=30, n_obs=1000, K=5,
                                        seed=20260506, dgp="regime_switch")),
        ("violated_latent_weak", dict(n_markets=30, n_obs=1000, K=5,
                                      seed=20260507, dgp="latent_conf",
                                      confounder_strength=0.5)),
        ("violated_latent_strong", dict(n_markets=30, n_obs=1000, K=5,
                                        seed=20260508, dgp="latent_conf",
                                        confounder_strength=1.5)),
    ]:
        r = run_one_config(**params)
        results[label] = r
        print(f"  {label}: rho={r['rho_ils_L']:+.4f}, perm p={r['permutation_p_two_sided']:.4f}")

    print("\n--- Stage 3: multi-seed effect-size sweep ---")
    results["multiseed_beta_sweep_n16"] = multi_seed_sweep(
        betas=[0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60],
        n_markets=16, n_obs=1000, K=5, n_seeds=30, base_seed=20260601)

    print("\n--- Stage 4: multi-seed panel-size sweep ---")
    results["multiseed_n_sweep_beta030"] = multi_seed_sweep(
        n_markets=[8, 12, 16, 20, 25, 30, 40, 60],
        n_obs=1000, K=5, n_seeds=30, base_seed=20260700)

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
