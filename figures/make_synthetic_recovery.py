"""Figure 2: synthetic Monte Carlo recovery curves.

(a) effect-size sweep at empirical n=16
(b) panel-size sweep at fixed beta=0.30
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
JSON_PATH = REPO / "outputs/ius_synthetic_validation.json"
OUT_DIR = REPO / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not JSON_PATH.exists():
        raise SystemExit(f"missing {JSON_PATH}; run experiments/synthetic_validation.py first")
    with JSON_PATH.open() as f:
        res = json.load(f)
    beta_rows = res["multiseed_beta_sweep_n16"]["rows"]
    n_rows = res["multiseed_n_sweep_beta030"]["rows"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    betas = [r["beta"] for r in beta_rows]
    mean_b = [r["mean_rho_ils_L"] for r in beta_rows]
    lo_b = [r["ci_lo_rho_ils_L"] for r in beta_rows]
    hi_b = [r["ci_hi_rho_ils_L"] for r in beta_rows]
    ax1.fill_between(betas, lo_b, hi_b, color="#1f77b4", alpha=0.20,
                     label="95% CI (30 seeds)")
    ax1.plot(betas, mean_b, "o-", color="#1f77b4", lw=1.6, ms=5,
             label=r"$\bar\rho(\mathrm{ILS}, L)$")
    ax1.axhline(-0.046, color="#d62728", lw=1.2, ls="--",
                label=r"empirical $\rho = -0.046$")
    ax1.axhline(0, color="#888", lw=0.7, ls=":")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Cross-coupling magnitude $\beta$ (log)")
    ax1.set_ylabel(r"$\rho(\mathrm{ILS}, L)$")
    ax1.set_title(r"(a) Effect-size sweep, $n_\mathrm{markets} = 16$")
    ax1.set_ylim(-0.2, 1.05)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, ls=":", lw=0.4, alpha=0.5)

    n_m = [r["n_markets"] for r in n_rows]
    mean_n = [r["mean_rho_ils_L"] for r in n_rows]
    lo_n = [r["ci_lo_rho_ils_L"] for r in n_rows]
    hi_n = [r["ci_hi_rho_ils_L"] for r in n_rows]
    ax2.fill_between(n_m, lo_n, hi_n, color="#2ca02c", alpha=0.20,
                     label="95% CI (30 seeds)")
    ax2.plot(n_m, mean_n, "s-", color="#2ca02c", lw=1.6, ms=5,
             label=r"$\bar\rho(\mathrm{ILS}, L)$")
    ax2.axhline(-0.046, color="#d62728", lw=1.2, ls="--",
                label=r"empirical $\rho = -0.046$")
    ax2.axhline(0, color="#888", lw=0.7, ls=":")
    ax2.axvline(16, color="#888", lw=0.7, ls=":")
    ax2.text(16.5, -0.15, r"$n_\mathrm{empirical}=16$", fontsize=8, color="#666")
    ax2.set_xlabel(r"Panel size $n_\mathrm{markets}$")
    ax2.set_ylabel(r"$\rho(\mathrm{ILS}, L)$")
    ax2.set_title(r"(b) Panel-size sweep, $\beta = 0.30$")
    ax2.set_ylim(-0.2, 1.05)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, ls=":", lw=0.4, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "synthetic_recovery.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "synthetic_recovery.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_DIR / 'synthetic_recovery.pdf'}")


if __name__ == "__main__":
    main()
