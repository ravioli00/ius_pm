"""Figure 1: pre-registered 16-market panel scatter (ILS vs L)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
ILS_CSV = REPO / "data/ius_panel.csv"
SWEEP_CSV = REPO / "data/sweep_summary.csv"
OUT_DIR = REPO / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ils = pd.read_csv(ILS_CSV).set_index("id")
    sweep = pd.read_csv(SWEEP_CSV).set_index("id")
    df = ils.join(sweep[["leadership_score_L"]], how="inner") \
            .rename(columns={"leadership_score_L": "L"})

    labels = {
        "polymarket_btc_2025": "PolyBTC",
        "us_strikes_iran": "Iran",
        "trump_fed_chair": "TrumpFed",
        "polyfed_jan_2026": "PolyFed",
        "kxfed": "KXFED",
    }
    tier_colors = {"high": "#d62728", "mid": "#7f7f7f", "low": "#1f77b4"}

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axvline(1.0, color="#888", linestyle=":", lw=0.8)
    ax.axhline(0.5, color="#888", linestyle=":", lw=0.8)

    for tier in ("low", "mid", "high"):
        sub = df[df["tier"] == tier]
        xerr_lo = np.clip(sub["ils"].values - sub["ils_ci_lower"].values, 1e-6, None)
        xerr_hi = np.clip(sub["ils_ci_upper"].values - sub["ils"].values, 1e-6, None)
        sizes = 10 + np.sqrt(sub["n_used_for_ils"].values) * 1.4
        ax.errorbar(sub["ils"], sub["L"], xerr=[xerr_lo, xerr_hi],
                    fmt="none", ecolor=tier_colors[tier], alpha=0.35,
                    elinewidth=0.7, capsize=2)
        ax.scatter(sub["ils"], sub["L"], s=sizes, c=tier_colors[tier],
                   alpha=0.85, edgecolors="white", linewidths=0.6,
                   label=f"{tier}-IUS tier (n={len(sub)})", zorder=5)

    for mid, lab in labels.items():
        if mid in df.index:
            x, y = df.loc[mid, "ils"], df.loc[mid, "L"]
            ax.annotate(lab, (x, y), xytext=(6, 4),
                        textcoords="offset points", fontsize=8, alpha=0.85)

    rho, _ = stats.spearmanr(df["ils"], df["L"])
    ax.text(0.025, 0.96,
            f"Spearman $\\rho$(ILS, $L$) = {rho:+.3f}\nperm. $p$ = 0.868",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", alpha=0.9))

    ax.set_xscale("log")
    ax.set_xlim(0.005, 600)
    ax.set_ylim(-0.02, 0.7)
    ax.set_xlabel("Information Leadership Score (ILS), log scale")
    ax.set_ylabel("Bootstrap leadership score $L$")
    ax.set_title("Pre-registered 16-market IUS panel: ILS vs $L$",
                 fontsize=11, pad=8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    out = OUT_DIR / "ius_scatter.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "ius_scatter.png", dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
