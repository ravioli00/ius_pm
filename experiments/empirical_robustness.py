"""Empirical subset-robustness re-tests on the pre-registered 16-market panel.

Re-runs the Spearman test on the panel under a series of post-hoc subsets:
tier-internal partitions, leverage-point removal (drop PolyBTC),
short-panel exclusions, high-uncertainty exclusions, and combinations.

Inputs:
    data/ius_panel.csv          (per-market ILS + bootstrap CI)
    data/sweep_summary.csv      (per-market leadership score L from PCMCI+ bootstrap)

Output: outputs/empirical_robustness.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ius import permutation_spearman

REPO = Path(__file__).resolve().parents[1]
ILS_CSV = REPO / "data/ius_panel.csv"
SWEEP_CSV = REPO / "data/sweep_summary.csv"
OUT_PATH = REPO / "outputs/empirical_robustness.json"


def _load_panel() -> pd.DataFrame:
    ils = pd.read_csv(ILS_CSV).set_index("id")
    sweep = pd.read_csv(SWEEP_CSV).set_index("id")
    panel = ils.join(sweep[["leadership_score_L"]], how="inner")
    panel["ci_width_log10"] = np.log10(
        panel["ils_ci_upper"].clip(lower=1e-10) /
        panel["ils_ci_lower"].clip(lower=1e-10))
    return panel


def _run_subset(panel: pd.DataFrame, label: str, mask: pd.Series) -> dict:
    sub = panel[mask]
    if len(sub) < 4:
        return {"label": label, "n": int(len(sub)),
                "note": "n < 4, skipped"}
    perm = permutation_spearman(sub["ils"].values,
                                 sub["leadership_score_L"].values,
                                 n_perm=10_000,
                                 rng=np.random.default_rng(20260502))
    return {"label": label, "n": int(len(sub)),
            "rho_ILS_L": perm.rho_observed,
            "perm_p_two": perm.p_two_sided,
            "perm_p_one": perm.p_one_sided,
            "markets": list(sub.index)}


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel = _load_panel()
    results: dict = {}

    results["full_panel_n16"] = _run_subset(panel, "full panel",
                                             pd.Series(True, index=panel.index))
    for tier in ("high", "mid", "low"):
        results[f"tier_{tier}"] = _run_subset(panel, f"tier={tier}",
                                              panel["tier"] == tier)
    results["drop_polybtc"] = _run_subset(panel, "drop polybtc",
                                          panel.index != "polymarket_btc_2025")
    for thr in (300, 500, 800):
        results[f"n_used_ge_{thr}"] = _run_subset(
            panel, f"n_used >= {thr}", panel["n_used_for_ils"] >= thr)
    for thr in (1.5, 2.0, 2.5):
        results[f"ci_width_le_{thr:.1f}"] = _run_subset(
            panel, f"log10(CI ratio) <= {thr}",
            panel["ci_width_log10"] <= thr)
    results["drop_polybtc_and_n_ge_500"] = _run_subset(
        panel, "drop polybtc, n>=500",
        (panel.index != "polymarket_btc_2025") & (panel["n_used_for_ils"] >= 500))

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print("Empirical-subset robustness summary:")
    for k, v in results.items():
        if "rho_ILS_L" in v:
            print(f"  {k:35s} n={v['n']:2d}  rho={v['rho_ILS_L']:+.4f}  "
                  f"p_two={v['perm_p_two']:.4f}  p_one={v['perm_p_one']:.4f}")
        else:
            print(f"  {k:35s} {v.get('note', '')}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
