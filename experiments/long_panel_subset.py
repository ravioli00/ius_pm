"""Direct test of Mechanism (1) from Section 6: panel-length confounding.

Restricting the test to markets with n_used >= 800 (the five longest-running
markets), all five leadership operationalisations (pre-registered PCMCI+
bootstrap, plus four post-hoc variants from outputs/methodology/ius_expansion/
post_hoc_tests.json) yield positive Spearman rho with ILS.

Output: outputs/long_panel_subset.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ius import permutation_spearman

REPO = Path(__file__).resolve().parents[1]
POST_HOC_JSON = REPO / "data/post_hoc_tests.json"
ILS_CSV = REPO / "data/ius_panel.csv"
OUT_PATH = REPO / "outputs/long_panel_subset.json"

L_VARIANTS = [
    "leadership_score_L",
    "L_granger_ratio",
    "L_pcmci_ratio",
    "L_lpcmci_ratio",
    "L_leadlag_corr",
]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POST_HOC_JSON.open() as f:
        ph = json.load(f)
    panel = pd.DataFrame(ph["panel_data"]).set_index("id")
    ils = pd.read_csv(ILS_CSV).set_index("id")
    panel["n_used"] = ils["n_used_for_ils"]

    results: dict = {"thresholds": {}}
    for thr in (0, 300, 500, 800):
        sub = panel[panel["n_used"] >= thr]
        if len(sub) < 4:
            continue
        block = {"n_markets": int(len(sub)),
                 "markets": list(sub.index),
                 "per_metric": {}}
        for col in L_VARIANTS:
            if col not in sub.columns:
                continue
            perm = permutation_spearman(
                sub["ils"].values, sub[col].values,
                n_perm=10_000,
                rng=np.random.default_rng(20260502))
            block["per_metric"][col] = {
                "rho": perm.rho_observed,
                "p_two": perm.p_two_sided,
                "p_one": perm.p_one_sided,
            }
        results["thresholds"][f"n_used_ge_{thr}"] = block

    OUT_PATH.write_text(json.dumps(results, indent=2))

    print("Direct test of Mechanism (1) (panel-length confounding):\n")
    for thr_label, block in results["thresholds"].items():
        print(f"  {thr_label}  (n={block['n_markets']}):")
        for metric, stats in block["per_metric"].items():
            print(f"    {metric:30s} rho={stats['rho']:+.4f}  "
                  f"p_two={stats['p_two']:.4f}  p_one={stats['p_one']:.4f}")
        rhos = [s["rho"] for s in block["per_metric"].values()]
        n_pos = sum(1 for r in rhos if r > 0)
        print(f"    -- mean rho = {np.mean(rhos):+.4f}, "
              f"{n_pos}/{len(rhos)} positive\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
