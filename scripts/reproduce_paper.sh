#!/usr/bin/env bash
# Reproduce all paper tables, figures, and outputs end-to-end.
# Assumes you have run `pip install -e ".[test]"` in the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Running test suite ==="
pytest -q

echo
echo "=== Synthetic Monte Carlo (Stage 1-4) ==="
python experiments/synthetic_validation.py

echo
echo "=== Empirical subset robustness ==="
python experiments/empirical_robustness.py

echo
echo "=== Direct test of Mechanism (1): long-panel subset ==="
python experiments/long_panel_subset.py

echo
echo "=== Figures ==="
python figures/make_ius_scatter.py
python figures/make_synthetic_recovery.py

echo
echo "=== Done. Outputs in outputs/, figures in figures/ ==="
