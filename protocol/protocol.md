# Study protocol

The analysis plan that was frozen before any IUS or causal-discovery
computation on the 16-market panel. The git commit SHA that first introduced
this file is the binding identifier; subsequent edits are tracked in
**Deviations** below.

The market panel, tier assignments, and per-market financial channels are
specified in [`manifest.yaml`](manifest.yaml).

## Primary test (H1)

| | |
|---|---|
| Statistic   | Spearman $\rho$ between $\mathrm{ILS}_m$ and bootstrap leadership score $L_m$ across the 16 panel markets |
| Test        | Two-sided permutation, 10{,}000 reps |
| Threshold   | $p < 0.05$ |
| Decision    | $p < 0.05$ → positive finding; $p \ge 0.05$ → null, paper pivots to methodology-first framing |

$L_m \in [0, 1]$ is the bootstrap fraction of PCMCI+ replicates classifying
PM as the leader, with the configuration below.

## Secondary tests

**H2.** Mann-Whitney $U$ on $L$ across the (high, mid, low) IUS tiers, BH-FDR
at $q \le 0.10$ across the three pairwise contrasts.

**H3.** Iran pre-strike event-window case study: pairwise Granger causality
on a 7-day pre-strike window before the 22 June 2025 US strikes, with
$K = 30$ calendar-matched placebo windows $\ge 14$ days prior, max-lag
3 hours, BH-FDR at $q \le 0.10$ across five financial channels (Oil, Gold,
VIX, SPY, TLT).

## Acceptance criteria

A market enters the panel iff:
- $n_\text{active} \ge 300$ hourly observations (relaxed to $\ge 200$ for
  political contracts whose lifetime is bounded by an event date *ex-ante*:
  Canada PM 2025, Germany Parl. 2025).
- $\hat\pi_1 \ge 0.10$ trade-arrival fraction.
- ILS bootstrap CI is finite.
- $\ge 2$ primary financial channels with $\ge 80\%$ coverage.

12 high-IUS candidates were dropped at the acceptance gate.

## Configuration

| | |
|---|---|
| Time resolution | 1 hour |
| Block-bootstrap block length | 10 hours |
| Block-bootstrap resamples (ILS CI) | 500 |
| PCMCI+ $\tau_\text{max}$ | 4 |
| PCMCI+ $\alpha$ | 0.05 |
| PCMCI+ CI test | ParCorr |
| PCMCI+ bootstrap replicates per market | 50 |
| Permutation reps (H1) | 10{,}000 |

## Tier assignment

Tiers were assigned by author judgment of expected information-uniqueness
profile prior to any IUS computation. No quantitative tier rule is proposed,
and tier-level tests (H2) are reported as exploratory.

| Tier | Expected $L$ | Domains |
|---|---|---|
| high | high | political, geopolitical, legal events |
| mid  | intermediate | secondary political contracts |
| low  | low | price- or rate-based contracts |

## Honest scope

This is an internal frozen analysis plan, not a public time-stamped
pre-registration on OSF or AsPredicted. The 16-market panel is a convenience
sample from publicly accessible contracts meeting the acceptance criteria;
generalisation beyond this panel is not warranted by the panel test alone.
The H1 decision rule was committed to before computation; the resulting
pivot to methodology-first framing is documented in the paper.

## Deviations

None.
