# IUS Panel Pre-Registration

_Author: Saman Sayahpour._
_Frozen on committing this file to git. Commit SHA is the binding identifier._
_Plan ref: `/IUS_EXPANSION_PLAN.md` Phase A._

**Purpose.** Lock the market panel, per-market IUS tier, per-market financial-asset channel, resolution, and hypothesis tests *before* any IUS or causal-discovery computation on the expanded panel. Any post-freeze change is a deviation and is logged in §6 below.

---

## 1. Primary hypothesis (locked)

**H1.** Spearman $\rho$ between ILS rank and bootstrapped leadership-score rank across the 16-market panel is positive at permutation $p < 0.05$ (two-sided, 10{,}000 permutations). Leadership score $L_m \in [0, 1]$ is the bootstrap fraction of (method, hyperparameter, block-resample) triples that classify PM as leader, per `/IUS_EXPANSION_PLAN.md` §5.2.

**Decision rule:** $p < 0.05$ → positive empirical finding; $0.05 \le p < 0.10$ → suggestive, reported as such; $p \ge 0.10$ → null, pivot to methodology-first paper framing.

## 2. Secondary hypotheses (locked)

**H2.** Mann-Whitney U on $L$ between pre-registered IUS tiers (high vs low, high vs mid, mid vs low). BH FDR at $q = 0.10$ across the 3 sub-tests.

**H3.** Stratified JCI (Theorem `unified_jci`) applied to any same-event cross-platform PM pair in the panel (at minimum KXFED ↔ PolyFed Fed-decisions) yields an invariant-edge set consistent across (platform, regime) strata.

**H4.** Iran pre-strike window (Jan 2025 US strikes) case study — LPCMCI + signature-(a, b) tests + K=20 matched placebos — reports per-event $p$ with BH across two signatures. Auxiliary to H1 (appendix-grade evidence).

---

## 3. Market panel (locked, 16 primary + 2 backup)

Tokens are the Polymarket `clobTokenIds` on the YES outcome (extracted via Gamma API, see `configs/ius_panel_raw.json`). The NO token is used for asymmetry checks (§YES/NO asymmetry in `paper/sections/results_causal.tex`). For Kalshi markets the "token" field is the Kalshi event slug.

### 3.1 High-IUS tier (expected leaders; n = 8)

| # | id | title | volume | yes_token | end_date | n_hist_pts_verified |
|---|---|---|---|---|---|---|
| 1 | `us_pres_election_2024` | Presidential Election Winner 2024 | \$3686M | `21742633143463906290569050155826241533067272736897614950488156847949938836455` | 2024-11-05 | 2{,}209 |
| 2 | `us_strikes_iran` | US strikes Iran by...? | \$529M | *(from `ius_panel_raw.json`)* | 2026-06-30 | 4{,}096 |
| 3 | `trump_fed_chair` | Who will Trump nominate as Fed Chair? | \$617M | *(see raw.json)* | 2026-12-31 | 4{,}988 |
| 4 | `tiktok_ban` | TikTok banned in the US before May 2025? | \$120M | *(see raw.json)* | 2025-04-30 | 3{,}024 |
| 5 | `germany_parl_2025` | Germany Parliamentary Election Winner | \$135M | *(see raw.json)* | 2025-02-23 | 1{,}674 |
| 6 | `canada_pm_2025` | Next Prime Minister of Canada | \$121M | `304223829867910478181460849192...` | 2025-04-28 | 2{,}900 |
| 7 | `khamenei_out` | Khamenei out by Feb 28, 2026? | \$131M | `393178854220263942590563281445...` | 2026-02-28 | 1{,}102 |
| 8 | `gov_shutdown_2026` | US government shutdown Saturday? | \$157M | `526073159005071568466228207704...` | 2026-01-31 | 1{,}904 |

### 3.2 Mid-IUS tier (expected mixed; n = 4)

| # | id | title | volume | yes_token | end_date | n_hist_pts_verified |
|---|---|---|---|---|---|---|
| 9 | `sk_president_2025` | Next president of South Korea? | \$291M | `274378653046145435487760359002...` | 2025-06-03 | 1{,}440 |
| 10 | `poland_pres_2025` | Poland Presidential Election | \$129M | `523783104469534651638453380483...` | 2025-06-01 | 4{,}396 |
| 11 | `speaker_house` | Who will be the Speaker of the House? | \$188M | `295123917582081896677478647657...` | 2025-02-01 | 1{,}247 |
| 12 | `portugal_pres_2026` | Portugal Presidential Election | \$136M | `282383049631153914685200846117...` | 2026-02-08 | 3{,}901 |

### 3.3 Low-IUS tier (expected followers; n = 4)

| # | id | title | platform | source |
|---|---|---|---|---|
| 13 | `kxfed` | KXFED Fed-funds-rate range | Kalshi | `data/processed/ges_ready_hourly.parquet` (8{,}246 rows) |
| 14 | `kxcpi` | KXCPI CPI-inflation range | Kalshi | `data/processed/ges_ready_kxcpi_hourly.parquet` (2{,}839 rows) |
| 15 | `polyfed_jan_2026` | Polymarket Fed decision in January 2026 | Polymarket | `118621655667573459852404761644...` (2{,}127 pts verified) |
| 16 | `polymarket_btc_2025` | What price will Bitcoin hit in 2025? | Polymarket | `112540911653160777059655478391...` (8{,}786 pts verified) |

### 3.4 Backup (used only if a primary market fails acceptance §4)

| # | id | title | yes_token | n_hist_pts_verified |
|---|---|---|---|---|
| B1 | `ireland_pres_2025` | Ireland Presidential Election | `623596801110872600136059360017...` | 5{,}257 |
| B2 | `romania_pres_2025` | Romania Presidential Election Winner | `433898080839912703499783522268...` | 1{,}330 |

---

## 4. Per-market IUS classification rationale + channel vector (locked)

### 4.1 High-IUS (expected leaders)

| id | Rationale | Primary channels | Secondary | Placebo |
|---|---|---|---|---|
| `us_pres_election_2024` | Political outcome with diffuse financial impact; information aggregates from polls, debates, scandals. | `SPY_logret, XLE_logret, XLF_logret, XLV_logret` | `IWM_logret, XLC_logret` | `USO_logret` |
| `us_strikes_iran` | Geopolitical outcome; information leaks through intelligence, deployment signals. Documented insider trading on this market. | `USO_logret, Gold_logret, VIX_logchg` | `SPY_logret, TLT_logret` | `XLP_logret` |
| `trump_fed_chair` | Regulatory appointment; information leaks through WH sources, candidate shortlists. | `TLT_logret, IEF_logret` | `SPY_logret, DXY_logret` | `USO_logret` |
| `tiktok_ban` | Legal/regulatory; information leaks through DoJ filings, court schedules. | `META_logret, GOOGL_logret, AAPL_logret` | `SPY_logret` | `TLT_logret` |
| `germany_parl_2025` | Electoral outcome with direct EUR/DAX impact; information via polls. | `EWG_logret, FXE_logret, IEF_logret` | `SPY_logret` | `USO_logret` |
| `canada_pm_2025` | Electoral outcome with CAD/energy impact; information via polls. | `FXC_logret, USO_logret` | `SPY_logret` | `TLT_logret` |
| `khamenei_out` | Geopolitical event with oil/defense impact; rumor-driven. | `USO_logret, Gold_logret` | `VIX_logchg, SPY_logret` | `XLP_logret` |
| `gov_shutdown_2026` | Legislative brinksmanship; information via congressional leaks. | `SPY_logret, TLT_logret, VIX_logchg` | `IWM_logret` | `USO_logret` |

### 4.2 Mid-IUS (expected mixed)

| id | Rationale | Primary channels | Secondary | Placebo |
|---|---|---|---|---|
| `sk_president_2025` | Electoral, but KRW/KOSPI impact is narrower than US markets. | `EWY_logret` | `SPY_logret, Gold_logret` | `USO_logret` |
| `poland_pres_2025` | Electoral with EUR/PLN impact but smaller-economy. | `FXE_logret` | `SPY_logret, IEF_logret` | `USO_logret` |
| `speaker_house` | Procedural legislative outcome; impact is short-lived and diffuse. | `SPY_logret, TLT_logret` | `IWM_logret` | `USO_logret` |
| `portugal_pres_2026` | Electoral with EUR impact; very small-economy. | `FXE_logret` | `SPY_logret, IEF_logret` | `USO_logret` |

### 4.3 Low-IUS (expected followers)

| id | Rationale | Primary channels | Secondary | Placebo |
|---|---|---|---|---|
| `kxfed` | Scheduled event (FOMC decision) already priced by Fed-funds futures and rate markets. | `TLT_logret, d_effr, d_sofr` | `SPY_logret, IEF_logret` | `USO_logret` |
| `kxcpi` | Scheduled event (CPI release) already priced by TIPS breakevens and rate markets. | `TLT_logret, d_t10yie` | `SPY_logret, DXY_logret` | `USO_logret` |
| `polyfed_jan_2026` | Same underlying event as KXFED; competing platform. | `TLT_logret, d_effr` | `SPY_logret` | `USO_logret` |
| `polymarket_btc_2025` | PM is a derivative on BTC spot price; redundant by construction. | `BTC_logret` | `ETH_logret (if collected)` | `SPY_logret` |

---

## 5. Acceptance criteria per market (locked)

A market is included in the final H1 panel only if **all** of the following hold after panel construction (Phase C):

1. **Sufficient active data:** $n_\text{active} := \sum_t T_t \ge 300$ where $T_t = 1$ iff at least one PM trade fired in hour $t$.
2. **Non-degenerate activity:** $\hat\pi_1 = n_\text{active} / n_\text{total} \ge 0.10$.
3. **ILS stability:** block-bootstrap 95% CI on ILS has finite upper and lower bounds (i.e., no estimator blowup).
4. **Channel coverage:** at least 2 of the pre-registered primary channels have $\ge 80\%$ non-null hourly coverage over the market's lifetime.

Markets failing any of (1)–(4) are excluded. If a high-IUS primary fails and a backup is available, the backup is promoted. Any exclusion is logged in §6 with which criterion failed.

Minimum final panel size for H1: **$n \ge 13$**. If <13 markets pass, H1 is reported as underpowered and the paper pivots per `/IUS_EXPANSION_PLAN.md` §10 Ship 1.

---

## 6. Deviations log

_Any deviation from §1–§5 that is made after this pre-registration is committed is documented here with timestamp, committed-to-commit-SHA before/after, and explicit justification._

_(currently empty — pre-registration is freshly committed)_

---

## 7. Commit trailer

This pre-registration is committed at `/methodology/ius_panel_preregistration.md`. The commit SHA that *includes this file* is the binding identifier for the frozen design. Any later change to §1–§5 is a deviation and must be logged in §6 with rationale.

**Frozen metadata file:** `configs/ius_panel_raw.json` contains the complete token identifiers, fetch ranges, and verified data-point counts at freeze time.

**Reproducibility:** all subsequent Phase B through H artifacts trace back to this commit SHA via the `frozen_prereg_sha` field in their metadata JSON outputs.
