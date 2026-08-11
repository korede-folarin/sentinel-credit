# Sentinel Credit — Validation Results

**Run against:** extended-horizon dataset (post Decision 16)
**Data:** 50,000 customers · 174,421 transactions · 483,579 repayments
**Observation window:** origination to 2024-10-31, performance observed to **2025-12-31**
**Right-censoring:** **0.00%** — every approved loan reaches an observed outcome
**Script:** `validation/validation.py`
**Artifacts:** `reports/validation/` — `fidelity.csv`, `utility.csv`, `privacy.csv`, `validation_metadata.json`

> **Horizon change.** `data_end_date` moved from 2024-12-31 to 2025-12-31 (`origination_end_date` unchanged) to eliminate right-censoring. Same customers, same transactions — the change gives existing loans time to resolve. See Decision 16.

---

## Fidelity — 8/8 passed

| Check | Hellinger | JS divergence | Max abs diff | Result |
|---|---|---|---|---|
| age_distribution — Klarna Wikipink UK 2024 | 0.0019 | 0.0000 | 0.0022 | PASS |
| income_bands — ONS FYE 2023 shifted | 0.0043 | 0.0000 | 0.0033 | PASS |
| merchant_categories — Klarna UK directory | 0.0029 | 0.0000 | 0.0016 | PASS |

| Prevalence check | Target | Observed | Diff | Result |
|---|---|---|---|---|
| thin_file — PwC/TotallyMoney 2022 | 31.00% | 31.22% | 0.22% | PASS |
| resilience driver — FCA Financial Lives 2022 | 24.00% | 24.08% | 0.08% | PASS |
| health driver — NHS Digital | 17.00% | 16.95% | 0.05% | PASS |
| ab_test group split — engineered 50/50 | 50.00% | 49.40% | 0.60% | PASS |

Merchant categories — completeness: PASS.

**Every fidelity score is identical to the pre-extension run.** That is the expected and reassuring result: extending the observation window changes *when outcomes resolve*, not *who the customers are*. Population characteristics, income distributions, merchant mix and vulnerability prevalences are all generated before any repayment simulation occurs, so a horizon change cannot touch them. Identical scores are therefore evidence the change did what it was supposed to and nothing else.

---

## Utility — PASS (TSTR + feature direction, real Home Credit)

| Metric | This run | Previous run | Change |
|---|---|---|---|
| Synthetic Gini | 0.4151 | 0.3954 | +0.0197 |
| Real (Home Credit) Gini | 0.3071 | 0.3110 | −0.0039 |
| Gini retention | 0.7398 | 0.7865 | **−0.0467** |
| Gini gap | 0.1080 | 0.0844 | +0.0236 |
| Sign agreement | 75.00% | 75.00% | — |
| Spearman rank correlation | 0.2000 | 0.8000 | **−0.6000** |

**Pass condition met** (`real Gini > 0.10`, `sign agreement ≥ 0.75`).

Two movements worth recording rather than passing over:

**Retention fell from 0.7865 to 0.7398.** The synthetic-trained model now retains ~74% of its discriminatory power when scored on real Home Credit data, down from ~79%. The synthetic model itself got *stronger* (Gini 0.3954 → 0.4151) while transfer weakened slightly — consistent with the model learning patterns more specific to the fully observed synthetic population. Not investigated further.

**Rank correlation fell from 0.8000 to 0.2000.** This is the larger movement. Feature *importance ordering* between synthetic-trained and real-trained models now agrees much less, while *sign* agreement is unchanged at 75%. The four cross-market features (`age`, `annual_income_gbp`, `existing_monthly_obligations_gbp`, `credit_score`) still point the same directions, but their relative importance ranks differently. With only four features, rank correlation is highly sensitive — a single position swap moves it substantially — so this is noted as a change to watch rather than a diagnosed problem.

The Gini gap (0.1080) sits outside the 0.05 tolerance. As documented in `run_utility()` and Decision 2's addendum, the gap is reported as supporting context rather than a pass/fail gate: a genuinely harder real dataset can widen it without indicating a broken generator.

---

## Privacy — 2/2 passed

| Check | Result | Threshold |
|---|---|---|
| Singling-out — 3,952 unique records (7.90%) | PASS | 20.00% |
| Nearest-neighbour — mean distance 0.2088, near-duplicate rate 0.00% | PASS | 1.00% |

Identical to the previous run, for the same reason as fidelity: privacy is a property of the customer records, which the horizon change does not touch.

---

## Known gaps

- **CTGAN benchmark** — planned, not implemented (Decision 2)
- **Internal consistency check** (`n_runs`, `ks_threshold`) — planned, not implemented (Decision 2)
- **Gini gap** exceeds tolerance (0.1080 vs 0.0500) — reported for visibility, not a failure condition
- **Rank correlation drop** (0.80 → 0.20) — recorded, not diagnosed
- **Transaction-level default reference** — the generator reports 0.90% against a reference of ~0.40%. That reference may have been set under the previous, shorter follow-up; whether it still applies under full observation is unresolved. See Decision 16.

---

## Reproducing

```bash
python validation/validation.py
```

Requires `data_generator/output/{customers,transactions,repayments}.csv`. For the full TSTR path rather than the internal fallback, `validation/reference/home_credit.csv` must be present. All results are written to `reports/validation/`.
