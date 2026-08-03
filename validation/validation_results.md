# Sentinel Credit — Validation Results

**Run against:** hazard-based, emergent-default data (post
Decision 10 rewrite)
**Data:** 50,000 customers · 174,421 transactions · 483,579
repayments
**Script:** `validation/validate_synthetic.py` v1.1.0
**Date recorded:** 30th July 2026

This is the first validation run recorded against the
rebuilt hazard-based dataset. Prior numbers in
`engineering_decision_log.md` Appendix A (e.g. 1.47% emergent
default rate) reflect the pre-rewrite, probability-drawn
generator and are historical only — do not compare them
directly to the figures below, which are a different data
generation process end to end.

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

Merchant categories — completeness: PASS (all configured
categories present, no unexpected categories).

All fidelity scores sit well inside their thresholds
(`hellinger_threshold: 0.05`, `prevalence_tolerance: 0.01`),
consistent with the CREDIT 1 reasoning that a well-calibrated
generator at n=50,000 should score roughly 10x under the
Hellinger ceiling.

---

## Utility — PASS (TSTR + feature direction, real Home Credit)

| Metric | Value |
|---|---|
| Synthetic Gini | 0.3954 |
| Real (Home Credit) Gini | 0.3110 |
| Gini retention | 0.7865 |
| Gini gap | 0.0844 |
| Gini gap tolerance | 0.0500 |
| Gap within tolerance | **No — outside tolerance** |
| Sign agreement (Test B) | 75.00% |
| Spearman rank correlation | 0.8000 |

**Reading this honestly:** the pass condition (`real Gini >
0.10` and `sign agreement ≥ 0.75`) is met, so this is a PASS.
The Gini gap sits outside the 0.05 tolerance band — this is
reported as context, not a failure condition, per the
documented reasoning in `run_utility()`: a genuinely harder
or easier real dataset can widen this gap without indicating
a broken generator. Retention of 0.7865 means the
synthetic-trained model retains ~79% of its discriminatory
power when scored on real Home Credit data — a meaningfully
different (better) result than the pre-rewrite retention of
1.64 recorded in earlier sessions, which was itself flagged
as reflecting Home Credit's easier separability on those four
features rather than generator quality. This run's retention
being below 1.0 rather than above it is arguably a more
intuitive result to explain, though both are defensible given
the documented cross-market limitation.

Sign agreement at exactly 75% means 3 of 4 cross-market
features (`age`, `annual_income_gbp`,
`existing_monthly_obligations_gbp`, `credit_score`) agree in
direction between the synthetic-trained and real-trained
models — matching the CREDIT 1 finding that a single
disagreeing feature is consistent with the documented
BNPL-vs-consumer-loan market difference, not generator error.

---

## Privacy — 2/2 passed

| Check | Result | Threshold |
|---|---|---|
| Singling-out — unique records: 3,952 (7.90%) | PASS | 20.00% |
| Nearest-neighbour — mean distance 0.2088, near-duplicate rate 0.00% | PASS | 1.00% |

Both thresholds are now config-driven
(`singling_out_threshold`, `near_duplicate_threshold`) rather
than hardcoded, per Decision 2's "Known Gaps" correction.
Results are unchanged from what the previous hardcoded
values would have produced — moving the threshold into
config did not alter the underlying data or the computed
rates.

---

## Known gaps in this run

- **CTGAN benchmark** — planned, not implemented (Decision 2).
- **Internal consistency check** (`n_runs`, `ks_threshold`) —
  planned, not implemented (Decision 2).
- Gini gap (0.0844) exceeds the documented tolerance (0.0500)
  — flagged here for visibility, not treated as a failure,
  per the reasoning above.

---

## Reproducing this run

```bash
python validation/validate_synthetic.py
```

Requires `data_generator/output/{customers,transactions,repayments}.csv`
to exist (run the generator first) and, for the full TSTR
path rather than the internal fallback,
`validation/reference/home_credit.csv` to be present.
