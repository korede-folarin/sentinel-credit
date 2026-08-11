# Sentinel Credit - Business Cutoff Selection Report

**Report status:** Cutoff confirmed on both splits; separate operating-policy item opened
**Dataset version:** `full_horizon_2025_v1`
**Feature-set label:** `14_features_v1_frozen`
**Model version:** `v1`
**Selection date:** 7 August 2026
**Decision log version:** 3.22
**Primary scripts:** `models/select_business_cutoff.py`, `models/segment_threshold_diagnostics.py`

---

## 1. Purpose

This report documents how `model.business_cutoff` was selected for the
frozen 14-feature v1 XGBoost champion, what evidence and constraints
shaped the choice, and what is still outstanding before the cutoff can
be called final.

This is Decision 18's step 6, closed with one dependency still open —
see Section 6.

---

## 2. Why the previous cutoff had to be replaced

`business_cutoff: 0.137` was set against the pre-horizon-extension
population and had been flagged as provisional since Decision 16.
Evaluation of the frozen 14-feature champion made the problem concrete
rather than theoretical:

| Split | Flagged at 0.137 | True positives |
|---|---:|---:|
| Stratified (n=8,321) | 0 | 0 |
| Temporal (n=4,857) | 3 | 0 |

At this cutoff, the champion approved effectively everyone. No
precision, recall, or fairness comparison at an actual operating point
was possible until the cutoff was replaced.

---

## 3. Constraints on the replacement

Two rules governed how the new cutoff could be selected, both carried
over from standing project discipline rather than decided fresh here:

**Training-only.** A threshold selected on either held-out evaluation
set would contaminate that set's use as an independent performance
estimate — the same reason hyperparameter search never touches the
test partition (Decision 4). The selection script reconstructs the
canonical stratified *training* partition only, from
`split_ids_train.csv`, and never reads `test_predictions_stratified.csv`
or `test_predictions_temporal.csv`.

**No invented cost assumptions.** `evaluate_models.py`'s own docstring
records that a cost-matrix approach using assumed unit costs was
considered and declined for the main evaluation script, since Sentinel
Credit is PD-only (Decision 4 addendum) with no real LGD/EAD basis for
a cost ratio. An expected-cost-minimizing candidate was built into the
selection script but deliberately left unconfigured
(`false_negative_cost` / `false_positive_cost` both `null` in
`config.yaml`) rather than filled with an invented number.

---

## 4. Method

`models/select_business_cutoff.py`:

1. Reconstructs the exact canonical stratified training population
   (33,283 customers, 906 defaults, 2.72% bad rate) by `customer_id`
   from `split_ids_train.csv`, applying the same population-wide
   median imputation training used.
2. Freezes the champion's already-selected XGBoost hyperparameters
   from `champion_metadata.json` — no hyperparameter re-search.
3. Generates calibrated out-of-fold probabilities via a double
   cross-validation structure: 5 outer folds partition the training
   data for OOF prediction; each outer fold's model is a
   `CalibratedClassifierCV(method="sigmoid")` with its own 5-fold
   inner split for calibration — the same calibration method the
   deployed champion uses, so the threshold is chosen on the
   probability scale the decision engine actually outputs, not on
   raw uncalibrated scores.
4. Builds one precision-recall table from the OOF probabilities and
   reads every candidate threshold off that same table.

---

## 5. Candidates

| Method | Threshold | Flagged | Precision | Recall | Bad rate if approved |
|---|---:|---:|---:|---:|---:|
| F1-optimal | 0.0868 | 7.82% | 11.02% | 31.68% | 2.02% |
| **Review-capacity (15%)** | **0.0671** | **15.00%** | **9.35%** | **51.55%** | **1.55%** |
| Youden's J | 0.0376 | 24.02% | 7.74% | 68.32% | 1.13% |
| Expected-cost-minimizing | — | — | — | — | skipped, no cost assumptions configured |

A clear monotonic pattern runs across the three computed candidates:
looser thresholds (more flagged) trade precision for recall and a
lower residual bad rate among approved customers. Each answers a
different question:

- **F1** balances precision and recall equally, with no reference to
  how many customers the business can actually process.
- **Youden's J** balances sensitivity and specificity equally — a
  statistical operating point, not an operational one.
- **Review-capacity** asks the production question directly: *given
  what the review process can handle, how much risk can the model
  capture?*

---

## 6. Decision

**`business_cutoff` is set to `0.0671`** — the review-capacity-constrained
threshold.

**Rationale.** F1 leaves most of the stated review capacity unused
(7.82% flagged against a 15% ceiling) and misses roughly two-thirds of
defaults (31.68% recall). Youden exceeds the stated capacity by more
than half again (24.02% flagged) — a defensible statistical balance
point, but not one described as within the review process's capacity.
At 0.0671, 15.00% of the training population is flagged, capturing
51.55% of observed defaults at 9.35% precision, with a 1.55% residual
bad rate among approved customers. F1 and Youden are retained in
`reports/threshold_selection/threshold_candidates.csv` as statistical
comparators — the capacity constraint was an active choice among
computed alternatives, not the only threshold generated.

**This decision is conditional.** `config.yaml`'s
`review_capacity.max_flag_rate: 0.15` is currently documented as an
illustrative placeholder — "the review team can handle at most 15% of
applications" was given as an example when this mechanism was built,
not confirmed as the real operational figure. If the actual capacity
differs, rerun `select_business_cutoff.py` with the corrected value.
The OOF probabilities and calibration do not need to be regenerated —
only the capacity constraint and the resulting candidate threshold.

---

## 7. Held-out confirmation (both splits)

`evaluate_models.py` was rerun on both splits at `business_cutoff:
0.0671`. Results were consistent with the OOF selection evidence:

| Metric | OOF (training) | Stratified (held-out) | Temporal (held-out) |
|---|---:|---:|---:|
| Recall | 51.55% | 54.87% | 38.71% |
| Precision | 9.35% | 9.60% | 8.99% |
| Flagged rate | 15.00% | 15.51% | 10.99% |

Stratified lands close to the OOF estimate — expected, since it's
drawn from the same population distribution the cutoff was
selected against. Temporal's lower recall reflects a genuinely
out-of-time population with a different score distribution, not a
selection failure.

Cutoff-independent metrics (Gini, AUC, KS, PR-AUC, ECE, Brier) were
unchanged from the training summary — expected, since none of them
depend on `business_cutoff`.

---

## 8. Operating-policy finding: thin_file_flag

The `thin_file_flag` fairness table at 0.0671 surfaced a pattern
requiring its own investigation before this cutoff could be signed
off: **100% false-negative rate for `thin_file_flag=0` on both
splits**, confirmed via full `evaluate_models.py` runs on stratified
and temporal — every defaulted customer in that segment was missed
— while `thin_file_flag=1` carried both the recall and a
substantial false-positive rate (46.53% stratified, 32.93%
temporal). `thin_file_flag` is a credit-profile segment, not itself
a protected characteristic; this is an operating-policy finding,
not a formally established disparate-impact claim.

Three checks (`models/segment_threshold_diagnostics.py`, run on
both splits) established what kind of problem this is before any
conclusion was drawn:

| Check | Finding |
|---|---|
| Score distribution among defaulters, by segment | `thin_file_flag=0` defaulters' scores topped out at 0.0344 (stratified) / 0.0367 (temporal) — roughly half the cutoff, 0% above it on either split. A hard ceiling, not a few borderline cases. |
| Threshold sweep by segment, single global threshold | No threshold recovers `thin_file_flag=0` recall without collapsing `thin_file_flag=1`'s FPR to 75-86%+. Rules out the threshold choice itself — F1 and Youden sit at or above the same collapse point. |
| Within-segment AUC, bootstrap 95% CI | `thin_file_flag=0`: 0.5661-0.7220 (stratified), 0.6148-0.7613 (temporal). Both CIs clear of 0.5, consistent across splits, overlapping substantially with `thin_file_flag=1`'s CIs. |

**Conclusion: a policy-scale problem, not a model-signal failure.**
The model discriminates risk within `thin_file_flag=0` comparably
to `thin_file_flag=1` — what collapses is the absolute probability
scale, since a rare event (1.03% bad rate) within an already
low-base-rate segment gets compressed toward zero under one shared
calibration, even for that segment's genuinely higher-risk members.
This echoes, rather than duplicates, the earlier finding that
removing `thin_file_flag` as a model input didn't close the same
gap because `credit_score` reconstructs it (r=-0.75) — the same
mechanism recurring under a different diagnostic.

**This finding does not license segment-specific thresholds.**
Ranking surviving within the segment is also not evidence that this
is "just calibration" — the segment carries real, if compressed,
risk information, and what policy to build on that remains a
separate governance decision under Decision 12.

---

## 9. Evidence locations

```text
reports/threshold_selection/xgboost_training_oof_predictions.csv
reports/threshold_selection/xgboost_oof_threshold_tradeoff.csv
reports/threshold_selection/threshold_candidates.csv
reports/threshold_selection/threshold_recommendation.json
reports/threshold_selection/segment_diagnostics/score_distribution_by_segment_stratified.csv
reports/threshold_selection/segment_diagnostics/score_distribution_by_segment_temporal.csv
reports/threshold_selection/segment_diagnostics/threshold_sweep_by_segment_stratified.csv
reports/threshold_selection/segment_diagnostics/threshold_sweep_by_segment_temporal.csv
reports/threshold_selection/segment_diagnostics/within_segment_discrimination_stratified.csv
reports/threshold_selection/segment_diagnostics/within_segment_discrimination_temporal.csv
```

## 10. Status statement

`business_cutoff: 0.0671` is confirmed on both stratified and
temporal held-out evidence — both consistent with the training-only
OOF selection. The `thin_file_flag` operating-policy finding is
complete and confirmed stable on both splits: no alternative
candidate threshold would have avoided it. A separate governance
item is opened, not resolved here: whether segment-aware review
capacity is warranted, given ranking survives within
`thin_file_flag=0` but the shared probability scale suppresses it to
zero recall in practice. That item requires explicit Decision 12
policy sign-off before any segment-differentiated treatment is
built. Decision 19 (decision log v3.22) records the confirmed cutoff
and the opened governance item.
