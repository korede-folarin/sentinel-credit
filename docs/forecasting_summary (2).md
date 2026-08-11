# Sentinel Credit - Portfolio Forecasting Summary

**Report status:** Complete for portfolio time-series forecasting; LSTM and macro stress testing explicitly deferred
**Dataset version:** `full_horizon_2025_v1`
**Forecast date:** 11 August 2026
**Decision log version:** 3.24
**Primary scripts:** `forecasting/build_portfolio_timeseries.py`, `forecasting/forecast_portfolio.py`

---

## 1. Purpose and scope

This report covers portfolio-level time-series forecasting — origination
volume and origination-cohort default rate, forecast forward using
Prophet. This is a separate view from the customer-level credit-risk
model documented elsewhere: no `MODEL_FEATURES`, no calibrated PD, no
`business_cutoff`. It answers a different question — not "is this
customer risky" but "how is the portfolio, in aggregate, behaving over
time."

**What this report is not.** The architecture names Forecasting as
"Prophet + LSTM, macro scenarios, stress testing, IFRS 9 impact." Only
the Prophet portfolio-series piece is delivered here. LSTM, macro
scenarios, stress testing, and an IFRS 9 macro overlay are explicitly
out of scope for this report — Section 6 states why, rather than
leaving the gap unexplained.

---

## 2. Data check before building anything

Before writing any forecasting code, the database was checked directly
for what data actually exists:

```text
raw.customers
raw.transactions
raw.repayments
feature_store.customer_features
```

No macro or economic data exists anywhere in this project — no
unemployment series, interest rates, inflation, or any other external
indicator. This was confirmed by inspecting `information_schema.tables`
directly, not assumed. It splits "Forecasting" into two genuinely
different pieces of work: portfolio time-series forecasting, buildable
immediately from `raw.transactions`/`raw.repayments`, and macro-scenario
stress testing / IFRS 9 macro overlay, which is blocked on a real
data-sourcing decision (synthetic macro series, real published ONS/Bank
of England data, or explicit deferral — not decided here; see Section 6).

---

## 3. Series construction

Two monthly series, built directly from raw data — independent of the
customer-level `feature_store.customer_features` table used for credit
scoring.

### 3.1 Origination volume

Transactions grouped by month of `transaction_date`, restricted to
`application_approved = 1` — a declined application generated no
exposure and is not an origination.

### 3.2 Default rate by origination cohort

**This is not a count of defaults recorded per month.** `default_flag`
and `outcome_observed` live on `raw.repayments`, at instalment grain. A
default surfacing in month N from a transaction originated in month N-4
belongs to that earlier cohort's risk, not the month it happened to
resolve. Every repayment row is joined back to its transaction to
recover the origination month before aggregating. Only transactions
where every instalment's outcome is resolved (`outcome_observed = 1`
for all rows) contribute — an origination cohort with any unresolved
instalment is excluded from that month's rate entirely, rather than
silently treated as a non-default. This mirrors the right-censoring
discipline already established for the credit-risk model (Decision
10/12), applied here at cohort level instead of customer level.

The join logic was verified against synthetic data with a known
unresolved case before running against real data: a late-surfacing
default was confirmed correctly attributed to its origination month
(not the month it was recorded), and a deliberately unresolved
transaction was confirmed correctly excluded rather than counted as a
non-default.

---

## 4. What the data actually looks like

34 monthly cohorts, January 2022 through October 2024. Zero unresolved
transactions in any cohort — consistent with Decision 16's confirmed
0% right-censoring for the current observation horizon.

### 4.1 Origination volume

| | |
|---|---:|
| January 2022 | 60 |
| October 2024 | 17,111 |
| Growth | ~285x |

Near-exponential growth throughout the full 34-month window, with no
sign of levelling off in the observed data.

### 4.2 Default rate by cohort

| | |
|---|---:|
| Range | 0.00% – 1.11% |
| Early cohorts (2022) | ~0.3% – 0.9% |
| Later cohorts (2024) | ~0.7% – 1.1% |

Noisy month to month, but with a genuine mild upward drift over the
observed period — this is the more decision-relevant of the two
series, since it reflects actual portfolio risk evolving, not a
population-generation artifact.

---

## 5. Forecast results

Prophet fit on all but the last 6 months of each series; backtested
against those 6 held-out months before any forward-looking forecast was
produced. No yearly seasonality — ~34 points is barely 2.5 annual
cycles, too little to distinguish real seasonality from noise with any
confidence.

| Series | MAE | MAPE |
|---|---:|---:|
| Default rate | 0.0016 | 18.4% |
| Origination volume | 3,041 | 24.5% |

### 5.1 Default rate

The more legitimate forecasting target. Prophet's linear trend tracks
the genuine upward drift reasonably well; the 18.4% MAPE reflects real
month-to-month noise in a small-cohort series (early months have as
few as 60 origination events), not a modelling failure.

### 5.2 Origination volume — the caveat is empirical, not just asserted

`config.yaml`'s `data_generator.origination_end_date` (2024-10-31) and
`n_customers` (50,000) are fixed generation-schedule parameters, not
market dynamics. The near-exponential growth in this series reflects
the synthetic population's onboarding schedule reaching its endpoint —
**and the backtest confirms this directly, not just by assertion**: the
final held-out months (May–October 2024) curve sharply above Prophet's
fitted linear trend, and the actual October value lands far outside
what the model predicted from the training data. The 24.5% MAPE is the
direct, visible consequence — the model cannot track an acceleration
that exists because a generation schedule is running out, not because
demand is genuinely still climbing at that rate. See
`reports/forecasting/forecast_origination_volume.png`.

**Any forecast value beyond the data's own end date (visually and
programmatically flagged as `is_beyond_data` in the output) should be
read as an extrapolation of that generation-schedule artifact, not a
business projection.** This applies more to origination volume than to
default rate.

---

## 6. Explicitly out of scope, and why

**LSTM.** Not silently omitted — ~34 monthly points is far too little
history to train a neural sequence model without it memorizing noise
rather than learning a genuine pattern. A confidently-wrong LSTM
reporting high train-set fit while generalizing to nothing would be a
worse outcome than not building it — the same reasoning already applied
to declining invented cost ratios and illustrative thresholds elsewhere
in this project (Decision 19's `select_business_cutoff.py`, for
example). Recorded as future work once meaningfully more monthly
history exists — the series construction and backtesting harness built
here would not need to be redone, only extended with an LSTM branch
alongside Prophet.

**Macro scenarios / stress testing / IFRS 9 macro overlay.** No macro
data exists in this project (Section 2). Building this requires a real
decision — synthetic macro series (fast, consistent with the rest of
this project's openly synthetic scope, but should be labelled as such
everywhere it appears), real published data (ONS/Bank of England base
rate, unemployment, CPI — matches how the customer-population generator
already cites ONS/FCA statistics as calibration sources), or explicit
further deferral. Not decided in this report.

Adding macro data later does not require rebuilding anything here:
macro series become additional regressor columns joined onto the same
monthly time-indexed dataframes already constructed
(`Prophet.add_regressor()` for the Prophet models; additional input
channels for an eventual LSTM). The construction and backtesting work
in this report is additive infrastructure for that future step, not
something that gets replaced by it.

---

## 7. Evidence locations

```text
reports/forecasting/originations_monthly.csv
reports/forecasting/default_rate_monthly.csv
reports/forecasting/forecast_default_rate.csv
reports/forecasting/forecast_default_rate.png
reports/forecasting/forecast_origination_volume.csv
reports/forecasting/forecast_origination_volume.png
reports/forecasting/forecast_metadata.json
```

## 8. Status statement

Portfolio time-series forecasting is implemented, backtested, and
verified reproducible — run twice against identical real data (once
container-relayed, once native) with matching results to four decimal
places. Default-rate forecasting is a legitimate, usable view of
evolving portfolio risk. Origination-volume forecasting works
mechanically but its caveat — that growth reflects a synthetic
generation schedule, not market demand — is now backed by direct
backtest evidence, not just a documentation note. LSTM and macro-driven
stress testing remain explicit, scoped future work: gated on more
historical data and a real macro-data-sourcing decision respectively,
neither fabricated to appear complete today.
