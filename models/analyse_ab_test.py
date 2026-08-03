"""
=============================================================
Sentinel Credit — A/B Test Analysis
=============================================================
File:    analyse_ab_test.py
Purpose: Formal analysis of the affordability-threshold
         experiment defined in Decision 1: does relaxing the
         affordability threshold from 0.45 (A_strict) to 0.60
         (B_relaxed) produce a measurable increase in default
         rate?
Author:  Folarin Korede
Version: 1.0.0

EXPERIMENT DESIGN (config.yaml ab_test block)
    group_A_size: 0.50          strict_threshold:  0.45
    group_B_size: 0.50          relaxed_threshold: 0.60
    group_B_hazard_multiplier: 1.60

Assignment is at customer level, fixed at generation time,
independent of any customer attribute — so the two arms are
comparable by construction and no covariate adjustment is
required. The multiplier acts on the MISS HAZARD, not on a
pre-assigned default probability, because default is emergent
in the hazard state machine (Decision 10). The observed uplift
is therefore a downstream consequence of the design parameter,
not a direct restatement of it.

WHAT THIS TESTS
Primary: difference in default rate between arms, on the
modelable population (has_outcome = 1). Two-proportion
z-test, plus a Wald 95% confidence interval on the absolute
difference and the relative risk.

Secondary: approval rate and average exposure, to confirm the
threshold change did what it was supposed to do upstream of
the outcome — a default-rate difference means little if the
approval behaviour did not actually differ.

Also reports achieved power against the observed effect, and
the minimum detectable effect at the realised sample sizes,
so the result can be read against the 80%/5% design target
rather than assumed to have met it.

SCOPE
Analysed on the modelable population only. Customers without
an observed outcome cannot contribute to a default-rate
comparison. This inherits the observation-window caveat from
Decision 12: both arms are subject to the same maturity
requirement, so the comparison is internally consistent, but
the absolute rates are conditional on maturity rather than
portfolio-wide.
=============================================================
"""

import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import yaml
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_features import FILTER_COLUMN, FILTER_VALUE, TARGET_COLUMN

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "ab_test")

with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DB = CONFIG["database"]
AB_CFG = CONFIG["data_generator"]["ab_test"]

ALPHA = 0.05
TARGET_POWER = 0.80

# Fully mature cohort definition. The longest product tenor is
# 12 instalments at 30-day intervals, so the final instalment
# falls due around day 330. Default is confirmed at 90 DPD, so
# a loan needs roughly 420 days from origination before its
# outcome is guaranteed knowable regardless of what happened
# during it. Customers originating on or before
# (data_end_date - 420 days) therefore had an equal
# opportunity to produce an observable outcome whichever arm
# they were assigned to.
GEN_CFG = CONFIG["data_generator"]
DATA_END_DATE = pd.Timestamp(GEN_CFG["data_end_date"])
MAX_TENOR_DAYS = 330      # 12 instalments x 30 days
DEFAULT_WINDOW_DAYS = 90  # CRR/EBA 90 DPD
FULL_MATURITY_DAYS = MAX_TENOR_DAYS + DEFAULT_WINDOW_DAYS
MATURE_COHORT_CUTOFF = DATA_END_DATE - pd.Timedelta(days=FULL_MATURITY_DAYS)


def load_data() -> pd.DataFrame:
    password = os.environ.get("SENTINEL_DB_PASSWORD", "")
    conn = psycopg2.connect(
        host=DB["host"], port=DB["port"], dbname=DB["name"],
        user="postgres", password=password,
    )
    try:
        df = pd.read_sql(
            f"SELECT * FROM {DB['schema']}.customer_features", conn
        )
    finally:
        conn.close()
    return df


# ── ASSIGNMENT VALIDITY ───────────────────────────────────
def check_assignment_balance(df_all: pd.DataFrame) -> dict:
    """
    Confirms the randomisation actually produced balanced arms
    before any outcome is compared. A significant imbalance in
    arm SIZE would indicate the assignment itself is faulty,
    which would invalidate everything downstream — this runs
    first for that reason.

    Also compares pre-treatment attributes across arms. These
    should NOT differ: assignment is independent of customer
    characteristics by design, so a large difference here would
    signal a bug in the generator rather than a real effect.
    """
    counts = df_all["ab_test_group"].value_counts()
    n_a = int(counts.get("A_strict", 0))
    n_b = int(counts.get("B_relaxed", 0))
    n_total = n_a + n_b

    # Chi-square against the designed 50/50 split.
    expected = [n_total * AB_CFG["group_A_size"], n_total * AB_CFG["group_B_size"]]
    chi2, p_balance = stats.chisquare([n_a, n_b], f_exp=expected)

    # Pre-treatment covariate balance. Assignment precedes any
    # treatment effect, so these should be indistinguishable.
    covariates = ["age", "credit_score", "monthly_income_gbp",
                  "savings_balance_gbp", "thin_file_flag"]
    covariate_checks = []
    for c in covariates:
        if c not in df_all.columns:
            continue
        a_vals = df_all[df_all["ab_test_group"] == "A_strict"][c].dropna()
        b_vals = df_all[df_all["ab_test_group"] == "B_relaxed"][c].dropna()
        t_stat, p_val = stats.ttest_ind(a_vals, b_vals, equal_var=False)
        smd = standardised_mean_difference(a_vals, b_vals)
        covariate_checks.append({
            "covariate": c,
            "mean_A": round(float(a_vals.mean()), 4),
            "mean_B": round(float(b_vals.mean()), 4),
            "smd": round(smd, 4),
            "p_value": round(float(p_val), 4),
            "balanced": bool(abs(smd) < 0.1),
        })

    return {
        "n_A": n_a, "n_B": n_b,
        "pct_A": round(100 * n_a / n_total, 2),
        "pct_B": round(100 * n_b / n_total, 2),
        "size_balance_p": round(float(p_balance), 4),
        "size_balanced": bool(p_balance > ALPHA),
        "covariate_balance": covariate_checks,
    }


def check_outcome_observability(df_all: pd.DataFrame) -> dict:
    """
    Reports has_outcome rates by arm — the diagnostic that
    determines whether the modelable population can be used as
    a primary analysis population at all.

    has_outcome is a POST-TREATMENT variable. Group B's hazard
    multiplier changes repayment behaviour, which changes
    default timing, which changes whether an outcome has become
    observable by data_end_date. A fast-defaulting B customer
    becomes observable; a slow-maturing B non-defaulter does
    not. The maturity RULE is identical across arms, but its
    EFFECT need not be, because treatment alters the outcome
    process itself.

    If these rates differ materially, filtering on has_outcome
    compares two selected post-treatment populations rather
    than preserving the original randomisation, and the primary
    test must run on a fixed pre-treatment cohort instead.
    """
    rows = []
    for arm in ["A_strict", "B_relaxed"]:
        sub = df_all[df_all["ab_test_group"] == arm]
        n_total = len(sub)
        n_obs = int((sub[FILTER_COLUMN] == FILTER_VALUE).sum())
        rows.append({
            "arm": arm,
            "n_total": n_total,
            "n_has_outcome": n_obs,
            "observability_rate": round(n_obs / max(n_total, 1), 4),
        })

    a, b = rows[0], rows[1]
    diff = b["observability_rate"] - a["observability_rate"]

    # Two-proportion test on observability itself.
    p_pool = (a["n_has_outcome"] + b["n_has_outcome"]) / (a["n_total"] + b["n_total"])
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / a["n_total"] + 1 / b["n_total"]))
    z = diff / se if se > 0 else 0.0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "by_arm": rows,
        "difference": round(diff, 4),
        "p_value": float(f"{p_val:.6g}"),
        "differs_significantly": bool(p_val < ALPHA),
        "note": (
            "If observability differs by arm, has_outcome is affected by "
            "treatment and cannot be used as a filter for the primary "
            "causal test without breaking randomisation."
        ),
    }


def standardised_mean_difference(a_vals: pd.Series, b_vals: pd.Series) -> float:
    """
    SMD is the conventional balance metric for randomised
    comparisons because, unlike a p-value, it does not depend
    on sample size — with 50,000 customers a trivial difference
    can be 'significant' while being substantively irrelevant.
    Convention: |SMD| < 0.1 indicates adequate balance.
    """
    m_a, m_b = a_vals.mean(), b_vals.mean()
    s_a, s_b = a_vals.std(), b_vals.std()
    pooled_sd = np.sqrt((s_a ** 2 + s_b ** 2) / 2)
    return float((m_b - m_a) / pooled_sd) if pooled_sd > 0 else 0.0


def build_mature_cohort(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    The primary population: customers for whom EVERY loan had
    time to reach an observable outcome regardless of treatment.

    ANCHORING ON THE LAST TRANSACTION, NOT THE FIRST.
    target_default is MAX(default_flag) across all of a
    customer's repayments, so the outcome depends on their most
    recent loan, not their earliest. A customer who originated
    in early 2023 but took a further loan in late 2024 has an
    immature outcome even though their first transaction sits
    well inside any first-date cutoff. Anchoring on
    first_transaction_date would silently admit those
    customers. The rule is therefore:

        last_transaction_date <= data_end_date - 420 days

    THE TRADEOFF THIS CREATES, STATED RATHER THAN HIDDEN.
    last_transaction_date is itself partly post-treatment:
    treatment changes approval behaviour, which can change how
    many loans a customer takes and when the last one falls.
    So this cohort is not perfectly pre-treatment either — it
    trades one form of selection for a smaller one.

    Neither anchor is fully clean given the current label
    definition. The genuinely correct fix is a FIXED
    OBSERVATION HORIZON — define the target as "reached 90 DPD
    within N months of first origination", which decouples
    outcome maturity from how many loans a customer happened to
    take. That is logged as future work (Decision 10 addendum)
    and is not implemented here.

    Of the available options this is the most defensible:
    conditioning on has_outcome selects directly on the outcome
    process (fast defaulters become observable, slow
    non-defaulters do not), whereas conditioning on transaction
    timing selects on exposure behaviour, which is one causal
    step further from the target. The observability diagnostic
    above reports how much this matters in practice.
    """
    df = df_all.copy()
    df["last_transaction_date"] = pd.to_datetime(df["last_transaction_date"])

    # Customers with no approved transactions have NaT here and
    # no outcome to analyse — excluded, and this exclusion is
    # pre-treatment-neutral in the sense that it removes
    # customers from both arms who never entered the outcome
    # process at all.
    return df[
        df["last_transaction_date"].notna()
        & (df["last_transaction_date"] <= MATURE_COHORT_CUTOFF)
    ].copy()


# ── UPSTREAM EFFECT ───────────────────────────────────────
def check_treatment_delivered(df_all: pd.DataFrame) -> dict:
    """
    Confirms the threshold change actually altered approval
    behaviour. If approval rates are identical between arms,
    the treatment was never delivered and any outcome
    difference would be spurious.
    """
    rows = []
    for arm in ["A_strict", "B_relaxed"]:
        sub = df_all[df_all["ab_test_group"] == arm]
        rows.append({
            "arm": arm,
            "n_customers": int(len(sub)),
            "mean_n_applications": round(float(sub["n_applications"].mean()), 3),
            "mean_n_approved": round(float(sub["n_approved"].mean()), 3),
            "approval_rate": round(
                float(sub["n_approved"].sum() / max(sub["n_applications"].sum(), 1)), 4
            ),
            "mean_max_affordability_ratio": round(
                float(sub["max_affordability_ratio"].mean()), 4
            ),
        })
    return {"by_arm": rows}


# ── PRIMARY TEST ──────────────────────────────────────────
def two_proportion_test(n_a: int, x_a: int, n_b: int, x_b: int) -> dict:
    """
    Two-proportion z-test on default rates, with a Wald 95%
    confidence interval on the absolute difference and the
    relative risk.

    The CI matters more than the p-value here: a p-value says
    whether an effect is distinguishable from zero, while the
    interval says how large the effect plausibly is — which is
    the question a lending policy decision actually turns on.
    """
    p_a, p_b = x_a / n_a, x_b / n_b
    diff = p_b - p_a

    # Pooled proportion for the test statistic.
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Unpooled SE for the interval (standard for a CI on the
    # difference, since it does not assume the null).
    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - ALPHA / 2)
    ci_low = diff - z_crit * se_unpooled
    ci_high = diff + z_crit * se_unpooled

    relative_risk = p_b / p_a if p_a > 0 else float("nan")

    return {
        "rate_A": round(p_a, 6), "rate_B": round(p_b, 6),
        "n_A": n_a, "n_B": n_b, "defaults_A": x_a, "defaults_B": x_b,
        "absolute_difference": round(diff, 6),
        "ci_95_low": round(ci_low, 6),
        "ci_95_high": round(ci_high, 6),
        "relative_risk": round(relative_risk, 4),
        "z_statistic": round(float(z), 4),
        "p_value": float(f"{p_value:.6g}"),
        "significant_at_5pct": bool(p_value < ALPHA),
        "wording_note": (
            "Report as 'the results suggest', not 'we proved'. This is a "
            "sample. The confidence interval, not the point estimate, is "
            "the honest statement of the effect size the data supports."
        ),
    }


def power_analysis(n_a: int, p_a: float, n_b: int, p_b: float) -> dict:
    """
    Achieved power against the OBSERVED effect, and the minimum
    detectable effect at the realised sample sizes.

    Reported because the experiment was designed for 80% power
    at 5% significance, but the realised modelable sample is
    smaller than the generated population — the observation
    window requirement (Decision 12) removes customers whose
    outcomes have not matured. Whether the design target was
    actually met is therefore a question to answer, not assume.
    """
    p_pool = (p_a * n_a + p_b * n_b) / (n_a + n_b)
    se_null = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    se_alt = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - ALPHA / 2)

    effect = abs(p_b - p_a)
    achieved_power = (
        1 - stats.norm.cdf((z_crit * se_null - effect) / se_alt)
        + stats.norm.cdf((-z_crit * se_null - effect) / se_alt)
    ) if se_alt > 0 else float("nan")

    # Minimum detectable effect at target power.
    z_power = stats.norm.ppf(TARGET_POWER)
    mde = (z_crit + z_power) * se_null

    return {
        "observed_absolute_effect": round(effect, 6),
        "achieved_power": round(float(achieved_power), 4),
        "target_power": TARGET_POWER,
        "power_target_met": bool(achieved_power >= TARGET_POWER),
        "minimum_detectable_effect": round(float(mde), 6),
        "observed_effect_exceeds_mde": bool(effect > mde),
    }


# ── BUSINESS IMPACT ───────────────────────────────────────
def business_impact(df_cohort: pd.DataFrame) -> dict:
    """
    Translates the statistical result into the decision the
    business actually has to make.

    A significant default-rate increase does not by itself mean
    the relaxed threshold is wrong. Relaxing affordability
    approves MORE customers, generating more revenue as well as
    more losses. The policy question is whether the additional
    approvals are worth the additional defaults — a 2.5pp
    uplift means something very different if it costs thousands
    versus millions.

    POPULATION: computed entirely on the fully mature cohort,
    the same population as the primary test. An earlier version
    took exposure from the full population and defaulted
    exposure from the modelable subset, making the ratio
    between them meaningless — numerator and denominator drawn
    from different groups. Both now come from one population
    where every customer had an equal opportunity to produce an
    observable outcome.

    SCOPE LIMIT: this platform models PD only. LGD and EAD are
    explicitly out of scope (Decision 4 addendum), so a true
    monetary Expected Loss cannot be computed. What follows is
    an EXPOSURE-BASED comparison — it quantifies the volume
    tradeoff, not profit. Converting it requires a margin
    assumption and a loss-given-default assumption, neither of
    which this project has established.
    """
    rows = []
    for arm in ["A_strict", "B_relaxed"]:
        sub = df_cohort[df_cohort["ab_test_group"] == arm]
        defaulted = sub[sub[TARGET_COLUMN] == 1]

        total_exposure = float(sub["total_order_value_gbp"].sum())
        defaulted_exposure = float(defaulted["total_order_value_gbp"].sum())

        rows.append({
            "arm": arm,
            "n_customers": int(len(sub)),
            "n_approved_transactions": int(sub["n_approved"].sum()),
            "total_exposure_gbp": round(total_exposure, 2),
            "n_defaulters": int(len(defaulted)),
            "defaulted_exposure_gbp": round(defaulted_exposure, 2),
            "defaulted_exposure_pct": round(
                100 * defaulted_exposure / max(total_exposure, 1), 3
            ),
        })

    a, b = rows[0], rows[1]
    extra_approvals = b["n_approved_transactions"] - a["n_approved_transactions"]
    extra_exposure = b["total_exposure_gbp"] - a["total_exposure_gbp"]
    extra_defaulted = b["defaulted_exposure_gbp"] - a["defaulted_exposure_gbp"]

    return {
        "population": "fully_mature_cohort",
        "by_arm": rows,
        "extra_approved_transactions_B_vs_A": extra_approvals,
        "extra_exposure_gbp_B_vs_A": round(extra_exposure, 2),
        "extra_defaulted_exposure_gbp_B_vs_A": round(extra_defaulted, 2),
        "exposure_gained_per_pound_defaulted": (
            round(extra_exposure / extra_defaulted, 2)
            if extra_defaulted > 0 else None
        ),
        "interpretation_note": (
            "Exposure-based, not profit-based, and computed on the fully "
            "mature cohort so numerator and denominator come from the same "
            "population. Converting to a profit position requires a margin "
            "per pound lent and a loss-given-default rate, neither modelled "
            "here (PD-only scope, Decision 4 addendum). This establishes the "
            "SIZE of the tradeoff, not whether it is commercially worth "
            "taking."
        ),
    }


def main() -> None:
    print("=" * 62)
    print("SENTINEL CREDIT — A/B TEST ANALYSIS")
    print("=" * 62)
    print(f"Design: A_strict threshold {AB_CFG['strict_threshold']}, "
          f"B_relaxed {AB_CFG['relaxed_threshold']}")
    print(f"        group_B_hazard_multiplier {AB_CFG['group_B_hazard_multiplier']}")
    print(f"        designed split {AB_CFG['group_A_size']}/{AB_CFG['group_B_size']}, "
          f"powered for {TARGET_POWER:.0%} at {ALPHA:.0%} significance")

    os.makedirs(OUT_DIR, exist_ok=True)

    df_all = load_data()
    df = df_all[df_all[FILTER_COLUMN] == FILTER_VALUE].copy()
    print(f"\nFull population {len(df_all):,}, "
          f"modelable {len(df):,} ({100*len(df)/len(df_all):.1f}%)")

    # ---- 1. Assignment validity ----
    print("\n" + "-" * 62)
    print("1. ASSIGNMENT VALIDITY (full population)")
    print("-" * 62)
    balance = check_assignment_balance(df_all)
    print(f"A_strict {balance['n_A']:,} ({balance['pct_A']}%)  "
          f"B_relaxed {balance['n_B']:,} ({balance['pct_B']}%)")
    print(f"Size balance vs designed split: p = {balance['size_balance_p']} "
          f"({'balanced' if balance['size_balanced'] else 'IMBALANCED'})")
    print("\nPre-treatment covariate balance (should NOT differ):")
    cov_df = pd.DataFrame(balance["covariate_balance"])
    print(cov_df.to_string(index=False))
    if not cov_df["balanced"].all():
        print("\n  WARNING: a pre-treatment covariate differs between arms.")
        print("  Assignment is designed to be independent of customer")
        print("  attributes, so this would indicate a generator bug, not")
        print("  a treatment effect. Investigate before trusting the result.")

    # ---- 1b. Outcome observability (the causal gate) ----
    print("\n" + "-" * 62)
    print("1b. OUTCOME OBSERVABILITY BY ARM (causal gate)")
    print("-" * 62)
    obs = check_outcome_observability(df_all)
    print(pd.DataFrame(obs["by_arm"]).to_string(index=False))
    print(f"\nDifference: {obs['difference']:+.2%}  p = {obs['p_value']}")
    if obs["differs_significantly"]:
        print("\n  Observability DIFFERS by arm. has_outcome is affected by")
        print("  treatment (hazard -> default timing -> observability), so")
        print("  filtering on it would compare two selected post-treatment")
        print("  populations rather than the original randomisation.")
        print("  The primary test below therefore runs on a fixed")
        print("  pre-treatment cohort instead.")
    else:
        print("\n  Observability is comparable across arms. The mature-cohort")
        print("  primary test below is used regardless, since it does not")
        print("  depend on this holding.")

    # ---- Mature cohort (primary population) ----
    cohort = build_mature_cohort(df_all)
    print("\n" + "-" * 62)
    print("PRIMARY POPULATION — fully mature cohort")
    print("-" * 62)
    print(f"Rule: LAST transaction on or before {MATURE_COHORT_CUTOFF.date()}")
    print(f"  = data_end_date ({DATA_END_DATE.date()}) minus "
          f"{FULL_MATURITY_DAYS} days")
    print(f"  = longest tenor ({MAX_TENOR_DAYS}d) + default window "
          f"({DEFAULT_WINDOW_DAYS}d)")
    print(f"Cohort size: {len(cohort):,} of {len(df_all):,} "
          f"({100*len(cohort)/len(df_all):.1f}%)")
    print("\n  Anchored on LAST transaction, not first: target_default is")
    print("  MAX(default_flag) across all repayments, so the outcome")
    print("  depends on the most recent loan. A first-date anchor would")
    print("  admit customers whose later loans had not matured.")
    print("\n  CAVEAT: last_transaction_date is itself partly")
    print("  post-treatment — treatment affects approvals, which affects")
    print("  how many loans a customer takes and when the last falls. This")
    print("  trades a large selection problem for a smaller one. The clean")
    print("  fix is a fixed observation horizon in the label definition")
    print("  (Decision 10 addendum, not implemented).")

    # ---- 2. Treatment delivery ----
    print("\n" + "-" * 62)
    print("2. WAS THE TREATMENT DELIVERED? (upstream check)")
    print("-" * 62)
    delivery = check_treatment_delivered(df_all)
    print(pd.DataFrame(delivery["by_arm"]).to_string(index=False))
    print("\n  A relaxed affordability threshold should approve MORE")
    print("  applications and admit HIGHER affordability ratios. If these")
    print("  match between arms, the outcome difference below is spurious.")

    # ---- 3. Primary test ----
    print("\n" + "-" * 62)
    print("3. PRIMARY TEST — default rate (fully mature cohort)")
    print("-" * 62)
    a = cohort[cohort["ab_test_group"] == "A_strict"]
    b = cohort[cohort["ab_test_group"] == "B_relaxed"]
    result = two_proportion_test(
        len(a), int(a[TARGET_COLUMN].sum()),
        len(b), int(b[TARGET_COLUMN].sum()),
    )

    print(f"A_strict:   {result['defaults_A']:,} / {result['n_A']:,} "
          f"= {result['rate_A']:.2%}")
    print(f"B_relaxed:  {result['defaults_B']:,} / {result['n_B']:,} "
          f"= {result['rate_B']:.2%}")
    print(f"\nAbsolute difference: {result['absolute_difference']:+.2%}")
    print(f"95% CI:              [{result['ci_95_low']:+.2%}, "
          f"{result['ci_95_high']:+.2%}]")
    print(f"Relative risk:       {result['relative_risk']:.3f}x")
    print(f"z = {result['z_statistic']}, p = {result['p_value']}")
    print(f"Significant at {ALPHA:.0%}: {result['significant_at_5pct']}")
    print("\n  Wording: this is a sample, not the full population. The")
    print("  results SUGGEST the relaxed threshold is associated with a")
    print("  higher default rate within the stated interval. They do not")
    print("  prove a fixed effect size — the CI is the honest statement of")
    print("  what the data supports, not the point estimate alone.")

    # ---- 4. Power ----
    print("\n" + "-" * 62)
    print("4. POWER")
    print("-" * 62)
    power = power_analysis(
        result["n_A"], result["rate_A"], result["n_B"], result["rate_B"]
    )
    print(f"Observed effect:            {power['observed_absolute_effect']:.2%}")
    print(f"Minimum detectable effect:  {power['minimum_detectable_effect']:.2%} "
          f"(at {TARGET_POWER:.0%} power)")
    print(f"Achieved power:             {power['achieved_power']:.2%} "
          f"(target {TARGET_POWER:.0%}, "
          f"{'MET' if power['power_target_met'] else 'NOT MET'})")

    # ---- 5. Business impact ----
    print("\n" + "-" * 62)
    print("5. BUSINESS IMPACT — the tradeoff, not just the difference")
    print("-" * 62)
    impact = business_impact(cohort)
    print(pd.DataFrame(impact["by_arm"]).to_string(index=False))
    print(f"\nB_relaxed vs A_strict:")
    print(f"  Extra approved transactions:  "
          f"{impact['extra_approved_transactions_B_vs_A']:+,}")
    print(f"  Extra exposure written:       "
          f"GBP {impact['extra_exposure_gbp_B_vs_A']:+,.2f}")
    print(f"  Extra defaulted exposure:     "
          f"GBP {impact['extra_defaulted_exposure_gbp_B_vs_A']:+,.2f}")
    if impact["exposure_gained_per_pound_defaulted"] is not None:
        print(f"  Exposure gained per GBP defaulted: "
              f"{impact['exposure_gained_per_pound_defaulted']:.2f}")
    print("\n  This is EXPOSURE, not profit. Converting it to a commercial")
    print("  answer needs a margin rate and an LGD assumption, neither of")
    print("  which this platform models (PD-only, Decision 4 addendum).")
    print("  The statistical result says the default rate rose. This says")
    print("  how large a tradeoff the business would be accepting.")

    # ---- Save ----
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "design": dict(AB_CFG),
        "alpha": ALPHA,
        "target_power": TARGET_POWER,
        "population": {
            "full": int(len(df_all)),
            "modelable": int(len(df)),
            "modelable_pct": round(100 * len(df) / len(df_all), 2),
        },
        "assignment_validity": balance,
        "outcome_observability": obs,
        "mature_cohort": {
            "anchor": "last_transaction_date",
            "anchor_rationale": (
                "target_default is MAX(default_flag) across all repayments, "
                "so outcome maturity depends on the customer's LAST loan, "
                "not their first. A first-date anchor would admit customers "
                "whose later loans had not matured."
            ),
            "residual_limitation": (
                "last_transaction_date is partly post-treatment: treatment "
                "affects approvals, which affects loan count and timing. "
                "This is a smaller selection problem than conditioning on "
                "has_outcome (which selects directly on the outcome "
                "process), but it is not zero. The clean fix is a fixed "
                "observation horizon in the label definition — see "
                "Decision 10 addendum, not implemented."
            ),
            "cutoff_date": str(MATURE_COHORT_CUTOFF.date()),
            "full_maturity_days": FULL_MATURITY_DAYS,
            "max_tenor_days": MAX_TENOR_DAYS,
            "default_window_days": DEFAULT_WINDOW_DAYS,
            "n_cohort": int(len(cohort)),
            "pct_of_population": round(100*len(cohort)/len(df_all), 2),
        },
        "treatment_delivery": delivery,
        "primary_test": result,
        "power": power,
        "business_impact": impact,
        "scope_note": (
            "Primary test runs on a FULLY MATURE COHORT defined by "
            "origination date and generator parameters only — all "
            "pre-treatment. has_outcome is deliberately NOT used as the "
            "primary filter: it is a post-treatment variable, since the "
            "group B hazard multiplier changes default timing and therefore "
            "changes whether an outcome has become observable. Conditioning "
            "on it would compare two selected post-treatment populations "
            "rather than preserving randomisation. Business impact is "
            "computed on the same cohort so its numerator and denominator "
            "are drawn from one population."
        ),
    }
    with open(os.path.join(OUT_DIR, "ab_test_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    pd.DataFrame([result]).to_csv(
        os.path.join(OUT_DIR, "ab_primary_test.csv"), index=False)
    cov_df.to_csv(
        os.path.join(OUT_DIR, "ab_covariate_balance.csv"), index=False)
    pd.DataFrame(delivery["by_arm"]).to_csv(
        os.path.join(OUT_DIR, "ab_treatment_delivery.csv"), index=False)
    pd.DataFrame(impact["by_arm"]).to_csv(
        os.path.join(OUT_DIR, "ab_business_impact.csv"), index=False)

    print("\n" + "=" * 62)
    print(f"Saved to {OUT_DIR}")
    print("  ab_test_results.json, ab_primary_test.csv,")
    print("  ab_covariate_balance.csv, ab_treatment_delivery.csv,")
    print("  ab_business_impact.csv")
    print("=" * 62)


if __name__ == "__main__":
    main()
