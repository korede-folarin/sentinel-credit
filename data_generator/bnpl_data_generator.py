"""
=============================================================
Sentinel Credit — BNPL Risk Intelligence Platform
=============================================================
File:    bnpl_data_generator.py
Purpose: Synthetic BNPL customer, transaction, and repayment
         data generator for credit risk modelling
Author:  Folarin Korede
Version: 1.0.0

Generation method:
  Parametric domain-expert generator calibrated against
  published UK regulatory and industry statistics.
  See docs/engineering_decision_log.md — Decision 1.

Regulatory alignment:
  FCA Consumer Duty PS22/9 — vulnerability fields
  FCA CONC 5.2A — affordability_ratio and 0.45 threshold
  See docs/engineering_decision_log.md for full mapping.

Calibration sources:
  Klarna Wikipink UK 2024 · Klarna Q2 2025 Investor Report
  FCA Financial Lives Survey May 2022 · ONS FYE 2023
  PwC/TotallyMoney 2022 · World Bank March 2025
  See docs/engineering_decision_log.md — Appendix A.
=============================================================
"""

# ── IMPORTS ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import hashlib
import os
import json
import yaml
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


# ── CONFIGURATION ─────────────────────────────────────────
def load_config() -> dict:
    """Load project configuration from config.yaml."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG      = load_config()
GEN         = CONFIG["data_generator"]
RANDOM_SEED = GEN["random_seed"]

# ── DATE WINDOWS ──────────────────────────────────────────
# Origination: the period in which applications can be made.
# Performance: how long repayment behaviour is then observed.
# Transactions originated late in the window remain right
# censored at the performance cutoff, as in a live portfolio.
START_DATE           = pd.Timestamp(GEN["start_date"])
ORIGINATION_END_DATE = pd.Timestamp(GEN["origination_end_date"])
DATA_END_DATE        = pd.Timestamp(GEN["data_end_date"])

np.random.seed(RANDOM_SEED)


# ── HELPER FUNCTIONS ──────────────────────────────────────
def generate_customer_id(index: int) -> str:
    """
    Pseudonymised customer ID. Hashing serves GDPR
    pseudonymisation; the index guarantees uniqueness.
    Widened to 16 hex chars so the truncated digest cannot
    collide across the population.
    """
    raw = f"BNPL-CUST-{index:06d}-{RANDOM_SEED}"
    return "C" + hashlib.md5(raw.encode()).hexdigest()[:16].upper()


def income_band_to_numeric(band: str) -> tuple:
    """
    Converts an income band string to a (min, max, median)
    tuple of numeric values in GBP.

    Why: config.yaml stores income bands as human-readable
    strings like 'Under £15k'. The generator needs actual
    numbers to draw realistic income values from.

    Band boundaries sourced from ONS Household Income
    Inequality UK FYE 2023 quintile data.
    Median values are design assumptions informed by
    ONS age-income distribution data.
    The £200,000 ceiling for Over £75k is a practical
    design assumption — no single source defines this.

    Args:
        band: Income band string from config.yaml

    Returns:
        Tuple of (minimum, median, maximum) in GBP
    """
    mapping = {
        "Under £15k":  (8_000,   12_000,  15_000),
        "£15k-£25k":   (15_000,  20_000,  25_000),
        "£25k-£35k":   (25_000,  30_000,  35_000),
        "£35k-£50k":   (35_000,  42_000,  50_000),
        "£50k-£75k":   (50_000,  61_000,  75_000),
        "Over £75k":   (75_000,  95_000, 200_000),
    }
    return mapping[band]


def compute_affordability_ratio(
    monthly_income: float,
    monthly_obligations: float,
    order_value: float,
    n_instalments: int
) -> float:
    """
    Calculates the FCA CONC 5.2A affordability ratio.

    Formula:
        (existing obligations + new instalment) / monthly income

    Why: FCA CONC 5.2A requires lenders to assess whether
    a customer can afford new credit without financial
    difficulty.

    The 0.45 threshold used in Group A is a design
    assumption derived from the population characteristics
    in our synthetic dataset. It is not a number explicitly
    stated in FCA CONC 5.2A. See config.yaml ab_test section
    and docs/engineering_decision_log.md.

    Args:
        monthly_income:       Net monthly income in GBP
        monthly_obligations:  Existing monthly debt payments
        order_value:          Total BNPL purchase amount
        n_instalments:        Number of repayment instalments

    Returns:
        Affordability ratio rounded to 4 decimal places
    """
    instalment = order_value / n_instalments
    ratio = (monthly_obligations + instalment) / max(monthly_income, 1)
    return round(ratio, 4)


def vulnerability_flags(rng: np.random.RandomState) -> dict:
    """
    Generates FCA Consumer Duty vulnerability flags.
    Four drivers: health, life events, resilience, capability.
    See docs/engineering_decision_log.md Decision 1.

    IMPORTANT: mental_health_flag must NEVER be used as
    a direct model input — violates Equality Act 2010.
    """
    health     = int(rng.random() < GEN["mental_health_prevalence"])
    life_event = int(rng.random() < 0.15)
    resilience = int(rng.random() < GEN["vulnerability_prevalence"])
    capability = int(rng.random() < 0.20)

    # Composite score for Consumer Duty treatment only.
    # Health included here — NOT in model features.
    score = health + life_event + resilience + capability

    return {
        "mental_health_flag":       health,
        "vulnerability_health":     health,
        "vulnerability_life_event": life_event,
        "vulnerability_resilience": resilience,
        "vulnerability_capability": capability,
        "vulnerability_score":      score,
        "is_vulnerable":            int(score > 0),
    }


# ── CUSTOMER GENERATOR ────────────────────────────────────
def generate_customers() -> pd.DataFrame:
    """
    Generates the synthetic BNPL customer population.
    Population characteristics reference published UK figures.
    See docs/engineering_decision_log.md — Decision 1.
    """
    n = GEN["n_customers"]
    rng = np.random.RandomState(RANDOM_SEED)

    # ── Age ───────────────────────────────────────────────
    age_brackets = list(GEN["age_distribution"].keys())
    age_weights  = list(GEN["age_distribution"].values())
    age_ranges   = {k: tuple(v) for k, v in GEN["age_ranges"].items()}

    chosen_brackets = rng.choice(age_brackets, size=n, p=age_weights)
    ages = np.array([
        rng.randint(age_ranges[b][0], age_ranges[b][1] + 1)
        for b in chosen_brackets
    ])

    # ── Income ────────────────────────────────────────────
    income_brackets = list(GEN["income_bands"].keys())
    income_weights  = list(GEN["income_bands"].values())
    chosen_income   = rng.choice(
        income_brackets, size=n, p=income_weights
    )

    annual_incomes = np.array([
        rng.triangular(*income_band_to_numeric(b))
        for b in chosen_income
    ])
    monthly_incomes = annual_incomes / 12

    # ── Employment ───────────────────────────────────────
    # Distribution references ONS labour market data, with the
    # student and retired split a design assumption reflecting
    # the younger BNPL age profile.
    employment_options = [
        "Employed FT", "Employed PT", "Self-Employed",
        "Student", "Unemployed", "Retired"
    ]
    employment_weights = [0.52, 0.14, 0.097, 0.12, 0.073, 0.05]
    employment = rng.choice(
        employment_options, size=n, p=employment_weights
    )

    # ── Gender ────────────────────────────────────────────
    # References ONS population estimates. Non-binary and
    # prefer-not-to-say shares are design assumptions.
    gender = rng.choice(
        ["Female", "Male", "Non-binary", "Prefer not to say"],
        size=n, p=[0.51, 0.45, 0.02, 0.02]
    )

    # ── Thin file ─────────────────────────────────────────
    thin_file = (
        rng.random(n) < GEN["thin_file_prevalence"]
    ).astype(int)

    # ── Credit file characteristics ──────────────────────
    # Thin file customers score lower and hold shorter,
    # sparser credit histories.
    credit_scores = np.where(
        thin_file == 1,
        rng.randint(300, 580, n),
        rng.randint(500, 999, n)
    )

    months_history = np.where(
        thin_file == 1,
        rng.randint(0, 24, n),
        rng.randint(12, 180, n)
    )

    num_accounts = np.where(
        thin_file == 1,
        rng.randint(0, 2, n),
        rng.randint(1, 8, n)
    )

    num_defaults_24m = np.where(
        credit_scores < 550,
        rng.randint(0, 3, n),
        rng.randint(0, 1, n)
    )

    # ── Existing obligations ─────────────────────────────
    # Beta(2,5) because obligations are a proportion of income
    # and Beta is defined for proportions. Right skewed: most
    # customers carry moderate burdens, a minority carry heavy.
    obligation_ratio = rng.beta(2, 5, n)
    monthly_obligations = monthly_incomes * obligation_ratio

    # ── Savings ──────────────────────────────────────────
    savings = np.where(
        thin_file == 1,
        rng.triangular(0, 200, 1000, n),
        rng.triangular(0, 1500, 10000, n)
    )

    # ── Vulnerability ────────────────────────────────────
    vuln_records = [
        vulnerability_flags(np.random.RandomState(RANDOM_SEED + i))
        for i in range(n)
    ]

    # ── Alternative data ─────────────────────────────────
    device_types = rng.choice(
        ["Mobile", "Desktop", "Tablet"],
        size=n, p=[0.65, 0.25, 0.10]
    )
    acquisition_channels = rng.choice(
        ["Merchant Checkout", "App Direct",
         "Referral", "Social Media", "Email"],
        size=n, p=[0.45, 0.25, 0.15, 0.10, 0.05]
    )

    # ── A/B test group ───────────────────────────────────
    ab_groups = rng.choice(
        ["A_strict", "B_relaxed"],
        size=n,
        p=[GEN["ab_test"]["group_A_size"],
           GEN["ab_test"]["group_B_size"]]
    )

    # ── Registration dates ───────────────────────────────
    # Registration closes at the origination cutoff so every
    # customer has at least some chance to transact.
    days_range = (ORIGINATION_END_DATE - START_DATE).days
    reg_dates = [
        START_DATE + pd.Timedelta(days=int(d))
        for d in rng.randint(0, days_range, n)
    ]

    # ── Assemble ─────────────────────────────────────────
    customers = pd.DataFrame({
        "customer_id":                      [generate_customer_id(i) for i in range(n)],
        "registration_date":                reg_dates,
        "age":                              ages,
        "age_bracket":                      chosen_brackets,
        "gender":                           gender,
        "income_band":                      chosen_income,
        "annual_income_gbp":                annual_incomes.round(2),
        "monthly_income_gbp":               monthly_incomes.round(2),
        "employment_status":                employment,
        "existing_monthly_obligations_gbp": monthly_obligations.round(2),
        "savings_balance_gbp":              savings.round(2),
        "thin_file_flag":                   thin_file,
        "credit_score":                     credit_scores,
        "months_credit_history":            months_history,
        "num_credit_accounts":              num_accounts,
        "num_defaults_24m":                 num_defaults_24m,
        "mental_health_flag":               [v["mental_health_flag"] for v in vuln_records],
        "vulnerability_health":             [v["vulnerability_health"] for v in vuln_records],
        "vulnerability_life_event":         [v["vulnerability_life_event"] for v in vuln_records],
        "vulnerability_resilience":         [v["vulnerability_resilience"] for v in vuln_records],
        "vulnerability_capability":         [v["vulnerability_capability"] for v in vuln_records],
        "vulnerability_score":              [v["vulnerability_score"] for v in vuln_records],
        "is_vulnerable":                    [v["is_vulnerable"] for v in vuln_records],
        "device_type":                      device_types,
        "acquisition_channel":              acquisition_channels,
        "ab_test_group":                    ab_groups,
    })

    return customers


# ── TRANSACTION GENERATOR ─────────────────────────────────
def generate_transactions(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Generates BNPL applications per customer within the
    origination window. Approval follows the CONC 5.2A style
    affordability assessment, with the threshold varying by
    A/B test group.

    Instalment plans are drawn from the menu the merchant
    category offers. All plans are monthly.
    See docs/engineering_decision_log.md — Decision 1a.
    """
    rng = np.random.RandomState(RANDOM_SEED + 1)

    merchant_categories = list(GEN["merchant_categories"].keys())
    merchant_weights    = list(GEN["merchant_categories"].values())
    merchant_risk       = GEN["merchant_risk"]
    plans_by_category   = GEN["instalment_plans_by_category"]

    strict_threshold  = GEN["ab_test"]["strict_threshold"]
    relaxed_threshold = GEN["ab_test"]["relaxed_threshold"]

    records = []

    for _, customer in customers.iterrows():

        # Applications per customer. Design assumption
        # reflecting occasional to moderate BNPL usage across
        # the origination window.
        n_transactions = rng.randint(1, 7)

        for t in range(n_transactions):

            # ── Merchant ──────────────────────────────────
            merchant_cat = rng.choice(
                merchant_categories, p=merchant_weights
            )
            risk_multiplier = merchant_risk[merchant_cat]

            # ── Order value ───────────────────────────────
            # Triangular: minimum £10, mode £85 referencing
            # published UK average order values, maximum
            # £2,000 for higher value financed purchases.
            order_value = round(rng.triangular(10, 85, 2000), 2)

            # ── Instalment plan ───────────────────────────
            # The plan menu is a merchant configuration, not a
            # customer choice. Durable goods categories offer
            # longer tenors; fast moving retail does not.
            plans   = plans_by_category[merchant_cat]
            options = [int(k) for k in plans.keys()]
            weights = list(plans.values())
            n_instalments = int(rng.choice(options, p=weights))

            # All plans are monthly. The first instalment is
            # taken at checkout, the remainder at 30 day
            # intervals.
            instalment_freq   = "monthly"
            instalment_amount = round(order_value / n_instalments, 2)

            # ── Transaction date ──────────────────────────
            # After registration, before the origination cutoff.
            reg_date = pd.Timestamp(customer["registration_date"])
            days_available = (ORIGINATION_END_DATE - reg_date).days
            if days_available < 1:
                days_available = 1
            tx_date = reg_date + pd.Timedelta(
                days=int(rng.randint(0, days_available))
            )

            # ── Affordability ─────────────────────────────
            affordability_ratio = compute_affordability_ratio(
                customer["monthly_income_gbp"],
                customer["existing_monthly_obligations_gbp"],
                order_value,
                n_instalments
            )

            if customer["ab_test_group"] == "A_strict":
                fca_flag = int(affordability_ratio > strict_threshold)
            else:
                fca_flag = int(affordability_ratio > relaxed_threshold)

            application_approved = int(fca_flag == 0)

            # Natural key, unique by construction. No hash:
            # a transaction carries no personal data to
            # pseudonymise.
            tx_id = f"T-{customer['customer_id']}-{t}"

            records.append({
                "transaction_id":           tx_id,
                "customer_id":              customer["customer_id"],
                "transaction_date":         tx_date.date(),
                "merchant_category":        merchant_cat,
                "order_value_gbp":          order_value,
                "n_instalments":            n_instalments,
                "instalment_frequency":     instalment_freq,
                "instalment_amount_gbp":    instalment_amount,
                "affordability_ratio":      affordability_ratio,
                "fca_affordability_flag":   fca_flag,
                "application_approved":     application_approved,
                "merchant_risk_multiplier": risk_multiplier,
                "ab_test_group":            customer["ab_test_group"],
            })

    return pd.DataFrame(records)

# ── REPAYMENT STATE MACHINE ───────────────────────────────
# States:
#   CURRENT     — no unpaid instalments
#   DELINQUENT  — one or more unpaid, oldest under 90 DPD
#   DEFAULT     — oldest unpaid has reached 90 DPD, absorbing
#
# Transitions from CURRENT:
#   pay on time / pay late  -> CURRENT
#   miss                    -> DELINQUENT
#
# Transitions from DELINQUENT:
#   full cure    -> CURRENT     (clears all arrears + current)
#   partial cure -> DELINQUENT  (clears oldest only)
#   no payment   -> DELINQUENT  (arrears deepen)
#   oldest arrear reaches 90 DPD -> DEFAULT
#
# Default is emergent. Nothing draws it in advance.

def employment_stability_score(status: str) -> float:
    """
    Maps employment status to a stability score, 0 to 1.
    Design assumption: permanent full time work provides the
    most reliable income for recovering from arrears.
    """
    return {
        "Employed FT":   1.00,
        "Employed PT":   0.70,
        "Self-Employed": 0.60,
        "Retired":       0.65,
        "Student":       0.35,
        "Unemployed":    0.15,
    }.get(status, 0.50)


def financial_resilience(cust, instalment_amount: float) -> float:
    """
    Generator internal latent variable measuring capacity to
    recover from arrears. Deliberately NOT the inverse of the
    miss hazard: a customer can be likely to miss yet able to
    recover, or unlikely to miss yet unable to recover if they do.
    """
    w = GEN["repayment_process"]["resilience_weights"]
    inst = max(instalment_amount, 1.0)

    savings_buffer = min(cust["savings_balance_gbp"] / inst / 6.0, 1.0)
    disposable = max(
        cust["monthly_income_gbp"]
        - cust["existing_monthly_obligations_gbp"], 0.0
    )
    disposable_buffer = min(disposable / inst / 4.0, 1.0)
    stability = employment_stability_score(cust["employment_status"])

    return (
        w["savings_buffer"] * savings_buffer
        + w["disposable_income_buffer"] * disposable_buffer
        + w["employment_stability"] * stability
    )


def miss_hazard(cust, tx, base: float) -> float:
    """
    Per instalment probability of missing. Uses the credit risk
    factors. Acts on MISS likelihood, not on default.

    The A/B uplift is applied here rather than to a default
    probability, because default is emergent. Group B approves
    customers at a looser affordability threshold, so the
    effect enters through repayment behaviour.
    """
    h = base * tx["merchant_risk_multiplier"]
    if cust["thin_file_flag"] == 1:
        h *= 2.0
    if cust["is_vulnerable"] == 1:
        h *= 1.5
    if tx["affordability_ratio"] > 0.35:
        h *= 1.3
    if cust["credit_score"] < 550:
        h *= 1.4

    # A/B uplift, applied multiplicatively to the miss hazard.
    if tx["ab_test_group"] == "B_relaxed":
        h *= GEN["ab_test"]["group_B_hazard_multiplier"]

    return min(h, 0.60)


def resolve_arrears(rng, depth: int, dpd: int, resilience: float,
                    rp: dict, cure_scale: float = 1.0,
                    halflife: float = None) -> str:
    """
    Decides what a delinquent customer does this cycle.

    depth      — how many instalments behind.
    dpd        — age of the oldest arrear. Recovery chances
                 decay as it ages.
    cure_scale — global multiplier on cure probabilities.
    halflife   — days over which cure chances halve. Overrides
                 config during calibration.

    Returns 'full', 'partial', or 'none'.
    """
    table = rp["arrears_resolution"]
    weights = list(table.get(min(depth, max(table.keys())),
                             table[max(table.keys())]))

    if halflife is None:
        halflife = rp["dpd_decay_halflife"]

    scale = (0.5 + resilience) * cure_scale
    decay = 0.5 ** (dpd / halflife)

    full    = min(weights[0] * scale * decay, 0.98)
    partial = min(weights[1] * scale * decay, 0.98 - full)
    none    = max(1.0 - full - partial, 0.01)

    total = full + partial + none
    draw = rng.random() * total
    if draw < full:
        return "full"
    if draw < full + partial:
        return "partial"
    return "none"



def generate_repayments(customers, transactions,
                        base_miss_hazard: float = None,
                        cure_scale: float = None,
                        halflife: float = None) -> pd.DataFrame:
    """
    Generates instalment repayments via a hazard based state
    machine. Default is not drawn; it emerges when the oldest
    unpaid instalment reaches 90 days past due.

    Payments clear the oldest unpaid instalment first, so DPD
    moves to the age of the next outstanding item rather than
    resetting to zero.

    See docs/engineering_decision_log.md — Decision 1a.
    """
    rp  = GEN["repayment_process"]
    rng = np.random.RandomState(RANDOM_SEED + 2)

    if base_miss_hazard is None:
        base_miss_hazard = rp["base_miss_hazard"]
    if cure_scale is None:
        cure_scale = rp.get("cure_scale", 1.0)

    dpd_default = GEN["default_dpd_threshold"]
    dpd_sicr    = GEN["sicr_dpd_threshold"]
    

    approved = transactions[
        transactions["application_approved"] == 1
    ].copy()
    cust_lookup = customers.set_index("customer_id")

    records = []

    for _, tx in approved.iterrows():
        cust = cust_lookup.loc[tx["customer_id"]]

        n_inst      = int(tx["n_instalments"])
        inst_amount = float(tx["instalment_amount_gbp"])
        tx_date     = pd.Timestamp(tx["transaction_date"])

        hazard     = miss_hazard(cust, tx, base_miss_hazard)
        resilience = financial_resilience(cust, inst_amount)

        # All plans monthly. First instalment at checkout,
        # remainder at 30 day intervals.
        due_dates = [tx_date + pd.Timedelta(days=30 * i)
                     for i in range(n_inst)]
        final_due = due_dates[-1]

        # A transaction has a fully observed outcome only if its
        # final instalment plus the 90 day window has elapsed.
        outcome_required_by = final_due + pd.Timedelta(days=dpd_default)
        is_mature = outcome_required_by <= DATA_END_DATE

        state    = "CURRENT"
        unpaid   = []          # due dates of unpaid instalments
        max_dpd  = 0
        n_cycles_delinquent = 0
        default_date = None
        tx_rows = []

        for i, due in enumerate(due_dates, start=1):

            dpd = (due - unpaid[0]).days if unpaid else 0
            max_dpd = max(max_dpd, dpd)

            # DEFAULT check first: absorbing state.
            if state != "DEFAULT" and dpd >= dpd_default:
                state = "DEFAULT"
                default_date = unpaid[0] + pd.Timedelta(days=dpd_default)

            if state == "DEFAULT":
                status, days_late, paid = "Defaulted", 0, 0.0
                unpaid.append(due)

            elif state == "DELINQUENT":
                n_cycles_delinquent += 1
                action = resolve_arrears(
                    rng, len(unpaid), dpd, resilience, rp,
                    cure_scale=cure_scale, halflife=halflife
                )

                if action == "full":

                
                    # Clears every arrear AND the current
                    # instalment. Returns to CURRENT.
                    oldest = unpaid[0]
                    days_late = (due - oldest).days
                    paid = inst_amount * (len(unpaid) + 1)
                    unpaid = []
                    state = "CURRENT"
                    status = "Late"

                elif action == "partial":
                    # Clears the oldest arrear only. Current
                    # instalment goes unpaid. Still delinquent.
                    oldest = unpaid.pop(0)
                    days_late = (due - oldest).days
                    paid = inst_amount
                    unpaid.append(due)
                    status = "Late"

                else:
                    status, days_late, paid = "Missed", 0, 0.0
                    unpaid.append(due)

            else:  # CURRENT
                if rng.random() < hazard:
                    status, days_late, paid = "Missed", 0, 0.0
                    unpaid.append(due)
                    state = "DELINQUENT"
                elif rng.random() < 0.05:
                    # Paid a few days late without becoming
                    # delinquent.
                    status = "Late"
                    days_late = int(rng.randint(1, 29))
                    paid = inst_amount
                else:
                    status, days_late, paid = "Paid", 0, inst_amount

            # Outstanding is the CUMULATIVE unpaid balance.
            outstanding = inst_amount * len(unpaid)

            tx_rows.append({
                "repayment_id":            f"R-{tx['transaction_id']}-{i}",
                "transaction_id":          tx["transaction_id"],
                "customer_id":             tx["customer_id"],
                "instalment_number":       i,
                "instalment_amount_gbp":   inst_amount,
                "due_date":                due.date(),
                "payment_status":          status,
                "days_late":               days_late,
                "days_past_due":           dpd,
                "arrears_depth":           len(unpaid),
                "amount_paid_gbp":         round(paid, 2),
                "outstanding_balance_gbp": round(outstanding, 2),
                "ab_test_group":           tx["ab_test_group"],
                "merchant_category":       tx["merchant_category"],
            })

        # ── Post schedule collections ─────────────────────
        # The contractual schedule has ended but arrears remain.
        # In practice collections activity continues, so the
        # customer keeps receiving opportunities to clear the
        # balance until default or the observation window closes.
        # Without this a customer who misses the final instalment
        # has no chance to recover at all, which was found to
        # account for 42% of all defaults.
        if unpaid and state != "DEFAULT":
            collection_date = final_due
            while True:
                collection_date = collection_date + pd.Timedelta(days=30)

                if collection_date > DATA_END_DATE:
                    # Window closed before the arrears resolved.
                    terminal_dpd = (DATA_END_DATE - unpaid[0]).days
                    max_dpd = max(max_dpd, terminal_dpd)
                    break

                dpd = (collection_date - unpaid[0]).days
                max_dpd = max(max_dpd, dpd)

                if dpd >= dpd_default:
                    state = "DEFAULT"
                    default_date = unpaid[0] + pd.Timedelta(
                        days=dpd_default
                    )
                    break

                n_cycles_delinquent += 1
                action = resolve_arrears(
                    rng, len(unpaid), dpd, resilience, rp,
                    cure_scale=cure_scale, halflife=halflife
                )

                if action == "full":
                    unpaid = []
                    state = "CURRENT"
                    break
                elif action == "partial":
                    unpaid.pop(0)
                    if not unpaid:
                        state = "CURRENT"
                        break
                # 'none' leaves arrears in place; DPD keeps rising.

        default_observed = int(
            state == "DEFAULT" and default_date is not None
            and default_date <= DATA_END_DATE
        )
        outcome_observed = int(default_observed == 1 or is_mature)

        for r in tx_rows:
            r["default_flag"]        = default_observed
            r["outcome_observed"]    = outcome_observed
            r["max_dpd"]             = max_dpd
            r["reached_sicr"]        = int(max_dpd >= dpd_sicr)
            r["cycles_delinquent"]   = n_cycles_delinquent
            r["terminal_state"]      = state
            records.append(r)

    return pd.DataFrame(records)
       

def print_delinquency_spectrum(repayments: pd.DataFrame) -> None:
    """
    Checks whether a genuine delinquency spectrum emerged, or
    whether the process collapsed toward a binary
    recovered/defaulted outcome.
    """
    tx = repayments.groupby("transaction_id").agg(
        max_dpd=("max_dpd", "max"),
        cycles=("cycles_delinquent", "max"),
        state=("terminal_state", "last"),
        defaulted=("default_flag", "max"),
    )

    print("\n" + "=" * 58)
    print("DELINQUENCY SPECTRUM")
    print("=" * 58)

    print("\nMax DPD distribution (transactions that ever missed)")
    bins = [0, 1, 30, 45, 60, 75, 90, 10_000]
    labels = ["0", "1-29", "30-44", "45-59",
              "60-74", "75-89", "90+"]
    delinquent = tx[tx["max_dpd"] > 0]
    cut = pd.cut(delinquent["max_dpd"], bins=bins,
                 labels=labels, right=False)
    for label, count in cut.value_counts().sort_index().items():
        pct = count / max(len(delinquent), 1)
        bar = "#" * int(pct * 40)
        print(f"  {label:<7} {count:>7,}  {pct:>6.2%}  {bar}")

    print("\nTerminal state, transactions that ever missed")
    for state, count in delinquent_states(tx).items():
        print(f"  {state:<22} {count:>7,}  "
              f"{count / max(len(tx), 1):>6.2%}")

    print("\nDelinquency episode length")
    d = tx[tx["cycles"] > 0]["cycles"]
    if len(d):
        print(f"  mean {d.mean():.2f} cycles   "
              f"median {d.median():.0f}   max {d.max()}")

    print("\nOverall default rate (per transaction)"
          f"     {tx['defaulted'].mean():>7.2%}")
    print("=" * 58)


def delinquent_states(tx: pd.DataFrame) -> dict:
    """Terminal state counts for transactions that ever missed."""
    d = tx[tx["max_dpd"] > 0]
    return {
        "recovered (CURRENT)": int((d["state"] == "CURRENT").sum()),
        "still behind":        int((d["state"] == "DELINQUENT").sum()),
        "defaulted":           int((d["state"] == "DEFAULT").sum()),
    }

# ── SAVE OUTPUTS ──────────────────────────────────────────
def build_data_dictionary(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    repayments: pd.DataFrame
) -> dict:
    """
    Field level documentation for the generated dataset.
    Records every column with dtype, cardinality, and missing
    count, plus the windows and exclusions that govern use.

    Supports the documentation expectations in PRA SS1/23
    Principle 3.
    """
    def profile(df: pd.DataFrame) -> dict:
        return {
            col: {
                "dtype":     str(df[col].dtype),
                "n_unique":  int(df[col].nunique()),
                "n_missing": int(df[col].isna().sum()),
            }
            for col in df.columns
        }

    # Outcome observability at transaction level.
    tx_outcome = repayments.groupby("transaction_id").agg(
        observed=("outcome_observed", "max"),
        defaulted=("default_flag", "max"),
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "random_seed":  RANDOM_SEED,
        "project":      CONFIG["project"]["name"],
        "version":      CONFIG["project"]["version"],

        "windows": {
            "origination_start": GEN["start_date"],
            "origination_end":   GEN["origination_end_date"],
            "performance_end":   GEN["data_end_date"],
            "note": (
                "Applications originate within the origination "
                "window. Repayment behaviour is observed to the "
                "performance end date. Transactions whose final "
                "instalment plus the default window falls beyond "
                "that date are right censored."
            ),
        },

        "default_definition": {
            "rule": (
                f"Oldest unpaid instalment reaches "
                f"{GEN['default_dpd_threshold']} days past due."
            ),
            "sicr_threshold_dpd": GEN["sicr_dpd_threshold"],
            "note": (
                "Default emerges from repayment behaviour and is "
                "never assigned in advance. Payments clear the "
                "oldest unpaid instalment first."
            ),
        },

        "outcome_observability": {
            "transactions_total":    int(len(tx_outcome)),
            "outcome_observed":      int(tx_outcome["observed"].sum()),
            "right_censored":        int((tx_outcome["observed"] == 0).sum()),
            "pct_right_censored":    round(
                float((tx_outcome["observed"] == 0).mean()), 4
            ),
            "note": (
                "Right censored transactions carry default_flag 0 "
                "as a storage convention. outcome_observed is the "
                "field that determines modelling eligibility."
            ),
        },

        "reference_sources": [
            "Klarna Wikipink UK, published April 2024, FY23 figures",
            "Klarna UK product documentation",
            "Klarna Q2 2025 Investor Report",
            "FCA Financial Lives Survey May 2022",
            "ONS Household Income Inequality UK FYE 2023",
            "ONS Labour Market Overview 2024",
            "PwC and TotallyMoney YouGov February 2022",
            "NHS Digital Adult Psychiatric Morbidity Survey",
            "World Bank Alternative Data March 2025",
            "Woolard Review FCA February 2021",
        ],

        "excluded_from_model": [
            "mental_health_flag — Equality Act 2010",
            "gender — protected characteristic",
            "merchant_risk_multiplier — generator internal",
            "default_probability — removed, no longer generated",
        ],

        "tables": {
            "customers": {
                "n_rows":  int(len(customers)),
                "columns": profile(customers),
            },
            "transactions": {
                "n_rows":  int(len(transactions)),
                "columns": profile(transactions),
            },
            "repayments": {
                "n_rows":  int(len(repayments)),
                "columns": profile(repayments),
            },
        },
    }

def save_outputs(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    repayments: pd.DataFrame
) -> None:
    """
    Writes the three tables to CSV plus a data dictionary
    documenting every field, the observation windows, and the
    default definition. Output directory comes from config.
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    out_dir = os.path.join(project_root, GEN["output_dir"])
    os.makedirs(out_dir, exist_ok=True)

    customers.to_csv(
        os.path.join(out_dir, "customers.csv"), index=False
    )
    transactions.to_csv(
        os.path.join(out_dir, "transactions.csv"), index=False
    )
    repayments.to_csv(
        os.path.join(out_dir, "repayments.csv"), index=False
    )

    data_dict = build_data_dictionary(
        customers, transactions, repayments
    )
    with open(
        os.path.join(out_dir, "data_dictionary.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(data_dict, f, indent=2)

    print(f"\nOutputs written to: {out_dir}")
    print(f"  customers.csv         {len(customers):>9,} rows")
    print(f"  transactions.csv      {len(transactions):>9,} rows")
    print(f"  repayments.csv        {len(repayments):>9,} rows")
    print(f"  data_dictionary.json")

def print_summary(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    repayments: pd.DataFrame
) -> None:
    """
    Checks the generated population against the reference
    figures it was built to resemble, and reports the emergent
    portfolio behaviour.
    """
    print("\n" + "=" * 58)
    print("POPULATION CHECK")
    print("=" * 58)

    tf = customers["thin_file_flag"].mean()
    print(f"Thin file prevalence      {tf:>7.2%}   reference 31.00%")

    vu = customers["is_vulnerable"].mean()
    print(f"Vulnerable (any driver)   {vu:>7.2%}   composite")

    rs = customers["vulnerability_resilience"].mean()
    print(f"  resilience driver       {rs:>7.2%}   reference 24.00%")

    mh = customers["mental_health_flag"].mean()
    print(f"  health driver           {mh:>7.2%}   reference 17.00%")

    # ── Portfolio ─────────────────────────────────────────
    print("\n" + "=" * 58)
    print("PORTFOLIO")
    print("=" * 58)

    ar = transactions["application_approved"].mean()
    print(f"Approval rate             {ar:>7.2%}")

    print("\nInstalment plan mix (approved)")
    approved = transactions[transactions["application_approved"] == 1]
    for plan, share in (
        approved["n_instalments"].value_counts(normalize=True)
        .sort_index().items()
    ):
        print(f"  {plan:>2} months            {share:>7.2%}")

    # ── Outcome observability ─────────────────────────────
    tx = repayments.groupby("transaction_id").agg(
        observed=("outcome_observed", "max"),
        defaulted=("default_flag", "max"),
        max_dpd=("max_dpd", "max"),
    )

    print("\n" + "=" * 58)
    print("OUTCOME OBSERVABILITY")
    print("=" * 58)
    n_obs = int(tx["observed"].sum())
    n_cen = int((tx["observed"] == 0).sum())
    print(f"Transactions              {len(tx):>9,}")
    print(f"  outcome observed        {n_obs:>9,}  "
          f"{n_obs / len(tx):>7.2%}")
    print(f"  right censored          {n_cen:>9,}  "
          f"{n_cen / len(tx):>7.2%}")

    # ── Default ───────────────────────────────────────────
    print("\n" + "=" * 58)
    print("DEFAULT (emergent, 90 DPD)")
    print("=" * 58)

    observed = tx[tx["observed"] == 1]
    dr_all = tx["defaulted"].mean()
    dr_obs = observed["defaulted"].mean() if len(observed) else 0.0
    print(f"All transactions          {dr_all:>7.2%}")
    print(f"Observed outcomes only    {dr_obs:>7.2%}   "
          f"reference approx 0.40%")

    # Customer level.
    cust_tx = repayments.groupby("customer_id").agg(
        defaulted=("default_flag", "max"),
        all_observed=("outcome_observed", "min"),
    )
    modelable = cust_tx[
        (cust_tx["defaulted"] == 1) | (cust_tx["all_observed"] == 1)
    ]
    print(f"\nCustomers with repayments {len(cust_tx):>9,}")
    print(f"  modelable               {len(modelable):>9,}  "
          f"{len(modelable) / max(len(cust_tx), 1):>7.2%}")
    if len(modelable):
        print(f"  customer default rate   "
              f"{modelable['defaulted'].mean():>7.2%}")

    # ── A/B test ──────────────────────────────────────────
    print("\n" + "=" * 58)
    print("A/B TEST — affordability threshold")
    print("=" * 58)

    rates = {}
    for grp in ["A_strict", "B_relaxed"]:
        sub = repayments[repayments["ab_test_group"] == grp]
        sub_tx = sub.groupby("transaction_id").agg(
            defaulted=("default_flag", "max"),
            observed=("outcome_observed", "max"),
        )
        sub_obs = sub_tx[sub_tx["observed"] == 1]
        rate = sub_obs["defaulted"].mean() if len(sub_obs) else 0.0
        rates[grp] = rate
        n_cust = int((customers["ab_test_group"] == grp).sum())
        print(f"  {grp:<12} {n_cust:>7,} customers  "
              f"default {rate:>6.2%}")

    diff = (rates["B_relaxed"] - rates["A_strict"]) * 100
    print(f"  Difference   {diff:>+6.2f}pp")

    # ── Merchant ──────────────────────────────────────────
    print("\n" + "=" * 58)
    print("DEFAULT RATE BY MERCHANT CATEGORY (observed only)")
    print("=" * 58)

    mtx = repayments.groupby(
        ["merchant_category", "transaction_id"]
    ).agg(
        defaulted=("default_flag", "max"),
        observed=("outcome_observed", "max"),
    ).reset_index()
    mtx = mtx[mtx["observed"] == 1]
    mc = mtx.groupby("merchant_category")["defaulted"].mean()
    for cat, rate in mc.sort_values(ascending=False).items():
        print(f"  {cat:<22} {rate:>6.2%}")

    print("=" * 58)



# ── CALIBRATION SWEEP ─────────────────────────────────────
def calibrate_grid(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    hazards: list = None,
    cure_scales: list = None,
    halflives: list = None,
    sample_frac: float = 0.25,
) -> None:
    """
    Sweeps the miss hazard, cure generosity, and DPD decay
    halflife, reporting both portfolio rates and modelling
    viability.

    Two constraints pull against each other. Matching a
    published default rate argues for a low hazard and generous
    cure. Retaining enough positive cases to train a model
    argues the other way. A calibration that hits the reference
    but leaves too few defaults to learn from has failed at the
    thing the dataset exists for.

    Reported per cell:
      missed  — transactions ever delinquent
      def tx  — transaction level default rate, observed only
      model   — customers with a fully observed outcome
      def cust— those customers who defaulted
      pos     — customer level default rate

    See docs/engineering_decision_log.md — Decision 1a.
    """
    if hazards is None:
        hazards = [0.004, 0.005, 0.006]
    if cure_scales is None:
        cure_scales = [1.1, 1.2, 1.3, 1.4, 1.5]
    if halflives is None:
        halflives = [45]

    approved = transactions[
        transactions["application_approved"] == 1
    ]
    sample = approved.sample(
        frac=sample_frac, random_state=RANDOM_SEED
    )
    scale_up = 1.0 / sample_frac

    print("\n" + "=" * 78)
    print("CALIBRATION SWEEP")
    print("=" * 78)
    print(f"Sample: {len(sample):,} approved transactions "
          f"({sample_frac:.0%}). Customer counts scaled to full "
          f"population.")
    print("References: delinquency around 5%, default around 0.4%")

    for hl in halflives:
        print(f"\n{'-' * 78}")
        print(f"DPD DECAY HALFLIFE: {hl} days")
        print(f"{'-' * 78}")
        print(f"{'hazard':>7} {'cure':>6} | {'missed':>7} "
              f"{'def tx':>7} | {'model':>8} {'def cust':>9} "
              f"{'pos rate':>9}")
        print("-" * 78)

        for h in hazards:
            for cs in cure_scales:
                rp = generate_repayments(
                    customers, sample,
                    base_miss_hazard=h,
                    cure_scale=cs,
                    halflife=hl,
                )

                # Transaction level.
                tx = rp.groupby("transaction_id").agg(
                    defaulted=("default_flag", "max"),
                    observed=("outcome_observed", "max"),
                    max_dpd=("max_dpd", "max"),
                )
                obs = tx[tx["observed"] == 1]
                tx_rate = obs["defaulted"].mean() if len(obs) else 0.0
                missed = (tx["max_dpd"] > 0).mean()

                # Customer level. A customer is modelable if they
                # defaulted, or if every transaction they hold has
                # completed its performance window.
                cust = rp.groupby("customer_id").agg(
                    defaulted=("default_flag", "max"),
                    all_observed=("outcome_observed", "min"),
                )
                modelable = cust[
                    (cust["defaulted"] == 1)
                    | (cust["all_observed"] == 1)
                ]
                n_model = int(len(modelable) * scale_up)
                n_pos = int(modelable["defaulted"].sum() * scale_up)
                pos_rate = (
                    modelable["defaulted"].mean()
                    if len(modelable) else 0.0
                )

                print(f"{h:>7.4f} {cs:>6.2f} | {missed:>6.2%} "
                      f"{tx_rate:>7.2%} | {n_model:>8,} "
                      f"{n_pos:>9,} {pos_rate:>8.2%}")

    print("-" * 78)
    print("\nPick a cell balancing three things: delinquency near")
    print("the reference, default rate plausible, and enough")
    print("positive cases to train on. Under roughly 300 positives")
    print("makes model evaluation unstable.")
    print("=" * 78)
def diagnose_default_timing(repayments: pd.DataFrame) -> None:
    """
    Splits defaults by where in the schedule they occurred.

    A customer who misses a mid schedule instalment has later
    due dates at which to cure. A customer who misses the final
    instalment has none: the schedule ends, no further cure
    opportunity exists, and default follows automatically once
    90 days elapse.

    If defaults are dominated by final instalment misses, the
    observed default floor is an artefact of that gap rather
    than a property of the product.
    """
    tx = repayments.groupby("transaction_id").agg(
        defaulted=("default_flag", "max"),
        observed=("outcome_observed", "max"),
        n_inst=("instalment_number", "max"),
    )

    missed = repayments[
        repayments["payment_status"].isin(["Missed", "Defaulted"])
    ]
    first_miss = missed.groupby("transaction_id")["instalment_number"].min()
    tx = tx.join(first_miss.rename("first_miss"))

    d = tx[(tx["defaulted"] == 1) & (tx["observed"] == 1)]

    print("\n" + "=" * 58)
    print("DEFAULT TIMING DIAGNOSTIC")
    print("=" * 58)
    print(f"Observed defaults: {len(d):,}")

    if not len(d):
        print("No observed defaults to analyse.")
        return

    final_miss = (d["first_miss"] == d["n_inst"]).sum()
    mid_miss   = (d["first_miss"] < d["n_inst"]).sum()

    print(f"\n  first miss on FINAL instalment  {final_miss:>7,}  "
          f"{final_miss / len(d):>6.2%}")
    print(f"  first miss MID schedule         {mid_miss:>7,}  "
          f"{mid_miss / len(d):>6.2%}")

    print("\nBy plan length")
    for n in sorted(d["n_inst"].unique()):
        sub = d[d["n_inst"] == n]
        fm = (sub["first_miss"] == n).sum()
        print(f"  {n:>2} month plans   {len(sub):>7,} defaults   "
              f"final instalment {fm / max(len(sub), 1):>6.2%}")

    print("\nCure opportunities available at first miss")
    d = d.copy()
    d["cure_chances"] = d["n_inst"] - d["first_miss"]
    for c in sorted(d["cure_chances"].unique()):
        n = (d["cure_chances"] == c).sum()
        print(f"  {c} remaining due dates   {n:>7,}  "
              f"{n / len(d):>6.2%}")

    print("=" * 58)


# ── ENTRY POINT ───────────────────────────────────────────
def main() -> None:
    """
    Runs the generation pipeline:
      1. Customers
      2. Transactions, within the origination window
      3. Repayments, hazard based, default emergent
      4. Save CSVs and the data dictionary
      5. Population, portfolio, and spectrum diagnostics
    """
    print("=" * 58)
    print("SENTINEL CREDIT — SYNTHETIC BNPL DATA GENERATOR")
    print("=" * 58)
    print(f"Seed:         {RANDOM_SEED}")
    print(f"Customers:    {GEN['n_customers']:,}")
    print(f"Origination:  {GEN['start_date']} to "
          f"{GEN['origination_end_date']}")
    print(f"Performance:  observed to {GEN['data_end_date']}")
    print(f"Default rule: {GEN['default_dpd_threshold']} DPD "
          f"on oldest unpaid instalment")

    print("\n[1/4] Generating customers...")
    customers = generate_customers()
    print(f"      {len(customers):,} customers created")

    print("[2/4] Generating transactions...")
    transactions = generate_transactions(customers)
    n_approved = int(transactions["application_approved"].sum())
    print(f"      {len(transactions):,} applications, "
          f"{n_approved:,} approved")

    print("[3/4] Generating repayments...")
    repayments = generate_repayments(customers, transactions)
    print(f"      {len(repayments):,} repayment records created")

    print("[4/4] Saving outputs...")
    save_outputs(customers, transactions, repayments)

    print_summary(customers, transactions, repayments)
    print_delinquency_spectrum(repayments)


if __name__ == "__main__":
    main()