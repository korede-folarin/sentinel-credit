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
    Generates synthetic BNPL customer population.
    50,000 customers calibrated against published sources.
    See docs/engineering_decision_log.md Decision 1.
    """
    n = GEN["n_customers"]
    rng = np.random.RandomState(RANDOM_SEED)

    # ── Age ───────────────────────────────────────────────
    # Source: Klarna Wikipink UK 2024 — age bracket chart
    age_brackets = list(GEN["age_distribution"].keys())
    age_weights  = list(GEN["age_distribution"].values())
    age_ranges   = {
        k: tuple(v) for k, v in GEN["age_ranges"].items()
    }

    chosen_brackets = rng.choice(
        age_brackets, size=n, p=age_weights
    )
    ages = np.array([
        rng.randint(age_ranges[b][0], age_ranges[b][1] + 1)
        for b in chosen_brackets
    ])

    # ── Income ────────────────────────────────────────────
    # Source: ONS FYE 2023 + BNPL downward shift
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
    # Source: ONS Labour Market Overview 2024
    # Employment rate 75.7% — Employed FT + PT + Self-Employed
    # Unemployment rate 4.2% — expanded to 7.3% to include
    # economic inactivity not captured by student/retired
    # Economic inactivity 20.9% — split between Student
    # and Retired based on BNPL young age distribution.
    # Student 12% and Retired 5% are design assumptions.
    employment_options = [
        "Employed FT", "Employed PT", "Self-Employed",
        "Student", "Unemployed", "Retired"
    ]
    employment_weights = [0.52, 0.14, 0.097, 0.12, 0.073, 0.05]
    employment = rng.choice(
        employment_options, size=n, p=employment_weights
    )

    # ── Gender ────────────────────────────────────────────
    # Source: ONS Population Estimates 2023
    # UK adult population approximately 51% female 49% male.
    # Applied directly. Non-binary and Prefer not to say
    # are design assumptions at 2% each.
    gender = rng.choice(
        ["Female", "Male", "Non-binary", "Prefer not to say"],
        size=n, p=[0.51, 0.45, 0.02, 0.02]
    )

    # ── Thin file ─────────────────────────────────────────
    # Source: PwC/TotallyMoney 2022
    thin_file = (
        rng.random(n) < GEN["thin_file_prevalence"]
    ).astype(int)

    # ── Credit score ──────────────────────────────────────
    # Thin file customers score lower on average
    credit_scores = np.where(
        thin_file == 1,
        rng.randint(300, 580, n),
        rng.randint(500, 999, n)
    )

    # ── Credit history ───────────────────────────────────
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
    # As proportion of monthly income — beta distribution
    # Beta(2, 5) is right-skewed: most customers spend
    # 20-40% of income on obligations with a long tail.
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
        vulnerability_flags(
            np.random.RandomState(RANDOM_SEED + i)
        )
        for i in range(n)
    ]

    # ── Alternative data ─────────────────────────────────
    # Source: World Bank Alternative Data March 2025
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
        p=[
            GEN["ab_test"]["group_A_size"],
            GEN["ab_test"]["group_B_size"],
        ]
    )

    # ── Registration dates ───────────────────────────────
    start = pd.Timestamp(GEN["start_date"])
    end   = pd.Timestamp(GEN["end_date"])
    days_range = (end - start).days
    reg_dates = [
        start + pd.Timedelta(days=int(d))
        for d in rng.randint(0, days_range, n)
    ]

    # ── Assemble DataFrame ───────────────────────────────
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
    Generates BNPL transaction applications per customer.
    Each customer makes 1 to 6 transactions on average.
    See docs/engineering_decision_log.md for calibration sources.
    """
    rng = np.random.RandomState(RANDOM_SEED + 1)

    merchant_categories = list(GEN["merchant_categories"].keys())
    merchant_weights    = list(GEN["merchant_categories"].values())
    merchant_risk       = GEN["merchant_risk"]

    strict_threshold  = GEN["ab_test"]["strict_threshold"]
    relaxed_threshold = GEN["ab_test"]["relaxed_threshold"]

    start = pd.Timestamp(GEN["start_date"])
    end   = pd.Timestamp(GEN["end_date"])

    records = []

    for _, customer in customers.iterrows():

        # Number of transactions per customer
        # Design assumption: 1 to 6 transactions per customer
        # reflecting occasional to moderate BNPL usage over
        # the 3 year window. Klarna Wikipink 2024 reports
        # approximately 10 transactions per active user per
        # year globally. We use a conservative lower range as
        # not all customers are active throughout the period.
        n_transactions = rng.randint(1, 7)

        for t in range(n_transactions):

            # Merchant category
            merchant_cat = rng.choice(
                merchant_categories, p=merchant_weights
            )
            risk_multiplier = merchant_risk[merchant_cat]

            # Order value — triangular distribution
            # Minimum £10: design assumption
            # Mode £85: informed by Klarna published average
            #           order value range of £70-£90 (UK)
            # Maximum £2,000: informed by Klarna Fair
            #           Financing high-value product range
            # Design assumption — no single source gives
            # exact UK BNPL order value distribution.
            order_value = round(rng.triangular(10, 85, 2000), 2)

            # Instalment plan
            n_instalments   = int(rng.choice([3, 4, 6, 12]))
            instalment_freq = rng.choice(
                ["biweekly", "monthly"], p=[0.60, 0.40]
            )
            instalment_amount = round(order_value / n_instalments, 2)

            # Transaction date — after registration
            reg_date = pd.Timestamp(customer["registration_date"])
            days_after_reg = (end - reg_date).days
            if days_after_reg < 1:
                days_after_reg = 1
            tx_date = reg_date + pd.Timedelta(
                days=int(rng.randint(0, days_after_reg))
            )

            # Affordability ratio — FCA CONC 5.2A
            affordability_ratio = compute_affordability_ratio(
                customer["monthly_income_gbp"],
                customer["existing_monthly_obligations_gbp"],
                order_value,
                n_instalments
            )

            # FCA affordability flag
            if customer["ab_test_group"] == "A_strict":
                fca_flag = int(affordability_ratio > strict_threshold)
            else:
                fca_flag = int(affordability_ratio > relaxed_threshold)

            # Approval decision
            # Declined if affordability flag fired
            application_approved = int(fca_flag == 0)

            # Transaction ID — natural key, no hash needed (no personal
            # data to pseudonymise). Unique by construction.
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


# ── REPAYMENT GENERATOR ───────────────────────────────────
def generate_repayments(
    customers: pd.DataFrame,
    transactions: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates instalment repayment records for each approved
    transaction. Contains the default probability engine.
    See docs/engineering_decision_log.md Decision 1.
    """
    rng = np.random.RandomState(RANDOM_SEED + 2)

    # Only generate repayments for approved transactions
    approved = transactions[
        transactions["application_approved"] == 1
    ].copy()

    # Customer lookup for fast access
    cust_lookup = customers.set_index("customer_id")

    base_rate = GEN["base_default_rate"]
    b_uplift  = GEN["ab_test"]["group_B_default_uplift"]

    records = []

    for _, tx in approved.iterrows():

        cust = cust_lookup.loc[tx["customer_id"]]

        # ── Default Probability Engine ────────────────────
        # Base rate = Klarna Wikipink UK 2024 published
        # UK default rate of 0.4%. The population uplift for
        # thin-file and vulnerable segments is applied by the
        # multipliers below — it is NOT baked into the base.
        # Realised portfolio rate emerges at approximately
        # 1.0% which is consistent with the 1.43x weighted
        # uplift documented in engineering_decision_log.md.
        risk = base_rate * tx["merchant_risk_multiplier"]

        # Thin file uplift — design assumption 2.0x
        if cust["thin_file_flag"] == 1:
            risk *= 2.0

        # Vulnerability uplift — design assumption 1.5x
        if cust["is_vulnerable"] == 1:
            risk *= 1.5

        # Affordability uplift — design assumption 1.3x
        if tx["affordability_ratio"] > 0.35:
            risk *= 1.3

        # Credit score adjustment — design assumption 1.4x
        if cust["credit_score"] < 550:
            risk *= 1.4

        # Cap at realistic maximum — design assumption 15%
        risk = min(risk, 0.15)

        # A/B test group uplift — applied LAST as a flat
        # additive term so the engineered effect size is
        # exactly group_B_default_uplift and is not amplified
        # or absorbed by the multipliers above.
        # Group B relaxed threshold approves riskier customers.
        if tx["ab_test_group"] == "B_relaxed":
            risk += b_uplift

        default_probability = risk

        # Draw default outcome for this transaction
        will_default = rng.random() < default_probability

        # ── Generate Instalments ──────────────────────────
        n_inst      = int(tx["n_instalments"])
        inst_amount = tx["instalment_amount_gbp"]
        tx_date     = pd.Timestamp(tx["transaction_date"])

        # Instalment frequency
        if tx["instalment_frequency"] == "biweekly":
            delta = pd.Timedelta(days=14)
        else:
            delta = pd.Timedelta(days=30)

        # Default occurs at a random instalment
        if will_default:
            default_at = rng.randint(1, n_inst + 1)
        else:
            default_at = None

        for inst_num in range(1, n_inst + 1):

            due_date = tx_date + (delta * inst_num)

            # Payment status
            if will_default and inst_num >= default_at:
                if inst_num == default_at:
                    payment_status = "Missed"
                else:
                    payment_status = "Defaulted"
                days_late    = 0
                amount_paid  = 0.0
                outstanding  = inst_amount
                default_flag = 1

            elif rng.random() < 0.05:
                # 5% chance of late payment per instalment.
                # LendingTree 2025 reports 41% of BNPL users
                # made at least one late payment. We use 5%
                # per instalment as a conservative rate.
                payment_status = "Late"
                days_late      = int(rng.randint(1, 30))
                amount_paid    = inst_amount
                outstanding    = 0.0
                default_flag   = 0

            else:
                payment_status = "Paid"
                days_late      = 0
                amount_paid    = inst_amount
                outstanding    = 0.0
                default_flag   = 0
                
            # Repayment ID — natural key, unique by construction.
            rp_id = f"R-{tx['transaction_id']}-{inst_num}"

            records.append({
                "repayment_id":            rp_id,
                "transaction_id":          tx["transaction_id"],
                "customer_id":             tx["customer_id"],
                "instalment_number":       inst_num,
                "instalment_amount_gbp":   inst_amount,
                "due_date":                due_date.date(),
                "payment_status":          payment_status,
                "days_late":               days_late,
                "amount_paid_gbp":         round(amount_paid, 2),
                "outstanding_balance_gbp": round(outstanding, 2),
                "default_flag":            default_flag,
                "default_probability":     round(default_probability, 6),
                "ab_test_group":           tx["ab_test_group"],
                "merchant_category":       tx["merchant_category"],
            })

    return pd.DataFrame(records)


# ── SAVE OUTPUTS ──────────────────────────────────────────
def build_data_dictionary(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    repayments: pd.DataFrame
) -> dict:
    """
    Builds a data dictionary documenting every column in
    every table with dtype, source, and modelling notes.

    Why: PRA SS1/23 Principle 3 requires model development
    to be documented. The data dictionary is the field-level
    documentation for the entire dataset.
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

    return {
        "generated_at":   datetime.now().isoformat(timespec="seconds"),
        "random_seed":    RANDOM_SEED,
        "project":        CONFIG["project"]["name"],
        "version":        CONFIG["project"]["version"],
        "date_range":     f"{GEN['start_date']} to {GEN['end_date']}",
        "calibration_sources": [
            "Klarna Wikipink UK 2024",
            "Klarna Q2 2025 Investor Report",
            "FCA Financial Lives Survey May 2022",
            "ONS Household Income Inequality UK FYE 2023",
            "ONS Labour Market Overview 2024",
            "PwC and TotallyMoney YouGov February 2022",
            "NHS Digital Adult Psychiatric Morbidity Survey",
            "World Bank Alternative Data March 2025",
        ],
        "excluded_from_model": [
            "mental_health_flag — Equality Act 2010",
            "gender — protected characteristic",
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
    Saves all three tables to CSV plus a data dictionary
    JSON documenting every field.

    Output directory is read from config.yaml and created
    if it does not exist.
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
    Prints calibration checks against published sources so
    we can verify the generator produced what it was
    designed to produce.
    """
    print("\n" + "=" * 58)
    print("CALIBRATION CHECK")
    print("=" * 58)

    # Thin file — target 31% (PwC/TotallyMoney 2022)
    tf = customers["thin_file_flag"].mean()
    print(f"Thin file prevalence      {tf:>7.2%}   target 31.00%")

    # Vulnerable — composite of four drivers
    vu = customers["is_vulnerable"].mean()
    print(f"Vulnerable (any driver)   {vu:>7.2%}   composite")

    # Resilience driver — target 24% (FCA FLS May 2022)
    rs = customers["vulnerability_resilience"].mean()
    print(f"  resilience driver       {rs:>7.2%}   target 24.00%")

    # Mental health — target 17% (NHS Digital)
    mh = customers["mental_health_flag"].mean()
    print(f"  health driver           {mh:>7.2%}   target 17.00%")

    # Approval rate
    ar = transactions["application_approved"].mean()
    print(f"\nApproval rate             {ar:>7.2%}")

    # Default rate per transaction
    tx_defaults = repayments.groupby(
        "transaction_id"
    )["default_flag"].max()
    dr = tx_defaults.mean()
    print(f"Default rate (per tx)     {dr:>7.2%}   target ~1.00%")

    # A/B test result
    print("\nA/B TEST")
    for grp in ["A_strict", "B_relaxed"]:
        sub = repayments[repayments["ab_test_group"] == grp]
        sub_def = sub.groupby("transaction_id")["default_flag"].max()
        n_cust = (customers["ab_test_group"] == grp).sum()
        print(
            f"  {grp:<12} {n_cust:>7,} customers  "
            f"default {sub_def.mean():>6.2%}"
        )

    a = repayments[repayments["ab_test_group"] == "A_strict"]
    b = repayments[repayments["ab_test_group"] == "B_relaxed"]
    a_dr = a.groupby("transaction_id")["default_flag"].max().mean()
    b_dr = b.groupby("transaction_id")["default_flag"].max().mean()
    print(f"  Difference   {(b_dr - a_dr) * 100:>+6.2f}pp")

    # Default by merchant category
    print("\nDEFAULT RATE BY MERCHANT CATEGORY")
    mc = repayments.groupby(
        ["merchant_category", "transaction_id"]
    )["default_flag"].max().groupby("merchant_category").mean()
    for cat, rate in mc.sort_values(ascending=False).items():
        print(f"  {cat:<22} {rate:>6.2%}")

    print("=" * 58)


# ── ENTRY POINT ───────────────────────────────────────────
def main() -> None:
    """
    Runs the full generation pipeline in order:
      1. Customers
      2. Transactions (depends on customers)
      3. Repayments (depends on customers + transactions)
      4. Save to CSV + data dictionary
      5. Print calibration checks
    """
    print("=" * 58)
    print("SENTINEL CREDIT — SYNTHETIC BNPL DATA GENERATOR")
    print("=" * 58)
    print(f"Seed:      {RANDOM_SEED}")
    print(f"Customers: {GEN['n_customers']:,}")
    print(f"Period:    {GEN['start_date']} to {GEN['end_date']}")

    print("\n[1/4] Generating customers...")
    customers = generate_customers()
    print(f"      {len(customers):,} customers created")

    print("[2/4] Generating transactions...")
    transactions = generate_transactions(customers)
    print(f"      {len(transactions):,} transactions created")

    print("[3/4] Generating repayments...")
    repayments = generate_repayments(customers, transactions)
    print(f"      {len(repayments):,} repayment records created")

    print("[4/4] Saving outputs...")
    save_outputs(customers, transactions, repayments)

    print_summary(customers, transactions, repayments)


if __name__ == "__main__":
    main()
