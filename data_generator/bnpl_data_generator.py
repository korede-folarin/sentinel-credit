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
    Generates a pseudonymised customer ID using MD5 hashing.

    Why: GDPR requires that personal data is pseudonymised
    where possible. Real customer IDs are never stored.
    Instead we hash a number to produce an untraceable ID.

    Args:
        index: The customer's position in the generation loop

    Returns:
        A pseudonymised ID string e.g. 'C3F2A1B4E9D7'
    """
    raw = f"BNPL-CUST-{index:06d}-{RANDOM_SEED}"
    return "C" + hashlib.md5(raw.encode()).hexdigest()[:10].upper()



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
    design assumption, no single source defines this.

    Args:
        band: Income band string from config.yaml

    Returns:
        Tuple of (minimum, maximum, median) in GBP
    """
    mapping = {
        "Under £15k":  (8_000,   15_000,  12_000),
        "£15k-£25k":   (15_000,  25_000,  20_000),
        "£25k-£35k":   (25_000,  35_000,  30_000),
        "£35k-£50k":   (35_000,  50_000,  42_000),
        "£50k-£75k":   (50_000,  75_000,  61_000),
        "Over £75k":   (75_000, 200_000,  95_000),
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

Why:
    FCA CONC 5.2A requires lenders to assess whether new
    credit can be afforded without causing financial
    difficulty. This ratio measures the share of monthly
    income committed to debt repayments.

    The 0.45 threshold used in Group A is a project design
    assumption selected to create a meaningful distinction
    from Group B (0.60) for experimental evaluation. It is
    not prescribed by FCA guidance and is documented in
    docs/engineering_decision_log.md.

Args:
    monthly_income: Net monthly income in GBP
    monthly_obligations: Existing monthly debt payments
    order_value: Total BNPL purchase amount
    n_instalments: Number of repayment instalments

Returns:
    Affordability ratio rounded to 4 decimal places.
"""
   
    instalment = order_value / n_instalments
    ratio = (monthly_obligations + instalment) / max(monthly_income, 1)
    return round(ratio, 4)