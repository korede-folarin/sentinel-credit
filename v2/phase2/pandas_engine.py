"""
Sentinel Credit v2, Phase 2: the pandas engine (V2-04).

Plain English: the same feature store as v1's SQL, built with pandas in one
Python process: the usual single-machine data-science tool, and the
baseline Spark has to beat.

Exactness: PostgreSQL rounds exact decimals half away from zero (an average
of exactly 75.125 becomes 75.13). Floating-point rounding would round some
ties the other way, so every rounded average or ratio here is computed in
whole units (pence, or ten-thousandths) with integer arithmetic, then
rounded half up: round(a / b, p) = floor((2 * a * 10^p + b) / (2 * b)) / 10^p
for a >= 0, b > 0. All the quantities involved are non-negative (checked).

Timing: read Parquet -> compute -> write Parquet, plus peak memory of the
process.

  python v2/phase2/pandas_engine.py --dataset <name> --repeats 3
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dataset_dir  # noqa: E402

HIGH = ["Travel", "Electronics"]
MEDIUM = ["Fashion & Clothing", "Home", "Other"]
LOW = ["Sports", "Health & Beauty", "Food & Grocery"]
SNAPSHOT = pd.Timestamp("2024-12-31")


def units(series: pd.Series, scale: int) -> pd.Series:
    """Exact whole units of a value with at most `scale` decimal places."""
    return np.rint(series.astype(float) * 10 ** scale).astype("int64")


def round_half_up(num: pd.Series, den: pd.Series, places: int) -> pd.Series:
    """num / den rounded half away from zero to `places`, from integers; NaN where den is 0 or missing."""
    num, den = num.astype("float64"), den.astype("float64")
    ok = den.notna() & num.notna() & (den > 0)
    if (num[ok] < 0).any():
        raise ValueError("round_half_up expects non-negative numerators")
    n = num[ok].to_numpy(dtype="int64") * 10 ** places
    d = den[ok].to_numpy(dtype="int64")
    if len(n) and (n.max() > 2 ** 61 or d.max() > 2 ** 61):    # keep 2 * n + d inside int64
        raise OverflowError("round_half_up: values too large for exact int64 arithmetic")
    out = pd.Series(np.nan, index=num.index)
    out[ok] = ((2 * n + d) // (2 * d)) / 10 ** places
    return out


def read(folder: str, table: str) -> pd.DataFrame:
    path = os.path.join(folder, f"{table}.parquet")
    return pd.read_parquet(path if os.path.exists(path) else os.path.join(folder, table))


def build(customers: pd.DataFrame, tx: pd.DataFrame, rep: pd.DataFrame) -> pd.DataFrame:
    # customer_default
    cd = rep.groupby("customer_id").agg(default_flag=("default_flag", "max"),
                                        all_observed=("outcome_observed", "min"))

    # application_behaviour (all applications)
    t = tx.assign(aff_u=units(tx["affordability_ratio"], 4))
    g = t.groupby("customer_id")
    ab = g.agg(n_applications=("transaction_id", "size"), n_approved=("application_approved", "sum"),
               aff_sum=("aff_u", "sum"), aff_max=("aff_u", "max"),
               n_affordability_flags=("fca_affordability_flag", "sum"))
    ab["n_declined"] = ab["n_applications"] - ab["n_approved"]
    ab["approval_rate"] = round_half_up(ab["n_approved"], ab["n_applications"], 4)
    ab["avg_affordability_ratio"] = round_half_up(ab["aff_sum"], ab["n_applications"] * 10_000, 4)
    ab["max_affordability_ratio"] = ab["aff_max"] / 10_000
    ab["pct_affordability_flagged"] = round_half_up(ab["n_affordability_flags"], ab["n_applications"], 4)

    # exposure (approved only). Parquet dates arrive as Python date objects;
    # as datetime64 the min/max run in compiled code (as objects they fell
    # back to a pure-Python loop: 24 of 27 seconds on v1's data, measured).
    a = tx[tx["application_approved"] == 1]
    a = a.assign(transaction_date=pd.to_datetime(a["transaction_date"]))
    a = a.assign(ov_u=units(a["order_value_gbp"], 2), inst_u=units(a["instalment_amount_gbp"], 2),
                 hi=a["merchant_category"].isin(HIGH).astype("int64"),
                 med=a["merchant_category"].isin(MEDIUM).astype("int64"),
                 lo=a["merchant_category"].isin(LOW).astype("int64"))
    g = a.groupby("customer_id")
    ex = g.agg(n_transactions=("transaction_id", "size"), ov_sum=("ov_u", "sum"), ov_max=("ov_u", "max"),
               ni_sum=("n_instalments", "sum"), inst_sum=("inst_u", "sum"),
               n_merchant_categories=("merchant_category", "nunique"),
               hi=("hi", "sum"), med=("med", "sum"), lo=("lo", "sum"),
               first_transaction_date=("transaction_date", "min"), last_transaction_date=("transaction_date", "max"))
    n = ex["n_transactions"]
    ex["total_order_value_gbp"] = ex["ov_sum"] / 100
    ex["avg_order_value_gbp"] = round_half_up(ex["ov_sum"], n * 100, 2)
    ex["max_order_value_gbp"] = ex["ov_max"] / 100
    ex["avg_n_instalments"] = round_half_up(ex["ni_sum"], n, 2)
    ex["total_instalment_exposure_gbp"] = ex["inst_sum"] / 100
    ex["pct_high_risk_category"] = round_half_up(ex["hi"], n, 4)
    ex["pct_medium_risk_category"] = round_half_up(ex["med"], n, 4)
    ex["pct_low_risk_category"] = round_half_up(ex["lo"], n, 4)

    # repayment_behaviour
    late = rep["payment_status"] == "Late"
    dl_pos = rep["days_late"].where(rep["days_late"] > 0)
    depth = rep["arrears_depth"].where(rep["arrears_depth"] != 0)
    r = rep.assign(late=late.astype("int64"), dl_pos=dl_pos, dl_pos_n=dl_pos.notna().astype("int64"),
                   depth=depth, depth_n=depth.notna().astype("int64"),
                   inst_u=units(rep["instalment_amount_gbp"], 2))
    g = r.groupby("customer_id")
    rb = g.agg(n_instalments_total=("repayment_id", "size"),
               n_transactions_with_repayments=("transaction_id", "nunique"),
               n_late_payments=("late", "sum"), dl_sum=("dl_pos", "sum"), dl_n=("dl_pos_n", "sum"),
               max_days_late=("days_late", "max"), total_days_late=("days_late", "sum"),
               max_dpd_ever=("max_dpd", "max"), dpd_sum=("max_dpd", "sum"),
               n_reached_sicr=("reached_sicr", "sum"),
               max_cycles_delinquent=("cycles_delinquent", "max"), cyc_sum=("cycles_delinquent", "sum"),
               max_arrears_depth=("arrears_depth", "max"), depth_sum=("depth", "sum"), depth_n=("depth_n", "sum"),
               inst_sum=("inst_u", "sum"))
    rows = rb["n_instalments_total"]
    rb["pct_late_payments"] = round_half_up(rb["n_late_payments"], rows, 4)
    rb["avg_days_late"] = round_half_up(rb["dl_sum"], rb["dl_n"], 2)
    rb["avg_max_dpd"] = round_half_up(rb["dpd_sum"], rows, 2)
    rb["pct_reached_sicr"] = round_half_up(rb["n_reached_sicr"], rows, 4)
    rb["avg_cycles_delinquent"] = round_half_up(rb["cyc_sum"], rows, 2)
    rb["avg_arrears_depth_when_behind"] = round_half_up(rb["depth_sum"], rb["depth_n"], 2)
    rb["avg_instalment_amount_gbp"] = round_half_up(rb["inst_sum"], rows * 100, 2)
    rb["total_scheduled_repayment_gbp"] = rb["inst_sum"] / 100
    lt = r[late].groupby("customer_id")["transaction_id"].nunique().rename("n_transactions_with_late_payment")
    term = (r.groupby(["customer_id", "terminal_state"])["transaction_id"].nunique().unstack(fill_value=0)
            .reindex(columns=["CURRENT", "DELINQUENT", "DEFAULT"], fill_value=0))
    term.columns = ["n_terminal_current", "n_terminal_delinquent", "n_terminal_default"]
    rb = rb.join(lt).join(term)

    # final select
    c = customers.set_index("customer_id")
    out = pd.DataFrame(index=c.index)
    out["target_default"] = cd["default_flag"].reindex(c.index).fillna(0).astype("int64")
    has = cd["default_flag"].eq(1) | cd["all_observed"].eq(1)
    out["has_outcome"] = has.reindex(c.index).fillna(False).astype("int64")
    for col in ["age", "annual_income_gbp", "monthly_income_gbp", "existing_monthly_obligations_gbp",
                "savings_balance_gbp", "credit_score", "months_credit_history", "num_credit_accounts",
                "num_defaults_24m", "thin_file_flag", "vulnerability_score", "is_vulnerable"]:
        out[col] = c[col]
    ab, ex, rb = ab.reindex(c.index), ex.reindex(c.index), rb.reindex(c.index)
    for col in ["n_applications", "n_approved", "n_declined"]:
        out[col] = ab[col].fillna(0).astype("int64")
    for col in ["approval_rate", "avg_affordability_ratio", "max_affordability_ratio"]:
        out[col] = ab[col]
    out["n_affordability_flags"] = ab["n_affordability_flags"].fillna(0).astype("int64")
    out["pct_affordability_flagged"] = ab["pct_affordability_flagged"]
    out["n_transactions"] = ex["n_transactions"].fillna(0).astype("int64")
    out["total_order_value_gbp"] = ex["total_order_value_gbp"].fillna(0)
    for col in ["avg_order_value_gbp", "max_order_value_gbp", "avg_n_instalments"]:
        out[col] = ex[col]
    out["total_instalment_exposure_gbp"] = ex["total_instalment_exposure_gbp"].fillna(0)
    out["n_merchant_categories"] = ex["n_merchant_categories"].fillna(0).astype("int64")
    for col in ["pct_high_risk_category", "pct_medium_risk_category", "pct_low_risk_category"]:
        out[col] = ex[col]
    zero_counts = {"n_instalments_total", "n_transactions_with_repayments", "n_late_payments", "total_days_late",
                   "n_transactions_with_late_payment", "n_reached_sicr", "n_terminal_current",
                   "n_terminal_delinquent", "n_terminal_default"}
    for col in ["n_instalments_total", "n_transactions_with_repayments", "n_late_payments", "pct_late_payments",
                "avg_days_late", "max_days_late", "total_days_late", "n_transactions_with_late_payment",
                "max_dpd_ever", "avg_max_dpd", "n_reached_sicr", "pct_reached_sicr", "max_cycles_delinquent",
                "avg_cycles_delinquent", "max_arrears_depth", "avg_arrears_depth_when_behind",
                "n_terminal_current", "n_terminal_delinquent", "n_terminal_default", "avg_instalment_amount_gbp"]:
        out[col] = rb[col].fillna(0).astype("int64") if col in zero_counts else rb[col]
    out["total_scheduled_repayment_gbp"] = rb["total_scheduled_repayment_gbp"].fillna(0)
    out["debt_to_income_ratio"] = round_half_up(units(c["existing_monthly_obligations_gbp"], 2),
                                                units(c["monthly_income_gbp"], 2), 4)
    out["bnpl_exposure_to_monthly_income_ratio"] = round_half_up(ex["ov_sum"].fillna(0),
                                                                 units(c["monthly_income_gbp"], 2), 4)
    first = pd.to_datetime(ex["first_transaction_date"])
    last = pd.to_datetime(ex["last_transaction_date"])
    out["first_transaction_date"] = first.dt.date
    out["last_transaction_date"] = last.dt.date
    out["transaction_span_days"] = (last - first).dt.days
    out["days_since_last_transaction"] = (SNAPSHOT - last).dt.days
    for col in ["age_bracket", "income_band", "employment_status", "ab_test_group"]:
        out[col] = c[col]
    return out.reset_index()


class PeakMemory:
    """Samples this process's resident memory every 50 ms."""

    def __init__(self):
        import psutil  # only needed when timing (not in the slim CI image)

        self.peak, self._stop = 0, threading.Event()
        self._proc = psutil.Process()

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(0.05):
            self.peak = max(self.peak, self._proc.memory_info().rss)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def run(dataset: str, out_dir: str, repeats: int) -> dict:
    folder = dataset_dir(dataset)
    os.makedirs(out_dir, exist_ok=True)
    runs, peak = [], 0
    for i in range(repeats):
        with PeakMemory() as mem:
            t = time.perf_counter()
            result = build(read(folder, "customers"), read(folder, "transactions"), read(folder, "repayments"))
            result.to_parquet(os.path.join(out_dir, "part-0.parquet"), index=False)
            runs.append(round(time.perf_counter() - t, 2))
        peak = max(peak, mem.peak)
        print(f"run {i + 1}: {runs[-1]} s, peak memory {mem.peak / 1e9:.2f} GB", flush=True)
        del result
    return {"engine": "pandas", "dataset": dataset, "runs_s": runs, "peak_memory_gb": round(peak / 1e9, 2),
            "pandas_version": pd.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.out, args.repeats)))


if __name__ == "__main__":
    main()
