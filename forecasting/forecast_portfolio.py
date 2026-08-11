"""
Sentinel Credit — Portfolio Forecasting (Prophet)

Purpose
-------
Forecasts two monthly portfolio series built by
forecasting/build_portfolio_timeseries.py: origination volume and
origination-cohort default rate. Backtests against held-out
months before producing any forward-looking forecast, so forecast
quality is evidence-based rather than asserted.

Prophet only — no LSTM
-----------------------
The architecture's Intelligence Core / Forecasting block names
"Prophet + LSTM". Only Prophet is implemented here, deliberately,
not as an oversight: the available series has ~34 monthly points.
An LSTM (or any neural sequence model) needs substantially more
history to learn a real pattern rather than memorize noise: with
this little data a trained LSTM would report high train-set
"accuracy" while having learned nothing generalizable, which is a
worse outcome than not building it — the same reasoning this
project has applied to declining invented cost ratios and
illustrative thresholds elsewhere. LSTM is recorded as future work
once meaningfully more monthly history exists, not silently
dropped.

The originations series requires its own caveat, more important
than the modelling choice: config.yaml's data_generator sets a
fixed origination_end_date (2024-10-31) and a fixed total
population (n_customers: 50000). The near-exponential growth
visible in originations_monthly.csv reflects the SYNTHETIC
POPULATION'S ONBOARDING SCHEDULE reaching that fixed endpoint, not
organic market demand. A forecast extending past the data's own
end date extrapolates a synthetic generation-schedule artifact,
not a business trend — this is stated in every output this script
produces, not just here.

The default-rate series does not have this problem to the same
degree — it reflects portfolio risk evolving over the same period,
which is a legitimate thing to forecast, though still on
synthetic data (Decision 4 addendum's PD-only, synthetic-population
scope applies here as everywhere else in this project).

No yearly seasonality is fit by default: ~34 points is barely 2.5
annual cycles, too little to distinguish genuine seasonality from
noise with any confidence — forcing it on risks fitting noise as
a seasonal pattern. Available via --yearly-seasonality if a
specific reason exists to test for it regardless.

See docs/engineering_decision_log.md — Decision 4 addendum
(PD-only, synthetic scope).
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "forecasting")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "forecasting")


def load_series(filename: str, value_column: str) -> pd.DataFrame:
    path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run "
            "forecasting/build_portfolio_timeseries.py first."
        )
    df = pd.read_csv(path, parse_dates=["month"])
    return df[["month", value_column]].rename(
        columns={"month": "ds", value_column: "y"}
    )


def fit_and_backtest(
    series: pd.DataFrame,
    series_name: str,
    holdout_months: int,
    forecast_horizon_months: int,
    yearly_seasonality: bool,
) -> tuple:
    """
    Fits on all but the last holdout_months, predicts across the
    holdout AND forecast_horizon_months beyond the data's actual
    end, then reports backtest error against the held-out actuals
    — the forecast beyond the data's end is never validated
    against anything (there is nothing to validate it against),
    which is exactly why it is reported separately and captioned
    as unvalidated, not blended into the backtest metrics.
    """
    n = len(series)
    if n < holdout_months + 5:
        raise ValueError(
            f"{series_name}: only {n} months available, need at least "
            f"{holdout_months + 5} (holdout + minimum viable training "
            "history) to backtest meaningfully."
        )

    train = series.iloc[: n - holdout_months].copy()
    holdout = series.iloc[n - holdout_months :].copy()

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(train)

    future = model.make_future_dataframe(
        periods=holdout_months + forecast_horizon_months, freq="MS"
    )
    forecast = model.predict(future)

    forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
        series[["ds", "y"]], on="ds", how="left"
    )
    forecast_out["is_holdout"] = forecast_out["ds"].isin(holdout["ds"])
    forecast_out["is_beyond_data"] = forecast_out["ds"] > series["ds"].max()

    backtest = forecast_out[forecast_out["is_holdout"]].copy()
    mae = float((backtest["y"] - backtest["yhat"]).abs().mean())
    # MAPE guarded against near-zero actuals (default_rate is a
    # small percentage) — reported alongside MAE, not instead of
    # it, since MAPE alone can be misleading at small denominators.
    nonzero = backtest[backtest["y"].abs() > 1e-9]
    mape = (
        float(((nonzero["y"] - nonzero["yhat"]).abs() / nonzero["y"].abs() * 100).mean())
        if len(nonzero) else None
    )

    metrics = {
        "n_total_months": n,
        "n_train_months": len(train),
        "n_holdout_months": holdout_months,
        "mae": round(mae, 6),
        "mape_pct": round(mape, 2) if mape is not None else None,
        "backtest_actual_mean": round(float(backtest["y"].mean()), 6),
    }
    return forecast_out, metrics, model


def plot_forecast(forecast_out: pd.DataFrame, series: pd.DataFrame, series_name: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    history = forecast_out[~forecast_out["is_beyond_data"]]
    beyond = forecast_out[forecast_out["is_beyond_data"]]

    ax.plot(series["ds"], series["y"], "o", color="#1B2A4A", markersize=3, label="Actual")
    ax.plot(history["ds"], history["yhat"], color="#1B2A4A", linewidth=1, alpha=0.5, label="Fitted / backtest")
    ax.fill_between(history["ds"], history["yhat_lower"], history["yhat_upper"], color="#1B2A4A", alpha=0.1)

    ax.plot(beyond["ds"], beyond["yhat"], "--", color="#B0413E", linewidth=1.5,
             label="Forecast beyond data (unvalidated)")
    ax.fill_between(beyond["ds"], beyond["yhat_lower"], beyond["yhat_upper"], color="#B0413E", alpha=0.15)

    ax.axvline(series["ds"].max(), color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"Sentinel Credit — {series_name} (Prophet)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_series(
    filename: str, value_column: str, series_name: str, caveat: str,
    holdout_months: int, forecast_horizon_months: int, yearly_seasonality: bool,
) -> dict:
    print("\n" + "-" * 62)
    print(series_name.upper())
    print("-" * 62)
    print(f"  {caveat}")

    series = load_series(filename, value_column)
    forecast_out, metrics, _ = fit_and_backtest(
        series, series_name, holdout_months, forecast_horizon_months, yearly_seasonality
    )

    print(f"\n  Backtest (last {holdout_months} months held out):")
    print(f"    MAE:  {metrics['mae']}")
    print(f"    MAPE: {metrics['mape_pct']}%" if metrics['mape_pct'] is not None else "    MAPE: n/a")

    slug = series_name.lower().replace(" ", "_")
    forecast_path = os.path.join(OUTPUT_DIR, f"forecast_{slug}.csv")
    plot_path = os.path.join(OUTPUT_DIR, f"forecast_{slug}.png")
    forecast_out.to_csv(forecast_path, index=False)
    plot_forecast(forecast_out, series, series_name, plot_path)

    print(f"\n  Saved: {forecast_path}")
    print(f"  Saved: {plot_path}")

    return {
        "series_name": series_name,
        "caveat": caveat,
        "metrics": metrics,
        "forecast_path": forecast_path,
        "plot_path": plot_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-months", type=int, default=6)
    parser.add_argument("--forecast-horizon-months", type=int, default=6)
    parser.add_argument("--yearly-seasonality", action="store_true", default=False)
    args = parser.parse_args()

    print("=" * 62)
    print("SENTINEL CREDIT — PORTFOLIO FORECASTING (PROPHET)")
    print("=" * 62)
    print("No LSTM — see module docstring for why, given the current "
          "~34-month series length.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []

    results.append(run_series(
        filename="default_rate_monthly.csv",
        value_column="default_rate",
        series_name="Default Rate",
        caveat=(
            "Reflects portfolio risk evolving over the observed period — "
            "a legitimate forecasting target, though still synthetic "
            "data (PD-only scope, Decision 4 addendum)."
        ),
        holdout_months=args.holdout_months,
        forecast_horizon_months=args.forecast_horizon_months,
        yearly_seasonality=args.yearly_seasonality,
    ))

    results.append(run_series(
        filename="originations_monthly.csv",
        value_column="n_originations",
        series_name="Origination Volume",
        caveat=(
            "CAUTION: growth reflects the synthetic population's fixed "
            "onboarding schedule (config.yaml data_generator.n_customers "
            "= 50000, origination_end_date = 2024-10-31) reaching its "
            "endpoint, not organic demand. Forecast beyond the data's "
            "own end date extrapolates a generation-schedule artifact, "
            "not a business trend."
        ),
        holdout_months=args.holdout_months,
        forecast_horizon_months=args.forecast_horizon_months,
        yearly_seasonality=args.yearly_seasonality,
    ))

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "holdout_months": args.holdout_months,
        "forecast_horizon_months": args.forecast_horizon_months,
        "yearly_seasonality": args.yearly_seasonality,
        "lstm_status": (
            "not implemented — insufficient history (~34 monthly points) "
            "to train a neural sequence model without overfitting to "
            "noise; see module docstring"
        ),
        "results": [
            {k: v for k, v in r.items() if k != "forecast_path" and k != "plot_path"}
            for r in results
        ],
    }
    metadata_path = os.path.join(OUTPUT_DIR, "forecast_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 62)
    print(f"Saved: {metadata_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
