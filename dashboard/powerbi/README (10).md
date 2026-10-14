# Sentinel Credit, Power BI Setup

## What this is, and what it is not

A .pbix file is a binary format Power BI Desktop produces. It
cannot be generated as text, so it is not included here. What is
included is everything needed to build the real report by hand in
Power BI Desktop, connected directly to your live PostgreSQL
database, the same way a production risk reporting stack would
connect, not from a static export.

The setup mirrors a real production pattern on purpose. Live
portfolio and customer data comes from a DirectQuery connection to
PostgreSQL, through the views in sql/reporting/create_reporting_views.sql.
Model governance evidence, SHAP, drift, forecasting, threshold
selection, comes from a Folder connection to the reports directory,
since that evidence lives in files, not the operational database,
in every ML platform, not only this one.

## Step 1, create the reporting views

Run this once against your database, from PowerShell or psql:

```
psql -h localhost -U postgres -d sentinel_credit -f sql/reporting/create_reporting_views.sql
```

This creates a reporting schema with four read only views. Nothing
in raw or feature_store is modified.

## Step 2, connect Power BI Desktop to PostgreSQL

1. Open Power BI Desktop.
2. Home, Get Data, More, Database, PostgreSQL database.
3. Server: localhost. Database: sentinel_credit.
4. Data Connectivity mode: DirectQuery. This is what makes the
   report reflect the live database, not a snapshot taken at
   import time.
5. When prompted for credentials, use the postgres user and the
   same password used everywhere else in this project.
6. In Navigator, expand the reporting schema and select all four
   views: originations_monthly, default_rate_monthly,
   segment_summary, customer_risk_snapshot.
7. Select Load.

## Step 3, connect Power BI Desktop to the reports folder

1. Home, Get Data, More, File, Folder.
2. Point it at the reports directory inside this project.
3. Choose Combine, then Transform Data rather than Load directly,
   since the CSVs in this folder have different shapes for
   different purposes and should not be combined into one table.
4. In Power Query Editor, keep only the specific CSVs you want as
   separate queries, model_comparison_stratified.csv,
   model_comparison_temporal.csv, threshold_selection then
   threshold_candidates.csv, monitoring then feature_drift.csv,
   forecasting then the two forecast CSVs. Each becomes its own
   table.
5. Close and Apply.

## Step 4, the M query for the live views

If you prefer to write the PostgreSQL connection directly rather
than clicking through Navigator, use this M query as a starting
point in Power Query Advanced Editor. Replace the password
placeholder with your actual credential, or better, use a Power BI
parameter so the password is not stored in the .pbix file directly.

```
let
    Source = PostgreSQL.Database(
        "localhost",
        "sentinel_credit",
        [
            CommandTimeout = #duration(0, 0, 5, 0)
        ]
    ),
    reporting_schema = Source{[Schema="reporting"]}[Data],
    originations_monthly = reporting_schema{[Name="originations_monthly"]}[Data],
    default_rate_monthly = reporting_schema{[Name="default_rate_monthly"]}[Data],
    segment_summary = reporting_schema{[Name="segment_summary"]}[Data],
    customer_risk_snapshot = reporting_schema{[Name="customer_risk_snapshot"]}[Data]
in
    originations_monthly
```

## Step 5, relationships

Set a relationship between originations_monthly and
default_rate_monthly on the month column, one to one. Set a
relationship between segment_summary and customer_risk_snapshot on
thin_file_flag and is_vulnerable together, if you want drillthrough
from the segment summary down to individual customers. Do not
relate customer_risk_snapshot to any of the model governance CSVs
from the reports folder, since those are model run level evidence,
not customer level, joining them would misrepresent the grain of
the data.

## Step 6, DAX measures

Create these as new measures, not calculated columns, so they
recompute live against whatever filter context the report page
applies.

```
Total Originations = SUM(originations_monthly[n_originations])

Total Origination Value =
SUM(originations_monthly[total_value_gbp])

Overall Default Rate =
DIVIDE(
    SUM(default_rate_monthly[n_defaulted]),
    SUM(default_rate_monthly[n_resolved]),
    0
)

Features With Active Alerts =
CALCULATE(
    COUNTROWS(feature_drift),
    feature_drift[any_alert] = TRUE
)

Champion Gini =
CALCULATE(
    MAX('model_comparison_stratified'[gini]),
    'model_comparison_stratified'[model] = "xgboost"
)

Thin File Default Rate =
CALCULATE(
    AVERAGE(segment_summary[actual_default_rate]),
    segment_summary[thin_file_flag] = 1
)

Non Thin File Default Rate =
CALCULATE(
    AVERAGE(segment_summary[actual_default_rate]),
    segment_summary[thin_file_flag] = 0
)
```

These last two use segment_summary directly rather than
recomputing anything, since that view already applies the
has_outcome equal to 1 filter this project's whole modelling
pipeline relies on. Do not build a version of this measure against
raw.transactions or raw.repayments directly, it would double count
customers with multiple transactions and would not match the
customer level default rate reported everywhere else in this
project.

## Step 7, suggested pages

Page 1, Portfolio Overview. Originations and default rate over
time as line charts, using the live DirectQuery views, plus the
Champion Gini card from the reports folder data.

Page 2, Fairness and Segments. A table from segment_summary,
thin_file_flag and is_vulnerable on rows, actual_default_rate and
n_customers as values. Add a text box stating plainly that the
thin_file_flag equal to zero segment shows a structurally
suppressed recall at the current business cutoff, and that this is
an open governance item, Decision 12 and Decision 19, not
something this report resolves.

Page 3, Model Governance. The threshold_candidates and
feature_drift tables from the reports folder, plus the approval
gate status if you also bring in
reports/governance/approval_gate_result.json through a JSON
connector, Get Data, JSON, same Folder approach as the CSVs.

## A note on refresh

DirectQuery pages query the live database every time a user
interacts with a filter or slicer, so Portfolio Overview and
Fairness and Segments always reflect the current database state
with no manual refresh needed, matching a production reporting
pattern. The Folder connected reports pages only update when the
underlying CSV or JSON files change on disk, which happens each
time the relevant pipeline script is rerun, evaluate_models.py,
monitor_drift.py, forecast_portfolio.py. Schedule those scripts,
Decision 20 already schedules monitor_drift.py through Airflow, if
you want the governance pages to stay current without a manual
Power BI refresh as well.
