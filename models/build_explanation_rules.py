"""
Sentinel Credit — Build the explanation reference files (Decisions 30 and 31)

Everything the plain-English reasons and the decision checks compare
against, computed from the champion's canonical TRAINING partition:

serving/typical_applicant.json
    The typical applicant (Decision 30): training medians of the profile
    fields and order value, rounded to whole units, 3 instalments, the most
    common merchant category. Plus the typical existing customer: medians
    of all 13 model features, rounded to the precision they are shown at.

serving/explanation_rules.json  (Decision 31)
    - training ranges of every model feature (data-input check);
    - consistency limits seen in training: thin-file customers' longest
      history and most accounts, full-file customers' shortest history and
      fewest accounts, the highest commitments-to-income ratio;
    - the blind-spot line: the configured percentile of established-file
      customers' training out-of-fold scores, with held-out confirmation;
    - "past customers like you" bands: default rate per band of each
      feature, groups of at least min_group_size, text pre-rendered;
    - the model identity the rules were built for.

The declared choices (percentile, single-detail share, minimum group,
band quantiles) come from config.yaml explainability.decision_review.
They are choices, not findings; see Decision 31.

website/public/data/review_evidence.json
    How often each check fires on the champion's held-out test customers,
    and which details the "rests on one detail" cases rest on. Aggregates
    only (counts, rates, medians over groups); no customer row.

Usage:  python models/build_explanation_rules.py
Needs the PostgreSQL feature store (password from SENTINEL_DB_PASSWORD).
Writes copies to website/public/model/ for the public site.
"""

import json
import math
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from serving.features import PROFILE_FIELDS  # noqa: E402

SERVING = os.path.join(PROJECT_ROOT, "serving")
SITE_MODEL = os.path.join(PROJECT_ROOT, "website", "public", "model")
MOST_COMMON_CATEGORY = "Fashion & Clothing"   # config.yaml data_generator.merchant_categories (0.30)

# How each feature is shown in a "past customers like you" band:
# fmt is the display format, the phrases fill data.band's {condition}.
BAND_SPEC = {
    "monthly_income_gbp": ("money", "with a monthly income under {hi}",
                           "with a monthly income from {lo} to under {hi}",
                           "with a monthly income of {lo} or more"),
    "existing_monthly_obligations_gbp": ("money", "with monthly commitments under {hi}",
                                         "with monthly commitments from {lo} to under {hi}",
                                         "with monthly commitments of {lo} or more"),
    "savings_balance_gbp": ("money", "with savings under {hi}", "with savings from {lo} to under {hi}",
                            "with savings of {lo} or more"),
    "credit_score": ("int", "with a credit score under {hi}", "with a credit score from {lo} to under {hi}",
                     "with a credit score of {lo} or more"),
    "months_credit_history": ("int", "with under {hi} months of credit history",
                              "with {lo} to under {hi} months of credit history",
                              "with {lo} or more months of credit history"),
    "num_credit_accounts": ("count", "credit account", "credit accounts"),
    "num_defaults_24m": ("count", "default in the last two years", "defaults in the last two years"),
    "thin_file_flag": ("flag", "with a full credit file", "with a limited credit file"),
    "max_affordability_ratio": ("pct", "whose repayments took under {hi} of income",
                                "whose repayments took {lo} to under {hi} of income",
                                "whose repayments took {lo} or more of income"),
    "avg_order_value_gbp": ("money10", "whose orders averaged under {hi}",
                            "whose orders averaged {lo} to under {hi}",
                            "whose orders averaged {lo} or more"),
    "avg_n_instalments": ("dec1", "whose plans averaged under {hi} instalments",
                          "whose plans averaged {lo} to under {hi} instalments",
                          "whose plans averaged {lo} instalments or more"),
    "pct_high_risk_category": ("pct", "where under {hi} of purchases were travel or electronics",
                               "where {lo} to under {hi} of purchases were travel or electronics",
                               "where {lo} or more of purchases were travel or electronics"),
    "pct_medium_risk_category": ("pct", "where under {hi} of purchases were fashion, home or other goods",
                                 "where {lo} to under {hi} of purchases were fashion, home or other goods",
                                 "where {lo} or more of purchases were fashion, home or other goods"),
}


def half_up(v):
    return int(math.floor(v + 0.5))


def snap(fmt, v):
    """Round a band edge to the precision it is shown at, so shown == used."""
    if fmt == "money":
        return float(half_up(v))
    if fmt == "money10":
        return float(10 * half_up(v / 10))
    if fmt == "int":
        return float(half_up(v))
    if fmt == "pct":
        return half_up(v * 100) / 100
    if fmt == "dec1":
        return half_up(v * 10) / 10
    return float(v)


def show(fmt, v):
    if fmt in ("money", "money10"):
        return f"£{half_up(v):,}"
    if fmt == "pct":
        return f"{half_up(v * 100)}%"
    if fmt == "dec1":
        return f"{v:.1f}"
    return f"{half_up(v):,}"


def per_1000(bad, n):
    return half_up(1000 * bad / n)


def make_bands(feature, x, y, quantiles, min_group, t):
    spec = BAND_SPEC[feature]
    fmt = spec[0]
    if fmt in ("count", "flag"):
        edges = sorted(set(float(v) for v in np.unique(x)))[1:]
    else:
        edges = sorted(set(snap(fmt, q) for q in np.quantile(x, quantiles)))
    edges = [e for e in edges if x.min() < e <= x.max()]
    # [lo, hi) bands; None = open end
    bounds = [None] + edges + [None]
    bands = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        m = np.ones(len(x), bool)
        if lo is not None:
            m &= x >= lo
        if hi is not None:
            m &= x < hi
        bands.append({"lo": lo, "hi": hi, "n": int(m.sum()), "bad": int(y[m].sum())})
    # merge small groups into a neighbour until each meets min_group
    while len(bands) > 1 and min(b["n"] for b in bands) < min_group:
        i = min(range(len(bands)), key=lambda k: bands[k]["n"])
        j = i - 1 if i == len(bands) - 1 or (i > 0 and bands[i - 1]["n"] <= bands[i + 1]["n"]) else i + 1
        a, b = sorted((i, j))
        merged = {"lo": bands[a]["lo"], "hi": bands[b]["hi"], "n": bands[a]["n"] + bands[b]["n"],
                  "bad": bands[a]["bad"] + bands[b]["bad"]}
        bands[a:b + 1] = [merged]
    out = []
    for b in bands:
        out.append({"lo": b["lo"], "hi": b["hi"], "n": b["n"], "per_1000": per_1000(b["bad"], b["n"]),
                    "text": t["data"]["band"].format(n=f"{b['n']:,}", condition=condition(feature, b),
                                                     k=per_1000(b["bad"], b["n"]))})
    return out


def condition(feature, b):
    spec = BAND_SPEC[feature]
    fmt, lo, hi = spec[0], b["lo"], b["hi"]
    if fmt == "flag":
        if lo is None and hi is None:
            return "in the data"
        return spec[1] if lo is None else spec[2]
    if fmt == "count":
        one, many = spec[1], spec[2]
        lo_i = 0 if lo is None else half_up(lo)
        if hi is None:
            return f"with {lo_i} or more {many}"
        hi_i = half_up(hi) - 1
        if lo_i == hi_i:
            return f"with {lo_i} {one if lo_i == 1 else many}"
        return f"with {lo_i} to {hi_i} {many}"
    if lo is None:
        return spec[1].format(hi=show(fmt, hi))
    if hi is None:
        return spec[3].format(lo=show(fmt, lo))
    return spec[2].format(lo=show(fmt, lo), hi=show(fmt, hi))


def typical_values(frame):
    med = frame.median(numeric_only=True)
    new = {f: half_up(float(med[f])) for f in PROFILE_FIELDS}
    new["order_value_gbp"] = half_up(float(med["avg_order_value_gbp"]))
    new["n_instalments"] = 3
    new["merchant_category"] = MOST_COMMON_CATEGORY
    existing = {f: half_up(float(med[f])) for f in PROFILE_FIELDS}
    existing.update({
        "max_affordability_ratio": half_up(float(med["max_affordability_ratio"]) * 100) / 100,
        "avg_order_value_gbp": half_up(float(med["avg_order_value_gbp"])),
        "avg_n_instalments": half_up(float(med["avg_n_instalments"]) * 10) / 10,
        "pct_high_risk_category": half_up(float(med["pct_high_risk_category"]) * 100) / 100,
        "pct_medium_risk_category": half_up(float(med["pct_medium_risk_category"]) * 100) / 100,
    })
    return new, existing


def main():
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    choice = cfg["explainability"]["decision_review"]
    gov = cfg["model_governance"]
    pkg = os.path.join(PROJECT_ROOT, gov["artifacts_dir"])
    with open(os.path.join(pkg, "feature_list.json"), encoding="utf-8") as f:
        features = json.load(f)["features"]
    with open(os.path.join(pkg, "champion_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(SERVING, "reason_templates.json"), encoding="utf-8") as f:
        t = json.load(f)
    cutoff = float(cfg["model"]["business_cutoff"])

    ids = set(pd.read_csv(os.path.join(pkg, "split_ids_train.csv"))["customer_id"].astype(str))
    conn = psycopg2.connect(
        host=os.environ.get("SENTINEL_DB_HOST", cfg["database"]["host"]),
        port=cfg["database"]["port"], dbname=cfg["database"]["name"], user="postgres",
        password=os.environ.get("SENTINEL_DB_PASSWORD", ""))
    try:
        frame = pd.read_sql(f"SELECT customer_id, {', '.join(features)}, target_default "
                            "FROM feature_store.customer_features WHERE has_outcome = 1", conn)
    finally:
        conn.close()
    frame = frame[frame["customer_id"].astype(str).isin(ids)].reset_index(drop=True)
    y = frame["target_default"].to_numpy()
    print(f"Training partition: {len(frame):,} customers, {int(y.sum()):,} defaults")

    # ── typical applicant / customer (Decision 30) ─────────────
    new_typ, existing_typ = typical_values(frame)
    typical_doc = {
        "_about": ("Typical applicant for plain-English reasons (Decision 30): medians of the "
                   "champion's canonical training partition, rounded to whole units; 3 instalments "
                   "(Pay in 3, the dominant plan); the most common merchant category. "
                   "existing_customer_values: medians of the 13 model features, rounded to the "
                   "precision they are shown at, for existing customers."),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_version": gov["model_version"],
        "n_training_customers": int(len(frame)),
        "values": new_typ,
        "existing_customer_values": existing_typ,
    }

    # ── consistency limits (data-input check) ──────────────────
    thin = frame["thin_file_flag"] == 1
    ratio = frame["existing_monthly_obligations_gbp"] / frame["monthly_income_gbp"]
    consistency = {
        "thin_file_max_months_credit_history": int(frame.loc[thin, "months_credit_history"].max()),
        "thin_file_max_num_credit_accounts": int(frame.loc[thin, "num_credit_accounts"].max()),
        "full_file_min_months_credit_history": int(frame.loc[~thin, "months_credit_history"].min()),
        "full_file_min_num_credit_accounts": int(frame.loc[~thin, "num_credit_accounts"].min()),
        "max_commitments_to_income": round(float(ratio.max()), 4),
    }
    # Automatic thin-file status (Decision 34): the applicant is not asked.
    # In the data a thin file never exceeds these three limits together,
    # so an applicant inside all three is treated as thin-file.
    limits = {"max_credit_score": int(frame.loc[thin, "credit_score"].max()),
              "max_months_credit_history": int(frame.loc[thin, "months_credit_history"].max()),
              "max_num_credit_accounts": int(frame.loc[thin, "num_credit_accounts"].max())}
    derived = ((frame["credit_score"] <= limits["max_credit_score"])
               & (frame["months_credit_history"] <= limits["max_months_credit_history"])
               & (frame["num_credit_accounts"] <= limits["max_num_credit_accounts"])).astype(int)
    thin_rule = {**limits, "n": int(len(frame)),
                 "agree": int((derived == frame["thin_file_flag"]).sum()),
                 "derived_thin_but_full": int(((derived == 1) & ~thin).sum()),
                 "derived_full_but_thin": int(((derived == 0) & thin).sum())}
    skip_low = set(choice.get("no_lower_range_check", []))
    ranges = {f: {"min": float(frame[f].min()), "max": float(frame[f].max()),
                  "check_min": f not in skip_low} for f in features}

    # ── blind spot: established-file customers ─────────────────
    oof = pd.read_csv(os.path.join(PROJECT_ROOT, "reports", "threshold_selection",
                                   "xgboost_training_oof_predictions.csv"))
    rec = json.load(open(os.path.join(PROJECT_ROOT, "reports", "threshold_selection",
                                      "threshold_recommendation.json"), encoding="utf-8"))
    if rec.get("model_version") != gov["model_version"]:
        raise SystemExit(f"OOF predictions are for {rec.get('model_version')}, champion is "
                         f"{gov['model_version']}: re-run select_business_cutoff.py first.")
    oof = oof.merge(frame[["customer_id", "thin_file_flag"]], on="customer_id", how="inner")
    if len(oof) != len(frame):
        raise SystemExit(f"OOF covers {len(oof):,} of {len(frame):,} training customers.")
    seg = oof[oof["thin_file_flag"] == 0]
    pctl = float(choice["blind_spot_percentile"])
    line = float(np.quantile(seg["oof_probability_xgboost"], pctl))
    top = seg[seg["oof_probability_xgboost"] >= line]
    test = pd.read_csv(os.path.join(pkg, "test_predictions_stratified.csv"))
    tseg = test[test["thin_file_flag"] == 0]
    ttop = tseg[tseg["proba_xgboost"] >= line]
    top_pct = half_up((1 - pctl) * 100)
    blind = {
        "segment": "thin_file_flag = 0 (full credit file)",
        "percentile": pctl,
        "line": line,
        "line_text": f"{line * 100:.2f}%",
        "segment_max_oof": float(seg["oof_probability_xgboost"].max()),
        "segment_flagged_at_cutoff": int((seg["oof_probability_xgboost"] >= cutoff).sum()),
        "training": {"segment_n": int(len(seg)), "segment_defaults": int(seg["target_default"].sum()),
                     "top_n": int(len(top)), "top_defaults": int(top["target_default"].sum())},
        "heldout": {"segment_n": int(len(tseg)), "segment_defaults": int(tseg["target_default"].sum()),
                    "top_n": int(len(ttop)), "top_defaults": int(ttop["target_default"].sum()),
                    "segment_max_score": float(tseg["proba_xgboost"].max())},
        "text_segment": t["data"]["segment"].format(
            n=f"{len(seg):,}", k=per_1000(seg["target_default"].sum(), len(seg))),
        "text_top": t["data"]["segment_top"].format(
            top=top_pct, k=per_1000(top["target_default"].sum(), len(top))),
        "top_percent": top_pct,
    }

    # ── "past customers like you" bands ────────────────────────
    bands = {f: make_bands(f, frame[f].to_numpy(dtype=float), y, choice["band_quantiles"],
                           int(choice["min_group_size"]), t) for f in features}

    rules = {
        "_about": ("Decision 31 reference limits, computed from the champion's training partition by "
                   "models/build_explanation_rules.py. Declared choices (percentile, single-detail "
                   "share, minimum group, band quantiles) come from config.yaml "
                   "explainability.decision_review and are choices, not findings."),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rules_version": str(choice["rules_version"]),
        "model": {"model_version": gov["model_version"], "feature_set_label": gov["feature_set_label"],
                  "trained_at": meta["trained_at"], "cutoff": cutoff},
        "policy": {"affordability_limit": float(cfg["model"]["affordability_limit"])},
        "choices": {"blind_spot_percentile": pctl,
                    "single_detail_share": float(choice["single_detail_share"]),
                    "material_effect": float(choice["material_effect"]),
                    "min_group_size": int(choice["min_group_size"]),
                    "band_quantiles": list(choice["band_quantiles"]),
                    "no_lower_range_check": sorted(skip_low)},
        "n_training_customers": int(len(frame)),
        "ranges": ranges,
        "consistency": consistency,
        "thin_file_rule": thin_rule,
        "blind_spot": blind,
        "overall": {"n": int(len(frame)), "per_1000": per_1000(y.sum(), len(frame)),
                    "text": t["data"]["overall"].format(n=f"{len(frame):,}",
                                                        k=per_1000(y.sum(), len(frame)))},
        "data_method": t["data"]["method"].format(n=f"{len(frame):,}",
                                                  min_group=f"{int(choice['min_group_size']):,}"),
        "bands": bands,
    }

    for name, doc in [("typical_applicant.json", typical_doc), ("explanation_rules.json", rules)]:
        with open(os.path.join(SERVING, name), "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write("\n")
    os.makedirs(SITE_MODEL, exist_ok=True)
    for name in ("typical_applicant.json", "explanation_rules.json", "reason_templates.json"):
        shutil.copyfile(os.path.join(SERVING, name), os.path.join(SITE_MODEL, name))

    # ── held-out evidence for the site (aggregates only) ───────
    ev = heldout_evidence(pkg, features, cutoff, t, int(choice["min_group_size"]))
    ev["model_version"] = gov["model_version"]
    ev["rules_version"] = str(choice["rules_version"])
    os.makedirs(os.path.join(PROJECT_ROOT, "website", "public", "data"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "website", "public", "data", "review_evidence.json"),
              "w", encoding="utf-8") as f:
        json.dump(ev, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Held-out review: {ev['review_required']:,} of {ev['n_customers']:,} "
          f"({ev['review_share']:.1%}); by type "
          + ", ".join(f"{x['error_type']} {x['n']}" for x in ev["by_error_type"]))
    for x in ev["single_detail_by_detail"]:
        print("  one detail:", x)

    # ── evidence printout (recorded in Decision 31) ────────────
    print("Typical applicant:", new_typ)
    print("Typical existing customer:", existing_typ)
    print("Consistency limits:", consistency)
    print(f"Thin-file rule: {thin_rule} ({thin_rule['agree'] / thin_rule['n']:.2%} agreement)")
    b = blind
    print(f"Blind spot: full-file max OOF {b['segment_max_oof']:.4f}, flagged at cutoff "
          f"{b['segment_flagged_at_cutoff']}; {pctl:.0%} line {line:.4f}")
    for k in ("training", "heldout"):
        s = b[k]
        print(f"  {k}: segment {s['segment_n']:,} ({1000 * s['segment_defaults'] / s['segment_n']:.1f}/1000), "
              f"top {s['top_n']:,} ({1000 * s['top_defaults'] / max(s['top_n'], 1):.1f}/1000), "
              f"top captures {s['top_defaults']}/{s['segment_defaults']} segment defaults")
    print(rules["overall"]["text"])
    for f in features:
        print(f"  {f}: " + " | ".join(f"{x['n']:,}:{x['per_1000']}" for x in bands[f]))
    print("  e.g.", bands["credit_score"][0]["text"])


def heldout_evidence(pkg, features, cutoff, t, min_group):
    """Run the shipped checks on every held-out test customer; aggregates only."""
    import joblib
    from serving.reasons import effects, explain_existing, load_typical
    model = joblib.load(os.path.join(pkg, "champion_calibrated.joblib"))

    def score_many(ds):
        return model.predict_proba(pd.DataFrame(ds)[features])[:, 1].tolist()

    test = pd.read_csv(os.path.join(pkg, "test_predictions_stratified.csv"))
    typical = load_typical("existing")
    rows = []
    for _, r in test.iterrows():
        x = {f: r[f] for f in features}
        out = explain_existing(x, score_many, cutoff, typical)
        off = out["risk_officer"]
        detail, share, p_typ = None, None, None
        if off["error_type"] == "overreliance":
            eff = out["effects"]
            detail = max(eff, key=lambda k: abs(eff[k]))
            share = abs(eff[detail]) / sum(abs(v) for v in eff.values())
            p_typ = out["probability"] - eff[detail]
        rows.append({"error_type": off["error_type"], "refer": out["probability"] >= cutoff,
                     "p": out["probability"], "detail": detail, "share": share, "p_typ": p_typ,
                     "default": int(r["target_default"])})
    d = pd.DataFrame(rows)
    n = len(d)
    labels = {"none": "No check needed a person",
              **{k: v["name"] for k, v in t["officer"]["checks"].items()}}
    by_type = []
    for k in ("none", "missing_information", "overreliance", "data_input"):
        g = d[d["error_type"] == k]
        by_type.append({"error_type": k, "label": labels[k], "n": int(len(g)),
                        "share": round(len(g) / n, 4), "referred": int(g["refer"].sum()),
                        "defaults": int(g["default"].sum())})
    one = d[d["error_type"] == "overreliance"]
    by_detail = []
    for (det, refer), g in one.groupby(["detail", "refer"]):
        item = {"detail": det, "name": t["fields"][det]["name"], "recommendation": "refer" if refer else "pass",
                "n": int(len(g))}
        if len(g) >= 20:   # medians only over groups large enough not to describe a person
            item.update(median_probability=round(float(g["p"].median()), 4),
                        median_probability_if_typical=round(float(g["p_typ"].median()), 4),
                        median_share_of_effect=round(float(g["share"].median()), 2))
        by_detail.append(item)
    by_detail.sort(key=lambda x: -x["n"])
    review = d[d["error_type"] != "none"]
    return {
        "_about": ("Decision 31 checks run on every held-out test customer of the champion "
                   "(stratified split), by models/build_explanation_rules.py. Aggregates only."),
        "n_customers": n, "cutoff": cutoff,
        "review_required": int(len(review)), "review_share": round(len(review) / n, 4),
        "by_error_type": by_type, "single_detail_by_detail": by_detail,
    }


if __name__ == "__main__":
    main()
