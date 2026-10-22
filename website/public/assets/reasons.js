/* Sentinel Credit — applicant explanations and decision checks (Decisions 30, 31, 33).
 *
 * Line-for-line port of serving/reasons.py (new applicants), using the same
 * files as the API and the dashboard: model/reason_templates.json (wording),
 * model/typical_applicant.json and model/explanation_rules.json (limits
 * computed from the training partition). parity.html checks this port
 * against Python-generated outputs, word for word.
 *
 * Three outputs (Decision 7): the applicant's explanation (outcome first:
 * approved or needs a manual review; their own details; details to check;
 * model-tested changes; next steps; Decision 33), the risk officer view and
 * the decision record. Model issues are never shown to the applicant.
 */
import { buildFeatures, scoreMany, PROFILE_FIELDS } from "./model.js";

export const INPUTS = [...PROFILE_FIELDS, "order_value_gbp", "n_instalments", "merchant_category"];
export const FEATURES = [...PROFILE_FIELDS, "max_affordability_ratio", "avg_order_value_gbp",
  "avg_n_instalments", "pct_high_risk_category", "pct_medium_risk_category"];
const MONEY = new Set(["monthly_income_gbp", "existing_monthly_obligations_gbp",
                       "savings_balance_gbp", "order_value_gbp", "avg_order_value_gbp"]);
const PERCENT = new Set(["max_affordability_ratio", "pct_high_risk_category", "pct_medium_risk_category"]);
const FEATURE_TO_INPUT = { avg_order_value_gbp: "order_value_gbp", avg_n_instalments: "n_instalments" };

let files = null;

export async function loadReasonFiles() {
  if (!files) {
    const [templates, typical, rules] = await Promise.all([
      fetch("model/reason_templates.json").then(r => r.json()),
      fetch("model/typical_applicant.json").then(r => r.json()).then(d => d.values),
      fetch("model/explanation_rules.json").then(r => r.json()),
    ]);
    files = { templates, typical, rules };
  }
  return files;
}

const fill = (s, vals) => s.replace(/\{(\w+)\}/g, (m, k) => (k in vals ? String(vals[k]) : m));

export function formatValue(field, v, t) {
  if (field === "merchant_category") return String(v);
  if (MONEY.has(field)) return `£${Math.round(v).toLocaleString("en-GB")}`;
  if (PERCENT.has(field)) return `${Math.round(v * 100)}%`;
  if (field === "avg_n_instalments") return `${(Math.round(v * 10) / 10).toFixed(1)} instalments`;
  const n = Math.round(v);
  const units = t.units[field];
  if (units) return `${n} ${n === 1 ? units[0] : units[1]}`;
  return String(n);
}

export function featuresOf(inputs) {
  const profile = {};
  PROFILE_FIELDS.forEach(f => { profile[f] = inputs[f]; });
  return buildFeatures(profile, inputs.order_value_gbp, inputs.n_instalments, inputs.merchant_category);
}

/** [probability, {input: effect}] for every input that differs from typical. */
export async function effects(inputs, typ, score = scoreMany) {
  const variants = [featuresOf(inputs)], names = [];
  for (const f of INPUTS) {
    if (inputs[f] !== typ[f]) { variants.push(featuresOf({ ...inputs, [f]: typ[f] })); names.push(f); }
  }
  const probs = await score(variants);
  const eff = {};
  names.forEach((f, i) => { eff[f] = probs[0] - probs[i + 1]; });
  return [probs[0], eff];
}

function sentence(field, inputs, typ, effect, t) {
  let direction;
  if (field === "merchant_category" || field === "thin_file_flag") direction = effect > 0 ? "higher" : "lower";
  else direction = inputs[field] > typ[field] ? "higher" : "lower";
  return fill(t.fields[field][direction], {
    value: formatValue(field, inputs[field], t), typical: formatValue(field, typ[field], t) });
}

const round10 = v => 10 * Math.round(v / 10);
const capitalise = s => s.charAt(0).toUpperCase() + s.slice(1);

/** Same output as serving/reasons.py build_reasons(): {against, favour}. */
export function buildReasons(inputs, typ, effectByInput, t) {
  const tol = t.numerical_tolerance;
  const order = Object.fromEntries(INPUTS.map((f, i) => [f, i]));
  const entries = Object.entries(effectByInput);
  const against = entries.filter(([, e]) => e > tol).sort((a, b) => (b[1] - a[1]) || (order[a[0]] - order[b[0]]));
  const favour = entries.filter(([, e]) => e < -tol).sort((a, b) => (a[1] - b[1]) || (order[a[0]] - order[b[0]]));
  const n = t.max_reasons;
  return {
    against: against.slice(0, n).map(([f, e]) => ({ input: f, text: sentence(f, inputs, typ, e, t) })),
    favour: favour.slice(0, n).map(([f, e]) => ({ input: f, text: sentence(f, inputs, typ, e, t) })),
  };
}

// ── Decision 31: checks ───────────────────────────────────────────

function featureOf(name, feats) {
  if (name === "order_value_gbp") return "avg_order_value_gbp";
  if (name === "n_instalments") return "avg_n_instalments";
  if (name === "merchant_category") {
    return (feats.pct_medium_risk_category >= 1 && feats.pct_high_risk_category < 1)
      ? "pct_medium_risk_category" : "pct_high_risk_category";
  }
  return name;
}

export function bandText(feature, value, rules) {
  for (const b of rules.bands[feature]) {
    if ((b.lo === null || value >= b.lo) && (b.hi === null || value < b.hi)) return b.text;
  }
  return rules.overall.text;
}

function dataInputIssues(feats, inputs, rules, t) {
  const issues = [], words = t.issues;
  for (const f of FEATURES) {
    const r = rules.ranges[f], v = feats[f];
    if ((r.check_min && v < r.min) || v > r.max) {
      const key = FEATURE_TO_INPUT[f] || f;
      const shown = key in inputs ? inputs[key] : v;
      issues.push(fill(words.out_of_range, { label: t.fields[key].label, value: formatValue(key, shown, t) }));
    }
  }
  const c = rules.consistency;
  const months = formatValue("months_credit_history", feats.months_credit_history, t);
  const accounts = formatValue("num_credit_accounts", feats.num_credit_accounts, t);
  if (feats.thin_file_flag === 1) {
    if (feats.months_credit_history > c.thin_file_max_months_credit_history)
      issues.push(fill(words.thin_long_history, { months }));
    if (feats.num_credit_accounts > c.thin_file_max_num_credit_accounts)
      issues.push(fill(words.thin_many_accounts, { accounts }));
  } else {
    if (feats.months_credit_history < c.full_file_min_months_credit_history)
      issues.push(fill(words.full_short_history, { months }));
    if (feats.num_credit_accounts < c.full_file_min_num_credit_accounts)
      issues.push(fill(words.full_no_accounts, { accounts }));
  }
  if (feats.existing_monthly_obligations_gbp / feats.monthly_income_gbp > c.max_commitments_to_income) {
    issues.push(fill(words.commitments_vs_income, {
      obligations: formatValue("existing_monthly_obligations_gbp", feats.existing_monthly_obligations_gbp, t),
      income: formatValue("monthly_income_gbp", feats.monthly_income_gbp, t) }));
  }
  return issues;
}

function singleDetail(p, effectByInput, cutoff, share) {
  const entries = Object.entries(effectByInput);
  const total = entries.reduce((s, [, e]) => s + Math.abs(e), 0);
  if (total <= 0) return null;
  const rank = Object.fromEntries(INPUTS.map((f, i) => [f, i]));
  const [name, e] = [...entries].sort((a, b) => (Math.abs(b[1]) - Math.abs(a[1])) || (rank[a[0]] - rank[b[0]]))[0];
  const flips = (p >= cutoff) !== ((p - e) >= cutoff);
  return (flips && Math.abs(e) / total >= share) ? name : null;
}

// ── Decision 33: what could change the outcome ────────────────────

const ACTIONABLE = ["order_value_gbp", "n_instalments", "existing_monthly_obligations_gbp", "savings_balance_gbp"];
const SPAN_FEATURE = { order_value_gbp: "avg_order_value_gbp", n_instalments: "avg_n_instalments" };

export function changeCandidates(inputs, rules) {
  const out = [];
  let cur = inputs.order_value_gbp, seen = [];
  for (let k = 1; k < 20; k++) {
    const v = Math.max(20, round10(cur * (1 - k / 20)));
    if (v < cur && !seen.includes(v)) seen.push(v);
  }
  seen.forEach(v => out.push(["order_value_gbp", v]));
  [3, 6, 12].filter(n => n !== inputs.n_instalments).forEach(n => out.push(["n_instalments", n]));
  cur = inputs.existing_monthly_obligations_gbp; seen = [];
  for (let k = 1; k < 11; k++) {
    const v = round10(cur * (1 - k / 10));
    if (v < cur && !seen.includes(v)) seen.push(v);
  }
  seen.forEach(v => out.push(["existing_monthly_obligations_gbp", v]));
  cur = inputs.savings_balance_gbp; seen = [];
  const top = rules.ranges.savings_balance_gbp.max;
  for (let k = 1; k < 21; k++) {
    const v = round10(cur + (top - cur) * k / 20);
    if (v > cur && !seen.includes(v)) seen.push(v);
  }
  seen.forEach(v => out.push(["savings_balance_gbp", v]));
  return out;
}

async function outcomeChanges(inputs, cutoff, rules, t, score) {
  const cands = changeCandidates(inputs, rules);
  if (!cands.length) return [];
  const probs = await score(cands.map(([f, v]) => featuresOf({ ...inputs, [f]: v })));
  const line = rules.blind_spot.line;
  const fullFile = inputs.thin_file_flag === 0;
  const best = {};
  cands.forEach(([f, v], i) => {
    const p = probs[i];
    if (p >= cutoff || (fullFile && p >= line)) return;
    const r = rules.ranges[SPAN_FEATURE[f] || f];
    const d = Math.abs(v - inputs[f]) / (r.max - r.min);
    if (!(f in best) || d < best[f][0]) best[f] = [d, v];
  });
  return Object.entries(best)
    .sort((a, b) => (a[1][0] - b[1][0]) || (ACTIONABLE.indexOf(a[0]) - ACTIONABLE.indexOf(b[0])))
    .slice(0, 3)
    .map(([f, [, v]]) => fill(t.changes[f], { value: formatValue(f, v, t), current: formatValue(f, inputs[f], t) }));
}

// ── the three outputs ─────────────────────────────────────────────

/** Same three outputs, keys and text as serving/reasons.py review(). */
export async function review(inputs, feats, p, effectByInput, reasons, cutoff, t, rules, score = scoreMany) {
  const refer = p >= cutoff;
  const bs = rules.blind_spot;
  const issues = dataInputIssues(feats, inputs, rules, t);
  const fullFile = feats.thin_file_flag === 0;
  const blind = fullFile && p >= bs.line;
  const decisive = singleDetail(p, effectByInput, cutoff, rules.choices.single_detail_share);
  let errorType;
  if (issues.length) errorType = "data_input";
  else if (blind) errorType = "missing_information";
  else if (decisive) errorType = "overreliance";
  else errorType = "none";
  const required = errorType !== "none";
  const outcome = (refer || required) ? "manual_review" : "approved";

  const o = t.outcome[outcome];
  const why = outcome === "manual_review" ? reasons.against : reasons.favour;
  const showChanges = outcome === "manual_review" && errorType !== "data_input";
  const changes = showChanges ? await outcomeChanges(inputs, cutoff, rules, t, score) : [];
  const explanation = {
    outcome,
    headline: o.headline,
    summary: o.summary,
    why_intro: why.length ? o.why_intro : o.why_none,
    why,
    check_intro: issues.length ? t.check.intro : "",
    check: issues.map(s => capitalise(s) + "."),
    changes_intro: showChanges ? (changes.length ? t.changes_text.intro : t.changes_text.none) : "",
    changes,
    changes_note: changes.length ? t.changes_text.note : "",
    next_steps: [...o.next_steps],
    method: t.method.new,
  };

  const w = t.officer, cw = w.checks;
  const pct = Math.round(bs.percentile * 100);
  const blindDetail = !fullFile ? cw.missing_information.not_applicable
    : fill(cw.missing_information[blind ? "triggered" : "ok"], { pct, line: bs.line_text });
  const context = [];
  if (reasons.against.length) {
    const f = featureOf(reasons.against[0].input, feats);
    context.push(bandText(f, feats[f], rules));
  }
  if (blind) context.push(bs.text_segment, bs.text_top);
  const officer = {
    error_type: errorType,
    error_label: w.error_labels[errorType],
    review_required: required,
    model_recommendation: refer ? "refer" : "pass",
    customer_outcome: outcome,
    divergence: required && !refer,
    divergence_note: (required && !refer) ? w.divergence_note : "",
    checks: [
      { id: "data_input", name: cw.data_input.name, triggered: issues.length > 0,
        detail: issues.length ? issues.join("; ") : cw.data_input.ok },
      { id: "missing_information", name: cw.missing_information.name, triggered: blind, detail: blindDetail },
      { id: "overreliance", name: cw.overreliance.name, triggered: decisive !== null,
        detail: decisive ? fill(cw.overreliance.triggered, { name: t.fields[decisive].name }) : cw.overreliance.ok },
    ],
    context,
  };

  const m = rules.model;
  const record = {
    record_type: "sentinel_credit_decision_record",
    record_version: "2",
    applicant_type: "new",
    model_version: m.model_version,
    feature_set_label: m.feature_set_label,
    model_trained_at: m.trained_at,
    business_cutoff: m.cutoff,
    model_recommendation: refer ? "refer" : "pass",
    risk_officer_review_required: required,
    customer_outcome: outcome,
    divergence: required && !refer,
    error_type: errorType,
    rules_version: rules.rules_version,
    rules_generated_at: rules.generated_at,
    approver_id: null,
    approver_note: t.record.approver_note,
    lineage: t.record.lineage.map(s => fill(s, { model_version: m.model_version, cutoff: m.cutoff,
                                                 rules_version: rules.rules_version })),
  };
  return { explanation, risk_officer: officer, decision_record: record };
}

/** Score, explain and check one new applicant (raw inputs). */
export async function explain(inputs, cutoff, score = scoreMany) {
  const { templates: t, typical: typ, rules } = await loadReasonFiles();
  const [p, eff] = await effects(inputs, typ, score);
  const reasons = buildReasons(inputs, typ, eff, t);
  return { probability: p, effects: eff, typical: typ,
           ...(await review(inputs, featuresOf(inputs), p, eff, reasons, cutoff, t, rules, score)) };
}
