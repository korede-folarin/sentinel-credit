/* Sentinel Credit — plain-English reasons and decision checks (Decisions 30, 31).
 *
 * Line-for-line port of serving/reasons.py (new applicants), using the same
 * files as the API and the dashboard: model/reason_templates.json (wording),
 * model/typical_applicant.json and model/explanation_rules.json (limits
 * computed from the training partition). parity.html checks this port
 * against Python-generated outputs, word for word.
 *
 * Three outputs (Decision 7): the explanation for the applicant, in a genre
 * matched to the error type detected; the risk officer view; the decision
 * record. "Risk officer review required" is a statement, not a routing.
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

function join(labels) {
  if (labels.length <= 1) return labels.join("");
  return labels.slice(0, -1).join(", ") + " and " + labels[labels.length - 1];
}

/** Same output structure and text as serving/reasons.py build_reasons(). */
export function buildReasons(inputs, typ, effectByInput, refer, t) {
  const tol = t.numerical_tolerance;
  const order = Object.fromEntries(INPUTS.map((f, i) => [f, i]));
  const entries = Object.entries(effectByInput);
  const against = entries.filter(([, e]) => e > tol).sort((a, b) => (b[1] - a[1]) || (order[a[0]] - order[b[0]]));
  const favour = entries.filter(([, e]) => e < -tol).sort((a, b) => (a[1] - b[1]) || (order[a[0]] - order[b[0]]));
  let main, side, block, sideText;
  if (refer) {
    main = against.slice(0, t.max_reasons); side = favour.slice(0, t.max_favourable); block = t.refer;
    sideText = side.length ? `${block.favour_intro} ${join(side.map(([f]) => t.fields[f].label))}.` : "";
  } else {
    main = favour.slice(0, t.max_reasons); side = against.slice(0, t.max_favourable); block = t.pass;
    sideText = side.length ? `${block.against_intro} ${join(side.map(([f]) => t.fields[f].label))}.` : "";
  }
  return {
    decision: refer ? "refer" : "pass",
    title: block.title,
    intro: main.length ? block.intro : block.none,
    reasons: main.map(([f, e]) => ({ input: f, text: sentence(f, inputs, typ, e, t) })),
    link: (refer && main.length) ? (block.link || "") : "",
    other: sideText,
    next_steps: [...t.next_steps[refer ? "refer" : "pass"]],
    method: t.method.new,
  };
}

// ── Decision 31: checks, genre, three outputs ─────────────────────

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
  const issues = [], words = t.review.issues;
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

/** Same three outputs, keys and text as serving/reasons.py review(). */
export function review(inputs, feats, p, effectByInput, reasons, cutoff, t, rules) {
  const refer = p >= cutoff;
  const bs = rules.blind_spot;
  const issues = dataInputIssues(feats, inputs, rules, t);
  const fullFile = feats.thin_file_flag === 0;
  const blind = fullFile && p >= bs.line;
  const decisive = singleDetail(p, effectByInput, cutoff, rules.choices.single_detail_share);

  let errorType, genre;
  if (issues.length) { errorType = "data_input"; genre = "features"; }
  else if (blind) { errorType = "missing_information"; genre = "data_centric"; }
  else if (decisive) { errorType = "overreliance"; genre = "data_centric"; }
  else { errorType = "none"; genre = "combination"; }
  const required = errorType !== "none";

  const ex = { ...reasons };
  const words = t.review;
  let data = [];
  if (genre === "combination") {
    if (ex.reasons.length) {
      const f = featureOf(ex.reasons[0].input, feats);
      data = [bandText(f, feats[f], rules)];
    } else data = [rules.overall.text];
  } else if (errorType === "overreliance") {
    const f = featureOf(decisive, feats);
    data = [bandText(f, feats[f], rules), rules.overall.text];
  } else if (errorType === "missing_information") {
    data = [bs.text_segment, bs.text_top, rules.overall.text];
  }
  if (genre === "data_centric") {
    ex.intro = words.data_only_intro; ex.reasons = []; ex.link = ""; ex.other = "";
  }
  let reviewText = "";
  if (errorType === "data_input") reviewText = fill(words.data_input, { issues: issues.join("; ") });
  else if (errorType === "missing_information") reviewText = fill(words.missing_information, { top: bs.top_percent });
  else if (errorType === "overreliance") reviewText = words.overreliance;
  let nextSteps = [...ex.next_steps];
  if (required && !refer) nextSteps = [words.next_step, ...nextSteps];

  const explanation = {
    decision: ex.decision,
    title: ex.title,
    review: { required, heading: required ? words.heading : "", text: reviewText },
    intro: ex.intro,
    reasons: ex.reasons,
    link: ex.link,
    other: ex.other,
    data_intro: (data.length && genre === "combination") ? words.data_intro : "",
    data,
    next_steps: nextSteps,
    method: genre === "data_centric" ? rules.data_method : ex.method,
  };

  const o = t.officer, cw = o.checks;
  const pct = Math.round(bs.percentile * 100);
  const blindDetail = !fullFile ? cw.missing_information.not_applicable
    : fill(cw.missing_information[blind ? "triggered" : "ok"], { pct, line: bs.line_text });
  const officer = {
    error_type: errorType,
    error_label: o.error_labels[errorType],
    genre,
    genre_reason: o.genre_reasons[genre],
    review_required: required,
    model_recommendation: refer ? "refer" : "pass",
    divergence: required && !refer,
    divergence_note: (required && !refer) ? o.divergence_note : "",
    checks: [
      { id: "data_input", name: cw.data_input.name, triggered: issues.length > 0,
        detail: issues.length ? issues.join("; ") : cw.data_input.ok },
      { id: "missing_information", name: cw.missing_information.name, triggered: blind, detail: blindDetail },
      { id: "overreliance", name: cw.overreliance.name, triggered: decisive !== null,
        detail: decisive ? fill(cw.overreliance.triggered, { name: t.fields[decisive].name }) : cw.overreliance.ok },
    ],
  };

  const m = rules.model;
  const record = {
    record_type: "sentinel_credit_decision_record",
    record_version: "1",
    applicant_type: "new",
    model_version: m.model_version,
    feature_set_label: m.feature_set_label,
    model_trained_at: m.trained_at,
    business_cutoff: m.cutoff,
    model_recommendation: refer ? "refer" : "pass",
    risk_officer_review_required: required,
    divergence: required && !refer,
    error_type: errorType,
    explanation_genre: genre,
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
  const reasons = buildReasons(inputs, typ, eff, p >= cutoff, t);
  return { probability: p, effects: eff, typical: typ,
           ...review(inputs, featuresOf(inputs), p, eff, reasons, cutoff, t, rules) };
}
