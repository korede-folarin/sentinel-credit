/* Sentinel Credit — plain-English reasons for a new applicant (Decision 30).
 *
 * Line-for-line port of serving/reasons.py, using the same wording file
 * (model/reason_templates.json) and typical applicant
 * (model/typical_applicant.json) as the API and the dashboard. parity.html
 * checks this port against Python-generated explanations, word for word.
 *
 * Method: each input's effect = P(applicant) − P(applicant with that one
 * input set to the typical value). It describes how the model responds,
 * not what causes default.
 */
import { buildFeatures, scoreMany, PROFILE_FIELDS } from "./model.js";

export const INPUTS = [...PROFILE_FIELDS, "order_value_gbp", "n_instalments", "merchant_category"];
const MONEY = new Set(["monthly_income_gbp", "existing_monthly_obligations_gbp",
                       "savings_balance_gbp", "order_value_gbp"]);

let templates = null, typical = null;

export async function loadReasonFiles() {
  if (!templates) {
    [templates, typical] = await Promise.all([
      fetch("model/reason_templates.json").then(r => r.json()),
      fetch("model/typical_applicant.json").then(r => r.json()).then(d => d.values),
    ]);
  }
  return { templates, typical };
}

export function formatValue(field, v, t) {
  if (field === "merchant_category") return String(v);
  if (MONEY.has(field)) return `£${Math.round(v).toLocaleString("en-GB")}`;
  const n = Math.round(v);
  const units = t.units[field];
  if (units) return `${n} ${n === 1 ? units[0] : units[1]}`;
  return String(n);
}

function featuresOf(inputs) {
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
  return t.fields[field][direction]
    .replace("{value}", formatValue(field, inputs[field], t))
    .replace("{typical}", formatValue(field, typ[field], t));
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
    method: "Each detail was compared with a typical applicant by changing that one detail "
            + "and re-scoring with the model.",
  };
}

/** Score and explain one new applicant (raw inputs). */
export async function explain(inputs, cutoff, score = scoreMany) {
  const { templates: t, typical: typ } = await loadReasonFiles();
  const [p, eff] = await effects(inputs, typ, score);
  return { probability: p, effects: eff, typical: typ, ...buildReasons(inputs, typ, eff, p >= cutoff, t) };
}
