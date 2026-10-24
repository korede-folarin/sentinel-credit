/* Sentinel Credit — live assessment page logic. Uses model.js only; the
 * score, the comparison with a typical applicant and the "what would
 * change" search are all real model evaluations run in this browser. */
import { loadModel, buildFeatures, scoreMany, MERCHANT_CATEGORIES, PROFILE_FIELDS,
         fmtPct, fmtGbp } from "./model.js";
import { explain, loadReasonFiles, deriveThinFile } from "./reasons.js";

const $ = id => document.getElementById(id);

// Hard bounds for what a visitor may type; separate from the training-range warning.
const BOUNDS = {
  monthly_income_gbp: [1, 50000], existing_monthly_obligations_gbp: [0, 50000],
  savings_balance_gbp: [0, 1000000], credit_score: [0, 1000], months_credit_history: [0, 720],
  num_credit_accounts: [0, 50], num_defaults_24m: [0, 20], order_value_gbp: [1, 20000],
};
const LABELS = {
  monthly_income_gbp: "Monthly income", existing_monthly_obligations_gbp: "Existing obligations",
  savings_balance_gbp: "Savings balance", credit_score: "Credit score",
  months_credit_history: "Credit history (months)", num_credit_accounts: "Credit accounts",
  num_defaults_24m: "Defaults in 24 months", thin_file_flag: "Thin credit file",
  order_value_gbp: "Order value", n_instalments: "Instalments", merchant_category: "Merchant category",
  max_affordability_ratio: "Affordability ratio", avg_order_value_gbp: "Order value",
  avg_n_instalments: "Instalments",
};
const MONEY = new Set(["monthly_income_gbp", "existing_monthly_obligations_gbp",
                       "savings_balance_gbp", "order_value_gbp"]);
const INTEGER = new Set(["credit_score", "months_credit_history", "num_credit_accounts",
                         "num_defaults_24m"]);

let contract, bands, rules;

function show(field, v) {
  if (field === "thin_file_flag") return v ? "yes" : "no";
  if (field === "merchant_category" || field === "n_instalments") return String(v);
  if (MONEY.has(field)) return fmtGbp(v);
  return INTEGER.has(field) ? String(Math.round(v)) : v.toFixed(2);
}

function readInputs() {
  const v = {};
  for (const [f, [lo, hi]] of Object.entries(BOUNDS)) {
    const raw = $(f).value.trim();
    const x = Number(raw);
    if (raw === "" || !Number.isFinite(x)) throw new Error(`${LABELS[f]}: enter a number.`);
    if (x < lo || x > hi) throw new Error(`${LABELS[f]}: must be between ${lo.toLocaleString()} and ${hi.toLocaleString()}.`);
    if (INTEGER.has(f) && !Number.isInteger(x)) throw new Error(`${LABELS[f]}: must be a whole number.`);
    v[f] = x;
  }
  // Decision 34: worked out from the score, history and accounts; never asked
  v.thin_file_flag = deriveThinFile(v.credit_score, v.months_credit_history, v.num_credit_accounts, rules);
  v.n_instalments = Number($("n_instalments").value);
  v.merchant_category = $("merchant_category").value;
  return v;
}

function featuresOf(inp) {
  const profile = {};
  PROFILE_FIELDS.forEach(f => { profile[f] = inp[f]; });
  return buildFeatures(profile, inp.order_value_gbp, inp.n_instalments, inp.merchant_category);
}

function percentileOf(p) {
  // Share of held-out customers scoring below p, from aggregated counts.
  const { edges, groups } = bands;
  const all = groups.all;
  let below = 0, total = 0;
  for (let i = 0; i < edges.length - 1; i++) {
    const n = all.bad[i] + all.good[i];
    total += n;
    if (edges[i + 1] <= p) below += n;
  }
  return below / total;
}

function candidates(field, current) {
  if (field === "merchant_category") return MERCHANT_CATEGORIES.filter(c => c !== current);
  if (field === "n_instalments") return [3, 6, 12].filter(x => x !== current);
  if (field === "thin_file_flag") return [1 - current];
  const key = field === "order_value_gbp" ? "avg_order_value_gbp" : field;
  const { min, max } = contract.training_ranges[key];
  const out = [];
  for (let i = 0; i <= 60; i++) {
    let x = min + (max - min) * i / 60;
    x = INTEGER.has(field) ? Math.round(x) : Math.round(x);
    if (field === "monthly_income_gbp" && x < 1) continue;
    if (x !== current) out.push(x);
  }
  return [...new Set(out)];
}

async function assess(event) {
  event.preventDefault();
  $("form-error").textContent = "";
  let inp;
  try { inp = readInputs(); } catch (e) { $("form-error").textContent = e.message; return; }

  const btn = $("submit");
  btn.disabled = true; btn.textContent = "Scoring…";
  try {
    const cutoff = contract.cutoff;
    const base = featuresOf(inp);
    const fields = [...Object.keys(BOUNDS), "n_instalments", "merchant_category"];

    // Plain-English reasons (shared logic with the API and dashboard, Decision 30).
    const ex = await explain({ ...inp, thin_file_flag: null }, cutoff);   // derived inside, recorded as such

    // One batch: the applicant and every what-if candidate.
    const rows = [base], meta = [{ kind: "base" }];
    for (const f of fields) {
      for (const c of candidates(f, inp[f])) {
        const variant = { ...inp, [f]: c };
        variant.thin_file_flag = deriveThinFile(variant.credit_score, variant.months_credit_history,
                                                variant.num_credit_accounts, rules);
        rows.push(featuresOf(variant)); meta.push({ kind: "whatif", f, v: c });
      }
    }
    const probs = await scoreMany(rows);
    const p = probs[0], refer = p >= cutoff;

    $("r-pd").textContent = fmtPct(p, p < 0.1 ? 1 : 0);
    $("r-rec").innerHTML = refer ? '<span class="pill refer">Refer for manual review</span>'
                                 : '<span class="pill ok">Below review threshold</span>';
    $("r-rank").textContent = `Higher than about ${Math.round(percentileOf(p) * 100)}% of `
      + `${bands.n_customers.toLocaleString("en-GB")} held-out customers`;
    $("r-aff").textContent = base.max_affordability_ratio.toFixed(2);
    $("r-cutoff").textContent = `Review cutoff: probability of default ≥ ${fmtPct(cutoff, 2)}. `
      + "Affordability ratio = (existing obligations + this instalment) ÷ monthly income.";

    // Decision 7 / 30 / 31: three outputs from one shared module
    const esc = x => String(x).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    const x = ex.explanation, off = ex.risk_officer, rec = ex.decision_record;
    // 1 · for the applicant (Decision 33): outcome, why, check, changes, next
    const list = (id, items) => { $(id).innerHTML = items.map(i => `<li>${esc(i)}</li>`).join(""); };
    $("r-ex-headline").textContent = x.headline;
    $("r-ex-headline").className = { approved: "outcome-ok", manual_review: "outcome-review",
                                      declined: "outcome-declined" }[x.outcome];
    $("r-ex-summary").textContent = x.summary;
    $("r-ex-why-intro").textContent = x.why_intro;
    list("r-ex-why", x.why.map(r => r.text));
    $("r-ex-why").classList.toggle("hidden", !x.why.length);
    $("r-ex-changes-box").classList.toggle("hidden", !x.changes_intro);
    $("r-ex-changes-intro").textContent = x.changes_intro;
    list("r-ex-changes", x.changes);
    $("r-ex-changes-note").textContent = x.changes_note;
    // repayment plan (approved): real dates from today, 30 days apart
    $("r-ex-plan-box").classList.toggle("hidden", !x.plan);
    if (x.plan) {
      const day = d => new Date(Date.now() + d * 864e5).toLocaleDateString("en-GB",
        { day: "numeric", month: "long", year: "numeric" });
      const dated = { 3: day(x.plan.first_due_days), 4: day(x.plan.final_due_days) };
      $("r-ex-plan").innerHTML = x.plan.rows.map(([k, v], i) => `<tr><td>${esc(k)}</td>
        <td class="num"><b>${esc(v)}</b>${i in dated ? `<br><span class="note">${dated[i]}</span>` : ""}</td></tr>`).join("");
      $("r-ex-plan-note").textContent = x.plan.note;
    }
    list("r-ex-next", x.next_steps);
    // 2 · for a risk officer (Decision 34: plain language)
    $("r-off-pill").innerHTML = `<span class="pill ${{ "No concerns": "ok", "Declined": "refer" }[off.status]
      || "warn"}">${esc(off.status)}</span>`;
    $("r-off-detail").textContent = off.status_detail;
    $("r-off-why-box").classList.toggle("hidden", !off.review_reasons.length);
    list("r-off-why", off.review_reasons);
    list("r-off-risks", off.risk_factors.length ? off.risk_factors : ["None that moved the risk score."]);
    list("r-off-strengths", off.strengths.length ? off.strengths : ["None that moved the risk score."]);
    $("r-off-checks").innerHTML = off.checks.map(c => `<tr><td>${esc(c.name)}</td>
      <td>${esc(c.detail)}</td></tr>`).join("");
    list("r-off-track", off.track_record);
    $("r-off-method").textContent = off.method;
    // 3 · the decision record (shown, never stored or sent)
    const fullRecord = { ...rec, calibrated_probability: Number(ex.probability.toFixed(6)),
                         generated_at: new Date().toISOString().slice(0, 19) + "Z" };
    $("r-record").textContent = JSON.stringify(fullRecord, null, 2);

    // Comparison with a typical applicant (same effects the reasons use)
    const drivers = Object.entries(ex.effects).map(([f, delta]) => ({ f, delta }))
      .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 8);
    const scale = Math.max(...drivers.map(d => Math.abs(d.delta)), 1e-9);
    $("r-drivers").innerHTML = drivers.length ? drivers.map(d => {
      const w = Math.abs(d.delta) / scale * 50;
      const left = d.delta >= 0 ? 50 : 50 - w;
      const colour = d.delta >= 0 ? "var(--risk)" : "var(--accent)";
      return `<div class="bar-row"><div>${LABELS[d.f]}: <span class="note">${show(d.f, inp[d.f])}</span></div>
        <div class="bar-track"><div class="mid"></div><div class="bar-fill" style="left:${left}%;width:${w}%;background:${colour}"></div></div>
        <div class="bar-value">${d.delta >= 0 ? "+" : "−"}${(Math.abs(d.delta) * 100).toFixed(2)} pts</div></div>`;
    }).join("") : '<p class="note">This applicant is the typical applicant on every input.</p>';

    // Smallest single change that flips the recommendation
    const best = {};
    meta.forEach((m, i) => {
      if (m.kind !== "whatif" || (probs[i] >= cutoff) === refer) return;
      let dist;
      if (typeof m.v === "number" && typeof inp[m.f] === "number" && !["thin_file_flag", "n_instalments"].includes(m.f)) {
        const key = m.f === "order_value_gbp" ? "avg_order_value_gbp" : m.f;
        const r = contract.training_ranges[key];
        dist = Math.abs(m.v - inp[m.f]) / Math.max(r.max - r.min, 1e-9);
      } else dist = 0.5;   // categorical switches rank after small numeric moves
      if (!best[m.f] || dist < best[m.f].dist) best[m.f] = { ...m, dist, p: probs[i] };
    });
    const changes = Object.values(best).sort((a, b) => a.dist - b.dist).slice(0, 5);
    $("r-changes").innerHTML = changes.length ? changes.map(c =>
      `<li><b>${LABELS[c.f]}</b> ${show(c.f, inp[c.f])} → <b>${show(c.f, c.v)}</b>
        <span class="note">· probability ${fmtPct(c.p)}, ${c.p >= cutoff ? "refer for review" : "below threshold"}</span></li>`).join("")
      : '<li class="note">No single change within the training range moves this applicant across the cutoff.</li>';

    // Inputs outside the training range: an extrapolation, say so.
    const outside = Object.entries(base).filter(([f, v]) => {
      const r = contract.training_ranges[f]; return r && (v < r.min || v > r.max);
    }).map(([f]) => LABELS[f] || f);
    $("r-range").classList.toggle("hidden", !outside.length);
    if (outside.length) $("r-range").innerHTML = "<b>Outside the training range:</b> "
      + [...new Set(outside)].join(", ") + ". The score is an extrapolation.";

    $("r-scope").textContent = contract.scope;
    $("result").classList.remove("hidden");
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    $("form-error").textContent = `Scoring failed: ${e.message}`;
  } finally {
    btn.disabled = false; btn.textContent = "Assess application";
  }
}

(async () => {
  const sel = $("merchant_category");
  MERCHANT_CATEGORIES.forEach(c => sel.add(new Option(c, c)));
  try {
    ({ contract } = await loadModel());
    bands = await fetch("data/cutoff_simulator.json").then(r => r.json());
    const files = await loadReasonFiles();   // start from the typical applicant
    const typical = files.typical;
    rules = files.rules;
    for (const f of PROFILE_FIELDS) {
      if (f === "thin_file_flag") continue;
      $(f).value = typical[f];
    }
    $("order_value_gbp").value = typical.order_value_gbp;
    $("n_instalments").value = String(typical.n_instalments);
    $("merchant_category").value = typical.merchant_category;
    $("chips").innerHTML = [
      ["Model", `${contract.model} ${contract.model_version}`],
      ["Features", contract.features.length],
      ["Review cutoff", `PD ≥ ${fmtPct(contract.cutoff, 2)}`],
      ["Browser vs Python", "verified, 0 decisions changed"],
    ].map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join("");
    const showThin = () => {
      const v = ["credit_score", "months_credit_history", "num_credit_accounts"].map(f => Number($(f).value));
      if (v.some(x => !Number.isFinite(x))) return;
      $("thin-status").textContent = (deriveThinFile(v[0], v[1], v[2], rules)
        ? "Credit file: thin (little borrowing history)" : "Credit file: established")
        + ". Worked out from the credit score, history and accounts above.";
    };
    ["credit_score", "months_credit_history", "num_credit_accounts"].forEach(f => $(f).addEventListener("input", showThin));
    showThin();
    $("submit").disabled = false; $("submit").textContent = "Assess application";
    $("form").addEventListener("submit", assess);
  } catch (e) {
    $("form-error").textContent = `The model could not be loaded in this browser: ${e.message}`;
    $("submit").textContent = "Model unavailable";
  }
})();
