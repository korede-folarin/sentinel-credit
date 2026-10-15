/* Sentinel Credit — in-browser scoring with the governed champion.
 *
 * Loads the ONNX model and its contract (model/model_contract.json) and
 * scores entirely in this browser with the bundled onnxruntime-web: nothing
 * typed is sent anywhere, stored, or logged.
 *
 * buildFeatures() is a line-for-line port of serving/features.py, the
 * module the FastAPI service and the Streamlit dashboard also use. The
 * parity page (parity.html) checks this port against Python-computed
 * examples, and the ONNX model against the Python champion.
 */
const ORT_DIR = "vendor/onnxruntime-web-1.19.2/";

// Same tiers as serving/features.py and sql/02_feature_store.sql.
export const MERCHANT_CATEGORIES = ["Fashion & Clothing", "Electronics", "Home", "Health & Beauty",
                                    "Sports", "Travel", "Food & Grocery", "Other"];
const HIGH_RISK = new Set(["Travel", "Electronics"]);
const MEDIUM_RISK = new Set(["Fashion & Clothing", "Home", "Other"]);
export const PROFILE_FIELDS = ["monthly_income_gbp", "existing_monthly_obligations_gbp",
  "savings_balance_gbp", "credit_score", "months_credit_history", "num_credit_accounts",
  "num_defaults_24m", "thin_file_flag"];

let session = null;
let contract = null;

const LOAD_TIMEOUT_MS = 45000;
let loading = null;

export function loadModel() {
  if (session) return Promise.resolve({ session, contract });
  if (!loading) {
    const work = (async () => {
      if (!window.ort) throw new Error("Scoring engine failed to load.");
      // Absolute URL: onnxruntime-web resolves a relative wasmPaths against its own
      // script, not this page, which would point it at a non-existent folder.
      window.ort.env.wasm.wasmPaths = new URL(ORT_DIR, document.baseURI).href;
      window.ort.env.wasm.numThreads = 1;   // static hosting cannot enable cross-origin isolation
      contract = await fetch("model/model_contract.json").then(r => {
        if (!r.ok) throw new Error("Model contract unavailable.");
        return r.json();
      });
      session = await window.ort.InferenceSession.create("model/sentinel_champion.onnx",
                                                         { executionProviders: ["wasm"] });
      return { session, contract };
    })();
    // Never hang silently: a stalled load becomes a visible error.
    const timeout = new Promise((_, reject) => setTimeout(
      () => reject(new Error("the model took too long to load")), LOAD_TIMEOUT_MS));
    loading = Promise.race([work, timeout]).catch(e => { loading = null; throw e; });
  }
  return loading;
}

/** Port of serving/features.py build_new_customer_features(). */
export function buildFeatures(profile, orderValueGbp, nInstalments, merchantCategory) {
  const missing = PROFILE_FIELDS.filter(f => profile[f] === null || profile[f] === undefined
                                             || Number.isNaN(profile[f]));
  if (missing.length) throw new Error(`New-customer profile fields are required: ${missing}`);
  if (!(orderValueGbp > 0) || !(nInstalments > 0) || !(profile.monthly_income_gbp > 0))
    throw new Error("order value, instalments and monthly income must be positive");
  const instalmentAmount = orderValueGbp / nInstalments;
  const affordability = (profile.existing_monthly_obligations_gbp + instalmentAmount)
                        / profile.monthly_income_gbp;
  const features = {};
  PROFILE_FIELDS.forEach(f => { features[f] = profile[f]; });
  // Python round(x, 4): correctly rounded from the exact binary value, as toFixed is.
  features.max_affordability_ratio = Number(affordability.toFixed(4));
  features.avg_order_value_gbp = orderValueGbp;
  features.avg_n_instalments = nInstalments;
  features.pct_high_risk_category = HIGH_RISK.has(merchantCategory) ? 1.0 : 0.0;
  features.pct_medium_risk_category = MEDIUM_RISK.has(merchantCategory) ? 1.0 : 0.0;
  return features;
}

/** Calibrated probability of default for each feature object, in contract order. */
export async function scoreMany(featureObjects) {
  const { session, contract } = await loadModel();
  const k = contract.features.length, n = featureObjects.length;
  const flat = new Float32Array(n * k);
  featureObjects.forEach((obj, i) => contract.features.forEach((f, j) => {
    if (!(f in obj)) throw new Error(`Missing model feature: ${f}`);
    flat[i * k + j] = obj[f];
  }));
  const out = await session.run({ [contract.input_name]: new window.ort.Tensor("float32", flat, [n, k]) });
  const probs = out[contract.probability_output].data;
  return Array.from({ length: n }, (_, i) => probs[i * 2 + 1]);
}

export async function scoreOne(features) {
  return (await scoreMany([features]))[0];
}

export function getContract() { return contract; }

export const fmtPct = (p, d = 1) => `${(p * 100).toFixed(d)}%`;
export const fmtGbp = v => `£${Math.round(v).toLocaleString("en-GB")}`;
