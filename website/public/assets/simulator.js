/* Sentinel Credit — cutoff and fairness simulator. Pure arithmetic over
 * aggregated counts (data/cutoff_simulator.json): flagged = score >= cutoff.
 * The same arithmetic was checked against evaluate_models.py's saved
 * results when the file was exported (website/tools/export_site_data.py). */
const $ = id => document.getElementById(id);
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const NS = "http://www.w3.org/2000/svg";
// Light theme: "gold" and "grey" keep the chart captions true; both pass 3:1 on white.
const C = { gold: "#A9741B", stone: "#6B7280", line: "#E4DCCF", text: "#18243A" };
const NAMES = { f1_optimal: "F1-optimal", youden_j: "Youden's J",
                review_capacity_constrained: "Review capacity 15%" };

let D;

function metrics(group, k) {          // k = first band flagged (edges[k] >= cutoff)
  let tp = 0, fp = 0, fn = 0, tn = 0;
  group.bad.forEach((n, i) => { if (i >= k) tp += n; else fn += n; });
  group.good.forEach((n, i) => { if (i >= k) fp += n; else tn += n; });
  const n = tp + fp + fn + tn;
  return { n, tp, fp, fn, tn, flagged: (tp + fp) / n, recall: tp / Math.max(tp + fn, 1),
           precision: tp + fp ? tp / (tp + fp) : null, fnr: fn / Math.max(tp + fn, 1),
           fpr: fp / Math.max(fp + tn, 1), badNotFlagged: (fn) / Math.max(fn + tn, 1) };
}

function el(tag, attrs, text) {
  const e = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
  if (text !== undefined) e.textContent = text;
  return e;
}

function lineChart(svg, series, k, yLabel) {
  svg.innerHTML = "";
  const W = 520, H = 300, L = 46, R = 14, T = 14, B = 40;
  const xMax = D.edges[D.edges.length - 1];
  const x = v => L + (v / xMax) * (W - L - R), y = v => T + (1 - v) * (H - T - B);
  for (let t = 0; t <= 1.0001; t += 0.25) {
    svg.append(el("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), stroke: C.line }));
    svg.append(el("text", { x: L - 8, y: y(t) + 4, "text-anchor": "end", fill: C.stone,
                            "font-size": 11 }, `${t * 100}%`));
  }
  const step = xMax > 0.1 ? 0.025 : 0.01;
  for (let v = 0; v <= xMax + 1e-9; v += step) {
    svg.append(el("text", { x: x(v), y: H - B + 18, "text-anchor": "middle", fill: C.stone,
                            "font-size": 11 }, `${(v * 100).toFixed(1)}%`));
  }
  svg.append(el("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", fill: C.stone,
                          "font-size": 11 }, "Review cutoff (probability of default)"));
  series.forEach(s => {
    const pts = D.edges.slice(0, -1).map((e, i) => `${x(e).toFixed(1)},${y(s.values[i]).toFixed(1)}`);
    svg.append(el("polyline", { points: pts.join(" "), fill: "none", stroke: s.colour,
                                "stroke-width": 2 }));
  });
  const cx = x(D.edges[k]);
  svg.append(el("line", { x1: cx, x2: cx, y1: T, y2: H - B, stroke: C.text,
                          "stroke-dasharray": "3 3", "stroke-width": 1 }));
  series.forEach(s => svg.append(el("circle", { cx, cy: y(s.values[k]), r: 4, fill: s.colour })));
  svg.append(el("title", {}, yLabel));
}

function render() {
  const k = Number($("cut").value);
  const cut = D.edges[k];
  const all = metrics(D.groups.all, k);
  $("cut-value").textContent = pct(cut, 2);
  const named = D.comparators.find(c => Math.abs(c.threshold - cut) < 1e-7);
  $("cut-label").textContent = Math.abs(cut - D.configured_cutoff) < 1e-7
    ? "Configured cutoff (review capacity 15%)" : named ? NAMES[named.method] || named.method : "";
  document.querySelectorAll("#marks button").forEach(b =>
    b.classList.toggle("active", Number(b.dataset.k) === k));

  $("s-flag").textContent = pct(all.flagged);
  $("s-recall").textContent = pct(all.recall);
  $("s-prec").textContent = all.precision === null ? "—" : pct(all.precision);
  $("s-bad").textContent = pct(all.badNotFlagged, 2);

  const seg = [["thin_file_flag=0", "Established credit file"], ["thin_file_flag=1", "Thin credit file"],
               ["is_vulnerable=0", "Not vulnerable"], ["is_vulnerable=1", "Vulnerable"]]
    .map(([key, name]) => ({ name, ...metrics(D.groups[key], k) }));
  $("seg-table").innerHTML = "<thead><tr><th>Segment</th><th class=num>Customers</th>"
    + "<th class=num>Defaulters</th><th class=num>Sent to review</th><th class=num>Missed</th>"
    + "<th class=num>Wrongly reviewed</th></tr></thead><tbody>"
    + seg.map(s => `<tr><td>${s.name}</td><td class=num>${s.n.toLocaleString("en-GB")}</td>`
      + `<td class=num>${(s.tp + s.fn).toLocaleString("en-GB")}</td><td class=num>${pct(s.flagged)}</td>`
      + `<td class=num>${pct(s.fnr)}</td><td class=num>${pct(s.fpr)}</td></tr>`).join("") + "</tbody>";

  const est = seg[0], thin = seg[1];
  $("finding").textContent = `At this cutoff the model misses ${pct(est.fnr, 0)} of defaulters with an `
    + `established credit file, against ${pct(thin.fnr, 0)} of thin-file defaulters, while `
    + `${pct(thin.fpr, 0)} of thin-file non-defaulters are sent to review (established: ${pct(est.fpr, 0)}). `
    + "Established-file customers default rarely, so even their riskiest members score low on the "
    + "shared probability scale, although the model still separates higher- from lower-risk customers "
    + "within that segment (segment diagnostics, Decision 19). Lowering the cutoff catches more "
    + "established-file defaults only by sending a much larger share of thin-file customers to review: "
    + "move the slider left to see it.";

  const fl = [], rc = [], fnEst = [], fnThin = [];
  for (let i = 0; i < D.edges.length - 1; i++) {
    const a = metrics(D.groups.all, i);
    fl.push(a.flagged); rc.push(a.recall);
    fnEst.push(metrics(D.groups["thin_file_flag=0"], i).fnr);
    fnThin.push(metrics(D.groups["thin_file_flag=1"], i).fnr);
  }
  lineChart($("chart-tradeoff"), [{ values: fl, colour: C.stone }, { values: rc, colour: C.gold }], k,
            "Share sent to review and defaults caught by cutoff");
  lineChart($("chart-fnr"), [{ values: fnEst, colour: C.stone }, { values: fnThin, colour: C.gold }], k,
            "Defaulters missed by credit-file segment and cutoff");
}

(async () => {
  try {
    D = await fetch("data/cutoff_simulator.json").then(r => r.json());
  } catch (e) {
    $("finding").textContent = "The simulator data could not be loaded.";
    return;
  }
  const slider = $("cut");
  slider.max = D.edges.length - 2;
  // Nearest band edge: edges are stored rounded to 8 dp, thresholds are not.
  const kOf = t => D.edges.reduce((best, e, i) =>
    Math.abs(e - t) < Math.abs(D.edges[best] - t) ? i : best, 0);
  slider.value = kOf(D.configured_cutoff);
  const marks = [{ label: `Configured ${pct(D.configured_cutoff, 2)}`, k: kOf(D.configured_cutoff) }]
    // The configured cutoff IS the review-capacity threshold, rounded for config
    // (0.0657 vs 0.06574): one policy point, one button.
    .concat(D.comparators.filter(c => Math.abs(c.threshold - D.configured_cutoff) > 1e-3)
      .map(c => ({ label: `${NAMES[c.method] || c.method} ${pct(c.threshold, 2)}`, k: kOf(c.threshold) })));
  $("marks").innerHTML = marks.map(m => `<button type="button" data-k="${m.k}">${m.label}</button>`).join("");
  $("marks").addEventListener("click", e => {
    if (e.target.dataset.k) { slider.value = e.target.dataset.k; render(); }
  });
  slider.addEventListener("input", render);
  $("chips").innerHTML = [["Model", `xgboost ${D.model_version}`],
    ["Held-out customers", D.n_customers.toLocaleString("en-GB")],
    ["Defaulters", D.n_defaulters.toLocaleString("en-GB")]]
    .map(([k, v]) => `<span class="chip">${k} <b>${v}</b></span>`).join("");
  render();
})();
