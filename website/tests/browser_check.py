"""
Sentinel Credit — headless browser check of the public demo site.

Launches headless Chrome (or Edge), serves nothing itself: point it at a
running static server (default http://127.0.0.1:8600, i.e.
`python -m http.server 8600 --directory website/public`). Drives each page
over the Chrome DevTools Protocol and asserts on real behaviour:

- parity.html: both checks (model; JS feature construction) report pass;
- index.html: a submitted application scores exactly as the Python champion
  does for the same applicant; invalid input is rejected with a message;
- simulator.html: metrics at the configured cutoff equal the saved
  evaluation evidence; comparator buttons move the cutoff;
- every page: no horizontal overflow at 1440 px and 390 px; screenshots
  written for visual review.

Usage:  python website/tests/browser_check.py [--out DIR] [--base URL]
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websocket

BROWSERS = [os.environ.get("CHROME_PATH", ""),
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "/usr/bin/google-chrome", "/usr/bin/chromium-browser", "/usr/bin/chromium"]


class Page:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=120, suppress_origin=True)
        self.n = 0

    def cmd(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr):
        r = self.cmd("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description",
                                                                             r["exceptionDetails"]))
        return r["result"].get("value")

    def wait_for(self, expr, timeout=90):
        end = time.time() + timeout
        while time.time() < end:
            if self.js(expr):
                return True
            time.sleep(0.25)
        raise TimeoutError(f"timed out waiting for: {expr}")

    def goto(self, url, width, height, mobile=False):
        self.cmd("Emulation.setDeviceMetricsOverride", width=width, height=height,
                 deviceScaleFactor=1, mobile=mobile)
        self.cmd("Page.navigate", url=url)
        self.wait_for("document.readyState === 'complete'")

    def screenshot(self, path):
        data = self.cmd("Page.captureScreenshot", format="png", captureBeyondViewport=True)["data"]
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))

    def overflow(self):
        return self.js("""(() => { const vw = document.documentElement.clientWidth;
            const off = [...document.querySelectorAll('body *')].filter(e => {
              const r = e.getBoundingClientRect(); return r.width > 0 && r.right > vw + 1
                && !e.closest('.table-scroll'); });
            return { pageScrolls: document.documentElement.scrollWidth > vw + 1,
                     offenders: off.slice(0, 5).map(e => e.tagName + '.' + e.className) }; })()""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8600/")
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "sentinel_site_check"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    exe = next(b for b in BROWSERS if b and os.path.exists(b))
    import socket
    with socket.socket() as sock:          # a free port, so stale runs cannot collide
        sock.bind(("127.0.0.1", 0))
        PORT = sock.getsockname()[1]
    profile = tempfile.mkdtemp(prefix="sc_browser_")
    flags = [exe, "--headless=new", f"--remote-debugging-port={PORT}",
             f"--user-data-dir={profile}", "--no-first-run", "--disable-extensions"]
    if os.name != "nt":                    # disposable CI runners (GitHub Actions)
        flags += ["--no-sandbox", "--disable-dev-shm-usage"]
    proc = subprocess.Popen(flags + ["about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failures = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    try:
        tabs = None
        for _ in range(120):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        if tabs is None:
            raise RuntimeError("Browser did not open its debugging port within 60 s.")
        target = next(t for t in tabs if t.get("type") == "page")
        page = Page(target["webSocketDebuggerUrl"])
        page.cmd("Runtime.enable")

        # ── parity.html ───────────────────────────────────────────
        page.goto(args.base + "parity.html", 1440, 900)
        page.wait_for("window.__parity && (window.__parity.model || window.__parity.modelError) "
                      "&& (window.__parity.features || window.__parity.featureError)", 120)
        par = page.js("window.__parity")
        m, f = par.get("model", {}), par.get("features", {})
        check("parity: model (browser vs Python)", bool(m.get("ok")),
              f"n={m.get('n')} max|d|={m.get('max')} flips={m.get('flips')} err={par.get('modelError')}")
        check("parity: JS feature construction end to end", bool(f.get("ok")),
              f"n={f.get('n')} feature mismatches={f.get('featMismatch')} max|d|={f.get('max')} "
              f"flips={f.get('flips')} err={par.get('featureError')}")

        # ── index.html ────────────────────────────────────────────
        page.goto(args.base, 1440, 900)
        page.wait_for("!document.getElementById('submit').disabled", 120)
        applicant = {"monthly_income_gbp": 2200, "existing_monthly_obligations_gbp": 300,
                     "savings_balance_gbp": 150, "credit_score": 610, "months_credit_history": 18,
                     "num_credit_accounts": 2, "num_defaults_24m": 0, "order_value_gbp": 500}
        fill = "".join(f"document.getElementById('{k}').value='{v}';" for k, v in applicant.items())
        page.js(fill + "document.getElementById('thin_file_flag').checked=false;"
                "document.getElementById('n_instalments').value='6';"
                "document.getElementById('merchant_category').value='Electronics';"
                "document.getElementById('form').requestSubmit(); true")
        page.wait_for("!document.getElementById('result').classList.contains('hidden')", 120)
        res = page.js("""(async () => {
            const m = await import('./assets/model.js');
            const p = await m.scoreOne(m.buildFeatures({monthly_income_gbp:2200,
              existing_monthly_obligations_gbp:300, savings_balance_gbp:150, credit_score:610,
              months_credit_history:18, num_credit_accounts:2, num_defaults_24m:0,
              thin_file_flag:0}, 500, 6, 'Electronics'));
            return { p, shown: document.getElementById('r-pd').textContent,
                     rec: document.getElementById('r-rec').innerText,
                     drivers: document.querySelectorAll('#r-drivers .bar-row').length,
                     changes: document.getElementById('r-changes').innerText };
        })()""")
        # Python champion, same applicant (dashboard/API parity test): 0.031834
        check("assessment: score equals Python champion", abs(res["p"] - 0.031834) < 5e-6,
              f"browser {res['p']:.6f} vs python 0.031834; shown {res['shown']}, {res['rec']}")
        check("assessment: comparison and what-if rendered",
              res["drivers"] > 0 and len(res["changes"]) > 0,
              f"{res['drivers']} driver bars; changes: {res['changes'][:120]!r}")
        page.screenshot(os.path.join(args.out, "index_desktop.png"))
        ov = page.overflow()
        check("assessment: no overflow at 1440px", not ov["pageScrolls"] and not ov["offenders"], str(ov))
        page.js("document.getElementById('monthly_income_gbp').value='-5';"
                "document.getElementById('form').requestSubmit(); true")
        time.sleep(0.5)
        err = page.js("document.getElementById('form-error').textContent")
        check("assessment: invalid input rejected", "Monthly income" in err, err)

        # ── simulator.html ───────────────────────────────────────
        page.goto(args.base + "simulator.html", 1440, 900)
        page.wait_for("document.getElementById('s-recall').textContent.length > 0", 60)
        sim = page.js("""({ cut: document.getElementById('cut-value').textContent,
            flag: document.getElementById('s-flag').textContent,
            recall: document.getElementById('s-recall').textContent,
            prec: document.getElementById('s-prec').textContent,
            rows: document.querySelectorAll('#seg-table tbody tr').length,
            paths: document.querySelectorAll('svg polyline').length })""")
        # evaluate_models.py at 0.0657: 1323 flagged of 8321, recall 0.5796, precision 0.0990
        check("simulator: matches saved evaluation at cutoff",
              sim["cut"] == "6.57%" and sim["flag"] == "15.9%" and sim["recall"] == "58.0%"
              and sim["prec"] == "9.9%", str(sim))
        check("simulator: charts and segment table rendered", sim["rows"] == 4 and sim["paths"] == 4, str(sim))
        page.screenshot(os.path.join(args.out, "simulator_desktop.png"))
        marks = page.js("[...document.querySelectorAll('#marks button')].map(b => b.textContent)")
        landed = []
        for i in range(len(marks)):
            page.js(f"document.querySelectorAll('#marks button')[{i}].click(); true")
            landed.append(page.js("document.getElementById('cut-value').textContent"))
        # each button must land on the threshold printed on it (e.g. "F1-optimal 7.64%")
        ok = len(marks) == 3 and all(m.endswith(v) for m, v in zip(marks, landed))
        check("simulator: each comparator button lands on its threshold", ok,
              f"{list(zip(marks, landed))}")

        # ── mobile ────────────────────────────────────────────────
        for name, path in [("index", ""), ("simulator", "simulator.html"), ("parity", "parity.html")]:
            page.goto(args.base + path, 390, 844, mobile=True)
            time.sleep(2.5)
            ov = page.overflow()
            check(f"{name}: no overflow at 390px", not ov["pageScrolls"] and not ov["offenders"], str(ov))
            page.screenshot(os.path.join(args.out, f"{name}_mobile.png"))
    finally:
        # Close the whole browser process tree, not just the launcher.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc.kill()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(1)
        shutil.rmtree(profile, ignore_errors=True)

    print(f"\nScreenshots: {args.out}")
    print("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
