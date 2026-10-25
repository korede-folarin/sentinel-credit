"""
Sentinel Credit — capture the README screenshots (docs/images/) from the demo site.

Serve the site first, then run from the project root:
    python -m http.server 8600 --directory website/public
    python website/tools/capture_readme_images.py

Uses the browser driver from website/tests/browser_check.py and waits for the
count-up and risk-scale animations to finish, so the images show final values.
Re-run after any change to the site's look.
"""
import base64
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
spec = importlib.util.spec_from_file_location("bc", os.path.join(root, "website/tests/browser_check.py"))
bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)
BASE = "http://127.0.0.1:8600/"
exe = next(b for b in bc.BROWSERS if b and os.path.exists(b))
with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
prof = tempfile.mkdtemp()
proc = subprocess.Popen([exe, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={prof}",
                         "--no-first-run", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json")); break
        except Exception:
            time.sleep(0.5)
    page = bc.Page(next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"])
    page.cmd("Runtime.enable")

    def clip(path, top_sel, bottom_sel, pad=16):
        r = page.js(f"""(() => {{ const a = document.querySelector('{top_sel}').getBoundingClientRect(),
            b = document.querySelector('{bottom_sel}').getBoundingClientRect();
            return {{ x: a.left + scrollX, y: a.top + scrollY, w: Math.max(a.right, b.right) - a.left,
                     h: b.bottom - a.top }}; }})()""")
        data = page.cmd("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                        clip={"x": r["x"] - pad, "y": r["y"] - pad, "width": r["w"] + 2 * pad,
                              "height": r["h"] + 2 * pad, "scale": 1})["data"]
        with open(path, "wb") as f:
            f.write(base64.b64decode(data))

    page.goto(BASE, 1280, 900)
    page.wait_for("!document.getElementById('submit').disabled", 120)
    fill = {"monthly_income_gbp": 2200, "existing_monthly_obligations_gbp": 300, "savings_balance_gbp": 150,
            "credit_score": 610, "months_credit_history": 18, "num_credit_accounts": 2,
            "num_defaults_24m": 0, "order_value_gbp": 500}
    page.js("".join(f"document.getElementById('{k}').value='{v}';" for k, v in fill.items())
            + "document.getElementById('n_instalments').value='6';"
            "document.getElementById('merchant_category').value='Electronics';"
            "document.getElementById('form').requestSubmit(); true")
    page.wait_for("!document.getElementById('result').classList.contains('hidden')", 120)
    time.sleep(2.5)   # count-up and knob slide finish
    assert page.js("document.getElementById('r-pd').textContent") == "3.2%"
    clip(os.path.join(root, "docs/images/demo-assessment.png"), "#result .surface", "#result .panel.explain")

    page.goto(BASE + "simulator.html", 1280, 900)
    page.wait_for("document.getElementById('s-recall').textContent.length > 0", 60)
    time.sleep(1.5)
    clip(os.path.join(root, "docs/images/demo-simulator.png"), ".slider-panel", "section.grid.half")
    print("ok")
finally:
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1); shutil.rmtree(prof, ignore_errors=True)
