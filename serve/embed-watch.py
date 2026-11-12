#!/usr/bin/env python3
"""serve/embed-watch.py - the embedding sidecar's Monitor: a stdlib-only observer of :8090 and of the box.

Why a second, tiny page instead of a panel in serve/webui.py: webui.py's whole machinery is vLLM-shaped --
parse_metrics() speaks vLLM's Prometheus dialect, and its body is the amdgpu sysfs. llama.cpp speaks another
dialect and owns no GPU, so sharing the page would mean patching 2,639 lines for one card. This is the same
observe-mode contract in 200 lines: GET /health and GET /metrics every POLL_S, nothing written anywhere,
nothing in the inference path.

What the page is FOR: the 2026-11-01 incident (notes/load-oom-svm-wedge.md) was memory overcommit -> kswapd
-> amdgpu SVM deadlock -> box unrecoverable. The embedding sidecar is the first new resident process this box
has taken on in months, so the first card is the memory pressure the GGUF's mmap must NOT add: PSI some_avg,
MemAvailable, swap used. The second card is llama.cpp's own /metrics deltas (needs llama-server --metrics, which
serve/embed.sh passes): requests/s, tokens/s, average latency. Third: the embed.sh recipe, read from the
launcher's own ${VAR:=default} lines the way launcher_env() in setup.py reads them.
"""
import argparse
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE = {"t": 0.0, "health": None, "metrics": None, "err": None, "snaps": [], "meta": {}}
LOCK = threading.Lock()


def get(url, timeout=4.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, OSError):
        return None, ""


COUNTERS = {"req": "llama_cpp_n_requests", "failed": "llama_cpp_n_failed",
            "ptok": "llama_cpp_prompt_tokens_total", "ptok_s": "llama_cpp_prompt_tokens_seconds_total",
            "ms": "llama_cpp_seconds_total"}


def parse_prom(text):
    """Sum each counter name across its label lines; the names are read defensively because the exact
    llama.cpp metric names are upstream's to change -- an absent one reads as None, not as a zero."""
    out = {}
    for line in text.splitlines():
        name = line.split("{")[0].split(" ")[0].strip()
        if name in COUNTERS.values():   # COUNTERS maps short name -> full metric name; the line carries the full one
            try:
                val = line.rsplit("}", 1)[1] if "{" in line else line.rsplit(" ", 1)[1]
                out[name] = out.get(name, 0.0) + float(val)   # keyed by the full metric name
            except (IndexError, ValueError):
                pass
    return {k: out.get(v) for k, v in COUNTERS.items()}   # short name -> value under the full name



def poller(url, poll_s):
    while True:
        code, _ = get(url + "/health")
        mcode, mtxt = get(url + "/metrics")
        with LOCK:
            STATE["t"] = time.time()
            STATE["health"] = code
            STATE["metrics"] = parse_prom(mtxt) if mcode == 200 else None
            STATE["err"] = None if code else "unreachable"
            if STATE["metrics"]:
                STATE["snaps"].append((time.time(), STATE["metrics"]))
                STATE["snaps"] = STATE["snaps"][-120:]
        time.sleep(poll_s)


def num(path):
    try:
        return float(Path(path).read_text().split()[0])
    except (OSError, IndexError, ValueError):
        return None


def read_host():
    h = {}
    for kind in ("cpu", "memory", "io"):
        try:
            m = re.search(r"some_avg=(\d+\.\d+|\d+)", Path(f"/proc/pressure/{kind}").read_text())
            if m:
                h[f"psi_{kind}"] = float(m.group(1))
        except OSError:
            pass
    try:
        txt = Path("/proc/meminfo").read_text()
        for k, g in (("MemTotal", "mem_total_kb"), ("MemAvailable", "mem_avail_kb"),
                     ("SwapTotal", "swap_total_kb"), ("SwapFree", "swap_free_kb")):
            m = re.search(rf"^{k}:\s+(\d+)", txt, re.M)
            if m:
                h[g] = int(m.group(1))
    except OSError:
        pass
    try:  # the sidecar's own footprint: RSS of the llama-server process, in kB
        pid = subprocess.run(["pgrep", "-f", "llama-server"], capture_output=True, text=True,
                             timeout=3).stdout.split()
        if pid:
            rss = 0
            for p in pid:
                for ln in Path(f"/proc/{p}/status").read_text().splitlines():
                    if ln.startswith("VmRSS:"):
                        rss += int(ln.split()[1])
            h["embed_rss_kb"] = rss
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return h


def rates():
    """Deltas over the kept snapshots: what the counters mean per second. llama.cpp's prompt-tokens
    metric is a rate (tokens/second), so the average latency is ptok / ptok_s, not ptok / n_requests."""
    with LOCK:
        snaps = list(STATE["snaps"])
    if len(snaps) < 2:
        return {}
    (t0, a), (t1, b) = snaps[0], snaps[-1]
    dt = max(t1 - t0, 1e-6)
    r = {"window_s": dt}
    for k in COUNTERS:
        if a.get(k) is not None and b.get(k) is not None:
            r[k] = (b[k] - a[k]) / dt
    if r.get("ptok") is not None and r.get("ptok_s"):
        r["avg_ms"] = 1000.0 * r["ptok"] / r["ptok_s"]
    return r


def state_json():
    with LOCK:
        health, err, t = STATE["health"], STATE["err"], STATE["t"]
    return {"up_at": t, "health": health, "err": err,
            "rates": rates(), "host": read_host(), "meta": STATE["meta"]}


def card(title, body, cls=""):
    return f'<div class="c {cls}"><h3>{title}</h3>{body}</div>'


def html():
    s = state_json()
    h, r = s["host"], s["rates"]
    up = s["health"] == 200

    def f(x, n=1):
        return "-" if x is None else f"{x:,.{n}f}"

    mem = h.get("mem_avail_kb", 0) / 1e6
    mtot = h.get("mem_total_kb", 0) / 1e6
    swp = (h.get("swap_total_kb", 0) - h.get("swap_free_kb", 0)) / 1e6
    stot = h.get("swap_total_kb", 0) / 1e6
    psi_m, psi_c, psi_i = h.get("psi_memory"), h.get("psi_cpu"), h.get("psi_io")
    rss = h.get("embed_rss_kb", 0) / 1e6
    cls_m = "g" if (psi_m is None or psi_m < 5) else ("y" if psi_m < 20 else "r")
    cls_s = "" if swp < stot * 0.7 else " y"

    upcls = "g" if up else "r"
    status = card("embedding :8090",
                  f'<p><b class="{upcls}">{"UP" if up else "DOWN / unreachable"}</b><br>'
                  f'最后探测 {time.strftime("%H:%M:%S", time.localtime(s["up_at"])) if s["up_at"] else "-"}'
                  + ("<br><small>llama-server 需带 --metrics（serve/embed.sh 默认带）</small>"
                     if not r else "") + "</p>", "g" if up else "r")
    perf = card("吞吐（差分窗口）",
                f"<p>req/s <b>{f(r.get('req'), 2)}</b> ｜ 失败 {f(r.get('failed'), 2)}/s<br>"
                f"tokens/s <b>{f(r.get('ptok'), 0)}</b> ｜ 平均 {f(r.get('avg_ms'), 0)} ms/req<br>"
                f"<small>窗口 {f(r.get('window_s'), 0)} s；无流量时归零是正常的</small></p>") if r else \
        card("吞吐", "<p>没有 /metrics 数据 —— 起服务时缺 --metrics，或还没有流量</p>", "y")
    memc = card("主机内存（11-01 红线）",
                f"<p>MemAvailable <b>{mem:.1f}</b> / {mtot:.1f} GiB<br>"
                f"swap 已用 <b>{swp:.1f}</b> / {stot:.1f} GiB<br>"
                f"llama-server RSS <b>{rss:.2f}</b> GiB（GGUF 走 page cache，应远小于模型大小）<br>"
                f"PSI some_avg 10s: memory <b>{f(psi_m)}</b> ｜ cpu {f(psi_c)} ｜ io {f(psi_i)}</p>",
                cls_m + cls_s)
    meta = STATE["meta"]
    recipe = card("embed.sh recipe",
                  "<p>" + "<br>".join(f"{k} <b>{v}</b>" for k, v in meta.items() if v) + "</p>"
                  if meta else "<p>未读到 serve/embed.sh</p>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>r9700 embedding monitor</title><style>
body{{background:#111;color:#ddd;font:14px/1.5 system-ui;margin:24px}}
h1{{font-size:18px}} h3{{margin:0 0 6px;font-size:13px;color:#8ab}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;max-width:1200px}}
.c{{background:#1c1f26;border:1px solid #2c313a;border-radius:8px;padding:12px 14px}}
b.g{{color:#6c6}} b.y{{color:#ec6}} b.r{{color:#e66}} .c.g{{border-color:#3a5}} .c.y{{border-color:#a83}} .c.r{{border-color:#a44}}
small{{color:#778}} p{{margin:6px 0}} a{{color:#8ab}}</style></head><body>
<h1>r9700 · embedding sidecar monitor <small>observe only · poll {STATE.get('poll_s', 5)}s · <a href="/state.json">state.json</a></small></h1>
<div class="grid">{status}{perf}{memc}{recipe}</div>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    KEY = None

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/health", "/monitor/health"):
            return self.send(200, json.dumps({"status": "ok"}), "application/json")
        if Handler.KEY and self.headers.get("X-API-Key") != Handler.KEY:
            return self.send(401, json.dumps({"error": "bad key"}), "application/json")
        if self.path == "/state.json":
            return self.send(200, json.dumps(state_json()), "application/json")
        if self.path in ("/", "/monitor"):
            return self.send(200, html())
        self.send(404, json.dumps({"error": "not found"}), "application/json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--embed", default="http://127.0.0.1:8090")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--key", default=None)
    ap.add_argument("--launcher", default=None, help="path to serve/embed.sh for the recipe card")
    a = ap.parse_args()
    Handler.KEY = a.key
    STATE["poll_s"] = a.poll
    src = Path(a.launcher) if a.launcher else Path(__file__).resolve().parent / "embed.sh"
    try:  # the ${VAR:=default} lines ARE the launcher's recipe -- the read setup.py's launcher_env does
        meta = {}
        for k, v in re.findall(r"\$\{([A-Z_]+)(?::-|\?=)([^}]*)\}", src.read_text(errors="replace")):
            if k in ("PORT", "CORES", "THREADS", "CTX", "BATCH", "MODEL", "HF_REPO", "HF_FILE"):
                meta[k] = v.strip().strip('"')
        STATE["meta"] = meta
    except OSError:
        pass
    threading.Thread(target=poller, args=(a.embed, a.poll), daemon=True).start()
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    print(f"embed-watch: http://{a.bind}:{a.port}/  observing {a.embed} every {a.poll}s (observe only)")
    srv.serve_forever()


if __name__ == "__main__":
    main()




