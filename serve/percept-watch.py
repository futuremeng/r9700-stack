#!/usr/bin/env python3
"""serve/percept-watch.py - the perception layer's Monitor: a stdlib-only observer of the run journal and of the box.

The perception layer is CLI-shaped (serve/percept.sh runs sherpa binaries per file), so there is no
:PORT to poll -- the data source is the run journal percept.sh appends (.r9700-state/percept-runs.jsonl,
one line per run) plus a live look at the same things the launcher's own guards check: the binary, the
two model files, the pinned core range, MemAvailable against MEM_FLOOR. Observe only, like embed-watch:
nothing written, nothing in any path a diarization takes.

Why it looks like embed-watch.py: the contract is the same -- a tiny page per sidecar family, webui.py's
machinery is vLLM-shaped and a sherpa binary publishes nothing. Same first card as the embedding page
carries: the memory the incident notes watch (MemAvailable, Committed_AS, swap, PSI), because the
acceptance in serve/cpu-sidecars.md is stated in exactly those numbers.

GET /health answers 200 (the gateway's liveness probe), GET /state.json the payload, GET / the page.
"""
import argparse
import json
import re
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE = {"meta": {}}


def journal_tail(path, n=200):
    try:
        lines = Path(path).read_text(errors="replace").splitlines()[-n:]
        return [json.loads(ln) for ln in lines if ln.strip()]
    except (OSError, ValueError):
        return []


def read_host():
    """The acceptance numbers: MemAvailable, Committed_AS, swap, PSI. Same reads as embed-watch.py's
    read_host() minus the per-serve RSS (the perception layer owns no resident RSS between runs --
    that is what CLI shape means, and it is the point)."""
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
                     ("SwapTotal", "swap_total_kb"), ("SwapFree", "swap_free_kb"),
                     ("Committed_AS", "committed_as_kb")):
            m = re.search(rf"^{k}:\s+(\d+)", txt, re.M)
            if m:
                h[g] = int(m.group(1))
    except OSError:
        pass
    return h


def cpus_in(rng):
    for part in re.split(r"[,\s]+", rng or ""):
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                yield from range(int(a), int(b) + 1)
            except ValueError:
                pass
        elif part.isdigit():
            yield int(part)


def online_cpus():
    try:
        txt = Path("/sys/devices/system/cpu/online").read_text().strip()
        for part in txt.split(","):
            a, _, b = part.partition("-")
            try:
                yield from range(int(a), int((b or a)))
            except ValueError:
                pass
    except OSError:
        return


def expand(p, meta):
    """The launcher resolves $MODEL_DIR/$HOME in a shell at run time; the regex over its source sees the
    literal. Resolve the same two variables so the inventory card stats the file a real run would stat."""
    if "$" not in str(p):
        return str(p)
    home = str(Path.home())
    md = str(meta.get("MODEL_DIR") or "") or str(Path.home() / "models/percept")
    return str(p).replace("$MODEL_DIR", md.replace("$HOME", home)).replace("$HOME", home)


def inventory(meta):
    """The same three checks serve/percept.sh refuses on, reported instead of enforced."""
    inv = {}
    bin_name = meta.get("SHERPA_BIN") or "sherpa-onnx-offline-speaker-diarization"
    # shutil.which covers both a bare name on PATH and an absolute path from native.env (it checks the
    # executable bit when the value contains a slash), which is what the launcher's own guard sees.
    inv["binary"] = shutil.which(expand(bin_name, meta))
    for key, var in (("SEG_MODEL", "segmentation"), ("SPK_MODEL", "embedding")):
        p = meta.get(key) or (Path(expand(meta.get("MODEL_DIR", ""), meta)) /
                             {"segmentation": "segmentation-3.0.onnx", "embedding": "campplus.onnx"}[var])
        p = Path(expand(p, meta))
        try:
            inv[var] = {"path": str(p), "mb": round(p.stat().st_size / 1e6, 1)}
        except OSError:
            inv[var] = {"path": str(p), "mb": None}
    onl = set(online_cpus())
    pinned = list(cpus_in(meta.get("CORES", "48-63")))
    # None (rendered "-") where /sys has no online list -- a dev box -- rather than claiming every
    # pinned core is online; on the native box onl is populated and this is the guard's real number.
    inv["cores"] = {"range": meta.get("CORES", "48-63"),
                    "online": len([c for c in pinned if c in onl]) if onl else None}
    h = read_host()
    floor = meta.get("MEM_FLOOR", "10")
    # None (rendered "-") when /proc has no meminfo at all, so the card cannot show a fake 0.0 next to
    # the "-" the host card shows; would_refuse stays False -- an unmeasurable floor refuses nothing.
    avail = h["mem_avail_kb"] / 1e6 if "mem_avail_kb" in h else None
    inv["guard"] = {"mem_avail_gb": (round(avail, 1) if avail is not None else None), "floor_gb": floor,
                    "would_refuse": bool(avail) and avail < float(re.sub(r"\D.*", "", floor) or 10)}
    return inv


def state_json():
    runs = journal_tail(STATE["journal"])
    now = time.time()
    day = [r for r in runs if now - r.get("ts", 0) < 86400]
    ok = [r for r in day if r.get("rc") == 0]
    rts = [r["rt"] for r in ok if isinstance(r.get("rt"), (int, float)) and r["rt"]]
    return {"runs_total": len(runs), "runs_24h": len(day), "fail_24h": len(day) - len(ok),
            "last": runs[-1] if runs else None,
            "rt_last": (runs[-1].get("rt") if runs else None),
            "rt_mean_24h": (sum(rts) / len(rts)) if rts else None,
            "host": read_host(), "meta": STATE["meta"], "inventory": inventory(STATE["meta"])}


def card(title, body, cls=""):
    return f'<div class="c {cls}"><h3>{title}</h3>{body}</div>'


def html():
    s = state_json()
    h, inv, last = s["host"], s["inventory"], s["last"]

    def f(x, n=1):
        return "-" if x is None else f"{x:,.{n}f}"

    age = "-" if not last else f'{int((time.time() - last["ts"]) / 3600)} h ago'
    lastline = ("-" if not last else
                f'{time.strftime("%m-%d %H:%M", time.localtime(last["ts"]))} · {last.get("task")} · '
                f'audio {f(last.get("audio_s"))}s in {f(last.get("wall_s"))}s · <b class="{"g" if last.get("rc") == 0 else "r"}">'
                f'rt {f(last.get("rt"), 1)}× · rc {last.get("rc")}</b>')
    runs_c = card("运行台账（CLI 形态：这里就是指标源）",
                  f"<p>最近运行 {lastline}<br>距上次 {age}<br>"
                  f"24 h：{s['runs_24h']} 次，失败 {s['fail_24h']}，rt 均值 {f(s['rt_mean_24h'], 1)}×<br>"
                  f"<small>台账 = .r9700-state/percept-runs.jsonl（percept.sh 每次运行一行）。rt < 1× 先查 §4 的线程/绑核；"
                  f"两个已知声音并成一个说话人是聚类阈值问题，不是管道坏了 —— 先试 num_clusters=2。</small></p>",
                  "" if s["runs_24h"] else "y")
    # Precomputed so no backslash lands inside an f-string expression (PEP 701 is 3.12+; the box may run 3.10).
    refuse = '<b class="r">会拒跑</b>' if inv['guard']['would_refuse'] else '放行'
    inv_c = card("运行时与模型（launcher 守卫的现场版）",
                 f"<p>binary <b>{inv['binary'] or '未在 PATH'}</b><br>"
                 f"segmentation {inv['segmentation']['path']} ({f(inv['segmentation']['mb'])} MB)<br>"
                 f"embedding {inv['embedding']['path']} ({f(inv['embedding']['mb'])} MB)<br>"
                 f"绑核 {inv['cores']['range']}（在线 {f(inv['cores']['online'], 0)}）｜ MEM_FLOOR 现况 {refuse} "
                 f"{f(inv['guard']['mem_avail_gb'])} / {inv['guard']['floor_gb']} GiB</p>",
                 "y" if inv['guard']['would_refuse'] else "")
    meta = STATE["meta"]
    # "-" where /proc has no meminfo (dev box) instead of a fake 0.0; the real box fills all of these.
    mem = h["mem_avail_kb"] / 1e6 if "mem_avail_kb" in h else None
    mtot = h["mem_total_kb"] / 1e6 if "mem_total_kb" in h else None
    cas = h["committed_as_kb"] / 1e6 if "committed_as_kb" in h else None
    swp = ((h["swap_total_kb"] - h["swap_free_kb"]) / 1e6
           if "swap_total_kb" in h and "swap_free_kb" in h else None)
    psi_m = h.get("psi_memory")
    cls_m = "g" if (psi_m is None or psi_m < 5) else ("y" if psi_m < 20 else "r")
    mem_c = card("主机内存（验收的四项里有三项在这里）",
                 f"<p>MemAvailable <b>{f(mem)}</b> / {f(mtot)} GiB ｜ 验收线 ≥ 12 GiB<br>"
                 f"Committed_AS <b>{f(cas)}</b> / {f(mtot)} GiB（11-01 红线）<br>"
                 f"swap 已用 <b>{f(swp)}</b> GiB ｜ PSI memory some_avg {f(psi_m)}<br>"
                 f"<small>probe32 ±1% 与 pswpout 增量是 bench 协议的一部分，不在本页；本页盯常驻与红线。"
                 f"开发机（无 /proc）这几项为 - ，以真机读数为准。</small></p>",
                 cls_m)
    recipe = card("percept.sh recipe",
                  "<p>" + "<br>".join(f"{k} <b>{v}</b>" for k, v in meta.items() if v) + "</p>"
                  if meta else "<p>未读到 serve/percept.sh</p>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>r9700 perception monitor</title><style>
body{{background:#111;color:#ddd;font:14px/1.5 system-ui;margin:24px}}
h1{{font-size:18px}} h3{{margin:0 0 6px;font-size:13px;color:#8ab}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;max-width:1200px}}
.c{{background:#1c1f26;border:1px solid #2c313a;border-radius:8px;padding:12px 14px}}
b.g{{color:#6c6}} b.y{{color:#ec6}} b.r{{color:#e66}} .c.g{{border-color:#3a5}} .c.y{{border-color:#a83}} .c.r{{border-color:#a44}}
small{{color:#778}} p{{margin:6px 0}} a{{color:#8ab}}</style></head><body>
<h1>r9700 · perception layer monitor <small>observe only · <a href="/state.json">state.json</a></small></h1>
<div class="grid">{runs_c}{inv_c}{mem_c}{recipe}</div>
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
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--journal", default=None, help="percept-runs.jsonl path (default: <repo>/.r9700-state/)")
    ap.add_argument("--key", default=None)
    ap.add_argument("--launcher", default=None, help="path to serve/percept.sh for the recipe card")
    ap.add_argument("--sherpa-bin", default=None, help="what native.env says SHERPA_BIN is, if it says anything")
    ap.add_argument("--seg-model", default=None)
    ap.add_argument("--spk-model", default=None)
    a = ap.parse_args()
    Handler.KEY = a.key
    here = Path(__file__).resolve().parent
    STATE["journal"] = a.journal or str(here.parent / ".r9700-state" / "percept-runs.jsonl")
    src = Path(a.launcher) if a.launcher else here / "percept.sh"
    try:
        meta = {}
        for k, v in re.findall(r"\$\{([A-Z_]+)(?::-|\?=)([^}]*)\}", src.read_text(errors="replace")):
            if k in ("TASK", "CORES", "THREADS", "MEM_FLOOR", "MODEL_DIR", "SHERPA_BIN",
                     "SEG_MODEL", "SPK_MODEL", "JOURNAL"):
                v = v.strip().strip('"')
                # THREADS defaults to the nested ${N_ONLINE:-8} form; the regex cannot finish it and the
                # real number is the online-cpu guard, so render the intent instead of the truncated text.
                if k == "THREADS" and (not v or "${" in v):
                    v = "(在线核数)"
                meta[k] = v
        # resolved at startup from --journal/default and what the server actually tails; the $( ) form in
        # the source would render as an expanded-looking literal in the recipe card.
        meta["JOURNAL"] = STATE["journal"]
        STATE["meta"] = meta
    except OSError:
        pass
    # native.env values arrive through the wrapper, not through the regex: the file is the real-box home
    # of SHERPA_BIN/SEG_MODEL/SPK_MODEL, and the launcher would resolve them at run time.
    for k, v in (("SHERPA_BIN", a.sherpa_bin), ("SEG_MODEL", a.seg_model), ("SPK_MODEL", a.spk_model)):
        if v:
            STATE["meta"][k] = v
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    print(f"percept-watch: http://{a.bind}:{a.port}/  journal {STATE['journal']} (observe only)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
