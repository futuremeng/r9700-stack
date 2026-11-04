#!/usr/bin/env python3
"""The monitor MERGED into the serve: pages and /api on the engine's own port, records from the engine.

Three tiers, because the plugin spans three worlds, and each is run where its imports exist:

  1. `webui.EngineRecords` / `webui.web_asset` -- stdlib, runs anywhere (this file works next to
     tests/test_webui.py under .venv).
  2. the ROUTES -- need FastAPI. Skipped, not failed, where it is absent; run with the native venv
     (`~/venv-r9700/bin/python tests/test_webui_plugin.py`) to exercise them.
  3. vLLM's own plugin wiring -- `attach_endpoint_plugins` deciding that a plugin's `GET /` replaces the
     server's. Only run when vLLM imports, because the ordering is its code, not ours, and pretending to
     test it with our own sort would test nothing.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))
import webui as W                                                    # noqa: E402

fail = 0


def ck(name, got, extra=""):
    global fail
    if got is True:
        print(f"  ok   {name}")
        return
    fail += 1
    print(f"  FAIL {name} {extra}")


# A fake amdgpu tree: this test must never read a real card (a live one answers EBUSY, and the point of
# the check is not to perturb the launch that happens to be running).
FAKE_SYSFS = "/tmp/r9700-webui-plugin-sysfs"
for card, busy in (("card1", "41"), ("card2", "0")):
    d = Path(FAKE_SYSFS) / card / "device"
    d.mkdir(parents=True, exist_ok=True)
    (d / "gpu_busy_percent").write_text(busy + "\n")
    (d / "mem_info_vram_used").write_text("59916288\n")
    (d / "mem_info_vram_total").write_text("34359738368\n")
    (d / "uevent").write_text("MAJOR=226\nMINOR=1\nDEVNAME=drm/card1\nPCI_SLOT_NAME=0000:c3:00.0\n")


class Fin:
    """vllm/v1/metrics/stats.py:288 FinishedRequestStats, duck-typed."""

    def __init__(self, **kw):
        d = dict(request_id="cmpl-1", finish_reason="stop", num_prompt_tokens=1187, num_generation_tokens=64,
                 num_preemptions=0, max_tokens_param=64, queued_time=0.0012, prefill_time=0.244,
                 inference_time=1.011, decode_time=0.767, mean_time_per_output_token=0.012,
                 is_corrupted=False, num_cached_tokens=1152)
        d.update(kw)
        self.__dict__.update(d)


# ------------------------------------------------------------------ 1. the engine-side record (stdlib)
er = W.EngineRecords(keep=3)
er.add(Fin())
er.add(Fin(request_id="cmpl-2", num_prompt_tokens=4399, num_generation_tokens=128, decode_time=1.6,
           queued_time=0.5, prefill_time=1.2, inference_time=3.0))
ck("newest first, like the relay ring", er.listing()[0]["id"] == "cmpl-2", str(er.listing()))
r = er.listing()[0]
ck("state and timing keys the page already reads",
   r["state"] == "completed" and r["wallclock_s"] == 3.0 and r["first_token_s"] == 1.7
   and r["response_format"] == "engine", json.dumps(r))
ck("no endpoint claim: the engine's stats do not carry one", r["path"] is None)
# The ROW, not just the detail: the cards read the list. While usage/timings stayed out of it, the
# "Last decode speed ... One stream" card fell back to the engine-wide 1/step -- 27.4 tok/s for a 36.4 ms
# inter-token interval -- on a page whose selected request decoded at 87.8, and a 250k prompt's prefill
# appeared nowhere except as a mean over the last 2 s.
ck("a row carries the engine's own split and its token counts",
   r["timings"]["prefill_s"] == 1.2 and r["timings"]["queue_s"] == 0.5 and r["timings"]["decode_s"] == 1.6
   and r["usage"]["prompt_tokens"] == 4399 and r["usage"]["completion_tokens"] == 128, json.dumps(r)[:200])
# Prefill speed per request, computed when the record is made: the card's "prompt tok/s" is a counter credited
# at request end (a 180,066-token prefill reads 0 tok/s until it lands, then 89,580 in one poll), so the record
# has to carry the mean the engine's own tokens and seconds support.
ck("prefill speed uses computed tokens, not tokens the cache handed over",
   abs(r["timings"]["prefill_tok_s"] - 3247 / 1.2) < 0.01
   and r["timings"]["prefill_basis"] == "3247 computed tokens / prefill time (1152 cached, not read)",
   json.dumps(r["timings"]))
er.add(Fin(request_id="nc", num_prompt_tokens=60061, num_cached_tokens=0, num_generation_tokens=7,
           prefill_time=6.74, queued_time=0.0, decode_time=0.04, inference_time=6.8))
nct = er.get("nc")["timings"]
ck("with nothing cached the numerator is the whole prompt",
   abs(nct["prefill_tok_s"] - 60061 / 6.74) < 0.01 and nct["prefill_basis"] == "prompt tokens / prefill time",
   json.dumps(nct))
er.add(Fin(request_id="q0", queued_time=0.0, prefill_time=0.04, decode_time=0.2, num_generation_tokens=8))
q0 = er.get("q0")["timings"]
ck("a request that never queued reads 0.000 s, not a dash (found live: 0 -> None made 'no queue wait' "
   "indistinguishable from 'not reported')", q0["queue_s"] == 0.0 and q0["prefill_s"] == 0.04, json.dumps(q0))
d = er.get("cmpl-2")
u, t = d["usage"], d["timings"]
ck("the per-request queue/prefill/decode split, measured rather than averaged over a window",
   (t["queue_s"], t["prefill_s"], t["decode_s"]) == (0.5, 1.2, 1.6), json.dumps(t))
ck("token counts from the engine, cache hits included",
   u["prompt_tokens"] == 4399 and u["completion_tokens"] == 128
   and u["prompt_tokens_details"]["cached_tokens"] == 1152, json.dumps(u))
ck("a speed is stated with its basis -- and the basis IS the engine's published mean, not our arithmetic: this "
   "fixture's mean_time_per_output_token (12.0 ms) and decode_time/(gen-1) (12.6 ms) disagree on purpose, and a "
   "row that recomputes would disagree with the engine's own card",
   abs(t["predicted_per_second"] - 1.0 / 0.012) < 1e-9 and "mean 12.0 ms" in t["speed_basis"]
   and abs(t["tpot_ms"] - 12.0) < 1e-9, json.dumps(t))
_nof = Fin(request_id="nofield", num_generation_tokens=128, decode_time=1.6, queued_time=0.5, prefill_time=1.2,
           inference_time=3.0)
del _nof.mean_time_per_output_token                 # an older build: the field is not there to ask
er.add(_nof)
_t2 = er.get("nofield")["timings"]
ck("on a build that does not publish the mean, the fallback is the engine's OWN formula (gen-1 over the span), "
   "so a 128-token request reads 79.4 tok/s and not the 80 that counts the prefill's first token as a decode token",
   abs(_t2["predicted_per_second"] - 127.0 / 1.6) < 1e-9 and "127 tokens after the first" in _t2["speed_basis"],
   json.dumps(_t2))
ck("the gaps say what is missing instead of showing an empty body",
   d["input"] is None and "serve/webui.sh with WEBUI_MODE=proxy" in d["input_note"])
ck("detail lookup by id only", er.get("nope") is None)
for i in range(5):
    er.add(Fin(request_id=f"x{i}"))
ck("ring bounded", len(er) == 3 and er.listing()[0]["id"] == "x4", str(len(er)))
ck("the /web guard refuses the repo's own source",
   W.web_asset(W.WEB, "../webui.py") is None and W.web_asset(W.WEB, "monitor.js").name == "monitor.js")

# ------------------------------------------------------------------ 2. the hub, fed from the process (stdlib)
MET = ('# HELP\nvllm:num_requests_running{model_name="Qwen3.8",engine="0"} 2\n'
       'vllm:generation_tokens_total{model_name="Qwen3.8",engine="0"} 400.0\n'
       'vllm:prompt_tokens_total{model_name="Qwen3.8",engine="0"} 9000.0\n'
       'vllm:kv_cache_usage_perc{model_name="Qwen3.8",engine="0"} 0.12\n'
       'vllm:cache_config_info{model_name="Qwen3.8",block_size="448",num_gpu_blocks="1230",'
       'kv_cache_size_tokens="497588",engine="0"} 1\n')
hub = W.Hub({"mode": "merged", "upstream": "http://127.0.0.1:1", "key": None, "poll": 2.0, "keep": 3,
             "gpus": FAKE_SYSFS, "smi": None, "meta": None, "version": "vllm x (this process)",
             "timeout": 1.0, "engine_records": er, "metrics_fn": lambda: MET})
hub.poll.__self__  # just to be explicit that poll is the hub's own loop, driven below
cur, cap, info = W.parse_metrics(MET)
payload = W.derive(cur, {}, None, cap, info, "merged", hub.kept(), [])
ck("merged mode is a mode the payload carries",
   payload["mode"] == "merged" and payload["rate"]["generation_tok_s"] is None)
hub.prev, hub.prev_t, hub.status = cur, time.time(), payload
hub.hist.append({"at": time.time(), "gen_tok_s": None, "kv_used_pct": 0.12, "running": 2})
st = hub.api("/api/status")
ck("the KV pool comes from the registry snapshot", st["live"]["kv_tokens"] == 497588, str(st["live"]))
ck("the request count comes from the engine ring", st["requests_kept"] == 3, str(st["requests_kept"]))
rq = hub.api("/api/requests")
ck("the list says where it came from", rq["source"] == "engine" and len(rq["requests"]) == 3, str(rq)[:120])
ck("and the detail agrees", hub.record_detail("x4")["timings"]["finish_reason"] == "stop")

# Two bugs the live merged launch showed, both now asserted.
#  (a) cfg["gpus"] defaulted to None, so the page came up with FOUR CARDS MISSING -- the one panel the
#      sidecar is for. (b) payload["merged"] = bool(self.eng): an EMPTY EngineRecords is falsy, so the
#      first seconds of a launch reported the merged monitor as if it were an observing sidecar.
import threading                                          # noqa: E402
live = W.Hub({"mode": "merged", "upstream": "http://127.0.0.1:1", "key": None, "poll": 0.2, "keep": 3,
              "gpus": FAKE_SYSFS, "pci": None, "journal": "", "smi": None, "meta": None, "version": "v",
              "timeout": 1.0, "engine_records": W.EngineRecords(3), "metrics_fn": lambda: MET})
threading.Thread(target=live.poll, daemon=True).start()
threading.Thread(target=live.gpu_poll, daemon=True).start()   # the cards are a separate loop on purpose:
threading.Thread(target=live.slow_poll, daemon=True).start()  # a sysfs read may block, a metrics read may not
time.sleep(1.6)
ck("a merged hub knows it is merged before any request has finished",
   live.status.get("merged") is True and live.status.get("reachable") is True, str(live.status)[:160])
ck("and it reads the cards from the same default path the sidecar uses",
   len(live.status["gpus"]) == 2 and live.status["gpus"][0]["card"] == "card1", str(live.status["gpus"])[:120])
# The ring has to KEEP filling. A reader added to gpu_poll without a key in this cfg raised KeyError inside the
# loop, one frame was stored, and the thread died: the panel kept showing its first frame forever and looked
# healthy. Found live on 2026-11-02 with the PCIe card, which is why the Hub reads cfg with .get.
time.sleep(1.0)
ck("the card ring keeps filling rather than freezing on one frame",
   len(live.status.get("gpus_history") or []) >= 3,
   f"{len(live.status.get('gpus_history') or [])} samples in ~2.6 s at a 0.2 s poll")
ck("the plugin's own cfg names every reader the Hub can start",
   all(f'"{k}"' in (Path(__file__).resolve().parent.parent / "serve/plugin/r9700_webui/__init__.py").read_text()
       for k in ("gpus", "pci", "journal", "smi")))

# --------------------------------------------------------------------- 3. the routes (needs FastAPI)
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except ImportError:
    print("  skip FastAPI routes: not installed in this interpreter "
          "(run with the native venv: ~/venv-r9700/bin/python tests/test_webui_plugin.py)")
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve" / "plugin"))
    import r9700_webui as P

    P.HUB = hub
    P.KEY[0] = None
    app = FastAPI()

    @app.get("/")
    def engine_root():                                # what vLLM itself serves on /
        return {"version": "engine"}

    P.WebMonitor().attach_router(app)
    c = TestClient(app)
    ck("/monitor is the page", "id=\"kv-pct\"" in c.get("/monitor").text, c.get("/monitor").status_code)
    ck("/about is the page", "id=\"facts-engine\"" in c.get("/about").text, c.get("/about").status_code)
    ck("/favicon.svg and /web/* are served",
       c.get("/favicon.svg").status_code == 200
       and "javascript" in c.get("/web/monitor.js").headers["content-type"],
       c.get("/web/monitor.js").headers.get("content-type"))
    ck("traversal refused on the merged port too", c.get("/web/%2e%2e/webui.py").status_code == 404)
    ck("a /web SUBPATH (the fonts) is served on the merged port", c.get("/web/fonts/outfit-latin-wght.woff2").status_code == 200)
    # 200 before the payload: a handler parameter FastAPI reads as a QUERY parameter answers 422 and the
    # page shows a blank panel -- which is how the first version of these routes failed.
    ck("/api/status answers the merged payload",
       c.get("/api/status").status_code == 200 and c.get("/api/status").json()["mode"] == "merged",
       str(c.get("/api/status").json())[:120])
    j = c.get("/api/requests").json()
    ck("/api/requests carries engine rows", j["source"] == "engine" and j["requests"][0]["id"] == "x4", str(j)[:120])
    dd = c.get("/api/requests?id=x4").json()
    ck("and the detail pane's queue/prefill/decode", dd["timings"]["prefill_s"] == 0.244, str(dd)[:160])
    ck("a missing id is a 404, not an empty object", c.get("/api/requests?id=zzz").status_code == 404)
    ck("the pages are open, the API is not, once the serve has a key",
       (P.KEY.__setitem__(0, "sekret") or c.get("/monitor").status_code == 200)
       and c.get("/api/status").status_code == 401
       and c.get("/api/status", headers={"Authorization": "Bearer sekret"}).status_code == 200)
    P.KEY[0] = None

    # vLLM's own wiring, if this interpreter has vLLM: it is what decides that a plugin's GET / wins.
    try:
        from vllm.plugins.endpoint_plugins.interface import attach_endpoint_plugins
    except Exception as e:
        print(f"  skip vLLM route ordering: import failed ({type(e).__name__}: {str(e)[:80]})")
    else:
        app2 = FastAPI()

        @app2.get("/")
        def root2():
            return {"version": "engine"}

        import vllm.plugins as VP
        real = VP.load_endpoint_plugins
        VP.load_endpoint_plugins = lambda tasks=None: [P.WebMonitor()]
        try:
            attach_endpoint_plugins(app2, ("generate",))
        finally:
            VP.load_endpoint_plugins = real
        c2 = TestClient(app2)
        ck("vLLM's attach_endpoint_plugins puts the Monitor at /, replacing the engine's root",
           "id=\"kv-pct\"" in c2.get("/").text, c2.get("/").text[:80])
        ck("and leaves the engine's other routes alone", c2.get("/api/host").status_code == 200)

# --------------------------------------------------------------------------- the stat logger itself
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve" / "plugin"))
    import r9700_webui as P2
except ImportError:
    print("  skip stat logger: the plugin package did not import")
else:
    before = len(P2.STORE)
    P2.RequestLog(None, 0).record(None, type("It", (), {"finished_requests": [Fin(request_id="log-1")]})())
    ck("the engine's finished-request list lands in the ring the page reads",
       len(P2.STORE) == before + 1 and P2.STORE.listing()[0]["id"] == "log-1", str(len(P2.STORE)))
    ck("a step with no finished requests records nothing",
       (P2.RequestLog(None, 0).record(None, None) or len(P2.STORE)) == before + 1)

print("PASS" if fail == 0 else "FAIL", f"({fail} failing)")
sys.exit(1 if fail else 0)
