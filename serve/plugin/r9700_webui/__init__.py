"""The Monitor and About pages served by the vLLM serve itself, on its own port (r9700-stack).

One package, two entry points, which is the pairing vLLM documents for exactly this case
(`vllm/plugins/endpoint_plugins/interface.py`: an endpoint plugin "must not open new paths into the
engine"; pair it with a plugin registered under another group that does the engine-side work):

  `vllm.stat_logger_plugins` -> `RequestLog`
      Sees every FINISHED request as the engine reports it: prompt / generation tokens and the
      queued / prefill / inference / decode split (`vllm/v1/metrics/stats.py:288-306`). No relay, no
      byte-sniffing, no estimating -- this is a strictly better per-request record than `--proxy` can
      build, which is why merging the pages into the serve's port is not a downgrade of the Monitor.

  `vllm.endpoint_plugins` -> `WebMonitor`
      The routes: `/`, `/monitor`, `/about`, `/web/*`, `/api/*`. Replacing `GET /` is supported by
      design (`attach_endpoint_plugins` logs "Endpoint plugin overrides GET /").

Both are opt-in twice over: the entry points have to be pip-installed, and an endpoint plugin is never
loaded unless it is named in `VLLM_PLUGINS` (`vllm/plugins/__init__.py`: a stricter posture than other
plugin groups, because it is a larger exposed network surface). `WEBUI=same` in the launch scripts sets
that; `setup.sh` installs the package.

Three things merged mode does not have, and no in-process route can invent them:

- **the request text.** The engine's per-request stats carry tokens and timings, not bodies. Captured
  bodies need the sidecar in proxy mode (`serve/webui.sh`, `WEBUI_MODE=proxy`).
- **in-flight rows.** A stat logger sees a request when it finishes. The running / waiting COUNTS are
  on the page anyway, from the gauges.
- **a page that outlives the engine.** Same port, same process: when the launch dies the monitor dies
  with it -- which is the whole reason the sidecar on 8081 exists (notes/webui.md) and why both stay.

The polling, the amdgpu sysfs reads (which can block on EBUSY) and the rocm-smi subprocess (when
`R9700_WEBUI_SMI=1`) all run on their own daemon threads, and every route handler is a plain `def`, so
FastAPI runs it on its threadpool rather than on the event loop. Nothing here may sit on the loop that
produces the numbers the page displays.
"""

import os
import sys
import threading
from pathlib import Path

# The Monitor's own code -- parser, rings, sysfs reader, page serving -- is one implementation, shared
# with the sidecar: serve/webui.py. This package is installed editable from serve/plugin, so the repo
# layout is known; R9700_WEBUI_SERVE overrides it for the container path, where the repo is mounted.
_SERVE = Path(os.environ.get("R9700_WEBUI_SERVE") or Path(__file__).resolve().parents[2])
if str(_SERVE) not in sys.path:
    sys.path.insert(0, str(_SERVE))

import webui as W                                             # noqa: E402

STORE = W.EngineRecords(int(os.environ.get("R9700_WEBUI_KEEP", "100")))
HUB = None
KEY = [None]                                                  # the serve's own --api-key, if it has one


def registry_text():
    """/metrics without a socket: the same registry the serve's own /metrics renders.

    Reading it directly is what makes merging cost nothing -- the sidecar has to fetch its own upstream
    over HTTP, which is fine when it is a different process and silly when it is this one. Same bytes,
    same parser (vllm/entrypoints/serve/instrumentator/metrics.py mounts make_asgi_app(registry) on the
    very registry get_prometheus_registry() returns)."""
    from prometheus_client import generate_latest
    from vllm.v1.metrics.prometheus import get_prometheus_registry
    return generate_latest(get_prometheus_registry()).decode("utf-8", "replace")


def _version():
    try:
        from vllm.version import __version__
        return f"vllm {__version__} (this process)"
    except Exception:
        return "vllm (version not read)"


# -------------------------------------------------------------------------------------------- engine side
try:                                                  # inside vLLM this import always resolves; the shim is
    from vllm.v1.metrics.loggers import StatLoggerBase     # for importing this module without vLLM (tests)
except Exception:                                     # and vLLM's own loader then never runs here
    class StatLoggerBase:                            # noqa: B024  - a shim, not an interface claim
        def __init__(self, vllm_config, engine_index=0): ...


class RequestLog(StatLoggerBase):
    """The engine's own per-request records, kept in this process.

    vLLM calls `record()` once per engine step batch, with `IterationStats.finished_requests` holding a
    `FinishedRequestStats` for every request that finished in it: request id, finish reason, prompt and
    generation tokens, cached tokens, and queued / prefill / inference / decode seconds
    (vllm/v1/metrics/stats.py:288-306). The queue/prefill/decode split per request is the number the
    sidecar can only give as a window mean -- here it is measured, per row.
    """

    def __init__(self, vllm_config, engine_index: int = 0):
        self.vllm_config, self.engine_index = vllm_config, engine_index

    def log_engine_initialized(self):
        pass

    def record(self, scheduler_stats=None, iteration_stats=None, mm_cache_stats=None, engine_idx: int = 0):
        if iteration_stats is None:
            return
        for fin in getattr(iteration_stats, "finished_requests", ()) or ():
            STORE.add(fin)


# ----------------------------------------------------------------------------------------------- routes
class WebMonitor:
    name = "r9700_webui"
    required_tasks = None                             # not tied to a supported task: it serves pages

    # ---- pages and assets (the same files the sidecar serves, the same guard) ------------------------
    def attach_router(self, app):
        from fastapi import APIRouter, Request
        from fastapi.responses import JSONResponse, Response

        rt = APIRouter()

        def send(path):
            return Response(path.read_bytes(), media_type=W.KIND.get(path.suffix, "application/octet-stream"),
                            headers={"Cache-Control": "no-cache"})

        def page(name):
            p = W.WEB / name
            return send(p) if p.is_file() else JSONResponse({"error": {"message": f"{name} not found"}},
                                                             status_code=404)

        def gated(request):
            """The serve's --api-key guards /api/*, exactly as --key guards the sidecar's: the HTML holds
            no secrets, and the numbers it would leak are already on an unauthenticated /metrics."""
            if not KEY[0]:
                return None
            got = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
            return None if got and got == KEY[0] else JSONResponse(
                {"error": {"message": "this monitor needs the serve's API key", "type": "auth"}}, status_code=401)

        @rt.get("/")
        def root():                                   # attach_endpoint_plugins puts a plugin route ahead
            return page("monitor.html")               # of the operation it replaces (interface.py:95ff)

        @rt.get("/monitor")
        def monitor():
            return page("monitor.html")

        @rt.get("/about")
        def about():
            return page("about.html")

        @rt.get("/favicon.svg")
        def favicon():
            return page("favicon.svg")

        @rt.get("/web/{rel:path}")
        def asset(rel: str):
            p = W.web_asset(W.WEB, rel)               # component check, not prefix: see webui.web_asset
            return send(p) if p else JSONResponse({"error": {"message": f"{rel} not found"}}, status_code=404)

        @rt.get("/api/status")
        def status(request: Request):
            return gated(request) or JSONResponse(HUB.api("/api/status") or {"error": {"message": "no hub"}})

        @rt.get("/api/requests")
        def requests(request: Request):
            rid = request.query_params.get("id")
            if rid:
                d = HUB.record_detail(rid)
                return JSONResponse(d or {"error": {"message": f"{rid} not kept"}}, status_code=200 if d else 404)
            return JSONResponse(HUB.api("/api/requests"))

        @rt.get("/api/host")
        def host(request: Request):
            return gated(request) or JSONResponse(HUB.api("/api/host"))

        app.include_router(rt)

    # ---- and the hub that feeds them ---------------------------------------------------------------
    async def init_state(self, engine_client, state, args):
        global HUB
        port = getattr(args, "port", None) or 8000
        KEY[0] = getattr(args, "api_key", None) or None
        cfg = {"mode": "merged", "upstream": f"http://127.0.0.1:{port}", "key": KEY[0],
               "poll": float(os.environ.get("R9700_WEBUI_POLL", "2")), "keep": STORE.keep,
               # the SAME knobs and defaults serve/webui.py uses, so one env works for both. Defaulting
               # gpus to None was a live bug: the merged page came up with no cards at all.
               "gpus": os.environ.get("R9K_WEB_GPUS", "/sys/class/drm"),
               "smi": (W.SMI_DEFAULT if os.environ.get("R9K_WEB_SMI", "") == "1"
                       else os.environ.get("R9K_WEB_SMI", "").split()) or None,
               "meta": os.environ.get("WEBUI_META"),
               "version": _version(), "timeout": 4.0, "engine_records": STORE, "metrics_fn": registry_text}
        HUB = W.Hub(cfg)
        for fn in (HUB.poll, HUB.slow_poll, HUB.gpu_poll):
            threading.Thread(target=fn, name=f"r9700-webui-{fn.__name__}", daemon=True).start()
