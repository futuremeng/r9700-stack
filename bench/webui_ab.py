#!/usr/bin/env python3
"""bench/webui_ab.py - what does the Monitor's relay cost, and what does recording a request cost?

serve/webui.py can sit in front of a serve (--proxy) to get the per-request records stock vLLM cannot
give anybody. The question the page cannot answer is what that hop is worth. Three arms, one process,
the same requests:

  direct   the serve itself                       (the baseline every client sees)
  relay    the sidecar in observe mode, relaying  (the hop, no bookkeeping)
  record   the sidecar in proxy mode, relaying    (the hop + parsing the SSE deltas as they pass)

Prompt length is the axis that matters: the hop is a fixed cost, so its share of a request should fall
as the prompt grows. The tiers walk it -- a ~45 token prompt is almost pure hop, a ~3000 token one is
almost pure prefill -- and the summary prints the share at each, which is the curve.

Two shapes per tier: max_tokens=1, where the whole request is the first engine step (and its first
token is the prefill), and max_tokens=64 streamed, where the interesting numbers are the first delta's
arrival (the relay must not buffer) and the per-stream decode speed. Reported as client-side medians:
the engine's histograms would hide a millisecond that lives in the client's path.

usage: bench/webui_ab.py [URL] [--n 25] [--tiers 45,1500,3000] [--shape all|first|stream] [--fake]
  URL     a running serve (default http://127.0.0.1:8080)
  --n     requests per arm per tier (drop it for the long tiers on a real serve: 25 x 3000-token
          prefills is minutes, and nothing about a fixed hop needs 25 samples to see)
  --tiers approximate prompt sizes in tokens, measured by the serve itself and reported per row
  --shape first (max_tokens=1), stream (64 streamed) or all (both, the default)
  --fake  answer from a loopback stand-in instead: that isolates our own cost (no GPU, no engine, an
          upstream that answers in ~0 ms). Its prompt "cost" is a fiction, so the tier percentages
          there only prove the plumbing works -- read the ms, not the %.
Run it against a real launch the way you run bench/itl.py: same box, warmed, nothing else on it.
"""
import http.client
import json
import math
import os
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "serve"))
import webui as W  # noqa: E402

ARGS = sys.argv[1:]
N = int(next((ARGS[i + 1] for i, a in enumerate(ARGS) if a == "--n"), 25))
FAKE = "--fake" in ARGS
URL = next((a for a in ARGS if a.startswith("http")), "http://127.0.0.1:8080")
MODEL = os.environ.get("MODEL_NAME", "Qwen3.8")
TIERS = [int(t) for t in next((ARGS[i + 1] for i, a in enumerate(ARGS) if a == "--tiers"),
                              "45,1500,3000").split(",")]
SHAPE = next((ARGS[i + 1] for i, a in enumerate(ARGS) if a == "--shape"), "all")
SHAPES = {"all": [("max_tokens=1", 1, False), ("stream 64 tok", 64, True)],
          "first": [("max_tokens=1", 1, False)],
          "stream": [("stream 64 tok", 64, True)]}[SHAPE]
ITEM = "Item {i}: the quick brown fox jumps over the lazy dog near river bank {i}."
TOK_PER_ITEM = 15                     # close enough to pick a tier; the serve's usage line reports the truth


def prompt_for(approx_tokens):
    items = max(1, math.ceil(approx_tokens / TOK_PER_ITEM))
    return "Summarize in one sentence: " + " ".join(ITEM.format(i=i) for i in range(items))


# ------------------------------------------------------------------ the stand-in upstream -----------
class Fake:
    """An upstream that costs nothing: /v1/models, /metrics, and completions that answer at once.
    Everything in these millisecond numbers is therefore ours (or the loopback's)."""

    def __init__(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                raw = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, "")
                elif self.path == "/v1/models":
                    self._send(200, json.dumps({"data": [{"id": MODEL, "max_model_len": 32768}]}))
                elif self.path == "/metrics":
                    self._send(200, "vllm:generation_tokens_total 0.0\n", "text/plain")
                else:
                    self._send(404, "{}")

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                msgs = body.get("messages") or [{}]
                ptok = max(1, len(msgs[0].get("content", "")) // 4)      # an estimate, labelled as one
                mt = body.get("max_tokens", 1)
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    # no Content-Length and no chunked framing: the body ends at the connection, so say
                    # so (vLLM, like our relay, closes the stream instead)
                    self.close_connection = True
                    self.end_headers()
                    for i in range(mt):
                        self.wfile.write(("data: " + json.dumps({"id": "chatcmpl-f", "choices": [
                            {"index": 0, "delta": {"content": f" t{i}"}}]}) + "\n\n").encode())
                        self.wfile.flush()
                        time.sleep(0.004)                                # ~250 tok/s: a real serve is slower
                    self.wfile.write(('data: ' + json.dumps({"id": "chatcmpl-f", "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": ptok, "completion_tokens": mt}})
                                     + "\n\n\ndata: [DONE]\n\n").encode())
                    self.wfile.flush()
                else:
                    self._send(200, json.dumps({"id": "chatcmpl-f", "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "x"}}],
                        "usage": {"prompt_tokens": ptok, "completion_tokens": mt}}))

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{srv.server_address[1]}"


# ------------------------------------------------------------------ the client --------------------
def ask(base, max_tokens, stream, approx_tokens, tries=3):
    """One request; returns (client_s, first_token_s or None, prompt_tokens, completion_tokens)."""
    u = urllib.parse.urlsplit(base)
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt_for(approx_tokens)}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": stream,
                       **({"stream_options": {"include_usage": True}} if stream else {}),
                       **({"chat_template_kwargs": {"enable_thinking": False}} if not FAKE else {})})
    for _ in range(tries):
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
        try:
            t_start = time.perf_counter()          # from the moment we start sending, not from the answer
            conn.request("POST", "/v1/chat/completions", body=body,
                         headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
            res = conn.getresponse()
            first, ptok, ctok = None, None, None
            if "text/event-stream" in (res.getheader("Content-Type") or ""):
                buf = b""
                while True:
                    chunk = res.read(4096)
                    if not chunk:
                        break
                    if first is None and b"data:" in chunk:
                        first = time.perf_counter() - t_start
                    buf += chunk
                for line in buf.split(b"\n"):
                    if line.startswith(b"data:") and b"usage" in line:
                        try:
                            u2 = json.loads(line[5:].strip()).get("usage") or {}
                            ptok, ctok = u2.get("prompt_tokens"), u2.get("completion_tokens")
                        except ValueError:
                            pass
            else:
                raw = json.loads(res.read() or b"{}")
                first = time.perf_counter() - t_start
                u2 = raw.get("usage") or {}
                ptok, ctok = u2.get("prompt_tokens"), u2.get("completion_tokens")
            if res.status != 200:
                raise RuntimeError(f"HTTP {res.status}")
            return time.perf_counter() - t_start, first, ptok, ctok
        except Exception as e:
            last = e
        finally:
            conn.close()
    raise last


def med(xs):
    return statistics.median(xs) * 1e3 if xs else float("nan")


def main():
    global MODEL
    if FAKE:
        url = Fake().url
        print(f"webui_ab: stand-in upstream {url} -- its prompt cost is a fiction; read the ms, not the %")
    else:
        url = URL.rstrip("/")
        models = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=10).read())
        MODEL = next((m["id"] for m in models.get("data", [])), MODEL)
        print(f"webui_ab: serve {url} model {MODEL}; warming", flush=True)
        for _ in range(3):
            ask(url, 1, False, 45)

    def sidecar(mode):
        cfg = {"port": 0, "upstream": url, "mode": mode, "key": "", "poll": 2.0, "keep": 200,
               "meta": "", "gpus": "/sys/class/drm", "smi": None, "timeout": 600.0, "verbose": False,
               "version": ""}
        _, srv = W.build(cfg)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, f"http://127.0.0.1:{srv.server_address[1]}"

    srvs = [sidecar(m) for m in ("observe", "proxy")]
    arms = [("direct", url), ("relay", srvs[0][1]), ("record", srvs[1][1])]
    hdr = f"{'tier':>8s} {'shape':14s} {'arm':7s} {'n':>3s} {'prompt tok':>10s} {'median ms':>10s} "\
          f"{'p95 ms':>8s} {'first tok ms':>13s} {'tok/s':>7s}  {'vs direct':>18s}"
    print(hdr)
    rows = []
    for approx in TIERS:
        for shape, mt, stream in SHAPES:
            base_median = base_first = None
            ptok = None
            for name, target in arms:
                ws, fs, toks = [], [], []
                for _ in range(N):
                    w, f, p, k = ask(target, mt, stream, approx)
                    ws.append(w)
                    if f is not None:
                        fs.append(f)
                    if p:
                        ptok = p
                    if k:
                        toks.append(k)
                ws.sort()
                if base_median is None:
                    base_median, base_first = med(ws), med(fs)
                    d = dfirst = 0.0
                else:
                    d, dfirst = med(ws) - base_median, med(fs) - base_first
                tok_s = (sum(toks) / sum(ws)) if toks and sum(ws) else float("nan")
                share = 100.0 * d / base_median if base_median else float("nan")
                rows.append((approx, shape, name, d, dfirst, share))
                print(f"{approx:8d} {shape:14s} {name:7s} {N:3d} {str(ptok or '-'):>10s} {med(ws):10.2f} "
                      f"{ws[min(N - 1, int(0.95 * (N - 1)))] * 1e3:8.2f} "
                      f"{(f'{med(fs):.2f}' if fs else '-'):>13s} {tok_s:7.1f}  "
                      f"{('%+.2f ms (%+.2f%%)' % (d, share)) if name != 'direct' else '-':>18s}", flush=True)
                time.sleep(0.2)
    print(f"\nhop cost against prompt length ({'stand-in: the ms are ours, the %s are not' if FAKE else 'client medians'}):")
    for approx, shape, name, d, dfirst, share in rows:
        if name == "direct":
            continue
        print(f"  {approx:>6d} tok  {shape:14s} {name:6s}  request {d:+8.2f} ms ({share:+6.2f}%)   "
              f"first token {dfirst:+7.2f} ms")
    print(f"(the sidecar's own threads sit in this process too, so the run cost {time.process_time():.2f} "
          f"CPU-s of relaying and parsing)")
    for srv, _ in srvs:
        srv.shutdown()


if __name__ == "__main__":
    main()
