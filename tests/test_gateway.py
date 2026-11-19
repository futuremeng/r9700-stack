#!/usr/bin/env python3
"""tests/test_gateway.py - the framing of the thing that fronts every port on this box.

serve/gateway.py is deliberately not a web framework: it reads the request line, opens the upstream, and pipes
bytes both ways. That is why it exists (serve/webui.py's proxy costs 2-3 ms a request because it parses the
body and rebuilds it; the gateway measured +0.8 ms, notes/embed-sidecar-box-test.md) and it is also the whole
risk: every bug in it is an HTTP framing bug, which a client sees as a hang or a truncated reply, not as a
nonzero exit. So this file tests framing, not features:

  (1) routing, including the /api/sidecars the gateway answers itself, and a path that merely STARTS with a
      routed prefix (/v1/embeddings-x) taking the default instead -- plus the paths it refuses to route at all:
      the engine's control plane (/sleep, /reset_prefix_cache, /collective_rpc, /update_weights, ...), which
      vLLM's --api-key does not guard and which therefore may not be reachable through a door that answers on
      a LAN interface. GATEWAY_ADMIN=1 forwards them, loopback only.
  (2) a large body arriving byte-identical. A 60 KB chat request through the gateway: the upstream's copy must
      match by length and digest.
  (3) `Expect: 100-continue`. read_headers() used to loop until the bytes it had collected ended in \\r\\n\\r\\n,
      which made it read straight past the upstream's "HTTP/1.1 100 Continue\\r\\n\\r\\n" and swallow the real
      status line: a 96 KB body hung with no error anywhere. A first interim response must now be forwarded,
      not eaten. This check reproduced that hang before the fix and is the reason the function stops at the
      first blank line instead of at the end of what the kernel happened to hand it.
  (4) streaming is not buffered: a server that writes three chunks with a gap between them must reach the
      client with the first chunk before the last one is written.
  (5) keep-alive: two requests down one connection, both answered, both counted upstream.
  (6) the hop is cheap. 40 sequential small GETs through the gateway vs straight to the same upstream: the
      median difference must stay under 2 ms, which is what a body-parsing proxy costs. If someone replaces
      this file with a framework, this check fails rather than quietly shipping the latency.
  (7) the admission gate: it closes on the ENGINE's own harm signals -- a preemption counter that moved, a KV
      pool near full -- NOT on a pool that is merely busy or an engine that is merely queueing its own work, so a
      batch of 100k-token prompts cannot shut the door on everything for its whole duration. It holds a new chat
      request (no upstream connection) instead of refusing it, re-opens on a window that ends, lets a request
      declaring a small body through a shut door (the ones v1 starved), bounds how many it will hold at all, keeps
      its ceiling promise, refuses only when told to, and OPENS when the gauge cannot be read.

Nothing here needs a GPU, a model, or the box's real serve: the upstreams are three sockets on loopback.
"""
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
fail = 0


def ck(name, cond, detail=""):
    global fail
    if not cond:
        fail += 1
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond or not detail else f"  {detail}"))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# -- fake upstreams -------------------------------------------------------------------------------------
# One server, three roles: it answers with what it received, so the body checks need no fixture.
LOG = {"engine": [], "embed": [], "rerank": []}
STATE_HITS = {"n": 0}
# What the fake engine says about itself. Mutable from the checks, because the gate's decision variable is the
# engine's own gauge and the only honest way to test a threshold is to move the number it reads.
GAUGE = {"kv": 0.10, "run": 0, "wait": 0, "prem": 0}
METRICS = {}                                    # port -> bytes, for an engine whose /metrics has no gauges
SLOW_METRICS = set()                            # ports whose /metrics hangs, so the gauge goes stale


def metrics_text(port):
    """vLLM 0.32's shape, byte for byte where it matters. The two `by_reason` lines are there on purpose: the
    gateway matches `vllm:num_requests_waiting{` with its brace, and without that it would read a scheduler
    waiting-for-a-reason counter as a queue it grew -- this fixture makes that failure visible."""
    j = METRICS.get(port, GAUGE)
    if isinstance(j, bytes):
        return j
    return ("# TYPE vllm:kv_cache_usage_perc gauge\n"
            'vllm:kv_cache_usage_perc{engine="0",model_name="Qwen3.8"} %.4f\n' % j["kv"] +
            'vllm:num_requests_running{engine="0",model_name="Qwen3.8"} %d.0\n' % j["run"] +
            'vllm:num_requests_waiting{engine="0",model_name="Qwen3.8"} %d.0\n' % j["wait"] +
            'vllm:num_preemptions_total{engine="0",model_name="Qwen3.8"} %d.0\n' % j.get("prem", 0) +
            'vllm:num_requests_waiting_by_reason{engine="0",model_name="Qwen3.8",reason="capacity"} 7.0\n'
            'vllm:num_requests_waiting_by_reason{engine="0",model_name="Qwen3.8",reason="deferred"} 9.0\n').encode()


def _read_request(c, on_headers=None):
    """The request line + headers, then exactly Content-Length bytes. Stops at the first blank line -- the
    same rule the gateway has to get right, so a test that reuses it cannot hide the gateway's mistake.
    on_headers fires between the two, which is where a server that honours Expect: sends its 100 Continue."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = c.recv(65536)
        if not d:
            return None, buf
        buf += d
    head, rest = buf.split(b"\r\n\r\n", 1)
    lines = head.split(b"\r\n")
    method, path, _vers = lines[0].split(b" ")
    hdrs = {}
    for l in lines[1:]:
        k, _, v = l.partition(b": ")
        hdrs[k.lower().decode()] = v.decode()
    if on_headers:
        on_headers(method, path, hdrs, len(rest))
    need = int(hdrs.get("content-length", 0))
    body = rest
    while len(body) < need:
        body += c.recv(65536)
    return (method.decode(), path.decode(), hdrs), body


def upstream(role, port, slow_stream=False):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(64)

    def loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=one, args=(c,), daemon=True).start()

    def one(c):
        try:
            while True:                                    # one connection may carry several requests
                def interim(method, path, hdrs, already):
                    # uvicorn/h11 (the real engine) answers an Expect: before the body arrives; a fake that
                    # skipped it would let a gateway that EATS the interim pass a check it should fail.
                    if hdrs.get("expect", "").lower() == "100-continue" and already == 0:
                        c.sendall(b"HTTP/1.1 100 Continue\r\n\r\n")
                req, body = _read_request(c, interim)
                if req is None:
                    return
                method, path, hdrs = req
                LOG[role].append((method, path, len(body), hdrs.get("expect")))
                if role == "engine" and slow_stream and path == "/slow":
                    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: "
                              b"chunked\r\n\r\n")
                    for i in range(3):
                        c.sendall(b"%x\r\ndata: chunk%d\n\n\r\n" % (14, i))   # 14 = len("data: chunkN\\n\\n")
                        time.sleep(0.4)
                    c.sendall(b"0\r\n\r\n")
                    return
                payload = json.dumps({"role": role, "path": path, "n": len(body),
                                      "sha": hashlib.sha256(body).hexdigest()[:16],
                                      "count": len(LOG[role])}).encode()
                if role == "engine" and path == "/metrics":     # what the gate reads, not an echo
                    if port in SLOW_METRICS:                    # a /metrics that never answers
                        time.sleep(6)
                    txt = metrics_text(port)
                    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: "
                              + str(len(txt)).encode() + b"\r\n\r\n" + txt)
                    if hdrs.get("connection", "").lower() == "close":
                        return
                    continue
                if method == "HEAD":                        # headers and no body, whatever the method asks
                    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                              + str(len(payload)).encode() + b"\r\n\r\n")
                else:
                    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                              + str(len(payload)).encode() + b"\r\n\r\n" + payload)
                if hdrs.get("connection", "").lower() == "close":
                    return
        except OSError:
            pass
        finally:
            c.close()

    threading.Thread(target=loop, daemon=True).start()
    return port


def state_server(port, slow_s=0.45):
    """A watcher's state.json. Slow on purpose (0.45 s, a little over what this box's watchers actually take):
    the gateway's answer is only correct if a slow watcher can hold up the rest of the port."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(16)

    def loop():
        while True:
            c, _ = srv.accept()
            _read_request(c)
            STATE_HITS["n"] += 1
            time.sleep(slow_s)
            body = json.dumps({"health": 200, "rates": {"rate": 1112.0}, "serve": {"threads": "15", "ctx": "8192"},
                               "host": {"committed_as_kb": 89397784, "embed_rss_kb": 1618092}}).encode()
            c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                      + str(len(body)).encode() + b"\r\n\r\n" + body)
            c.close()

    threading.Thread(target=loop, daemon=True).start()
    return port


def read_one(s, no_body=False):
    """One response: status line, headers, then exactly Content-Length bytes (or none for a HEAD). Reading to
    EOF instead is the client bug the first version of this file had: the gateway is allowed to keep a
    keep-alive connection open after answering, so `recv` until EOF sat there until the socket timed out and
    every check in the file looked slow -- and the timing check became unmeasurable. A real client stops at
    the end of the body, so this does too."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        d = s.recv(65536)
        if not d:
            return None, buf
        buf += d
    head, rest = buf.split(b"\r\n\r\n", 1)
    hdr = {}
    for l in head.split(b"\r\n")[1:]:
        k, _, v = l.partition(b": ")
        hdr[k.lower()] = v
    need = 0 if no_body else int(hdr.get(b"content-length", 0))
    while len(rest) < need:
        rest += s.recv(65536)
    return head.split(b"\r\n")[0].decode(), rest


def http(port, path, method="GET", body=None, headers=None, timeout=10):
    s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    data = body if isinstance(body, bytes) else (body or "").encode()
    h = {"host": f"127.0.0.1:{port}", "connection": "close", "content-length": str(len(data))}
    h.update(headers or {})
    s.sendall(f"{method} {path} HTTP/1.1\r\n".encode()
              + b"".join(f"{k}: {v}\r\n".encode() for k, v in h.items()) + b"\r\n" + data)
    out = read_one(s, no_body=(method == "HEAD"))
    s.close()
    return out


# -- bring it up ---------------------------------------------------------------------------------------
random.seed(os.getpid())
ENG, EMB, RER = (free_port() for _ in range(3))
STA, GW = free_port(), free_port()
SLOW = free_port()
upstream("engine", ENG)
upstream("engine", SLOW, slow_stream=True)
upstream("embed", EMB)
upstream("rerank", RER)
state_server(STA)

env = {**os.environ, "GATEWAY_PORT": str(GW), "GATEWAY_BIND": "127.0.0.1",
       # ENGINE is the streaming fake: it answers every path the way the plain one does except for /slow, which
       # is the one the streaming check needs. PAGES stays on the plain fake so the /monitor check proves the
       # pages really go to a different upstream than the chat path does.
       "ENGINE": f"http://127.0.0.1:{SLOW}", "EMBED": f"http://127.0.0.1:{EMB}",
       "RERANK": f"http://127.0.0.1:{RER}", "PAGES": f"http://127.0.0.1:{ENG}",
       "SIDECAR_STATE": f"/v1/embeddings=http://127.0.0.1:{STA}/state.json,embed=http://127.0.0.1:{STA}/state.json",
       # no caching in the main run, so the checks below really do drive watcher fetches; the cache gets its own
       # gateway a few lines further down, with its own port
       "SIDECAR_TTL": "0",
       # the gate is off in THIS gateway on purpose: its poller would add GET /metrics to the upstream's log, and
       # the keep-alive check below counts the requests the upstream saw. The gate gets its own gateways.
       "GATE_DISABLE": "1"}
GWLOG = tempfile.NamedTemporaryFile(prefix="r9700-gateway-test-", suffix=".log", delete=False)
gw = subprocess.Popen([sys.executable, os.path.join(ROOT, "serve/gateway.py")], env=env,
                      stdout=subprocess.DEVNULL, stderr=GWLOG)
for _ in range(100):
    try:
        socket.create_connection(("127.0.0.1", GW), timeout=1).close()
        break
    except OSError:
        time.sleep(0.1)
else:
    print("the gateway did not come up")
    sys.exit(1)

try:
    # (1) routing
    st, b = http(GW, "/v1/chat/completions", "POST", '{"x":1}')
    ck("a chat request reaches the engine", json.loads(b)["role"] == "engine", b[:80])
    st, b = http(GW, "/v1/embeddings", "POST", '{"input":["a"]}')
    ck("/v1/embeddings reaches the embedder", json.loads(b)["role"] == "embed", b[:80])
    for p in ("/v1/rerank", "/rerank", "/reranking"):
        st, b = http(GW, p, "POST", '{"query":"q"}')
        ck(f"{p} reaches the reranker", json.loads(b)["role"] == "rerank", b[:80])
    st, b = http(GW, "/monitor")
    ck("/monitor goes to the pages upstream (the engine's plugin)", json.loads(b)["path"] == "/monitor", b[:80])
    st, b = http(GW, "/v1/embeddings-x", "POST", "{}")
    ck("a path that merely starts with a routed prefix is not captured by it",
       json.loads(b)["role"] == "engine", b[:80])
    st, b = http(GW, "/api/sidecars")
    d = json.loads(b)
    ck("/api/sidecars is answered by the gateway itself, with the configured name and the watcher's numbers",
       d["sidecars"][0]["name"] == "/v1/embeddings" and d["sidecars"][0]["rates"]["rate"] == 1112.0, b[:120])
    # the name is allowed to be a route: the split has to land on the FIRST = and only when it precedes ://, or
    # "/v1/embeddings=http://host/s?x=1" would come back named "/v1/embeddings=http://host/s?x".
    ck("a name that is itself a path, in front of a url that carries =, still splits once",
       [x["name"] for x in d["sidecars"]] == ["/v1/embeddings", "embed"], str([x["name"] for x in d["sidecars"]]))
    _hd = http(GW, "/monitor", "HEAD")
    ck("HEAD gets headers and no body", "200" in _hd[0] and _hd[1] == b"", repr(_hd)[:160])

    # The loop must stay free while a page polls /api/sidecars: the watcher fetches are blocking work, and a
    # Monitor tab asks for this every 2 s. The first version called urllib straight from the async handler, so
    # each poll parked the event loop for the whole fetch -- on the box that showed up as streamed chat chunks
    # arriving in bursts (median gap 0.0 ms against an engine decoding at its normal 13 ms), which is the kind
    # of harm no client attribute in the response would ever reveal.
    ths = [threading.Thread(target=lambda: http(GW, "/api/sidecars", timeout=20)) for _ in range(3)]
    for t in ths:
        t.start()
    time.sleep(0.05)
    t0 = time.time()
    st, b = http(GW, "/v1/models", timeout=10)
    hop = time.time() - t0
    for t in ths:
        t.join()
    ck("three slow /api/sidecars polls in flight do not hold up another request (the fetches leave the loop)",
       hop < 0.2, f"/v1/models took {hop*1000:.0f} ms behind three 0.45 s watcher fetches")

    # (1b) what the door REFUSES to forward. vLLM's --api-key guards /v1 /v2 /inference /cohere and nothing else,
    # so the engine's control plane sits outside it: a door whose rule is "everything else -> ENGINE" hands any
    # client that can reach the door POST /sleep (the model stops), /reset_prefix_cache (the cache anyone is
    # hitting goes away) and /collective_rpc (run code on the ranks). The list is the installed wheel's route
    # table, not the docs' (serve/gateway.py:ADMIN).
    before = len(LOG["engine"])
    for p in ("/sleep", "/wake_up", "/reset_prefix_cache", "/abort_requests", "/collective_rpc",
              "/update_weights", "/release_kv_cache_memory", "/start_profile"):
        st, b = http(GW, p, "POST", "{}")
        ck(f"{p} is not forwarded to the engine",
           st.startswith("HTTP/1.1 404") and json.loads(b)["error"]["type"] == "gateway_admin", f"{st} {b[:70]}")
    ck("and none of them reached the engine: the refusal happens before an upstream connection exists",
       len(LOG["engine"]) == before, f"{len(LOG['engine']) - before} request(s) got through")
    st, b = http(GW, "/api/gate")
    d = json.loads(b)
    ck("the refusals are counted where the Monitor can read them, and the knob's state is named with them",
       d["admin"]["denied"] >= 8 and d["admin"]["GATEWAY_ADMIN"] is False and d["admin"]["paths"] >= 20,
       str(d["admin"])[:120])
    for p in ("/version", "/load", "/server_info", "/tokenizer_info"):
        st, b = http(GW, p)
        ck(f"a read-only engine diagnostic ({p}) still goes through",
           json.loads(b)["path"] == p, b[:80])
    # GATEWAY_ADMIN=1 forwards -- to a loopback peer, which is all this test can be. The peer comes from the
    # socket: a header cannot make a LAN connection look like a tunnel, so the loopback rule is not negotiable
    # by the client.
    AP = free_port()
    agw = subprocess.Popen([sys.executable, os.path.join(ROOT, "serve/gateway.py")],
                           env={**env, "GATEWAY_ADMIN": "1", "GATEWAY_PORT": str(AP)},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", AP), timeout=1).close()
            break
        except OSError:
            time.sleep(0.1)
    st, b = http(AP, "/sleep", "POST", "{}")
    ck("GATEWAY_ADMIN=1 does forward it, so the operator's escape hatch is real",
       json.loads(b)["path"] == "/sleep", b[:80])
    st, b = http(AP, "/api/gate")
    ck("and that door says so in its own state", json.loads(b)["admin"]["GATEWAY_ADMIN"] is True, b[:80])
    agw.terminate()

    # (2) a big body arrives whole
    big = os.urandom(60 * 1024)
    st, b = http(GW, "/v1/chat/completions", "POST", big, {"content-type": "application/octet-stream"})
    r = json.loads(b)
    ck("a 60 KB body arrives byte-identical (length and digest)",
       r["n"] == len(big) and r["sha"] == hashlib.sha256(big).hexdigest()[:16], str(r)[:120])

    # (3) Expect: 100-continue, sent in two writes with a pause: the gateway must forward the interim response
    # and then the body, rather than reading the interim's blank line as the end of the upstream's headers.
    s = socket.create_connection(("127.0.0.1", GW), timeout=15)
    s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\nconnection: close\r\ncontent-length: 40000\r\n'
              b'expect: 100-continue\r\n\r\n')
    time.sleep(0.7)                       # the client is waiting for 100 Continue before it sends the body
    s.sendall(b"x" * 40000)
    out = b""
    s.settimeout(8)
    while True:                                   # stop as soon as the final response is complete, the way a
        try:                                      # client does -- not at EOF, which the gateway may withhold
            d = s.recv(65536)
        except Exception:
            break
        if not d:
            break
        out += d
        if out.count(b"\r\n\r\n") >= 2:
            try:
                if json.loads(out.split(b"\r\n\r\n")[-1]).get("n") == 40000:
                    break
            except Exception:
                pass
    s.close()
    tail = out.split(b"\r\n\r\n")[-1]
    try:
        r = json.loads(tail)
    except Exception:
        r = {}
    ck("Expect: 100-continue with a 40 KB body does not hang, and the interim response reaches the client",
       b"100 continue" in out.lower() and r.get("n") == 40000, repr(out[:200]))

    # (4) streaming: the first chunk must arrive before the upstream has written the last one
    t0 = time.time()
    s = socket.create_connection(("127.0.0.1", GW), timeout=15)
    s.sendall(b"GET /slow HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n")
    t_first = None
    got = b""
    while True:                                   # chunked: read to the zero-length terminator, not to EOF
        try:
            d = s.recv(65536)
        except (socket.timeout, TimeoutError):
            break
        if not d:
            break
        if t_first is None:
            t_first = time.time() - t0
        got += d
        if got.endswith(b"0\r\n\r\n"):
            break
    s.close()
    ck("an SSE stream is forwarded as it is written, not held to the end",
       t_first is not None and t_first < 0.5 and b"chunk2" in got, f"first byte after {t_first:.2f}s")

    # (5) keep-alive: two requests, one connection. read_one() is a real response reader (status line, headers,
    # then exactly Content-Length bytes) rather than the "read until it looks done" loop the first version had,
    # which spun on a connection that stayed open and made the whole file hang with no output at all.
    s = socket.create_connection(("127.0.0.1", GW), timeout=10)
    n_before = len(LOG["engine"])
    s.sendall(b"GET /a HTTP/1.1\r\nhost: x\r\n\r\n")
    st1, r1 = read_one(s)
    s.sendall(b"GET /b HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n")
    st2, r2 = read_one(s)
    s.close()
    ck("keep-alive: two requests down one connection both get answered",
       b"/a" in r1 and b"/b" in r2 and len(LOG["engine"]) - n_before == 2,
       f"{st1} {st2} upstream saw {len(LOG['engine']) - n_before}")

    # The same, across the endpoint the gateway answers ITSELF. It used to `return` after /api/sidecars, which
    # closed the keep-alive connection under a browser that had more polls to send down it: the next request
    # met an EOF or a BrokenPipe and Chrome labelled it "canceled". Every endpoint was answering in under
    # 200 ms the whole time -- the Monitor "kept dropping" because of the connection, not the numbers.
    s = socket.create_connection(("127.0.0.1", GW), timeout=10)
    seq = []
    for p in ("/api/sidecars", "/v1/models", "/api/sidecars", "/v1/models"):
        s.sendall(f"GET {p} HTTP/1.1\r\nhost: x\r\n\r\n".encode())
        stx, body = read_one(s)
        seq.append(stx)
    s.close()
    ck("a keep-alive connection survives the gateway's own /api/sidecars (four polls, one socket)",
       seq.count("HTTP/1.1 200 OK") == 4, str(seq))

    # (7) the front door must not amplify its own polling. The watchers refresh every 2 s; a Monitor tab polls
    # every 2 s; behind a cache-less fan-out every extra tab and every aborted retry buys two more watcher
    # fetches, and on the box that multiplier fed a fork storm that made the page give up.
    GW2 = free_port()
    gw2 = subprocess.Popen([sys.executable, os.path.join(ROOT, "serve/gateway.py")],
                           env={**env, "GATEWAY_PORT": str(GW2), "SIDECAR_TTL": "8"},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", GW2), timeout=1).close()
            break
        except OSError:
            time.sleep(0.1)
    before = STATE_HITS["n"]
    ths = [threading.Thread(target=lambda: http(GW2, "/api/sidecars", timeout=20)) for _ in range(5)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    hits = STATE_HITS["n"] - before
    gw2.terminate()
    gw2.wait(timeout=5)
    ck("five concurrent /api/sidecars polls cost ONE round of watcher fetches (single flight + short TTL)",
       hits <= 2, f"the state server saw {hits} hits for 5 polls (2 = one round, both entries point at it)")

    # (6) the hop stays cheap: 40 small requests through the gateway vs the same upstream directly
    def med(path_via_gw):
        ts = []
        for _ in range(40):
            t0 = time.time()
            http(GW if path_via_gw else SLOW, "/v1/models", timeout=5)
            ts.append(time.time() - t0)
        ts.sort()
        return ts[len(ts) // 2]
    direct, via = med(False), med(True)
    ck("the hop costs under 2 ms on a small request (a body-parsing proxy costs 2-3 ms, which is what this "
       "file exists to keep out)",
       (via - direct) < 0.002, f"direct {direct*1000:.1f} ms, via gateway {via*1000:.1f} ms")

    # (8) the admission gate. Why these thresholds and not a sequence cap: the engine's own 10 s stats fit
    # step_ms ~= 13.6 + 1.79 x N + 14.9 ns x Sum(resident tokens) on this box, so a sequence costs about 1.8 ms
    # a step while its TOKENS are what eat the 902,153-token pool -- one 80k row is 8.9% of it, the footprint
    # of 53 short rows. The slow windows of the 6.8-day launch sat at kv 84-100% with a waiting queue (TTFT p99
    # 376 s, max 1 960 s); the fine launch ran the SAME --max-num-seqs 32. So the gate reads the pool.
    def bring(extra, base=None):
        p = free_port()
        pr = subprocess.Popen([sys.executable, os.path.join(ROOT, "serve/gateway.py")],
                              env={**(base or GENV), **extra, "GATEWAY_PORT": str(p)},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", p), timeout=1).close()
                return p, pr
            except OSError:
                time.sleep(0.1)
        raise AssertionError("the gate gateway did not come up")

    GENV = {**env, "GATE_DISABLE": "0", "GATE_POLICY": "hold", "ENGINE": f"http://127.0.0.1:{ENG}",
            "GATE_METRICS_S": "0.1", "GATE_STALE_S": "2", "GATE_KV_HIGH": "0.85", "GATE_KV_LOW": "0.78",
            "GATE_WAIT": "1", "GATE_HOLD_S": "1.5"}
    # Two things are shipped defaults here and neither is what these v1-era thresholds are: the MODE is observe (so
    # UNARMED drops GATE_POLICY and gets the shipped one), and the thresholds are 0.92/0.85 with WAIT=0 and a 5 s
    # ceiling (so DEFAULTS drops the tuned ones). The gate has to be tested as configured AND as shipped: the armed
    # door's thresholds are what a re-armed box will run, and observe is what the box runs now.
    UNARMED = {k: v for k, v in GENV.items() if k != "GATE_POLICY"}
    DEFAULTS = {k: v for k, v in GENV.items() if k not in ("GATE_KV_HIGH", "GATE_KV_LOW", "GATE_WAIT", "GATE_HOLD_S")}
    # A body that declares more than GATE_SMALL_BYTES: the hold path needs a request big enough to be the kind of
    # request the door exists for. The v1-era one-line body is now, correctly, a request that always passes.
    BIG = '{"pad":"' + "y" * 9000 + '"}'
    GAUGE.update({"kv": 0.10, "run": 1, "wait": 0, "prem": 0})
    GP, ggate = bring({})

    def gstate(p=GP):
        return json.loads(http(p, "/api/gate", timeout=5)[1])

    g = gstate()
    ck("the gate reads the ENGINE's own gauges (not an estimate of the body it has not parsed)",
       abs(g["gauges"]["kv"] - 0.10) < 0.001 and g["gauges"]["wait"] == 0 and not g["closed"], str(g)[:200])

    # a closed gate HOLDS: no upstream connection, the client waits, and the request is served once the pool
    # falls -- never refused, never re-sent.
    GAUGE["kv"] = 0.95
    time.sleep(0.3)
    ck("kv above the high threshold closes the gate, and the reason it names carries the token count it is "
       "protecting", gstate()["closed"] and "of 902,153 tokens" in gstate()["why"], gstate()["why"][:160])
    threading.Timer(0.5, lambda: GAUGE.__setitem__("kv", 0.20)).start()   # the pool frees itself mid-hold
    t0 = time.time()
    st, b = http(GP, "/v1/chat/completions", "POST", BIG, timeout=15)
    held = time.time() - t0
    ck("a closed gate holds a chat request and then SERVES it (held, not dropped, not answered by the gateway)",
       held >= 0.35 and json.loads(b)["role"] == "engine", f"waited {held:.2f} s, {st}")
    g = gstate()
    ck("the hold is counted as a hold and a release, so a late request is explainable afterwards",
       g["counters"]["held"] >= 1 and g["counters"]["released"] >= 1, str(g["counters"]))

    # hysteresis: one 8 192-token prefill moves this pool ~0.9%, so a single threshold would chatter closed/open
    # once an hour and turn the gate into a metronome.
    GAUGE["kv"] = 0.95
    time.sleep(0.3)
    closed_hi = gstate()["closed"]
    GAUGE["kv"] = 0.82                                # between LOW and HIGH: must STAY closed
    time.sleep(0.3)
    mid = gstate()
    GAUGE["kv"] = 0.70
    time.sleep(0.3)
    ck("hysteresis: closes at HIGH, stays shut down to LOW, opens below it",
       closed_hi and mid["closed"] and not gstate()["closed"],
       f"0.95->{closed_hi} 0.82->{mid['closed']} 0.70->{gstate()['closed']}")

    # The second signature of the collapse windows: the engine queueing. As a KNOB it still works (GATE_WAIT=1
    # here), and it must read num_requests_waiting{, NOT the *_by_reason family -- a request the scheduler is
    # holding for a reason is not a queue the engine grew. As SHIPPED policy it is off: see the defaults block
    # below, where a full pool plus a 17-deep engine queue and no preemption leaves the door open on purpose.
    GAUGE.update({"kv": 0.10, "wait": 3})
    time.sleep(0.3)
    g = gstate()
    ck("GATE_WAIT=1 makes the engine queue close the gate by itself, reading num_requests_waiting{ rather than "
       "a *_by_reason counter", g["closed"] and "waiting 3" in g["why"], g["why"][:160])
    GAUGE["wait"] = 0

    # what must never be held: /health is what setup.py's watcher polls, /metrics is what the gate reads itself,
    # /api/* is the Monitor, and the sidecar routes are CPU serves that cannot take a KV hit.
    GAUGE["kv"] = 0.99
    time.sleep(0.3)
    slow = {}
    for pth in ("/health", "/metrics", "/v1/models", "/api/gate", "/v1/embeddings", "/v1/rerank"):
        t0 = time.time()
        http(GP, pth, timeout=10)
        slow[pth] = time.time() - t0
    ck("with the gate shut, /health /metrics /v1/models /api/gate and both sidecar routes answer immediately",
       all(v < 0.3 for v in slow.values()), {k: f"{v*1000:.0f} ms" for k, v in slow.items()})
    p0 = gstate()["polls"]
    time.sleep(0.5)
    ck("a closed gate does not starve its own reading (the poller keeps running while nothing is admitted)",
       gstate()["polls"] > p0, f"polls {p0} -> {gstate()['polls']} while shut")
    # the strip the Monitor draws is the poller's own record, so it has to say which polls were shut -- and a
    # poll whose gauges were unusable draws OPEN rather than blank, because a gap would blame the traffic for a
    # telemetry fault (the page cannot tell "open" from "we stopped looking" any other way).
    h = gstate()["hist"]
    ck("the history strip is one entry per poll and says which of them were shut",
       2 <= len(h) <= gstate()["hist_n"] and sum(h) >= 1 and h == h[-len(h):],
       f"{len(h)} polls, {sum(h)} shut: {h[-12:]}")
    _g = gstate()
    ck("and the ring holds GATE_HIST=900 polls rather than the 180 that could only speak about the last three "
       "minutes, while the payload states the span the ring ACTUALLY covers at this poll rate -- what an afternoon "
       "of traffic leaves behind is the question 'was the door shut at 15:05?', and a strip without a span cannot "
       "answer it (notes/admission-gate.md §6)",
       _g["hist_n"] >= 600 and abs(_g.get("hist_span_s", 0) - _g["hist_n"] * float(GENV["GATE_METRICS_S"])) < 1,
       f"hist_n {_g['hist_n']}, span {_g.get('hist_span_s')} s, poll {GENV['GATE_METRICS_S']} s")
    _gsrc = open(os.path.join(ROOT, "serve/gateway.py")).read()
    ck("as SHIPPED that ring is 15 minutes: 900 polls at the shipped 1 s poll interval",
       'GATE_HIST", "900"' in _gsrc and 'GATE_METRICS_S", "1.0"' in _gsrc,
       [l.strip()[:70] for l in _gsrc.splitlines() if "GATE_HIST" in l or "GATE_METRICS_S\"" in l][:3])

    # the ceiling is a promise: a client never waits longer than GATE_HOLD_S, and the engine still gets the
    # request -- a gate that trades a 30-minute TTFT for a hung connection has not improved anything.
    t0 = time.time()
    st, b = http(GP, "/v1/chat/completions", "POST", BIG, timeout=20)
    waited = time.time() - t0
    g = gstate()
    ck("the hold ceiling holds: passed after ~GATE_HOLD_S with the ENGINE's answer, and counted as expired",
       1.4 <= waited <= 2.6 and json.loads(b)["role"] == "engine" and g["counters"]["expired"] >= 1,
       f"waited {waited:.2f} s, {st}, {g['counters']}")

    # GATE_HOLD_S=0 is the other policy: refuse, and name what was refused for. It must be the GATE answering,
    # not the engine -- the request never reaches it.
    GF, gfail = bring({"GATE_HOLD_S": "0"})
    GAUGE["kv"] = 0.99
    time.sleep(0.3)
    n_before = sum(1 for m, p, _n, _e in LOG["engine"] if p == "/v1/chat/completions")
    st, b = http(GF, "/v1/chat/completions", "POST", BIG, timeout=10)
    d = json.loads(b)
    n_after = sum(1 for m, p, _n, _e in LOG["engine"] if p == "/v1/chat/completions")
    ck("GATE_HOLD_S=0 refuses with a 503 that names the pressure, and the request never reaches the engine",
       "503" in st and d["error"]["type"] == "gateway_admission" and "kv 0.99" in d["error"]["gate"]["why"]
       and n_after == n_before, f"{st} {str(d)[:140]}")
    gfail.terminate()
    gfail.wait(timeout=5)

    # fail-open, twice over: a /metrics with no gauges in it (a rename, another server), and a gauge older than
    # GATE_STALE_S. Either way the gate must NOT be the thing that stops this box serving.
    JUNK = free_port()
    METRICS[JUNK] = b"# nothing of ours here\n"
    upstream("engine", JUNK)
    GJ, gjunk = bring({"ENGINE": f"http://127.0.0.1:{JUNK}"})
    GAUGE["kv"] = 0.99
    t0 = time.time()
    st, b = http(GJ, "/v1/chat/completions", "POST", '{"x":1}', timeout=10)
    g = json.loads(http(GJ, "/api/gate", timeout=5)[1])
    ck("an engine whose /metrics carries no gauges opens the gate and says so (a rename is not an outage)",
       time.time() - t0 < 0.4 and json.loads(b)["role"] == "engine" and not g["closed"]
       and "kv_cache_usage_perc" in (g["err"] or ""), f"{st} {str(g)[:160]}")
    # and the strip shows it: those polls read OPEN, not blank. A reader who sees a gap cannot tell "the pool
    # was free" from "we stopped being able to look", and the second one is the finding.
    ck("a poll with unusable gauges draws open in the history strip rather than a gap",
       g["hist"] and not any(g["hist"]), f"{len(g['hist'])} polls: {g['hist'][-10:]}")
    gjunk.terminate()
    gjunk.wait(timeout=5)
    # a /metrics that stops answering must not freeze the gate shut on the last number it saw: the gauge goes
    # stale, the gate opens, and the request path is untouched while one poller sits inside a hung read.
    HANG = free_port()
    upstream("engine", HANG)
    GH, ghang = bring({"ENGINE": f"http://127.0.0.1:{HANG}", "GATE_STALE_S": "1", "GATE_HOLD_S": "10"})
    GAUGE["kv"] = 0.99
    time.sleep(0.4)
    shut = gstate(GH)["closed"]
    SLOW_METRICS.add(HANG)
    time.sleep(1.4)
    t0 = time.time()
    st, b = http(GH, "/v1/chat/completions", "POST", BIG, timeout=20)
    waited, g = time.time() - t0, gstate(GH)
    SLOW_METRICS.discard(HANG)
    GAUGE["kv"] = 0.10
    ck("a /metrics that stops answering OPENS the gate (stale gauge) rather than holding traffic shut on it",
       shut and waited < 1.0 and json.loads(b)["role"] == "engine" and not g["closed"],
       f"shut first: {shut}; waited {waited:.2f}s; {str(g)[:140]}")
    ghang.terminate()
    ghang.wait(timeout=5)

    # and the kill switch, on a gateway that really was configured to gate.
    GD, gdis = bring({"GATE_DISABLE": "1"})
    GAUGE["kv"] = 0.99
    time.sleep(0.3)
    t0 = time.time()
    st, b = http(GD, "/v1/chat/completions", "POST", '{"x":1}', timeout=10)
    g = json.loads(http(GD, "/api/gate", timeout=5)[1])
    ck("GATE_DISABLE=1 is a real switch: shut pool, instant chat, gate reports itself off -- and it is NOT reading "
       "the engine either, which is the difference between that switch and GATE_POLICY=observe (off removes the "
       "measurement too)",
       time.time() - t0 < 0.35 and json.loads(b)["role"] == "engine" and g["closed"] is False
       and g["why"] == "off" and g["polls"] == 0 and not g["hist"], f"{time.time()-t0:.2f}s {str(g)[:120]}")
    gdis.terminate()
    gdis.wait(timeout=5)

    # ---- what is shipped now: GATE_POLICY=observe, measured and not acted on -------------------------------
    # 2026-11-16, the second afternoon. Four requests with no max_tokens (prompt ~520 each, so vLLM's default
    # commitment was the rest of the 262,144-token window) filled a 902,153-token pool: the engine queue went to
    # 66-72 all on reason="capacity", aggregate decode fell to 8.9-51.7 tok/s, and num_preemptions_total ran 165 ->
    # 296 in ten minutes. Against that, the armed door could not re-open (a preemption every 20 s re-arms the
    # window, and kv p50 0.945 sits above GATE_KV_LOW=0.85), and its hold could not take one block back from the
    # rows already running: measured 5.019 s through the gate for a 20 KB body against 0.002 s direct to the
    # engine, with 25 of 28 holds ending at the ceiling. So the intervention is off and the measurement stays --
    # these are the assertions that mode needs.
    GO, gobs = bring({"GATE_KV_HIGH": "0.85", "GATE_KV_LOW": "0.78", "GATE_WAIT": "1"}, base=UNARMED)
    GAUGE.update({"kv": 0.95, "run": 5, "wait": 3, "prem": 2})
    time.sleep(0.4)
    g = gstate(GO)
    hist_seen = list(g["hist"])
    ck("as shipped the gate is OBSERVE, and the harm signal is still computed and still named: the report did not "
       "leave with the intervention",
       g["policy"] == "observe" and g["config"]["GATE_POLICY"] == "observe" and g["closed"]
       and "kv 0.950" in g["why"], g["why"][:140])
    t0 = time.time()
    st, b = http(GO, "/v1/chat/completions", "POST", BIG, timeout=10)     # a body the armed door would hold
    seen, g = time.time() - t0, gstate(GO)
    ck("and a big-body request met by the harm signal goes through with the ENGINE's answer and no delay: the "
       "5.019 s the armed door measured on this box is exactly what was removed",
       seen < 0.4 and json.loads(b)["role"] == "engine" and g["closed"], f"waited {seen:.3f} s, {st}")
    cn = g["counters"]
    ck("nothing is held, refused or expired while observing, and the arrivals a policy would have acted on are "
       "counted instead -- that is how 'what would this have cost?' is answered without running it on a live box",
       cn["held"] == 0 and cn["held_now"] == 0 and cn["expired"] == 0 and cn["refused"] == 0
       and cn["arrived_shut"] >= 1 and cn["arrived_shut_big"] >= 1 and cn["gated_seen"] >= 1
       and cn["declared_big"] >= 1, str(cn))
    ck("the door states its own duty cycle -- fraction of the ring shut, the run it is inside, the longest run -- "
       "because 900 strip dots cannot say 'effectively permanently shut', and on 2026-11-16 that was the finding",
       g["shut"]["frac"] is not None and g["shut"]["frac"] > 0 and g["shut"]["run_s"] >= 0
       and g["shut"]["max_s"] >= g["shut"]["run_s"], str(g["shut"]))
    ck("and the reading keeps running while it decides nothing: the poller and the ring are not the intervention",
       g["polls"] > 0 and len(hist_seen) >= 2 and sum(hist_seen) >= 1, f"{g['polls']} polls")
    gobs.terminate()
    gobs.wait(timeout=5)
    # A knob whose value is a typo is a knob that is off, and off has to mean the shipped mode rather than a
    # surprise -- the fail-open direction of this whole file is the request path.
    GB, gbad = bring({"GATE_POLICY": "halt"}, base=UNARMED)
    GAUGE["kv"] = 0.99
    time.sleep(0.4)
    t0 = time.time()
    st, b = http(GB, "/v1/chat/completions", "POST", BIG, timeout=10)
    g = gstate(GB)
    ck("GATE_POLICY spelled wrong observes and SAYS SO, rather than silently holding",
       time.time() - t0 < 0.5 and json.loads(b)["role"] == "engine" and g["policy"] == "observe"
       and "halt" in (g["policy_err"] or ""), f"{time.time()-t0:.2f}s {str(g)[:160]}")
    gbad.terminate()
    gbad.wait(timeout=5)
    _gmode = open(os.path.join(ROOT, "serve/gateway.py")).read()
    ck("the shipped default in the source is observe, and the launcher can name the knob (an unnamed knob is a knob "
       "the tmux launch cannot set)",
       'GATE_POLICY", "observe"' in _gmode
       and "GATE_POLICY" in open(os.path.join(ROOT, "serve/gateway.sh")).read(),
       [l.strip()[:64] for l in _gmode.splitlines() if "GATE_POLICY" in l][:2])
    # ---- the shipped THRESHOLDS, on an armed door (GATE_POLICY=hold) ----------------------------------------
    # 2026-11-15, read off the live box: a batch of ~100k-token data-processing prompts parked the pool at
    # 0.87-0.90 with 17 requests waiting on scheduling capacity, and num_preemptions_total moved ONCE all day.
    # v1 read exactly that as "shut the door": 180 of 180 polls closed, 14 requests held, 9 of them admitted
    # anyway after waiting the whole 25 s ceiling. The batch was not harming anything -- the KV was already held,
    # and the engine was queueing its own work correctly -- so the door was pure latency on every new arrival,
    # and the cheap small requests were the ones made to wait. These are the four assertions that policy needs.
    GD2, gdef = bring({}, base=DEFAULTS)
    GAUGE.update({"kv": 0.30, "run": 5, "wait": 0, "prem": 0})
    time.sleep(0.4)                       # start open: the pool's latch would otherwise carry the last test's 0.99
    GAUGE.update({"kv": 0.88, "wait": 17})
    time.sleep(0.4)
    t0 = time.time()
    st, b = http(GD2, "/v1/chat/completions", "POST", BIG, timeout=10)
    busy, g = time.time() - t0, gstate(GD2)
    ck("a full pool and a 17-deep engine queue and NO preemption leave the door open: the engine queueing its "
       "own work is the engine's job, and a second queue in front of it changes nothing about the KV already held",
       not g["closed"] and busy < 0.6 and json.loads(b)["role"] == "engine", f"{g['why'][:120]} after {busy:.2f} s")

    # A preemption counter that MOVED is the harm: blocks taken back from a running request, which is what the
    # 11-14 slow window's 68 victims paid a whole re-prefill for. It is cumulative since engine start, so the
    # gate compares polls; a jump covers a poll it missed.
    GAUGE["prem"] = 3
    time.sleep(0.5)
    g = gstate(GD2)
    t0 = time.time()
    st, b = http(GD2, "/v1/chat/completions", "POST", BIG, timeout=15)
    held = time.time() - t0
    ck("preemption activity closes the door, and the reason names the count and the window rather than a level",
       g["closed"] and "preemption" in g["why"], g["why"][:160])
    ck("the shipped hold ceiling is 5 s, not v1's 25 -- a bound that is usually reached is a delay, not a bound",
       5.0 <= held <= 7.0 and json.loads(b)["role"] == "engine" and gstate(GD2)["counters"]["expired"] >= 1,
       f"waited {held:.2f} s: {gstate(GD2)['counters']}")

    # ...and the window is a WINDOW. A pool threshold (re-open below 0.78) never happens under a long batch, which
    # is how v1 ended up shut forever; a quiet-preemptions window ends on its own.
    GW2, gwin = bring({"GATE_PREEMPT_S": "1.2"}, base=DEFAULTS)
    GAUGE["prem"] = 4
    time.sleep(0.4)                       # pool is 0.88 here: below HIGH 0.92, so only the preemption axis is shut
    shut = gstate(GW2)["closed"]
    time.sleep(1.6)
    back = gstate(GW2)
    ck("the preemption shutdown expires: after GATE_PREEMPT_S with nothing preempting, the door re-opens",
       shut and not back["closed"], f"shut={shut} -> {back['closed']}: {back['why'][:100]}")

    # The ones v1 starved: a 200-token question behind a shut door. It passes on its declared size, header only --
    # the pipe still never reads a body to decide anything.
    GAUGE.update({"kv": 0.97, "prem": 5})
    time.sleep(0.4)
    shut = gstate(GW2)["closed"]
    t0 = time.time()
    st, b = http(GW2, "/v1/chat/completions", "POST", '{"x":1}', timeout=10)
    small, g = time.time() - t0, gstate(GW2)
    ck("a request declaring a small body passes a shut door, counted, while a big one would be held",
       shut and small < 0.4 and g["counters"]["small_passed"] >= 1 and g["config"]["GATE_SMALL_BYTES"] == 8192,
       f"door shut={shut}; small waited {small:.2f} s; {g['counters']}")

    # Never a second queue: past GATE_MAX_HELD the gate admits instead of holding, because a gate with an
    # unbounded hold queue has just become a queue with extra steps -- and the engine already has one.
    GM, gheld = bring({"GATE_MAX_HELD": "1", "GATE_HOLD_S": "3"}, base=DEFAULTS)
    GAUGE.update({"kv": 0.99})
    time.sleep(0.4)
    t1 = threading.Thread(target=lambda: http(GM, "/v1/chat/completions", "POST", BIG, timeout=15))
    t1.start()
    time.sleep(0.4)                                                             # one request is now being held
    t0 = time.time()
    st, b = http(GM, "/v1/chat/completions", "POST", BIG, timeout=15)
    second = time.time() - t0
    t1.join(timeout=10)                                                         # the first leaves at its ceiling
    g = gstate(GM)
    ck("at most GATE_MAX_HELD are ever held: the second arrival is admitted instead of queued behind it",
       second < 1.0 and g["counters"]["held_over"] >= 1 and g["counters"]["expired"] >= 1,
       f"second waited {second:.2f} s: {g['counters']}")
    for pr in (gdef, gwin):
        pr.terminate()
        pr.wait(timeout=5)
    gheld.terminate()
    gheld.wait(timeout=5)
    GAUGE.update({"kv": 0.10, "run": 1, "wait": 0, "prem": 0})
    gg = gstate()
    ggate.terminate()
    ggate.wait(timeout=5)
    ck("the gate's own page carries its config and counters, so the Monitor can say why a request was late",
       gg["config"]["GATE_KV_HIGH"] == 0.85 and gg["gated"] == ["/v1/chat/completions", "/v1/completions"]
       and gg["config"]["pool_tokens"] == 902153, str(gg)[:180])
    METRICS.pop(JUNK, None)
finally:
    gw.terminate()
    gw.wait(timeout=5)

print("PASS" if fail == 0 else "FAIL", f"({fail} failing)")
if fail:                                       # a framing bug shows up as an exception in the pump, not as a
    print("--- the gateway's own stderr ---")  # nonzero exit, so its log is the shortest path to the cause
    GWLOG.flush()
    print(open(GWLOG.name).read()[-2500:])
sys.exit(0 if fail == 0 else 1)
