#!/usr/bin/env python3
"""tests/test_gateway.py - the framing of the thing that fronts every port on this box.

serve/gateway.py is deliberately not a web framework: it reads the request line, opens the upstream, and pipes
bytes both ways. That is why it exists (serve/webui.py's proxy costs 2-3 ms a request because it parses the
body and rebuilds it; the gateway measured +0.8 ms, notes/embed-sidecar-box-test.md) and it is also the whole
risk: every bug in it is an HTTP framing bug, which a client sees as a hang or a truncated reply, not as a
nonzero exit. So this file tests framing, not features:

  (1) routing, including the /api/sidecars the gateway answers itself, and a path that merely STARTS with a
      routed prefix (/v1/embeddings-x) taking the default instead.
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
       "SIDECAR_TTL": "0"}
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
finally:
    gw.terminate()
    gw.wait(timeout=5)

print("PASS" if fail == 0 else "FAIL", f"({fail} failing)")
if fail:                                       # a framing bug shows up as an exception in the pump, not as a
    print("--- the gateway's own stderr ---")  # nonzero exit, so its log is the shortest path to the cause
    GWLOG.flush()
    print(open(GWLOG.name).read()[-2500:])
sys.exit(0 if fail == 0 else 1)
