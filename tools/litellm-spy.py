#!/usr/bin/env python3
"""tools/litellm-spy.py - what LiteLLM actually puts on the wire to the upstream, measured not remembered.

Everyone reaches for source IP when asking "who sent this request", and behind a LiteLLM proxy that answer is one
address for every caller. The headers litellm forwards are a version-specific list nobody has read, so this tool
reads it: it listens on a port, records every request a litellm proxy sends it, and forwards the body verbatim to
the front door.

    # litellm config points its api_base at the spy instead of at the gateway:
    #   api_base: http://127.0.0.1:4010/v1
    python3 tools/litellm-spy.py --port 4010 --upstream http://127.0.0.1:8080
    ... send traffic through litellm ...
    python3 tools/litellm-spy.py --dump /tmp/litellm-spy.jsonl

It PARSES BODIES, and that is allowed only because this is a test stand: the question is what is inside the
request. The production gateway (serve/gateway.py) is a byte pipe and stays one; see
[notes/who-sent-it.md](../notes/who-sent-it.md) for what this tool proved on 2026-11-16 and for what the gateway
may therefore key on (a header it already receives) instead of on anything inside the body.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {}


class Spy(BaseHTTPRequestHandler):
    def _say(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _record(self, rec):
        with open(STATE["log"], "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def do_GET(self):
        self._record({"method": "GET", "path": self.path, "headers": dict(self.headers.items())})
        try:
            with urllib.request.urlopen(STATE["upstream"] + self.path, timeout=60) as r:
                self._say(r.status, r.read())
        except urllib.error.HTTPError as e:
            self._say(e.code, e.read())

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        rec = {"method": "POST", "path": self.path, "headers": dict(self.headers.items()), "body_bytes": n}
        try:
            j = json.loads(body)
            rec["body_keys"] = sorted(j.keys())
            for k in ("model", "user", "metadata", "extra_body", "stream"):
                if k in j:
                    rec["body_" + k] = j[k]
        except ValueError:
            rec["body_unparsed"] = True                       # a body that is not JSON is still a fact
        self._record(rec)
        req = urllib.request.Request(STATE["upstream"] + self.path, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                self._say(r.status, r.read())
        except urllib.error.HTTPError as e:
            self._say(e.code, e.read())

    def log_message(self, *a):                                        # no stderr spam per request
        pass


def dump(path):
    """Print what was recorded, request by request: the point is the header set and the body key set."""
    seen = 0
    for ln in open(path, encoding="utf-8", errors="replace"):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("method") != "POST":
            continue
        seen += 1
        print(f"\n=== request {seen}: {r['path']} ({r.get('body_bytes', 0)} B) ===")
        print("body keys:", r.get("body_keys"), "| body user:", r.get("body_user"),
              "| model:", r.get("body_model"))
        for k, v in sorted((r.get("headers") or {}).items()):
            print(f"  {k}: {v[:90]}")
    if not seen:
        print(f"{path}: no POSTs recorded -- litellm is not pointed at this spy, or the spy died on a stale "
              f"log handle (a deleted log file keeps its inode: restart the spy after deleting it)")
    return seen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=4010)
    ap.add_argument("--upstream", default="http://127.0.0.1:8080",
                    help="where to forward (the front door, so the answer is a real answer)")
    ap.add_argument("--log", default="/tmp/litellm-spy.jsonl")
    ap.add_argument("--dump", metavar="LOG", help="print a recorded log and exit")
    a = ap.parse_args(argv)
    if a.dump:
        return 0 if dump(a.dump) else 1
    STATE.update(upstream=a.upstream.rstrip("/"), log=a.log)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Spy)
    print(f"spy on 127.0.0.1:{a.port} -> {STATE['upstream']}   log {a.log}", flush=True)
    print("point the litellm model entry's api_base at http://127.0.0.1:%d/v1, send traffic, then "
          "re-run with --dump %s" % (a.port, a.log), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
