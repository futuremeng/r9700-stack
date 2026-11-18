#!/usr/bin/env python3
"""Inter-token latency via streaming at concurrency 1/2/4 (median gap between streamed chunks, ms).

ITL_BASE overrides the endpoint: on 2026-11-15 the box grew a gateway in front of the engine (port 8080 ->
127.0.0.1:8180), and the only way to show the hop costs nothing is to run the same bench down both paths."""
import json, os, statistics, sys, threading, time, urllib.request
BASE = os.environ.get("ITL_BASE", "http://localhost:8080/v1/chat/completions")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from served_model import resolve          # the model name comes from the serve, never from this file
MODEL = resolve(BASE)
def stream(i, out):
    body = {"model": MODEL, "messages": [{"role": "user", "content": f"Topic {i}: write a long essay about rivers."}],
            "max_tokens": 150, "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    ts = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data: {"):
                ts.append(time.time())
    out.append([b - a for a, b in zip(ts, ts[1:])][5:])
for conc in [int(c) for c in (sys.argv[1:] or ["1", "2", "4"])]:
    res = []
    th = [threading.Thread(target=stream, args=(i, res)) for i in range(conc)]
    [t.start() for t in th]; [t.join() for t in th]
    gaps = [g for r in res for g in r]
    print(f"conc {conc}: median ITL {1000*statistics.median(gaps):.1f} ms, p90 {1000*sorted(gaps)[int(.9*len(gaps))]:.1f} ms")
