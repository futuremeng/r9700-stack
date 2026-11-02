#!/usr/bin/env python3
"""Put the Monitor's cards under a long-prefill backlog and keep the poll that shows it.

    python3 bench/capture-cards.py [N=8] [TOKENS=30000] [BASE=http://127.0.0.1:8080] [OUT=docs/r9700-stack-monitor-cards.svg]

Why this exists: docs/monitor-cards.py draws the five headline cards from one GET /api/status, and the values worth
drawing only exist under load. Polling by hand picks a sample by eye and cannot be repeated; the run that produced
the figure in the README was exactly that, and a reboot cut it off mid-flight. So the sampler here is the record:
it fires N concurrent long prompts, polls /api/status once a second while they run, keeps the whole payload of the
poll it selects, and hands THAT file to docs/monitor-cards.py. The figure is therefore one measured poll, not a
mixture of several -- which matters most on the Requests card, whose three numbers (running / waiting / waiting by
reason) only mean something as a set.

Which poll gets kept: the one with the longest waiting queue, tie-broken by requests actually running. That is the
shape the card exists to show -- on this launch the queue is long while the KV pool is nearly empty, so "waiting"
without the reason split would read as a memory wall that is not there (notes/context-262k.md).

Prompt length is measured, not assumed: one cheap calibration prompt decides the word count, and what each request
actually cost is printed from the serve's own usage.prompt_tokens. A ~0 token prompt would show a queue too, but
not one that stays queued.
"""
import json
import random
import subprocess
import sys
import threading
import time
import urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
BASE = (sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080").rstrip("/")
OUT = sys.argv[4] if len(sys.argv) > 4 else "docs/r9700-stack-monitor-cards.svg"
# The kept poll lives next to the figure it produced, in the repo: "drawn from a live poll" is only checkable if
# the poll itself is checkable. Re-running the figure is then `python3 docs/monitor-cards.py
# docs/r9700-stack-monitor-cards.json`, with no serve in the loop.
SAMPLE = sys.argv[5] if len(sys.argv) > 5 else "docs/r9700-stack-monitor-cards.json"
MAXOUT = 16                                   # the load is prefill; keep decode out of the way
CALIB_FILLERS = 1500                          # one short prompt, to measure this filler on this tokenizer
FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        return json.load(r)


def post(prompt):
    body = json.dumps({"model": "Qwen3.8", "max_tokens": MAXOUT, "stream": False,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)


def status():
    return get("/api/status")


stop = threading.Event()
polls = []                                    # [(t, payload)] -- payloads, not fields, so a whole poll survives


def spy():
    while not stop.is_set():
        try:
            polls.append((time.time(), status()))
        except Exception as e:                # a serve that dies mid-run must not lose the polls already kept
            print(f"poll failed: {e}", file=sys.stderr)
        time.sleep(1.0)


def worker(i, out, fillers):
    rng = random.Random(1000 + i)             # a different haystack per request: one shared prefix is one cache
    parts = []
    for k in range(fillers):
        if k % 40 == 0:
            parts.append(f"nonce {rng.randrange(1 << 30)}: ")
        parts.append(FILLER)
    t0 = time.time()
    try:
        j = post("".join(parts))
        u = j.get("usage") or {}
        out[i] = (u.get("prompt_tokens"), u.get("completion_tokens"), round(time.time() - t0, 2), None)
    except Exception as e:                    # a refused or truncated prompt is a fact about the run, not a gap
        out[i] = (None, None, round(time.time() - t0, 2), f"{type(e).__name__}: {e}")


base0 = status()
print(f"{BASE} reachable: mode {base0.get('mode')}, "
      f"running {base0['live'].get('running')}, waiting {base0['live'].get('waiting')}, "
      f"pool {base0['live'].get('kv_used_pct')}%")

# Size the prompts by measuring, not by a word count: this filler costs ~24.3 prompt_tokens a repeat on this
# checkpoint (measured twice: 4,893 at 200 repeats, 36,429 at 1,500), and a script that counts words instead lands
# 3x off the length it says it ran. One cheap prompt first, and its own usage decides the rest.
calib = {}
worker(-1, calib, CALIB_FILLERS)
tpf = (calib[-1][0] or 24.3) / CALIB_FILLERS
if calib[-1][3]:
    print(f"calibration failed ({calib[-1][3]}); sizing at a guessed 24.3 tokens per filler repeat", file=sys.stderr)
else:
    print(f"calibration: {CALIB_FILLERS} repeats of the filler = {calib[-1][0]} prompt_tokens "
          f"({tpf:.2f} tokens/repeat) -> {int(TOKENS / tpf):,} repeats for ~{TOKENS:,} tokens")
fillers = max(1, int(TOKENS / tpf))

results = {i: None for i in range(N)}          # keyed up front: an unfinished request is a None, not a missing key
thr = threading.Thread(target=spy, daemon=True)
thr.start()
t0 = time.time()
for i in range(N):
    threading.Thread(target=worker, args=(i, results, fillers), daemon=True).start()

print("t(s)  run wait cap def  kv%  kv_tok   gen tok/s  prompt tok/s")
printed, wall = 0.0, time.time()
while any(v is None for v in results.values()):
    time.sleep(1.0)
    if polls and polls[-1][1]["at"] > printed:                # one line per distinct poll
        printed = polls[-1][1]["at"]
        l, r = polls[-1][1]["live"], polls[-1][1]["rate"]
        print(f"{printed - t0:4.0f} {l['running']:4.0f} {l['waiting']:3.0f} "
              f"{(l.get('waiting_capacity') or 0):3.0f} {(l.get('waiting_deferred') or 0):3.0f} "
              f"{l['kv_used_pct']:5.1f} {l['kv_tokens']:7.0f} {r.get('generation_tok_s') or 0:9.1f} "
              f"{r.get('prompt_tok_s') or 0:12.1f}")
    if time.time() - wall > 1800:                            # a wedged serve must not hang the sampler forever
        print("giving up after 1800 s with some requests still outstanding", file=sys.stderr)
        break

stop.set()
thr.join(timeout=5)

# The poll with the longest queue: waiting first, because that is the number the card is being drawn to show;
# running second, so a tie goes to the poll where work was actually in flight rather than one already draining.
def key(p):
    l = p[1]["live"]
    return (l.get("waiting") or 0, l.get("running") or 0, l.get("kv_used_pct") or 0)


best_at, best = max(polls, key=key)
kept = polls.index((best_at, best))
path = SAMPLE
with open(path, "w") as f:
    json.dump(best, f, indent=1)

l, r = best["live"], best["rate"]
print(f"\nkept poll {kept + 1} of {len(polls)}: running {l['running']:.0f} waiting {l['waiting']:.0f} "
      f"capacity {l.get('waiting_capacity')} deferred {l.get('waiting_deferred')} "
      f"kv {l['kv_used_pct']:.1f}% ({l['kv_tokens']:.0f} tokens) -> {path}")
print("requests (prompt_tokens, completion, wall s, error):")
for i in sorted(results):
    print("  ", results[i])
print(f"the kept poll is the one with the longest waiting queue (tie: more running, then more of the pool); "
      f"engine clock {time.strftime('%H:%M:%S UTC', time.gmtime(best_at))} -- cross-check the serve's own log line")

subprocess.run([sys.executable, "docs/monitor-cards.py", path, OUT, BASE], check=True)
print("\npaste into notes/context-262k.md:")
print(f"  {N} x ~{TOKENS:,}-token prompts through {BASE} (mode {best['mode']}, "
      f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(best_at))}): "
      f"running {l['running']:.0f} / waiting {l['waiting']:.0f} / capacity {l.get('waiting_capacity'):.0f} / "
      f"deferred {l.get('waiting_deferred'):.0f} at KV {l['kv_used_pct']:.1f}% of a {l['kv_tokens']:.0f}-token "
      f"pool; the serve's own prompt_tokens were "
      f"{sorted(set(v[0] for v in results.values() if v and v[0]))}")
