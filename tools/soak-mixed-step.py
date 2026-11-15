#!/usr/bin/env python3
"""tools/soak-mixed-step.py - drive the step shape that faulted, until it faults or until it doesn't.

Why this exists. Two device faults, 2026-11-14 20:59 and 2026-11-15 05:11:18, share one signature and share
nothing else that was measured: a step that mixes **new prefills with spec-decode rows**, on card 0000:c3:00.0, a
read fault (`PERMISSION_FAULTS: 0x3`, `RW: 0x0`, `MAPPING_ERROR: 0x0`) at a page-aligned address, and two
DIFFERENT named kernels (`_causal_conv1d_fwd_kernel` at grid [9856, 10, 1]; `at::native::CatArrayBatchedCopy_contig`
at grid [13312, 3, 1]). `tests/test_conv1d_mixed_step.py` bounds the arithmetic of both steps (arms A and E) and
replays their ops on one card (arms B and E): all of it is IN RANGE. What no static check can reach is an index
that is in range and no longer names live memory -- so the next answer comes from running the shape for a long time
against a real launch with `R9K_GDN_DEBUG=1` on, which is cheap now that the debug block is layer 0 only
(gdn.py `_dbg_here`, notes/bug-causal-conv1d-oob.md §8.2).

What it drives, taken from the engine's own two dumps rather than invented:

  the 05:11 step   23 spec rows of 3 tokens + TWO new prefills (160, 156 tokens), total 385 tokens, kv 0.591,
                   waiting 0. Both new requests had `num_computed_tokens=0`, i.e. no prefix hit -- which is what
                   puts 316 prefill tokens into one step, above R9K_GDN_PREFILL_MAX=256, i.e. on stock's path.
  the 11-14 step   30 spec rows + ONE new prefill of 610 tokens, kv 0.998, 13 waiting.

So: a standing population of spec-decoding requests (`--pop`, long outputs with `ignore_eos` so they really do
occupy decode rows for minutes), a PAIR of short arrivals every `--period` seconds (the 05:11 shape), a single
long arrival every `--long-period` seconds (the 11-14 shape), and `--deep` long-context requests to move
`kv_cache_usage` toward the 0.998 case when you want that arm. Prefix hits are what usually keep prefills out of a
step, so each arrival carries a fresh, unique filler -- otherwise the prefix cache swallows the prefill and the step
stops being mixed, which is the one thing the experiment needs.

It goes to the ENGINE port by default, not the gateway: the admission gate (serve/gateway.py,
notes/admission-gate.md) is supposed to stop exactly this regime, so leaving it in the path would measure the gate.
Point `--base` at the gateway when the gate is the thing under test.

It exits the moment a request fails or /health stops answering, and prints the wall moment so the engine log can be
read against it. Run it with the watcher armed (`WATCH_RESTART=1`) and a hit is survivable: the breaker brings the
launch back and the log keeps the fault line (notes/launch-methodology.md §4).

  ~/venv-r9700/bin/python tools/soak-mixed-step.py --minutes 45 --pop 24 --deep 0     # the low-kv arm (the 05:11 regime)
  ~/venv-r9700/bin/python tools/soak-mixed-step.py --minutes 30 --pop 28 --deep 8     # the high-kv arm (the 11-14 regime)
  --dry      build the prompts and print the plan, send nothing
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

fail = 0


def log(*a):
    print(f"{time.strftime('%H:%M:%S')}", *a, flush=True)


def post(base, path, payload, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json",
                                          # an API-key-protected serve still needs the header; this box's engine
                                          # binds loopback with no key, and sending an empty one is a way to ask
                                          # without learning the operator's secret
                                          "authorization": "Bearer " + os.environ.get("API_KEY", "none")})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gauge(base):
    """(kv_usage, running, waiting) from the engine's own /metrics -- the regime the run has to be in."""
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=5) as r:
            txt = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return None, None, None
    out = {}
    for line in txt.split("\n"):
        # the braces are load-bearing on the two counts: without the brace, `vllm:num_requests_waiting` also matches
        # `vllm:num_requests_waiting_by_reason`, which is a scheduler choosing to wait and not a queue growing
        # (the same lesson serve/gateway.py's admission gate learned the same day). The usage gauge carries
        # `{engine="0",model_name=...}` directly after the name, so it is matched without a trailing space.
        for name, key in (("vllm:kv_cache_usage_perc", "kv"), ("vllm:num_requests_running{", "run"),
                          ("vllm:num_requests_waiting{", "wait")):
            if line.startswith(name):
                out.setdefault(key, []).append(float(line.rsplit(" ", 1)[1]))
    return (max(out["kv"]) if out.get("kv") else None, sum(out["run"]) if out.get("run") else None,
            sum(out["wait"]) if out.get("wait") else None)


class Prompt:
    """A prompt of about `tokens` tokens, and unique every time.

    Uniqueness is not decoration: prefix caching is what normally keeps a prefill OUT of a mixed step (the 05:11
    dump's two new requests both had num_computed_tokens=0 -- no hit -- and that is exactly why 316 prefill tokens
    landed in one step). Repeating the same filler would make the engine read the prompt from cache and the step
    would stop being mixed. Calibrated against the serve's own token accounting, not guessed at 4 chars/token."""

    def __init__(self, base, model):
        self.base, self.model, self.per_word = base, model, 1.35
        # Calibrate on the filler this class actually emits, measured by the serve's own token accounting
        # (/v1/tokenize is not served by this build -- 404 on 2026-11-15). One probe request: build 400 words of the
        # real filler, read usage.prompt_tokens, and the rate follows. It carries the ChatML template's ~20 tokens
        # inside it, which at this size is a few percent -- the dumps' steps were 156/160/610 tokens, and what the
        # experiment needs is the REGIME (a couple of hundred prefill tokens landing at once, above PREFILL_MAX=256),
        # not an exact count.
        try:
            n = self._ask(self.make(400, "calib"))
            if n and n > 40:
                self.per_word = (n - 20) / (400 / self.per_word)
        except Exception as e:                                                  # noqa: BLE001
            log(f"  note: calibration request failed ({type(e).__name__}); assuming {self.per_word} tokens per word")

    def _ask(self, content):
        r = post(self.base, "/v1/chat/completions",
                 {"model": self.model, "messages": [{"role": "user", "content": content}],
                  "max_tokens": 1, "temperature": 0}, timeout=120)
        return (r.get("usage") or {}).get("prompt_tokens")

    def make(self, tokens, tag):
        words = max(1, int(tokens / self.per_word))
        # one word per iteration, and every word distinct: a repeated filler tokenizes at a predictable rate AND
        # gives no two arrivals a shared prefix (see the class docstring on why a shared prefix kills the mixed step)
        salt = f"{tag}-{time.time_ns()}-{random.randrange(1 << 30)}"
        body = " ".join(f"t{i}" for i in range(words))
        return f"{salt}\n{body}\nContinue the list above at length."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8180",
                    help="engine port by default: the gateway's admission gate deliberately holds this regime")
    ap.add_argument("--model", default="Qwen3.8")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--pop", type=int, default=24, help="standing spec-decoding requests (the dumps had 23 and 30)")
    ap.add_argument("--short", type=int, default=165, help="the paired arrival's prompt tokens (dump: 160, 156)")
    ap.add_argument("--long", type=int, default=610, help="the single arrival's prompt tokens (dump: 610)")
    ap.add_argument("--period", type=float, default=1.2, help="seconds between PAIRS of short arrivals")
    ap.add_argument("--long-period", type=float, default=6.0, help="seconds between 610-token arrivals")
    ap.add_argument("--max-load", type=int, default=30, help="arrival valve: skip a round when running+waiting is "
                       "at or above this (the dumps sat at 25 running / 0 waiting and 31 / 13)")
    ap.add_argument("--deep", type=int, default=0, help="long-context requests to raise kv_cache_usage (0 = the "
                                                       "05:11 regime; 8 got near the 11-14 regime's 0.998)")
    ap.add_argument("--deep-tokens", type=int, default=40000)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--out", default=str(os.path.expanduser("~/logs/soak-mixed.jsonl")))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    stop = threading.Event()
    dead = threading.Event()
    why = {}
    lock = threading.Lock()
    live = {"dec": 0}
    fh = open(a.out, "a", buffering=1)

    def note(rec):
        with lock:
            fh.write(json.dumps(rec) + "\n")

    def chat(prompt, kind, maxtok):
        t0 = time.time()
        body = {"model": a.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": maxtok, "temperature": 0, "ignore_eos": True}
        try:
            r = post(a.base, "/v1/chat/completions", body)
            u = r.get("usage") or {}
            note({"t": t0, "kind": kind, "ok": True, "prompt": u.get("prompt_tokens"),
                  "out": u.get("completion_tokens"), "s": round(time.time() - t0, 2)})
            return True
        except Exception as e:                                                  # noqa: BLE001
            m = f"{type(e).__name__}: {e}"
            note({"t": t0, "kind": kind, "ok": False, "s": round(time.time() - t0, 2), "err": m[:200]})
            # any failure ends the run: a device fault shows up as a 500 / a closed connection / an engine dead
            # error, and the wall moment is the thing the engine log has to be lined up against
            why.setdefault("first", f"{kind}: {m}")
            dead.set()
            return False

    def decoder(pm):
        while not stop.is_set() and not dead.is_set():
            with lock:
                live["dec"] += 1
            ok = chat(pm.make(400, "dec"), "decoder", a.max_tokens)
            with lock:
                live["dec"] -= 1
            if not ok:
                return

    def arrivers(pm):
        """The mixed part: a PAIR of ~160-token arrivals every --period (05:11), one 610-token arrival every
        --long-period (11-14). Two at a time matters: one 165-token prefill is under PREFILL_MAX=256 and would go
        to OUR path; the pair is 330 and goes to stock's -- which is the door both faults went through.

        `--max-load` is the load valve. Without it the arrival rate is a metronome and the queue grows without
        bound (the 2-minute smoke run reached 37 waiting), and a queue that long is a DIFFERENT step shape from
        both dumps: 05:11 had running=25 waiting=0, 11-14 had running=31 waiting=13. So the driver asks the engine
        how full it is and skips a round rather than inventing a regime neither fault had."""
        next_pair, next_long = time.time() + 1.0, time.time() + a.long_period
        while not stop.is_set() and not dead.is_set():
            now = time.time()
            if now >= next_pair or now >= next_long:
                kv, run, wait = gauge(a.base)
                load = (run or 0) + (wait or 0)
                if load >= a.max_load:
                    next_pair = now + a.period                 # skipped, not queued: the valve is the point
                    next_long = max(next_long, now + a.period)
                    skipped = getattr(arrivers, "skipped", 0) + 1
                    arrivers.skipped = skipped
                    if skipped % 20 == 1:
                        log(f"  valve: running {run} + waiting {wait} = {load} >= --max-load {a.max_load}, "
                            f"holding arrivals ({skipped} round(s) skipped so far)")
                elif now >= next_pair:
                    next_pair = now + a.period
                    threading.Thread(target=chat, args=(pm.make(a.short, "s1"), "short", a.max_tokens),
                                     daemon=True).start()
                    threading.Thread(target=chat, args=(pm.make(a.short, "s2"), "short", a.max_tokens),
                                     daemon=True).start()
                else:
                    next_long = now + a.long_period
                    threading.Thread(target=chat, args=(pm.make(a.long, "long"), "long", a.max_tokens),
                                     daemon=True).start()
            time.sleep(0.05)

    try:
        with urllib.request.urlopen(a.base + "/v1/models", timeout=10) as r:
            names = [m.get("id") for m in json.loads(r.read()).get("data", [])]
        log(f"serve answers on {a.base}: models {names}")
    except (urllib.error.URLError, OSError) as e:
        log(f"BAD  nothing answering on {a.base}: {type(e).__name__}: {e}")
        return 2

    pm = Prompt(a.base, a.model)
    kv, run, wait = gauge(a.base)
    log(f"plan: pop={a.pop} pairs of {a.short} tok every {a.period}s + one {a.long} tok every {a.long_period}s"
        f" + deep={a.deep} x {a.deep_tokens} tok | minutes={a.minutes} | kv={kv} running={run} waiting={wait}"
        f" | tokens/word {pm.per_word:.2f}")
    if a.dry:
        log(f"dry: a {a.short}-token prompt is {len(pm.make(a.short, 'x'))} chars; nothing sent")
        return 0

    threads = [threading.Thread(target=decoder, args=(pm,), daemon=True) for _ in range(a.pop)]
    for t in threads:
        t.start()
    for i in range(a.deep):                          # the high-kv arm: long contexts, fresh prefixes
        threading.Thread(target=chat, args=(pm.make(a.deep_tokens, f"d{i}"), "deep", 120), daemon=True).start()
    threading.Thread(target=arrivers, args=(pm,), daemon=True).start()

    t_end, t_gauge, t0 = time.time() + a.minutes * 60, 0.0, time.time()
    while not dead.is_set() and time.time() < t_end:
        if time.time() - t_gauge >= 10:
            t_gauge = time.time()
            kv, run, wait = gauge(a.base)
            log(f"  kv={kv if kv is None else round(kv * 100, 1)}% running={run} waiting={wait} "
                f"decoders={live['dec']} elapsed={time.time() - t0:.0f}s")
        time.sleep(0.5)
    stop.set()

    kv, run, wait = gauge(a.base)
    health = None
    try:
        with urllib.request.urlopen(a.base + "/health", timeout=5) as r:
            health = r.status
    except (urllib.error.URLError, OSError):
        pass
    if dead.is_set():
        log(f"FAULT at {time.strftime('%Y-%m-%d %H:%M:%S')}: {why.get('first')}")
        log(f"  right after: kv={kv} running={run} waiting={wait} /health={health}")
        log("  read the engine log from that second: `r9700 gdn debug` lines carry idx/cu/acc, and the driver's "
            "[gfxhub] page fault line names the kernel (journalctl -k --since ...)")
        return 1
    log(f"no fault in {a.minutes:.0f} min (kv now {kv}, running {run}, waiting {wait}) -- a negative, timed and "
        f"recorded in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
