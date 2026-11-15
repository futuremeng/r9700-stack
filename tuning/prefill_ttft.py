#!/usr/bin/env python3
"""TTFT probe for R9K_GDN_PREFILL_MAX: where the short-prefill core stops being the cheaper kernel.

`R9K_GDN_PREFILL_MAX` routes a step's GDN core by that step's prefill-token count: under it the core is our
token-by-token recurrence (one launch per layer), above it stock's chunked delta rule (~30 launches and ~1.3 ms of
Python per layer whatever the prompt length). The first is cheaper on short prompts and costs the GPU ~1.8 us per
token per layer, so there is a crossover -- measured at ~240 tokens on this model, ~440 on the 27B, when the knob
was introduced. The fault hunt (notes/bug-causal-conv1d-oob.md §16) wants to raise the knob to 1024, which pushes
steps well past that crossover onto the token-by-token path. That costs something, and the box should know what
before the knob moves in production.

Why a probe and not a soak comparison: a soak's per-request wall is dominated by ~700 tokens of decode, so a
20 ms prefill difference is inside its noise. Here `max_tokens=1` makes the wall essentially the prefill, and the
sizes bracket the knob:

    208 tok   ours under either value          610 tok   stock at 256, ours at 1024
    900 tok   stock at 256, ours at 1024      1300 tok   stock under BOTH -- the control

The control is the load-bearing row: if 1300 moves between two runs, the runs are not comparable (clocks, another
launch, KV occupancy) and the 610/900 rows say nothing about the knob.

Prompts are unique per request -- a per-request tag inside the filler -- because the prefix cache would otherwise
answer the repeats and report ~0 ms of prefill. The filler length is calibrated by asking the serve what it
counted (`usage.prompt_tokens`), since the ChatML template adds ~20 tokens and a guessed word count lands off
target.

Run it against a quiet engine: after a soak is fine, during one is not.
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request


def post(base, path, payload, key=None, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def filler(words, tag):
    """`words` filler words, tagged so no two prompts share a prefix worth caching."""
    body = " ".join(f"{i} alpha{i % 7} bravo charlie delta echo" for i in range(words))
    return f"{body} [probe {tag}]"


def calibrate(base, model, target, key):
    """Filler unit count that lands the prompt within a few tokens of `target`, as the serve counts it.

    The step is a ratio, not a difference: one filler unit is ~7 tokens, so adding `target - got` UNITS overshoots
    by the same factor (the first version of this did exactly that and "calibrated" 610 tok to 5 336."""
    n = max(8, int(target * 0.72))
    last = None
    for attempt in range(9):
        u = post(base, "/v1/chat/completions", {"model": model, "max_tokens": 1, "temperature": 0,
                                                "messages": [{"role": "user", "content": filler(n, f"cal{attempt}")}]},
                 key).get("usage") or {}
        got = u.get("prompt_tokens") or 0
        last = got
        if abs(got - target) <= 4:
            return n
        n = max(8, round(target * n / max(1, got)))      # tokens per unit, measured, not assumed
    print(f"    calibration for {target} tok settled at {last} tok -- the rows will say what was actually sent")
    return n


def one(base, model, key, words, out, bad, tag):
    try:
        t0 = time.time()
        post(base, "/v1/chat/completions", {"model": model, "max_tokens": 1, "temperature": 0,
                                            "messages": [{"role": "user", "content": filler(words, tag)}]}, key)
        out.append(time.time() - t0)
    except (urllib.error.URLError, OSError) as e:
        bad.append(str(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8180")
    ap.add_argument("--model", default="Qwen3.8")
    ap.add_argument("--key", default=None)
    ap.add_argument("--sizes", default="208,610,900,1300", help="prompt tokens (calibrated against the serve)")
    ap.add_argument("--conc", default="1,8")
    ap.add_argument("--rounds", type=int, default=4, help="repeats per cell; the report is the median of the rounds")
    a = ap.parse_args()

    print(f"# TTFT probe against {a.base} (model {a.model}); every probe is max_tokens=1, so the wall IS the prefill")
    bad_total = 0
    for size in [int(s) for s in a.sizes.split(",")]:
        words = calibrate(a.base, a.model, size, a.key)
        for conc in [int(c) for c in a.conc.split(",")]:
            meds = []
            for rnd in range(a.rounds):
                out, bad = [], []
                ths = [threading.Thread(target=one, args=(a.base, a.model, a.key, words, out, bad,
                                                          f"{size}-{conc}-{rnd}-{i}"))
                       for i in range(conc)]
                [t.start() for t in ths]
                [t.join() for t in ths]
                meds += out
                bad_total += len(bad)
                for e in bad[:2]:
                    print(f"    probe failed: {e}")
            if meds:
                med = statistics.median(meds)
                print(f"  {size:>5} tok (filler {words:>4} words)  conc {conc:>2}   median wall {med*1000:>7.1f} ms"
                      f"   over {len(meds)} probes")
            else:
                print(f"  {size:>5} tok  conc {conc:>2}   no successful probes")
    if bad_total:
        print(f"\n{bad_total} probe(s) failed -- read the rows as a floor, not a comparison")
        return 1
    print("\nThe 1300-token row is stock's path under every knob value: compare two runs by it. If it moved, the")
    print("other rows moved with the box, not with the knob.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
