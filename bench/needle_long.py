#!/usr/bin/env python3
"""Long-context gate for this box: needle-in-haystack out to the checkpoint's own context ceiling.

usage: needle_long.py [TIERS=30000,67000,150000,250000] [BASE=http://127.0.0.1:8080] [MODEL=Qwen3.8]
                     [TOKENS=/home/jiulu/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ] [MAXOUT=32] [WARM=1]

bench/quality.py's needle stops at 31,166 tokens because the recipe it gates runs at MAXLEN=32768. Asking
"how far does this box actually go" needs the same test taken to the model's own max_position_embeddings, and
three things that ladder does not have to say about itself:

  1. **Token counts are measured, not guessed.** The filler compresses to ~0.65 tokens a word, so counting
     words lands 35 % off. The haystack is built against the checkpoint's tokenizer and the tier is only
     accepted when usage.prompt_tokens says so.
  2. **Cold and warm are different runs.** This serve may or may not have prefix caching on, and a haystack
     made by repeating one sentence is exactly the case prefix caching eats. Every cold tier therefore carries
     a random nonce every ~40 sentences, so the shared prefix ends almost immediately; WARM=1 re-sends the
     identical haystack afterwards and the delta is the cache, not the model.
  3. **TTFT is measured to the first streamed chunk.** If the serve buffers the body, first-chunk time equals
     total time and the decode column is meaningless; a run that wants a real decode-at-depth number should
     ask for a few hundred output tokens (MAXOUT) and watch that the two times differ.
  4. **The wall is usually not the KV pool.** What has actually died on this line at long prompt was the
     prefill-side indexer scratch, and VRAM outside PyTorch's allocator. So every tier reports the per-card
     VRAM high-water while the request ran, and a tier that errors prints where.

A retrieval pass at 250k says the path is correct end to end; it does not say the answer would be right for
your prompt, and a highly repetitive haystack is far cheaper to attend to than prose.
"""
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

TIERS = [int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "30000,67000,150000,250000").split(",")]
BASE = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080").rstrip("/")
MODEL = sys.argv[3] if len(sys.argv) > 3 else "Qwen3.8"
TOKENIZER = sys.argv[4] if len(sys.argv) > 4 else "/home/jiulu/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ"
MAXOUT = int(sys.argv[5]) if len(sys.argv) > 5 else 32
WARM = (sys.argv[6] if len(sys.argv) > 6 else "1") != "0"
SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 0   # offset so two runs of the same tier are different haystacks
URL = BASE + "/v1/chat/completions"
TIMEOUT = float(os.environ.get("NEEDLE_TIMEOUT", "900"))

FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ",
          "A quick brown fox jumps over the lazy dog near the river bank. ",
          "Twelve spoons, a brass lamp, and one folded map sat on the shelf. ",
          "Rain arrived at four, stopped at six, and the tram was late again. ")


SAMPLES = []   # per-card VRAM in use, collected while a tier runs; sysfs reads, no rocm-smi (no /dev/kfd open)


def vram_samples(stop):
    """Sample every card's mem_info_vram_used until told to stop."""
    out, seen = [], []
    for card in sorted(os.listdir("/sys/class/drm")):
        p = f"/sys/class/drm/{card}/device/mem_info_vram_used"
        if card.startswith("card") and os.path.exists(p):
            seen.append((card, p))
    while not stop.is_set():
        row = []
        for card, p in seen:
            try:
                row.append(int(open(p).read().strip()))
            except OSError:
                row.append(0)
        if any(row):
            out.append(row)
            SAMPLES.append(row)
        time.sleep(0.4)
    return out


def ask(prompt, max_tokens):
    """One streaming chat call. Returns (text, prompt_tokens, completion_tokens, ttft, total, timings_ok)."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    chunks = []
    usage = {}
    fin = ""
    for raw in urllib.request.urlopen(req, timeout=TIMEOUT):
        if first is None and raw.startswith(b"data: "):
            first = time.monotonic() - t0
        if not raw.startswith(b"data: "):
            continue
        payload = raw[6:].strip()
        if payload == b"[DONE]":
            break
        try:
            d = json.loads(payload)
        except ValueError:
            continue
        if d.get("usage"):
            usage = d["usage"]
        for c in d.get("choices") or []:
            if c.get("finish_reason"):
                fin = c["finish_reason"]
            piece = (c.get("delta") or {}).get("content")
            if piece:
                chunks.append(piece)
    end = time.monotonic() - t0
    text = "".join(chunks)
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    if ct is None:
        ct = len(chunks)
    return text, pt, ct, first, end, fin


def build(n_tokens, seed):
    """Haystack of about n_tokens tokens with a passcode at 30-70 % depth.

    Sized against the checkpoint's own tokenizer (the filler compresses, so counting words lands ~35 % off):
    measure tokens-per-sentence, rebuild with the corrected count, stop when within 2 %. Nonces every ~40
    sentences so a prefix cache cannot reuse the earlier part of the haystack.
    """
    rng = random.Random(seed)
    tok = build.tok
    tps = getattr(build, "tps", 16.0)
    sentences = []
    got = 0
    for _ in range(4):
        n = max(8, int(n_tokens / tps) + 2)
        sentences = []
        for i in range(n):
            s = FILLER[i % len(FILLER)].strip()
            if i % 40 == 37:
                s = f"Nonce {rng.randint(100000, 999999)} marks a break in the text."
            sentences.append(s)
        text = " ".join(sentences)
        got = len(tok(text))
        build.tps = tps = got / len(sentences)
        if abs(got - n_tokens) <= 0.02 * n_tokens:
            break
    key = rng.randint(100000, 999999)
    cut = int(len(text) * rng.uniform(0.3, 0.7))
    while cut < len(text) and text[cut] != " ":
        cut += 1
    # ASK overrides the question, so the same haystack can be used to measure decode-at-depth (a long
    # answer) instead of a 7-token recall hit.
    ask = os.environ.get("ASK", "What is the secret passcode? Reply with the number only.")
    prompt = (text[:cut] + f" The secret passcode is {key}. " + text[cut:] + f"\n\n{ask}")
    return prompt, key, got, cut / max(1, len(text))


def load_tokenizer(path):
    """Text -> token ids. `tokenizers` alone, so sizing a haystack does not drag in torch (and with it the
    native venv's OpenMPI LD_LIBRARY_PATH requirement); transformers is the fallback."""
    try:
        from tokenizers import Tokenizer
        local = os.path.join(path, "tokenizer.json") if os.path.isdir(path) else path
        t = Tokenizer.from_file(local) if os.path.exists(local) else Tokenizer.from_pretrained(path)
        return lambda text: t.encode(text).ids
    except ImportError:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(path)
        return lambda text: t(text)["input_ids"]


def main():
    build.tok = load_tokenizer(TOKENIZER)
    print(f"base={BASE} model={MODEL} tiers={TIERS} warm={WARM}")
    print(f"{'tier':>8} {'prompt':>8} {'depth':>7} {'ttft':>8} {'prefill':>9} {'decode':>8} "
          f"{'VRAM hi/card':>13}  result")
    for n in TIERS:
        prompt, key, approx, frac = build(n, seed=n + SEED * 7919)
        stop = threading.Event()
        th = threading.Thread(target=vram_samples, args=(stop,), daemon=True)
        th.start()
        try:
            text, pt, ct, ttft, total, fin = ask(prompt, MAXOUT)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            stop.set()
            th.join(timeout=2)
            SAMPLES.clear()
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode("utf-8", "replace")[:400]
                except Exception:
                    pass
            print(f"{n:>8} {approx:>8} {'-':>7} {'-':>8} {'-':>9} {'-':>8} {'-':>13}  "
                  f"FAILED {type(exc).__name__}: {exc} {detail}")
            continue
        stop.set()
        th.join(timeout=2)
        hi = [max(col) for col in zip(*SAMPLES)] if SAMPLES else []
        SAMPLES.clear()
        # the depth column is where the passcode was inserted (character fraction of the haystack, applied to
        # the measured prompt length) -- not half the context.
        depth = int(pt * frac) if pt else int(approx * frac)
        gen = bool(os.environ.get("ASK"))
        res = "generation (ASK set)" if gen else ("hit" if str(key) in text else "MISS")
        prefill = pt / ttft if ttft else 0
        decode = (ct - 1) / (total - ttft) if ttft and ct > 1 else 0
        hi_s = "/".join(f"{v / 2**30:.1f}" for v in hi) or "-"
        print(f"{n:>8} {pt:>8} {depth:>7} {ttft:>7.2f}s {prefill:>7.0f}/s {decode:>7.1f}/s {hi_s:>13}  "
              f"{res} ({ct} tok, {fin})")
        if WARM:
            try:
                t2, pt2, ct2, ttft2, total2, fin2 = ask(prompt, MAXOUT)
                res2 = "generation (ASK set)" if gen else ("hit" if str(key) in t2 else "MISS")
                print(f"{n:>8} {pt2:>8} {depth:>7} {ttft2:>7.2f}s {pt2 / ttft2 if ttft2 else 0:>7.0f}/s "
                      f"{(ct2 - 1) / (total2 - ttft2) if ttft2 and ct2 > 1 else 0:>7.1f}/s {'(same ctx)':>13}  "
                      f"{res2} warm (prefix cache)")
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"{n:>8} {'-':>8} {'-':>7} {'-':>8} {'-':>9} {'-':>8} {'-':>13}  warm FAILED: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
