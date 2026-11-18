#!/usr/bin/env python3
"""bench/mmvl.py -- does the vision path actually work on a Nemotron-H omni launch, and what does an image cost?

Text throughput on this checkpoint is measured in notes/nemotron-super-vl.md; the image path was never exercised. The
checkpoint's processor is not Qwen's: patch 16, merge 2, `min_patches 1024` and `max_patches 13312`, so an image
arrives as between 256 and 3,328 merged tokens regardless of how big the file is, and the launcher's `MM_MAX_PIX` size
cap does not apply (the family passes `--mm-processor-kwargs {}`). Every number here is read off `usage.prompt_tokens`
against a text-only control of the same text, so the image's cost is a difference, not a guess.

The images are generated, not fixtures, so the run cannot drift against a stale PNG. Each one carries a code the model
has to read out of the pixels -- that is the comprehension check, and it is the only part that separates "the tokens
were produced" from "the model saw the picture".

  python3 bench/mmvl.py                      # against http://localhost:8080, model SuperVL
  SERVED_NAME=Other BASE=http://other:8080 python3 bench/mmvl.py
  ONLY=cost,limit python3 bench/mmvl.py      # run a subset
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("BASE", "http://localhost:8080")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from served_model import resolve          # the model name comes from the serve, never from this file
SERVED = os.environ.get("SERVED_NAME") or resolve(BASE)
ONLY = {s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()}
DIR = os.path.expanduser("~/r9700-mm-vl")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def make(name, w, h, code, note):
    """A documentation-like page: a headline, a bar chart, and a code in the pixels to read back."""
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 16), "R9700 stack -- vision probe", font=font(max(18, w // 45)), fill="black")
    d.text((20, 16 + max(26, h // 12)), f"code: {code}", font=font(max(24, w // 34)), fill=(180, 20, 20))
    d.text((20, h - 60), note, font=font(max(16, w // 60)), fill=(60, 60, 60))
    bars = [0.3, 0.62, 0.48, 0.85, 0.22, 0.7]
    x0, y0, bw = 20, h - 90, max(18, w // 14)
    for i, v in enumerate(bars):
        bh = int(v * (h * 0.35))
        d.rectangle([x0 + i * (bw + 10), y0 - bh, x0 + i * (bw + 10) + bw, y0], fill=(30, 90, 190))
    p = os.path.join(DIR, name)
    img.save(p)
    return p


os.makedirs(DIR, exist_ok=True)
IMG_DOC = make("doc.png", 1200, 654, "BRAVO-2288", "a documentation-sized page (about 0.8 MP)")
IMG_SMALL = make("small.png", 200, 160, "ALPHA-7731", "tiny -- below min_patches, gets upscaled")
IMG_HUGE = make("huge.png", 2560, 1440, "CHARLIE-9915", "a 3.7 MP screenshot: the max-cost case")
IMG_EXTRA = make("extra.png", 900, 600, "DELTA-4402", "a fourth picture, for the per-prompt limit")


def b64(p):
    with open(p, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def post(payload, stream=False, timeout=600):
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(payload).encode(),
                                headers={"content-type": "application/json", "authorization": "Bearer x"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300], time.time() - t0, None
    if not stream:
        return 200, body, time.time() - t0, None
    # streaming: the first content chunk is TTFT; the total is the wall of the whole completion
    first, txt, n = None, [], 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            if "[DONE]" in line:
                break
            n += 1
            try:
                d = json.loads(line[5:].strip())
                piece = (d.get("choices", [{}])[0].get("delta") or {}).get("content") \
                    or (d.get("choices", [{}])[0].get("delta") or {}).get("reasoning")
            except (json.JSONDecodeError, IndexError, KeyError):
                piece = None
            if piece and first is None:
                first = time.time() - t0
            if piece:
                txt.append(piece)
    return 200, "".join(txt), time.time() - t0, first


def part(p):
    return {"type": "image_url", "image_url": {"url": b64(p)}}


def ask(text, images=(), max_tokens=64, stream=False):
    content = [{"type": "text", "text": text}] + [p if isinstance(p, dict) else part(p) for p in images]
    pl = {"model": SERVED, "max_tokens": max_tokens, "temperature": 0,
          "messages": [{"role": "user", "content": content}]}
    if stream:
        pl["stream"] = True
        st, txt, wall, ttft = post(pl, stream=True)
        return st, {"text": txt}, wall, ttft
    st, body, wall, _ = post(pl)
    try:
        j = json.loads(body)
    except json.JSONDecodeError:
        return st, {"raw": body}, wall, None
    if st != 200:
        return st, {"error": (j.get("error") or {}).get("message", body)[:200]}, wall, None
    ch = j["choices"][0]
    return st, {"text": (ch.get("message") or {}).get("content") or "",
                "reasoning": (ch.get("message") or {}).get("reasoning") or "",
                "finish": ch.get("finish_reason"),
                "pt": j.get("usage", {}).get("prompt_tokens"), "ct": j.get("usage", {}).get("completion_tokens")}, wall, None


def run(what, label, want_http=200, want_in=None, text="What code is printed on this page? Answer with the code only.",
        images=(IMG_DOC,), control_text=None, max_tokens=900, stream=False):
    if ONLY and what not in ONLY:
        return
    st, j, wall, ttft = ask(text, images, max_tokens=max_tokens, stream=stream)
    ok = st == want_http
    got = (j.get("text") or "") + (j.get("reasoning") or "") or j.get("error") or ""
    if want_in:
        ok = ok and want_in.upper() in got.upper()
    line = f"  {'ok  ' if ok else 'FAIL'} {label:<46} http {st}  {wall:5.2f}s"
    if j.get("finish"):
        line += f"  finish={j['finish']}"
    if j.get("pt") is not None:
        line += f"  prompt {j['pt']:>7}"
        ctl = control(control_text or text)
        if ctl is not None:
            line += f"  (= +{j['pt'] - ctl} over the same text with no image)"
    if ttft is not None:
        line += f"  ttft {ttft:.2f}s"
    print(line + f"  -> {got[:90]!r}")
    return j


_CTL = {}


def control(text):
    """prompt_tokens of the same text with no images -- so an image's cost is a difference, not a guess."""
    if text in _CTL:
        return _CTL[text]
    st, j, _, _ = ask(text, (), max_tokens=8)
    _CTL[text] = j.get("pt") if st == 200 else None
    return _CTL[text]


print(f"vision checks against {BASE} model {SERVED}")
print("cost and comprehension:")
run("cost", "one doc page (1200x654)", want_in="BRAVO-2288")
run("cost", "tiny (200x160) -- min_patches floor", want_in="ALPHA-7731", images=(IMG_SMALL,))
run("cost", "huge (2560x1440) -- max_patches cap", want_in="CHARLIE-9915", images=(IMG_HUGE,))
run("count", "four pictures in one prompt (the limit)", images=(IMG_DOC, IMG_SMALL, IMG_HUGE, IMG_EXTRA), max_tokens=300,
    text="List every code printed on these pages, one per line, codes only.")
run("limit", "five pictures: the mm_last_turn filter keeps the newest four (so no 400, and the cost of four)",
    images=(IMG_DOC, IMG_SMALL, IMG_HUGE, IMG_EXTRA, IMG_DOC),
    text="List every code on these pages.", max_tokens=300)
run("video", "video is off on this launch (MM_VIDEO=0)", want_http=400,
    text="Describe this video.", images=({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,"
                                                                                   + base64.b64encode(b"\x00" * 64).decode()}},))
run("stream", "streaming an image answer (TTFT is the encoder + prefill)", stream=True, max_tokens=64)
# The tiny page failed its read-back at 900 tokens with finish=length: was the picture understood and the answer never
# emitted, or did the read fail? Give it the budget and ask where the code appears.
run("deep", "tiny page again with a 1,500-token budget", want_in="ALPHA-7731", images=(IMG_SMALL,), max_tokens=1500)
# Is the tiny-image miss about upscaling or about the text itself? Same page drawn at a size where every label is
# proportionally larger, and a plain "what can you read at all" on the tiny one.
IMG_MID = make("mid.png", 800, 640, "ECHO-5150", "mid-sized: every label is proportionally bigger")
run("mid", "mid page (800x640) -- is the fine text legible here?", want_in="ECHO-5150", images=(IMG_MID,), max_tokens=1200)
run("legible", "tiny page, 'what text can you read at all?'", images=(IMG_SMALL,), max_tokens=700,
    text="Transcribe every piece of text you can read on this page, exactly as written, one per line.")
run("ttfthuge", "streaming a 2560x1440 page: TTFT carries the encoder + 3.3k prefill", images=(IMG_HUGE,),
    stream=True, max_tokens=64)

# ------------------------------------------------------------------ images under concurrency: the unmeasured one --
# The launch runs images ON, and notes/nemotron-super-vl.md says the encoder steps interleaving with concurrent decode
# has never been measured. Same shape as bench/wave: N text streams, once alone and once while image requests are in
# flight, and the honest caveat is printed rather than hidden: this box's wave-to-wave spread is +-10%, so anything
# under that is a observation, not a claim.
def concur(n=8, toks=600, imgs=4):
    from concurrent.futures import ThreadPoolExecutor

    def text_run(i):
        st, j, wall, _ = ask("Count from 1 to 400, one number per line.", (), max_tokens=toks)
        return (j.get("ct") or 0), wall

    def img_run(i):
        st, j, wall, _ = ask("What code is printed on this page? Answer with the code only.", (IMG_HUGE,), max_tokens=200)
        return wall

    def longtext_run(i):
        # The control that makes the number above mean something: the same volume of PREFILL tokens (a ~3.3k-token
        # prompt, which is what the 2560x1440 page costs) with no encoder step at all. If the drop here matches the
        # drop with images, images are expensive because they are long, not because the encoder interleaves.
        filler = "The quick brown fox jumps over the lazy dog. " * 470      # ~3.3k tokens
        st, j, wall, _ = ask(filler + " Answer with the single word: pear.", (), max_tokens=200)
        return wall

    def wave(with_images, mode="image"):
        runner = img_run if mode == "image" else longtext_run
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n + imgs) as ex:
            fs = [ex.submit(text_run, i) for i in range(n)]
            if with_images:
                for i in range(imgs):
                    ex.submit(runner, i)
            done = [f.result() for f in fs]
        wall = time.time() - t0
        ct = sum(c for c, _ in done)
        per = sorted(c / w for c, w in done)
        return ct / wall, per[len(per) // 2], wall

    a_ct, a_per, a_wall = wave(False)
    b_ct, b_per, b_wall = wave(True, "image")
    c_ct, c_per, c_wall = wave(True, "text")
    print(f"\nimages under concurrency ({n} text streams x {toks} tokens, {imgs} injected requests)")
    print(f"  text only        : aggregate {a_ct:6.1f} tok/s, per-stream median {a_per:5.1f} tok/s, wall {a_wall:.1f}s")
    print(f"  + 4 image (3.3k) : aggregate {b_ct:6.1f} tok/s, per-stream median {b_per:5.1f} tok/s, wall {b_wall:.1f}s")
    print(f"  + 4 long text    : aggregate {c_ct:6.1f} tok/s, per-stream median {c_per:5.1f} tok/s, wall {c_wall:.1f}s")
    if a_ct:
        print(f"  images cost {100.0 * (b_ct / a_ct - 1):+.1f}% of the text aggregate; the same prefill volume without "
              f"an encoder costs {100.0 * (c_ct / a_ct - 1):+.1f}% -- the gap is the encoder, the rest is length "
              f"(this box's wave-to-wave spread is +-10%, so read the band, not the sign)")


if not ONLY or "concur" in ONLY:
    concur()
