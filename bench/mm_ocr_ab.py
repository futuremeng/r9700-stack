#!/usr/bin/env python3
"""bench/mm_ocr_ab.py - does reading a picture line by line beat asking for one long transcription of it?

What it settled (notes/multimodal-input.md, 2026-11-21): a picture this box could not read at all -- `i il i il i`,
`机机机机`, whole pages answering `''` -- was not a reading problem. `--mm-encoder-tp-mode weights` (vLLM's default)
splits the vision tower across the TP ranks, and on 4x R9700 that quarter-tower is not numerically equivalent; with
`MM_TP_MODE=data` the same requests come back character-exact (CER 0.0%). Under the broken mode this bench's pre-pass
was the difference between garbage and a usable answer (96.8% -> 32.3% CER, collapses 2/2 -> 0/2), which is what made
the shape of the failure legible: a SHORT answer was always right, a long transcription collapsed.

It also killed two plausible ideas with numbers: equal-height BAND cuts, the obvious way to slice a page, measured
WORSE than the baseline (99.8% / 8.5 s) -- a band of a big-font picture is still a long answer, and the ask itself
collapsed into `机机机机`; per-text-line crops are the shape that worked.

The in-engine version of this (`r9700_vllm/compat/mm_ocr_pass.py`) was implemented, measured, and deleted the same
day: with the tower replicated it buys nothing and costs 40-90% more wall time on the picture turn. It is in git at
`6cc716c`; the policy lives here, self-contained, so the experiment replays against any live engine with no restart
and no plugin:

  baseline         one request: the whole picture + "逐行转录"
  pre-pass lines   one crop per text line (row-darkness segmentation below), one narrow ask each, then one request
                   carrying the merged, labelled text block AND the picture
  pre-pass bands   the same with equal-height bands, kept because it is the counterexample

Reported per card: character error rate against the ground truth, whether the answer COLLAPSED (a repetition cycle: one
character dominating, or an empty answer with the budget burned), the prompt tokens the main request paid, and wall
time including the asks.

Needs a CJK font (this box has none by default: `fc-list :lang=zh` is empty and drawing Chinese with DejaVu draws
tofu, which is how one earlier "Chinese OCR works" claim turned out to be meaningless). Point it at one with
--font, e.g. /tmp/wqy.ttc.

Run: ~/venv-r9700/bin/python3 bench/mm_ocr_ab.py [--font /tmp/wqy.ttc] [--repeat 3] [--bands 4] [--sizes 34,110] [--url ...]
"""
import argparse
import base64
import difflib
import io
import json
import os
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFont

# The policy, owned here since the in-engine pass was deleted (git 6cc716c). Kept verbatim from it, because these
# numbers are what the CER table in notes/multimodal-input.md was measured with.
ASK = ("逐行转录这张图里的文字，只输出文字本身，一行一条；"
       "看不清或图中没有文字就只输出：无")
BLOCK_OPEN = "[图片文字·自动识别，可能有错]"
BLOCK_CLOSE = "[/图片文字]"
NOTE_EMPTY = f"{BLOCK_OPEN}\n（没有识别出文字）\n{BLOCK_CLOSE}"
MIN_BAND_H, OVERLAP = 160, 0.08          # a band shorter than a glyph row answers 无 and wastes an ask


def band_slices(h, n, min_h=MIN_BAND_H, overlap=OVERLAP):
    """(y0, y1) covering 0..h in reading order, overlapping `overlap` of their own height so a text line is not cut
    at a boundary; a picture that cannot host n bands gets as many as fit."""
    if h <= 0:
        return []
    n = max(1, int(n))
    if h < min_h * n:
        n = max(1, h // min_h)
    step = h / n
    pad = int(step * overlap)
    return [(max(0, int(i * step) - (pad if i else 0)), min(h, int((i + 1) * step) + (pad if i < n - 1 else 0)))
            for i in range(n) if min(h, int((i + 1) * step) + (pad if i < n - 1 else 0)) >
            max(0, int(i * step) - (pad if i else 0))]


def line_slices(profile, min_h=10, pad=6, merge_gap=6):
    """(y0, y1) per TEXT LINE from a per-row dark-pixel count: runs of rows, gaps <= `merge_gap` merged (CJK lines have
    holes -- 川, 三 -- and a wrapped sentence leaves a hole inside one line), runs shorter than `min_h` dropped (a rule,
    a dot), each line padded by `pad` for ascenders and descenders. [] means no line structure: a photo, a chart."""
    h = len(profile)
    if not h:
        return []
    runs, y = [], 0
    while y < h:
        if profile[y] > 0:
            y0 = y
            while y < h and profile[y] > 0:
                y += 1
            runs.append([y0, y])
        else:
            y += 1
    if not runs:
        return []
    out = [runs[0]]
    for r in runs[1:]:
        if r[0] - out[-1][1] <= merge_gap:
            out[-1][1] = r[1]
        else:
            out.append(r)
    return [(max(0, a - pad), min(h, b + pad)) for a, b in out if b - a >= min_h]


def merge(cuts):
    """The cut answers in reading order, empty ones (无 / blank) dropped."""
    out = []
    for c in cuts or ():
        t = (c or "").strip()
        if not t:
            continue
        keep = [ln.strip() for ln in t.splitlines() if ln.strip() and ln.strip() not in ("无", "無", "无文字", "（无）")]
        if keep:
            out.append("\n".join(keep))
    return "\n".join(out)


def block_of(text):
    return NOTE_EMPTY if not text else f"{BLOCK_OPEN}\n{text}\n{BLOCK_CLOSE}"

FONTS = ("/tmp/wqy.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         os.path.expanduser("~/.fonts/wqy-microhei.ttc"))


def pick_font(arg):
    for p in ([arg] if arg else []) + list(FONTS) + [os.environ.get("R9K_CJK_FONT", "")]:
        if p and os.path.isfile(p):
            return p
    return None


def card(lines, px, w=1400, h=560, extra=None):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, px)
    y = 24
    for ln in lines:
        d.text((36, y), ln, font=f, fill="black")
        y += int(px * 1.5)
    if extra:                                            # a cover-style page: a small-print line too
        f2 = ImageFont.truetype(FONT, extra[1])
        d.text((36, h - 40), extra[0], font=f2, fill="black")
    return im


def b64(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def post(parts, max_tokens=800):
    # thinking OFF for the whole bench: with it on, an ask can spend its entire budget reasoning and return content='' --
    # which the first run of this bench scored as "the pass found nothing" and "the baseline collapsed". Measured on
    # 2026-11-21: several empty rows were the thinking budget, not the picture. Both arms run without it, so the table
    # measures the read.
    pl = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0,
          "chat_template_kwargs": {"enable_thinking": False},
          "messages": [{"role": "user", "content": parts}]}
    r = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(pl).encode(),
                              headers={"content-type": "application/json"})
    t0 = time.time()
    j = json.loads(urllib.request.urlopen(r, timeout=300).read().decode())
    ch = j["choices"][0]
    m = ch["message"]
    return {"pt": j["usage"]["prompt_tokens"], "ct": j["usage"]["completion_tokens"],
            "fin": ch.get("finish_reason"), "wall": time.time() - t0,
            "text": (m.get("content") or "").strip(), "reasoning": (m.get("reasoning") or "").strip()}


def body_of(r):
    return r["text"] or r["reasoning"]


def collapsed(r):
    """A repetition cycle: one character carrying a whole window, or the token budget burned without an answer."""
    b = body_of(r)
    if not b:
        return True
    w = b[:160]
    top = max((w.count(c) for c in set(w) if not c.isspace()), default=0)
    return top / max(1, len(w.replace(" ", ""))) > 0.35 or (r["fin"] == "length" and len(b) > 400)


def cer(got, want):
    """Character error rate against the card's own text: what a transcription benchmark would call CER."""
    g = "".join(body_of(got).split())
    w = "".join(want.split())
    if not w:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, g, w).ratio()


def bands_of(im, n):
    out = []
    for y0, y1 in band_slices(im.height, n):
        buf = io.BytesIO()
        im.crop((0, y0, im.width, y1)).save(buf, format="PNG")
        out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    return out


def lines_of(im):
    """One crop per TEXT LINE, from the row-darkness profile -- the shape the plugin prefers, because a line is a
    SHORT answer, and a short answer is the one this checkpoint does not collapse on (measured: an equal-height
    band of a big-font picture is still a long answer, and the ask itself collapsed into 机机机机)."""
    g = im.convert("L")
    w, h = g.size
    px = g.load()
    profile = [sum(1 for x in range(0, w, 2) if px[x, y] < 160) for y in range(h)]
    out = []
    for y0, y1 in line_slices(profile):
        buf = io.BytesIO()
        im.crop((0, y0, im.width, y1)).save(buf, format="PNG")
        out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    return out


def crops_of(im, policy, n):
    got = lines_of(im) if policy == "lines" else bands_of(im, n)
    return got or bands_of(im, n)              # no line structure -> equal bands, exactly as the plugin falls back


def pre_pass(im, url, n, policy="lines"):
    """The plugin's slice policy, run from here: k narrow asks, merged into one labelled block."""
    texts = []
    t0 = time.time()
    for u in crops_of(im, policy, n):
        r = post([{"type": "image_url", "image_url": {"url": u}}, {"type": "text", "text": ASK}], 512)
        texts.append(body_of(r))
    return merge(texts), time.time() - t0, sum(1 for t in texts if t.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("R9K_URL", "http://127.0.0.1:8180"))
    ap.add_argument("--font", default="")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument("--sizes", default="", help="comma list of glyph heights to run (e.g. 34,110); empty = all four cards")
    a = ap.parse_args()
    global URL, FONT, MODEL
    URL, FONT = a.url.rstrip("/"), pick_font(a.font)
    if not FONT:
        sys.exit("no CJK font found (--font PATH / R9K_CJK_FONT). Drawing Chinese without one produces tofu, and a\n"
                 "tofu probe measures nothing -- notes/multimodal-input.md has the incident. Tried: "
                 + ", ".join(FONTS))
    MODEL = json.load(urllib.request.urlopen(URL + "/v1/models"))["data"][0]["id"]
    print(f"url {URL}  model {MODEL}  font {os.path.basename(FONT)}  bands {a.bands}  repeat {a.repeat}\n")

    L = ["机器学习模型识别", "合同需在月底前签署", "本季度销售额增长 18.6%"]
    all_cards = [("dense 34 px (cover small print)", 34, card(L, 34), "\n".join(L)),
                 ("medium 110 px (a slide title)", 110, card(L, 110), "\n".join(L)),
                 ("one line only", 96, card(L[:1], 96), L[0]),
                 ("page + small print", 96, card(L[:2], 96, extra=("图中另有小字第七行", 26)),
                  "\n".join(L[:2] + ["图中另有小字第七行"]))]
    cards = [(n, im, want) for n, px, im, want in all_cards if not a.sizes or str(px) in a.sizes.split(",")]
    Q = "图片里的文字写的是什么？逐行转录，不要解释。"
    hdr = f"{'card':30} {'mode':17} {'CER':>6} {'collapse':>9} {'prompt tok':>10} {'wall s':>7}"
    print(hdr); print("-" * len(hdr))
    tot = {}
    for name, im, want in cards:
        url = b64(im)
        for mode in ("baseline", "pre-pass lines", f"pre-pass bands x{a.bands}"):
            cer_v, col, pt, wall = [], 0, 0, []
            for _ in range(a.repeat):
                if mode == "baseline":
                    r = post([{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": Q}])
                    pt, w = r["pt"], r["wall"]
                else:
                    text, ask_w, _ = pre_pass(im, url, a.bands, "lines" if "lines" in mode else "bands")
                    # the same fidelity as the plugin: nothing recognized means the request is NOT rewritten with an
                    # empty block -- an image the model can read beats a pass that found nothing
                    parts = [{"type": "image_url", "image_url": {"url": url}}]
                    if text:
                        parts.append({"type": "text", "text": block_of(text)})
                    parts.append({"type": "text", "text": Q})
                    r = post(parts)
                    pt, w = r["pt"], r["wall"] + ask_w
                cer_v.append(cer(r, want))
                col += 1 if collapsed(r) else 0
                wall.append(w)
                samp = body_of(r).replace("\n", " | ")[:74]
            k = (name, mode)
            tot[k] = (sum(cer_v) / len(cer_v), col, pt, sum(wall) / len(wall))
            c, cc, p, w = tot[k]
            print(f"{name:30} {mode:17} {c * 100:5.1f}% {cc}/{a.repeat}{'':3} {p:10d} {w:7.2f}")
            print(f"{'':30} {'':17} {samp}")
        print()

    print("verdict (mean over cards):")
    for mode in ("baseline", "pre-pass lines", f"pre-pass bands x{a.bands}"):
        v = [tot[(n, mode)] for n, _, _ in cards]
        print(f"  {mode:17} CER {sum(x[0] for x in v) / len(v) * 100:5.1f}%   "
              f"collapsed {sum(x[1] for x in v)}/{a.repeat * len(cards)}   "
              f"wall {sum(x[3] for x in v) / len(v):5.2f} s")


if __name__ == "__main__":
    main()
