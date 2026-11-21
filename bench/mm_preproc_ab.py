#!/usr/bin/env python3
"""bench/mm_preproc_ab.py -- same card, different cut: what the picture is RESIZED to, by whom, before the tower.

The encoder is not the variable between this stack and Strata: the tower is the same 27-block Qwen3-VL vision net
(333 BF16 tensors here; Strata's mmproj metadata says the same depth/hidden/patch/merge/mean/std), and feeding
pre-computed embeddings is not a seam this arch can use -- the LM takes merged embeds PLUS three `deepstack_input_embeds`
injections (vllm/models/qwen4_exp/amd/model.py:495-517), and `image_embeds` carries one tensor.

What IS ours to choose is the preprocessing: vLLM runs the checkpoint's HF processor, whose `smart_resize` rounds to
multiples of patch*merge=32 and clamps to the `--mm-processor-kwargs` band, with the checkpoint's own resample filter
(resample=3 in preprocessor_config.json). mtmd (llama.cpp) does the same job with its own filter, its own band
(min_pixels/max_pixels), and its own rounding. This A/B sends the SAME card cut each way, and checks the engine took
our cut unchanged (the prompt-token delta must equal rows*cols/1024 exactly -- if it resized, the number would not match).

Run: ~/venv-r9700/bin/python3 bench/mm_preproc_ab.py [--font /tmp/wqy.ttc] [--reps 2] [--img PATH ...]
"""
import argparse
import base64
import io
import json
import os
import statistics
import time
import urllib.request

from PIL import Image

MIN_PIX, MAX_PIX = 1_048_576, 4_194_304
GRID = 32                                             # patch 16 * merge 2 -- what smart_resize must round to
FILTERS = {"bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS, "box": Image.BOX, "nearest": Image.NEAREST}
NARROW = "图片里第一行文字是什么？只给那一行的文字。"
LONG = "图片里的文字写的是什么？逐行转录，不要解释。"
BASE = os.environ.get("BASE", "http://127.0.0.1:8180")


def model():
    return json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]


def norm(s):
    return "".join(str(s or "").split())


def cer(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / len(ref)


def collapsed(s):
    s = norm(s)
    return len(s) > 12 and len(set(s)) <= 8


def b64(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def fit_size(w, h, min_px, max_px):
    """mtmd's fit: scale into the band, then snap each side to a multiple of GRID (never below one grid cell)."""
    f = 1.0
    for _ in range(60):
        a = w * h * f * f
        if min_px <= a <= max_px:
            break
        f *= (min_px / a) ** 0.5 if a < min_px else (max_px / a) ** 0.5
    h2 = max(GRID, int(round(h * f / GRID)) * GRID)
    w2 = max(GRID, int(round(w * f / GRID)) * GRID)
    return w2, h2


def cut(img, filt, min_px=MIN_PIX, max_pix=MAX_PIX):
    w, h = img.size
    w2, h2 = fit_size(w, h, min_px, max_pix)
    return img.convert("RGB").resize((w2, h2), FILTERS[filt]), w2, h2


def ask(md, text, img_b64):
    pl = {"model": md, "max_tokens": 256, "temperature": 0.0,
          "messages": [{"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
              {"type": "text", "text": text}]}]}
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(pl).encode(),
                               headers={"content-type": "application/json"})
    t0 = time.time()
    j = json.loads(urllib.request.urlopen(r, timeout=300).read().decode())
    return norm(j["choices"][0]["message"]["content"]), j["usage"]["prompt_tokens"], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", action="append", default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--filters", default="bicubic,lanczos,box")
    a = ap.parse_args()
    cards = a.img or [os.path.expanduser(p) for p in
                      ("~/vision-probes/zh-dense-1400x560.png", "~/vision-probes/zh-dense-2800x1120.png",
                       "~/vision-probes/latin-2800x1120.png")]
    md = model()
    for path in cards:
        img = Image.open(path)
        ref = open(path[:-4] + ".txt").read()
        print(f"\n### {os.path.basename(path)} {img.size} band {MIN_PIX}..{MAX_PIX} px "
              f"({img.size[0]*img.size[1]} px -> engine's own cut)")
        variants = [("as-sent", img)]
        for f in a.filters.split(","):
            c, w2, h2 = cut(img, f)
            variants.append((f"{f} {w2}x{h2} -> {(w2//GRID)*(h2//GRID)} tok", c))
        for name, im in variants:
            b = b64(im)
            for q, refq, tag in ((NARROW, ref.splitlines()[0], "narrow"), (LONG, ref, "long")):
                outs = [ask(md, q, b) for _ in range(a.reps)]
                cer_ = statistics.mean(cer(refq, o[0]) for o in outs)
                tok = outs[0][1]
                print(f"  {name:34} {tag:6} CER {cer_*100:5.1f}%  collapse {sum(collapsed(o[0]) for o in outs)}/{a.reps}  "
                      f"pt {tok:6d}  {statistics.mean(o[2] for o in outs):4.2f} s   {outs[0][0][:44]!r}")


if __name__ == "__main__":
    main()
