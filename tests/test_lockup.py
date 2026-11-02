#!/usr/bin/env python3
"""tests/test_lockup.py - the README's title lockup, checked as a rendered picture.

docs/lockup.py lays the lockup out and asserts from the font's own metrics that the wordmark and the
mark stay inside the canvas. That is the geometry. This is the pixels, and it catches what geometry
cannot see: an SVG that is valid XML and draws nothing (a path a renderer refuses), ink clipped at an
edge, the mark's four bars merged into a slab, letters merged into a blob or one letter missing, the
mark touching the word, and dark and light quietly drifting into two different drawings.

What it reads is the alpha channel's column and row profile: where ink starts and stops, how many
separate letter clusters the wordmark renders as, and how many bands the mark renders as.

NEEDS A RENDERER -- cairosvg and Pillow, and cairo itself underneath. Without root, the system
library comes out of a .deb into a directory and LD_LIBRARY_PATH:

    python3 -m venv /tmp/venv-render && /tmp/venv-render/bin/pip install -q cairosvg pillow
    mkdir -p /tmp/svglibs && cd /tmp/svglibs
    apt-get download libcairo2 libpixman-1-0 libxcb-shm0 libxcb-render0 libxrender1 libxext6 \\
        libxcb1 libxau6 libxdmcp6 libglib2.0-0t64 libgraphite2-3
    for f in *.deb; do dpkg -x "$f" /tmp/svglibs; done
    LD_LIBRARY_PATH=/tmp/svglibs/usr/lib/x86_64-linux-gnu /tmp/venv-render/bin/python tests/test_lockup.py

With no cairo on the box the import fails and the file skips rather than failing, so it never breaks
a machine that has no renderer. The rest of the README checks -- geometry, not pixels -- are in
tests/test_readme.py and run with stdlib alone.
"""
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SIZES = {"dark": DOCS / "r9700-stack-lockup-dark.svg", "light": DOCS / "r9700-stack-lockup-light.svg"}
OUT = 640                                                # the README shows 320; 640 is that at 2x

fail = 0


def ck(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        fail += 1


def clusters(vals, thresh=0):
    """Runs of consecutive entries above thresh, as (start, end) in index units."""
    out, s = [], None
    for i, v in enumerate(vals):
        if v > thresh:
            if s is None:
                s = i
        elif s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(vals) - 1))
    return out


try:
    import cairosvg
    from PIL import Image
except Exception as e:
    print(f"SKIP no SVG renderer here ({type(e).__name__}: {e}); see this file's docstring")
    sys.exit(0)


def alpha(path, width):
    try:
        png = cairosvg.svg2png(url=str(path), output_width=width)
    except Exception as e:                      # a file that parses and cannot be drawn
        return None, f"renderer refused it: {type(e).__name__}: {e}"
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    return im.getchannel("A"), ""


for scheme, path in SIZES.items():
    if not path.exists():
        ck(f"{scheme}: the lockup is in the repo", False, str(path))
        continue
    a, err = alpha(path, OUT)
    if a is None:
        ck(f"{scheme}: a renderer can draw it", False, err)
        continue
    w, h = a.size
    px = list(a.getdata())
    inked = sum(1 for v in px if v > 8)
    cols = [sum(1 for r in range(h) if px[r * w + c] > 8) for c in range(w)]
    rows = [sum(1 for c in range(w) if px[r * w + c] > 8) for r in range(h)]
    # the split between mark and word: the widest blank column run inside the drawing
    xs = [c for c in range(w) if cols[c] > 0]
    blank = clusters([cols[c] == 0 for c in range(xs[0], xs[-1] + 1)])
    gap = max(blank, key=lambda g: g[1] - g[0]) if blank else None
    split = xs[0] + gap[1] if gap else w // 2
    scale = w / float(re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', path.read_text()).group(1))

    ck(f"{scheme}: it renders as ink, not an empty canvas", inked > 4000,
       f"{inked} inked px of {w * h}, {w} x {h} at 2x the README's 320 px")
    ys = clusters(rows)
    ck(f"{scheme}: nothing is clipped at the canvas edge",
       xs[0] >= 2 and xs[-1] <= w - 3 and ys[0][0] >= 2 and ys[-1][1] <= h - 3,
       f"ink x {xs[0]}..{xs[-1]} of {w}, y {ys[0][0]}..{ys[-1][1]} of {h}")

    bands = clusters([sum(1 for c in range(split + 1) if px[r * w + c] > 8) for r in range(h)])
    ck(f"{scheme}: the mark renders as four separate bars, not a slab", len(bands) == 4,
       f"{len(bands)} bands at y " + str([f"{s / scale:.1f}-{e / scale:.1f}" for s, e in bands]))

    letters = clusters([cols[c] > 0 for c in range(split + 1, w)])
    ck(f"{scheme}: the word renders as eleven letters, none merged, none missing", len(letters) == 11,
       f"{len(letters)} clusters")

    wy = clusters([sum(1 for c in range(split + 1, w) if px[r * w + c] > 8) for r in range(h)])
    word_h = (wy[-1][1] - wy[0][0]) / scale if wy else 0
    ck(f"{scheme}: the wordmark is title-sized, not shrunk or blown up",
       0.35 * h / scale <= word_h <= 0.75 * h / scale,
       f"{word_h:.1f} units tall in a {h / scale:.0f}-unit canvas")
    mark_c = (bands[0][0] + bands[-1][1]) / 2 if bands else 0
    word_c = (wy[0][0] + wy[-1][1]) / 2 if wy else 0
    ck(f"{scheme}: mark and word are on the same middle line", abs(mark_c - word_c) <= 0.03 * h,
       f"centres {mark_c / scale:.1f} vs {word_c / scale:.1f} units")
    ck(f"{scheme}: mark and word do not touch", (gap[1] - gap[0]) / scale >= 8 if gap else False,
       f"{(gap[1] - gap[0]) / scale:.1f} units of air" if gap else "no gap at all")

d, _ = alpha(SIZES["dark"], OUT)
l, _ = alpha(SIZES["light"], OUT)
if d is not None and l is not None:
    ck("dark and light are one drawing in two inks, not two drawings",
       list(d.getdata()) == list(l.getdata()))

print("FAIL" if fail else "PASS", f"({fail} failing)")
sys.exit(1 if fail else 0)
