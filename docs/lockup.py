#!/usr/bin/env python3
"""The README's title: docs/r9700-stack-lockup-dark.svg and -light.svg.

Strata's README opens with a lockup instead of a markdown heading: a mark, the word drawn as
outlines, and dark/light variants switched by `prefers-color-scheme` (the Monitor/About design is
credited to Strata in CREDITS.md, MIT; this is the same shape, not the same file).

The word is set in the repo's own face -- the Outfit variable subset the Monitor and About pages use,
instantiated at weight 650 -- and converted to path outlines, so the wordmark is identical in every
viewer instead of whatever font that viewer happens to have, and so the reader downloads paths
rather than a font. An SVG `<text>` inside an `<img>` can do neither: no font is fetched, and what
renders is a guess. Outfit is OFL 1.1 and already credited in CREDITS.md; the licence covers exactly
this, embedding the outlines.

Four bars for the mark: the stack, and the four R9700s the repo is built around -- the same mark the
release page carries, drawn here in the blue step the badges use.

Needs fontTools and brotli (brotli only to read the woff2), and nothing else -- in a throwaway venv,
not the one that runs the serve:

    python3 -m venv /tmp/venv-lockup && /tmp/venv-lockup/bin/pip install -q fonttools brotli
    /tmp/venv-lockup/bin/python docs/lockup.py

The .svg files it writes are what the README uses; nothing here runs at serve time.
"""
import os
import sys

from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.varLib import instancer

FONT = "outfit-latin-wght.woff2"      # the Monitor/About face, in this repo (OFL 1.1)
WEIGHT = 650                          # the variable axis's wght; 650 reads as a title, 400 is UI
WORD = "r9700-stack"
VERSION = 34.0                        # the wordmark's em size in the lockup's own units
TRACK = 0.012                         # extra tracking in ems: geometric faces run tight on 'ck'
H, BASE = 48.0, 37.0                  # canvas height and the wordmark's baseline, like Strata's 48
MARK_W, GAP = 46.0, 15.0              # the bars and the space between mark and word
INK = {"dark": "#f0efec", "light": "#1a1a19"}
ACCENT = {"dark": "#3987e5", "light": "#2f81f7"}       # the blue steps used in the figures and badges


def outlines(font, word, size, x, baseline, tracking):
    """The word as (paths, x at its end, box its ink covers). Each glyph is placed by its own advance.

    fontTools 4.68's pens keep the pen they wrap privately, so the path pen and the bounds pen are
    both held here and each driven through its own transform pen.
    """
    cmap, hmtx, glyphs = font.getBestCmap(), font["hmtx"], font.getGlyphSet()
    upem = font["head"].unitsPerEm
    scale, x, paths, box = size / upem, x, [], None
    for ch in word:
        name = cmap.get(ord(ch))
        if name is None:
            sys.exit("no glyph for %r in %s" % (ch, FONT))
        sp = SVGPathPen(glyphs)
        glyphs[name].draw(TransformPen(sp, (scale, 0, 0, -scale, x, baseline)))
        d = sp.getCommands()
        if d:
            paths.append('<path d="%s"/>' % d)
        bp = BoundsPen(glyphs)
        glyphs[name].draw(TransformPen(bp, (scale, 0, 0, -scale, x, baseline)))
        bnd = bp.bounds or (x, baseline, x, baseline)
        box = bnd if box is None else (min(box[0], bnd[0]), min(box[1], bnd[1]),
                                       max(box[2], bnd[2]), max(box[3], bnd[3]))
        x += hmtx[name][0] * scale + tracking * size
    return paths, x - tracking * size, box


def bars(scheme):
    """Four right-aligned bars: the stack, all four in the accent -- the same mark the release page carries, so
    the README's title and its one-image summary are the same lockup. Their centre sits on the wordmark's
    cap-box centre, and the widest leaves 2 units of left inset. Returns the elements and the box they cover."""
    out, y = [], 5.5
    for w in (22.0, 30.0, 38.0, 44.0):
        out.append('<rect x="%.2f" y="%.2f" width="%.2f" height="7" rx="3.5" fill="%s"/>'
                   % (MARK_W - w, y, w, ACCENT[scheme]))
        y += 10.0
    return out, 5.5, y - 10.0 + 7.0


def render(scheme, font):
    paths, word_end, box = outlines(font, WORD, VERSION, MARK_W + GAP, BASE, TRACK)
    w = round(word_end + 2.0, 2)
    marks, mark_top, mark_bot = bars(scheme)
    body = marks + ['<g fill="%s">%s</g>' % (INK[scheme], "".join(paths))]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="%g" height="%g" viewBox="0 0 %g %g" role="img">'
           % (w, H, w, H)]
    svg.append("".join(body))
    svg.append("</svg>")
    return "\n".join(svg) + "\n", w, box, (mark_top, mark_bot)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(here, os.pardir, "serve", "web", "fonts", FONT)
    if not os.path.exists(font_path):
        sys.exit("no %s: the wordmark is outlined from it" % font_path)
    base = TTFont(font_path)
    font = instancer.instantiateVariableFont(base, {"wght": WEIGHT}, inplace=False, updateFontNames=False)
    for scheme in ("dark", "light"):
        svg, w, box, mark = render(scheme, font)
        out = os.path.join(here, "r9700-stack-lockup-%s.svg" % scheme)
        with open(out, "w") as f:
            f.write(svg)
        # both the wordmark and the mark have to sit inside the canvas: a lockup clipped at the edge is
        # the one failure a reader sees, and the one thing a generator can check before it writes
        assert 0 <= box[0] and box[2] <= w and 0 <= box[1] and box[3] <= H, \
            "the wordmark leaves the %g x %g canvas: %s" % (w, H, box)
        assert 0 <= mark[0] and mark[1] <= H, "the mark leaves the canvas: %s" % (mark,)
        print("%s: %s  %g x %g units, wordmark %g-%g wide, ink y %g..%g, mark y %g..%g"
              % (scheme, out, w, H, MARK_W + GAP, w, box[1], box[3], mark[0], mark[1]))


if __name__ == "__main__":
    main()
