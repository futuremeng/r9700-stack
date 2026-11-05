#!/usr/bin/env python3
"""docs/r9700-stack-page.svg -- the README's one-page release figure, drawn from the numbers below. The name is
version-free on purpose: the version is data in the script (VERSION below), so the figure the README points at does
not go stale at the next tag and the tag's own copy is a git artifact, not a filename.

Why a script and not the file: the figure is a stack of claims, and every release moves some of them.
Hand-edited, it drifts -- the version this replaces still said Flash-Next 210 tok/s and 27B 216 under a
"210 W cap" banner while the README's own table said 225 and 222, and it drew uncapped numbers under
that same banner. The same argument is why this page, not the tag's own hand-drawn
``docs/r9700-stack-v0.4.0.svg``, is what the README carries after the v0.4.0 merge: measured, that one's smallest
type is 11.0 px on a 1200-unit canvas, 9.0 px rendered in the column; this one's floor is 14 px on a 1000-unit
canvas. tests/test_readme.py measures both rather than asserting the preference. Here the numbers are data, each row names the gate it was measured at, and
tests/test_readme.py re-draws the figure and compares it string for string, so a number can move in one
place only.

Canvas: 1000 px wide, because github.com's README column is ~980 px -- the figure lands at about 1.0
scale, so the type sizes below are the sizes the reader gets. The 1200-unit canvas it replaces put its
smallest type at 9.8 px in that column, which is why it had to be clicked to read. The colour tokens are
serve/web/tokens.css's dark theme, the same set docs/monitor-cards.py draws with, so the README's two
figures are one drawing.

Every string is measured before it is placed (CHAR em per glyph, conservative), every box is sized from
the content that has to fit in it, and anything that would still run past its box is reported on stderr
as OVERFLOW -- the test fails on that rather than shipping a figure with text under a border.

    python3 docs/summary-page.py                       # writes the .svg
    rsvg-convert -w 2000 docs/r9700-stack-page.svg -o docs/r9700-stack-page.png
"""
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/r9700-stack-page.svg"

BG, CARD, CARD2, LINE, INK, SOFT, MUTED = ("#0e1113", "#161a1d", "#1d2226", "#2a3036",
                                           "#eef1f3", "#c7cdd3", "#8e979f")
ACCENT, BAR, TAG = "#34d399", "#7b93ff", "#f5b53d"       # accent, the decode bars, the card-count marks
FONT = "Outfit, 'DejaVu Sans', system-ui, sans-serif"
W, PAD = 1000, 16
CHAR = 0.60                                              # em per glyph, as in docs/monitor-cards.py
HEAD, BODY, SMALL = 19, 15, 14                           # SMALL is the floor: 14 px at the column's width

VERSION, RELEASED, LINE_OF = "v0.4.2", "2026-11-04", "vLLM 0.32 line, nightly 2026-10-25"
TAGLINE = "LLM inference on 2 to 8 AMD Radeon AI PRO R9700"
WHAT = "a vLLM plugin plus HIP and Triton kernels for gfx1201 (RDNA4)"
BADGES = [("stock vLLM 0.32", 1), ("stock ROCm 10", 1), ("no fork", 0),
          ("unmodified checkpoints", 0), ("loaded at runtime", 0)]

# model, the one line under it, what it brings, how many cards, and the note under the card count
MODELS = [
    ("Qwen3.8-Flash-Next", "MoE, 288 experts", [
        "MXFP4 experts, fp8 dense",
        "MTP-2/3 speculative decoding",
        "int6 embeddings in host RAM",
        "262k context, fp8 e4m3 main KV"], 4,
     "or two cards, experts in RAM"),
    ("Qwen3.8-27B-NVFP4", "dense + DFlash2 drafter", [
        "NVFP4 in, MXFP4 at load",
        "fp8 KV cache, fp8 drafter",
        "DFlash2 shares the MTP kernels"], 2, None),
    ("GLM-5.3-Flash", "sparse MLA + indexer", [
        "NVFP4 checkpoint + MTP layer",
        "NVFP4 in, MXFP4 at load",
        "fp8 KV cache, MTP-3"], 8,
     "one-kernel 8-rank collectives"),
    ("DeepSeek-V4.1-Flash", "552B MoE, EXL3 2.9 bpw", [
        "384 routed experts, Engram tables",
        "EXL3 GEMV + 48-row expert decode",
        "DSpark k=5 drafter, adaptive",
        "387k-token KV pool at 32k"], 8,
     "128k recipe: 392k KV, three needles"),
]

# single-stream decode. Every row names its gate: the 210 W figures and the uncapped ones come from
# different machines and have never been measured on both, so they do not share a banner.
GATE, UNCAP = "210 W gate", "the native box, uncapped"
DSGATE = "8 cards, -500 MHz"                      # upstream's eight-card gate: a clock offset, not a power cap
BARMAX, BARLEN = 250.0, 300
BARS = [
    ("Qwen3.8-Flash-Next", "TP4, four cards", 225, "14.3-14.4 ms a step; best sample 228", GATE),
    ("Qwen3.8-27B-NVFP4", "TP2, two cards", 222, "4.5 tokens a step with the DFlash2 drafter", GATE),
    ("GLM-5.3-Flash", "TP8, eight cards", 100, "75-80 before v0.3.8's host-memory collective", GATE),
    ("DeepSeek-V4.1-Flash", "TP8+EP, EXL3", 215, "on code; 104 prose (v0.4.1: 202, 87-97)", DSGATE),
]

# the column beside the bars: what the same four cards do in aggregate, and what the context costs.
# One gate label for the whole column, in its header -- every row in it is the uncapped native box.
FACTS = [
    ("1,792 tok/s", "aggregate decode, 32 in flight, 1,200-token waves, fp8 main KV"),
    ("9,702 tok/s", "reading a 7,792-token prompt on that same fp8 launch"),
    ("935,216 tokens", "KV pool on this box's standing launch (MAXLEN 262,144, fp8 e4m3 main KV, image inputs "
     "on): 1,009,899 without the vision knobs, 540,202 at bf16; needle found at 250,078"),
    ("115 tok/s", "what a real request actually finishes at, median over 1,525 of them on this launch "
     "(162 tok/s with the box to itself)"),
]

FOOT = ["Each number names its machine: \"210 W gate\" is the release configuration (210 W a card, -42 mV, 32k); "
        "\"the native box\" is one 4x R9700 box, uncapped. They are not comparable across.",
        "The Flash-Next column is the fp8 e4m3 main-KV launch this box runs by default: the pool doubles (1,009,899 "
        "against 540,202 tokens; with image inputs on, as today's launch runs, 935,216) and GSM8K 800 reads 96.0 "
        "against 95.6 %, HumanEval 164 98.17 against 96.95.",
        "v0.4.2 is the DeepSeek step measured a launch at a time: a per-kernel ledger of the step, an eight-rank "
        "all-reduce at 160 KiB-1.5 MiB, an expert placement map for the rank skew, the EXL3 GEMV trimmed, and the "
        "48-token expert tier released -- the drafter collapse that held it back in v0.4.1 was a graph-pool-layout "
        "latent bug, not that kernel.",
        "github.com/bkvargyas/r9700-stack"]

warn, out = [], []


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wid(s, fs):
    """How wide this line will be: conservative, so what fits here fits in the browser."""
    return len(s) * fs * CHAR


def fits(where, s, fs, box):
    if wid(s, fs) > box:
        warn.append(f"{where}: {s[:44]!r} at {fs}px needs {wid(s, fs):.0f}px, the box holds {box:.0f}px")


def wrap(s, fs, box):
    per = max(1, int(box / (fs * CHAR)))
    words, lines, cur = s.split(" "), [], ""
    for w_ in words:
        cand = (cur + " " + w_).strip()
        if wid(cand, fs) <= box or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    for ln in lines:
        if wid(ln, fs) > box:
            warn.append(f"unwrappable {ln!r} at {fs}px: needs {wid(ln, fs):.0f}px, box holds {box:.0f}px")
    return lines


def text(x, y, s, fs, fill=SOFT, weight=None):
    w_ = f' font-weight="{weight}"' if weight else ""
    out.append(f'<text x="{round(x, 1)}" y="{round(y, 1)}" font-size="{fs}"{w_} fill="{fill}">{esc(s)}</text>')


def rect(x, y, w_, h, fill, rx=0, stroke=None):
    s_ = f' stroke="{stroke}"' if stroke else ' stroke="none"'
    out.append(f'<rect x="{round(x, 1)}" y="{round(y, 1)}" width="{round(max(w_, 0), 1)}" '
               f'height="{round(max(h, 0), 1)}" rx="{rx}" fill="{fill}"{s_}/>')


def pill(x, y, s, fs, fg, solid, bg=None, stroke=None):
    """A rounded label; returns the width it took, so the caller can lay the next one out."""
    w_ = wid(s, fs) + 22
    rect(x, y - fs - 6, w_, fs + 13, (bg or CARD2) if solid else "none", rx=(fs + 13) / 2,
         stroke=stroke or (None if solid else LINE))
    text(x + 11, y, s, fs, fg, "700" if solid else None)
    return w_


def mark(x, y, s=1.0, colour=ACCENT):
    """The stack: four right-aligned bars -- the title lockup's mark, at whatever scale this page needs."""
    for i, bw in enumerate((22.0, 30.0, 38.0, 44.0)):
        rect(x + (44 - bw) * s, y + i * 10 * s, bw * s, 7 * s, colour, rx=3.5 * s)
    return 46 * s


# ---------------------------------------------------------------- the page is its content: measure first
L = PAD                                            # everything is inside this frame
R = W - PAD
SECX = 128                                         # section labels own the left column; see the guard below
SECW = R - SECX                                    # and the content area runs to the frame

what_lines = wrap(WHAT, HEAD, SECW)
badges, bx, row = [], SECX, []
for s, solid in BADGES:
    w_ = wid(s, SMALL) + 22
    if bx + w_ > R:                                # a badge that no longer fits starts the next row
        badges.append(row)
        row, bx = [], SECX
    row.append((bx, s, solid, w_))
    bx += w_ + 8
badges.append(row)
badge_rows = len(badges)
for r_ in badges:
    for x, s, solid, w_ in r_:
        if x + w_ > R:
            warn.append(f"01 badges: {s!r} still runs past the frame at x={x + w_:.0f}")

COLS = 2                                           # four models, two cards across: the name still fits at 19 px
CW = (SECW - (COLS - 1) * 14) / COLS               # one 14 px gutter between the cards of a row
cards = []
for name, sub, bullets, n, note in MODELS:
    bl = [ln for b in bullets for ln in wrap(b, SMALL, CW - 24)]
    nl = wrap(note, SMALL, CW - 24) if note else []
    fits(f"02 {name}", name, HEAD, CW - 24)
    fits(f"02 {name} sub", sub, SMALL, CW - 24)
    fits(f"02 {name} tp", f"TP{n} · {n} cards", BODY, CW - 24 - n * 16)
    cards.append((name, sub, bl, n, nl))


def card_h(nlines, notes):
    """The model card's panel height: the name and the line under it, the bullets, the card-count row, the
    note, and the descent of the last line. Measured, because how many bullet lines a card needs is a
    property of its text, not of the layout."""
    return 108 + 20 * max(nlines - 1, 0) + 22 + 20 * notes + 12


CARDH = max(card_h(len(c[2]), len(c[4])) for c in cards)
ROWS = (len(cards) + COLS - 1) // COLS

bar_rows = []
for name, tp, v, note, gate in BARS:
    fits(f"03 {name}", name, BODY, 210)
    fits(f"03 {name} sub", f"{tp} · {gate}", SMALL, 300)
    fits(f"03 {name} note", note, SMALL, SECW - BARLEN - 130)
    bar_rows.append((name, f"{tp} · {gate}", v, note))
BARH = 64 + 86 * (len(bar_rows) - 1) + 62 + 16        # model, gate, bar, note, descent

FACTW = 340
FACTX = R - FACTW
facts = [(v, wrap(what, SMALL, FACTW)) for v, what in FACTS]
FACTH = 56 + 18 + sum(22 + 18 * len(f[1]) + 12 for f in facts)   # +18: the column names its gate

SEC1H = 110 + 26 * (len(what_lines) - 1) + 30 * (badge_rows - 1)
SEC2H = 52 + ROWS * CARDH + (ROWS - 1) * 14        # 26 px of panel above and below each card row
SEC3H = max(BARH, FACTH) + 16
HEADH = 84

ys, y = [], HEADH + 6
for h in (SEC1H, SEC2H, SEC3H):
    ys.append(y)
    y += h + 14
foot_lines = sum(len(wrap(f, SMALL, W - 2 * PAD)) for f in FOOT)
H = ys[-1] + SEC3H + 14 + 18 * foot_lines + 8

out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="{FONT}" role="img">')
rect(0, 0, W, H, BG)

# ---------------------------------------------------------------- the header: the lockup, the line, the version
mw = mark(L, 26, 1.0)
text(L + mw + 16, 58, "r9700-stack", 34, INK, "800")
tx = L + mw + 16 + wid("r9700-stack", 34) + 16
pill(R - wid(f"{VERSION} · {RELEASED} · {LINE_OF}", SMALL) - 22, 52, f"{VERSION} · {RELEASED} · {LINE_OF}",
     SMALL, ACCENT, 0, stroke=ACCENT)
fits("header tagline", TAGLINE, SMALL, W - 2 * PAD)
text(L, 78, TAGLINE, SMALL, MUTED)

# ---------------------------------------------------------------- the three section frames
for (n, name, y, h) in (("01", "WHAT IT IS", ys[0], SEC1H), ("02", "WHAT RUNS", ys[1], SEC2H),
                        ("03", "HOW FAST", ys[2], SEC3H)):
    fits(f"section label {name}", name, SMALL, SECX - L - 16)   # a label this column cannot hold gets painted
                                                             # over by the content that starts at SECX
    rect(L, y, W - 2 * PAD, h, CARD, rx=12, stroke=LINE)
    text(L + 16, y + 30, n, SMALL, ACCENT, "700")
    text(L + 16, y + 48, name, SMALL, MUTED, "700")

# 01 -- what it is, and the labels that say what it does not touch
y = ys[0]
ly = y + 34
for i, ln in enumerate(what_lines):
    text(SECX, ly + i * 26, ln, HEAD, INK, "600")
by = ly + HEAD + 26 * (len(what_lines) - 1) + 22
for row in badges:
    by += 30
    for x, s, solid, w_ in row:
        pill(x, by, s, SMALL, INK if solid else MUTED, solid, bg=CARD2, stroke=LINE)

# 02 -- the models this release runs, and the cards each one runs on
y = ys[1]
for i, (name, sub, bl, n, nl) in enumerate(cards):
    x = SECX + (i % COLS) * (CW + 14)
    p = y + 26 + (i // COLS) * (CARDH + 14)        # this card's panel top; the row below starts 14 px on
    rect(x, p, CW, CARDH, CARD2, rx=10, stroke=LINE)
    text(x + 12, p + 26, name, HEAD, INK, "700")
    text(x + 12, p + 46, sub, SMALL, TAG)
    ly = p + 74
    for j, ln in enumerate(bl):
        text(x + 12, ly, ("·  " if j == 0 else "   ") + ln, SMALL, SOFT)
        ly += 20
    ly += 22
    for j in range(n):
        rect(x + 12 + j * 16, ly - 13, 11, 13, TAG, rx=2)
    text(x + 18 + n * 16, ly, f"TP{n} · {n} cards", BODY, INK, "700")
    for ln in nl:
        ly += 20
        text(x + 12, ly, ln, SMALL, MUTED)

# 03 -- the bars, and what the same four cards do at scale
y = ys[2]
text(SECX, y + 30, "single-stream decode, tok/s", BODY, INK, "700")
for i, (name, sub, v, note) in enumerate(bar_rows):
    ry = y + 64 + i * 86
    text(SECX, ry, name, BODY, INK, "700")
    text(SECX, ry + 18, sub, SMALL, MUTED)
    rect(SECX, ry + 28, BARLEN * v / BARMAX, 18, BAR, rx=7)
    text(SECX + BARLEN + 12, ry + 44, f"{v}", 26, INK, "800")
    text(SECX + BARLEN + 12 + wid(f"{v}", 26) + 10, ry + 44, "tok/s", SMALL, MUTED)
    text(SECX, ry + 62, note, SMALL, MUTED)
text(FACTX, y + 30, "aggregate · prefill · context", BODY, INK, "700")
text(FACTX, y + 48, UNCAP, SMALL, MUTED)
fy = y + 56
for v, lines in facts:
    fy += 22
    text(FACTX, fy, v, BODY, INK, "700")
    for ln in lines:
        fy += 18
        text(FACTX, fy, ln, SMALL, SOFT if ln != UNCAP else MUTED)
    fy += 12

# ---------------------------------------------------------------- footer: which machine each number came from
fl = []
for f in FOOT:
    fl += wrap(f, SMALL, W - 2 * PAD)
if len(fl) != foot_lines:
    warn.append(f"footer: laid out {len(fl)} lines, the figure reserves {foot_lines}")
for i, ln in enumerate(fl):
    text(PAD, H - 16 - 18 * (len(fl) - 1 - i), ln, SMALL, MUTED)
out.append("</svg>\n")

with open(OUT, "w") as f:
    f.write("\n".join(out))
print(f"{OUT} written: 3 sections, {W} x {H}: header {HEADH}, sections {SEC1H} / {SEC2H} / {SEC3H} "
      f"(cards {CARDH}, bars {BARH}, facts {FACTH}, {badge_rows} badge rows), footer {foot_lines} lines")
for w_ in dict.fromkeys(warn):
    print("OVERFLOW: " + w_, file=sys.stderr)
