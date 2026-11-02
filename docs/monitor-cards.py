#!/usr/bin/env python3
"""Draw the Monitor's headline cards as a crisp SVG, from a live GET /api/status.

    python3 docs/monitor-cards.py [BASE|PAYLOAD.json] [OUT]        # stdlib only

BASE (a URL) polls once and draws that poll. A .json path is a payload recorded by bench/capture-cards.py during a
load run, and draws the poll that was actually chosen -- polling again here would draw a different second than the
one the sampler picked, which is how a figure ends up showing a queue shape that was never measured together.

Why this exists next to the screenshot rather than instead of it: a README image lands inside github.com's
~980 px column, so the 1688 x 1953 Monitor screenshot renders at 0.58 scale and its text is unreadable. Cropping
the screenshot into a wide strip does not fix that -- the column is the limit for the whole strip, so a banner of
five panels is SMALLER per panel than the full page. So the figure is drawn to the column instead of to a screen:
1000 px wide, which lands at ~1.0 scale, and it is five cards in two rows rather than one row of five, because a
row of five at 1000 px is 190 px a card and no honest sentence fits in that.

The text is laid out to fit, not assumed to: every line is measured against its own font size (0.6 em a character,
conservative for Outfit/DejaVu) and wrapped, and a line that still cannot fit is printed as a warning to stderr
instead of running out of the card unnoticed.

It is not a screenshot and says so: the footer names the base it polled, the time of the poll, and the launch
the meta file recorded. The values are the same ones the page's own cards read, formatted the same way.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

SRC = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/r9700-stack-monitor-cards.svg"
# Which serve a recorded payload came from, for the footer. Not in the payload itself: /api/status has no
# reason to name itself.
BASE = (sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8080").rstrip("/")

BG, CARD, LINE, INK, SOFT, MUTED, ACCENT = ("#0e1113", "#161a1d", "#2a3036", "#eef1f3", "#c7cdd3",
                                            "#8e979f", "#34d399")
FONT = "Outfit, 'DejaVu Sans', system-ui, sans-serif"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


REPO = str(Path(__file__).resolve().parent.parent) + "/"


def shown(path):
    """A path as the figure prints it: relative to this repo when it lies inside the repo, so the footer names a
    file the reader can open in the checkout instead of one machine's absolute location."""
    full = str(Path(path).resolve())
    return full[len(REPO):] if full.startswith(REPO) else path


def num(v, d=0):
    return f"{v:,.{d}f}" if isinstance(v, (int, float)) and v == v else "—"


def pct(v, d=1):
    return f"{v:.{d}f}%" if isinstance(v, (int, float)) and v == v else "—"


if SRC.startswith("http"):
    BASE = SRC.rstrip("/")
    with urllib.request.urlopen(BASE + "/api/status", timeout=10) as r:
        s = json.load(r)
    how, kept = "drawn from a live GET /api/status, not screenshotted", None
else:
    with open(SRC) as f:
        s = json.load(f)
    # "the poll the sampler kept" is the whole point of a .json source: re-polling here would draw some other
    # second, and a figure that shows a queue shape never measured in one poll is a figure made of two polls.
    how, kept = "drawn from the /api/status poll the sampler kept, not screenshotted", shown(SRC)
live, rate, meta = s.get("live") or {}, s.get("rate") or {}, s.get("meta") or {}

recipe = " · ".join(x for x in [
    f"TP{meta['tp']}" if meta.get("tp") else "",
    f"MAXLEN {num(int(meta['maxlen']))}" if meta.get("maxlen") else "",
    f"NSEQ {meta['nseq']}" if meta.get("nseq") else "",
    f"NBT {num(int(meta['nbt']))}" if meta.get("nbt") else "",
    "prefix cache off" if meta.get("prefix_cache") == "0" else
    ("prefix cache on" if meta.get("prefix_cache") == "1" else "")] if x)

waiting = f"{num(live.get('waiting'))} waiting"
if live.get("waiting_capacity") is not None:
    waiting += f" · {num(live.get('waiting_capacity'))} for scheduling capacity"
if live.get("waiting_deferred") is not None:
    waiting += f" · {num(live.get('waiting_deferred'))} deferred"

# The same pick() the page uses: a window value if this poll has one, else the cumulative one, labelled.
since = s.get("since") or {}
win = rate.get("window_s")
WIN = f" · last {win:.0f} s" if isinstance(win, (int, float)) and win == win else " · this poll"


def pick(k):
    """A value plus the span it was computed over.

    On the page a window value carries no label, because the page states its own poll window a few cards away. In a
    frozen figure there is nowhere else to look: an unlabelled mean over an unstated span is a number the reader
    cannot check, and "2 tokens per iteration record" next to "9,621 tokens per iteration record since launch" is
    the pair that gets read backwards."""
    v = rate.get(k)
    if isinstance(v, (int, float)) and v == v:
        return v, WIN
    w = since.get(k)
    return (w, " · since launch") if isinstance(w, (int, float)) and w == w else (None, "")


tpi, tpi_lbl = pick("tokens_per_iter")
accept, acc_lbl = pick("accept_pct")
per_draft, pd_lbl = pick("accepted_per_draft")
# acceptance rate and accepted-tokens-per-draft are two readings of the same two counters, so they are always on
# the same span and one basis line covers the pair. If they ever disagree, both get said.
spec_lbl = acc_lbl if acc_lbl == pd_lbl else f"{acc_lbl}{pd_lbl}"

# (label, big value, [(line, kind)]) -- kind picks the size: soft is a card's own sub line, faint is the
# small print under it (the recipe, the basis a number was computed on).
cards = [
    ("Model", s.get("model_name") or "—", [(f"context {num(int(meta['maxlen']))} tokens" if meta.get("maxlen")
                                           else "max context not in the recipe file", "soft"),
                                           (recipe, "faint")]),
    ("KV pool", pct(live.get("kv_used_pct")), [(f"pool {num(live.get('kv_tokens'))} tokens", "soft"),
                                               ("used now, not reserved", "faint")]),
    ("Requests", num(live.get("running")) + " running", [(waiting, "soft"),
                                                         (f"{num(rate.get('requests'), 0)} finished in the poll window",
                                                          "faint")]),
    # The iteration-record line is a Generation-card number, not a speculation one, and it is not a step size:
    # vllm:iteration_tokens_total credits a whole chunked prefill in one record. See serve/web/monitor.js for the
    # measurement that settled that phrasing.
    ("Generation", num(rate.get("generation_tok_s"), 1) + " tok/s",
     [(f"{num(rate.get('prompt_tok_s'))} prompt tok/s (credited at request end)"
       + (f" · {num(rate.get('requests_per_min'), 1)} req/min" if rate.get("requests_per_min") is not None else ""),
       "soft"),
      ((f"{num(tpi, 0)} tokens per iteration record (prompt whole, decode per step){tpi_lbl}" if tpi is not None
        else "iteration-record size not in /metrics"), "faint")]),
    ("Speculation", pct(accept, 1) + " accepted",
     [(f"{num(rate.get('drafts_per_req'), 2)} drafts per request"
       if rate.get("drafts_per_req") is not None else
       f"{num(since.get('requests'), 0)} requests finished since the engine started", "soft"),
      (f"{num(per_draft, 2)} accepted tokens per draft{spec_lbl}" if per_draft is not None
       else "no speculation counters in /metrics", "faint")]),
]

FS = {"soft": 14.5, "faint": 13.0}
W, TOP, CW, GAP, PAD, BAR, ROWGAP = 1000, 52, 320, 20, 18, 7, 20
COLS = 3                                     # 3 * 320 + 2 * 20 = 1000, full column, no margin to waste
CHAR = 0.60                                  # em per glyph, conservative for Outfit / DejaVu Sans
warn = []


def wrap(s, fs, width):
    """Break a line to fit `width` px at `fs`, at word boundaries where possible."""
    if not s:
        return []
    per = max(1, int(width / (fs * CHAR)))
    words, lines, cur = s.split(" "), [], ""
    for w_ in words:
        cand = (cur + " " + w_).strip()
        if len(cand) <= per or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    for ln in lines:                          # a single word longer than the card: say so, do not overflow
        if len(ln) > per:
            warn.append(f"{ln!r} at {fs}px needs {len(ln) * fs * CHAR:.0f}px, card holds {width}px")
    return lines


def layout(label, value, subs):
    """One card, measured: the value's font size, its wrapped sub-lines, and the height it needs.

    A fixed card height is how a figure ends with text under a bar or outside the box -- the number of
    sub-lines here depends on values this script does not control (a long model name, a 5-digit pool size,
    a queue split that arrives as three clauses), so the height comes from the content."""
    vfs = 30 if len(value) <= 12 else (24 if len(value) <= 17 else 20)
    if len(value) * vfs * CHAR > CW - 2 * PAD:
        warn.append(f"{label}: value {value!r} at {vfs}px wants {len(value) * vfs * CHAR:.0f}px, the card "
                    f"holds {CW - 2 * PAD}px")
    lines = [(ln, FS[kind], kind) for text, kind in subs for ln in wrap(text, FS[kind], CW - 2 * PAD)]
    below = sum(fs * 1.45 for _, fs, _ in lines)                    # the sub-lines' own height
    h = max(100 + below + 12, 150)
    if label == "KV pool":                                         # the pool bar sits under the text
        h = max(h, 100 + below + 16 + BAR)
    return {"label": label, "value": value, "vfs": vfs, "lines": lines, "h": h}


def wide(x, s, fs):
    """Where a line the card layout does not own -- the figure's header and footer -- would end. svg text has no
    wrapping and no overflow, so a line wider than the figure is a line that leaves the picture."""
    return x + len(s) * fs * CHAR


# The header and the footer are worked out before the height, because they are part of it: a second provenance
# line needs room, and a figure that clips its own source is worse than one that is 20 px taller.
head = f"{how} &#183; mode: {esc(s.get('mode'))}"
if wide(212, head, 14) > W:
    head = "drawn from a kept /api/status poll, not screenshotted"
    warn.append(f"header {how!r} is too long for the figure; shortened to {head!r}")
stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(s.get("at") or time.time()))
foot = [f"GET {esc(BASE)}/api/status at {stamp}"]
if meta.get("model"):
    m = f" &#183; {esc(meta.get('served_name'))} from {esc(meta.get('model'))}"
    if wide(0, foot[0] + m, 13) > W:                        # the full checkpoint path is a luxury, not a fact
        m = f" &#183; {esc(meta.get('served_name'))} from {esc(str(meta['model']).rstrip('/').split('/')[-1])}"
    foot[0] += m
if kept:                                                    # where the poll this was drawn from lives
    second = f"the poll it was drawn from: {esc(kept)}"
    if wide(0, second, 13) > W:
        warn.append(f"the footer cannot carry {SRC!r} and still fit {W}px; it is left out")
    else:
        foot.append(second)

laid = [layout(*c) for c in cards]
rows = [laid[:COLS], laid[COLS:]]
hs = [round(max(c["h"] for c in row)) for row in rows]
ys, y = [], TOP
for h in hs:
    ys.append(y)
    y += h + ROWGAP
H = ys[-1] + hs[-1] + 26 + 20 * (len(foot) - 1)
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="{FONT}">',
       f'<rect width="{W}" height="{H}" fill="{BG}"/>',
       f'<text x="0" y="30" font-size="17" font-weight="700" fill="{INK}">The Monitor\'s cards</text>',
       f'<text x="212" y="30" font-size="14" fill="{MUTED}">{head}</text>']
for row, y0, ch in zip(rows, ys, hs):
    x0 = (W - (len(row) * CW + (len(row) - 1) * GAP)) / 2          # a short second row sits centred
    for j, c in enumerate(row):
        x = x0 + j * (CW + GAP)
        out.append(f'<rect x="{x}" y="{y0}" width="{CW}" height="{ch}" rx="10" fill="{CARD}" stroke="{LINE}"/>')
        out.append(f'<text x="{x + PAD}" y="{y0 + 30}" font-size="12" letter-spacing="1.2" fill="{MUTED}">'
                   f'{esc(c["label"].upper())}</text>')
        out.append(f'<text x="{x + PAD}" y="{y0 + 72}" font-size="{c["vfs"]}" font-weight="900" '
                   f'fill="{INK}">{esc(c["value"])}</text>')
        sy = y0 + 100
        for ln, fs, kind in c["lines"]:
            out.append(f'<text x="{x + PAD}" y="{round(sy)}" font-size="{fs}" '
                       f'fill="{SOFT if kind == "soft" else MUTED}">{esc(ln)}</text>')
            sy += fs * 1.45
        if c["label"] == "KV pool":
            used = max(0.0, min(100.0, live.get("kv_used_pct") or 0))
            out.append(f'<rect x="{x + PAD}" y="{y0 + ch - BAR - 10}" width="{CW - 2 * PAD}" height="{BAR}" '
                       f'rx="{BAR / 2}" fill="{LINE}"/>')
            out.append(f'<rect x="{x + PAD}" y="{y0 + ch - BAR - 10}" width="{(CW - 2 * PAD) * used / 100:.1f}" '
                       f'height="{BAR}" rx="{BAR / 2}" fill="{ACCENT}"/>')
for i, line in enumerate(reversed(foot)):                          # the last line sits at the bottom
    out.append(f'<text x="0" y="{H - 14 - 20 * i}" font-size="13" fill="{MUTED}">{line}</text>')
out.append("</svg>\n")
with open(OUT, "w") as f:
    f.write("\n".join(out))
print(f"{OUT} written: {len(cards)} cards in {len(rows)} rows, {W} x {H}, from {BASE}")
for w_ in dict.fromkeys(warn):
    print("OVERFLOW: " + w_, file=sys.stderr)
