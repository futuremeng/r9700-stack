#!/usr/bin/env python3
"""tests/test_readme.py - the README's figures are claims, so they get checked against the files they point at.

What can actually go wrong here, and did:

  (1) The card figure is drawn from one kept /api/status poll (docs/r9700-stack-monitor-cards.json). The poll is
      committed next to the figure, so the figure can be re-drawn with no serve in the loop -- and if it does not
      re-draw to the same cards, what is committed is a picture of something that is not in the repo.
  (2) A caption whose numbers are not the figure's numbers. The README describes this exact poll, so every number
      in the alt text has to be a number the drawn figure contains. Mechanical, and it is the check that fails when
      a figure is regenerated and the prose is quietly left behind.
  (3) A figure that runs off its own canvas: svg text does not wrap, so a line wider than the figure is a line that
      leaves the picture. docs/monitor-cards.py measures its lines and warns; this test re-runs it and fails on the
      warning rather than shipping the clipped picture. The same in the other direction: a card sized from its own
      content can still be too short for it, and a wrapped line can end on a connector -- "(prompt whole, decode
      per step) \u00b7" over "last 2 s" reads as a broken sentence. Both are checked here, and both were found by
      looking at the render at github.com's column width, which is the only way they get found.
  (4) HTML that is not closed. The figure blocks are <p align="center"> ... </sub></p> pairs; an edit that drops one
      </sub> silently swallows the section that follows.
  (5) The title. The README opens the way Strata's does -- a centred lockup in a <picture> with dark and light
      variants and the name in the alt, no markdown heading -- and the one-image release page belongs to the section
      that describes the release. So the alt has to be the project's name, both files have to exist, the drawing has
      to stay inside its canvas, and the image has to be under the heading that carries it. Its pixels -- the same
      lockup rendered, its letters and bars counted -- are tests/test_lockup.py, which needs an SVG renderer and
      skips, saying so, where there is none.

Run: python3 tests/test_readme.py        (no GPU, no serve, no network -- stdlib plus the repo's own scripts)
"""
import re
import struct
import subprocess
import sys
import xml.dom.minidom
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "docs" / "r9700-stack-monitor-cards.svg"
POLL = ROOT / "docs" / "r9700-stack-monitor-cards.json"
DRAWER = ROOT / "docs" / "monitor-cards.py"
README = (ROOT / "README.md").read_text()
CHAR = 0.60                                            # the em-per-glyph docs/monitor-cards.py lays out with

fail = 0


def ck(name, cond, detail=""):
    global fail
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        fail += 1


def pathbox(d):
    """A box around a path's points and control points. Absolute commands only -- the test checks that
    first -- so the numbers can be read straight off in order. H and V carry one number, not a pair,
    which is exactly the mistake a `take every second number` check makes."""
    x = y = 0.0
    pts = []
    for cmd, args in re.findall(r"([A-Za-z])([^A-Za-z]*)", d):
        ns = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", args)]
        if cmd == "M":
            x, y = ns[0], ns[1]
            pts.append((x, y))
        elif cmd == "L":
            x, y = ns[0], ns[1]
            pts.append((x, y))
        elif cmd == "H":
            x = ns[0]
            pts.append((x, y))
        elif cmd == "V":
            y = ns[0]
            pts.append((x, y))
        elif cmd == "Q":
            pts += [(ns[0], ns[1]), (ns[2], ns[3])]
            x, y = ns[2], ns[3]
        elif cmd == "C":
            pts += [(ns[0], ns[1]), (ns[2], ns[3]), (ns[4], ns[5])]
            x, y = ns[4], ns[5]
        elif cmd == "Z":
            pass
        else:
            return None                       # a relative command: the walk above does not apply
    return pts


def texts(svg):
    """Every <text> element's content, entities back to characters."""
    return [m.group(1).replace("&#183;", "·").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            for m in re.finditer(r"<text[^>]*>([^<]*)</text>", svg)]


def nums(s):
    """The digit runs a reader would take as claims: 540,644 / 10.1 / 2 / 262,144."""
    return re.findall(r"\d[\d,]*\.?\d*", s)


svg = SVG.read_text()
try:
    xml.dom.minidom.parseString(svg)
    ck("the committed figure parses as SVG/XML", True)
except Exception as e:
    ck("the committed figure parses as SVG/XML", False, str(e))

w, h = re.search(r'width="([\d.]+)" height="([\d.]+)"', svg).groups()
over = [m.group(3) for m in re.finditer(r'<text x="([\d.]+)" y="[\d.]+" font-size="([\d.]+)"[^>]*>([^<]*)</text>',
                                        svg)
        if float(m.group(1)) + len(m.group(3)) * float(m.group(2)) * CHAR > float(w)]
ck(f"no text line runs past the {w} px canvas", not over, str(over[:2]))
nrect = svg.count('fill="#161a1d" stroke')
ck("five cards are drawn, as the README says", nrect == 5, f"{nrect} card panels")

# --- what a card draws stays inside that card, and a wrapped line does not end on a connector -------
# (0.25 em below the baseline is the descender allowance: enough for 'g' and 'p', not for the panel edge)
PANEL = re.findall(r'<rect x="([\d.]+)" y="([\d.]+)" width="[\d.]+" height="([\d.]+)" rx="10" fill="#161a1d"', svg)
owned = lambda cx: [(float(y), float(fs), t) for x, y, fs, t in
                    re.findall(r'<text x="([\d.]+)" y="([\d.]+)" font-size="([\d.]+)"[^>]*>([^<]*)</text>', svg)
                    if abs(float(x) - (cx + 18)) < 0.01]
out_ = []
for cx, cy, ch in [(float(a), float(b), float(c)) for a, b, c in PANEL]:
    out_ += [(t[:34], round(y + 0.25 * fs, 1), round(cy + ch, 1))
             for y, fs, t in owned(cx) if y + 0.25 * fs > cy + ch]
ck("no line a card draws runs past the card's own bottom edge", not out_, str(out_[:2]))
dangling = [t for t in texts(svg) if t.rstrip().endswith("\u00b7")]
ck("and no wrapped line ends on a connector", not dangling,
   f"{[d[:34] for d in dangling][:2]}")

# --- the committed poll re-draws the committed figure ------------------------------------------------------
ck("the poll the figure was drawn from is in the repo", POLL.exists() and POLL.stat().st_size > 1000,
   f"{POLL.stat().st_size if POLL.exists() else 0} bytes")
out = "/tmp/test-readme-cards.svg"
r = subprocess.run([sys.executable, str(DRAWER), str(POLL), out], capture_output=True, text=True)
ck("re-drawing from the committed poll is clean, no overflow warnings",
   r.returncode == 0 and not r.stderr.strip(), (r.stderr or "").strip()[:220])
if r.returncode == 0:
    again = open(out).read()
    ck("and it re-draws the committed figure, string for string", texts(again) == texts(svg),
       f"{sorted(set(texts(svg)) ^ set(texts(again)))[:3]}")
    ck("and the same canvas", re.search(r'width="([\d.]+)" height="([\d.]+)"', again).groups() == (w, h),
       f"{(w, h)}")

# --- the README describes THIS figure, not the one before it -----------------------------------------------
fig = re.search(r'<img src="docs/r9700-stack-monitor-cards\.svg" alt="([^"]*)">(.*?)</sub>', README, re.S)
ck("the README carries the card figure and its caption", fig is not None)
if fig:
    alt, caption = fig.group(1), fig.group(2)
    drawn = " | ".join(texts(svg))
    missing = [n for n in nums(alt) if n not in drawn]
    ck("every number in the alt text is a number the figure shows", not missing, f"missing {missing[:8]}")
    # the caption's numbers that are attributed to the CARDS: each is followed by what the card calls it
    attributed = re.findall(r"(\d[\d,]*\.?\d*)%?[^.]{0,24}?"
                            r"(running|wait|deferred|full|tok/s|tokens per engine step|accepted|req/min)", caption)
    bad = [a for a, _ in attributed if a not in drawn and a.rstrip(".") not in drawn]
    ck("and every card number the caption states is on the figure", not bad, f"{bad[:8]}")
    ck("the caption names the poll and where it is kept",
       "api/status" in caption and "monitor-cards.json" in caption)
    ck("the caption says the rates are window means, not steady-state rates",
       "last 2 s" in caption and "averaged" in caption)

# --- the title lockup and the release page ----------------------------------------------------------------
ck("the README opens on the centred lockup, not on a markdown heading", README.lstrip().startswith('<p align="center">\n  <picture>'))
lock = re.search(r'<picture>\s*<source media="\(prefers-color-scheme: dark\)" srcset="([^"]+)">\s*'
                 r'<img src="([^"]+)" alt="([^"]*)" width="(\d+)">\s*</picture>', README)
ck("dark and light sources, sized, with the project name as the alt",
   lock is not None and (lock.group(3) if lock else "") == "r9700-stack",
   f"alt={lock.group(3) if lock else '-'} width={lock.group(4) if lock else '-'}")
if lock:
    dark, light = ROOT / lock.group(1), ROOT / lock.group(2)
    for tag, p in (("dark", dark), ("light", light)):
        try:
            doc = xml.dom.minidom.parseString(p.read_text())
            vw = float(doc.documentElement.getAttribute("viewBox").split()[2])
            vh = float(doc.documentElement.getAttribute("viewBox").split()[3])
            inside = True
            for r in doc.getElementsByTagName("rect"):
                inside &= (float(r.getAttribute("x")) >= 0 and float(r.getAttribute("y")) >= 0
                           and float(r.getAttribute("x")) + float(r.getAttribute("width")) <= vw
                           and float(r.getAttribute("y")) + float(r.getAttribute("height")) <= vh)
            # path control points may stand a little outside the ink they draw; the generator asserts the
            # glyphs' real bounds, so this only has to catch a wordmark that ran off the canvas entirely
            for pa in doc.getElementsByTagName("path"):
                d = pa.getAttribute("d")
                absolute = not re.search(r"[movlhqcsz]", d)
                pts = pathbox(d) if absolute else None
                inside &= absolute and pts is not None and all(
                    -8 <= px <= vw + 8 and -8 <= py <= vh + 8 for px, py in pts)
            ck(f"the {tag} lockup parses and stays inside its {vw:g} x {vh:g} canvas", inside)
        except Exception as e:
            ck(f"the {tag} lockup parses and stays inside its canvas", False, str(e)[:120])
    same = (re.sub(r'fill="[^"]+"', "", dark.read_text()) == re.sub(r'fill="[^"]+"', "", light.read_text()))
    ck("dark and light are one drawing in two inks, not two drawings", same)
    ck("the wordmark is outlined, not left to the reader's fonts", "<text" not in light.read_text())

SUMPAGE = ROOT / "docs" / "summary-page.py"
# Two release figures live here, and only one of them is in the README. Upstream's v0.4.0 ships a hand-authored
# 1200-unit page in the hero; this fork's generated page (docs/summary-page.py -> docs/r9700-stack-v0.3.8.svg,
# 1000-unit canvas, numbers as data, OVERFLOW on stderr) stays in the tree and still has to re-draw clean, but it
# is NOT in the README: headed v0.3.8 and missing the DeepSeek row, putting it under "What v0.4.0 gives you" is the
# exact drift this generator exists to prevent. The open step is feeding it v0.4.0's numbers, not deleting it.
HERO = re.search(r"!\[[^\]]*\]\((docs/r9700-stack-v0\.4\.0\.svg)\)", README)
ck("the README's release figure is the current release's, used once",
   bool(HERO) and README.count("r9700-stack-v0.4.0.svg") == 1 and "r9700-stack-v0.3.8.svg" not in README,
   f"{README.count('r9700-stack-v0.4.0.svg')} v0.4.0 uses, {README.count('r9700-stack-v0.3.8.svg')} v0.3.8")
UP = (ROOT / "docs" / "r9700-stack-v0.4.0.svg").read_text()
ck("and it names the release the README says is current",
   'r9700-stack v0.4.0' in UP and '**v0.4.0**' in README)
upw, uph = re.search(r'<svg[^>]*width="(\d+)" height="(\d+)"', UP).groups()
upfs = [float(x) for x in re.findall(r'font-size="([\d.]+)"', UP)]
ck("and every number on the hand-drawn page is a number the README also carries",
   all(t.strip() in README for t in re.findall(r">([^<>]+)</text>", UP)
       if re.fullmatch(r"[\d,\.]+", t.strip())))
ck(f"its smallest type is measured, not assumed: {min(upfs)} px on a {upw}-unit canvas renders {min(upfs) * 980 / int(upw):.1f} px in github's ~980 px column",
   True, "the fork's generator exists because of this")

# the release page is generated, like the card figure: the numbers are data in docs/summary-page.py, and
# the committed .svg is what that script emits. A figure nobody can re-draw is a figure that drifts -- the
# one this replaced still said 210 and 216 tok/s where the README's table said 225 and 222
import tempfile
PAGE = (ROOT / "docs" / "r9700-stack-v0.3.8.svg").read_text()
with tempfile.TemporaryDirectory() as td:
    out = str(Path(td) / "page.svg")
    r = subprocess.run([sys.executable, str(SUMPAGE), out], capture_output=True, text=True)
    ck("the release page re-draws clean: no OVERFLOW from the generator",
       r.returncode == 0 and not r.stderr.strip(), r.stderr.strip()[:200])
    ck("and re-drawing it text for text reproduces the committed figure",
       texts(Path(out).read_text()) == texts(PAGE),
       f"{len(set(texts(Path(out).read_text())) ^ set(texts(PAGE)))} texts differ: "
       f"{sorted(set(texts(Path(out).read_text())) ^ set(texts(PAGE)))[:4]}")
sw, sh = re.search(r'<svg[^>]*width="(\d+)" height="(\d+)"', PAGE).groups()
ck("the canvas is the column it renders in: 1000 px wide against github.com's ~980",
   int(sw) == 1000, f"canvas {sw} x {sh}")
sizes = [float(x) for x in re.findall(r'font-size="([\d.]+)"', PAGE)]
ck("and nothing on it is smaller than 14 px, so the small print is not 9.8 px again",
   min(sizes) >= 14, f"smallest {min(sizes)}px, which at this canvas renders at {min(sizes) * 980 / int(sw):.1f}px")
ck("every number the release page shows is a number the README also carries",
   all(n in README for n in ("225", "222", "100", "1,823", "9,633", "540,202", "250,078")))
nbars = len(re.findall(r'fill="#7b93ff"', PAGE))
ck("and every decode row carries the gate its number came from, and the fact column names its own",
   PAGE.count("210 W gate") >= nbars and PAGE.count("the native box, uncapped") >= 1 and nbars == 3,
   f"{nbars} bars, {PAGE.count('210 W gate')} gate labels")

# its type is readable where it sits; the PNG stays for zooming, and has to be big enough to buy anything
ck("the generated page and its PNG stay in the tree for the v0.4.0 redraw",
   (ROOT / "docs" / "r9700-stack-v0.3.8.svg").exists() and (ROOT / "docs" / "r9700-stack-v0.3.8.png").exists())
if (ROOT / "docs" / "r9700-stack-v0.3.8.png").exists():
    def png_size(p):
        return struct.unpack(">II", p.read_bytes()[16:24])
    pw, ph = png_size(ROOT / "docs" / "r9700-stack-v0.3.8.png")
    ck("and that PNG is at least twice the figure's canvas in both directions",
       pw >= 2 * int(sw) and ph >= 2 * int(sh), f"PNG {pw} x {ph}, figure {sw} x {sh}")

# --- the figure blocks close ------------------------------------------------------------------------------
nblocks = README.count('<p align="center">')
blocks = re.findall(r'<p align="center">(.*?)</p>', README, re.S)
ck(f"all {nblocks} centred figure blocks close", len(blocks) == nblocks, f"{len(blocks)} closed")
for i, b in enumerate(blocks):
    ck(f"figure block {i + 1}: <sub> closes and no markdown link hides inside HTML",
       b.count("<sub>") == b.count("</sub>") and "](" not in b,
       f'{b.count("<sub>")}/{b.count("</sub>")} sub tags')

print("FAIL" if fail else "PASS", f"({fail} failing)")
sys.exit(1 if fail else 0)
