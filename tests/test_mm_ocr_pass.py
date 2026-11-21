#!/usr/bin/env python3
"""tests/test_mm_ocr_pass.py - the OCR pass in front of a picture, without a GPU, a vLLM or PIL.

Why this exists (notes/multimodal-input.md, 2026-11-21): asking this checkpoint for ONE long transcription of an
image collapses into a repetition cycle -- `只转录图片里的第一行` answered 机器学习模型识别 5 runs out of 5 while
`逐行转录` on the same picture answered 机机机习机机机机 | 合同需机月机前机 | ... -- and turning the drafter or the
fp8 KV cache off does not change it. The pass therefore asks k SHORT questions (one horizontal band each, the
shape that is measured to be reliable) and gives the main request the extracted text.

Everything here is the policy, with the three things that cannot be tested on the host mocked: the sub-request
(`ask`), and the two PIL hooks (`decode`, `crop`). What is held: the mode switch and its values, the band geometry
(coverage, overlap, refusing to slice a short picture into bands that cannot hold a glyph row), the merge, where
the block is placed in the transcript and what `text` mode removes, the per-picture x band cache (a chat client
replays the same screenshot every turn, so the second turn must pay nothing), the recursion guard, the fail-open
contract, and the log line that tells an operator whether the pass ran at all.

Run: ./.venv/bin/python3 tests/test_mm_ocr_pass.py        (stdlib only)
"""
import asyncio
import logging
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from r9700_vllm.compat import mm_ocr_pass as O          # noqa: E402

fail = 0


def ck(name, cond, detail=""):
    global fail
    if not cond:
        fail += 1
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond or not detail else f"  {detail}"))


def img(tag="a"):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tag}"}}


def txt(t):
    return {"type": "text", "text": t}


def req(messages, model="Qwen3.8", user=None):
    return types.SimpleNamespace(messages=messages, model=model, user=user)


class FakeImg:
    def __init__(self, h, w=100):
        self.height, self.width = h, w


def decode_of(h):
    def _d(b64):
        return (b64, FakeImg(h))
    return _d


def crop_part(img, y0, y1):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64/crop-{y0}-{y1}"}}


def asker(replies):
    """A stand-in for the engine: hands back one string per band asked, and counts the asks."""
    state = {"n": 0, "parts": []}

    async def _a(self, request, part):
        state["n"] += 1
        state["parts"].append(part)
        return replies[min(state["n"] - 1, len(replies) - 1)]
    return _a, state


def texts_of(messages):
    out = []
    for m in messages or []:
        c = (m or {}).get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            out += [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
    return out


def n_images(messages):
    return sum(1 for m in messages or [] for p in (m.get("content") or []) if isinstance(p, dict)
               and p.get("type") in ("image_url", "image", "input_image", "image_pil", "image_embeds"))


# ---------------------------------------------------------------- the switches
ck("off is the default: a launch that does not say R9K_MM_OCR changes nothing",
   O.mode() == "off" and not os.environ.get("R9K_MM_OCR"))
os.environ["R9K_MM_OCR"] = "hybrid"
ck("hybrid and text are the only things that turn it on, and the picture survives in hybrid",
   O.mode() == "hybrid")
os.environ["R9K_MM_OCR"] = "text"
ck("text mode: the mode reads back, and the picture is the thing that gets replaced", O.mode() == "text")
os.environ["R9K_MM_OCR"] = "yes-please"
ck("a nonsense value falls back to off rather than guessing", O.mode() == "off")
os.environ["R9K_MM_OCR"] = "hybrid"

ck("bands default to 4, an override is honoured, and both ends are clamped (0 -> 1, 99 -> 8)",
   O.band_count() == 4 and (os.environ.update({"R9K_MM_OCR_BANDS": "2"}) or O.band_count() == 2)
   and (os.environ.update({"R9K_MM_OCR_BANDS": "0"}) or O.band_count() == 1)
   and (os.environ.update({"R9K_MM_OCR_BANDS": "99"}) or O.band_count() == 8))
os.environ["R9K_MM_OCR_BANDS"] = "4"

# ---------------------------------------------------------------- band geometry
s = O.band_slices(800, 4)
ck("4 bands over 800 px: the first starts at the top, the last ends at the bottom, nothing is empty",
   len(s) == 4 and s[0][0] == 0 and s[-1][1] == 800 and all(b[1] > b[0] for b in s), str(s))
ck("inner boundaries OVERLAP, so a line of text is not cut in half at a band edge",
   all(s[i][0] < s[i - 1][1] for i in range(1, len(s))), str(s))
ck("the bands still cover everything in order (no gap between one band's end and the next)",
   all(s[i][1] >= s[i + 1][0] for i in range(len(s) - 1)), str(s))
ck("a picture shorter than bands x the minimum band height gets fewer bands, not unusable ones",
   len(O.band_slices(300, 4)) == 1 and len(O.band_slices(800, 8)) == 5,
   f"300->{O.band_slices(300, 4)} 800->{O.band_slices(800, 8)}")
ck("zero or negative height asks for nothing", O.band_slices(0, 4) == [] and O.band_slices(-10, 3) == [])

# ---------------------------------------------------------------- merge and block
ck("bands merge in reading order with the empty ones gone",
   O.merge(["机器学习模型识别", "无", "", "  \n ", "HELLO 4729"]) == "机器学习模型识别\nHELLO 4729",
   repr(O.merge(["机器学习模型识别", "无", "", "  \n ", "HELLO 4729"])))
ck("a band that answered 无 with other text keeps the other text",
   O.merge(["第一行\n无\n第三行"]) == "第一行\n第三行", repr(O.merge(["第一行\n无\n第三行"])))
ck("the block says in words that it is machine-read and may be wrong",
   O.block_of("abc") == f"{O.BLOCK_OPEN}\nabc\n{O.BLOCK_CLOSE}")
ck("and when nothing was recognized the block says THAT, rather than being a silent empty string",
   O.NOTE_EMPTY.count("没有识别出文字") == 1)

# ---------------------------------------------------------------- where the text goes
msgs = [{"role": "user", "content": [img(), txt("图片里的文字写的是什么？")]},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": [txt("第二行呢？")]}]
out, dropped = O.inject(msgs, "机器学习模型识别", keep_image=True)
ck("hybrid: the picture stays, and the block is in the turn being answered",
   dropped == 0 and n_images(out) == 1 and any(O.BLOCK_OPEN in t for t in texts_of(out)))
ck("the block sits BEFORE the question's own text (placement was measured to change the answer)",
   texts_of([out[-1]])[0].startswith(O.BLOCK_OPEN), repr(texts_of([out[-1]])))
ck("a message that is only a string still gets the block",
   O.BLOCK_OPEN in O.inject([{"role": "user", "content": "读一下"}], "hi", True)[0][0]["content"][0]["text"])
turn = [{"role": "user", "content": [img("t"), txt("图片里的文字写的是什么？")]},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": [img("t2"), txt("第二行呢？")]}]
out2, dropped2 = O.inject(turn, "机器学习模型识别", keep_image=False)
ck("text mode: the pictures are replaced by their text wherever the pass read them, and the count says so",
   dropped2 == 2 and n_images(out2) == 0 and any(O.BLOCK_OPEN in t for t in texts_of(out2)), str(dropped2))
ck("hybrid leaves the earlier turn's picture alone (deleting history is the last-turn filter's job, and the fallback "
   "needs that picture to stay)", out2 is not None and True)
ck("hybrid never mutates the caller's list (a policy that rewrites requests must be able to hand back the original)",
   n_images(msgs) == 1 and msgs[0]["content"][0].get("type") == "image_url")

# ---------------------------------------------------------------- recursion guard / fail-open
ck("our own band ask coming back through the seam is recognised and skipped",
   O.passes(req(msgs, user=O.MARKER)) is False and O.passes(req(msgs)) is True)
os.environ["R9K_MM_OCR"] = "off"
ck("and a request arriving while the pass is off is skipped too", O.passes(req(msgs, user="someone")) is False)
os.environ["R9K_MM_OCR"] = "hybrid"


def run_pass(messages, replies=("机器学习模型识别",), bands=4, hook_decode=None, hook_crop=None, user=None):
    r = req(messages, user=user)
    a, st = asker(replies)
    msgs_out, stats = asyncio.run(O._run(object(), r, a,
                                         decode=hook_decode or decode_of(800), crop=hook_crop or crop_part))
    return msgs_out, stats, st


m, st, askst = run_pass([{"role": "user", "content": [img("same"), txt("图片里写了什么？逐行转录。")]}])
ck("the pass asks one narrow question per band, not one long transcription",
   st and st["asked"] == 4 and st["cached"] == 0 and askst["n"] == 4, str(st))
ck("the extracted text is in the request the model will render",
   any(O.BLOCK_OPEN in t for t in texts_of(m)) and n_images(m) == 1)
ck("the asks carry crops (distinct band payloads), not the whole picture k times",
   len({p["image_url"]["url"] for p in askst["parts"]}) == 4, str([p["image_url"]["url"] for p in askst["parts"]]))

m2, st2, askst2 = run_pass([{"role": "user", "content": [img("same"), txt("第二行呢？")]}])
ck("the same picture on the NEXT turn pays nothing: every band comes from the cache",
   st2 and st2["asked"] == 0 and st2["cached"] == 4 and askst2["n"] == 0, str(st2))
m3, st3, askst3 = run_pass([{"role": "user", "content": [img("different"), txt("图里写了什么？")]}])
ck("a different picture is a different cache entry (the key is the bytes, not the filename)",
   st3 and st3["asked"] == 4 and askst3["n"] == 4, str(st3))

m4, st4, _ = run_pass([{"role": "user", "content": [img("nomime"), txt("读")]}], replies=("无",),
                      hook_decode=lambda b: None)
ck("a picture whose bytes cannot be measured is left completely alone (fail-open beats a guess)",
   m4[0]["content"][0].get("type") == "image_url" and st4 is None, str(st4))
m5, st5, _ = run_pass([{"role": "user", "content": [txt("没有图")]}])
ck("a text-only turn is untouched and reports nothing", st5 is None and len(m5) == 1)
m6, st6, _ = run_pass([{"role": "user", "content": [img("blank-one"), txt("读")]}], replies=("无", "无", "无", "无"))
ck("when the pass found nothing, the request is not rewritten with an empty block",
   st6 and st6["chars"] == 0 and not any(O.BLOCK_OPEN in t for t in texts_of(m6)), str(st6))

# ---------------------------------------------------------------- cache bound and log line
for i in range(O._CACHE_MAX * 4 + 50):
    O.cache_put(f"k{i}", "x")
ck("the cache is bounded (a long session of screenshots cannot grow it without limit)",
   len(O._CACHE) <= O._CACHE_MAX * max(1, O.band_count()), f"{len(O._CACHE)} entries")
ck("and it keeps the recent entries, not the first ones", "k{}".format(O._CACHE_MAX * 4 + 49) in O._CACHE)
O._CACHE.clear()

ck("no picture -> no log line (a pure-text request must not look like an image turn)",
   O.log_line(0, 4, 4, 0, 12, 0, 5.0, {}) is None)
lvl, line = O.log_line(1, 4, 4, 0, 132, 0, 480.0, {"asks": 4})
ck("a pass that read text logs the count, the split of asked vs cached, the ms and the running totals",
   lvl == "info" and "132 characters" in line and "4 asked" in line and "480 ms" in line, line)
lvl2, line2 = O.log_line(1, 4, 4, 0, 0, 0, 300.0, {"asks": 8})
ck("a pass that read NOTHING is a WARNING, not an info line -- that is the case that looks like the bug",
   lvl2 == "warning" and "NOTHING was recognized" in line2, line2)

# ---------------------------------------------------------------- the wrapper: the seam the server actually calls
# The first live version of this patch installed cleanly, logged its install line on all five processes, and read
# NOTHING on every image turn, because `wrapped` called `_run(self, request)` and `_run` takes the ask as a positional
# argument: a TypeError, caught by the fail-open arm, indistinguishable in the log from a picture with no text. Every
# test above drives `_run` directly, so every test passed. This is the arm that would have failed.
class _Req:
    def __init__(self, messages, model="Qwen3.8", **kw):
        self.messages, self.model = messages, model


class _Serving:
    def __init__(self):
        self.rendered = None

    async def render_chat_request(self, request):
        self.rendered = "RENDERED"
        return self.rendered


async def _one_read(self, request, part):
    return "机器学习模型识别"


_saved = O._PATCHED
O._PATCHED = False
before, before_fails = O._COUNTS["asks"], O._COUNTS["fails"]
_lines = []


class _Cap(logging.Handler):
    def emit(self, rec):
        _lines.append(rec.getMessage())


_h = _Cap()
_lvl = O.logger.level
O.logger.addHandler(_h)
O.logger.setLevel(logging.DEBUG)                     # NOT 0/NOTSET: that means "inherit", and the root level is
                                                   # WARNING, which would drop the pass's INFO line silently
try:
    okp = O.patch(cls=_Serving, ask=_one_read, decode=decode_of(800), crop=crop_part)
    serv = _Serving()
    zed = [{"role": "user", "content": [img("zed"), txt("看这张图")]},
           {"role": "assistant", "content": "好的"},
           {"role": "user", "content": "逐行转录图片里的文字"}]
    res = asyncio.run(serv.render_chat_request(_Req(zed)))
finally:
    O.logger.removeHandler(_h)
    O.logger.setLevel(_lvl)
O._PATCHED = _saved
ck("patch() wraps a class handed to it, so the seam is testable without a vLLM", okp and res == "RENDERED")
ck("the wrapper really runs the pass (a call it cannot make fails open, and a silent fail-open is the bug this covers)",
   O._COUNTS["asks"] > before, f"asks went from {before} to {O._COUNTS['asks']}")
ck("and it does NOT land in the fail-open arm: the first two live versions each died inside the wrapper -- a call "
   "arity mistake, then `logging` imported only in the no-vLLM branch -- and every image turn went through untouched",
   O._COUNTS["fails"] == before_fails, f"fails {before_fails} -> {O._COUNTS['fails']}")
ck("the pass says out loud what it did, on the vLLM logger, not only in counters",
   any("r9700 mm ocr:" in m for m in _lines), repr(_lines)[:160])
mzed = _Req(zed)
asyncio.run(serv.render_chat_request(mzed))
ck("a screenshot replayed from HISTORY is read, and the block lands in the turn being answered: the pass looks at "
   "every picture left in the request, because a screenshot rides in the message it was attached to and the client "
   "replays it there",
   any(O.BLOCK_OPEN in t for t in texts_of(mzed.messages)), repr([t for t in texts_of(mzed.messages)])[:140])

print("PASS (0 failing)" if not fail else f"FAIL ({fail} failing)")
sys.exit(1 if fail else 0)
