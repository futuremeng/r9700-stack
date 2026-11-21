#!/usr/bin/env python3
"""tests/test_mm_last_turn.py - the history-image filter and the shared pixel budget, without a GPU or a vLLM.

Two behaviors, one seam. First (2026-11-09): a chat client replays the transcript every turn, so the images of
earlier turns arrive again, and stock vLLM re-encodes all of them -- until the replay alone hits
--limit-mm-per-prompt and the turn that adds a NEW picture is refused. Only the last submitted turn's pictures may
survive, capped to the engine's own limit (newest first). Second (2026-11-21): the pictures that stay SHARE one
pixel budget -- the engine's own cap, 4,194,304 px (~4,096 tokens, the band the model reads in: notes
/multimodal-input.md has the sweep that found the ceiling and the floor) -- and from k = 2 up an oversized
picture shrinks to budget/k. The resize itself is PIL (the container's); these
tests swap in measurement and resize hooks and check the arithmetic and the fail-open contract. Third (later the
same day, and the shapes below are the client shapes that made it necessary): the strict last-turn rule left a
follow-up question about a screenshot with NO picture in the prompt and no word saying one had been removed, so
the filter now keeps the newest replayed picture when the turn being answered carries none, and every message that
loses a picture carries a text part saying so.

Run: ./.venv/bin/python3 tests/test_mm_last_turn.py        (stdlib only)
"""
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from r9700_vllm.compat import mm_last_turn as M
from r9700_vllm.compat.mm_last_turn import (strip_history_images, adapt_pixels, pixel_budget,   # noqa: E402
                                            history_keep, log_line, _is_image)

fail = 0


def ck(name, cond, detail=""):
    global fail
    if not cond:
        fail += 1
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond or not detail else f"  {detail}"))


def img(tag="a"):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tag}"}}


def turn(*parts):
    return list(parts)


# the shape that broke: three turns, two pictures in the first, one new one in the last
msgs = [{"role": "user", "content": turn({"type": "text", "text": "look"}, img("old1"), img("old2"))},
        {"role": "assistant", "content": "noted"},
        {"role": "user", "content": turn({"type": "text", "text": "and this?"}, img("new"))}]
out, kept, dropped, from_hist = strip_history_images(msgs, keep=4)
ck("history images are dropped and the new one is kept (2 dropped, 1 kept, no fallback needed)",
   (kept, dropped, from_hist) == (1, 2, 0), f"{kept} kept, {dropped} dropped, {from_hist} from history")
ck("the transcript keeps its shape and its text: three messages, the first one still says 'look'",
   len(out) == 3 and out[0]["content"][0]["text"] == "look" and out[2]["content"][1]["image_url"] is not None)
ck("a message that was only pictures does not render as an empty turn",
   strip_history_images([{"role": "user", "content": turn(img())},
                         {"role": "user", "content": turn(img())}])[1:3] == (1, 1))

# the 400 case: five pictures in the turn being answered, the engine's limit is 4
five = [{"role": "user", "content": turn(*[img(str(i)) for i in range(5)])}]
out, kept, dropped, _ = strip_history_images(five, keep=4)
ck("more pictures than the engine allows keeps the NEWEST four instead of answering 400, and leaves them where the "
   "client put them (part order is what the model reads a picture in)",
   (kept, dropped) == (4, 1)
   and [p["image_url"]["url"] for p in out[0]["content"] if _is_image(p)] == [img(i)["image_url"]["url"]
                                                                             for i in "1234"],
   f"{kept} kept, {dropped} dropped")
ck("a same-turn picture that got dropped is named in the content too -- nothing disappears in silence",
   any(p.get("text", "").startswith("[1 image(s) dropped") for p in out[0]["content"] if p["type"] == "text"))

# text-only requests must come back as the SAME object: no copy, no behavior change for the 99% case
plain = [{"role": "user", "content": "hi"}]
ck("a request with no images is returned untouched (same list object, nothing copied)",
   strip_history_images(plain, keep=4)[0] is plain)
ck("R9K_MM_LAST_TURN=0 is the way back to stock behavior (patch() installs nothing)",
   os.environ.get("R9K_MM_LAST_TURN", "1") != "0")

# ------------------------------------------------- the client shapes that made the third policy necessary ------
# Both shapes below were replayed against the live serve and are logged in notes/multimodal-input.md: the client
# replays the screenshot inside the message it was attached to, and the question is asked on the NEXT turn. The
# strict last-turn rule then sent the model an 84-token prompt where the same question with its picture is 232 --
# no picture in the prompt at all, and nothing in the transcript saying one had been removed. The model answered
# "我当前没有看到可读取的图片内容", and in other turns by invention with same-family glyphs (猫 read out as 猛, a
# sentence as 冗冗冗), which from the client's side is "the model cannot read Chinese".
def txt(t):
    return {"type": "text", "text": t}


followup = [{"role": "user", "content": turn(txt("看这张截图"), img("SHOT"))},
            {"role": "assistant", "content": "好的，我看过了。"},
            {"role": "user", "content": turn(txt("图片里的文字写的是什么？"))}]
out, kept, dropped, from_hist = strip_history_images(followup, keep=4)
ck("a follow-up turn that carries no picture keeps the newest replayed one (kept 1, nothing dropped, nothing copied)",
   (kept, dropped, from_hist) == (1, 0, 1) and out is followup, f"{kept}/{dropped}/{from_hist}")
out, kept, dropped, from_hist = strip_history_images(followup, keep=4, older=0)
urls = [p["image_url"]["url"] for m in out for p in (m["content"] if isinstance(m["content"], list) else [])
        if _is_image(p)]
ck("older=0 (R9K_MM_HISTORY_IMAGE=0) is the strict rule back -- and the message that lost the picture now says so",
   (kept, dropped, from_hist) == (0, 1, 0) and urls == []
   and any(p.get("text", "").startswith("[1 image(s) from an earlier turn") for p in out[0]["content"]),
   f"{kept}/{dropped}/{from_hist} {urls}")

two_replayed = [{"role": "user", "content": turn(txt("第一张"), img("S1"))}, {"role": "assistant", "content": "ok"},
                {"role": "user", "content": turn(txt("第二张"), img("S2"))}, {"role": "assistant", "content": "ok"},
                {"role": "user", "content": turn(txt("图里写的什么？"))}]
out, kept, dropped, from_hist = strip_history_images(two_replayed, keep=4)
ck("two replayed pictures and an empty turn: the NEWEST one survives, not the oldest, and the other is named",
   (kept, dropped, from_hist) == (1, 1, 1)
   and [p["image_url"]["url"] for p in out[0]["content"] if _is_image(p)] == []
   and [p["image_url"]["url"] for p in out[2]["content"] if _is_image(p)] == [img("S2")["image_url"]["url"]],
   f"{kept}/{dropped}/{from_hist}")

ck("older=2 keeps exactly two replayed pictures across three messages -- the counts are per message, not running "
   "totals added again (a doubled 'kept' is a log line that lies)",
   strip_history_images([{"role": "user", "content": turn(txt("a"), img("A"), img("B"))},
                         {"role": "user", "content": turn(txt("b"), img("C"))},
                         {"role": "user", "content": turn(txt("问"))}], keep=4, older=2)[1:4] == (2, 1, 2),
   str(strip_history_images([{"role": "user", "content": turn(txt("a"), img("A"), img("B"))},
                             {"role": "user", "content": turn(txt("b"), img("C"))},
                             {"role": "user", "content": turn(txt("问"))}], keep=4, older=2)[1:4]))

three_turns = [{"role": "user", "content": turn(txt("1"), img("T1"))}, {"role": "assistant", "content": "ok"},
               {"role": "user", "content": turn(txt("2"), img("T2"))}, {"role": "assistant", "content": "ok"},
               {"role": "user", "content": turn(txt("3"), img("T3"))}]
out, kept, dropped, from_hist = strip_history_images(three_turns, keep=4)
ck("the fallback does NOT fire when the turn being answered carries a picture (2 dropped, 1 kept, 0 from history)",
   (kept, dropped, from_hist) == (1, 2, 0), f"{kept}/{dropped}/{from_hist}")
ck("and the two dropped pictures of that shape are each accounted for where they were",
   sum(1 for m in out[:4] for p in (m["content"] if isinstance(m["content"], list) else [])
       if p.get("text", "").startswith("[1 image(s) from an earlier turn")) == 2)

ck("the fallback cannot lift the engine's own count cap (older=9 asked, keep=1 allowed)",
   strip_history_images(two_replayed, keep=1, older=9)[1:4] == (1, 1, 1),
   str(strip_history_images(two_replayed, keep=1, older=9)))

os.environ["R9K_MM_HISTORY_IMAGE"] = "0"
ck("R9K_MM_HISTORY_IMAGE moves the fallback, and a value that is not a number does not break the read",
   history_keep() == 0)
os.environ["R9K_MM_HISTORY_IMAGE"] = "off"
ck("R9K_MM_HISTORY_IMAGE=off falls back to the default 1 rather than to 0 (a typo must not silently re-strict)",
   history_keep() == 1)
del os.environ["R9K_MM_HISTORY_IMAGE"]
ck("the shipped default is one replayed picture", history_keep() == 1)

# ------------------------------------------------------ what the log may and may not claim ------------------
# The first version logged a request whose OWN turn carried the picture not at all. That is the same silence the
# fallback exists to break: "the model answered about a screenshot" and "the model was never shown one" read
# identical in the log, and the Zed round at 17:28:37 proved it could not be told apart afterwards.
ck("a request whose own turn carries the picture is logged too (that is what was silent)",
   log_line(1, 0, 0, 1, 0)[0] == "info" and "1 image(s) in the prompt (1 from the turn being answered" in log_line(1, 0, 0, 1, 0)[1])
ck("a rescued replayed picture is named as rescued, not as if the turn carried it",
   "0 from the turn being answered, 1 the newest replayed one" in log_line(1, 0, 1, 1, 0)[1])
ck("nothing left in the prompt is a WARNING naming the knob that allowed it",
   log_line(0, 2, 0, 0, 2)[0] == "warning" and "R9K_MM_HISTORY_IMAGE=0" in log_line(0, 2, 0, 0, 2)[1])
ck("a text-only request logs NOTHING (no line per request on the 99% path)",
   log_line(0, 0, 0, 1, 0) is None)
ck("the running total rides every line, so 'how often does this fire' is answerable without grepping",
   log_line(1, 3, 0, 1, 41)[1].endswith("running total dropped 41 (R9K_MM_HISTORY_IMAGE=0 is the strict rule, "
                                        "R9K_MM_LAST_TURN=0 restores stock)"))

# ---------------------------------------------------------- the shared pixel budget (2026-11-21) ------------
# Hooks instead of PIL: the host python has neither PIL nor a vLLM, and what is under test is the arithmetic and
# the identity/fail-open contract, not resampling itself (that lives in _measure_px/_downscale_px, container-side).
B = 16777216                                            # one turn's budget for these cases (the launcher's default
                                                      # is 4,194,304 -- a round number here keeps the arithmetic legible)
PX = {}


def measure(part):
    return PX.get(part["image_url"]["url"])


def downscale(part, target):
    return {"type": "image_url", "image_url": {"url": part["image_url"]["url"] + f"#fit:{target}"}}


big, big2, small = img("big"), img("big2"), img("small")
PX[big["image_url"]["url"]] = 20_000_000
PX[big2["image_url"]["url"]] = 20_000_000
PX[small["image_url"]["url"]] = 800_000

# one picture may take the whole budget: nothing is decoded, nothing copied; the raised engine cap trims one that
# still exceeds it -- which is exactly the fix for the blurred-screenshot incident notes/multimodal-input.md logs
one = [{"role": "user", "content": turn(big)}]
ck("one oversized image is left untouched: it may take the whole budget (no decode, same object)",
   adapt_pixels(one, B, measure, downscale) == (one, 0))

# two pictures whose total exceeds the budget: only the oversized one moves, and to budget/2
two = [{"role": "user", "content": turn(big, small)}]
out, n = adapt_pixels(two, B, measure, downscale)
ck("two images over the budget: the oversized one is resized to budget/2, the fitting one is untouched",
   n == 1 and f"#fit:{B // 2}" in out[0]["content"][0]["image_url"]["url"]
   and out[0]["content"][1] is small, f"n={n}")

fit = [{"role": "user", "content": turn(small, small)}]
ck("two images that fit the budget come back as the SAME object (nothing resized, nothing copied)",
   adapt_pixels(fit, B, measure, downscale)[0] is fit)

# the seam: history is gone BEFORE the budget runs, so an old oversized picture is never measured or resized
hist = [{"role": "user", "content": turn(big2)}, {"role": "assistant", "content": "ok"},
        {"role": "user", "content": turn(big)}]
msgs, kept, dropped, _ = strip_history_images(hist, keep=4)
out, n = adapt_pixels(msgs, B, measure, downscale)
ck("history is dropped before the budget runs: the one kept image takes it whole, the old one was never measured",
   (kept, dropped) == (1, 1) and n == 0 and out is msgs)

unk = img("unmeasurable")
unk_pair = [{"role": "user", "content": turn(big, unk)}]
_uk = adapt_pixels(unk_pair, B, measure, downscale)
ck("an unmeasurable picture means NOTHING is resized (fail-open: guessing at pixels is worse than letting the "
   "engine's cap stand), and the same list object comes back", _uk[1] == 0 and _uk[0] is unk_pair)

# the stock hooks on fake payloads: base64 "a" is not a PNG, PIL may not even import on this host -- the request
# must come back byte-identical, which is what makes a garbage-tolerant measure() safe in the live path either way
fake = [{"role": "user", "content": turn(img("a"), img("b"))}]
r = adapt_pixels(fake, B)
ck("the stock hooks fail open on fake base64 (same object, nothing resized) -- no host needs PIL for this patch",
   r[1] == 0 and r[0] is fake)

ck("the budget reads the engine's own cap, falls back to the checkpoint's 16,777,216, and no attribute shape breaks it",
   pixel_budget(types.SimpleNamespace(model_config=types.SimpleNamespace(
       mm_processor_kwargs={"size": {"longest_edge": 1048576}}))) == 1048576
   and pixel_budget(types.SimpleNamespace(model_config=None)) == 16777216)
os.environ["R9K_MM_TOTAL_PIX"] = "12345"
ck("R9K_MM_TOTAL_PIX overrides the budget read from the config", pixel_budget(None) == 12345)
del os.environ["R9K_MM_TOTAL_PIX"]

print("PASS" if fail == 0 else "FAIL", f"({fail} failing)")
sys.exit(1 if fail else 0)
