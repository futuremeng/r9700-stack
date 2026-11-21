#!/usr/bin/env python3
"""tests/test_mm_last_turn.py - the history-image filter and the shared pixel budget, without a GPU or a vLLM.

Two behaviors, one seam. First (2026-11-09): a chat client replays the transcript every turn, so the images of
earlier turns arrive again, and stock vLLM re-encodes all of them -- until the replay alone hits
--limit-mm-per-prompt and the turn that adds a NEW picture is refused. Only the last submitted turn's pictures may
survive, capped to the engine's own limit (newest first). Second (2026-11-21): the pictures that stay SHARE one
pixel budget -- the engine's own cap, now 16,777,216 px so one image may take it whole and is never resampled --
and from k = 2 up an oversized picture shrinks to budget/k. The resize itself is PIL (the container's); these
tests swap in measurement and resize hooks and check the arithmetic and the fail-open contract.

Run: ./.venv/bin/python3 tests/test_mm_last_turn.py        (stdlib only)
"""
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from r9700_vllm.compat import mm_last_turn as M
from r9700_vllm.compat.mm_last_turn import strip_history_images, adapt_pixels, pixel_budget  # noqa: E402

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
out, kept, dropped = strip_history_images(msgs, keep=4)
ck("history images are dropped and the new one is kept (2 dropped, 1 kept)",
   (kept, dropped) == (1, 2), f"{kept} kept, {dropped} dropped")
ck("the transcript keeps its shape and its text: three messages, the first one still says 'look'",
   len(out) == 3 and out[0]["content"][0]["text"] == "look" and out[2]["content"][1]["image_url"] is not None)
ck("a message that was only pictures does not render as an empty turn",
   strip_history_images([{"role": "user", "content": turn(img())},
                         {"role": "user", "content": turn(img())}])[1:3] == (1, 1))

# the 400 case: five pictures in the turn being answered, the engine's limit is 4
five = [{"role": "user", "content": turn(*[img(str(i)) for i in range(5)])}]
out, kept, dropped = strip_history_images(five, keep=4)
ck("more pictures than the engine allows keeps the NEWEST four instead of answering 400",
   (kept, dropped) == (4, 1)
   and [p["image_url"]["url"] for p in out[0]["content"]] == [img(i)["image_url"]["url"] for i in "1234"],
   f"{kept} kept, {dropped} dropped")

# text-only requests must come back as the SAME object: no copy, no behavior change for the 99% case
plain = [{"role": "user", "content": "hi"}]
ck("a request with no images is returned untouched (same list object, nothing copied)",
   strip_history_images(plain, keep=4)[0] is plain)
ck("R9K_MM_LAST_TURN=0 is the way back to stock behavior (patch() installs nothing)",
   os.environ.get("R9K_MM_LAST_TURN", "1") != "0")

# ---------------------------------------------------------- the shared pixel budget (2026-11-21) ------------
# Hooks instead of PIL: the host python has neither PIL nor a vLLM, and what is under test is the arithmetic and
# the identity/fail-open contract, not resampling itself (that lives in _measure_px/_downscale_px, container-side).
B = 16777216                                            # one turn's budget, = MM_MAX_PIX's new default
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
msgs, kept, dropped = strip_history_images(hist, keep=4)
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
