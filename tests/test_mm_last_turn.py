#!/usr/bin/env python3
"""tests/test_mm_last_turn.py - the history-image filter, without a GPU or a vLLM.

One behavior, and it exists because of a cost and a 400: a chat client replays the transcript every turn, so the
images of earlier turns arrive again, and stock vLLM re-encodes all of them -- until the replay alone hits
--limit-mm-per-prompt and the turn that adds a NEW picture is refused. Only the last submitted turn's pictures may
survive, capped to the engine's own limit (newest first).

Run: ./.venv/bin/python3 tests/test_mm_last_turn.py        (stdlib only)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from r9700_vllm.compat.mm_last_turn import strip_history_images  # noqa: E402

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

print("PASS" if fail == 0 else "FAIL", f"({fail} failing)")
sys.exit(1 if fail else 0)
