"""Only the images of the LAST submitted turn are processed: the images a chat client replays from history are
dropped, not re-encoded.

The bug this closes (reported 2026-11-09, and it is a cost AND a hard failure, not a nicety): an OpenAI-style chat
client replays the whole conversation on every turn, so turn N carries N-1 earlier pictures again. Stock vLLM
processes every one of them -- each is fetched, resized and pushed through the vision tower, and each one spends
image tokens in the window (`serve/native-flashnext.sh`: one 2,560x6,236 screenshot measured 15,664 prompt tokens,
6% of a 262,144-token window; `MM_MAX_PIX` caps what one turn's pictures cost). And `--limit-mm-per-prompt` counts
images **per prompt**, so the replayed history hits that ceiling by itself: with the standing `image: 4` cap, a
conversation that already holds four pictures answers

    400 Bad Request: At most 4 image(s) may be provided in one prompt

on the turn that adds the fifth, even though exactly one image is new. The engine has no notion of "what is new in
this turn", because by the time it sees the request the transcript is one flat token sequence.

So the filter sits at the one place every chat request passes, before the transcript is rendered:
`OpenAIChatServingContext.render_chat_request`. It keeps the pictures of the last submitted turn -- the last `user`
message, plus anything after it, which is where tool results carrying a screenshot live -- and drops the image parts
of everything older, leaving one short text part in a message that would otherwise render empty. Within the kept
turn the LAST `N` pictures survive, `N` being the engine's own image limit (read from the model config, so there is
no second number to keep in sync): if a client sends six, the four newest are what the model sees rather than a 400.

A second policy rides on the same seam (2026-11-21): the pictures that stay also SHARE one pixel budget, the
engine's own cap, and since that cap is now the checkpoint's own 16,777,216 px (`serve/native-flashnext.sh`'s
`MM_MAX_PIX`) a lone big screenshot may take the whole budget and is never resampled. The incident that moved
this: a low-resolution screenshot's small Chinese text came back unreadable, and the measurements in
`notes/multimodal-input.md` put the cause on the old 4,194,304 px PER-IMAGE cap downsampling the pixels before
the model saw them, not on fonts or the client. From two pictures up, when the kept turn's total exceeds the
budget each oversized one is shrunk here to budget/k -- the same 16,384-token worst case the old 4 x 4,096 cost,
but paid jointly. History pictures are gone before the budget runs, so they are never measured or resized.

A THIRD policy rides on the same seam (2026-11-21, later the same day, and it exists because the second one was
not enough). "Only the last turn's pictures" is exactly right about cost and exactly wrong about a follow-up
question: the client replays the screenshot inside the message it was attached to, the NEXT user message is plain
text, and the strict rule then hands the model a request with no picture in it and not one word saying a picture
was removed. Measured on the live serve (notes/multimodal-input.md, 2026-11-21): the same screenshot and the same
question cost 232 prompt tokens in one message and 84 tokens asked on the next turn -- 84 is the question alone,
the picture is not there. What came back at 84 tokens was "我当前没有看到可读取的图片内容", and in other turns the
model answered by invention with same-family glyphs (猫 read out as 猛, a whole sentence as 冗冗冗), which from the
other end of the wire is precisely "汉字都认不出来". So two things changed, both about not failing silently:

1. every message that loses pictures keeps one text part saying so, whether or not the message still has text; and
2. when the turn being answered carries NO picture at all, the newest replayed picture stays --
   `R9K_MM_HISTORY_IMAGE=N`, default 1, `0` restores the strict rule. At most one extra picture, still inside the
   same count cap and the same per-turn pixel budget.

What this deliberately does NOT do: it does not make an arbitrary earlier picture still visible -- asking "what was
in the first of six screenshots?" needs the client to send it again. That is the trade, and it is what
"只处理最后一次提交内容中的图片" asks for; the fallback covers the one shape where the strict rule stops the
question itself from being answerable. `R9K_MM_LAST_TURN=0` restores stock behavior (every image in the prompt is
processed, 400 and all), `R9K_MM_MAX_IMAGE=N` overrides the count cap when reading the model config is not enough,
`R9K_MM_TOTAL_PIX=N` the pixel budget, and `R9K_MM_HISTORY_IMAGE=N` how many replayed pictures stay when this turn
carries none.
"""
from __future__ import annotations

import logging
import os

try:                                                   # the pure helper has to be importable without vLLM
    from vllm.logger import init_logger                # (tests/test_mm_last_turn.py runs on the host python)
except Exception:                                      # noqa: BLE001
    import logging
    init_logger = logging.getLogger

logger = init_logger("vllm." + __name__)

_PATCHED = False
# Running totals, printed with every line: "does this fire in production, and how often" is the first question an
# operator asks about a filter that edits other people's requests, and the answer used to be "once per launch, in
# one INFO line, then nothing at all".
_DROPPED = 0
_RESIZED = 0

# The part types a chat request can carry a picture in. TypedDicts in vLLM's chat_utils spell it four ways: the
# OpenAI wire format ("image_url"), the simple form some clients send ("image_url" as a bare string is normalized
# to this same type), the embeds form ("image_embeds"), and the in-process PIL form.
IMG_TYPES = ("image_url", "image", "input_image", "image_pil", "image_embeds")

# The fallback shared pixel budget -- preprocessor_config.json's own size.longest_edge, the checkpoint's number
# rather than one invented here. The launchers pass the same value as MM_MAX_PIX, so pixel_budget() normally
# reads it back from the engine rather than using this.
_TOTAL_PIX = 16777216

# How many REPLAYED pictures stay when the turn being answered carries none at all. One, because the shape this
# covers is a follow-up question about the screenshot the user attached one message ago, and one picture is what
# that question needs; the count cap and the pixel budget still decide what it may cost.
_HISTORY_IMAGE = 1

# The vision patch grid: a resize must round dimensions down to a multiple of this (patch 16 x merge 2) or the
# engine resamples the result anyway.
_PATCH_GRID = 32


def _is_image(part) -> bool:
    return isinstance(part, dict) and part.get("type") in IMG_TYPES


def _turn_cut(messages) -> int:
    """Index where the last submitted turn starts: the last `user` message, else the last message."""
    for i in range(len(messages) - 1, -1, -1):
        if (messages[i] or {}).get("role") == "user":
            return i
    return len(messages) - 1


def _image_slots(msg):
    """The part indexes a picture sits at in this message's content, or [] for a string/absent content."""
    content = (msg or {}).get("content")
    return [j for j, p in enumerate(content) if _is_image(p)] if isinstance(content, list) else []


def strip_history_images(messages, keep: int = 4, older: int = _HISTORY_IMAGE):
    """(messages, kept, dropped, kept_from_history) -- `messages` untouched unless something had to change.

    `keep` is the engine's own per-prompt picture limit and it bounds the TOTAL the request carries, newest first.
    `older` is the fallback slot: when the turn being answered carries no picture at all, the newest `older`
    pictures of the replayed history stay, so a follow-up question about a screenshot is still a question about a
    picture. `older` is clamped to `keep`, so the fallback can never lift the engine's limit.

    Every message that loses at least one picture gets one text part saying so -- including a message that still
    has its text. That is not decoration: "看这张截图" with the screenshot quietly deleted reads to the model as a
    picture it is looking at, and it answers by invention. Returns the SAME list object when nothing was dropped,
    so a request that needs no change is not copied.
    """
    if not messages:
        return messages, 0, 0, 0
    # the last submitted turn: the last user message, and anything after it (tool results with a screenshot in
    # them belong to the turn being answered, not to history)
    cut = _turn_cut(messages)
    slots = min(older, keep) if older > 0 else 0
    # what the turn being answered carries on its own decides whether a replayed picture gets a slot at all
    if sum(len(_image_slots(m)) for m in messages[cut:]) or not slots:
        slots = 0
    # the newest `slots` replayed pictures, counted back from the end of the transcript
    older_spots = [(i, j) for i in range(min(cut, len(messages))) for j in _image_slots(messages[i])]
    stay = set(older_spots[len(older_spots) - slots:]) if slots else set()

    out, kept, dropped, kept_older = list(messages), 0, 0, 0
    for i, msg in enumerate(messages):
        content = (msg or {}).get("content")
        if not isinstance(content, list):
            continue                                    # a plain string content carries no parts
        spots = _image_slots(msg)
        if i >= cut:
            if len(spots) <= keep:
                kept += len(spots)
                continue
            gone = spots[:len(spots) - keep]            # the OLDEST of this turn's own pictures
            dropped += len(gone)
            kept += len(spots) - len(gone)
            # the kept pictures stay where the client put them (only the dropped ones are pulled out), and the
            # message says how many went -- the old code also moved the survivors to the end of the content, which
            # is exactly the part order that was measured to change what the model reads out of a picture
            new = [p for j, p in enumerate(content) if j not in set(gone)]
            out[i] = dict(msg, content=new + [{"type": "text", "text": f"[{len(gone)} image(s) dropped: this "
                                                          f"request processes only the newest {keep}]"}])
            continue
        gone = [j for j in spots if (i, j) not in stay]
        if not gone:
            # every picture in this replayed message stayed -- that is the fallback doing its job, and the caller
            # has to be able to tell that from "the turn being answered carried them"
            kept += len(spots)
            kept_older += len(spots)
            continue
        stayed = len(spots) - len(gone)
        dropped += len(gone)
        kept += stayed
        kept_older += stayed
        rest = [p for j, p in enumerate(content) if j not in set(gone)]
        # a message that was only pictures would render as an empty turn, which reads to the model as a glitch in
        # the transcript, and a message that still has its text would read as a picture it can see. One line
        # saying what happened is cheaper than a confident answer about a picture that is not in the prompt.
        rest += [{"type": "text", "text": f"[{len(gone)} image(s) from an earlier turn not re-processed]"}]
        out[i] = dict(msg, content=rest)
    return (out, kept, dropped, kept_older) if dropped else (messages, kept, dropped, kept_older)


def _data_url(part):
    """The base64 ``data:`` URL an image part carries, or None. Remote URLs are left alone: this layer neither
    fetches nor resizes what it has not got (fetching here would also sidestep --limit-media-domains)."""
    for key in ("image_url", "image", "input_image"):
        v = part.get(key)
        if isinstance(v, dict):
            v = v.get("url")
        if isinstance(v, str) and v.startswith("data:"):
            return v
    return None


def _measure_px(part):
    """The pixels an image part carries, or None if undecidable -- which must leave the request untouched. PIL
    is the container's (vLLM depends on it); the host test swaps these hooks wholesale, and the stock hook here
    must not raise on a payload that is not a real image."""
    url = _data_url(part)
    if url is None:
        return None
    try:
        import base64
        import io
        from PIL import Image
        with Image.open(io.BytesIO(base64.b64decode(url.partition(",")[2]))) as im:
            return int(im.width) * int(im.height)
    except Exception:                                   # noqa: BLE001 -- fail-open: no picture is resized blind
        return None


def _downscale_px(part, target: int):
    """A COPY of `part` resized to at most `target` pixels -- or None if it already fits or cannot be resized
    here. Dimensions round DOWN to the vision patch grid so the engine does not resample the result, and the
    re-encode is lossless PNG: this step exists because resampling hurt drawn text once (notes/multimodal-input.md,
    2026-11-21), so it must not also add JPEG artifacts to whatever pixels survive it."""
    url = _data_url(part)
    if url is None:
        return None
    try:
        import base64
        import io
        from PIL import Image
        with Image.open(io.BytesIO(base64.b64decode(url.partition(",")[2]))) as im:
            im.load()
            if im.width * im.height <= target:
                return None                             # already fits; this one is left untouched
            if im.mode != "RGB":
                rgba = im.convert("RGBA")               # composite alpha onto WHITE: a transparent-background
                bg = Image.new("RGB", rgba.size, (255, 255, 255))  # text screenshot must not flip to inverse
                bg.paste(rgba, mask=rgba.split()[-1])
                im = bg
            scale = (target / (im.width * im.height)) ** 0.5
            size = tuple(max(_PATCH_GRID, int(d * scale) // _PATCH_GRID * _PATCH_GRID) for d in im.size)
            buf = io.BytesIO()
            im.resize(size, Image.LANCZOS).save(buf, format="PNG")
        new_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:                                   # noqa: BLE001 -- fail-open
        return None
    v = part.get("image_url")                           # rebuild the part's own shape, whichever form it came in
    rebuilt = dict(part)
    if isinstance(v, dict):
        rebuilt["image_url"] = dict(v, url=new_url)
    elif isinstance(v, str):
        rebuilt["image_url"] = new_url
    else:
        return None                                     # a PIL/embeds part is not this function's to rewrite
    return rebuilt


def adapt_pixels(messages, total: int, measure=None, downscale=None):
    """(messages, resized_count) -- make the kept turn's pictures SHARE the pixel budget `total`.

    With one picture there is nothing to do: it may take the whole budget, and the engine's own cap trims one
    that exceeds it without a decode here (decoding is for k >= 2, and only then does this touch a byte). From
    two up, an oversized picture shrinks to total/k so the turn's images cost at most what the old per-image cap
    times the old count cap allowed, paid jointly. Returns the SAME list object when nothing was resized, and
    leaves everything untouched when a picture's size cannot be measured -- guessing at a pixel count is worse
    than letting the engine's cap stand. `measure`/`downscale` are the two PIL hooks, swappable for tests.
    """
    if not messages or total <= 0:
        return messages, 0
    measure = _measure_px if measure is None else measure
    downscale = _downscale_px if downscale is None else downscale
    cut = _turn_cut(messages)
    spots = []                                          # (message index, part index, pixels)
    for i in range(cut, len(messages)):
        content = (messages[i] or {}).get("content")
        if not isinstance(content, list):
            continue
        for j, part in enumerate(content):
            if _is_image(part) and part.get("type") != "image_embeds":   # embeds carry no pixels to measure
                px = measure(part)
                if not isinstance(px, int) or px <= 0:
                    return messages, 0
                spots.append((i, j, px))
    if len(spots) < 2 or sum(s[2] for s in spots) <= total:
        return messages, 0
    per = total // len(spots)
    out, done = list(messages), 0
    for i, j, px in spots:
        if px <= per:
            continue                                    # already within its share; never shrunk further
        msg = out[i] if isinstance(out[i], dict) else {}
        content = list(msg.get("content") or [])
        new = downscale(content[j], per)
        if new is None:
            continue
        content[j] = new
        out[i] = dict(msg, content=content)
        done += 1
    return (out, done) if done else (messages, 0)


def image_limit(self) -> int:
    """How many pictures one prompt may carry -- the engine's own number, not a copy of it."""
    v = os.environ.get("R9K_MM_MAX_IMAGE", "")
    if v.isdigit():
        return int(v)
    try:
        lim = self.model_config.multimodal_config.limit_per_prompt
        got = lim.get("image") if hasattr(lim, "get") else getattr(lim, "image", None)
        if isinstance(got, int) and got > 0:
            return got
    except Exception:                                   # noqa: BLE001 -- a moved attribute must not break a request
        pass
    return 4                                            # serve/native-flashnext.sh's MM_IMAGE default


def history_keep() -> int:
    """How many replayed pictures stay when the turn being answered carries none -- 1 by default, 0 for the strict
    rule. Not read from the model config: there is no engine number for it, this stack's policy is the number."""
    v = os.environ.get("R9K_MM_HISTORY_IMAGE", "")
    if v.isdigit():
        return int(v)
    return _HISTORY_IMAGE


def pixel_budget(self) -> int:
    """What one turn's pictures may cost TOGETHER, in pixels -- the engine's own size cap again, not a copy of
    it: the launchers pass MM_MAX_PIX as --mm-processor-kwargs' longest_edge, and the budget IS that number so
    one image may take it whole."""
    v = os.environ.get("R9K_MM_TOTAL_PIX", "")
    if v.isdigit():
        return int(v)
    try:
        kw = self.model_config.mm_processor_kwargs or {}
        longest = (kw.get("size") or {}).get("longest_edge")
        if isinstance(longest, int) and longest > 0:
            return longest
    except Exception:                                   # noqa: BLE001 -- a moved attribute must not break a request
        pass
    return _TOTAL_PIX


def log_line(kept, dropped, from_history, older, total):
    """(level, message) for one chat request that carried pictures, or None when it carried none.

    Pure on purpose: `patch()` needs a vLLM to run, and the question this answers -- "did the picture the user is
    asking about reach the model" -- has to be held by tests/test_mm_last_turn.py without one. It exists at all
    because a request whose OWN turn carries the picture used to log nothing whatsoever, so "the model answered
    about a screenshot" and "the model was never shown one" were indistinguishable in the log.
    """
    if not kept and not dropped:
        return None
    if not kept:
        return "warning", (f"r9700 mm last-turn: dropped {dropped} image(s) replayed from history and NOTHING is "
                           f"left in the prompt: the turn being answered carries no picture and "
                           f"R9K_MM_HISTORY_IMAGE={older} kept none as a fallback, so the model is being asked "
                           f"about a picture it was not shown (the transcript now says so in words); "
                           f"running total dropped {total}")
    return "info", (f"r9700 mm last-turn: {kept} image(s) in the prompt ({kept - from_history} from the turn being "
                    f"answered, {from_history} the newest replayed one kept as a fallback), dropped {dropped} "
                    f"replayed from history; running total dropped {total} "
                    f"(R9K_MM_HISTORY_IMAGE=0 is the strict rule, R9K_MM_LAST_TURN=0 restores stock)")


def patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("R9K_MM_LAST_TURN", "1") == "0":
        logger.info("r9700 mm last-turn: off (R9K_MM_LAST_TURN=0) -- every image in a prompt is processed again")
        return False
    from . import gate
    gate.check("mm_last_turn")
    try:
        from vllm.entrypoints.openai.chat_completion import serving as S
    except Exception as e:                              # noqa: BLE001
        logger.warning("r9700 mm last-turn: no chat serving module to patch: %r", e)
        return False
    cls = next((c for c in vars(S).values()
                if isinstance(c, type) and "render_chat_request" in vars(c)), None)
    if cls is None:
        logger.warning("r9700 mm last-turn: %s has no render_chat_request to wrap -- images from every turn of a "
                       "conversation will be processed again", S.__name__)
        return False
    orig = cls.render_chat_request

    async def wrapped(self, request):
        global _DROPPED, _RESIZED
        try:
            msgs, kept, dropped, from_history = strip_history_images(getattr(request, "messages", None),
                                                                     image_limit(self), history_keep())
            if dropped:
                request.messages = msgs
                _DROPPED += dropped
            line = log_line(kept, dropped, from_history, history_keep(), _DROPPED)
            if line:
                logger.log(logging.WARNING if line[0] == "warning" else logging.INFO, line[1])
            msgs, shrunk = adapt_pixels(msgs, pixel_budget(self))
            if shrunk:
                request.messages = msgs
                _RESIZED += shrunk
                logger.info("r9700 mm last-turn: shrank %d oversized image(s) so the turn's pictures share the "
                            "%d px budget (one image may take it whole; R9K_MM_TOTAL_PIX overrides); running "
                            "total shrunk %d", shrunk, pixel_budget(self), _RESIZED)
        except Exception as e:                          # noqa: BLE001 -- a dropped request is worse than an old bug
            logger.warning("r9700 mm last-turn: leaving the request untouched: %r", e)
        return await orig(self, request)

    cls.render_chat_request = wrapped
    _PATCHED = True
    logger.info("r9700 mm last-turn installed on %s.render_chat_request (only the last submitted turn's images are "
                "processed; when it carries none, the newest %d replayed picture stays, and every message that loses "
                "a picture says so)", cls.__name__, history_keep())
    return True
