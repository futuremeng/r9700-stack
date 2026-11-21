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

What this deliberately does NOT do: it does not make an earlier picture still visible -- asking "what was in the
first screenshot?" needs the client to send it again. That is the trade, and it is what "只处理最后一次提交内容中的图片"
asks for. `R9K_MM_LAST_TURN=0` restores stock behavior (every image in the prompt is processed, 400 and all),
`R9K_MM_MAX_IMAGE=N` overrides the count cap when reading the model config is not enough, and
`R9K_MM_TOTAL_PIX=N` the pixel budget.
"""
from __future__ import annotations

import os

try:                                                   # the pure helper has to be importable without vLLM
    from vllm.logger import init_logger                # (tests/test_mm_last_turn.py runs on the host python)
except Exception:                                      # noqa: BLE001
    import logging
    init_logger = logging.getLogger

logger = init_logger("vllm." + __name__)

_PATCHED = False
_SHOWN = False
_RESIZED = False

# The part types a chat request can carry a picture in. TypedDicts in vLLM's chat_utils spell it four ways: the
# OpenAI wire format ("image_url"), the simple form some clients send ("image_url" as a bare string is normalized
# to this same type), the embeds form ("image_embeds"), and the in-process PIL form.
IMG_TYPES = ("image_url", "image", "input_image", "image_pil", "image_embeds")

# The fallback shared pixel budget -- preprocessor_config.json's own size.longest_edge, the checkpoint's number
# rather than one invented here. The launchers pass the same value as MM_MAX_PIX, so pixel_budget() normally
# reads it back from the engine rather than using this.
_TOTAL_PIX = 16777216

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


def strip_history_images(messages, keep: int = 4):
    """(messages, kept, dropped) -- `messages` untouched unless something had to change.

    `keep` is the number of pictures the last turn may carry; the newest ones win. Returns the SAME list object
    when nothing was dropped, so a request that needs no change is not copied.
    """
    if not messages:
        return messages, 0, 0
    # the last submitted turn: the last user message, and anything after it (tool results with a screenshot in
    # them belong to the turn being answered, not to history)
    cut = _turn_cut(messages)
    out, kept, dropped = list(messages), 0, 0
    for i, msg in enumerate(messages):
        content = (msg or {}).get("content")
        if not isinstance(content, list):
            continue                                    # a plain string content carries no parts
        if i >= cut:
            imgs = [p for p in content if _is_image(p)]
            if len(imgs) <= keep:
                kept += len(imgs)
                continue
            gone = imgs[:len(imgs) - keep]              # the OLDEST of this turn's own pictures
            keep_imgs = imgs[len(imgs) - keep:]
            dropped += len(gone)
            kept += len(keep_imgs)
            new = [p for p in content if not _is_image(p)] + keep_imgs
            out[i] = dict(msg, content=new)
            continue
        gone = [p for p in content if _is_image(p)]
        if not gone:
            continue
        dropped += len(gone)
        rest = [p for p in content if not _is_image(p)]
        if not rest:
            # a message that was only pictures would render as an empty turn, which reads to the model as a
            # glitch in the transcript; one line saying what happened is cheaper than a confused answer
            rest = [{"type": "text", "text": f"[{len(gone)} image(s) from an earlier turn not re-processed]"}]
        out[i] = dict(msg, content=rest)
    return (out, kept, dropped) if dropped else (messages, kept, dropped)


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
        global _SHOWN, _RESIZED
        try:
            msgs, kept, dropped = strip_history_images(getattr(request, "messages", None), image_limit(self))
            if dropped:
                request.messages = msgs
                if not _SHOWN:
                    _SHOWN = True
                    logger.info("r9700 mm last-turn: dropped %d image(s) replayed from history, kept %d from the "
                                "turn being answered (R9K_MM_LAST_TURN=0 restores stock behavior)", dropped, kept)
                else:
                    logger.debug("r9700 mm last-turn: dropped %d, kept %d", dropped, kept)
            msgs, shrunk = adapt_pixels(msgs, pixel_budget(self))
            if shrunk:
                request.messages = msgs
                if not _RESIZED:
                    _RESIZED = True
                    logger.info("r9700 mm last-turn: shrank %d oversized image(s) so the turn's pictures share the "
                                "%d px budget (one image may take it whole; R9K_MM_TOTAL_PIX overrides)",
                                shrunk, pixel_budget(self))
                else:
                    logger.debug("r9700 mm last-turn: shrank %d image(s) to the per-turn pixel budget", shrunk)
        except Exception as e:                          # noqa: BLE001 -- a dropped request is worse than an old bug
            logger.warning("r9700 mm last-turn: leaving the request untouched: %r", e)
        return await orig(self, request)

    cls.render_chat_request = wrapped
    _PATCHED = True
    logger.info("r9700 mm last-turn installed on %s.render_chat_request (only the images of the last submitted "
                "turn are processed, and they share one pixel budget)", cls.__name__)
    return True
