"""Only the images of the LAST submitted turn are processed: the images a chat client replays from history are
dropped, not re-encoded.

The bug this closes (reported 2026-11-09, and it is a cost AND a hard failure, not a nicety): an OpenAI-style chat
client replays the whole conversation on every turn, so turn N carries N-1 earlier pictures again. Stock vLLM
processes every one of them -- each is fetched, resized and pushed through the vision tower, and each one spends
image tokens in the window (`serve/native-flashnext.sh`: one 2,560x6,236 screenshot measured 15,664 prompt tokens,
6% of a 262,144-token window; `MM_MAX_PIX` caps one image at ~4,096 tokens). And `--limit-mm-per-prompt` counts
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

What this deliberately does NOT do: it does not make an earlier picture still visible -- asking "what was in the
first screenshot?" needs the client to send it again. That is the trade, and it is what "只处理最后一次提交内容中的图片"
asks for. `R9K_MM_LAST_TURN=0` restores stock behavior (every image in the prompt is processed, 400 and all), and
`R9K_MM_MAX_IMAGE=N` overrides the cap when reading the model config is not enough.
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

# The part types a chat request can carry a picture in. TypedDicts in vLLM's chat_utils spell it four ways: the
# OpenAI wire format ("image_url"), the simple form some clients send ("image_url" as a bare string is normalized
# to this same type), the embeds form ("image_embeds"), and the in-process PIL form.
IMG_TYPES = ("image_url", "image", "input_image", "image_pil", "image_embeds")


def _is_image(part) -> bool:
    return isinstance(part, dict) and part.get("type") in IMG_TYPES


def strip_history_images(messages, keep: int = 4):
    """(messages, kept, dropped) -- `messages` untouched unless something had to change.

    `keep` is the number of pictures the last turn may carry; the newest ones win. Returns the SAME list object
    when nothing was dropped, so a request that needs no change is not copied.
    """
    if not messages:
        return messages, 0, 0
    # the last submitted turn: the last user message, and anything after it (tool results with a screenshot in
    # them belong to the turn being answered, not to history)
    cut = len(messages) - 1
    for i in range(len(messages) - 1, -1, -1):
        if (messages[i] or {}).get("role") == "user":
            cut = i
            break
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
        global _SHOWN
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
        except Exception as e:                          # noqa: BLE001 -- a dropped request is worse than an old bug
            logger.warning("r9700 mm last-turn: leaving the request untouched: %r", e)
        return await orig(self, request)

    cls.render_chat_request = wrapped
    _PATCHED = True
    logger.info("r9700 mm last-turn installed on %s.render_chat_request (only the images of the last submitted "
                "turn are processed)", cls.__name__)
    return True
