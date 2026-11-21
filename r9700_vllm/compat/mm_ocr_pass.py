"""Read the text out of a picture FIRST, with the narrow questions this checkpoint answers reliably, and hand the
model the RESULT as text.

Why (measured 2026-11-21, `notes/multimodal-input.md`): the vision path is not blind and it is not a quantization
problem. The same hand-drawn card, asked two different ways:

    只转录图片里的第一行, 逐字。   ->  机器学习模型识别           exact, 5 runs out of 5, 0.54 s
    图片里的文字写的是什么？逐行转录。 ->  机机机习机机机机 | 合同需机月机前机 | ...   a repetition cycle

The picture is the same and so are its image tokens; only the LENGTH OF THE EXPECTED ANSWER changed, and the long
answer collapses onto whatever token it committed first (`机机机…`, a period-2 `机会机会…`, `杌杌杌…`, or 2,500
generated tokens with an empty `content` at `finish_reason=length` while the loop runs in the reasoning channel).
Turning the MTP drafter off and moving the main KV cache back to bf16 change nothing, and the text-only control is
exact (three-line dictation, "repeat this line three times", a passage that has to contain 篡改 three times) -- so
the language model, its CJK vocabulary and the PLE n-gram path are fine, and what fails is one long generation
that has to keep reading a picture.

So do not ask for one long transcription: ask for k SHORT ones -- a horizontal band at a time, the shape that is
measured to be reliable -- and give the main request the extracted text. Three modes:

  * `R9K_MM_OCR=off`   (the default, and what a launch does until this is said out loud): nothing happens.
  * `R9700/R9K_MM_OCR=hybrid`: the picture stays in the prompt (scene, layout, charts, "what is in this
    screenshot" questions keep working) and the extracted text is added to it as a labelled block.
  * `R9K_MM_OCR=text`: the picture parts of the turn being answered are REPLACED by the text, which is the cheap
    shape for a document/OCR workload -- the turn pays no image tokens at all.

The asks are real requests to this same engine (`self.create_chat_completion`), issued concurrently, marked with
`user=R9K_MM_OCR_MARKER` so the wrapper skips them instead of recursing. They cost `bands x ~1k` prompt tokens and
about one band's latency, so the result of each (image bytes, band) pair is kept in a bounded in-process cache:
a chat client replays the same screenshot on every turn (measured: 93.2% prefix-cache reuse on such a replay),
and the second and later turns of a conversation pay nothing for the pass.

What this deliberately does NOT do: it does not try to understand the picture (no scene description, no chart
reading -- the picture itself still does that in `hybrid`), it does not add an OCR engine (no tesseract, no
onnxruntime, nothing to install), and it never touches a request it cannot complete: every step is wrapped, and on
any error the request goes through exactly as it would have with the pass off.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging                                        # NOT inside the except below: the wrapper names logging.WARNING
import os                                              # on its hot path, and a module that only imports it when vLLM
import time                                            # is MISSING is a module that NameErrors on every image turn
from collections import OrderedDict

try:                                                   # the pure helpers must import without a vLLM
    from vllm.logger import init_logger                # (tests/test_mm_ocr_pass.py runs on the host python)
except Exception:                                      # noqa: BLE001
    init_logger = logging.getLogger

logger = init_logger("vllm." + __name__)

_PATCHED = False
# What the marker is for: the sub-requests go back through the very seam this module wraps, and reading it is what
# stops them from OCR-ing themselves. It also names itself in the log, so an operator counting requests on the
# Monitor page can see why an image turn arrived as several.
MARKER = "r9700-ocr-pass"

# The one question every band gets. Fixed wording on purpose: it is part of the cache key's version, so changing it
# is a deliberate act (an old cache entry cannot answer a different question).
ASK = ("逐行转录这张图里的文字，只输出文字本身，一行一条；"
       "看不清或图中没有文字就只输出：无")

BLOCK_OPEN = "[图片文字·自动识别，可能有错]"
BLOCK_CLOSE = "[/图片文字]"
NOTE_EMPTY = f"{BLOCK_OPEN}\n（没有识别出文字）\n{BLOCK_CLOSE}"

_BANDS = 4                       # horizontal bands per picture
_MIN_BAND_H = 160                # do not slice a picture into bands shorter than a glyph row's worth
_OVERLAP = 0.08                  # of a band's height, so a line of text is not cut in half at a boundary
_CACHE_MAX = 64                  # pictures (each holds up to `bands` strings)
_VERSION = "v1"                  # bump with ASK or the band policy: it is inside the cache key
_CACHE: "OrderedDict[str, list[str]]" = OrderedDict()

_COUNTS = {"asks": 0, "hits": 0, "chars": 0, "drops": 0, "fails": 0}


def mode() -> str:
    """`off` (default), `hybrid` or `text`. Nothing here is on until a launch says it."""
    v = (os.environ.get("R9K_MM_OCR") or "").strip().lower()
    return v if v in ("hybrid", "text") else "off"


def band_count() -> int:
    """How many horizontal bands a picture is read in. Clamped: 1 asks one narrow question, 8 is the ceiling
    before the asks cost more than the picture they replace."""
    v = os.environ.get("R9K_MM_OCR_BANDS", "")
    return max(1, min(8, int(v))) if v.isdigit() else _BANDS


def band_slices(h: int, n: int, min_h: int = _MIN_BAND_H, overlap: float = _OVERLAP):
    """(y0, y1) pairs covering 0..h in reading order, or [] if the picture is too short to slice.

    Pure so the geometry is testable without an engine or a decoder: the bands must cover the whole picture,
    overlap by `overlap` of their own height so a text line does not get cut at a boundary, and never be shorter
    than `min_h` when the picture allows -- a band that holds less than a glyph row answers 无 and wastes an ask.
    """
    if h <= 0:
        return []
    n = max(1, int(n))
    if h < min_h * n:                      # a picture that cannot host n bands gets as many as fit
        n = max(1, h // min_h)
    step = h / n
    pad = int(step * overlap)
    out = []
    for i in range(n):
        y0 = max(0, int(i * step) - (pad if i else 0))
        y1 = min(h, int((i + 1) * step) + (pad if i < n - 1 else 0))
        if y1 > y0:
            out.append((y0, y1))
    return out


def line_slices(profile, min_h: int = 10, pad: int = 6, merge_gap: int = 6):
    """(y0, y1) pairs -- one per TEXT LINE, in reading order -- from a per-row count of dark pixels.

    This is the shape the pass actually wants. Equal-height bands were measured (bench/mm_ocr_ab.py) and they only
    help small print: a band of a big-font picture is still a long answer, and the ask itself collapses into
    `机机机机`. One line at a time is the shape measured to be reliable (5 runs out of 5 exact on a 110 px card).

    Pure on purpose: `profile` is just the dark-pixel count of each row, so the segmentation -- runs of rows, gaps
    smaller than `merge_gap` merged (a line of CJK has holes: 川, 三, and the gap between two lines of a wrapped
    sentence), runs shorter than `min_h` dropped (a rule line, a stray dot) and each line padded by `pad` so its
    ascenders and descenders are not cropped -- is testable without an image or a decoder. [] means "no text-line
    structure found": a photograph, a chart, a scan with no clean baseline -- and the caller falls back to bands.
    """
    h = len(profile)
    if not h:
        return []
    runs, y = [], 0
    while y < h:
        if profile[y] > 0:
            y0 = y
            while y < h and profile[y] > 0:
                y += 1
            runs.append([y0, y])
        else:
            y += 1
    if not runs:
        return []
    out = [runs[0]]
    for r in runs[1:]:                       # close small gaps: one line of text is not always one run
        if r[0] - out[-1][1] <= merge_gap:
            out[-1][1] = r[1]
        else:
            out.append(r)
    out = [[max(0, a - pad), min(h, b + pad)] for a, b in out if b - a >= min_h]
    return [tuple(o) for o in out]


def _row_profile(img):
    """The dark-pixel count of each row -- the input `line_slices` needs. A PIL hook, so tests do not need one."""
    try:
        w = max(1, min(img.width, 900))                      # a profile at ~900 columns is enough and cheap
        im = img.convert("L").resize((w, img.height))
        px = im.load()
        return [sum(1 for x in range(w) if px[x, y] < 160) for y in range(img.height)]
    except Exception:                                        # noqa: BLE001 -- no profile, fall back to bands
        return []


def _b64_of(part) -> bytes:
    """The bytes of an image part's base64 payload ("" when it is not a data URL / not decodable here)."""
    if not isinstance(part, dict):
        return b""
    url = part.get("image_url") or {}
    url = url.get("url") if isinstance(url, dict) else url
    if isinstance(url, str) and url.startswith("data:") and "," in url:
        return url.split(",", 1)[1].encode("ascii", "ignore")
    return b""


def _decode(b64: bytes):
    """(bytes, PIL image) for an image part, or None when PIL is not around or the bytes are not a picture."""
    try:
        from PIL import Image
        raw = __import__("base64").b64decode(b64) if isinstance(b64, bytes) else b64
        return raw, Image.open(io.BytesIO(raw))
    except Exception:                                    # noqa: BLE001 -- fail-open, the caller keeps the picture
        return None


def _crop_part(img, y0: int, y1: int):
    """An `image_url` part holding the crop, PNG, or None if PIL cannot produce it."""
    try:
        buf = io.BytesIO()
        img.convert("RGB").crop((0, y0, img.width, y1)).save(buf, format="PNG")
        return {"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + __import__("base64").b64encode(buf.getvalue()).decode()}}
    except Exception:                                    # noqa: BLE001
        return None


def cache_key(raw: bytes, i: int, y0: int, y1: int) -> str:
    return f"{_VERSION}:{i}:{y0}:{y1}:{hashlib.sha256(raw).hexdigest()}"


def cache_get(key: str):
    v = _CACHE.get(key)
    if v is not None:
        _CACHE.move_to_end(key)
        _COUNTS["hits"] += 1
    return v


def cache_put(key: str, text: str):
    _CACHE[key] = text
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX * max(1, band_count()):
        _CACHE.popitem(last=False)


def merge(bands) -> str:
    """The band answers in reading order, with the empty ones (无 / blank) dropped."""
    out = []
    for b in bands or ():
        t = (b or "").strip()
        if not t:
            continue
        keep = [ln.strip() for ln in t.splitlines() if ln.strip() and ln.strip() not in ("无", "無", "无文字", "（无）")]
        if keep:
            out.append("\n".join(keep))
    return "\n".join(out)


def block_of(text: str) -> str:
    return NOTE_EMPTY if not text else f"{BLOCK_OPEN}\n{text}\n{BLOCK_CLOSE}"


def inject(messages, text: str, keep_image: bool):
    """(messages, dropped_images) -- the extracted text lands in the turn being answered, as a labelled text part
    placed BEFORE the question's own text (measured: where the picture and the text sit relative to each other
    changes what comes back, so placement is a decision, not a detail).

    `keep_image=False` replaces the turn's picture parts with that text instead: a document turn then costs no
    image tokens at all. Returns the SAME list object when there is nothing to do.
    """
    if not messages:
        return messages, 0
    from .mm_last_turn import _is_image                      # one notion of "a picture in a message", not two
    out = list(messages)
    dropped = 0
    if not keep_image:
        # every picture the pass could have read, wherever it sits: a screenshot replayed from history is read too now,
        # so leaving it in would cost the image tokens the mode exists to save (and mix a read with its source).
        for i in range(len(out)):
            content = (out[i] or {}).get("content")
            if not isinstance(content, list):
                continue
            keep = [p for p in content if not _is_image(p)]
            if len(keep) == len(content):
                continue
            out[i] = dict(out[i] or {}, content=keep)
            dropped += len(content) - len(keep)
    target = None
    for i in range(len(out) - 1, -1, -1):
        content = (out[i] or {}).get("content")
        if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "text" for p in content):
            target = i
            break
        if (out[i] or {}).get("role") == "user":
            target = i
            break
    if target is None:
        return out, dropped
    msg = dict(out[target] or {})
    content = [{"type": "text", "text": msg["content"]}] if isinstance(msg.get("content"), str) \
        else list(msg.get("content") or [])
    where = next((j for j, p in enumerate(content) if isinstance(p, dict) and p.get("type") == "text"), len(content))
    content.insert(where, {"type": "text", "text": block_of(text)})
    msg["content"] = content
    out[target] = msg
    return out, dropped


def log_line(pic: int, cuts: int, asked: int, cached: int, chars: int, dropped: int, ms: float, counts):
    """(level, message) for one image turn that went through the pass, or None when nothing did. An operator has
    to be able to tell "the pass ran and read N characters" from "the pass did not run" without a second probe --
    which is not a hypothetical: the first live version died on a call arity and asked zero questions on every
    request, and from the log that was indistinguishable from a picture with no text in it."""
    if not pic or not cuts:
        return None
    if not chars and asked:
        return "warning", (f"r9700 mm ocr: {pic} picture(s) cut into {cuts} line/band ask(s), {asked} asked, "
                           f"{cached} from cache, and NOTHING was recognized in {ms:.0f} ms -- the prompt is "
                           f"unchanged apart from an empty block (an image the model can read beats a pass that "
                           f"found nothing); running totals {counts}")
    return "info", (f"r9700 mm ocr: {pic} picture(s) -> {cuts} line/band ask(s) -> {chars} characters "
                    f"({asked} asked, {cached} served from cache) in {ms:.0f} ms; "
                    f"running totals {counts}")


def passes(request) -> bool:
    """Should this request go through the pass? Pure so the two ways it says no are testable without a vLLM: a
    band ask of our own coming back through the seam (recursion), and a launch that has the pass off."""
    return getattr(request, "user", None) != MARKER and mode() != "off"


async def _ask(self, request, part):
    """One narrow question about one crop, through this same serving object. The MARKER is what keeps the wrapper
    from recursing when the sub-request comes back through the seam."""
    cls = type(request)
    payload = {"model": getattr(request, "model", None),
               "messages": [{"role": "user", "content": [part, {"type": "text", "text": ASK}]}],
               "temperature": 0.0, "max_tokens": 512, "stream": False, "user": MARKER,
               # thinking OFF: with it on the ask can spend the whole budget reasoning and answer content='' (measured
               # 2026-11-21), which from here looks exactly like "the picture had no text in it". A read is not a
               # discussion -- notes/multimodal-input.md.
               "chat_template_kwargs": {"enable_thinking": False}}
    try:
        sub = cls.model_validate(payload)
    except Exception:                                    # noqa: BLE001 -- older/newer request models differ
        sub = cls(**{k: v for k, v in payload.items() if v is not None})
    res = await self.create_chat_completion(sub, None)
    try:
        return (res.choices[0].message.content or "").strip()
    except Exception:                                    # noqa: BLE001
        return ""


def _content(msg):
    """A message's content parts as a list, [] when it is a string or absent. At this layer the messages are the
    openai TypedDicts (plain dicts), which is why the pass reads `content` as a key -- and why it reads it through
    one helper instead of a dozen call sites that can each be wrong in their own way."""
    c = (msg or {}).get("content")
    return c if isinstance(c, list) else []


async def run(self, request, ask=None, decode=None, crop=None):
    """(messages, stats) -- the pass itself. `ask`/`decode`/`crop` are injectable for the same reason they always
    have been in this file: the host that runs the tests has neither PIL nor a vLLM. Passing them through the
    wrapper is what lets tests/test_mm_ocr_pass.py drive the WRAPPER, which is the only way the first live bug -- a
    call arity mistake between `wrapped` and `_run` that failed open on every image turn -- would have been caught.
    """
    return await _run(self, request, ask or _ask, decode, crop)


async def _run(self, request, ask, decode=None, crop=None):
    """The policy, with the three things it cannot be tested without mocked: the sub-request (`ask`), and the two
    PIL hooks (`decode`, `crop`) -- the host python that runs the tests has neither PIL nor a vLLM, exactly as in
    tests/test_mm_last_turn.py. Fails open: a picture whose bytes cannot be measured leaves the request untouched.
    """
    msgs = getattr(request, "messages", None)
    if not msgs:
        return msgs, None
    from .mm_last_turn import _is_image
    n, keep_img = band_count(), mode() != "text"
    decode = _decode if decode is None else decode
    crop = _crop_part if crop is None else crop
    # EVERY picture still in the request, not just the turn being answered. The client replays the conversation and a
    # screenshot rides in the message it was attached to; mm_last_turn keeps the newest one for exactly that case, and
    # a pass that only looked at the current turn would read nothing on the shape a chat client actually sends
    # (measured 2026-11-21: Zed's follow-up turn kept the picture in the prompt, the pass asked zero questions, and
    # the log said nothing because "nothing asked" and "the pass is broken" looked identical from the outside).
    spots = [(i, j) for i in range(len(msgs)) for j, p in enumerate(_content(msgs[i])) if _is_image(p)]
    if not spots:
        return msgs, None
    texts, asked, cached, cuts = [], 0, 0, 0
    loop = asyncio.get_running_loop()
    for i, j in spots:
        part = _content(msgs[i])[j]
        dec = await loop.run_in_executor(None, decode, _b64_of(part))
        if not dec:
            return msgs, None                       # cannot measure: leave the request exactly as it was
        raw, img = dec
        # lines where the page has lines (measured to be the reliable ask), equal bands where it does not (a photo,
        # a chart) -- and a picture whose structure cannot be read at all is left to the engine rather than guessed.
        slices = line_slices(await loop.run_in_executor(None, _row_profile, img)) or band_slices(img.height, n)
        cuts += len(slices)
        for bi, (y0, y1) in enumerate(slices):
            key = cache_key(raw, bi, y0, y1)
            got = cache_get(key)
            if got is not None:
                texts.append(got)
                cached += 1
                continue
            c = await loop.run_in_executor(None, crop, img, y0, y1)
            if c is None:
                continue
            asked += 1
            txt = await ask(self, request, c)
            cache_put(key, txt or "")
            texts.append(txt or "")
    text = merge(texts)
    if not text:
        return msgs, {"pictures": len(spots), "cuts": cuts, "asked": asked, "cached": cached, "chars": 0, "dropped": 0}
    msgs, dropped = inject(msgs, text, keep_img)
    return msgs, {"pictures": len(spots), "cuts": cuts, "asked": asked, "cached": cached, "chars": len(text),
                  "dropped": dropped}


def patch(cls=None, ask=None, decode=None, crop=None) -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    m = mode()
    if m == "off":
        logger.info("r9700 mm ocr: off (R9700/R9K_MM_OCR unset) -- pictures go to the model without an OCR pass")
        return False
    if cls is None:
        from . import gate
        gate.check("mm_ocr_pass")                    # the gate asks a real vLLM what it offers; a test host has none
        try:
            from vllm.entrypoints.openai.chat_completion import serving as S
        except Exception as e:                          # noqa: BLE001
            logger.warning("r9700 mm ocr: no chat serving module to patch: %r", e)
            return False
        cls = next((c for c in vars(S).values()
                    if isinstance(c, type) and "render_chat_request" in vars(c)), None)
    if cls is None:
        logger.warning("r9700 mm ocr: the chat serving class has no render_chat_request to wrap")
        return False
    orig = cls.render_chat_request

    async def wrapped(self, request):
        global _COUNTS
        if not passes(request):
            return await orig(self, request)            # a band ask of ours, or switched off mid-flight
        try:
            t0 = time.monotonic()
            msgs, st = await run(self, request, ask, decode, crop)
            if st and (st["asked"] or st["cached"]):
                request.messages = msgs
                for k in ("asks", "hits", "chars", "drops"):
                    _COUNTS[k] += st["asked"] if k == "asks" else (st["cached"] if k == "hits" else
                                                                   (st["chars"] if k == "chars" else st["dropped"]))
                line = log_line(st["pictures"], st["cuts"], st["asked"], st["cached"], st["chars"], st["dropped"],
                                (time.monotonic() - t0) * 1000.0, dict(_COUNTS))
                if line:
                    logger.log(logging.WARNING if line[0] == "warning" else logging.INFO, line[1])
        except Exception as e:                          # noqa: BLE001 -- a dropped request is worse than the bug
            _COUNTS["fails"] += 1
            logger.warning("r9700 mm ocr: leaving the request untouched: %r", e)
        return await orig(self, request)

    cls.render_chat_request = wrapped
    _PATCHED = True
    logger.info("r9700 mm ocr installed on %s.render_chat_request (mode=%s, one narrow ask per text line it can "
                "segment, %d band(s) as the fallback when it finds none, results cached per picture x cut in RAM; %s)",
                cls.__name__, mode(), band_count(),
                "the picture stays too" if mode() == "hybrid" else "the picture is replaced by its text")
    return True
