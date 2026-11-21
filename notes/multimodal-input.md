# Multimodal input: is it on, what does it cost, and what did measuring it cost us

Asked on 2026-11-04: *"目前的模型打开图像输入了吗?"* — is image input enabled on this model? The short answer is
**no, and it is a flag we pass, not a limit of the weights.** The longer answer needed a launch, because the cost
side was unknown and "it is probably fine" is the kind of sentence this repository keeps turning into a correction.

## The three facts that answer the question

1. **The running serve passes `--language-model-only`.** It is in the launch line (`serve/native-flashnext.sh`,
   inside `ARGS=(…)`), and it is position 26 of the live process's argv — readable without touching the box:
   `tr '\0' '\n' < /proc/$(pgrep -f 'vll[m] serve' | head -1)/cmdline | grep -n language`.
2. **The engine says so twice at boot**:
   `All limits of multimodal modalities supported by the model are set to 0, running in text-only mode`
   (`model.py:951`), and `Disabled mm_prefix attention mode because multimodal inputs are configuration-disabled`
   (`model.py:1008`). Note the wording of the second one: with images on, a different attention mode (`mm_prefix`)
   is what would run.
3. **An image request comes back 400 `At most 0 image(s) may be provided in one prompt. (parameter=image)`.** That
   error is the measurable form of the flag: it says the *limit* is zero, not that the model cannot see.

And the checkpoint disagrees with anyone who assumed the model is text-only: `config.json` carries `vision_config`,
`image_token_id`, `vision_start_token_id`, `vision_end_token_id`, and the directory holds
`preprocessor_config.json` **and** `video_preprocessor_config.json`. Architecture
`Qwen4ExpForConditionalGeneration`. Images and video are in the weights; the flag is the only thing keeping them out.

## The named test config (and its revert, proved)

`MM=1 MM_IMAGE=2` in `serve/native-flashnext.sh` swaps `--language-model-only` for
`--limit-mm-per-prompt {"image": 2, "video": 0}`. Run per notes/launch-methodology.md:

- session `r9700-native-TEST-mm1`, log `~/logs/TEST-mm1-1105-0007.log`, geometry otherwise identical to production
  (`MAXLEN=262144 NSEQ=32 NBT=8192`).
- Reverted at 00:15 by relaunching the untouched production command, and proved four ways: the live argv contains
  `--language-model-only` and not `--limit-mm-per-prompt`; an image request answers 400 `At most 0 image(s)` again;
  the KV pool is back to **1,242 blocks / 977,726 tokens**; `tools/recipe-check.py` returns
  `PRODUCTION recipe … every launch flag in the live argv matches`, exit 0.

## What images cost, measured

| | mm off (production) | mm on (`MM=1 MM_IMAGE=2`) |
|---|---:|---:|
| KV pool | 1,242 blocks / **977,726** tokens | 1,199 blocks / **943,875** tokens |
| cost of the vision tower + encoder profiling | — | **−33,851 prompt tokens of KV, −3.5%** |
| boot to `/health` 200 | ~200 s | ~260 s (**+60 s**) |
| preemptions, illegal-access lines, `EngineDeadError` | 0 / 0 / 0 | **0 / 0 / 0** |
| text gate (`bench/quality.py 10 8`), GSM8K / needles | 10/10, 100-question runs 97-99% / 3 of 3 | **10/10 / 3 of 3** |
| accepted tokens per MTP draft on that gate | 1.715 / 1.716 (hot), 1.740 (genuinely cold) | **1.725** |

So the answer to "does turning vision on slow the text path down" is: it takes 3.5% of the KV pool and 60 seconds of
boot, and on identical fixed prompts the decode side is indistinguishable (1.725 inside the 1.715-1.740 band of the
same gate with images off). It does not corrupt the language path: no crash, no preemption, no acceptance loss.

## What one image costs in prompt tokens

This is the part worth knowing before pointing a client at it, because the number is not intuitive:

| image | pixel size | `usage.prompt_tokens` |
|---|---|---:|
| question text alone | — | 68 |
| a 360 × 140 rendered PNG (a four-digit number, drawn by PIL) | 360 × 140 | 135 (**+67**) |
| same, two images in one request | 2 × 360 × 140 | 203 (**+135**, so 67.5 each) |
| a repo documentation PNG | 1,200 × 654 | 824 |
| a long screenshot from the model directory | **2,560 × 6,236** | **15,664** |

A single tall screenshot is **15,664 prompt tokens** — 6% of a 262,144-token window — and the boot log says the
encoder cache is `initialized with a budget of 16384 tokens`. One such image nearly fills the encoder cache, and
`--limit-mm-per-prompt` caps *how many* images, not *how big*: cabling a client to this box means deciding a pixel
budget (`--mm-processor-kwargs`) rather than trusting the default.

Did it actually see? On drawn-number images, yes, including the ordering of two images:

- one image, "what four-digit number is written here?" → **`4729`** (drawn 4729), 0.49 s wall, 77 tokens out;
- two images, "both numbers, comma separated, in order" → **`4729, 1371`**, 1.52 s;
- third image over the limit → 400 `At most 2 image(s)`, which is the limit doing its job;
- a documentation screenshot → *"A dark-themed data visualization dashboard displaying tokenized text strings and
  frequency bar charts"* — plausible, but this one is a judgement rather than a match, so it is recorded as weaker
  evidence than the digits.

**A trap for whoever writes the client**: with `--reasoning-parser qwen3`, a small `max_tokens` yields
`content: ""` and `finish_reason: length`, because the reasoning tokens consume the whole budget. Four of the first
probes looked like silent failures for exactly that reason. Read `reasoning_content` as well, or give the request a
real budget.

## Prefix caching and images: what the counters showed, and what is still unresolved

Repeating an identical image request is the cheapest prefix-cache probe, and the answer came out block-shaped
(`block_size` on this hybrid stack is **832**):

| repeated 3 × | prompt tokens each | `prompt_tokens_cached_total` delta | reading |
|---|---:|---:|---|
| image alone | 824 | **0** | under one 832-token block: nothing reusable |
| same image + ~1,300 tokens of text | 2,125 | 1,664 | 2 blocks |
| text only | 1,359 | **0** | one full block, still nothing reused |
| text only | 2,759 | 3,328 | 4 blocks over the three calls |
| the 15,664-token screenshot | 15,664 | 42,432 | 51 blocks × 832 — every call, including the first, hit blocks left resident by an earlier probe |

The honest summary: **reuse happens in whole 832-token blocks, and a prompt under roughly two blocks gets nothing.**
The off-by-one is not fully modelled here — 1,359 tokens contains one full block and reused zero, while 2,125 reused
two — so this is a measured table and not a rule. Anyone who needs the rule should read it out of the scheduler
rather than infer it from three calls.

## The one boot warning to keep an eye on

`WARNING [speculator.py:243] Draft model R9kQwen4ExpMTP does not support external multimodal embeddings.
Embeddings from the target model will not be passed …` Our MTP is internal, so this is about externally supplied
embeddings, and nothing in the measurements above looks worse for it — but it is the line to re-read if an mm + MTP
answer ever comes out wrong.

## What is *not* claimed

- No claim about vision **quality** on photographs: no photographs were available offline, and drawn digits plus one
  plausible caption is a pipeline test, not a benchmark.
- No claim about **video**: `video_preprocessor_config.json` exists and `MM_VIDEO` exists in the launcher, but the
  limit was left at 0 and nothing was sent.
- No claim that images are safe under **concurrency**: every probe above was one request at a time on an idle box.
  Encoder steps interleaving with 32 concurrent deep-context decodes — the situation this box actually lives in —
  is unmeasured, and that is the missing experiment before any client is pointed at `MM=1` in production.
- No claim about the **`mm_prefix` attention mode** beyond the boot log's statement that it exists and is off today.

## Reproduce

```
# what the flag does, without touching the box:
DRY=1 bash serve/native-flashnext.sh | grep -m1 'launch:'
DRY=1 MM=1 MM_IMAGE=2 bash serve/native-flashnext.sh | grep -m1 'launch:'
# is the live serve the one this tree would start (flags included)?
tools/recipe-check.py
# is the running box answering image requests at all? (a 400 is the honest answer when it should not)
# see the argv probe above, or POST one image and read the code
```

Decision as of 2026-11-05: **the default stays text-only.** `MM=1` is now a named, reversible switch that the gate
can see, images demonstrably work, and the cost is 3.5% of the KV pool plus 60 seconds of boot. What is missing
before production is the concurrency experiment, not the flag.

## Is there a size cap? (asked next, and the answer used to be "no")

`--limit-mm-per-prompt` caps **how many** images a prompt may carry. It says nothing about what **one** image may
cost, and the checkpoint's own default is enormous: `preprocessor_config.json` has

```json
"size": { "longest_edge": 16777216, "shortest_edge": 65536 }, "patch_size": 16, "merge_size": 2
```

With `patch 16 × merge 2`, one image token covers `32 × 32 = 1,024` pixels, so `longest_edge` is
**16,777,216 px = 16,384 image tokens per image** — which is exactly the number in the boot line
`Encoder cache will be initialized with a budget of 16384 tokens`. That is where the 15,664-token screenshot came
from: nothing was capping it.

`MM_MAX_PIX` (default `4194304` px ≈ 4,096 image tokens) now passes

```
--mm-processor-kwargs {"size": {"longest_edge": 4194304, "shortest_edge": 65536}}
```

vLLM folds that into the resize (`_get_vision_size`: `size.longest_edge` → `max_pixels`, `shortest_edge` →
`min_pixels`, `factor = patch × merge = 32`). The arithmetic is predictable offline, and it matched: the
2,560 × 6,236 canvas is predicted at 4,059 image tokens under the cap and 15,600 without, and the live
`usage.prompt_tokens` read 4,123 and 15,664 — the ~64-token difference in both cases being the question text and
the chat template.

### Measured, on a named test config (`r9700-native-TEST-mmcap`, log `TEST-mmcap-1105-0043.log`, `MM=1 MM_IMAGE=2`)

| | mm on, uncapped | mm on, capped at 4,096 tokens/image | production (mm off) |
|---|---:|---:|---:|
| boot encoder-cache budget | 16,384 tokens, 1 item profiled | **8,192 tokens, 2 items profiled** | — |
| the 2,560 × 6,236 screenshot | **15,664** prompt tokens | **4,123** prompt tokens | refused (400) |
| KV pool | 1,199 blocks / 943,875 | 1,188 blocks / 935,216 | 1,242 / 977,726 |

- **The cap does not buy back KV.** 935,216 with the cap against 943,875 without it: slightly *less*, because the
  encoder now profiles two items instead of one and the peak it reserves moves with that. Anyone hoping a smaller
  image budget returns the 3.5% of the pool that vision took away should read this row again. What the cap bounds is
  **what one image costs a prompt**, and the encoder cache — which is what protects a 262,144-token window from a
  single screenshot.
- **Detail survives at this cap, in the sizes tested.** Three 2,560 × 6,236 canvases (so the same rescale as the
  screenshot: down to ~1,312 × 3,168) with a six-digit number drawn at **28 px, 48 px, 96 px** were all read
  correctly: `308714`, `519264`, `740153`. The 28 px digits land at roughly 14 px after the cap's rescale, and the
  model still got them. That is evidence the cap is not costing legibility at document-screenshot scale; it is not
  evidence about a 6 pt font in a 4K dashboard.
- **The cap is a ceiling, not a suggestion a client can lift.** A request carrying
  `"mm_processor_kwargs": {"size": {"longest_edge": 16777216, ...}}` is refused:
  `Per-request mm_processor_kwargs are disabled by default because they can change multimodal preprocessing
  resource usage.` vLLM has a server flag to allow it; this stack does not use it, so the launch flag is the last
  word. That is the right default for a shared box: a client should not be able to decide how much of the context
  window its own image eats.
- Worst case per request is now arithmetic rather than luck: `MM_IMAGE × MM_MAX_PIX / 1024` image tokens. The
  default is `4 × 4,096 = 16,384` (6.2% of a 262,144 window); the test used `MM_IMAGE=2`, i.e. 8,192, which is why
  its encoder budget read 8,192.

Reverted and proved again: argv carries `--language-model-only` and neither `--limit-mm-per-prompt` nor
`--mm-processor-kwargs`; the image request is back to 400 `At most 0 image(s)`; KV pool 1,242 blocks / 977,726
tokens; `tools/recipe-check.py` exit 0 naming all three flags as matching. A size cap is a **flag**, so the flag
face of the gate is what catches it — and `MM_MAX_PIX`/`MM_MIN_PIX` are in `OPERATOR_KNOBS`, so the shell that asks
the question cannot move the answer.

## Where the box answers this today: the About page's Image input card

`/about` has a card for it, because the engine cannot answer it: **0** of the labels the engine publishes mention
multimodal (`mmLabels` in `serve/web/about.js` counts them, and the card prints that count, so if a future vLLM
starts publishing the limits the card says so instead of staying silent).

Two rows, and they name their witness:

- **The running serve accepts** — read from the live process's argv, which the merged plugin publishes: today
  `no images, no video (--language-model-only)`. On a **sidecar** page this row changes to "whose command line this
  is: the sidecar's … it proves nothing about what the serve accepts", because a sidecar holds its own argv. Same
  rule as the Configuration card: a sidecar's `MAXLEN` is not the engine's.
- **The recipe file records** — the *effective* values, not the nominal ones: `off · no image can be sent at all`,
  or `on · image 2, video 0 · one image up to 4,194,304 px (~4,096 tokens), floor 65,536 px`, plus the knobs
  (`MM_IMAGE=… MM_VIDEO=… MM_MAX_PIX=… MM_MIN_PIX=…`) so the page can say what `MM=1` would do without implying it
  is on.

`px per image token: 1,024 (patch 16 x merge 2 = 32 px a side)` is printed because a pixel cap is not a cost until
the token basis is next to it. That row is the one static claim on the card — it comes from this checkpoint's
`preprocessor_config.json`, and it says so.

**One stale row to be honest about:** the granular keys (`mm_image`, `mm_max_pix`, …) land in the recipe file from
the launch *after* this change. The launch running when this was written records only `mm` and `mm_limits`, so its
card says `off` and stops there rather than inventing the rest. Refreshing it by hand (`METAONLY=1`) would record
the environment of the shell that ran it, which is exactly how a recipe file starts lying, so it was not refreshed.

Two things the fixture work taught, both kept in `tests/render-about.mjs`:

- A saved payload can be the bare `/api/status` JSON or the page's own `{status: …, host: …}` dump. Reading
  `FIX.status` in one place and `FIX` in another made a fixture render an **empty** status: the page came up with
  blank cards and the checker was inspecting a different object from the one the page saw. Now `PAY` is normalized
  once and both the page stub and the checks use it.
- `/api/host` is a **separate endpoint**; a bare `/api/status` payload saved from the merged serve does not contain
  it. Handing the fixture a 404 for it blanked the whole page (only `error` and `updated` rendered), which is also
  a fact about the page: it treats a 404 on that endpoint as fatal. File mode now answers `{}` for a fixture with
  no host block; URL mode still passes the real 404 through so the page's behaviour stays visible.

## The default flipped: images are ON from 2026-11-05 01:19

`serve/native-flashnext.sh` now has `: ${MM:=1}`. `MM=0` is the way back and it is a restart, not a code change, and
`tools/recipe-check.py` sees the choice because it compares launch **flags** (section 13 of
notes/launch-methodology.md) -- nothing in `vllm:*` does.

The claim "text-only stays the default" earlier in this file is superseded by this section. It was the right claim
when the only numbers were the ones above; what changed the decision was having the cost measured, not new
information about the benefit.

### Measured on the production launch itself, not a test config

| | images off (00:49 launch) | images on (01:19 launch) |
|---|---:|---:|
| KV pool | 1,242 blocks / **977,726** tokens | 1,188 blocks / **935,216** tokens (**-4.35%**) |
| `Maximum concurrency for 262,144 tokens per request` | **3.73x** | **3.57x** |
| boot to `/health` 200 | 200 s / 220 s | **200 s** |
| encoder cache budget | -- (mm disabled) | **8,192 tokens, profiled with 2 image items** |

Two things in that table correct earlier claims in this file:

- **Boot cost was +60 s and is not.** mm-on has now been timed at 260 s and at 200 s; mm-off at 200 s and 220 s. The
  spread swamps the difference, so the honest statement is "boot time does not move measurably".
- **`MM_IMAGE=4` does not mean 4 images fit one step.** The encoder budget came up as 8,192 tokens with **2** items
  profiled even though the cap allows 4 x 4,096. The bound is `max_num_batched_tokens` (8,192), not the mm limits:
  you may *send* four images and the encoder will process them **two at a time**. That is a scheduling fact worth
  knowing before anyone reads "image: 4" as "four images land in one step".

### What the box does now, checked on production

- one drawn number → **`4729`** (133 prompt tokens); two → **`4729, 1371`**; four →
  **`1113, 2227, 3341, 4458`**, all correct, in order; a fifth → `400 At most 4 image(s) may be provided in one
  prompt`, which is the count cap doing its job.
- the 2,560 x 6,236 screenshot → **4,123 prompt tokens** (it was 15,664 uncapped) and a plausible caption.
- the **reasoning-parser trap fired again**: the four-image probe first came back with `content: ""` at
  `max_tokens=400`. With a 1,200-token budget it returned `finish_reason: stop`, 457 generated tokens and 1,188
  characters of reasoning, and the right answer. Anything wiring images to this box must budget for reasoning or
  read `reasoning_content`, or it will believe the model is blind.
- the text path did not move: GSM8K 10/10, needles 3/3, **1.741 accepted tokens per MTP draft** (the same gate gives
  1.715-1.740 with images off), decode 128.1 tok per stream-second, preemptions **0**.
- the deep-context path did not move either, which matters because enabling mm switches on a different attention
  mode: a **150,057**-token needle prompt took **19.33 s** to first token at **7,764 tok/s** prefill and the needle
  hit; the same prompt again took **0.62 s** (241,004 tok/s, prefix cache). Those are the same numbers the
  cache-on ladder in README reports for 150k (19.32 s, 7,768 tok/s), so mm has not touched them.

### What this default is knowingly taking on

**Images under concurrency are still unmeasured.** Every image number above is one request at a time. What has not
been watched yet: encoder steps interleaving with a box already carrying 32 concurrent deep-context decodes, and
whether `mm_prefix` (the mode that boots on with mm) changes decode-step timing under that load. The decode card on
the Monitor is the instrument: `rate-gen` per wall second and `itl_ms` should be watched against the
`15.5 ms / ~128 tok per stream-second` recorded here. If they move, `MM=0` and a restart is the whole rollback.

And one capacity number to keep straight: production's KV pool is now **935,216 tokens**, so the KV figures quoted
throughout this repository for older launches (540,202 / 977,726) are history, not this launch.

## A page that quotes the tree's default has to read it from somewhere (found on the post-reboot launch)

The About card used to end its knobs row with a hardcoded `MM=0 is this tree's default`. That sentence was written
when it was true, had no test asserting it, and survived the default flipping to `MM=1` -- the live page was still
saying it an hour later. Same class of bug as the recipe file that was read once at startup: a fact copied into a
place that never re-reads its source.

So the launcher now has a name for it -- `: ${MM_DEFAULT:=1}` feeding `: ${MM:=$MM_DEFAULT}` -- and records
`mm_default` in the recipe. `MM_DEFAULT` went into `OPERATOR_KNOBS` in `tools/recipe-check.py` for the same reason
`MM` did: it decides what an unset `MM` means, so it is part of the recipe, not a comment. The page prints
`this tree's default is MM=<recipe value>` when the recipe says so, and
`this recipe does not record the tree's default, so this page does not state one` when it does not -- which is the
state of the running launch right now, since its recipe was written before this change and will be rewritten by the
next restart. No hand-refresh: `METAONLY=1` records the environment of the shell that runs it, and this shell is not
the launch environment.

Two checks keep the two ends joined: the recipe's `mm_default` must equal the launcher's own `:=` value, and the
launch line for an unset `MM` must be byte-identical to the line for `MM=<that default>` -- one source of truth
rather than a recorded default and an applied default that can drift apart.

## 2026-11-09: a chat client replays its history, so the engine was encoding the same pictures again every turn

The default flipped to `MM=1` (images accepted) with `MM_IMAGE=4` and `MM_MAX_PIX=4,194,304`, and the first thing the
multi-turn shape showed was that "accept images" and "process the images of this turn" are different statements.
A client that keeps a conversation sends the whole transcript on every turn, so turn N carries N-1 pictures again,
and stock vLLM has no notion of what is new: by the time the engine sees the request the transcript is one flat token
sequence. Two consequences, both measured on this box before any patch was in place
(`bench/mmturns.py`, against the standing 262k launch):

```
turns   three messages: 2 images in the first, 1 in the last   http 200  prompt_tokens 297
five    one message carrying 5 images                           http 400  At most 4 image(s) may be provided in one prompt
```

The first line pays for three images when one was asked about; the second is a hard refusal of a request whose newest
content is a single picture, because `--limit-mm-per-prompt` counts images **per prompt** and the replayed history
reaches the ceiling by itself. So the bug is not a nicety: past four pictures in a conversation the endpoint stops
answering at all, and each stale picture costs a vision-tower step plus its image tokens (`MM_MAX_PIX` caps one image
at ~4,096 tokens; this fixture's 240x160 PNG costs 72 tokens each).

`r9700_vllm/compat/mm_last_turn.py` wraps the one method every chat request passes --
`OpenAIServingChat.render_chat_request` -- before the transcript is rendered. It keeps the pictures of the last
submitted turn (the last `user` message, plus anything after it, where tool results carrying a screenshot live),
drops the image parts of everything older, and leaves one short text part in a message that would otherwise render
as an empty turn. Within the kept turn the **newest** `N` survive, `N` read from the engine's own
`limit_per_prompt["image"]` rather than a second copy of the number, so six pictures become four answered requests
instead of a 400. Re-run against the same launch with the patch installed:

```
turns   http 200  prompt_tokens 153      (297 -> 153: 144 tokens = the 2 replayed images, not encoded)
five    http 200  prompt_tokens 346      (was 400; the newest 4 of the 5 are what the model sees)
```

and the serve says it out loud the first time it happens:

```
r9700 mm last-turn: dropped 2 image(s) replayed from history, kept 1 from the turn being answered
(R9K_MM_LAST_TURN=0 restores stock behavior)
```

The knob is baked into `serve/native-flashnext.sh` (`: ${R9K_MM_LAST_TURN:=1}`, exported, so `tools/recipe-check.py`
holds the running engine to it) and into the container path's `serve/serve.sh`, which also gained the same image
defaults the native launcher has -- it was still passing `--language-model-only` by default, i.e. the container path
was text-only while the native one accepted pictures.

**What the trade costs, stated plainly:** a picture from an earlier turn is not visible any more. "What was in the
first screenshot?" needs the client to send it again, and that is the deliberate choice -- keeping every old picture
is what made the endpoint refuse at four. `R9K_MM_LAST_TURN=0` restores the old behavior for anybody who wants the
history visible and pays tokens for it.

Unmeasured, and worth knowing before pointing a heavy vision client at this: everything above is one request at a
time on an idle box. Vision-tower steps interleaving with 32 concurrent deep-context decodes is still unmeasured
(the same gap the `MM=1` default took on), and `mm_processor_cache` behavior across many distinct pictures in a long
conversation has not been looked at.

## 2026-11-21: the size cap becomes ONE budget the last turn's pictures share -- and one image may take it whole

The incident: a low-resolution screenshot of small Chinese text came back unreadable -- the model saw the layout
but not the characters. Re-sending the same text as a high-resolution image recognized every character, which
localized the failure to pixels, not fonts and not the chat client: at the standing 4,194,304 px PER-IMAGE cap a
capture larger than ~4,096 image tokens was resampled DOWN before the vision tower saw it, and small glyphs do
not survive that. (This section's cap column already carried the evidence: the 2,560x6,236 screenshot arrived at
15,664 tokens uncapped and 4,123 capped; the 28 px digits that survived were never evidence about 6 pt text.)

Two policy changes, both riding on the seam `r9700_vllm/compat/mm_last_turn.py` already owns:

1. **The count cap is last-turn-only** -- in force since 2026-11-09 and now stated as policy: `MM_IMAGE=4` applies
   to the pictures of the turn being answered, never to the replayed history, and history images are dropped
   before anything measures or encodes them. (Anthropic is the useful contrast: every turn's images stay visible
   in its payload model, re-sent and re-billed each turn -- dropping history is this box's cost trade, not the
   industry default.)
2. **The pixel cap becomes a SHARED budget.** `MM_MAX_PIX` goes 4,194,304 -> 16,777,216 -- back to the
   checkpoint's own `preprocessor_config.json` `size.longest_edge` -- and it now bounds what one turn's kept
   pictures cost TOGETHER: one image may take the whole budget and is never resampled (a full-HD 16.7M-px
   screenshot passes through byte-for-byte intact), and from k >= 2 up, when the kept turn's total exceeds the
   budget, each oversized picture is shrunk to budget/k in the compat layer (data: URLs only, dimensions floored
   to the 32 px patch grid, re-encoded lossless PNG -- this step exists because resampling hurt drawn text, it
   must not also add JPEG artifacts; anything undecodable is left untouched and the engine's cap stands).
   The worst case did not grow: 4 x 4,096 = 16,384 image tokens then, 16,384 now, paid jointly. `R9K_MM_TOTAL_PIX`
   overrides the budget; normally it reads the engine's own `mm_processor_kwargs` cap back, so there is one number.

How the others do it (fetched 2026-11-21 unless marked; general knowledge is labelled):

| engine | count limit | size limit |
|---|---|---|
| vLLM stock | `--limit-mm-per-prompt` per PROMPT -- the very bug this patch fixes; per-request kwargs refused unless `--trust-request-mm-kwargs` | server-wide `max_pixels` PER IMAGE (this box: one number, shared by policy here) |
| SGLang | `--limit-mm-data-per-request` JSON | `--mm-process-config '{"image":{"max_pixels":...}}'`, per modality; 64 MiB download cap, `--allowed-media-domains` |
| Anthropic API | 10-600 images/request; >20 images in one request tightens EVERY image to a 3,000 px long edge | per-image tiers (1,568 px standard; 2,576 px on 4.7+ models); docs warn lossy pre-compression hurts text |
| OpenAI (general knowledge) | per request | `detail=auto` tiles an image at 768 px then 512 px tiles -- many tokens for a big one, none for a small one |
| Gemini (general knowledge) | per request | ~258 fixed tokens/image (768x768) -- small text is a known loss |

Nobody counts only the last turn -- the last-turn scoping is this stack's own; Anthropic's >20-images tightening
and this budget/k are the same shape: the more pictures a request carries, the smaller each may stay.

Unmeasured after this change, in order of what to look at first: the PIL decode/resize inside
`render_chat_request` adds latency only when a turn keeps k >= 2 oversized pictures (k = 1 never decodes here --
the raised engine cap makes it a no-op); the boot encoder-cache reservation returns to 16,384 tokens
("1 item profiled" -- the uncapped column above is the evidence that number runs fine); and both launchers'
new default wants one live soak before the About-page wording reads as settled: it says "shared by one turn"
because that is what the code enforces, not what has been measured under load.

## 2026-11-21 (later the same day): the size cap was never the Chinese-text story, and a follow-up turn had no picture in it at all

Asked from a Mac: *"图片识别有问题，汉字都认不出来，最近改了图片压缩策略还是不行"*. The section above was written on
the assumption that the 4,194,304 px per-image cap had blurred a screenshot's small Chinese text. Re-measuring on the
live serve (Qwen3.8-Flash-Next, `--mm-processor-kwargs` carrying the checkpoint's own 16,777,216 px) says that
assumption was wrong, and that a different thing was happening at the same time.

**1. The pixel budget is not involved.** The raised cap is in the argv and it does what it says: a 1,200 x 400 PNG
arrives as +458 image tokens (480,000 px / 1,024 = 469, smart_resize to the 32 px grid), a 3,200 x 1,280 PNG as
+4,000, and `adapt_pixels` never fires for a lone picture. What kills the old theory outright is the size sweep: the
same four CJK words drawn at **16, 32, 64, 128 and 256 px** (63 -> 4,000 image tokens) do not get read better as
they get bigger -- the 256 px row was the *worst* of the five. "More pixels fixed it" was one lucky sample.

**2. The picture is often simply not in the prompt.** `mm_last_turn.py` keeps only the last submitted turn's
pictures, and a chat client (here: Zed's agent, which replays the transcript) attaches the screenshot to the message
it was typed in and asks the follow-up one message later. Measured, same screenshot, same question:

| the request | `usage.prompt_tokens` | what came back |
|---|---:|---|
| screenshot + question in ONE user message | **232** | (the transcription quality problem, below) |
| screenshot in turn 1, the question asked on turn 2 | **84** | *"我当前没有看到可读取的图片内容，无法逐字转录。请重新上传图片"* |
| the same picture re-attached on the turn that asks | 254 | a real answer about a real picture |

84 is the question with nothing else: the picture is gone. The engine said so out loud in seven launches
(`r9700 mm last-turn: dropped ... kept N`) and in **three of them `kept 0`** -- a request carrying images that
reached the model with none. Those lines were also the *first* one per launch; everything after ran at DEBUG, which
is invisible at the default log level, so the box had this happening and was not saying it.

**3. What the model can and cannot do with a CJK glyph it is actually shown.** Discriminative tasks pass, free-form
transcription does not:

| one character drawn at 260 px, asked five ways | 猫 | 梨 |
|---|---|---|
| 偏旁? | **犭** ✓ | **禾** ✓ |
| 右边? | **苗** ✓ | (finish=length, empty content) |
| 拼音? | **māo** ✓ | **lí** ✓ |
| 几画? | **11** ✓ | **11** ✓ |
| 组词? | **小猫** ✓ — it can *emit* the character | **梨子** ✓ |
| 四选一 / 是·否 | 3/4 · 6/6 ✓ | |
| **"只输出这个字"** | **猛** ✗ | (empty) |

Latin OCR is fine at the same and at smaller sizes (14 px read exactly: `HELLO VISION 4729`), and Chinese *text*
ability is fine (组词, 翻译). So the vision tower, the mm positions and the CJK vocabulary are all working; what
fails is the see-it-then-write-that-glyph step. **One controllable input flips it**: putting the image *before* the
text in the same message turned `猛` into `猫` on the identical image and question. Part order is worth a probe of
its own (owed, below).

**4. The host has no CJK font.** `fc-list :lang=zh` is empty and `bench/mmvl.py` looks for DejaVu only, so any
"Chinese" image this repo generates for itself is tofu. Every CJK number above was drawn with `wqy-microhei.ttc`
downloaded for the purpose; a probe that silently falls back to DejaVu would have "measured" blank boxes.

## What changed because of it

`r9700_vllm/compat/mm_last_turn.py`, third policy, one knob:

* **A turn that carries no picture keeps the newest replayed one.** `R9K_MM_HISTORY_IMAGE=N`, default **1**, `0` is
  the strict rule back. Still bounded by `--limit-mm-per-prompt` (the fallback cannot lift the engine's own count
  cap) and by the per-turn pixel budget, so the worst case grows by exactly one picture.
* **Nothing disappears in silence.** Every message that loses pictures now carries one text part saying how many,
  whether or not the message still has its text -- before, only an image-only message got the placeholder, so
  "看这张截图" survived and the screenshot behind it did not, which reads to the model as a picture it is looking at.
* **The log answers "did it fire on THAT turn".** Every image-bearing request logs dropped/kept at INFO with a
  running total, and the case where nothing is left in the prompt is a WARNING naming the knob.

Not claimed: that this makes the model transcribe Chinese well. It removes the class where the picture was not
there at all; the `猫 -> 猛` class is a model-side limit and the levers found so far are the prompt (candidates,
not transcription: 3/4 and 6/6) and part order.

Owed, in order: a `bench/mm_cjk.py` that fails when no CJK font is installed rather than drawing tofu; a part-order
probe (image-first vs text-first, N images x M sizes); the mrope/position question if order turns out to matter; and
`tools/recipe-check.py` will report `env:R9K_MM_HISTORY_IMAGE` as a deviation until the box is relaunched, because
the tree now sets a knob the running process was not started with.

## 2026-11-21 (evening): dropping a replayed picture costs the WHOLE replay's prefix cache -- measured, and the fallback buys it back

Two rounds from the client, read off the engine's own `vllm:prompt_tokens_cached_total` and `usage` (17.8k-token
prompt, one screenshot in the first user message, i.e. everything else sits *after* it):

| the request | `prompt_tokens` | cached | wall |
|---|---:|---:|---:|
| A cold, picture in this turn | 17,834 | **0** | 2.13 s |
| B text-only follow-up -- the fallback keeps the replayed picture | 17,853 | **16,640 (93.2%)** | **0.35 s** |
| C follow-up carrying a NEW picture -- the old one is dropped | 17,868 | **0** | **2.09 s** |
| D B again, verbatim | 17,853 | 16,640 (93.2%) | 0.35 s |

The mechanism is not alignment, it is the hash chain: `generate_block_hash_extra_keys` folds the mm feature
identifier into the block hash, and a block's hash chains its parent -- so removing 148 image tokens near the start
changes every block after it, and the replay is cold all the way down. **A dropped picture is therefore not free:**
it is a full re-prefill of everything that follows it (this box's own client: 35,607 tokens, TTFT 3.97 s, where the
same transcript with its picture kept ran in 0.35 s at 93% reuse). And the new fallback is, on top of being correct,
the cheaper option in the shape it covers.

The instrument catch, because it hid this for one run: the first version of the probe built its prefix from one
sentence repeated 700 times. A periodic prefix matches *itself* under a token shift, so all four rows came back at
92% reuse and the effect was invisible. A prefix-cache probe needs unique content per block.

## 2026-11-21 (later): the per-picture FLOOR was a real OCR failure -- it just is not the ceiling

The client sent a TIME-cover-style image and the answer described the layout perfectly (red border, the head, the
barcode strip) while every word came out as repeated thin strokes (`l l l l`, and in my own probe `i il i il i`).
The same card drawn by hand reproduces it, and the first thing that changes the outcome is not the font, the
quantization or the history filter -- it is how many image tokens the picture arrives with.

One card, three lines of 34 px text, same bytes, only the scale changing (`temperature 0`, `MM_MAX_PIX=16777216`):

| sent | image tokens (approx) | read |
|---|---:|---|
| 1,400 x 560 = 0.78 Mpx | 765 | `'i il i il i il i \| il i il i il i \| ... 18.6% \| I IL L VI IL 4729'` |
| 2,800 x 1,120 = 3.14 Mpx | 3,062 | `'机 器 学 习 模 型 识 别 \| 合 同 需 求 月 底 前 签 署 \| 本 季 度 销 售 额 增 长 18.6% \| HELL0 W0RLD 4729'` |
| 4,200 x 1,680 = 7.06 Mpx | 6,878 | `'机机机机机机机机机机 \| 合回需在月床前签署 \| 本季售额增长18.6% \| HELL0 VI5I0N 4729'` |

So there is a cliff under ~1,024 image tokens and a soft worse zone above ~4,000. The checkpoint's own
`shortest_edge` of 65,536 px is **64 tokens**, i.e. a floor the model was never meant to be run at, and a client
that resizes pasted pictures down (the Zed client's screenshots arrive at 450-800 image tokens -- read off the
`prompt_tokens` deltas) lands right in the cliff. What the other engines serving this checkpoint do:

* llama.cpp prints it as advice: *"Qwen-VL models require at minimum 1024 image tokens to function correctly on
  grounding tasks ... try adding --image-min-tokens 1024"*.
* Strata's GPU vision runs `vision.max_tokens=1024`; its CPU reader, which defaults to `max_tokens: 300`, measures
  **CER 12.1% against 1.66%** on dense pages on this same model (their #5443), and their own #1530 shows even the
  GPU reader at 1024 dropping a character out of an eight-character confirmation code.
* their engine's defaults for this family are `min_pixels=4096, max_pixels=4194304` -- a 4,096-token ceiling, half
  of what this box lets a single picture cost.

`MM_MIN_PIX=1048576` (1,024 tokens) is therefore the shape the rest of the ecosystem uses. Price, measured on the
live serve with the floor on: a 400 x 300 picture costs **1,099 prompt tokens where it used to cost ~120**, and the
request runs 0.82 s. A 1,200 x 400 costs 1,127 and runs 1.07 s. Nothing about the ceiling changed.

## What the floor does NOT explain: a repetition collapse on long image-conditioned transcription

Same card, and the failure changes shape with the LENGTH OF THE ANSWER rather than with the picture:

* ask narrowly -- `只转录图片里的第一行` -- and it is exact: `机器学习模型识别`, 5 runs out of 5, 0.54 s.
* ask for all three lines and the same picture gives `'机机机习机机机机 | 合同需机月机前机 | 本季机销机额机机 18.6%'`;
  two lines gives a period-2 cycle (`机会机会机会…`); a Latin-only card at the same size gives `I A I I A I A I…`;
  with thinking on, the loop runs in the reasoning channel and the content comes back empty at
  `finish_reason=length` (2,500 tokens generated, nothing in `content`).
* **the text-only control is exact**: dictating the same three lines (no picture) returns them verbatim, "repeat
  this line three times" returns three correct copies, a passage required to contain 篡改 three times gets all
  three right, and a 60-character explanation of 机器学习 reads clean. The language model, its CJK table and the
  PLE n-gram path are not broken.

Two suspects were tested by turning them off and re-running the same probes (this box is running that experiment
now: `MTP=0` -- the launcher learned that spelling, since vLLM rejects `num_speculative_tokens: 0` -- plus
`R9K_QSA_KVF8=0 KVDTYPE=`, i.e. a bf16 main KV): **the collapse is unchanged.** It is not the drafter and it is
not the fp8 KV cache. The upstream fingerprint that fits best also does not apply here: vllm#55357 (same
`qwen4_exp` family, "correct first token, garbage from the second", 0% draft acceptance) traces to a strided
column view meeting a unit-stride PLE kernel, and the 0.32.0 wheel this box runs already passes the stride
(`state_idx_stride = state_indices.stride(0)` in `models/qwen4_exp/common/ops/ple.py`); this box's acceptance
counters read 9,010 accepted of 16,676 drafted (54%), not 0%.

Recorded as a suspect and not proven: this checkpoint's `aa_report.json` says `vision_sequences: 0` -- the
activation-aware FP8/MXFP4 calibration saw 1.9 M tokens of code, math, wiki and multilingual text and **not one
image** -- while its README claims 95.17% top-1 agreement with the bf16 model. The pack Strata serves from is
IQ3_S, which by their own yardstick agrees with Q8_0 on only 87.69% of positions, and it reads these pictures. So
quantization damage is a weak explanation of the gap; the picture A/B is the one that separates them, and the four
probe PNGs and the fixed question are kept in `~/vision-probes/` for exactly that.

## 2026-11-21 (night): the band, not the ceiling -- and the operator refused the expensive answer

The floor fix above was measured with `MTP=0` and a bf16 main KV cache, because that was the cheapest way to
disprove the two suspects the fingerprint pointed at. Refusing to run that as production is correct -- it costs
~18% of aggregate decode and halves the KV pool (1,021,931 -> 540,202 tokens, `notes/kv-fp8-qsa.md`) -- and it turns
out nothing had to be paid: the collapse is the same with and without the drafter and the fp8 KV, so the box went
back to `MTP=2` + `--kv-cache-dtype fp8_e4m3` and the ONLY permanent change is the token band.

`MM_MAX_PIX=4194304` (4,096 tokens, the number that was there before the same-day raise to the checkpoint's own
16,777,216) and `MM_MIN_PIX=1048576` (1,024 tokens). The raise is reversed because the top of the band is also
where the reading gets worse: the 4,200 x 1,680 card measured 6,878 tokens and answered `杌杌杌杌杌杌杌杌` per line,
and with the 4,096-token ceiling the same card costs 4,107 tokens and still repeats. Halving the ceiling also
halves what the encoder may hold, which is capacity the box keeps: the boot line reads
**"Encoder cache will be initialized with a budget of 8192 tokens"** where it read 16384.

What the band does NOT fix, written down because it is the remaining half of the complaint: at a FIXED 1,138-token
prompt (one 1,400 x 560 card, so the only variable is the height of the glyphs), transcription is not a function of
size at all --

| glyph height | ~tokens/字 | read |
|---:|---:|---|
| 18 px | 0.6 | `机器学模型识别 \| 合词需在月底前签署 \| 本季度销售额增长 18.6%` (2 errors) |
| 24 px | 0.8 | `机器模型 \| 台需月前言 \| 本消消长13.6` |
| 32 px | 1.0 | `111111111 \| 111111111111…` |
| 40-104 px | 1.2-3.2 | `机机机机…` per line |
| 144 px | 4.5 | `别别别… \| 辟辟辟… \| 谣谣谣…` (a different char per line, then a run) |

18 px is the best row and 32 px the worst: this is not acuity, it is an unstable decode -- whichever token it
commits first is the token it then emits for the rest of the line. Two hints for the next step, neither quantified
yet: (a) samples from the `MTP=0` run in the same band read *substantially better* than the `MTP=2` samples
(`机民器人学习机器人 \| 合月需在月底前签署 \| 本季度销售额增长18.6%` against `机机机…`), so the drafter may be an
amplifier even though it is not the origin -- a same-probe collapse-rate count with and without it is the test; and
(b) the other engine's numbers for the same model are in the same class but an order of magnitude rarer (their
#1530 drops one character in eight, their #5443 measures CER 1.66% at 1,024 tokens), so the four PNGs kept in
`~/vision-probes/` are meant to be run there unchanged before this box's path is blamed further.

## It was the vision tower being sliced into quarters (2026-11-21) — the answer to "this box cannot read pictures"

Everything above this line was measured with `--mm-encoder-tp-mode` at **vLLM's default, `weights`**, which splits
each vision-tower layer across the TP ranks. On this box that is four ranks. Changing that one flag to **`data`**
(replicate the tower on every rank, split the pictures instead) turns the failure off. Nothing else in the launch
changed in any of the runs below.

Same cards (`bench/mm_ocr_ab.py`, drawn Chinese, ground truth in the file), same questions, temperature 0, thinking
off, two reps per arm:

| launch | long transcription of the whole card, CER | collapsed | notes |
|---|---|---|---|
| `weights`, the pre-pass off | **96.8%** (dense 34 px) / **100%** (110 px) | 2/2 | `机机机机…`, `ｌｉ　ｌｉ　ｌｉ…`, empty answers |
| `weights`, the pre-pass on | **32.3%** | 0/4 | a readable answer built out of narrow per-line reads; ~1/3 of the characters still wrong |
| `data`, the pre-pass on | **0.0%** | 0/4 | exact, and every pre-pass arm exact too |
| `data`, the pre-pass off (what the box ships) | **0.0%** | 0/4 | `机器学习模型识别 / 合同需在月底前签署 / 本季度销售额增长 18.6%`, 0.38 s |

The four hand-drawn PNGs in `~/vision-probes/` — the ones this note spent days calling unreadable — now come back
`CER 0.0%` narrow *and* long, Chinese and Latin, through the plain engine path:

```
### zh-dense-1400x560.png   as-sent  narrow CER 0.0%  long CER 0.0%   pt 1138   0.41 s
### zh-dense-2800x1120.png  as-sent  narrow CER 0.0%  long CER 0.0%   pt 3147   0.51 s
### latin-2800x1120.png     as-sent  narrow CER 0.0%  long CER 0.0%   pt 3147   0.55 s
```

### What had to be ruled out first (all of it measured, all of it not the cause)

- **The quantized pack.** The checkpoint's visual tensors are BF16 and unquantized (333/333 in the first shard, no
  scale companions), and `aa_report.json` says `vision_sequences: 0` — suspicious, but not it.
- **The other engine's mmproj.** Its GGUF metadata is the *same* tower: `clip.projector_type=qwen3vl_merger`, 27
  blocks, hidden 1152, patch 16, merge 2, mean/std 0.5, projection_dim 2560 — the same 333 tensors this checkpoint
  carries, the same class vLLM builds at `vllm/models/qwen4_exp/amd/model.py:871`. There was never a better encoder
  on the other side of the comparison.
- **The pixel band, the resample filter, the placement of the picture.** Equal bands vs per-text-line crops, bicubic
  vs lanczos vs box vs nearest, 1,020 vs 3,080 image tokens: the output moves, none of it fixes the read.
- **The drafter and the KV dtype.** `MTP=0` and bf16 KV: the collapse stays. Spec-decode counters were healthy
  (~54% accepted), so the MTP+PLE index bug upstream describes (#55357) is not what this build has.
- **The sampler.** `no_repeat_ngram_size` 2/3/4 and `repetition_penalty` 1.05/1.15: the pattern changes shape
  (`合合同同床床在在前前`) and stays wrong — it is emitting repeated *characters*, not a repeated n-gram.
- **The language.** Text-only dictation of the same three lines, with the same "篡改 it three times" instructions, is
  exact. CJK vocabulary and the PLE/ngram path are fine.

### What is claimed and what is not

Proven: on this build, this checkpoint and TP4, `weights` gives 32.3% character error (96.8-100% without the pass)
and `data` gives 0.0%, with the rest of the launch byte-identical, both directions re-run.

Not proven: the mechanism. The tower is Qwen3-VL's — 27 blocks, hidden 1152, 16 heads so **head_dim 72**, position
embeddings bilinearly interpolated over a 48×48 grid, 3D RoPE, and a `deepstack` set of three extra feature maps
injected into the first LM layers (`model.py:495-517`). All of that has to be sliced correctly for a sharded tower to
be numerically equivalent, and stroke-soup-then-repeat is what a broken vision encoder looks like from the outside.
Nobody upstream reports it as an *accuracy* bug on RDNA4 — searching for it returns nothing — the closest thing is
vllm-project/vllm#49851, multimodal models failing to load on gfx1201, where vLLM's ROCm maintainer's answer was
`--mm-encoder-tp-mode data`. That the same knob erases the accuracy failure here is the finding.

The two leads left dangling at the end of the previous section are void: the drafter is not an amplifier (MTP off
changes nothing), and "the other engine reads the same PNGs better" is not a comparison of engines — it is a
comparison of a replicated tower against a sharded one.

### What it costs, measured in the pair of boots

```
weights  Available KV cache memory: 6.55 GiB   GPU KV cache size: 887,983 tokens   concurrency 3.39x
data     Available KV cache memory: 5.9  GiB   GPU KV cache size: 797,453 tokens   concurrency 3.04x
```

0.65 GiB of KV per rank — the tower exists on all four cards instead of a quarter of it on each. 10% of the pool for
pictures that are actually read: this box takes that. Not measured: whether `data` changes decode throughput or
time-to-first-token with a picture in the batch (the replicated tower encodes on every rank), images under
concurrency, photographs, video.

### What the box does now

- `MM_TP_MODE=data` is the launcher's default in `serve/native-flashnext.sh` and `serve/serve.sh`, with the table
  above in its comment; `weights` is one env away (`MM_TP_MODE=weights bash serve/serve.sh`).
- The OCR pre-pass is **gone** (removed the same day, kept in git at `6cc716c`). It was `r9700_vllm/compat/mm_ocr_pass.py`:
  one narrow ask per segmented text line (a row-profile segmentation; equal-height bands measured *worse* than the
  baseline, 99.8% / 8.5 s), thinking off, results cached per picture × cut, fail-open. Under `weights` it was the
  difference between garbage and a usable answer (96.8%/100% → 32.3%, collapses 2/2 → 0/2); under `data` every arm is
  already 0.0%, so it bought nothing and cost 40-90% more wall time on the picture turn. The strategy and its CER table
  live on in `bench/mm_ocr_ab.py`, which does the same thing client-side — cut, ask per line, paste the text in as an
  ordinary text part — so nothing about the measurement was lost with the plugin.
- `MM_EMBEDS=1` stays: `--enable-mm-embeds` lets a client send its own picture encoding. It is not a path to a better
  encoder for this arch — the LM needs merged embeds **plus** three deepstack feature maps and the wire carries one
  tensor — but it is the seam `bench/mm_encoder.py` uses to feed the tower's own output through, and it is measured
  to be what makes `image_embeds` requests answerable at all.
- The 1,024-token floor (`MM_MIN_PIX=1048576`) is still in place and now looks unnecessary — it was set to dodge a
  "stroke soup" that this flag explains. Retest at the checkpoint's own floor before changing it: send a small
  picture at `MM_MIN_PIX=65536` and re-run `bench/mm_preproc_ab.py`.
- The encoder cache line in the boot log still reads `a budget of 8192 tokens`.

### Reproduce

```bash
cd ~/r9700-stack
python3 bench/mm_ocr_ab.py --font /tmp/wqy.ttc --sizes 34,110 --repeat 2 --bands 4   # CER per arm
python3 bench/mm_preproc_ab.py --reps 2 --filters bicubic                            # the four PNGs
MM_TP_MODE=weights python3 setup.py --yes --launch native-flashnext.sh --webui same --config 256k   # the failure back
```

`bench/mm_ocr_ab.py` draws its cards with a CJK font and refuses to run without one (`--font /tmp/wqy.ttc`,
`R9K_CJK_FONT`): a box with no CJK font draws tofu and the probe measures nothing. `bench/mm_encoder.py` is the
out-of-engine tower runner (`transformers.models.qwen3_vl`, the checkpoint's 333 BF16 tensors, CPU) and needs
`LD_LIBRARY_PATH=/home/jiulu/opt/openmpi/usr/lib/x86_64-linux-gnu` to import torch at all.
