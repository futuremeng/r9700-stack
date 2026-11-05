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
