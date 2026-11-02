# 262,144 tokens of context on the 4×R9700 box (2026-11-02)

Asked: is there enough resource to run this checkpoint at its full context. `Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ`
says its own ceiling is `max_position_embeddings: 262144`, with `rope_parameters: {rope_type: default,
rope_theta: 10000000, partial_rotary_factor: 0.25, mrope_interleaved: true, mrope_section: [11, 11, 10]}` --
**no YaRN, no extrapolation knob to argue about**. The answer is yes, and the interesting part is what the
limit turns out to be: not the KV pool.

## Geometry: why 262k is cheap for this model

48 layers, `full_attention_interval: 4` → **12 full-attention layers, 36 linear-attention (GDN) layers**,
`num_key_value_heads: 2`, `head_dim: 256`, KV dtype auto (bf16):

```
attention KV = 12 x 2 x 256 x 2 (K,V) x 2 B = 24,576 B/token across the four ranks = 24 KiB a token
262,144 tokens = 6.00 GiB per full-length sequence (1.5 GiB a rank)
```

Measured pool sizes (`gpu_worker.py:759`, `kv_cache_utils.py:2327` in the launch log):

| launch | KV pool | vLLM's own concurrency line | attention block | kv cache group sizes |
|---|---|---|---|---|
| MAXLEN=32768 (the recipe up to now) | 497,588 tok (`num_gpu_blocks 1230`, `mamba_block_size 32768`) | 15.19× by the same arithmetic; the line was recorded as `15.2x at 32k` in [native-4xr9700.md](native-4xr9700.md) | 448 | not recorded |
| **MAXLEN=262144 NSEQ=32** | **540,202 tok** (546,391 on the identical run the day before) | **2.06× at 262,144** | 448 | `262144, 262144, 262144, 262144, 8, 448` |
| MAXLEN=262144 NSEQ=4 | 544,919 tok | 2.08× | 416 | `416, 416, 416, 416, 8, 416` |

Two things that cost nothing: the attention pool does **not** shrink when MAXLEN grows (the memory-profiling run
is bounded by `--max-num-batched-tokens`, not by MAXLEN, so raising MAXLEN from 32k to 262k left the pool
*bigger* than it was), and the GDN recurrent state is per-sequence, not per-token -- `interface.py:1096` sets
the attention block size (448, or 416 at NSEQ=4) purely so the attention page and the mamba page match exactly.
What it does cost: per-card VRAM in use went from ~29.4 GiB to **30.47-30.54 of 31.86 GiB**, so ~1.35 GiB of
headroom left on each card, and `Available KV cache memory` reads 7.31 GiB a rank.

## The retrieval ladder (`bench/needle_long.py`)

Haystacks sized against the checkpoint's own tokenizer (the filler compresses to ~0.65 tokens a word, so
counting words lands ~35% off), passcode inserted at 30-70% depth, a random nonce every ~40 sentences so a
prefix cache cannot reuse the earlier part. Cold rows are first-seen context; warm rows re-send the identical
haystack.

| prompt tokens | cold TTFT | cold prefill | result | same context again (PREFIX_CACHE=1) |
|---|---|---|---|---|
| 30,070 | 4.25 s | 7,068 tok/s | hit | 0.28 s |
| 67,063 | 7.73 s | 8,671 tok/s | hit | 0.29 s |
| 150,057 | 19.32 s | 7,768 tok/s | hit | 0.51 s |
| **250,078** | **36.52 s** | 6,847 tok/s | **hit** | **0.70 s** |

On the final NSEQ=32 / PREFIX_CACHE=0 launch: 250,061 tokens → 36.60 s, 6,832 tok/s, hit; and the same
quality gate that has gated every launch of this recipe still passes (GSM8K 10/10, `bench/quality.py`
needle 3/3 at 3.8k / 15.2k / 30.4k). Prefill degrades gently with length -- 9,097 tok/s at 30k (the recorded
32k number is 9.2k) against 6,832 at 250k, i.e. **-25% for 8× the context**, which is what 12 of 48 layers
attending plus the QSA indexer buys. A 400-token generation *inside* a 250k context decoded at 141 tok/s.

## What actually queues, and why

Fired from one process at once (separate processes stagger on their own tokenizer work, which fakes a
nicer-looking queue):

| load | wall | running peak | waiting peak | KV pool peak |
|---|---|---|---|---|
| 32 × 300 tok | 3.7-4.0 s (1,126-1,257 tok/s, wall clock from the issuing process, not the `bench/probe32.sh` window) | **32** | 0 | 20.7% |
| 8 × 30k tok | 26.2 s | **2** | 7 | **11.7%** |
| 3 × 150k tok | 59.1 s | 2 | 2 | 32.3% |
| 4 × 150k tok | 79.1 s | 2 | 3 | 32.3% |
| 3 × 250k tok | 109.2 s | 2 | 2 | 50.9% |

**Long prefills serialize, and that is the wall you hit first.** 8 × 30k arrives as TTFT
3.6 / 7.1 / 9.8 / 13.4 / 16.9 / 19.6 / 23.1 / 26.2 s -- an arithmetic series whose step, 3.3 s, is exactly one
30k prefill at 9,097 tok/s, and whose total (26.2 s) is 8 prefills end to end. The KV pool is **11.7% used**.
32 short requests all run at once because 32 × 300 tokens is about two 8192-token steps. So:

- The 2.06× from vLLM's startup line is a **residency** ceiling (how many full-length contexts can be *held*),
  not the concurrency the box delivers at 262k. With long prefills serialized the pool is never more than half
  full, and the binding constraint is prefill throughput (~6.8-9.1k tok/s aggregate).
- Above roughly 30k tokens per prompt, "more than 2 in flight" already means queueing -- but it is queueing for
  the prefill pipe, not exhaustion of memory.
- A third 250k request is **not** an error: it waits and succeeds (109 s TTFT). Nothing OOMed at any tier, and
  three concurrent 250k prefills peaked at 30.6 GiB a card.

`vllm:num_requests_waiting_by_reason` is worth reading with care, because its labels do not mean what they
look like. vLLM's own HELP text says "'capacity' = waiting for scheduling capacity; 'deferred' = deferred by
transient constraints (LoRA budget, KV transfer, blocked status). Sum of all reasons equals
`vllm:num_requests_waiting`". What the code actually assigns (`v1/metrics/loggers.py:1150-1159` at 0.32.0,
and `v1/core/sched/scheduler.py:2947-2948`): `waiting = num_waiting_reqs + num_skipped_waiting_reqs`, with
`capacity = num_waiting_reqs` (the ordinary queue) and `deferred = num_skipped_waiting_reqs` (requests the
scheduler passed over this step). So "capacity" is a name for the queue, not a claim about blocks: **neither
reason proves KV exhaustion**. The 8 × 30k load showed 7 "capacity" waiting with the pool at 11.7%, and the same
load re-sampled through the Monitor says the same thing more exactly: `bench/capture-cards.py` fired eight
concurrent prompts (the serve's own `prompt_tokens`: 29,984-29,994) at the standing 262k launch and kept the poll
with the longest queue -- at 2026-11-02 06:47:35 UTC, `running 2 · waiting 6 · capacity 6 · deferred 0` at KV
10.1% of a 540,644-token pool, with 8.8-10.2% across the seconds around it. Six requests waiting for a slot in
the schedule while the pool sits nearly empty. That kept poll is what the README's card figure draws, and its
`/api/status` payload is committed beside the figure (`docs/r9700-stack-monitor-cards.json`), so the sentence and
the picture are checkable against the same sample.
`vllm:kv_cache_usage_perc` has the opposite
failure: it read 32.3% during 4 × 150k, when the hybrid groups were the resource that was actually short. Read
`running`, `waiting` and the pool together, never one alone -- which is what the Requests card now prints.

## `--long-prefill-token-threshold`: it moves latency, it does not create capacity

0.32 exposes `long_prefill_token_threshold` (+ `_adaptive`, which floors it at `max_num_batched_tokens / queued
and running`). A/B against the same launch, one flag added (`EXTRA="--long-prefill-token-threshold 2048
--long-prefill-token-threshold-adaptive"`, everything else identical):

| load | threshold 0 (the recipe) | threshold 2048 + adaptive |
|---|---|---|
| 1 × 250k + 6 × 30k, the six 30k TTFT | 40.3 / 43.0 / 46.5 / 50.0 / 52.7 / 55.8 s | **7.8 / 16.2 / 16.2 / 22.3 / 27.2 / 27.7 s** |
| ...the 250k TTFT | 36.8 s | 56.8 s |
| ...wall | 55.9 s | 56.8 s |
| ...running peak / KV peak | 2 / 50.1% | 6 / 46.4% |
| 8 × 30k | wall 26.2 s, running 2, KV 11.7% | wall **26.3 s**, running **7**, KV 27.4% |
| 3 × 250k | wall 109.2 s, running 2, KV 50.9% | wall **133.6 s**, running 3, KV **99.1%** |

Read it as fairness, not speed: total prefill work is fixed, so the wall clock does not move (55.9 vs 56.8,
26.2 vs 26.3); the flag stops a 250k prompt from monopolising the 8192-token step, and six ordinary requests
start 2-5× sooner. **It costs you when every request is long**: three 250k prefills run in parallel only by
each holding its own KV, which drives the pool to 99.1% and makes the whole load 22% slower. Quality and
short-context throughput are untouched (GSM8K 10/10, needle 3/3, 32 × 300 wave 1,257 tok/s).

Recommendation as measured: leave it off in the 32k-or-262k production recipe; if mixed lengths arrive at once
(agent traffic behind the same endpoint as long documents), use **4096** -- `NBT/2`, which caps parallel long
prefills at two and so matches the 2.06× the memory can actually hold. 2048 over-parallelizes at 262k.
This box is left running the recipe without it.

## Re-measured on a later day, on the standing recipe (what the README's 262k table now quotes)

Same box, same recipe (`MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same`), launched fresh from this repo
after the Monitor changes, `~/logs/ctx262k-prefillrate-052720.log.out`; probe bands in
`~/logs/probe32-262k-054506.log`. Prefill and throughput, nothing else changed.

| tier (`bench/needle_long.py`, prefix cache off) | prompt tokens | cold TTFT | cold prefill | the same context again |
|---|---|---|---|---|
| 60k | 60,061 | 6.93 s | 8,672 tok/s | 6.87 s, 8,742 tok/s -- **no reuse, the prefill was repeated**, which is what `--no-enable-prefix-caching` means |
| 120k | 120,066 | 14.89 s | 8,062 tok/s | 14.89 s, 8,064 tok/s |
| 180k | 180,070 | 24.22 s | 7,436 tok/s | 24.28 s, 7,416 tok/s |

Every tier was found (needle hit, 7-token answer), and per-card VRAM high-water was 30.3-30.4 GiB at 120k and 180k.
The ladder now fills in between its ends on one launch: 8,672 / 8,062 / 7,436 tok/s at 60k / 120k / 180k, and 6,832
at 250k on the standing recipe. Read the shape rather than the endpoints as one series: the 30k figure quoted
elsewhere (9,097) comes from the 8 × 30k concurrency wave and not from `needle_long`, and the `PREFIX_CACHE=1`
ladder above reads 7,068 at 30k.

What the engine says about the same 60k request, from its own `FinishedRequestStats` (the Monitor's per-request
row): 60,061 prompt tokens over 6.74 s of prefill = **8,908 and 8,917 tok/s** on the two runs. That is ~2.5% above
the client-side 8,672 / 8,742: the client clock includes connect, the JSON body crossing the socket, and the
engine's queue, the engine's clock does not. Both are honest; they are just measuring from different places.

| `bench/probe32.sh` protocol | this launch (262k) | the 32k recipe's band, same protocol |
|---|---|---|
| 32 in flight, 300-token waves | 1,568.3 / 1,573.7 / 1,587.5 -> **1,576** | 1,598 (three samples), and 1,574 on another instance of the same recipe |
| 32 in flight, 1,200-token waves | 1,799.5 / 1,832.7 -> **1,816** | 1,823 |
| 16 in flight, 300-token waves | 1,133.1 / 1,135.8 / 1,142.7 -> **1,137** | 1,107 |

So the full 262,144-token context costs **nothing measurable** in short-request throughput: 1,576 at 32 × 300 sits
inside the spread between instances of the 32k recipe (1,574-1,598), and 1,816 at 1,200-token waves is at that
band's 1,823. The 1,126-1,257 tok/s row earlier in this note is not in conflict with that: it is a **wall-clock**
figure from the issuing process, while `probe32.sh` measures the `bench.aggregate(conc, max_tokens)` window. Quoting
the first against the second's band makes 262k look 25% slower than it is, which is why both rows now name their
protocol.

Power while `gpu_busy_percent` read 100: 132 samples, 179-303 W, **median 299 W a card** -- the same ~300 W ceiling
[notes/native-4xr9700.md](native-4xr9700.md) records for this box uncapped (299 W median, 300.0 W max there), so
the full context changes where the cards sit, not how hard they work. The standing gate on this launch:
GSM8K **99/100** with 33,909 prompt tokens in 40 s, and `bench/quality.py`'s needle 3/3 at 3,828 / 15,196 /
30,354 tokens. (The `GSM8K 10/10` line earlier in this note is the 10-question sample taken during the first 262k
launch; the 100-question gate is 99%, where every other launch of this recipe has been.)

## Real traffic through it, observed and not manufactured (2026-11-02 14:11–14:17 UTC)

Someone used the box. Actual concurrent tasks arrived on the standing 262k launch, and the only instrument pointed
at them was `tools/watch-live.py`, which **samples and never sends** — measuring this properly requires not adding
load to the very concurrency being read. What the serve reported, over 150 one-second samples and the 100 request
rows it still held (the journal held 279 back to 12:53):

| what | reading |
|---|---|
| prompt tokens, kept rows | p50 **53,855** · p95 149,913 · max 153,568 · 37% ≥ 60k, 14% ≥ 120k |
| generated tokens | p50 403 · p95 2,434 |
| wall clock per request | p50 25.6 s · p95 90.8 s |
| **first token** | p50 **8.5 s** · p95 **25.1 s** · max 35.6 s |
| decode, that request's own mean | p10 11.1 · p50 **33.1** · p90 116.4 tok/s |
| in flight | p50 **4** · p95 5 (of `NSEQ=32`) · waiting max 2 · **preemptions 0** |
| KV pool | p50 48% · p95 74% · max 78% of 540,202 tokens |
| engine step interval | p50 **951 ms** · p95 1,160 ms |
| aggregate generation | avg **58 tok/s** (0–416 second to second) |
| aggregate prefill, from the engine's own row timings | 6,410,492 prompt tok / 839 s = **7,638 tok/s** |
| cards | 100% busy all four · 2.5–3.4 GHz · 111–491 W · **card4 76–78 °C against 64–68 °C for the other three** |

**What it confirms.** "Long prefills serialize, and that is the wall you hit first" — with a vengeance: 7,638 tok/s
aggregate prefill sits inside the 6.8–9.1k band measured by the synthetic ladder, and a 150k prompt costs 21 s of
prefill before its first token. The step interval p50 of ~950 ms is one 8192-token chunk at that rate. Summed
per-request time splits 27% prefill / 73% decode, yet the decode seconds are stretched: a stream's decode shares
the pipe with other people's prefills.

**What it adds.** The synthetic loads in "What actually queues" peaked at **2** running for ≥30k prompts; real
traffic held **4–5** at once, with waiting never above 2 and the pool at 48–74%. Mixed lengths are the difference:
short requests slip between the long prefills, so the scheduler runs more than two while the prefill pipe stays
the constraint. Still no KV exhaustion and zero preemptions — the queue is a prefill queue, exactly as that
section argued.

**A reading trap this exposes in the Monitor.** Per-second live rates look awful under this load — per-stream
`stream_tok_s_live` p50 **1.8 tok/s**, aggregate often 0 — while the same requests report 33 tok/s each end to
end. Both are right: vLLM credits generation tokens **per engine step**, and a step here is ~1 s long, so a
1-second sampler aliases against the step cadence and lands between credits. The page's per-second sparkline is
therefore an *activity* trace, not a rate you should quote; the request row's `wallclock − first token` figure is
the rate. Worth remembering before anyone screenshots the aggregate card during long-context work.

**The flag this traffic was made for.** "If mixed lengths arrive at once (agent traffic behind the same endpoint
as long documents), use **4096**" — this is the first real traffic on the box that fits that description (prompt
p50 54k with a 14% tail at ≥120k, four to five at once). It was **not** switched: it needs a restart, the box had
15 boots that morning, and the gain it buys is TTFT fairness for the short requests, not throughput. Re-measure
when a restart is happening anyway.

The hardware side of the same window is in
[the incident note](../incident-2026-11-02/README.md): six correctable PCIe events in 34 minutes, five of them
while a 72k–144k prompt was prefilling — long-context work moves the link counters where short-request bursts do
not.

## Why the GLM-5.3-Flash 262k failure did not repeat here

[release-v0.3.7.md](release-v0.3.7.md) records GLM-5.3-Flash TP8 coming up at 262k and then dying inside
stock's prefill indexer (`sparse_attn_indexer_kpool -> rocm_fp8_mqa_logits`) at 150k even with `NBT=2048`.
That model compiles the same family of ops this one does, so the failure was worth expecting. It did not
appear: this checkpoint's QSA scoring runs through `r9700_vllm/attn/qsa_score.py`, which the module docstring
describes precisely because of that buffer -- stock "writes every column of an fp32 `[rows, capacity]` logits
buffer", which at `NBT=8192` and 262,144 capacity is **8.6 GiB a card** against 1.35 GiB of headroom, while ours
stages K per 16-row tile and stops at the tile's last visible column. The plugin's own scratch for the same
shape is a u32 `[rows, bw]` bitmap plus positions: 8192 × 586 × 4 B ≈ 19 MB. The condition to keep is
**`R9K_QSA_SCORE` unset** -- `R9K_QSA_SCORE=stock` puts the 8.6 GiB allocation back (that number is arithmetic
from the buffer shape, not a measured OOM).

## What these numbers do not say

- The haystack is four repeated sentences with nonces. A pass at 250k proves the path end to end (tokenizer →
  chat template → chunked prefill → hybrid KV → QSA → sampler) and nothing about prose recall; repeated text is
  also cheaper to attend to than real documents.
- 141 tok/s decode at depth is measured on that same repetitive context, where MTP acceptance is high. Do not
  quote it as long-document decode speed.
- One 30k TTFT sample read 11.58 s; taken again on an idle engine the same tier read 3.31 s. TTFT sampled right
  after a throughput wave is not a number to record.
- The pool figure moves by ~1% between identical launches (540,202 / 546,391 / 544,919 tokens). Nothing here
  explains that; don't read it as an effect.
- Not measured: `--kv-cache-dtype fp8` (halves the 24 KiB/token attention term for 12 of 48 layers, so less than
  a doubling of the residency ceiling -- the GDN state does not shrink), TP8, and any long-document (non-repetitive)
  recall quality at 262k.

## Reproduce

```
cd ~/r9700-stack && MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same bash serve/native-flashnext.sh
~/venv-r9700/bin/python bench/needle_long.py 30000,67000,150000,250000    # ladder; argv: TIERS BASE MODEL TOKENIZER MAXOUT WARM SEED
ASK="Summarise the text above in detail, at least 300 words." ~/venv-r9700/bin/python bench/needle_long.py 250000 \
    http://127.0.0.1:8080 Qwen3.8 ~/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ 400 1 11     # decode at depth
~/venv-r9700/bin/python bench/quality.py 10 8          # the standing gate: GSM8K + needle 3/3
```

Roll back to the 32k recipe: `STOP=1 bash serve/native-flashnext.sh` then
`MAXLEN=32768 NSEQ=32 NBT=8192 WEBUI=same bash serve/native-flashnext.sh`.
Logs: `~/logs/ctx262k-final-045115.log.out` (262k, recipe defaults), `~/logs/ctx262k-nseq32-041130.log.out`,
`~/logs/ctx262k-035005.log.out` (NSEQ=4 / PREFIX_CACHE=1), `~/logs/lpt-044037.log.out` (long-prefill A/B).
