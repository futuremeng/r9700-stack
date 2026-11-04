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

**These are cache-off numbers, and the recipe is no longer cache-off** (since 2026-11-04, aligned with upstream:
notes/prefix-cache-decision.md). Re-running this section today reproduces the cold columns and not the last one --
`the same context again` was the special case then and is what every repeated prefix pays now, so a fresh table
needs cold corpora if it wants to compare against the rows below.

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

### Then it went one request too far, and the box preempted for the first time

The load kept arriving and got longer. Read through the whole journal (385 rows, 12:53 → 14:34 — the Monitor only
keeps the last 100), in ten-minute buckets:

| bucket | rows | prompt p50 | first token p50 / p95 | that requests' own decode p50 | busy streams | aggregate generation |
|---|---:|---:|---:|---:|---:|---:|
| 13:30 | 72 | 49,384 | 5.6 / 9.8 s | 163.2 tok/s | 1.1 | 40 tok/s |
| 13:40 | 48 | 86,833 | 10.8 / 23.9 s | 127.0 | 1.5 | 51 |
| 13:50 | 42 | 108,515 | 15.0 / 29.8 s | 52.0 | 2.0 | 53 |
| 14:00 | 62 | 48,395 | 6.6 / 27.1 s | 50.3 | 2.4 | 52 |
| 14:10 | 63 | 66,923 | 9.5 / 35.6 s | 21.0 | 4.0 | 56 |
| 14:20 | 57 | 78,096 | 11.5 / 31.5 s | 18.3 | 4.9 | 64 |
| 14:30 | 25 | 28,985 | **26.1 / 48.9 s** | **12.2** | 4.6 | 39 |

The aggregate never leaves ~40–65 tok/s while TTFT rises ~5× and per-request decode falls ~13×. **That is the
prefill-bound regime stated out loud: past the pipe's ceiling, concurrency is not bought with throughput, it is
paid for in latency.**

And then `vllm:num_preemptions_total` — 0.0 all morning through 32 × 300 waves, 250k prefills and the reset storm —
went to **1** and then **2**:

| started | prompt | preemptions | first token | wall | its own decode |
|---|---:|---:|---:|---:|---:|
| 14:24:39 | **157,542** | 1 | 20.9 s | 63.0 s | 14.8 tok/s |
| 14:29:27 | **159,950** | 1 | **30.2 s** | 60.8 s | **3.4 tok/s** (129 tokens) |

Those are the two longest prompts in the journal, and they are what the pool math predicts: the pool holds
**540,202 tokens**, so it fits **3.4** 160k-token contexts, 5.4 at 100k, 9.0 at 60k — while `running` was sitting
at p50 6 with prompt p50 78k and the pool touching **100.0%** (2 samples over 99%, 13 over 95% in a 300 s
window). A preempt is not free and the row says so: vLLM v1 re-prefills, so the victim's first token went to
20.9 / 30.2 s and its own decode average fell to 14.8 / 3.4 tok/s.

So the honest capacity statement for long-context work is not `NSEQ=32` at all — it is **pool ÷ prompt**: keep
`running × prompt_length` under ~500k tokens. At 150k prompts that is **three** in flight; the box will run more
than that, and will do it by stealing decode from everyone and then preempting whoever asked for the most.

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

## Per-stream decode, read off the whole journal rather than a window (2026-11-02, 16:2x UTC)

The journal (`~/logs/requests-8080.jsonl`, one JSONL row per finished request — the thing added to survive hard
resets) had **1,400 completed rows** by this afternoon, covering **13.8 h** of wall-clock of which **301 minutes
had traffic in them**. That is 14× more real traffic than the 14:11–14:17 window above, and it is long enough to
answer the question the window could not: **what does one stream actually decode at?**

Definitions matter more than the number here. Own-request rate = `completion_tokens / timings.decode_s` — the
engine's own decode span, which excludes queueing and prefill (`predicted_per_second` in the row is the same
quantity). Concurrency is reconstructed from the rows' own `started_at … started_at + wallclock_s` at 1 Hz: for
each request, the mean and the **maximum** number of requests in flight across its decode window.

| in flight during its decode | share of active seconds | own decode, median |
|---|---:|---:|
| 1 (never shared) | 32.7% | **157.7 tok/s** (n=274) |
| 2 | 18.6% | 119.1 tok/s (n=232) |
| 3–4 | 31.8% | **22.8 tok/s** |
| 5–6 | 12.4% | 15.6 tok/s |
| 7–9 | 4.6% | 17.3 tok/s |
| **all 1,400 requests** | — | **60.8 tok/s** (mean 75.7, p10 8.7, p90 163.0) |

So the honest answer to "average single-stream decode" is two numbers, not one: **~157 tok/s when the stream has
the box to itself, ~61 tok/s for the request as a user actually experiences it**, and the experienced number is
*mostly* not a decode-speed problem — it is the price of sharing a prefill-bound engine.

**Single stream is nearly context-independent, which is worth knowing about this model.** Never-shared requests with
≥200 generated tokens (n=188), by prompt size: **161.9** tok/s under 20k (n=9), **158.2** at 20–80k (n=96),
**149.7** at 80–160k (n=74), **140.3** above 160k (n=9). A 262k context costs a single stream ~13% of its decode
speed — small next to what concurrency does to it. Decode on this stack streams MoE weights, so attention's KV
read grows while the dominant cost does not.

**Two "alone" definitions, 19 tok/s apart — the trap is worth recording.** Bucketing concurrency at the *midpoint*
of the decode window puts a request in "alone" whenever nothing else happened to be in flight at that instant, even
if it shared the engine for most of its decode. That definition said 137 tok/s (n=361). Requiring that the request
**never** shared the engine across its whole decode window (`cmax == 1`) gives 157.7 (n=274), and every sane
variant agrees (mean-concurrency <1.5: 156.1; <1.5 with ≥200 tokens: 155.7). Midpoint sampling is the shape of lie
to avoid when reconstructing concurrency from a journal.

**Aggregate, and why the denominator decides.** `Σgenerated / Σwallclock` = **16.6 tok/s**, but 63% of that
wall-clock is an idle box; over active minutes only it is **45.4 tok/s**. Mean in-flight during traffic is **2.75**
(p50 2, p90 6, max 9). `Σgenerated / Σdecode_s` — token-weighted own decode — is **24.8 tok/s**, well below the
60.8 median, because long decodes are the ones that ran while the box was busy: weighting by tokens moves the
answer toward the crowded case, and quoting an unweighted mean of per-request rates moves it toward the quiet one.
For "what will my request feel like", the median of per-request rates is the right shape: **61 tok/s**.

**The same journal reads prefill honestly too**, since rows carry `timings.prefill_s`: median **7,911 tok/s**
overall, and by prompt size 8,593 (<20k) / 8,641 (20–80k) / 7,867 (80–160k) / **7,238** (>160k) — the same
6.8–9.1k band the synthetic ladder measured, now with a size-dependent falloff that the bench's fixed prompt
lengths could not show.

**Where the cliff is, and what it is made of.** Nothing interesting happens between 1 and 2 streams (157.7 →
119.1: decode sharing itself, roughly proportional). Then **2 → 3 collapses it by ~5×** (119.1 → 22.8), and 5–6
streams sit at 15.6. With prompts of 50–170k and `NBT=8192`, a third long prompt means engine steps of ~1 s
carrying one 8192-token prefill chunk (measured directly earlier in this note: step p50 951 ms), and each stream is
credited its accepted tokens once per step. That is a scheduling/chunking cost, not a decode-kernel cost, which is
why the lever named in this note's other section — `--long-prefill-token-threshold`, or a smaller chunk — is the one
that addresses it, and why no kernel work on decode does.

**Cross-check against the bench, so nobody reads 157 as a regression.** The standing numbers for this box are 186.7
tok/s single stream at 32k with MTP-3, and 210–225 tok/s at the 210 W release gate with draft acceptance ~3.1
tokens a step. This launch runs **MTP=2** — acceptance 2.0–2.4 measured on live traffic — and these real requests
carry 50–170k of context. 156–158 tok/s is what acceptance-limited decode at MTP-2 with long context should look
like; it is not evidence of anything broken. If single-stream decode ever wants to come back, the knob is the
drafter depth, not the attention path.

## The fp8 launch under real traffic: same method, a new launch (2026-11-04, 01:50)

The launch that runs today is the fp8 e4m3 main-KV one (`R9K_QSA_KVF8=1 KVDTYPE=fp8_e4m3`,
`MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 MTP=2`, started 2026-11-03 15:29, journal boot
`1793719607-24497`, plugin 0.4.0 at 953a3901). Ten hours of real traffic, same method as the 2026-11-02 window
above: journal rows for the per-request view, `/metrics` counters for the aggregate view, and the two computed
independently.

**Totals, both sources.** Journal: 1,568 completed rows, 10.25 h span, 10,818 busy seconds (29% of the span),
prompts p50 52,083 / p90 149,572 / p95 163,513 / max 198,073, 106,214,313 prompt tokens, generated p50 322 /
p95 1,853 / total 874,128. Counters: `prompt_tokens_total` 106,686,448, `generation_tokens_total` 877,747,
`request_success_total` 1,526 stop + 39 length = 1,565. The two agree to under 0.5% (the journal flushes per
request, so the counter is ahead by whatever is in flight).

**Concurrency, per second, from the rows' own stamps:** 1 → 5,141 s (47.5%), 2 → 2,717 (25.1%), 3 → 1,852 (17.1%),
4 → 836 (7.7%), 5 → 235 (2.2%), 6 → 30, 7 → 3, 8 → 4 (0.4% together); p50 2, p90 4, max 8. The earlier (bf16)
window measured p50 2, p90 6, max 9 — the same box seeing lighter fan-out, which is why its bands were worse.

**Own decode by max concurrency over its decode window** (n = requests generating ≥40 tokens, 1,525 of them):
one 162.0 (585), two 114.5 (294), three-four **41.0** (423), five-six 23.9 (127), seven-nine 47.0 (23); overall
median **115.4**, mean 100.9, p10 17.1, p90 171.1. Alone with ≥200 tokens generated: 182.6 (<20k, n=10), 164.6
(20–80k, n=258), 150.0 (80–160k, n=99), 156.5 (>160k, n=30) — the context cost curve reproduces, and the >160k
bucket is thin, so read it as "flat to 160k and no evidence past it".

**Aggregates, all four denominators:** 23.7 tok/s over the span, 80.8 over busy seconds, 42.1 token-weighted
(`Σgenerated / Σdecode_s`), 115.4 as the median of rates. From the counters: 877,747 / 37,200 s = **23.6** and
877,747 / 20,909.3 s (`request_decode_time_seconds_sum`) = **42.0**. Prefill: 106,686,448 / 13,420.7 s
(`request_prefill_time_seconds_sum`) = **7,949 tok/s while prefilling**, against journal medians of 7,542–8,937 by
prompt bucket.

**Prefill and TTFT by prompt size** (journal, `prompt_tokens / prefill_s`, `first_token_s`): <20k n=179 median
8,158 (p10 1,102 — a request whose prompt was mostly cached-shaped or short-credit noise); 20–80k n=882 median
**8,937**; 80–160k n=376 median 7,970; >160k n=101 median **7,542**. TTFT over 1,563 rows: p50 **6.50 s**, p95
**24.52 s**, max **56.1 s**. Longest single prompt served: 198,073 tokens, TTFT 27.0 s at 7,330 tok/s of prefill,
then 1,242 tokens generated.

**Speculation over the whole launch** (counters): 376,433 drafts, 752,866 draft tokens (depth 2, MTP=2 is the
launcher default at TP4), 501,271 accepted → **1.33 accepted per draft**, mean acceptance length **2.33**, 66.6%
of drafted tokens accepted. Compare the paired benchmark arms in notes/kv-fp8-qsa.md: 1.549 (fp8) against
1.604/1.620 (bf16) accepted per draft at 32 in flight. Real traffic sits lower than that because the prompts are
50–200k and the answers are shorter.

**The three spans, as the Monitor computes them** (counter deltas, taken 2026-11-04 01:49, from the observe-mode
sidecar on :8099 watching this engine — see notes/webui.md for how the spans are built): 1 min 45.8 tok/s
generation, 4,214 prefill, 17.2 per stream, 4.7 req/min, 90.7 ms a step, 1.29 accepted/draft, KV 28.4%; 15 min
23.7, 7,713, 12.8, 2.7, 100.6 ms, 1.45, 18.5%; 1 h 26.4, 7,737, 17.9, 3.5, 65.4 ms, 1.48, 12.6%. The 1 min
prefill number is a metric artefact — `prompt_tokens_total` credits at request END, so a 60 s span misses long
prefills still in flight — which is the bias notes/webui.md documents and the Monitor's second line exists to
show; over spans longer than a request's wall time it converges on the counters' 7,949.

**Not a controlled fp8 comparison.** The bf16 window above (1,400 requests, prompts p50 74,381, p90 concurrency
6, bands 157.7 / 119.1 / 22.8 / 15.6 / 17.3, overall median 60.8) and this one differ in traffic mix as well as
in KV dtype, and the mix is the dominant term. Saying "real traffic got faster" from these two windows would be
wrong. The controlled comparison is the paired arms: pool 540,202 → 1,009,899 tokens, 32-stream aggregate 1,823 →
1,792, prefill flat to ±0.5%, acceptance 1.60 → 1.55 per draft.

## Why real traffic decodes slower at three streams than the bench does at thirty-two (2026-11-04, 02:2x)

The two numbers that provoke the question: `bench/probe32.sh` at 32 in flight gives **57.6 tok/s per stream**
(1,792 aggregate, MTP=2), while the real-traffic bands above give **41.0 tok/s at three-to-four** and **23.9 at
five-to-six**. Lower fan-out, worse per-stream rate. The explanation is not row count and not attention depth;
it is that under real traffic most of the engine's time sits inside steps that carry a prefill chunk, and a
decoding stream advances exactly one acceptance per step whatever that step costs.

**The step census (counters, this launch at 10.6 h).** `vllm:iteration_tokens_total` sums to exactly the tokens the
engine handled, and it records a decode step individually -- but a whole chunked prefill lands in ONE record
(measured on 0.32.0 by the Monitor's own work, and recorded in README: a 29,969-token prompt produced records 29,969,
4 and 1 against NBT 8,192). So the record count below is a few percent below the true step count, which is said where
it matters.
`prompt_tokens_total` 110,091,658 + `generation_tokens_total` 891,750 to the last token. So:

- **99.2% of the tokens this box processed were prompt tokens.** One generated token cost 123 prompt tokens.
- Prompt work in 8,192-token units: 110,091,658 / 8,192 = **13,439 chunk-equivalents**, i.e. **5.9% of records**
  are prefill work -- and that share is a floor, because one record can hold a multi-chunk prompt. True steps are
  nearer 241,000, which moves the share to ~5.6%; nothing else below depends on the distinction, because the unit
  prices are fixed by decode records (whose length two independent rates pin down) and by 8,192-token time units.
- Time, from the journal's interval unions (not the per-request sums, which overlap): 4.97 h with something in
  flight out of a 10.62 h span (47%), of which queue+prefill spans cover 4.05 h and decode spans 3.18 h --
  overlapping, because one request's prefill runs inside another's decode.

**Two unit prices, and they close the books.** Take a chunk-bearing step at ~1.0 s and a step without one at
20.6 ms:

| step kind | count | unit | total |
|---|---:|---:|---:|
| carries an 8,192-token prefill chunk | 13,439 | 1.0 s | 13,439 s |
| no prefill in it | 216,186 | 20.6 ms | 4,453 s |
(the second row is `records - chunks`; taking records as steps makes it 227,925 and moves the unit price to 19.5 ms,
so the pair should be read as 1.0 s and 20 +/- 1 ms, not as four significant figures)
| | 229,625 | mean 78 ms | **17,892 s = 4.97 h** -- the journal's union |

75% of the box's busy seconds are prefill steps, and one chunk step is worth **49 ordinary decode steps**
(1.0 s / 20.6 ms), or 27 of the fat 32-row ones. Those two prices were not fitted to that total alone -- they sit
right next to two independent measurements. This launch accepts 501,271 of 376,433 drafts, a mean acceptance length
of **2.33 tokens a step**, so the 162 tok/s a lone stream gets is **14.4 ms a step** -- and the benchmark suite
times a step (its "ITL" is a step time, not a per-token latency) at **13.0 ms with one stream**, 20.8 ms with eight
and **37.1 ms with thirty-two**. That median is not the mean, and the difference is the mechanism showing through:
the fp8 arm's 1,792 aggregate is 56.0 tok/s a stream, and at its own 2.55 accepted tokens a step that is a **45.5 ms
mean step** -- 23% above its 37.1 ms median. Solving f x 1.0 + (1-f) x 0.0371 = 0.0455 gives **f = 0.9% of the
wave's steps carrying a chunk**, which is exactly what 4.7 chunk steps against ~471 decode steps says. The bench
wave and the live box agree on the unit prices and differ in f by a factor of six in steps (0.9% against 5.9%) and
four in seconds (20% of the wave's seconds inside chunk steps against 75% of the box's).

**And the fan-out inside the chunk step is why 20% is survivable and 75% is not.** A chunk step is not a wasted
step -- it still advances every decoding stream once. In the wave that yields 32 x 2.33 = **75 tokens per second of
stall**; live, the engine runs 382,167 drafts across 229,625 steps, i.e. **1.66 streams decoding per step**, so the
same one-second step yields **3.9 tokens**. The bench is not lucky in having cheap prefills -- its prefills cost the
same ~1.0 s each -- it is lucky in having 32 streams to pay for them, and in paying them before the decode starts.

**What that does to a stream.** Its rate is (accepted tokens per step) divided by (its mean step time), and 2.33 is
the first term all the way through this launch. Alone: 162 tok/s = 2.33 / **14.4 ms**. At three-to-four in flight the
measured 41.0 tok/s implies **57 ms a step**, which is what **~4% of its steps being ~1 s steps** buys
(0.036 x 1.0 + 0.964 x 0.022 = 0.057). At five-to-six, 23.9 tok/s implies 98 ms, about 8%. Meanwhile going from one
stream to 32 *decode* rows moves the measured step only 13.0 -> 37.1 ms, a factor of 2.9. Prefill injection is worth
4-7x; concurrency inside decode is worth 2.9x. That is the whole gap.

**And why the bench does not see it.** A `probe32` wave is 32 requests x 1,200 prompt tokens = 38,400 prompt tokens
= **4.7 chunk steps in the whole wave**, paid once as a ramp before the streams settle; each stream then gets
~471 decode steps (1,200 generated at the bench's own 2.55 a step) to amortize them, and every one of those steps advances all 32 streams. Real
traffic injects 6-7 chunk steps per arriving request (p50 prompt 52,083) at ~2.5 arrivals a minute into requests
that own only ~138 decode steps of their own (median 322 generated tokens). Same engine, same kernels, different
injection rate.

**Ruled out, with the counters that say so.** Preemption: `vllm:num_preemptions_total` 0, and all 1,600 journal
rows for this boot carry `preemptions: 0`. KV pressure: pool mean 15-30% over the spans, peak 45%, and
`num_requests_waiting_by_reason` shows capacity 0 / deferred 0 -- the queue is never a memory wall here. fp8: the
paired arms measured 1,823 -> 1,792 aggregate and 1.60 -> 1.55 accepted per draft, two orders of magnitude short of
4-7x. Acceptance: this launch accepts 1.33 per draft, so 2.33 tokens a step; the bench arms sit at 1.55-1.62, so
acceptance explains a few percent, not a factor of four. Context depth: 162 tok/s alone
at 20-80k against 186.7 at 32k uncapped, ~13%.

**A measurement trap on the way, kept because it is how this could go wrong.** The obvious first move is to fit
those two unit prices from 5-second buckets of `iteration_tokens_total` against `prompt_tokens_total`. It does not
work: prompt tokens credit when a request ENDS, so a bucket's prompt total says where requests finished, not
where the prefills ran -- the fit came out with f = 0 buckets costing 165 ms a step, which is nonsense. The same
quantity measured over a 150 s window says 36% of steps carried a chunk; over the launch it says 5.9%. Both are
true at their own timescale; only the launch-level counters and the interval unions can be reconciled, which is
why the decomposition above is built from those and then checked against the two rates it was not fitted to.

**What can be done about it, and what it costs.** `--long-prefill-token-threshold` (+ `_adaptive`) is a fairness
knob, not a throughput one: six 30k requests behind a 250k one go 40.3-55.8 s -> 7.8-27.7 s with the wall clock
unchanged (55.9 vs 56.8 s), while three 250k prefills in parallel push the pool to 99.1% and cost 22% -- so it
stays off and the cliff stays where the mechanism puts it. Lower `NBT` would shorten each stall and interleave
more finely, at the cost of prompt throughput (9,702 tok/s at NBT 8,192 is why it is 8,192). `MTP=3` buys
single-stream latency (197.6 against 179.8) and costs 18% of aggregate at 32 in flight. Turning prefix caching
back on is the one lever that removes prompt work rather than redistributing it, and it is off on purpose for the
262k recipe (notes/kv-fp8-qsa.md and the 262k tables above) — measured on 2026-11-04 with six copies of one
14,530-token prompt: wall 32.3 s → 3.6 s, median TTFT 25.3 s → 0.66 s, 133,120 of 346,754 prompt tokens served
from cache. That is the best case, not the expected one; the win scales with how much of the prompt is genuinely
shared (notes/launch-methodology.md §8, including the config revert and its check).

**"Shouldn't the seconds spent chewing prompts be out of the decode denominator?" -- asked of the right counters,
the answer is no, and the card that asked it was wrong (2026-11-04, same launch).** vLLM publishes two ways to
count decode seconds, and they cross-check to 0.08%: `inter_token_latency_seconds` carries 430,954 observations
summing to **24,236.14 s**, and `request_decode_time_seconds` carries 1,812 observations summing to **24,216.27 s**.
That agreement is not corroboration, it is tautology: it is ONE sum sampled twice -- the step family
per stream-step (`v1/metrics/stats.py:551`, `itl = engine_core_timestamp - req_stats.last_token_ts`, recorded in the
per-request loop of `v1/metrics/loggers.py:275-280`), the request family once at the end
(`stats.py:601`, `last_token_ts - first_token_ts`). So both are **stream-seconds**: a step with two streams
decoding accrues twice. Two consequences the Monitor had backwards:

- Dividing the engine's summed generated tokens by stream-seconds gives a **per-stream** rate, not a summed one.
  The live 15 min card read **17.3 tok/s labelled "summed, decode time"** in a span whose tokens summed to **45.1
  tok/s over wall seconds** -- a summed figure below the summed one, which is what gave it away.
- The step interval **contains** the chunked prefill computed in that step (a chunk step ~1.0 s against ~20 ms for
  a bare one), so an ITL-shaped denominator does not exclude prompt chewing at all. What it excludes is queue,
  prefill before the first token, and steps that produced no token.

| basis for a decode rate | denominator | this launch (10.6 h) | the 661 s window |
|---|---|---:|---:|
| **summed over streams, over wall seconds** (what a day of traffic produces) | span | 23.4 tok/s | 20.1 |
| summed over seconds with something in flight | journal union | 80.8 | 21.3 |
| **per stream actually decoding** (a per-stream rate; prompt chunks are inside it) | stream-seconds | 36.8 | 68.8 |
| one request's own decode window | per request | 115 (median) | 158 (median, p10 24.4 / p90 176.8) |

and the identity that ties the first to the third, which is what the sidecar publishes and the tests assert:

    summed tok/s = tok/s per stream-second  x  streams decoding (stream-seconds / wall seconds)
    launch:      23.4  = 36.8 x 0.64        (0.64 streams in decode phase at an average instant of a day)
    15 min:      42.5  = 17.2 x 2.47        (2.47 while traffic is on)

The old card's `decode_time_pct` was that last factor wearing a percent sign. It read 29% for the 661 s window and
**260%** for a 15-minute one, and the window table's sentence "the ratio of the two rates IS the decode-time
share" was right only because that window rarely had two streams decoding at once (at 1 Hz: 0 streams for 71% of
seconds, 1 for 28%, 2 for 1%). A count of stream-seconds per wall second has no 100% ceiling, so the card now
prints `x2.6 decoding`, never `260%`. The same naming discipline runs through the payload: `..._tok_s` is per wall
second, `..._stream_tok_s` is per stream-second, and `decode_streams` / `prefill_streams` are counts that may
exceed 1. `streams_per_record` -- ITL observations over `iteration_tokens_total` records, **2.85** live -- is the
same fan-out per *step* rather than per second, and it reconciles with the census from the other side: 2.85 streams
x ~2.5 accepted tokens = 7.05 tokens a record.

`prefill_streams` (1.04 live) is why the prefill headline did not move when renamed: long prefills take the whole
8,192-token step, so they serialize. Its denominator, `request_prefill_time_seconds_sum`, is the request's own
scheduled -> first-token span, which therefore contains decode interference from other requests -- that is the
difference between this launch's **~7,200 tok/s per prefill-second** and the census's **8,079 per
whole-prefill-record**, and why a long-context prompt reading 7,6xx in the Monitor is not contradicting the bench.

Three more calculation bugs the same pass found, each now pinned by a test:

- **A request's own decode rate was 14% fast, and 2x fast for a two-token request.** The row divided
  `num_generation_tokens` by `decode_time` while calling the result "the engine's own mean". The engine's mean is
  `decode_time / (gen - 1)` (`stats.py:606-610`), because `decode_time` spans first token -> last token and the
  first token came out of the prefill step. Live: mean 28.57 ms a token -> the engine says 35.0 tok/s, the row
  showed 37.5.
- **The PCIe rate divided a partial watch by the whole window.** `aer_poll_s = 5` needs 60 clean polls before it
  reports a rate at all, and then divided by `window_s = 300` whether it had watched 300 s or 60 -- so the first
  five minutes of every launch read a trickle at one fifth its true per-minute rate, which is exactly when the
  number matters (ramp-up). The denominator is now the span actually watched, floored at the 60 s the estimator
  needs, with each chain's `watched_s` alongside.
- **Every journalled relay row lost its timings.** `Journal.append` serialises `ROW_KEYS` when it is called, and
  the proxy path appended *before* computing `wallclock_s` and `timings`: the list looked right in memory while
  the file held nulls, so after a hard reset every restored row had no wall, no decode rate, and no span for
  `shared_decode` to count. Both paths append last now.

Plus two exactness wins taken while the counters were open: the skipped-prefix numerator prefers
`prompt_tokens_cached_total` (vLLM's exact count of tokens it did not read, KV-connector transfers included) to
`prefix_cache_hits_total` (local cache only, counted at admission), and `num_requests_running` /
`num_requests_waiting` are **summed across series** -- with TP4 each API-server instance reports its own pair, and
the old code took the max, reporting one instance's count as the engine's. `kv_used_pct_mean` also stopped
summing the gauge across the two KV pools and calling the total a mean: a pair of ranks 40% full used to read 80%.

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
cd ~/r9700-stack && MAXLEN=262144 NSEQ=32 NBT=8192 WEBUI=same bash serve/native-flashnext.sh
#   (PREFIX_CACHE is ON by default from 2026-11-04; add PREFIX_CACHE=0 to reproduce the recipe the tables above
#    were measured on. HOST comes from ~/.config/r9700-stack/native.env, notes/launch-methodology.md section 10.)
~/venv-r9700/bin/python bench/needle_long.py 30000,67000,150000,250000    # ladder; argv: TIERS BASE MODEL TOKENIZER MAXOUT WARM SEED
ASK="Summarise the text above in detail, at least 300 words." ~/venv-r9700/bin/python bench/needle_long.py 250000 \
    http://127.0.0.1:8080 Qwen3.8 ~/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ 400 1 11     # decode at depth
~/venv-r9700/bin/python bench/quality.py 10 8          # the standing gate: GSM8K + needle 3/3
```

Roll back to the 32k recipe: `STOP=1 bash serve/native-flashnext.sh` then
`MAXLEN=32768 NSEQ=32 NBT=8192 WEBUI=same bash serve/native-flashnext.sh`.
Logs: `~/logs/ctx262k-final-045115.log.out` (262k, recipe defaults), `~/logs/ctx262k-nseq32-041130.log.out`,
`~/logs/ctx262k-035005.log.out` (NSEQ=4 / PREFIX_CACHE=1), `~/logs/lpt-044037.log.out` (long-prefill A/B).
