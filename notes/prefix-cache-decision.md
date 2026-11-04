# Would the prefix cache pay for itself here? Decided against the box's own conversations.

Asked on 2026-11-04: 将本地的对话记录作为本地的测试集进行对照测试，决定是否常开. Every prefix-cache number this
box had was measured on a corpus built for the question -- six copies of one 14,530-token filler (notes/webui.md,
launch-methodology §8). That measures what a cache *can* do and says nothing about what it *does* to the traffic
this machine carries, because for this workload the answer is not about the cache at all. It is about arrival
order and pool size:

- an agent harness resends its whole transcript every step, so consecutive requests from one session are
  prefix-nested and the nominal hit is enormous;
- a hit only exists if those blocks are **still in the pool** when the next request lands, and the pool is 1,175
  blocks of 880 tokens (1,009,899 tokens) while one 226,100-token session wants 257 of them;
- requests from several sessions interleave, so the question is what survives that.

So `tools/prefix-replay.py` replays the box's own conversation records and simulates the engine's own rules.

## The two corpora, and what each is good for

| | local conversation records (`~/.pi/agent/sessions`) | this box's request journal (`logs/requests-8080.jsonl`) |
|---|---|---|
| calls / rows | 6,560 model calls (10 records unreadable: zero-filled lines where a write died mid-flight) | 9,010 rows |
| window | 2026-10-31 → 11-04 (4 days) | 2026-11-02 12:53 → 11-04 09:44 (~44 h) |
| prompt tokens | 581.9M | 396.8M |
| prompt size p50 / p90 / p99 / max | 84,630 / 171,028 / 226,100 / 263,298 | 15,033 / 137,063 / 191,352 / 262,059 |
| peak concurrent | 4 calls, 989 blocks held at once | 32 requests, **327 blocks** held at once |
| what it supplies | **content** -- the actual bytes, so block identity and prefix structure are real | **capacity profile** -- this box's real concurrency: the pool's low water is 848 blocks, never once did in-use demand exceed 1,175 |

The sessions are not this box's traffic: they go to `192.168.1.130:4000` (a proxy), and joining them against the
journal on prompt-token counts and arrival times matches 2 of 6,534. They are the same *kind* of traffic at the
same volume, and they are the only local records that contain prompt text -- which is what a prefix test needs.
Reconstruction is measured, not assumed: the client stored its own per-call prompt count (`usage.input`, the new
tokens, plus `usage.cacheRead`, the tokens it skipped), and the rebuilt prompt is a median **0.72x** of that
(p10 0.31, p90 0.98) because the record does not carry everything the harness sends (tool schemas, attachments).
The magnitudes are therefore conservative and the *structure* -- what shares a prefix with what -- is exact, and
the simulation only uses structure for hits.

## The comparison (block 880, pool 1,175, this box's measured read rate 7,428 tok/s)

| arrivals | run | prompt tokens | hit | prefill s | admission waits | evictions | peak blocks | blocks free at peak |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| as recorded | cache OFF | 581,891,351 | 0.0% | 78,314 | 0 s | 0 | 989 | 186 |
| as recorded | cache ON | 581,891,351 | **98.1%** | **1,456** | 0 s | 7,832 | 1,175 | 0 |
| ×2 | OFF / ON | same | 0 / 98.1% | 78,198 / 1,454 | **2 s / 0 s** | 0 / 7,820 | 1,138 / 1,175 | 37 / 0 |
| ×4 | OFF / ON | same | 0 / 98.1% | 78,198 / 1,454 | **1,696 s / 0 s** | 0 / 7,820 | 1,173 / 1,175 | 2 / 0 |
| ×8 | OFF / ON | same | 0 / 98.1% | 78,198 / 1,454 | **60,706 s / 0 s** | 0 / 7,821 | 1,175 / 1,175 | 0 / 0 |

**ON buys back 571,078,640 prompt tokens = 76,882 s of prefill (98.1% of the OFF run's prefill, ~21 hours over four
days of this traffic), and it never queues.** The cache does saturate the pool (peak 1,175 of 1,175, 7,832 retained
blocks evicted), but every eviction was of a block nobody needed: nothing had to wait for room. Meanwhile OFF,
which has 186 blocks free at the peak, starts queueing as soon as the load is squeezed 2x, and 60,706 s of queueing
at 8x -- because a cache-off request must allocate its whole context every time, while an ON request re-uses the
98% of it that is already resident. For prefix-nested traffic the cache is a **capacity win**, not a capacity cost.

An independent check on the 98.1%: these very conversations went to an endpoint that runs its cache ON, and its own
usage records say `cacheRead` = 775,363,595 of 782,710,453 prompt tokens = **99.1%** measured. Simulated 98.1%,
measured 99.1%, different instruments, same calls.

## The cost side, measured on this box rather than simulated

- **The pool gets 2.0-3.2% smaller.** The 2026-11-04 test boot with `PREFIX_CACHE=1` published
  `kv_cache_size_tokens = 989,534`, and the production launch that made it permanent (10:26 the same day) published
  **977,726** -- 1,242 blocks of 832 against the cache-off launch's 1,175 blocks of 880, i.e. -3.2% -- against
  production's `1,009,899`. On a hybrid (GDN/QSA) model the cached unit is the hybrid block plus its recurrent
  state, and that state has to live somewhere; the boot log shows the two page sizes being padded into each other
  ("pad the mamba page size by 1.71% to ensure that mamba page size and attention page size are exactly equal").
- Retained blocks make the KV gauge read fuller, so the Monitor's `pool used` line shifts (notes/webui.md): a
  fuller-looking pool is not a tighter pool while nothing is preempting, but the number moves and a reader has to
  know why.
- Evictions are not free in the general case -- they are free **here** because the box's own journal shows in-use
  demand peaking at 327 blocks of 1,175. On a box already near 90% pool occupancy the same replay would start
  evicting prefixes that were about to be needed.

## Verdict: turn it on, and watch two counters for an hour afterwards

For the traffic this box carries -- agentic, prefix-nested, 15k-262k prompts, pool occupancy peaking at 28% -- the
answer is not close: the downside is bounded at 2-3% of pool (which is what caching costs when nothing hits), and the
upside is 98% of prefill work. The two things that would change the verdict are both visible from the Monitor: a
traffic mix that stops repeating prefixes (then ON ≈ OFF minus 2% of pool), and a pool that is genuinely tight
(near preemption), where retained prefixes start competing with live work.

Landing it is one default in `serve/native-flashnext.sh` (`: ${PREFIX_CACHE:=1}`, which drops
`--no-enable-prefix-caching` from the line) plus the README/notes lines that say `PREFIX_CACHE=0` is the recipe --
those lines are load-bearing for the 262k measurement series (notes/kv-fp8-qsa.md), so they have to be re-stated
with the new numbers rather than deleted. It landed the same day: see "How it landed" below. After the flip, the check is `prompt_tokens_cached_total` and
`prefix_cache_hits_total` moving off zero on the live box, `num_preemptions_total` staying at 0, and the first-token
line dropping: predicted TTFT falls by (prompt tokens x 0.98) / 7,428 per request, which on a p90 137k prompt is
about 18 s.

Not yet done, deliberately: this is an offline replay. It is not a live A/B on this machine, and replaying 582M
prompt tokens into a box serving real clients is not how to find out -- the counters above are. Also worth saying:
the 07:53 death (launch-methodology §9, `hipErrorIllegalAddress` in a decode step at 111k computed tokens) happened
with the cache OFF, and the box ran 30 minutes with it ON earlier the same morning without one, so nothing here
implicates the cache -- but the hybrid-state path is a code path this box exercises far more with caching on, so the
crash log gets watched for a day after the flip.

## The objection this repo already contains, and what it actually measured

This is not an unmeasured knob. `5056943` (2026-09-25, "serve: prefix caching off by default for Flash-Next
(PREFIX_CACHE knob); short-prefill root cause") turned it off for this model, and `serve/serve.sh` still documents
the mechanism next to the ON default it kept:

> On hybrid (GDN + attention) models prefix caching puts the mamba cache in "align" mode, and the scheduler then
> aligns every prefill chunk end to the mamba block size, so a prompt that is not a multiple of it costs an extra
> forward pass (~285 ms of CPU-bound Python each; a 1.5k prompt ran as two). Measured 2026-09-25 (Flash-Next TP=4):
> 1325-token TTFT 563 -> 313 ms, 3086 tokens 564 -> 512 ms.

`serve/flashnext.sh` inherited that as `PREFIX_CACHE=0` ("avoids the mamba-aligned prefill chunking, 2k prefill
2.8k -> ~4.7k tok/s"), and `serve/native-flashnext.sh` inherited it from `flashnext.sh` -- which is why this box was
running with the cache off while `serve.sh` was not.

The measurement is real and its direction is against caching, but it is a **short-prompt** measurement, and the
quantity at stake scales the other way. The extra pass is bounded at roughly one per prompt (the alignment falls on
the chunk end, not on the token), so it is ~285 ms against a 1.5k-token prompt -- an 80% TTFT penalty at that tier,
563 -> 313 ms -- and ~285 ms against the 17-20 s of prefill a 130k-token prompt costs, under 2%. This box's ledger
says p99 prompt 245,792 tokens and 1,195 requests at or above 32,000, and its own 2026-11-04 A/B at 14,530 tokens a
prompt measured the whole thing cold: wall 13.7 s with the cache on against 32.3 s with it off, median TTFT 10.0 s
against 25.3 s. Already swamped at 14.5k.

The replay does **not** model the extra pass -- it prices tokens not read, nothing else. So the short tier has to be
re-measured on this stack instead of inherited, and it is one command:

```
ITEMS=0,66,152 bench/ttft_breakdown.py http://127.0.0.1:8080 cacheon     # ~45, ~1,365, ~3,065 tokens
```

If that reproduces a doubling of TTFT at the 1.3k tier on the current build, and this box is asked to serve that
tier, `PREFIX_CACHE=0` is the right answer again and `tools/recipe-check.py` will say in one command that the box is
not on the tree's recipe. Two years of inheriting a default is exactly how that argument gets lost, which is why the
line in `serve/native-flashnext.sh` now cites 5056943 and this command instead of asserting ON.

**That debt is paid, and it dissolved the objection (same day, after the flip).** Run on the production cache-on
launch, 15 requests per tier, each with a different prompt so nothing hits:

| tier | client median | TTFT (engine) | prefill | iterations per request |
|---|---:|---:|---:|---:|
| 45 tokens | 44 ms | 37 ms | 31.1 ms | **1.00** |
| 1,412 tokens | 202 ms | **188 ms** | 178.0 ms | **1.00** |
| 3,323 tokens | 387 ms | **371 ms** | 357.6 ms | **1.00** |

Against 2026-09-25's cache-**on** numbers (563 ms at 1,325 tokens, 564 ms at 3,086) the current stack is 3x faster
with the cache on, and better than that day's cache-**off** numbers (313 / 512 ms) too. Decisively: **1.00 iterations
per request** at every tier. The extra forward pass that the align-mode alignment used to cost is not happening on
this stack -- which is what the 2026-09-25 root-cause note was measuring, and it is gone. (Not a controlled A/B: the
old numbers came from an older build. But `iters/req = 1.00` is a mechanism-level answer that no baseline can argue
with, and it makes a controlled cache-off re-run a nicety rather than a decision.)

Two smaller objections, both answered by numbers rather than by argument:

- **notes/release-v0.2.4.md**: the plugin's one-page GDN state is Stage 1 for `mamba_cache_mode` "none"; with
  caching on, "vLLM's pages are used unchanged", so the +18% KV that work bought (Flash-Next, two cards) is not in
  play. That +18% belongs to a different configuration (two cards, mode none) and must not be subtracted from the
  -2.0% above: -2.0% is what this box's engine published with everything on, measured, and it is the number to use.
- **MTP and align mode.** `enable_mamba_shared_prefix_checkpoint` exists in this build precisely for the case of an
  EAGLE/MTP sibling resuming at a shared-prefix junction, and it needs `prefix_match_unit` finer than the mamba
  block. This box runs MTP=2, and after the flip the number to watch is accepted tokens per draft, which sits at
  1.44-1.46 today: if it drops, that knob plus `prefix_match_unit` (832 / 32 = 26, so 32 is legal) is where to look.

## How it landed (2026-11-04)

- `serve/native-flashnext.sh`: `: ${PREFIX_CACHE:=1}`, replacing `${PREFIX_CACHE:-0}` at both sites (the flag and
  the recipe file), with the 5056943 measurement and the re-measure command quoted in the comment so the next
  person inherits the argument, not just the default. `PREFIX_CACHE=0` still turns it off.
- `HOST`: the repo's default is `127.0.0.1`, which is also upstream's. This box binds `0.0.0.0` from
  `~/.config/r9700-stack/native.env`, outside the repo (notes/launch-methodology.md section 10). The recipe file now
  records `host`, and the About page shows it; `launch_hint()` deliberately does not put it in the restore command,
  because the launcher picks it up from the local env.
- Found on the way: **`DRY=1` wrote the recipe file.** A `DRY=1 WEBUI=same` run meant to read the launch line back
  overwrote the recipe of the serve that was actually running, so a cache-off engine's About page described a
  cache-on launch, and `recipe-check.py`'s meta-vs-engine agreement was a lie until a restart. `DRY` now writes and
  starts nothing (it also no longer spawns the sidecar); `METAONLY=1` is the deliberate version, for refreshing the
  recipe file of a launch that is already up.
- Proving the flip, in order: `tools/recipe-check.py` exits 0; `/api/status` config says `enable_prefix_caching
  True` with a 832-token block; the pool reads ~977,726 tokens; after real traffic
  `prompt_tokens_by_source_total{source="local_cache_hit"}` moves off zero and `prefix_cache_hits_total` with it;
  `num_preemptions_total` stays 0; accepted-per-draft stays ~1.45; the crash log stays quiet for a day
  (notes/incident-2026-11-03 section 9's illegal-memory-access was on a cache-off launch, but the align path is new
  exercise for this box).

## It is on, and the counters say so (2026-11-04 10:26 launch)

Boot: 210 s to `/health` (the compile-cache key from the 07:28 test boot was reused, no recompile),
`Mamba cache mode is set to 'align' for Qwen4ExpForConditionalGeneration by default when prefix caching is enabled`,
`kv lcm block sizes 832`, GPU KV cache 977,726 tokens, `recipe-check.py` exits 0.

Three identical 3,658-token prompts, back to back:

```
call 1: wall 0.51 s  prompt 3658
call 2: wall 0.23 s  prompt 3658      prefix_cache_hits_total        4,992
call 3: wall 0.23 s  prompt 3658      prompt_tokens_cached_total     4,992   = 3 x 832
                                       prompt_tokens_by_source{local_cache_hit} 4,992
                                       num_preemptions_total            0
```

The first ten minutes of **real** traffic after the flip, which is the number that actually matters: 1,150,598
prompt tokens counted, of which **955,136 (83.0%) were `local_cache_hit`** and 195,462 computed, with
`num_preemptions_total` still 0. The per-request rows changed shape -- a 98,541-token prompt with
`96,541 -> ... 96,512 cached` and a first token in 0.24-0.39 s, against a 15-minute TTFT mean of 22.9 s on the same
traffic an hour earlier. The replay said 98.2%; the box is reading 83% in its first ten minutes and the difference is
cold prefixes at the start of conversations, exactly as the model predicted.

Two things that fell out of that probe and matter more than the probe itself:

- **A hit costs 832 tokens of prefix.** Repeating a 538-token prompt three times produced 0 hits: nothing is
  cacheable until a whole physical block exists. That is the sharpest form of the 2026-09-25 objection -- a short
  prompt pays the align-mode alignment and cannot earn any reuse back -- and it puts the break-even for this box's
  traffic at well under one block, since its p50 prompt is in the tens of thousands.
- **The client cannot see it on this build.** `usage.prompt_tokens_details.cached_tokens` stayed `null` on those
  calls even while the engine counted 4,992 cached tokens, so a proxy's `cacheRead` figure is not a way to verify
  this box. The engine's counters are, and they are what the Monitor subtracts.

## One open number: MTP acceptance dropped, and it is not yet explained

**Resolved the same day, below ("It was traffic: the fixed-prompt gate answers it").** The numbers in this section
were written while the question was open and are kept as the record of what was measured and what was inferred --
they are not the current answer.

40 minutes after the flip, on live traffic (engine counters, not the Monitor's window maths):

| | drafts | accepted / draft | pos 0 | pos 1 |
|---|---:|---:|---:|---:|
| the cache-off boot it replaced | 88,106 | **1.438** | 80.9% | 62.9% |
| the cache-on launch, 40 min | 9,592 | **1.281** | 74.5% | 53.6% |

-11% accepted tokens per draft. Two explanations, and the traffic did change as drastically as the acceptance did
(TTFT fell from a 22.9 s mean to 0.57 s, decode fan-out went 0.26 -> 0.44, generation 12.4 -> 46.6 tok/s of wall),
so this is **not yet evidence of a mechanism**:

- **Traffic shape.** Different requests, different token mixes, different batch shapes. Acceptance is a property of
  what is being predicted, and the box is now serving a much wider mix because the queue moved.
- **The align-mode/MTP interaction the fork has a knob for.** `enable_mamba_shared_prefix_checkpoint` exists for
  exactly this: an MTP sibling resuming at a shared-prefix junction needs a mamba checkpoint *there*, and it only
  does anything if `prefix_match_unit` is finer than the mamba block (unset here, so hits land on whole 832-token
  blocks). If this is the cause, the fix is `prefix_match_unit=32` (832 / 32 = 26) plus that checkpoint on.

What separates them is the standing gate, whose prompts are fixed: `bench/quality.py 10 8` (GSM8K 10 + needles 8)
recorded 1.44 accepted per draft with the cache off; run it on this launch and compare. Until that is done, the
honest summary is: acceptance is 11% lower for reasons not yet pinned, and every other measured number moved the
right way by a lot (generation 3.7x, TTFT 40x, 80.4% of prompt tokens not computed, preemptions still 0).

### It was traffic: the fixed-prompt gate answers it (2026-11-04, 14:54)

Run the gate twice back to back on the cache-on launch and diff the engine's spec-decode counters around each run
(`vllm:spec_decode_num_drafts_total`, `..._accepted_tokens_total`, `..._accepted_tokens_per_pos_total`):

| the same gate, same prompts | drafts | accepted / draft | pos 0 | pos 1 | prompt tokens arriving as a cache hit |
|---|---:|---:|---:|---:|---:|
| cache-off launch, this gate (older record) | -- | **1.44** | -- | -- | 0% (cache off) |
| cache-on, run 1 | 1,836 | **1.644** | 89.8% | 74.6% | 70.6% |
| cache-on, run 2 (warmer prefixes) | 2,327 | **1.612** | 88.4% | 72.8% | 81.0% |
| cache-on, run 3 (same 13 calls, prefixes hot) | 1,658 | **1.702** | 91.4% | 78.8% | 90.8% |
| cache-on, the same gate at **100** questions | 13,240 | **1.713** | -- | -- | 87.0% |
| production config (`M<=8`), 100 questions, **genuinely cold** | 12,421 | **1.740** | 92.9% | 81.1% | 1.4% |
| production config, 100 questions, warm | 12,411 | **1.726** | 92.4% | 80.2% | 78.5% |
| production config, 10 questions, hot | 1,325 / 1,408 | **1.715 / 1.716** | 92.3% | 79.2% | 91.0% |

Three things follow, and none of them needs the live windows that confused the question:

- **Acceptance is not lower with the cache on; on identical prompts it is higher than the cache-off launch's own
  gate number** (1.64 against 1.44). Whatever the live 1.28-1.36 was, it was the requests, not the configuration.
- **Warm prefixes do not cost acceptance.** Going 70.6% -> 81.0% cached on the *same* prompts moved accepted per
  draft by **-2.0%**, and the windows were clean: 13 of the 14 journal rows in run 1 were the gate's own requests.
  So `mamba_cache_mode=align` is not silently throwing MTP siblings away. `prefix_match_unit` plus
  `enable_mamba_shared_prefix_checkpoint` stay unexplored as *latency/capacity* levers; the acceptance argument for
  them is gone.
- **Quality is unchanged by the flip and by the router experiment on this gate**: GSM8K 10/10 and needles 3/3
  (3,828 / 15,196 / 30,354 tokens), three runs. Ten questions is an anecdote, so the gate was re-run at **100**:
  **97/100 (97.0%)**, needles 3/3, 107 journal rows in the window of which 103 were the gate's own, preemptions 0.
  The fp8 A/B's 800-question arms measured 96.0 and 95.6 -- a different subset size, so that reads as *no
  regression*, not as +1%. That same run is the strongest acceptance number here: **1.713 accepted per draft over
  13,240 drafts**, 87.0% of prompt tokens arriving as cache hits.
- **The baseline is the weak side of the comparison, and it is named as such.** The 1.44 on the cache-off row came
  from the 10-question gate; 1.713 is the 100-question one, so those two are not like for like. What *is* like for
  like is runs 1-3: the same 13 calls, prefixes going 70.6% -> 81.0% -> 90.8% cached, acceptance 1.644 -> 1.612 ->
  1.702. Acceptance does not fall as the cache gets warmer; and if the cache-off launch is ever re-measured, run it
  at 100 so the headline numbers match.

Note what this does **not** settle -- at the time it was written; the next subsection closes it. The launch under
test had `R9K_ROUTER_MAX_M=4` (the crash discriminator,
notes/launch-methodology.md section 12), so the MoE gate ran on hipBLASLt for the 5-8 row shapes. That changes
which kernel computes the router -- different summation order, same bf16 -- and cannot be excluded as an influence
on acceptance without another pair of runs. It is not a likely one at 1.64-1.71 vs 1.44, but the claim on record is
"cache on, acceptance fine", not "router cap changes nothing".

### The router-cap confound, excluded (same day, after reverting)

The caveat above was closed by reverting to the production recipe (`M <= 8`, the launch of 23:47 proved it: boot log
`installed on 48 gates (M <= 8, split 2)`, no `R9K_ROUTER_MAX_M` in `/proc/<pid>/environ`, and `recipe-check` back to
exit 0 with the environment compared) and running the same gate on it:

| same gate, two configs | accepted / draft | prompt tokens arriving as a cache hit |
|---|---:|---:|
| `R9K_ROUTER_MAX_M=4` (hipBLASLt at 5-8 rows), 10 questions hot | 1.702 | 90.8% |
| production `M <= 8` (our kernel at 5-8 rows), 10 questions hot | **1.715 / 1.716** | 91.0% |
| `R9K_ROUTER_MAX_M=4`, 100 questions | 1.713 | 87.0% |
| production `M <= 8`, 100 questions warm | **1.726** | 78.5% |

The kernel choice moves accepted per draft by **+0.8% and +0.8%** at matched cache warmth -- inside the spread of
repeating the same run twice (1.715 against 1.716). The confound is gone: the acceptance conclusion stands on the
production configuration.

That pair also gave the missing end of the warmth series, because a restarted engine starts with an empty prefix
cache: **a genuinely cold 100-question run accepts 1.740 at 1.4% cached**, against 1.726 warm (78.5%) and 1.715-1.716
hot (91.0%). Across a 90-point swing in cache hits, acceptance moves 1.4%. That is the cleanest form of the answer to
the original question: **prefix caching and acceptance are independent on this box, in both router configurations.**

One thing the cold/warm pair exposed, kept because it is the kind of thing that gets forgotten: on the **same 100
questions at temperature 0**, the cold run scored **97.0%** and the warm run **99.0%** -- two answers flipped. Cached
prefix reuse is therefore not bit-identical to recomputing the prompt (different accumulation order in the resumed
blocks), and at n=100 two flips is noise-level -- the same magnitude as the fp8 A/B's 96.0 against 95.6 -- but it is
why "cold" and "warm" runs of a quality gate are not the same measurement, and a quality claim that matters should
say which one it is.

Re-run recipe (no new tool; the counters are the engine's own):

```
# snapshot -> run -> snapshot, and diff. A window is trustworthy when its journal rows are almost all the gate's:
# the 100-question run put 107 rows in the log and 103 of them were the gate's own calls.
nice -n 19 python3 bench/quality.py 100 8           # 10 is the fast anecdote; 100 is the number
curl -s http://127.0.0.1:8080/metrics | grep -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)' \
  | grep -v created
# accepted per draft = delta(accepted_tokens) / delta(drafts); per-position ratios come from ..._per_pos_total
```

## Re-running this

```
nice -n 19 /home/jiulu/venv-r9700/bin/python3 tools/prefix-replay.py              # offline, GPU-free
nice -n 19 /home/jiulu/venv-r9700/bin/python3 tools/prefix-replay.py --speed 4    # the same days at 4x concurrency
nice -n 19 /home/jiulu/venv-r9700/bin/python3 tools/prefix-replay.py --pool-blocks 600    # a smaller pool
nice -n 19 /home/jiulu/venv-r9700/bin/python3 tests/test_prefix_replay.py         # the small hand-checkable cases
```

The pool and the block size default to whatever the live serve publishes (`/api/status` -> `num_gpu_blocks`,
`effective_attention_block_size`), so the simulation cannot drift from the machine it is judging.

The geometry was re-run at the shape a cache-on box actually gets, since enabling changes it (832-token blocks,
1,189 of them): hit 98.2%, prefill 1,442 s against 79,293 s, still zero admission waits. The verdict does not
depend on the block size, and the prompt total moves 1.2% because the reconstruction trims each turn to the last
whole block, so a finer block keeps slightly more history. What the tests
pin is exactly the four ways this could lie: a hit that is a shared substring rather than a shared *prefix*,
identical content being shared when the cache is off, retained blocks being counted as free, and a corpus quietly
dropping the records it cannot parse.

### The launch that answered it, and the acceptance number it left

The cache-on launch was restarted at 13:32 on 2026-11-04 to pick up the About payload (an in-process plugin older
than the card cannot publish it). Counters at the moment before the stop, for the whole launch:

- `prefix_cache_hits_total` **80,180,672** prompt tokens arrived as cache hits, `num_preemptions_total` **0**.
- `spec_decode_num_drafts_total` 264,253, `spec_decode_num_accepted_tokens_total` 358,742 → **1.358 accepted per
  draft**, up from the 1.281 read 40 minutes into the launch and still below the **1.438** of the cache-off launch
  it replaced.

So the open number in the section above stays open: it moved with traffic shape (1.28 → 1.36 on the same config)
without any config change, which is exactly why the fixed-prompt `bench/quality.py 10 8` gate -- not a live window --
is the only thing that can decide whether `align` costs acceptance. Geometry was unchanged by the restart
(`block_size 832`, `num_gpu_blocks 1242`, `kv_cache_size_tokens 977,726`, `mamba_cache_mode align`).
