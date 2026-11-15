# The admission gate: why the front door holds on the KV pool, not on the sequence count

Box: `jiulu-9700`, 4 x R9700, engine recipe `native-flashnext.sh` (`--max-model-len 262144 --max-num-seqs 32
--max-num-batched-tokens 8192 --kv-cache-dtype fp8_e4m3`, MTP `num_speculative_tokens=2`).
Code: `serve/gateway.py` (the gate), `serve/gateway.sh` (its overrides), checks in `tests/test_gateway.py` §(8).
Status: **on** on this box since 2026-11-15 05:37, at the built-in defaults.

## 1. What the logs said, in their own numbers

The request journal (`~/logs/requests-8080.jsonl`, `requests-8180.jsonl`) and the engine's 10 s stats from the
6.8-day launch (`~/logs/setup-005730.log`, 27 508 samples) fit, on 28 clean pure-decode windows
(`Running==1`, `Waiting==0`, no prefill inside the window, two consecutive windows with identical `Running` and
KV), R^2 0.887:

```
step_ms ~= 13.6  +  1.79 x N(sequence)  +  14.9 ns x Sum(resident tokens)
```

with `Sum(resident tokens) = kv_cache_usage_perc x 902,153` (the pool the engine printed at boot: *"GPU KV cache
size: 902,153 tokens"*, from `Available KV cache memory: 6.65 GiB`, so ~7.7 KB/token). `N` came from the
`N>=2` windows the same way (`(step - 13.6 - 14.9ns x T) / N`), median 1.79 ms/seq, p25 1.61 / p75 1.91.

Checked forward against 103 windows of the 11-15 launch, which was fitted on a *different* launch:

```
04:55:41  N=6  T=251,701   77 tok/s/req measured   79 predicted
04:56:31  N=5  T=165,094   102                     99
05:05-05:08  N=9-10  T~500k  ~55-65                ~58
```

Two budgets fall out of it, with different exchange rates:

| | one 1.5k row | one 20k row | one 80k row | one 236k row |
|---|---|---|---|---|
| per-step time | 1.81 ms (1.0x) | 2.08 ms (1.2x) | 2.98 ms (**1.6x**) | 5.30 ms (**2.9x**) |
| pool | 0.17% | 2.2% | 8.9% (**= 53 short rows**) | 26.2% |

So "concurrency" is not one number. `run=25` appeared twice in the journal with completely different machines
behind it: 05:11 today = 25 sequences, Sum own-context 280,572 tokens (longest row 79,619, median row 370), KV
59.1%; 11-14 20:59 = 31 sequences, Sum own-context 596,090 (longest 236,153), KV 99.8%, 13 waiting. Both
crashed; `--max-num-seqs 32` was in the argv of both, and of the launch that behaved.

Measured per-request tok/s in the long launch (`Waiting==0`, no prefill in the window), which is the shape the
gate is aimed at:

```
 N       4-20k/req  20-60k/req  60-120k/req      total tok/s for the same row
 1          144        144         144              144
 2-4        127        110         108              233-378
 5-8         97         90          83              472-501
 9-16        -          64           -              600
```

Per-request rate falls with N (the 13.6 ms fixed part is the same step for everybody), total throughput *rises*
with N. Nothing in that table says "stop at N=12". What says it is the pool: the collapse windows of that same
launch (11-12 09:00-22:00, 11-14 17:00-21:00) sat at **kv 84-100% with a waiting queue** -- 24.6% / 18.1% of
samples with `Waiting>0`, 14.9% / 21.1% of samples at kv>=90%, 423 preemptions over 68 requests in the last
3.3 h of it, and TTFT p99 376 s / max 1 960 s. In those windows the model above stops describing anything
(measured 2.7-6.1 tok/s where it predicts ~60), which is the signature of recomputing rather than decoding.

## 2. What the gate does

`serve/gateway.py` is a byte pipe, and it stays one: the gate opens **no upstream connection** while it is
closed. The client waits; no body is read, parsed or buffered, so the 0.8 ms/req the hop costs
(`notes/embed-sidecar-box-test.md`) does not change.

It closes on what the engine already measured about itself -- one `GET /metrics` per `GATE_METRICS_S`
(12 ms and 68 KB on this box, in a coroutine, off the request path):

* `vllm:kv_cache_usage_perc >= GATE_KV_HIGH` (0.85 ~= 767k of 902,153 tokens) -- the pool axis, the one the
  slow windows were on;
* `vllm:num_requests_waiting >= GATE_WAIT` (1) -- the engine queueing is the thrash signature on its own.

It re-opens below `GATE_KV_LOW` (0.78). The hysteresis is not decoration: one 8 192-token prefill moves this
pool ~0.9%, so a single threshold would chatter. The prefix `vllm:num_requests_waiting{` carries its brace on
purpose -- without it the same prefix matches `..._waiting_by_reason`, which is a scheduler choosing to wait,
not a queue growing.

Only `/v1/chat/completions` and `/v1/completions` are gated: `/health` is what `setup.py`'s watcher polls,
`/metrics` is what the gate reads itself, `/api/*` is the Monitor, and `/v1/embeddings` / `/v1/rerank` are CPU
llama.cpp serves that cannot take a KV hit.

**Holding, not refusing.** `GATE_HOLD_S` (25 s) is the client-visible ceiling: after it the request goes through
anyway and is counted `expired`, because trading a 30-minute TTFT for a hung connection improves nothing.
`GATE_HOLD_S=0` is the other policy -- a 503 whose body carries the gate's own state, and the request never
reaches the engine.

**Fail-open, twice.** No gauges in the answer (a rename, another server) -> open, and `err` says so. A gauge
older than `GATE_STALE_S` (5 s), e.g. an engine whose `/metrics` stops answering -> open. A front door that
stops serving because its telemetry broke is worse than no gate.

What the gate deliberately does **not** do: estimate prompt tokens. A body-parsing proxy costs 2-3 ms a request
(`bench/webui_ab.py`), and a `Content-Length/4` estimate is a number no test could hold. `GATE_POOL_TOKENS` is
display-only (it renders 0.854 as "767,000 of 902,153 tokens"); the decision reads only the fraction. Known
inaccuracy, stated rather than hunted: 902,153 is the pool of the **production** launch, and a different launch
shape gets a different pool -- the 2026-11-15 debug TEST launch came up with **886,408** tokens, so the Monitor's
"... of 902,153 tok" overstates the denominator by ~1.8 % against that launch. No decision depends on it, because
the gate works on the fraction `/metrics` reports.

## 3. Knobs, and where they live

Defaults live in `serve/gateway.py` and are the ones this box runs; `serve/gateway.sh` passes an override
through and prints it, and nothing goes in `~/.config/r9700-stack/native.env` because these are measured
thresholds, not local bindings (§10's rule). The names the shell forwards:

```
GATE_DISABLE GATE_KV_HIGH GATE_KV_LOW GATE_PREEMPT_S GATE_WAIT GATE_MAX_INFLIGHT GATE_SMALL_BYTES
GATE_MAX_HELD GATE_HOLD_S GATE_METRICS_S GATE_STALE_S GATE_HIST GATE_POOL_TOKENS
```

`tests/test_setup.py` holds that list against the `os.environ.get("GATE_...")` names `gateway.py` actually reads,
in both directions: a knob the launcher cannot pass is a knob that does not exist, and a default written in the
shell as well as the program is two policies with only one of them visible in the running process.

Shipped values after §6: shut at `GATE_KV_HIGH=0.92`, re-open the pool axis at `GATE_KV_LOW=0.85`, stay shut
`GATE_PREEMPT_S=20` s after the last observed preemption, `GATE_WAIT=0` (an engine queue alone never closes the
door), `GATE_SMALL_BYTES=8192` (a declared body that small passes a shut door), `GATE_MAX_HELD=8` (never hold
more than this many at once), `GATE_HOLD_S=5` (the ceiling on any single hold), `GATE_HIST=900` polls of shut/open
history -- ~15 min at the shipped 1 s poll, which is §6's question made answerable after the fact: the afternoon
left behind "was the door shut at 15:05?", and a 180-poll ring could only speak about the last three minutes.

`GATE_MAX_INFLIGHT` (0 = off) is the sequence axis, kept because the two budgets are independent: a fan-out of
short requests is a different problem from one 200k request, and only the second one moves the pool gauge.

Live state: `GET /api/gate` -- closed/why, the gauges it read, `err`, counters (`held released expired refused
admitted`), the config, and `hist`: one entry per poll (1 shut / 0 open, and a poll with unusable gauges draws 0
rather than a gap) over `GATE_HIST` = 900 polls ~= 15 min at the shipped poll rate, with the span itself stated in
the payload (`hist_span_s`) rather than left for the reader to multiply -- the Monitor prints it in the strip's
title ("30 polls over 15 min"). The strip is the poller's own
record rather than something `/api/gate` reconstructs on read, so a page that asks during a shut window and a
page that asks after it cannot disagree.

**The Monitor row** (`serve/web/monitor.html` + `paintGate()` in `serve/web/monitor.js`): two cards, in a section
that hides itself when `/api/gate` is not answered -- the page is also reachable on the engine's own port, where
no gateway is in front, exactly as the sidecar row behaves.

* *Admission gate*: the state word (Open / Held / Off), the dot (live / busy / off -- a hold is the gate working,
  not an outage), the history strip on a fixed 0-1 axis, the reason and the pool in tokens
  (`775,852 of 902,153 tok`). The reason line carries the gateway's own first clause and its **title** carries
  the gateway's full string with the token count: that arithmetic lives in one place, and the 24-character budget
  of the tightest 4-up card is what puts the long form in the title.
* *Holds at the door*: the running total of held requests, then `expired E . small S`, with `refused F` and
  `over O` joining the line only when they are not zero -- either one is the gate choosing traffic over protection,
  which is exactly the count worth running wide for. `expired` against `held` is the ratio that says whether the
  ceiling is acting as a bound or just as a delay; `small` is the count of cheap requests that had to be let
  through a shut door, which is the sign a pool threshold was starving the wrong traffic. released / refused /
  admitted / held_now / held_over are all stated in the gloss, and a gateway old enough to lack the newer counters
  gets the older `released . expired` pairing rather than a column of zeros.

## 4. Proved, on this box

`tests/test_gateway.py` §(8), 14 checks against a fake engine whose `/metrics` the test drives:
it reads the engine's gauges rather than guessing; a shut gate holds a chat request and then serves it; the hold
is counted; hysteresis (0.95 shut, 0.82 still shut, 0.70 open); `Waiting` alone shuts it and reads
`num_requests_waiting{` and not `..._by_reason`; `/health /metrics /v1/models /api/gate` and both sidecar routes
answer immediately while it is shut; a shut gate keeps polling (it cannot starve its own reading); the ceiling
lets the request through with the *engine's* answer; `GATE_HOLD_S=0` answers 503 and the engine never sees the
request; a gauge-less `/metrics` opens it; a hung `/metrics` opens it; `GATE_DISABLE=1` is a real switch.

Live, against the real engine (2026-11-15 05:40), on a throwaway gateway on :18080 with the gate forced shut
(`GATE_KV_HIGH=0 GATE_WAIT=0`) and `GATE_HOLD_S=4`: the completion arrived after **4.098 s** with the engine's
answer, counters `held=1 expired=1`; the ungated :8080 path answered the same request in **0.084 s**. The
gateways were stopped by pid, not by pattern (`launch-methodology.md` §11).

The page was rendered, not eyeballed: `node tests/render-monitor.mjs` runs `monitor.js` unmodified in a stub DOM.
Against the live gateway it prints

```
gate row: hidden=false | Open | under the line | 0 of 902,153 tok | held 0 | released 0 . expired 0
          | hold 25s . poll 1s | strip 3 node(s), 147 polls of history
```

with every gate slot inside the 24-character budget and no BAD, and with `tools/monitor-fixture.py`'s gate payload
(a HELD pool reading, `held=41`) it prints the branch a live idle box will not show you:

```
gate row: hidden=false | Held | kv 0.860 >= 0.85 | 775,852 of 902,153 tok | held 41
          | released 33 . expired 6 . refused 2 | hold 25s . poll 1s | strip 3 node(s), 30 polls
```

Remove `gate` from the fixture and the same harness reports `hidden=true` with every slot empty -- the hiding rule
is checked, not assumed. The harness now keeps `LAST_GATE` the way it keeps `LAST_STATUS`: the payload the PAGE was
fed. Two checks were wrong before learning that -- one fetched `/api/gate` separately and compared (a freshly
restarted gateway paints with one poll of history while a second fetch seconds later has a dozen), and one counted
the sparkline element's children, which is 1 for every spark on the page because the paths live inside the svg.

## 5. What this does not fix

It is not the crash. Both device faults (`_causal_conv1d_fwd_kernel` at 11-14 20:59,
`at::native::CatArrayBatchedCopy_contig` at 11-15 05:11, same card `c3:00.0`, same read fault shape) happened on
a step that mixed new prefills with spec-decode rows, which with prefix caching on is handed to stock's
`_forward_core` -- and 05:11 took the fault at **kv 59% with nothing waiting**. The gate would not have closed
for either. It addresses the slow windows, which are a different failure with different numbers
(`notes/bug-causal-conv1d-oob.md` is the crash's file).

## 6. The batch that made the door permanent (2026-11-15 afternoon)

The gate went into production at 0.85/0.78 with `GATE_WAIT=1`, and a few hours later a batch of
data-processing requests arrived -- prompt tokens with a median of 542 but a **p95 of 105,058** and a max of
108,992 in the last 100 rows. What the gauges looked like while it ran, sampled over 36 s:

```
kv_cache_usage 0.87-0.90      running 5      waiting 14-17, all of it reason="capacity"
num_preemptions_total 1.0     (cumulative since launch: ONE preemption, all day)
```

and the gate's own history said:

```
hist: 180 of 180 polls closed      counters: held 14, released 4, expired 9, admitted 93
why:  kv 0.905 >= 0.78 (816,271 of 902,153 tokens); engine waiting
```

Three separate mistakes are visible in those four lines.

**A threshold below a workload's steady state is not a threshold, it is a permanent state.** Five resident
100k-token contexts are ~500k tokens = 55% of the pool on their own; the batch parked the pool at 0.87-0.90 for
its whole duration. `GATE_KV_LOW=0.78` is a line that batch can never cross, so the door never re-opened: 180 of
180 polls. Nothing about the pressure was transient.

**A queue is not thrash.** `waiting 17` with `reason="capacity"` is the scheduler admitting it has no blocks for
more rows; `num_preemptions_total` moving is the pool actually taking blocks back from a running request. The
first is saturation (normal, and the engine handles it in an order it chose), the second is the harm that made the
11-12 and 11-14 slow windows 68 victims pay a whole re-prefill each. v1 keyed on the first and never looked at the
second. On this batch the difference was not academic: **one preemption** while the door was shut for three
minutes and counting.

**Holding at the door does not reduce the pressure that is already there.** The KV those 17 requests are waiting
for is held by requests already running. A second queue in front of the engine cannot free a block; all it can do
is delay an arrival, and the delays land worst on the requests that would have cost almost nothing.

And the ceiling measured the damage: of 14 held requests, **4** went because the pool fell back under the re-open
line and **9** went because they waited the whole 25 s and were admitted anyway. A bound that is usually reached
is not a bound, it is a 25-second tax -- which is what "new requests can't get in" actually was.

So the policy changed from LEVEL to HARM, and the door got three escapes:

| | v1 | shipped now |
|---|---|---|
| closes on | `kv >= 0.85` **or** `waiting >= 1` | `num_preemptions_total` moved (shut 20 s after the last event) **or** `kv >= 0.92` |
| re-opens | below `kv 0.78` (unreachable under a long batch) | window expiry (always ends) / pool below 0.85 |
| hysteresis latch | the whole door | the pool axis only |
| small requests | held like everything else | a declared body <= 8192 B passes a shut door |
| hold queue | unbounded | at most 8 held, past that it admits |
| ceiling | 25 s (hit by 9 of 14 holds) | 5 s |

The preemption signal is a *delta* between polls, not the cumulative value: the counter has been sitting at 1
since the launch, and a gate that closed on a nonzero cumulative count would be shut forever in the other
direction. `num_requests_waiting_by_reason` is still deliberately not read as a queue -- the brace on
`num_requests_waiting{` is there for that reason, and the test fixture keeps emitting two `by_reason` lines with
values that would close the door wrongly.

Header-only, still. `GATE_SMALL_BYTES` reads the `Content-Length` the pipe already holds in order to frame the
upstream request; it never turns that number into a token estimate, and it never reads a body to make a decision.
The claim it needs is only "this is a 200-token question, not a 100k-token document", which is the distinction the
door was getting wrong.

Held in tests (`tests/test_gateway.py` §(8), against the shipped defaults rather than the tuned ones the other
checks use): a full pool plus a 17-deep engine queue and no preemption leaves the door **open**; a preemption
counter that moved closes it and names the window in the reason; the shipped ceiling is 5 s; the preemption
shutdown **expires** (1.2 s window, re-opened 1.6 s later with the pool at 0.88 -- the case v1 could not exit);
a small declared body passes a shut door and is counted; and at `GATE_MAX_HELD=1` the second arrival is admitted
instead of queued behind the first.

Went live at **14:43:50** by restarting the front door alone (`STOP=1 bash serve/gateway.sh`, then start; ~2 s).
The engine and its tmux session were untouched: same engine pid, `/health` 200, and no new `[gfxhub]` line in the
kernel journal (still 09:52:00 on `c3:00.0`). The last v1 readings before the bounce: **`held 45, released 4,
expired 41`** with the door shut on 180 of 180 polls, at kv 0.837 and `waiting 33`. The new process, three seconds
after it started, with the batch draining:

```
closed=False  why=""   gauges: kv 0.141  run 5  wait 0  prem 20
counters:     held 0  held_now 0  released 0  expired 0  refused 0  admitted 7  small_passed 4
```

and the Monitor row carries the same thing in words: `Open | no harm signal | 132,368 of 902,153 tok | held 0 |
expired 0 . small 4 | hold 5s . win 20s`.

Two things worth noticing in that line. `prem 20` does **not** close the door: 19 preemptions happened before this
process ever polled, and the baseline is the first reading rather than zero -- the cumulative count is history, and
only movement is harm. And restarting the front door resets the counters: `held/expired/small_passed` are since-
this-gateway-process, which is the same trade the engine's own counters make. The history that mattered from the
old process is in this file, not in the process.
