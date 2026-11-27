# Verbatim tool_call_id replay: a parked plan, waiting on one number nobody has measured.

Asked on 2026-11-27: "第 3 点展开" — the third takeaway from github.com/metaspartan/rad (RAD, the Rust+HIP
engine; not the same thing as `radiance`, see notes/radiance-test-run.md) is its prefix-cache trick: a tool
call's arguments and a tool result's JSON, if the client re-serializes them between requests (key order,
whitespace, escapes), fork every 880-token block after that point and the whole prompt-prefix hit is lost.
RAD's fix is one sentence of engineering discipline: render once, store the bytes keyed by `tool_call_id`,
and on every later request resend those exact bytes. 用户决定:先不做,列为备查方案.

Whether it is worth anything here depends entirely on one unmeasured number: **how much of the prefix miss is
reformatting-class**. If the replay's attribution says the class is ~0, or the miss is dominated by pool
eviction, this file is done and nothing changes.

## Where a prefix miss can come from in this stack

| source | where it lives | this trick applies? |
|---|---|---|
| compaction (pi rewrites its own context) | marked in `tools/prefix-replay.py` (`type == "compaction"`) | no — fork is inevitable, only classifiable |
| tools schema serialization order | `serve/templates/qwen-fixed-v22.3.jinja` renders tools first with `tool \| tojson`, inheriting the client's dict order | yes — canonical-rendering class, fixable template-side |
| tool arguments / tool results re-rendered | harness: `render()` in `tools/prefix-replay.py` does `json.dumps(b.get('arguments'))` per call on dict insertion order; external clients likewise | **yes — the bullseye of the RAD trick** |
| pool-capacity eviction | the LRU arm of `simulate()` (pool 1,175 blocks; one deep session holds hundreds) | no — this trick cannot help at all |
| first request / new session | — | no |

Do not anchor on the 94.8 % in serve/gateway.py: that batch was deliberately prefix-nested filler traffic. The
decision variable is the eviction arm of the replay.

## Step 1 — measure first (zero line cost)

1a. **First-divergence attribution in `tools/prefix-replay.py`.** Today it runs the OFF and ON arms and never
answers *why* a request misses. Minimal extension: keep each call's whole-block hash sequence alongside the
rebuilt prompt, find the first differing block between consecutive calls of one session, and classify the fork
point — compaction boundary → compaction; inside the system sections → schema; inside a rendered
`toolCall`/`toolResult` where the same message hashes differently across calls → **reformatting candidate**
(content genuinely changed = normal progression); everything else → suspected client re-serialization. Print
one extra line: miss blocks by class. The OFF/ON arms must stay byte-for-byte unchanged; the new logic only
adds.

1b. **Wire cross-check.** The rebuilt prompt is a guess at the harness's bytes. `tools/litellm-spy.py --dump`
captures real wire bodies; diff the two for a handful of sessions to decide whether schema/result serialization
happens pi-side or client-side, which picks the Step-2 branch.

## Step 2 — landing point by source

**(a) Fork in the pi harness (most likely, cheapest).** The record already stores the content; the fix is
harness-side (outside this repo): render arguments/results once, store the rendered bytes keyed by
`tool_call_id`, resend those bytes on later calls. That is the RAD trick, moved to the client — zero server
change, zero latency. Rerun the ON arm to see whether it eats the attributed class.

**(b) Fork in the tools schema (external clients with unstable dict order).** Canonical-render it in the
template (sort keys before tojson). This changes the whole prompt's bytes — qwen-fixed carries a +14 %
acceptance history — so it must re-pass acceptance + GSM8K gates, not ship as a cache-only change.

**(c) Fork in an untouchable external client only.** Never inside `serve/gateway.py`: the gate exists to
refuse reading the body (measured 2–3 ms/request for a body-parsing proxy, bench/webui_ab.py). If ever needed:
an opt-in sidecar following the embed/rerank precedent, routing only known-reformatting clients, bounded
storage (LRU by bytes, visible in /metrics), measured with bench/webui_ab.py — against ~230 ms/request of
prefill on a 43k-prompt box, 2–3 ms is noise *if and only if* the bytes are identical; byte-inequivalence
disqualifies the component outright.

## Step 3 — gates (what would have to be true to ship)

1. Replay gains a third arm, ON+replay, printed beside OFF/ON.
2. Live A/B: the uncached-token histogram in serve/webui.py shifts down and the `cached_unknown` rows do not
   grow — otherwise the miss merely became unattributable.
3. Parity is a hard gate, not an observation: temp=0 corpus replay must be bit-exact in output text and step
   count against the no-replay arm. Any tool_call_id that does not match or is not byte-equivalent must pass
   through untouched and be counted — never silently overwritten. RAD states this in one sentence; here it
   belongs in CI.

## Order of attack

1a (half a day) → if the reformatting class is ~0, close the file; else 1b to pick the side → prefer (a), it
covers the main traffic and changes no byte on the server → only traffic (a) cannot reach justifies evaluating
(c). For any external comparison use RAD's no-spec rows (110.8 tok/s decode, 4,435 prefill); never its
hybrid DFlash2+K=128 line. Related external surveys: notes/radiance-test-run.md; the rdna-boosts findings
(host-expert cache, chunked GDN WMMA prefill, hipBLASLt solution draw) live in this session's discussion and
share the same rule — benchmark rows with speculation excluded.
