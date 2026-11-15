# v0.4.5 — the front door learned to wait, and the fault hunt stopped guessing (2026-11-15)

## What was gated, and what was not

The eight-card gate that v0.4.4's note admits was not run has **still not been run** for this tag either. What
this tag changes inside the served model path is diagnosis behind a knob production does not set:
`R9K_GDN_DEBUG` / `R9K_GDN_DEBUG_ALL` in `r9700_vllm/models/gdn.py`. With the knobs off, `_dbg_here()` is False and
the changed code never executes; the live serve's own R9K_ environment is listed by `tools/recipe-check.py`, which
names neither. That is the whole defence, and it is thinner than a gate run.

What was measured on this box, on this day:

* the admission gate's arithmetic and its fail-open paths (tests), and its behaviour under a real soak;
* the device fault, which is now reproducible in **45–536 seconds** instead of once every day or two;
* one upstream fix, applied, and shown to be the wrong fix for this fault;
* the card-order discriminator, which is the result that changed what the box's next experiment should be.

## Changed

### The front door — the admission gate (`serve/gateway.py`, `serve/gateway.sh`)

`/v1/chat/completions` and `/v1/completions` are held, not rejected, while the KV pool is over the line
(`GATE_KV_HIGH=0.85`, reopen `GATE_KV_LOW=0.78`, hold `GATE_HOLD_S=25`). Three `/metrics` gauges drive it
(`kv_cache_usage_perc`, `num_requests_running`, `num_requests_waiting`); a closed gate never opens an upstream
connection; missing or stale gauges fail **open**, because a gate that trusts its own outage over the engine's
health turns a bad metric into an outage. Only those two routes are gated — `/health`, `/metrics`, `/monitor` and
`/api/*` stay free, so the page that shows the gate is never behind it. Defaults live in `gateway.py` alone;
`gateway.sh` passes overrides and holds no second copy of the numbers. Gauge names are matched with the brace
(`vllm:num_requests_waiting{`), because `vllm:num_requests_waiting_by_reason` is in the same payload and would be
read as a queue length. `notes/admission-gate.md` carries the reasoning and the knob table.

### The pages — the gate gets a row (`serve/web/monitor.*`)

One line, because the Monitor's slots are the constraint: `Open | under the line | 0 of 902,153 tok | held 0 |
released 0 · expired 0 | hold 25s · poll 1s`. `expired` is the number that matters — a rising count says pressure
is not coming down on its own. `tests/render-monitor.mjs` renders the real payload rather than trusting markup
(hidden when there is nothing to say, filled when held, reason clause present, one history node per sample), and
`tools/monitor-fixture.py` grew a HELD fixture so that state is reviewable without generating load.

### Launch supervision (`setup.py`) — a crashed-and-wedged launch is visible, and restart is opt-in

On 2026-11-15 the engine died and supervision reported a healthy box for 26 minutes: `Worker_TP0` gone, the other
three ranks alive at ~32 GiB each and ignoring SIGTERM, the tmux pane still up, `died_at` still `null` — and after
a manual STOP the empty session blocked the next launch ("already up"). Now: `WATCH_WEDGE_TICKS` silent probes + a
crash signature in the log tail + ranks present read as **crashed and wedged**, cleared with the recipe's own
`STOP=1 bash serve/<recipe>` rather than a hand-rolled pkill, and only when all five gates in
`stop_wedge_would()` hold. Leftover sessions are closed only under `corpse_session()`'s three conditions.
Restart requires `WATCH_RESTART`; the mechanism was field-proved first (`.r9700-state/restarts.json` holds two
restarts, sig `illegal memory access`) and then switched off on purpose. With restart off the record writes
`restart_max=0`, not the default 3.

### The venv is a carried artifact now (`patches/`, `setup.py`, the plugin)

`patches/` holds `vllm-0.32.0-61174-clear-stale-block-table-rows.patch`, the unpatched reference copy, and
README with apply/verify/revert. `setup.py`'s box walk reports `venv patch applied / NOT applied` beside the
version numbers, and `--dry-run`s the patch against the saved original so venv drift is caught there; an unreadable
install dir reads as unknown, never as "missing the fix". The engine says at boot what it holds:
`r9700 webui loaded in vllm 0.32.0 (this process) -- venv patch state: #61174 present`.

### Diagnosis in the model path (`r9700_vllm/models/gdn.py`)

Host-side bounds **and span** checks before the stock `_forward_core` (`non_spec_query_start_loc[-1] +
spec_query_start_loc[-1] <= num_actual_tokens`, `num_actual_tokens <= rows`, conv-state line indices), because
`CatArrayBatchedCopy_contig` reads ranges and a step whose spans do not add up is wrong even when every index is
in range. Cost is the caller's: layer 0 unless `R9K_GDN_DEBUG_ALL=1`. The helper used to filter layers itself, so
`DEBUG_ALL` charged every layer for syncs while only layer 0 was ever checked — the layer that died was never
checked at all; fixed, and pinned by tests.

### Two tools

`tools/soak-mixed-step.py` reproduces the faulting step shape on purpose (paired short prefills force the stock
path, fresh prompts defeat the prefix cache, `--max-load` keeps the queue shaped like the field dumps) — 45–536 s
to a fault. `tools/gpu-rank-map.py` answers "which physical card does each rank hold" from KFD's per-process
records, since `/proc/<pid>/fd` cannot (every rank keeps all four render nodes open for P2P).

## Measured on the launches of this day — none of which carries 0.4.5

The live serve still answers `vLLM 0.32.0 · plugin 0.4.4 · a3d1e53e*`: the string is computed when the sidecar
starts, and the tree moved under it. No number below comes from a launch carrying 0.4.5.

* **Six device faults, one card.** kv_usage 30 % → 99.8 %, waiting 0 → 13, uptime 4 min → 6.8 days, patched and
  unpatched, four instrumentations — and `0000:c3:00.0` in every row. Under `GPUS=3,0,1,2` rank 0 moved to
  `06:00.0`, and the fault named `c3:00.0` and `Worker_TP1`, the rank now sitting on it. The guard printed 0 `BAD`
  across ~1 150 steps, and the four ranks' stage counts were identical (63/63/63). `c3:00.0` carries zero AER
  counters; the box's noisy PCIe device is `0000:04:00.0` (BadTLP 579, Timeout 20) on the `06:00.0` chain — a
  different card, a different failure mode, spelled out so the two do not merge.
* **#61174 applied and disproved.** Upstream's own test: 7 failed → 7 passed against this wheel. As a fix for
  these faults: a soak hit the identical fault at **250 s** with the patch, against 431 s and 536 s without.
* **The gate has never actually held anything in production.** Polls ~11.5 k per launch, `closed=False`,
  `held 0`: on this box's traffic the pool did not cross 0.85 during normal service. The hold/release/expiry path
  is exercised by tests and by a soak that drives the pool up, not by a working day. Reporting it as though it were
  proven in service would be the kind of claim this tree keeps getting told not to make.
* KV pool 902,153 tokens on the production launch (886,408 on the debug TEST launch, the difference being the
  instrumented build's overhead, not a knob).

## Not in this tag, and owed

1. **Silicon or slot.** No config change can separate them; the card has to move by hand.
2. **Run without that card** (`GPUS=1,2,3 TP=3`): both a mitigation and a seventh confirmation — but three cards
   means ~33 % more weights per card, and whether the KV pool survives is a measurement, not a guess.
3. **A gate that has actually held** in production, and a bench of what holding costs the tail.
4. **The eight-card gate**, still not run for either tag.
5. **The fault itself.** Six data points and a card; no mechanism. `AMD_SERIALIZE_KERNEL=3` would name the
   kernel that tripped over the card, not the reason the card trips.
