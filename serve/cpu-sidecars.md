# CPU sidecars: the perception layer on the EPYC

The plan this directory's CPU processes live by. One page, because the budget (cores, memory, ports) is
one budget and it has to be readable next to the launchers that spend it.

## The measured starting point (this box)

* 4× R9700 under the standing Flash-Next launch: 30.3–30.5 GiB of 31.86 a card at util 0.96 — ~1.35 GiB
  spare a card, so no second serve on the cards, and every sub-15 ms model belongs on the CPU
  (same reasoning that put the embedding sidecar there: notes/embed-sidecar-box-test.md).
* Engine steady state: ~11 of 192 cpus (notes/native-4xr9700.md); embedding + rerank already pinned and
  measured to cost the launch 0.0–0.2 ms ITL / 0.3 % throughput at saturation.
* Free when this was written: a few dozen cores and ~18 GiB.
* Load windows are swap-I/O bound (110.6 GiB streamed through 93.8 GiB with 38.7 GiB pinned): anything
  started during a cold start pays for it, and 2026-11-01 showed what a big anon commit does
  (notes/load-oom-svm-wedge.md). mmap-only is the rule for every sidecar here.

## The budget

| Layer | What | Resident | Cores (taskset) | Form |
|---|---|---:|---|---|
| L0 — running | Qwen3-Embedding-0.6B (`embed.sh` TASK=embed, :8090), Qwen3-Reranker (`TASK=rerank`, :8092) | ~1.5 GiB | 16–31 / 32–47 (embed.sh defaults) | llama-server, mmap, daemon |
| L1 — Phase 1 | diarize (pyannote segmentation-3.0 + CAM++), align (Qwen3-ForcedAligner) | ~1.5 GiB | 48–63 (percept.sh default) | CLI-shaped (sherpa-onnx offline), no daemon yet |
| L2 — Phase 3 | SenseVoiceSmall GGUF + Kokoro-82M-zh, on-demand | ~1 GiB when up | 64–79 | TTL'd: idle unload, wake on request |
| L3 — the rest | ~13 GiB left as the load shock absorber (page cache / SVM margin) | — | 8–15, 64–191 mostly idle | no model; this is what the cold start spends |

Core map in words: the engine floats unpinned over everything and takes ~11; embed and rerank keep the
ranges embed.sh measured (16–31, 32–47); the perception layer takes the next free range (48–63) — one
range per serve, never two serves on one range; 8–15 stay empty for the engine's burst. cpu 17/113 are
offline and fall out of every range silently (host/offline-core.cpus).

Ordering rule, in both directions: **up** — engine first, L1 after it reaches serving; **down** —
`./stop.sh` already stops monitors → door → sidecars (embed, rerank, percept) → engine, and the percept
tasks are enumerated from serve/percept.sh the same way embed's are read out of serve/embed.sh.

## Guards

* `percept.sh` refuses to run below MEM_FLOOR (default 10 GiB MemAvailable) — the answer under memory
  pressure is "later", not "swap".
* No `--mlock` anywhere; weights by mmap/page cache only; KV/scratch anon sized deliberately (embed.sh's
  header carries the Committed_AS arithmetic).
* ONNX-based sidecars cap their intra-op threads (kokoro-js upstream #361 measured 1.9× aggregate
  throughput lost to default pools sized to all 192 cpus). sherpa binaries take --num-threads, which
  percept.sh sets to the ONLINE cpus in the pinned range, same arithmetic as embed.sh's -t.

## Acceptance protocol (Phase 1)

Executed as a step list with failure branches: `notes/cpu-phase1-checklist.md`. The four numbers it must
produce are restated here so this page stays the single source of truth for the budget: the sidecar must
cost the standing launch ±1 % aggregate, pswpout ≈ 0 while serving, MemAvailable ≥ 12 GiB steady, and a
cold start no slower than the recorded baseline.

1. `bash bench/probe32.sh` on the standing launch, before and after the first diarize run — aggregate
   within ±1 % — the pair is recorded in the checklist's results block.
2. During a diarize soak: pswpout delta ≈ 0 while serving (serving is allowed; the load window is not —
   see the ordering rule).
3. MemAvailable in steady state ≥ 12 GiB with L1 resident (that is also the threshold Phase 3's daemon
   uses to unload L2).
4. Cold-start time unchanged against the recorded baseline — the 13 GiB L3 buffer exists to protect it,
   so a cold start that got slower means L3 got spent and this page is stale.

## Monitor (:8094)

Phase 2's observer, pulled forward: `bash serve/percept-watch.sh` serves a stdlib page over the run
journal (`serve/percept-watch.py`; journal-driven because the layer is CLI-shaped — there is no resident
port to poll). It renders the journal's rt/rc ledger, the live inventory the launcher's guards check
(binary, the two model files, online cpus in the pinned range, MemAvailable vs MEM_FLOOR), and the
host-memory card this page's acceptance is stated in. Observe only, like embed-watch; `bash stop.sh
--embedding` stops it. To put it on the unified Monitor page: append
`,/diarize=http://127.0.0.1:8094/state.json` to `SIDECAR_STATE` when launching `serve/gateway.sh`.

## Runtime pinning status (honest column)

| Task | Status | Pin this |
|---|---|---|
| embed / rerank | shipped (`serve/embed.sh`) | — |
| diarize | pinned shape, binary+models to fetch | sherpa-onnx build with the diarization example; segmentation-3.0 and CAM++ sherpa exports into `~/models/percept/` — checklist in the "model checklist" section below |
| align | shape only, runtime pending | Qwen3-ForcedAligner via ONNX export (sherpa alignment support first) or its own package in the venv; set `ALIGN_CMD` when chosen |
| sense / kokoro (L2) | deferred to Phase 3 | SenseVoice GGUF runtime is single-binary with built-in FSMN-VAD; Kokoro: worker recycle (upstream issue #152 leaks on the CPU path) + capped intra-op threads |

## Model checklist (fetch targets for diarize)

Verified through the mirror 2026-11-27 (exact curl commands in notes/cpu-phase1-checklist.md §3):

* pyannote **segmentation-3.0**, sherpa export → `~/models/percept/segmentation-3.0.onnx`
  (`csukuangfj/sherpa-onnx-pyannote-segmentation-3-0`, ungated, 6.1 MB).
* **CAM++** speaker embedding, sherpa export → `~/models/percept/campplus.onnx`
  (`csukuangfj/speaker-embedding-models`, file `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx`, 27 MB).
  A zh-cn model on a zh-heavy corpus is the point; for predominantly-English corpora pick the
  `_en_voxceleb` sibling in the same repo and say so in the record.
* sherpa-onnx built with diarization on (upstream default since at least v2.0.3), binary on PATH or
  SHERPA_BIN set. DRY (`DRY=1 TASK=diarize bash serve/percept.sh`) prints the exact pinned command and
  what is missing, without starting or downloading anything — same DRY contract as every other launcher
  here.

## Explicitly not on the CPU

* A second 27B-class LLM: the cards carry Qwen3.8-27B at ~200 TPS single-stream in the pod's own
  catalogue; the CPU path tops out near 20 TPS for the same weights — 10× slower and 7 GiB, only ever
  worth it as a card-down fallback, which the 52-model catalogue already covers by switching launches.
* Image generation on the CPU: the cards do 512² in ~2 s; the CPU path is 20–40 s for comparable quality
  (estimates, no EPYC numbers found upstream). Never worth the 5 GiB while cards are up.
* Video generation on the CPU: minutes-per-second-of-video territory; don't.
