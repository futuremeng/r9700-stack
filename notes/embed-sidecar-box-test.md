# The embedding sidecar next to the standing launch: what the box says (2026-11-15)

The sidecar (`serve/embed.sh`, `serve/embed-watch.py`, `setup.py --embedding`) was written against the notes and
never run. It has now run on this box — the same box, the same **running** `serve/native-flashnext.sh`
(Qwen3.8-Flash-Next, TP4, 262k context, `:8080`), no restart in between — and it works. Four things in it did not
work the first time and are fixed with this change; the numbers below are what the fix bought and what the
configuration is worth.

## What the box is

Four R9700s, AMD EPYC 9654 (192 threads, cpu 17 and 113 offline — `host/offline-core.cpus`,
notes/incident-2026-11-03), 93.8 GiB RAM, 8 GiB swap. The engine's five processes (4 workers + EngineCore) carry
~87 % of a core each while idle (vLLM's spin) and are **not pinned** — their affinity mask is every online cpu.
Nothing had been added to the box's resident set in months.

## Four things that did not work first time

| what | what it did | the fix |
|---|---|---|
| `setup.py llama_build` | `cmake -B ~/llama.cpp/build` — `run()` has no cwd, so **this folder** became the source tree: `CMake Error: The source directory "/home/jiulu/r9700-stack" does not appear to contain CMakeLists.txt` | `cmake -S ~/llama.cpp -B ~/llama.cpp/build`; the same wrong one-liner in `embed.sh`'s own help text, fixed there too |
| `HF_REPO=unsloth/Qwen3-Embedding-0.6B-GGUF` | `401 {"error":"Invalid username or password."}` from the hub API: gated on HF. The download step dies on any box without a token | `Qwen/Qwen3-Embedding-0.6B-GGUF`, which is public: `Qwen3-Embedding-0.6B-Q8_0.gguf`, 639,150,592 B, sha256 `06507c7b…`, pulled on this box 2026-11-15 |
| `embed.sh`'s fetch | `pip3 install -U huggingface_hub[cli]` into whatever `pip3` is (system pip here: externally managed), then reached for `~/.local/bin/hf`, which does not exist here (`hf` lives in `~/venv-r9700/bin`) | use an `hf` already on PATH, never pip-install from a launcher; say what a proxied box needs (`https_proxy` — git carries its own `http.proxy`, curl and hf do not) |
| a second `bash serve/embed.sh` while one runs | llama-server dies in the bind (`couldn't bind HTTP server socket, port: 8090`) while the health loop sees the **first** serve answering 200 → `embedding up: pid 376676`, rc 0, for a pid already gone | refuse before launching (and wait for a mid-`STOP` serve to be gone before refusing); check our own pid before believing `/health` |
| `embed-watch.py`'s `COUNTERS` | every card read `-`: the guessed `llama_cpp_*` names do not exist | the names the serve really publishes (below) |

`cmake` was not on the box and there is no root here, so it came from pip into the repo's own venv
(`.venv/bin/pip install cmake ninja`) — recorded because `llama_build`'s advice ("sudo apt install cmake") is
unreachable on this box.

## llama.cpp's actual /metrics names (build 1, commit 5c008e0)

```
llamacpp:prompt_tokens_total      llamacpp:prompt_seconds_total     llamacpp:prompt_tokens_seconds
llamacpp:prompt_tokens_cached_total  llamacpp:requests_processing  llamacpp:requests_deferred
llamacpp:n_decode_total           llamacpp:n_tokens_max            llamacpp:n_busy_slots_per_decode
```

**There is no request counter in this build**, so the monitor's "req/s / ms per request" was not derivable even
with the right names. The card now leads with prompt tokens/s (the delta of `prompt_tokens_total`, cross-checked
against the serve's own `prompt_tokens_seconds` gauge), the busy-slot count, the deferred rate, and ms per 1k
tokens (`prompt_seconds_total / prompt_tokens_total`, which measured 1,436 ms/1k against a 694 tok/s gauge —
consistent). The names are still matched defensively: an absent one reads `-`.

## The memory claim, measured at its real size

The update's load-bearing rule — mmap, never `--mlock`, never `--no-mmap` — holds, and `embed-watch`'s RSS
expectation did not. From `/proc/<pid>/smaps` with the serve up at `-c 8192`:

```
/home/jiulu/models/qwen3-embedding-0.6b-q8_0.gguf   Rss 618,368 kB
    Shared_Clean 0  Private_Clean 618,368  Private_Dirty 0  Anonymous 0   VmFlags: rd mr me ms sd
process rollup:   Rss 1,613,800 kB   Anonymous 977,404 kB   Locked 0
/proc/meminfo:    Committed_AS 85,149,980 -> 87,915,288 kB  (MemTotal 98,389,904)
```

* The weights are **exactly** what the note promised: 618 MB resident, all of it clean file-backed, zero dirty,
  zero anon, nothing locked — reclaimable page cache, invisible to swap and to `Committed_AS`.
* What is anonymous is the **KV pool and the scratch**: 977 MB at `-c 8192` (llama.cpp runs 4 slots by default,
  so 4 × 8192 tokens), and the process takes **+2.77 GB of `Committed_AS`** (85.15 → 87.92 of 93.8 GiB). `STOP=1`
  gives all of it back (85.15 GB again).
* So "the sidecar's own RSS should read ~0 while the GGUF stays in the page cache" is wrong as written: RSS reads
  ~1.6 GiB (weights touched + KV). The number the 11-01 red line is actually about is `Committed_AS`, and the
  monitor now shows it. **This box's commit headroom went from ~8.7 GB to ~5.9 GB** — still far from 11-01's
  122.6 GB, but it is a real change to the box's margin and it should be read on the page, not remembered.

## Does it cost the running launch anything? No — measured, not argued

Same box, same engine process (`pid 320010`, never restarted), `bench/itl.py` (median inter-token latency, ms):

| engine load | sidecar idle | sidecar saturated |
|---|---|---|
| conc 1 | 13.0 | 12.9 |
| conc 2 | 15.0 | 15.0 |
| conc 4 | 18.4 | 18.3 (t=8) / 18.6 (t=31) |
| conc 8 | 21.2 | 21.1 |
| conc 16 | 27.3 | 27.3 |

and end-to-end, a fixed 16-request × 150-token chat load at concurrency 8: **705.5 tok/s** with the sidecar idle,
**703.7 tok/s** with it saturating its core range (−0.3 %, i.e. noise). The same shape run the other way: the
sidecar held 2,046 tok/s while the engine served that chat load. `PSI memory` stayed at 0, swap free did not move
(4,050 MiB used before, after, and during), MemAvailable moved only by the ~1 GB the model's clean pages occupy.

Why: 190 online cpus, of which the engine's five processes want ~5-7 (each worker reads 87 % of a core idle, its
own spin loop, and ~170 % under a concurrency-16 chat load — TP4 puts four workers and an EngineCore in the
mix). The sidecar takes a taskset range of 15-48 of them. There is no contention to find.

## What the sidecar is worth, and where the knob is

A saturated indexing run (documents of ~420 tokens, `/v1/embeddings`, dim 1024):

| threads | cores | tok/s | note |
|---|---|---|---|
| 8 | 16-31 | 689 | the default the update shipped |
| 15 | 16-31 | **1,112** | one thread per **online** cpu in the range |
| 16 | 16-31 | 748 | **oversubscription: −33 %.** cpu 17 is offline, so the range is 15 cores; llama.cpp's workers spin |
| 31 | 16-63 | 2,052 | what the box can do if an indexing run is what you are waiting on |

So `THREADS` now defaults to the number of **online** cpus inside `CORES` (counted from
`/sys/devices/system/cpu/online`), and a hand-set `THREADS` above that count is called out in the launcher's own
output. At the default the sidecar indexes **4.0 M tokens/hour** (1,112 tok/s) — 96 M tokens/day, ~210 k
documents of 450 tokens, if it were given the box all day.

* A **single query** embedding, box idle: **40-70 ms**. Under a saturated indexing run: 1.2-4.5 s (4 slots, queue).
* Batching documents into one request (`input: [d1..d8]`) does **not** beat concurrency: 748 vs 752 tok/s. The
  server is compute-bound at its thread count; batch only saves HTTP round trips.
* Concurrency past 4 (llama.cpp's default `--parallel 4` slots) buys nothing either — 690 tok/s at conc 1 and at
  conc 8. If a big indexing run matters, raise `--parallel` in `LLAMA_EXTRA` rather than the client count.

## The vectors themselves

Not a golden set — a sanity check that the pipeline is not returning noise: with the Qwen3 instruct prompt on the
query side and plain text on the document side, `cosine(query, on-topic doc) = 0.766` and
`cosine(query, off-topic doc) = 0.219`. Whether 0.6B is **good enough** is still the open question the change
note says it is; that needs retrieval runs with a fixed evaluation set, and per that note BM25 + a 0.6B reranker
(`--rerank`, same launcher, `LLAMA_EXTRA`) is the cheaper answer to it than a 4B embedder.

## Not tested here

Deliberately: no memory-pressure test (the point of the mmap rule is that clean pages get reclaimed instead of
swapped; forcing that on a box running a five-day-old launch to prove a point is not a trade worth making — the
smaps evidence above is direct instead). Also not tested: the 4B q8_0, the `--rerank` variant, what happens to
the metric names when llama.cpp updates (they are matched defensively, and this note pins the commit they were
read from), and the RAG quality question.

## Reproducing it

```
PATH=$PWD/.venv/bin:$PATH .venv/bin/python setup.py --no-docker --yes --embedding --launch none   # builds llama.cpp
LLAMA_SERVER=$HOME/llama.cpp/build/bin/llama-server bash serve/embed.sh                          # :8090
bash serve/embed-watch.sh                                                                        # :8091
.venv/bin/python bench/itl.py 1 2 4 8 16                                                          # engine latency
# and for the sidecar: N concurrent /v1/embeddings over ~420-token documents, /metrics read before and after.
```
