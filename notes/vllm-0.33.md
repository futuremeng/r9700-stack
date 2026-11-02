# vLLM 0.33.0 on this stack: what moved since the pin, and what following it costs (2026-11-02)

"vllm 有更新吗？我们要不要跟一下？" — checked 2026-11-02 against three things at once: the release line
(v0.33.0), the ROCm 10 nightly line our image pins, and `main`. Short answer: **yes, follow v0.33.0 — it is
107 commits past our pin, it needs no plugin change, and the wheel is a drop-in for the native venv. Do not
follow `main` yet**: one upstream rename breaks two of our imports there.

## What 0.33.0 is

- Tag `v0.33.0` → commit **`4ba787ec9`** (`[CI][Bugfix] Pin the ColUltraVBERT test checkpoint…`, commit dated
  2026-10-31 07:15 UTC, tagged 2026-11-01 09:21 UTC; rc0-rc3 tags precede it). As of 2026-11-02 there is **no
  GitHub release page and no PyPI 0.33.0** (PyPI still stops at 0.32.0) — but the wheels are built and served
  (last-modified 2026-11-01 09:45 UTC).
- **Our image pin is an ancestor of the release.** `e4a5b5537` (the 2026-10-25 `nightly-rocm100` pin) → `4ba787ec9`
  is **107 commits, status `ahead`**; `4ba787ec9` → `main` (`5689c2790`, 2026-11-02) is another **487 commits,
  `diverged`**. So the release line is the cheap step and `main` is the expensive one, not a superset ordering.
- The 107 commits are mostly KV-offload, Rust frontend, CI and new-model work. Only five touch ground we stand
  on (below).

## Wheels: the native box can follow without docker

`wheels.vllm.ai/rocm/<pin>/` carries **two flavors per pin**: `rocm72/` (the ROCm 7.2-era build our venv uses)
and `rocm100/` (the pip-SDK ROCm 10 build). For `4ba787ec9`:

- `vllm-0.33.0+rocm72-cp312…manylinux_2_39_x86_64.whl`, 325.7 MB, HTTP 200 verified.
- **torch and triton are the same versions we already run**: `torch-2.13.0+git733fca1`,
  `triton-3.8.0+git669b31ac`. So a pin bump is a vLLM-only change on this box — no torch, no SDK, no driver.
- Two dependency deltas in the wheel: it **needs `orjson`** (new hard dep; installed) and it pins
  **`amd-aiter==0.1.26`** while our venv has `0.1.24.post1`. AITER is off in our launch (`AITER=0`), so the test
  venv keeps `0.1.24.post1` and measures the same thing we serve; `0.1.26` is in the same index and goes in when
  we adopt.

**Two stale things this uncovered in our own docs** (fixed in the same commit as this note):

- The flavor is renamed upstream — `+rocm723` → **`+rocm72`** — and **per-commit indexes expire**: the URL
  `README.md` and `notes/native-4xr9700.md` documented (`…/rocm/5df63a72…/rocm723/`) now returns 404, so a fresh
  box could not be built from the recipe as written. `setup.py`'s `--check` hint was worse: it printed
  `download.pytorch.org/whl/rocm6.4`.
- Per-commit indexes exist for `main` commits too (`24324ef0f`, `45fd60d4d` → 200) but not for every one
  (`5689c2790`, `d492ee786` → 404): builds appear with CI and disappear. Track the release line, not a moving sha.

## The test venv, and how to re-run it

`~/venv-r9700-033` is a `cp -a` of `~/venv-r9700` (11 GB, shebangs in `bin/` re-pointed) with one swap — the
torch/triton/aiter/openmpi and both editable installs (`-e .`, `-e serve/plugin`) are the ones the box already
runs, and `libr9k.so` is untouched (host hipcc, ROCm 10.1):

```
~/venv-r9700-033/bin/python -m pip install --no-deps vllm==0.33.0+rocm72 \
    --extra-index-url https://wheels.vllm.ai/rocm/4ba787ec9959e566d3b587a796dff07673324da5/rocm72/
~/venv-r9700-033/bin/python -m pip install orjson
```

For serving, keep the caches off the production ones: `VENV=~/venv-r9700-033 VLLM_CACHE_ROOT=~/vllm-native-cache-033
TRITON_CACHE_DIR=~/vllm-native-triton-033 bash serve/native-flashnext.sh` (1.5 GB + 240 MB of cache get rebuilt,
and the 110.6 GiB checkpoint reads again — that is what a pin bump actually costs in wall time).

## Import-surface audit (static, before running anything)

Every `from vllm…` in `r9700_vllm/` — **63 modules, 91 symbols** — checked against the `v0.33.0` tree and against
`main` (blobless clone, `git show` per module):

| ref | result |
|---|---|
| `v0.33.0` | **all present.** Two apparent misses (`vllm.distributed.tensor_model_parallel_all_reduce`, `vllm.models.glm5next`) are lazy re-exports, not removals |
| `main` | **two modules gone**: `vllm.models.qwen4_exp.amd.indexer_qsa` and `vllm.models.qwen4_exp.common.qsa_cache`, renamed by [`#64110`](https://github.com/vllm-project/vllm/pull/64110) `d492ee786` (2026-10-30, "Remove FlashAttention dependency and standardize QSA naming") to `amd/qsa_indexer.py` and `common/qsa.py`. Our imports of `apply_qsa_rope` (`attn/indexer_glue.py`, `models/qwen4_exp.py`) and `QSAForwardMetadata` break there. `#64110` is **not** in 0.33.0 |

`vllm/plugins/__init__.py` and `vllm/v1/metrics/stats.py` have **no diff** between our pin and 0.33.0: the
Monitor/About plugin (endpoint + stat-logger entry points, `FinishedRequestStats`) is untouched by this bump.

## What 0.33.0 changes in our surface (5 items)

1. **`a55937f25` `#60084` — 32-bit buffer addressing restored for `TRITON_ATTN` KV views.** New
   `set_triton_ptr_range()` in `v1/attention/backends/utils.py`, called from `TritonAttentionImpl` on ROCm for the
   fp8 KV views. Triton's AMD backend only uses 32-bit addressing for pointers under 2 GiB, and a view into a big
   storage loses it; the fix sets `ptr_range` to the distance from the view to the **end of the region**. We run
   fp8 KV through our `R9kTriton3DImpl` subclass at 262k — this is the one item in 0.33.0 that could move our
   prefill/decode numbers, and the one that could bite: it asserts kernels never read past the end of the region.
   Our subclass overrides `forward`, so the measurement has to be ours, not stock's.
2. **`00b6c4c56` `#59774` — Octave KV (`ROCM_OCTAVE`), a native 3-bit KV cache backend.** +1,180 lines of
   `csrc/attention`, a new backend in the ROCm priority list, and it is what grew `models/qwen4_exp/amd/qsa.py`
   by 212 lines in this range. **Gated to RDNA3 and CDNA3** (`get_rdna_version() == 3 or get_cdna_version() == 3`,
   "decoder attention only") — so it is *not* available on gfx1201, and it will not show up as a KV-density option
   for the 262k context until RDNA4 is validated. It is now also in the backend priority list, so a mis-gated
   selection would be our problem, not stock's.
3. **`rocm.py` +45**: `get_rdna_version()` (4 / 3 / 0) is new — worth using in `r9700_vllm` instead of our own arch
   probes; `_maybe_default_dsv4_mxfp4_indexer` is CDNA4-only and DeepSeek-only, no effect on Qwen4Exp/GLM5next.
4. **`platforms/__init__.py`**: a platform plugin that raises during detection is now logged at **warning** level
   when it is out-of-tree (was debug). Our `r9700_vllm.platform:detect` must not throw — on a bad card that changes
   a silent line into the first thing a reader sees. It did not fire in the 0.33.0 gate below (detection succeeded).
5. **`0f6b26191` `#61879` added `FusedMoEExperts.post_gemm2_hook`** — "called with the unweighted GEMM2 rows and
   the output to weight and reduce into, **replacing the experts' own top-k reduction**". That is upstream building
   the extension point our `moe/fold.py` monkeypatches around (output alias + shared-expert fold, both OFF by
   default since 2026-10-07 at ~0.1 ms of a 16.7 ms step). Retiring those two patches onto the hook is now
   possible on a supported API; not done here, and it should be a separate change with its own A/B.

## Gates on this box (jiulu-9700, 2026-11-02, one card visible, launch up)

| gate | 0.32.0 (production venv) | 0.33.0 (`~/venv-r9700-033`) |
|---|---|---|
| `tests/test_integration.py` | ALL OK, 34 checks | **ALL OK, 34 checks** — no plugin change needed |
| `tests/test_ple_int6.py` | ALL OK | **ALL OK** (head_dim 160/256, device and UVA rows, rel 1.67e-03) |
| `tests/test_cache_shapes.py` | ALL OK | **ALL OK** — eager 900 shapes + graph capture + 12 replays equal |
| `tests/test_gemm_fp8.py` | not re-run today | **not run to completion: OOM**, see below |
| `tests/test_moe_mxfp4.py` | ALL OK (note) | not run |
| the three serving configs, `sanity_stress`, `soak`, GSM8K/HumanEval, probe32 | — | **not run** |

The OOM is capacity, not 0.33.0: the standing 262k/32-stream launch holds 0.97 of every card —
`rocm-smi` reads 32.6 of 34.2 GB used on all four, so a gate has ~1.5 GiB to work in and
`test_gemm_fp8`'s `x.float() @ w.float().T` reference asks for 1.19 GiB on its own. **The serving gate needs a
window on the cards**: stop the launch (`STOP=1 bash serve/native-flashnext.sh`), run the kernel gates that need
>1.5 GiB, then the 32k and 262k launches.

Nothing was changed on the production venv, the production caches, or the running launch, and `compat/gate.py`'s
TESTED lists were deliberately **not** given `0.33.0` — that line means "serving gate passed on it", and the
serving gate has not run. Two patches (`mtp_allowlist`, `glm5next_rdna4`) warn on both 0.32.0 and 0.33.0 exactly
as they did before.

## Hardware note, because it decides the schedule

[incident-2026-11-02/README.md](incident-2026-11-02/README.md) records seven self-resets in one night on this box,
three of them idle, cause still open (BMC/PSU/watchdog material). A pin bump's remaining steps are exactly the kind
that expose a platform problem: a 20-minute cold load, a 262k soak, and a full-context retrieval ladder. **Do the
serving gate after the box stops resetting**, or do it knowingly, with the BMC log running.

## Upstream to watch (not in 0.33.0)

| what | state 2026-11-02 | why it matters here |
|---|---|---|
| `#62799` Triton skinny GEMM for 6-128 rows on RDNA4 | open, updated 2026-10-30 | the only upstream item in this whole sweep with a two-digit R9700 win in it: the author measures **+28-29% output throughput with DSpark at conc 1-16 on one R9700** (gfx1201), 0.75-0.81x GEMM time over 36 shapes, because BF16 GEMMs with 6-128 rows fall from `wvSplitK` (600 GB/s) to hipBLASLt (420-470 GB/s). Watch for merge, or carry it |
| `#57861` avoid RowWise FP8 scaled MM on RDNA4 | open, updated 2026-11-01 | vLLM 0.28+ picks RowWise on gfx1200/1201 where ChannelWise decodes faster |
| `#62718` split `skinny_gemms.cu` per kernel family | open, updated 2026-11-02 | ROCm build time (~58 s serialized); matters if we start source-building |
| `#64874` `Workspace is locked` assert in `rocm_fp8_paged_mqa_logits` (GLM-5.3-Flash, high decode concurrency) | open, filed 2026-10-31 | our `models/glm5next.py:270` reserves the decode logits through `current_workspace_manager()` — read this one before moving any GLM launch onto a new pin |
| `#63735` RDNA launch config for `kernel_paged_attention_2d`, `#63054` / `#64614` 64-bit row offsets in the Triton and RDNA-hybrid W4A16 GEMMs | merged, **main only** | RDNA4 fixes that will not reach a wheel until 0.34 |

## Verdict and plan

Follow **v0.33.0**; do not follow `main`. Steps, in order:

1. Fix the wheel URLs/flavor (done with this note).
2. Kernel gates that need VRAM, plus the 32k and 262k launches on `~/venv-r9700-033`, with separate caches —
   measure the two things 0.33.0 could change: `set_triton_ptr_range` on our fp8 KV path, and the KV-cache token
   count (it moved 203k → 157k on a different pin for a different reason once already).
3. `sanity_stress` at NSEQ+1 / 1.5x / 2x, `soak` 1500 s, GSM8K + HumanEval, `bench/probe32.sh` bands — the
   standing release-gate list.
4. Only then: `0.33.0` into `compat/gate.py`'s TESTED lists, `amd-aiter 0.1.26`, the image `BASE` (the nightly
   line has no 0.33-era ROCm 10 image pinned yet, so this is a separate question from the venv), README/notes pin
   text, and the release note.
5. Separately: `post_gemm2_hook` vs `moe/fold.py`, and `get_rdna_version()` replacing our arch probes. Both are
   plugin changes with their own A/B, not part of a pin bump.
