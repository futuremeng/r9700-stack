# vLLM 0.33.0 on this stack: two evaluations, one release

This file carries **two independent evaluations of the same release**, on two machines and two paths. They were
written without knowing about each other, and both are kept whole; the file was a merge conflict by construction
(2026-11-08, merging `upstream/master` into the fork).

- **Part 1 -- VM100, container path (upstream, 2026-11-07/08).** All four serving configurations on
  `vllm/vllm-openai-rocm:v0.33.0-rocm100` at the v0.4.2 numbers, no vLLM-0.33-specific breakage, the pin moves,
  and it shipped as v0.4.3.
- **Part 2 -- jiulu-9700, native path (this fork, 2026-11-02).** The 107-commit import/wheel audit, the
  per-gate pairs against the standing 262k recipe, and the verdict reached that morning: level, so not adopted,
  on a box that was resetting four times a day.

**Status as of 2026-11-08: the pin moves -- Part 1 decides it.** Part 2's "not adopted" is superseded: it was a
judgement about this box's launch line ("a release that changes nothing measurable is not worth a second serving
line"), not about the platform, and Part 1 answers the question Part 2 left open -- the BetterBench row Part 2
never got, and the eight-card DeepSeek validation this box cannot run. Part 2's import audit, wheel-flavour
correction (`+rocm723` -> `+rocm72`) and its hardware caveat still stand and are the part worth keeping here.

**Owed on this box, and open until it runs** (jiulu, 2026-11-08: "follow it, but compare-test first, when the box
is idle"). The 0.33 venv and caches are already in place, so it costs no build time, only two launches of the
standing 262k recipe: see "On this box: the comparison to run" at the end of this file. Two patches this fork
carries and Part 1 never exercised -- `qsa_kvfp8` (fp8 main KV for Qwen4Exp QSA) and `moe_h2d_batch` (batched
expert placement) -- are deliberately still listed in `compat/gate.py` as tested on 0.32/`e4a5b5537` only, so a
0.33 launch here warns until that gate passes. That warning, not a documentation change, is the marker.

---

## Part 1 -- the plugin on vLLM v0.33.0 (VM100, container path)

Brian (2026-11-07): "Take a look at vllm .33 and see how it works in our stack." The production pin is the 2026-10-25 nightly
`e4a5b553` (0.32.1rc1.dev466, torch 2.13, ROCm 10.0); the current release v0.4.2 runs on it. Same method as
[vllm-0.32.md](vllm-0.32.md) and [vllm-upgrade-2026-10.md](vllm-upgrade-2026-10.md): build the plugin image on the new base,
regenerate the VM100 overlay, run the integration gate and the kernel gates, then the four serving configurations against the
v0.4.2 gate numbers ([release-v0.4.2.md](release-v0.4.2.md)).

### What "0.33" is on ROCm

- **For the first time there is an exact release on ROCm 10:** `vllm/vllm-openai-rocm:v0.33.0-rocm100` (2026-11-02; also tagged
  `v0.33.0`, the ROCm 10 image is now the default ROCm image, ROCm 7.2 moved to `-rocm72`, #58761). Version string `0.33.0`.
- **It is very close to our pin.** The GitHub release (title v0.33.0, tag `v0.33.0rc3` = `v0.33.0` = commit `4ba787ec9`; there is
  no commit 3a4d82f5) branched at `4135590ffe` on 2026-10-26, **one day and 95 main commits after `e4a5b553`**, and took 12
  cherry-picks (HiSparse, Humming, CI, a gfx950 GEMM guard, MTP in the Transformers backend). The "927 commits" of the release
  notes are counted from v0.32.0, most of which our pin already carries.
- **Same runtime as the pin:** torch `2.13.0+rocm10.0.0` (the notes' "PyTorch 2.14" is the CUDA build; the ROCm image keeps
  2.13), triton 3.8.0, aiter 0.1.26, flash_attn 2.8.3, torchvision 0.28.0, `rocm-sdk-core 10.0.0` from the same TheRock build
  (`the_rock_commit 16adc4d8`, run 32424980240), HIP 7.15.26333; `libamdhip64.so.7` and `librccl.so.1` are byte-identical to
  the pin's (md5 35e93d24 / a3963038). So the overlay's hostcall-free RCCL needs no rebuild.
- The newest ROCm 10 nightly, `nightly-rocm100-1648fefab9` (2026-11-07, `0.33.0rc1.dev992+g1648fefab`): same torch / triton / SDK /
  HIP / RCCL, **aiter 0.1.28.post1**, ~1,060 commits past the branch point.

### Release-note delta against what the plugin hooks

The code between the pin and the release is 95 main commits plus the 12 picks (`gh api .../compare/e4a5b553...4ba787ec9`; the
release notes describe all of 0.32.0 -> 0.33.0). Diffing the Python trees of the two images (`~/v033/src-old`, `src-new`, 164
files changed) against the 72 vLLM modules the plugin imports, the ones that touch our hooks:

| area | change (PR) | what it means for us |
|---|---|---|
| ROCm platform | `ROCM_OCTAVE` 3-4 bit KV backend appended to the non-MLA priority list after `TRITON_ATTN` / `TURBOQUANT` (#59774); `_maybe_default_dsv4_mxfp4_indexer` (MXFP4 indexer K cache for DeepSeek-V4, #63669) | gated on `get_cdna_version() == 4` and aiter on: not taken on gfx1201. Our `R9700Platform` (`get_device_communicator_cls`, `get_mla_prefill_backend_cls`) is unchanged and still applies. |
| platform plugins | out-of-tree detection failures now log a warning instead of debug (#63601: PIN_MEMORY eager, no platform resolution during `import vllm`) | nothing to change; our `detect()` does not raise. |
| attention | `TRITON_ATTN` sets `ptr_range()` on ROCm KV views for 32-bit buffer addressing (#60084); `attention.py` no longer re-quantizes an fp8 query; `customize_spec` runs under `set_current_vllm_config` | our CUSTOM backend pins `supports_quant_query_input = False` (0.32 fix 4) and keeps its own kernels: not affected. The integration gate's draft-metadata and kernel-block-remap checks still hold. |
| Flash-Next (Qwen4-Exp) QSA | Octave cache dtypes for the QSA main cache, a `customize_spec` that passes non-Octave specs through, `forward_qsa` signature unchanged | our `forward_qsa` rebinding (attn/qsa.py) still applies; BF16 cache as before. |
| DeepSeek-V4.1 | Engram GPU tables read from peers' HBM (#62091) and meta-init for Engram offload (#63068), both only for GPU tables with DP > 1 / meta builds; FP8 shared-expert FSE check (#63099); `krdy` ordering in index-source launches (#63576, AMD MonoKernel) | we run `cpu_offload` Engram at TP8 + EP: the CPU-offload path is unchanged apart from a `weight.is_meta` guard. |
| GLM-5.3 | FP8 indexer WK pairs kept across `load_weights` calls (#59223) | load path only. |
| speculative decoding | speculator `max_model_len` follows auto-fit (#60513); adaptive-verification cost profiling skipped in the graph memory-profiling pass (#63431); lookahead block hashes for EAGLE/MTP prefix caching on by default (#61589) | our launches run prefix caching off; no draft-metadata API change. |
| MoE | `post_gemm2_hook` / `route_major_moe_output` added to the modular kernel; hash-routing kwargs refactored | additive; our MoE methods do not use them. |
| sparse MLA | `idx_mapping` (HiSparse) variant of the sparse-utils kernels | additive. |

Not in v0.33.0 (they are on main after the branch point and in the 11-07 nightly): AITER 0.1.28, QSA top-k buffer shared across
layers (#65619), the Qwen4-Exp HyperConnection up/gate fusion (#64898), PLE shard copy via `copy_weight_` (#65692), Engram gate
sharding (#63367), the DeepSeek-V4.1 mHC gate overlap (#59802), DSpark sliding-window MLA draft layers (#63795).

### Image and overlay

- `r9700/vllm:dev-v0330` from `docker/Dockerfile` with `BASE=vllm/vllm-openai-rocm:v0.33.0-rocm100`: **683 s, 53.8 GB**.
  **exllamav3 v1.6.6 builds unchanged** (same torch), `exllamav3_ext ok`; libr9k builds unchanged; the plugin installs as
  `r9700-vllm 0.4.2` and registers under `vllm.general_plugins`.
- Overlay: the hostcall-free RCCL (`~/rccl10/.../librccl.so.1.0`) is reused (same SDK build, byte-identical `librccl.so.1` and HIP
  runtime). `patch-hostcall.sh` -> `~/p2p-patched-v0330`: `_C_stable_libtorch.abi3.so` 0 hostcall kernels after the patch;
  `_rocm_C.abi3.so` 782 still requesting hostcall, all gfx9-only (`paged_attention_ll4mi_QKV_mfma4` 768, `wvSplitKrc` 14), as on
  the pin.

### Gates (card 4 of VM100, `~/gates033.sh`, tree = branch v033)

The gates032 list plus the seven single-card tests v0.4 added (dsv4_decode, dsv4_glue, exl3_gemv, exl3_linear, exl3_moe,
exl3_moe_r9k, mhc_r9k), `BENCH=0`. `gates033.sh` now prints a per-test verdict from the full log: gates4.sh keeps each test's
last 25 lines, which hid the first FAIL below.

- First run (`~/v033/gates-v0330.log`): `tests/test_integration.py` **ALL OK** (24 checks, the int6-PLE scope section skips as on
  the pin); 40 of 42 tests pass; `test_dsv4_glue.py` SOME FAILED; `test_attn_r4d.py` no verdict.
- After the fixes below (`~/v033/gates-v0330-run2.log`): **41 of 42 pass**. `test_attn_r4d.py` needs `kernels/r4d.so`, which
  the gate image does not build (pre-existing, not a 0.33 matter).

#### Fixes (branch v033): what broke, why, and the fix

1. **`test_dsv4_glue.py`: "rotated input K=5120 / K=1280 M=1 vs exl3_gemm A_had" (1 element, 1 f16 ulp).** Identical on the
   production image (not a 0.33 matter). An op-for-op f32 emulation of exllamav3 v1.6.6's `had_hf_r_128_inner` matches our
   kernel, not exllamav3's output, and every mismatch is a near-tie (the exact value just past the f16 midpoint): exllamav3 is
   compiled with HIP's default `-ffp-contract=fast`, where `__floats2half2_rn(h * r_scale)` becomes a mixed-precision multiply
   with an f16 result (one rounding); libr9k builds with `-ffp-contract=off` and rounds twice (f32, then f16). **Fix**
   (`kernels/r9k_dsv4_glue.hip` `f2h_mul`, the rotated store): the f32 x f32 product is exact in f64 and rounded once to f16.
   Now 0 ulp at M = 1, 2, 6, 8 against both of exllamav3's rotations; the test asserts bit-equality at every M (it allowed
   2 ulp above M = 1). The suh pre-pass has no serving caller (package D's interface), so no served output changes.
2. **`test_exl3_linear.py` "[graph-cold] cold capture at M=32 failed: hipErrorStreamCaptureInvalidated".** An informational
   negative control (the test reports "0 failed"; same on the production image). The first error, from a rocgdb breakpoint on
   `hipMalloc` and `AMD_LOG_LEVEL=3`: `exl3_gemm -> exl3_gemv_try_launch -> exl3_gemv_graph_param_block(device,
   allow_alloc=true) -> hipMalloc(4128) = hipErrorStreamCaptureUnsupported`. exllamav3 1.6.6 sends 1..8 rows to its multi-row
   GEMV (tag 92); only 9+ rows reach `exl3_gemv_try_launch`, whose first line pre-warms the graph-GEMV parameter block on the
   assumption that "no exllamav3 `Graph`" means "not capturing", which a torch stream capture breaks. The failed launch and
   `EndCapture` are the reported symptom. With the block pre-warmed by an eager 16-row call, the next blocker is the cooperative
   GEMM autotuner's `hipStreamSynchronize` on the capture stream on a shape's first call. **Not a plugin defect:** `apply()`
   never sends more than 8 rows to one `exl3_gemm` and `_warm` runs every row count eagerly before capture; the test's
   docstring blamed the autotuner and now states the traced chain.
3. **`kernels/build.sh`: "MISSING EXPORT: r9k_ar_ipc_open"** in the image build on the nightly 1648 (the symbol was present).
   `set -o pipefail` with `nm -D | grep -q`: grep exits on its first match, nm takes SIGPIPE (141), the pipeline fails. Latent
   since the export check was written. **Fix:** the export list is read once and each name grepped against it.

4. **`serve/ds41-exl3.sh`: "drafter self-test FAILED (Connection refused)" on a still-loading server** (found in serving,
   6cbc5a2). The WARM and SELFTEST waits were a fixed 120 x 10 s; a cold-page-cache DeepSeek launch takes 35-45 minutes.
   **Fix:** `wait_ready()` polls `/v1/models` while the container lives (`READY_WAIT`, default 3600 s), exits 2 if it goes away.
5. **`compat/gate.py`: twelve "untested on vLLM 0.33.0" warnings on the Flash-Next launch** (f820bc2): `mtp_allowlist` and
   `glm5next_rdna4`, six each (one per process that imports the plugin). A release image reports the plain
   version string (`0.33.0`, no `+g<sha>`), so the TESTED tuples need `"0.33.0"` itself; both are now listed (validated by
   Flash-Next and GLM here). A nightly reports `+g<sha>`: the 1648 run needs `"1648fefab"` added after its gate. The DeepSeek
   launch also warned for `moe_shared_fold` (tested `e97573215`); its list stays until DeepSeek passes on 0.33.

### Serving (VM100, `~/gate033.sh CONFIG`, image `r9700/vllm:dev-v0330`, overlay on, REPO = `~/r9700-build/repo-033` = branch v033)

The v0.4.2 gate for each configuration, one at a time under the lock (210 W cards; GLM and DeepSeek at the -500 MHz offset).
v0.4.2's numbers are from `notes/release-v0.4.2.md` (same scripts, the pinned image). No configuration needed a fix.

**Qwen3.8-Flash-Next TP4** (cards 0-3; `~/v033/v0330-fn4.*`)

| | v0.33.0 | v0.4.2 (e4a5b553) |
|---|---:|---:|
| probe decode mean (code / prose / json / math) | **225.4** (224.1 / 161.9 / 264.2 / 251.3) | 222.0 (218 / 160 / 262 / 249) |
| tokens per step (MTP) | 3.11 | 3.11 |
| 8 / 16 streams | 476 / 938 | 466 / 909 |
| prefill 8k | 6,953 tok/s | 6,556 |
| BetterBench combined (update p99) | **185.7** (15.0 ms) | 184.8 (15.2 ms) |
| strict sanity | 0 bad of 2,460 | 0 bad of 2,460 |
| soak (16 clients, 200-30k, 900 s) | 454 requests, 0 errors, 6,668 prompt tok/s | 424, 0 errors, 6,483 |
| KV cache | 472,439 tokens | 479,076 |

**Qwen3.8-27B-NVFP4 TP2** (cards 0,1; `~/v033/v0330-27b.*`)

| | v0.33.0 | v0.4.2 |
|---|---:|---:|
| probe decode mean (code / prose / json / math) | **218.3** (236.2 / 107.6 / 275.3 / 253.9) | 220.4 (236 / 111 / 281 / 254) |
| tokens per step (DFlash2 drafter) | 4.49 | 4.64 |
| 8 / 16 streams | 532 / 646 | 512 / 607 |
| prefill 8k | 3,968 tok/s | 3,874 |
| BetterBench combined (update p99) | **197.9** (24.3 ms) | 195.5 (24.7 ms) |
| strict sanity | 0 bad of 2,460 | 0 bad of 2,460 |
| soak | 251 requests in 948 s, 0 errors, 3,082 prompt tok/s | 199 in 961 s, 0 errors, 3,050 |

The 0.32 move's draft-metadata fix (`R9kTriton3DMetadataBuilder`) still holds: the drafter accepts (4.49 a step, not 1.0).

**GLM-5.3-Flash TP8** (32k production recipe; `~/v033/v0330-glm.*`)

| | v0.33.0 | v0.4.2 |
|---|---:|---:|
| decode-only median (512 tokens x 3) | **96.6 tok/s** | 95.1 |
| four streams, aggregate (first / again) | 178 / **229.5** | 176 / 221 |
| 12.7k needle | correct, ttft 5.0 s, prefill 2,534 tok/s, 124.2 tok/s at depth, 82 % acceptance | correct, 118.7, 83 % |
| strict sanity (thinking on) | 0 bad of 780 | 0 bad of 780 |
| soak | 183 requests in 1,019 s, 0 errors | 168 in 978 s, 0 errors |
| KV cache at 0.94 | **155,437 tokens** (2.5 GiB available) | 203,329 (2.65 GiB) |

The GLM KV cache is the one regression. The KV layout is identical (groups 32768 x 3 / 1280 / 8, lcm block 163,840 tokens, the
same mamba padding) and the CUDA-graph estimate is unchanged (0.94 = 0.9229 without it on both); the profile run's peak is
~0.15 GiB higher on 0.33, and with blocks this coarse 0.15 GiB costs a quarter of the tokens. A candidate is #57926 (the sampler
is profiled with the parameters the warm-up uses). Not a plugin matter; `KVMEM=` pins the budget explicitly (see the open items).

**DeepSeek-V4.1-Flash TP8 (launcher defaults, SELFTEST on; after the host power cycle, 2026-11-08 03:41-04:22 UTC; `~/v033/v0330-ds41.*`):
at the v0.4.2 numbers, no fix needed.** KV cache 387,069 tokens (same), model load from a cold page cache, self-test prose 1.66 /
code 3.59. TTFT 1.25 s on 1.5k (1,236 tok/s), 3.36 s on 12.3k (3,669 tok/s); the 4 x 12k burst leaves **0.74 GiB** free on the
worst card (v0.4.2 0.52). spectune prose x1 **104.1-104.7** tok/s (MAL 1.75-1.77; v0.4.2 103.5), x4 202-231 aggregate (199-221);
code x1 **210.8-210.9** (4.24-4.27; 214.5), x4 **422-424** (422-423); GSM8K-style 160.6 / 190-230. divergence 24 / 24 and
22 / 24 (23 / 24, 24 / 24, 24 / 24). Needles 3 / 3 at 8k, 16k, 28k. Strict sanity 0 bad of 25 at 5 streams and 0 of 80 at 8.
Soak (8 clients, 200-28k, 928 s): 191 requests, 0 errors, median prompt 14.8k, 2,989 prompt tok/s, 0.74 GiB free throughout.
GSM8K 200: **196 / 200 = 98.0 %** (194 on the pin). 0 server errors; the 16 "untested patch" warnings were `moe_shared_fold`
(its TESTED tuple lacked e4a5b5537 and 0.33.0; added in bcf3622). The earlier two launches, before the power cycle (`~/v033/v0330-ds41-try1-cold.*`, `~/v033/v0330-ds41.*`):

- The first launch read the 196 GiB pack and the Engram tables from a cold page cache (the GLM gate had just run): ~21 minutes
  of Engram / pre-weight reads, then 45 shards at ~24 s each. `serve/ds41-exl3.sh`'s WARM / SELFTEST waits were a fixed 20
  minutes, so the launcher reported "drafter self-test FAILED (Connection refused)" against a server that was still loading, and
  the gate wrapper's own 25-minute wait then removed the container. **Fix (6cbc5a2):** `wait_ready()` polls `/v1/models` for as long
  as the container lives (`READY_WAIT`, default 3600 s) and exits 2 if it goes away.
- The relaunch exposed the real problem: **the .100 host's CPU was parked by the SCMP power-limit trip** (the 120 V feed; see the
  host notes): a 20M-iteration Python loop takes 62-66 s on VM100, the host and the mgmt VM (~1.5 s normally); host cores at
  400 MHz; the libr9k compile took 3.6 minutes at 20:20 and more than 12 at 23:45. GLM's decode, needle and sanity numbers
  (22:26-22:38) are normal and the build right after the GLM gate (22:55) was already slow, so the trip happened in the GLM gate's
  eight-card soak (22:38-22:55; 16 clients, prompts to 30k tokens, at the -500 MHz offset). The GLM soak counts as a stability
  pass (183 requests, 0 errors), not as a throughput number. The DeepSeek launch was stopped and nothing further was measured;
  the trip clears only with a power cycle of the host.

Nothing in the static delta touches the DeepSeek path we serve (Engram's changes are for GPU tables with DP > 1; the MXFP4 indexer
default is CDNA4-only; the MonoKernel / MegaMoE work is gfx94x / gfx950), and the integration gate and the DeepSeek-side unit
tests (dsv4_decode, dsv4_glue, exl3_gemv / linear / moe / moe_r9k, mhc) pass on the image, but the protocol's rule stands: a
configuration is validated only by the eight-card server run (validate-42.sh scope, self-test on).

### Newest nightly (`nightly-rocm100-1648fefab9`, 2026-11-07)

`r9700/vllm:dev-1648` is built (exllamav3 1.6.6 and libr9k build unchanged; **36.8 GB**, a slimmer base than the release's
53.8; the first build attempt hit the `build.sh` export-check race, fixed above), overlay pair `~/p2p-patched-1648` (38 CCOB
bundles in `_rocm_C`, 768 gfx9-only hostcall kernels, `_C_stable_libtorch` 0). Gates: not run. The DeepSeek launch (2026-11-08 04:23-04:43 UTC, `~/v033/1648-ds41.*`) **dies at engine start**: `module 'vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router' has no attribute '_get_padding_mask'` (our router hook's target moved in the nightly line). Informational: the recommendation is the release image, and the nightly line's next release is the next candidate

### Verdict

**All four configurations serve at the v0.4.2 numbers on v0.33.0-rocm100 with no serving fix.**

- **v0.33.0-rocm100 is the first exact vLLM release on the platform we target**, and it is a small step from our pin: 95 main
  commits + 12 picks, the same torch / triton / aiter / SDK build / HIP runtime / RCCL. The image builds from `docker/Dockerfile`
  with only `BASE` changed, exllamav3 v1.6.6 builds unchanged (EXL3_REF stays), the overlay's RCCL is reused.
- The integration gate and 41 of 42 kernel gates pass (the 42nd needs libr4d's `r4d.so`, as on the pin). Three plugin-side fixes
  came out of the gates (build-script race, a 1-ulp rotation rounding in a non-serving pre-pass, a mis-attributed negative
  control), one out of serving (the launcher's fixed readiness wait), and the compat allowlist gained `0.33.0`. **None of the four fixes is caused by 0.33**; all reproduce on the
  pinned image. **No vLLM-0.33-specific breakage was found** -- unlike the 0.32 move (five fixes).
- **Flash-Next TP4, 27B TP2 and GLM-5.3-Flash TP8 serve at or slightly above the v0.4.2 numbers** (BetterBench 185.7 / 197.9;
  GLM decode 96.6) with 0 bad answers and 0 soak errors. The one regression: **GLM's KV cache at 0.94 shrinks 203k -> 155k tokens**
  (a ~0.15 GiB larger profile peak against coarse 163,840-token blocks).
- **DeepSeek-V4.1-Flash TP8 -- the configuration v0.4.x is about -- passes its full validation** (prose 104 / code 211 single
  stream, 422 at four streams, GSM8K 98 %, soak 0 errors, self-test ok) after the host power cycle, with 0.2 GiB more headroom on
  the worst card than the pin.

### Recommendation: move the pin to `v0.33.0-rocm100` (DeepSeek passed 2026-11-08)

Reasons: (1) an exact release on ROCm 10 is the pin we want (reproducible, named, documented, no nightly churn), and nothing in
the four configurations argues against it; (2) the eight-card DeepSeek validation that the protocol requires before a pin moves
has now run and is level with v0.4.2; (3) the GLM KV regression is to be measured with `KVMEM=2.65` (the pin's budget) in the
release gate, so the 32k recipe either keeps its ~203k tokens or documents the new figure. Do not chase the nightly line: it is ~1,060 commits
further on and touches five of our hook sites; take the next release instead.

Steps after the host's power cycle: `ssh devops@192.168.0.123 'bash ~/gate033.sh ds41'` (launcher defaults, self-test, the
validate-42 scope via `~/v033/validate-033.sh`; ~1.5 h with a warm page cache -- run it first, or pre-read the pack); if it is
level with v0.4.2 (prose 103.5 / code 214.5 single stream, 199-221 / 422 at four, divergence 23-24 / 24, needles 3 / 3, GSM8K
97.0 %), set `docker/Dockerfile` `ARG BASE=vllm/vllm-openai-rocm:v0.33.0-rocm100`, merge v033 and gate the release as usual.

### Open items

- The nightly 1648 DeepSeek pass: running 2026-11-08 04:23 UTC (`~/v033/gate033-ds41-1648.out`); informational only, the
  recommendation does not depend on it.
- **GLM-5.3-Flash KV cache, for the pin move: 155,437 tokens on 0.33 vs 203,329 on the pin** (UTIL 0.94, 32k recipe). Same
  KV layout (groups 32768 x 3 / 1280 / 8, mamba page padded 16.79 %, lcm block 163,840 tokens) and the same CUDA-graph estimate;
  "Available KV cache memory" is 2.5 GiB a rank instead of 2.65 because the profile run peaks ~0.15 GiB higher (candidate:
  #57926, the sampler profiled with the warm-up's parameters), and with blocks this coarse 0.15 GiB is a quarter of the tokens.
  **Knob: `KVMEM=2.65`** in the GLM 32k launch (the pin's measured budget, a fixed GiB a rank that bypasses the profile
  estimate; equivalently `UTIL=0.945`, +0.16 GiB of a 31.86 GiB card). I would set `KVMEM=2.65` and confirm >= 0.5 GiB free on
  the worst card through the four-stream burst and the 15-minute soak before writing it into the recipe; if it does not hold,
  keep UTIL 0.94 and document 155k.
- The serving rotations round the rotated store twice where exllamav3 rounds once: `kernels/r9k_exl3_gemv.hip` (`rotate_*`, the
  `*= RS128` before `st_h4`) and `kernels/r9k_moe_exl3.hip` (the input pre-pass and the expert-tier rotations); 1 f16 ulp on
  near-ties (~1 in 15k rotated inputs). Moving them to `f2h_mul`-style single rounding makes them bit-identical to exllamav3's
  rotation but changes DeepSeek's outputs, so it needs a DeepSeek server acceptance run (protocol: spectune MAL, divergence,
  GSM8K) before it lands.

### Reproduce (VM100)

| what | where |
|---|---|
| images | `r9700/vllm:dev-v0330` (v0.33.0-rocm100 + plugin, 53.8 GB), `r9700/vllm:dev-1648` (nightly 1648fefab9, 36.8 GB) |
| overlay pairs | `~/p2p-patched-v0330`, `~/p2p-patched-1648`; RCCL `~/rccl10/.../build-nightly/librccl.so.1.0` (unchanged) |
| trees | `~/r9700-build/repo-033` (branch v033, served), `~/r9700-build/repo-033n` (nightly build tree), `~/r9700-build/repo-033t` (card-4 test tree); the old v0.3.3-candidate `repo-033` moved to `repo-033-old-v0.3.3rc` |
| build | `~/build-release.sh TAG LABEL` (full tag, e.g. `v0.33.0-rocm100 v0330`; `REPO=` the tree) -> `~/v033/build-release-LABEL.log` |
| gates | `~/gates033.sh` (IMG / REPO / LBL; verdict table at the end) -> `~/v033/gates-v0330.log`, `gates-v0330-run2.log`, `*.full.log`; one test with full output: `~/v033/one-test.sh IMG TEST [card]` |
| serving | `~/gate033.sh fn4\|27b\|glm\|ds41` (IMG / PATCHED / REPO / LBL, takes the lock, offset for 8-card) -> `~/v033/LBL-CONFIG.{gate,launch,server,bb,validate}.log`; DeepSeek validation `~/v033/validate-033.sh` |
| diagnosis | `~/v033/colddiag*.py`, `exl3lin-gdb.log`, `exl3lin-trace.log` (the cold-capture trace); `~/v033/rotdiag*.py` (the rotation rounding); `~/v033/src-old`, `src-new` (the two images' vLLM Python trees) |

---

## Part 2 -- what moved since the pin, and what following it costs (jiulu-9700, native path)

"vllm 有更新吗？我们要不要跟一下？" — checked 2026-11-02 against three things at once: the release line
(v0.33.0), the ROCm 10 nightly line our image pins, and `main`. What the sweep found: **v0.33.0 is 107 commits past
our pin, needs no plugin change, and its wheel is a drop-in for the native venv; `main` is not followable** (one
upstream rename breaks two of our imports). Measured against the standing 0.32 numbers, though, 0.33.0 came back
**level** on everything this box can gate — so the call was **not to move**. Decision and the re-measurement
command are under "Verdict and plan"; the rest of the note is the sweep itself, which is the part worth keeping.

### What 0.33.0 is

- Tag `v0.33.0` → commit **`4ba787ec9`** (`[CI][Bugfix] Pin the ColUltraVBERT test checkpoint…`, commit dated
  2026-10-31 07:15 UTC, tagged 2026-11-01 09:21 UTC; rc0-rc3 tags precede it). As of 2026-11-02 there is **no
  GitHub release page and no PyPI 0.33.0** (PyPI still stops at 0.32.0) — but the wheels are built and served
  (last-modified 2026-11-01 09:45 UTC).
- **Our image pin is an ancestor of the release.** `e4a5b5537` (the 2026-10-25 `nightly-rocm100` pin) → `4ba787ec9`
  is **107 commits, status `ahead`**; `4ba787ec9` → `main` (`5689c2790`, 2026-11-02) is another **487 commits,
  `diverged`**. So the release line is the cheap step and `main` is the expensive one, not a superset ordering.
- The 107 commits are mostly KV-offload, Rust frontend, CI and new-model work. Only five touch ground we stand
  on (below).

### Wheels: the native box can follow without docker

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

### The test venv, and how to re-run it

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

### Import-surface audit (static, before running anything)

Every `from vllm…` in `r9700_vllm/` — **63 modules, 91 symbols** — checked against the `v0.33.0` tree and against
`main` (blobless clone, `git show` per module):

| ref | result |
|---|---|
| `v0.33.0` | **all present.** Two apparent misses (`vllm.distributed.tensor_model_parallel_all_reduce`, `vllm.models.glm5next`) are lazy re-exports, not removals |
| `main` | **two modules gone**: `vllm.models.qwen4_exp.amd.indexer_qsa` and `vllm.models.qwen4_exp.common.qsa_cache`, renamed by [`#64110`](https://github.com/vllm-project/vllm/pull/64110) `d492ee786` (2026-10-30, "Remove FlashAttention dependency and standardize QSA naming") to `amd/qsa_indexer.py` and `common/qsa.py`. Our imports of `apply_qsa_rope` (`attn/indexer_glue.py`, `models/qwen4_exp.py`) and `QSAForwardMetadata` break there. `#64110` is **not** in 0.33.0 |

`vllm/plugins/__init__.py` and `vllm/v1/metrics/stats.py` have **no diff** between our pin and 0.33.0: the
Monitor/About plugin (endpoint + stat-logger entry points, `FinishedRequestStats`) is untouched by this bump.

### What 0.33.0 changes in our surface (5 items)

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

### Gates on this box (jiulu-9700, 2026-11-02, both venvs, paired wherever the box let us)

The small gates were run against the standing 0.32 launch (one card visible, ~1.5 GiB free); the serving gates ran
on a 0.33 launch with the cards to itself. Three boots went between them (see the hardware note), so where a number
has a pair, the pair is named.

| gate | 0.32.0 (production venv) | 0.33.0 (`~/venv-r9700-033`) |
|---|---|---|
| `tests/test_integration.py` | ALL OK, 34 checks | **ALL OK, 34 checks** — no plugin change needed |
| `tests/test_ple_int6.py` | ALL OK | **ALL OK** (head_dim 160/256, device and UVA rows, rel 1.67e-03) |
| `tests/test_cache_shapes.py` | ALL OK | **ALL OK** — eager 900 shapes + graph capture + 12 replays equal |
| `tests/test_gemm_fp8.py` | **ALL OK**: lm_head 124160x2560 M=1 fp8 522 µs (608 GB/s) vs bf16 torch 1,021 µs; fp8-block 8192x2560 M=4 15-17 µs | **ALL OK**: 523 µs / 1,021 µs and 17 µs. The first 0.33 sample read 30 µs (708 GB/s) on that line; re-runs interleaved 0.32/0.33/0.32/0.33 read 15, 17, 17, 17 µs — noise, recorded because it looked like a regression for about a minute |
| `tests/test_moe_mxfp4.py` | **ALL OK**: gate_up 202.3 µs (598 GB/s) at 139 touched experts, down 103.7 µs (584 GB/s) | **ALL OK**: 202.2 µs (598 GB/s), 103.9 µs (582 GB/s) |

The first OOM (`test_gemm_fp8` at 08:22) was capacity, not the version: the standing launch held 32.6 of 34.2 GB on
every card and that test's `x.float() @ w.float().T` reference asks for 1.19 GiB on its own. With the cards free both
venvs run it through.

#### Serving, same recipe both sides: `MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same`, uncapped

The 0.32 column is the record [context-262k.md](context-262k.md) measured **the same morning** (05:27-05:45 UTC) on
the same box and the same recipe; the 0.33 column is a fresh launch at 09:17 (log
`~/logs/ctx262k-033-091658.log.out`, KV line) and a second at 09:46 (`ctx262k-033b-094629.log.out`).

| probe | 0.32.0 | 0.33.0 |
|---|---|---|
| KV pool, tokens | 540,202 / 544,919 / 546,391 | **535,781** and **545,949** (two launches) |
| `probe32.sh` 32 in flight, 300-token waves | 1,568.3 / 1,573.7 / 1,587.5 → 1,576 | 1,353.7 (first request the launch ever served) / **1,577.8 / 1,562.5** |
| `probe32.sh` 32 in flight, 1,200-token waves | 1,799.5 / 1,832.7 → 1,816 | **1,812.8 / 1,817.0** |
| `probe32.sh` 16 in flight, 300-token waves | 1,133.1 / 1,135.8 / 1,142.7 → 1,137 | **1,107.3 / 1,145.0 / 1,122.0** |
| power with `gpu_busy_percent == 100` | median 299 W, max 300.0 W | **median 300 W, max 315 W** (n=139) |
| ladder 60,061 tok (cold TTFT / prefill) | 6.93 s / 8,672 tok/s | 8.42 s / 7,130 first-ever request, then **6.96 s / 8,628** |
| ladder 120,066 | 14.89 s / 8,062 | **15.03 s / 7,988** (repeat 15.10 / 7,951) |
| ladder 180,070 | 24.22 s / 7,436 | **24.36 s / 7,391** (repeat 24.39 / 7,383) |
| needle at each tier | found, 7-token answer | **found at all three** |
| `bench/quality.py 100 8` | GSM8K 99/100, 33,909 prompt tokens in 40 s; needle 3/3 | **GSM8K 98/100**, 34,248 tokens in 41 s; **needle 3/3** (3,828 / 15,196 / 30,354) |
| `bench/itl.py 1 2 4` | not recorded for this recipe (the 14.3 ms row is the 32k / MTP-3 launch) | **13.0 / 14.8 / 18.5 ms** median, p90 13.3 / 15.1 / 19.0 |
| single-stream decode | 186.7 tok/s at the 32k/MTP-3 recipe | **187.8 / 189.5 tok/s** at 262k/MTP-2 — not a pair, both listed so nobody pairs them later |
| `bench/sanity_stress.py --pre ""` 60 rounds at conc 33 / 48 / 64 (= NSEQ+1 / 1.5x / 2x) | release practice, run per launch | **0 bad of 1,980 / 0 of 2,880 / 0 of 3,840** |
| `--pre docs,bb --depth 24000 --conc 33` 10 rounds (long prefills first) | | **0 bad of 330** |
| `bench/soak.py --conc 16 --seconds 1500 --lo 200 --hi 30000` | not recorded on this box (the 664 / 792 / 690-request soaks in the release notes are VM100 at the cap) | **882 ok, 0 errors, 0 context-length rejections** in 1,505 s; prompts 192-28,271 tok (median 13,750), **8,135 prompt tok/s**, 150.0 output tok/s, median latency 27.3 s. VRAM **flat**: 16 samples, min == max on every card (30.53 / 30.55 / 30.55 / 30.48 GiB) |

**What `#60084` (32-bit buffer addressing on the `TRITON_ATTN` KV views) is worth on our path: nothing measurable,
and it broke nothing.** 1,562-1,578 against 1,576 aggregate at 32x300 (-0.4%), 7,988 / 7,391 against 8,062 / 7,436
prefill at 120k / 180k (-0.9% / -0.6%), which is inside the spread [context-262k.md](context-262k.md) already
records between instances of the same recipe. The check that would have caught a kernel reading past the end of the
region it now bounds is the ladder (needles found at 60k / 120k / 180k, 30.5 GiB/card high-water) and the
long-prefill sanity run (0 bad of 330) — both clean. The KV-pool figure needs the same treatment the 0.32 note
already gives it: 535,781 and 545,949 across two 0.33 launches against 540,202 / 544,919 / 546,391 across three
0.32 launches. Read it as +-1% launch spread, not as a version effect.

Nothing on the production venv, the production caches or the running launch was touched on 2026-11-02: `0.33.0` was
never installed into `~/venv-r9700`, and `compat/gate.py`'s TESTED lists were deliberately left without it — that
line means "serving gate passed on it", and BetterBench is still outstanding. Two patches (`mtp_allowlist`,
`glm5next_rdna4`) warn on both 0.32.0 and 0.33.0 exactly as they did before.

**One more probe, quoted so it is not mistaken for a delta later.** `bench/bench.py`'s concurrency sweep on the 0.33
launch: aggregate 176 / 459 / 826 / 1,231 / 1,775 / 1,583 / 1,796 tok/s and per-stream median 176 / 119 / 106 / 81 /
58 / 55 / 41 at 1 / 4 / 8 / 16 / 32 / 48 / 64 in flight (300-token outputs). Against the same sweep recorded in
[native-4xr9700.md](native-4xr9700.md) -- 193 / 544 / 900 / 1,203 / 1,537 / 1,454 / 1,532 -- it reads *both* below
at low concurrency and 15% above at 32 and up. That is not a version effect: that table is the **32k recipe on
2026-10-31**, and the paired measurements for today are the `probe32` rows and the ladder, which sit within 1%. The
NSEQ=32 knee the old note describes still shows up the same way: aggregate peaks, and the per-stream spread blows up
past 32 (55.5-63.4 at 32, 33.0-62.5 at 48).

### Hardware note, because it decides the schedule

[incident-2026-11-02/README.md](incident-2026-11-02/README.md) records seven self-resets in one night on this box,
three of them idle, cause still open (BMC/PSU/watchdog material) — and four more the same day. The gate ran anyway,
because the box was already down when the window opened, and it took two launches with it:

- the first 0.33.0 launch came up at 09:17, served `probe32` + the ladder + `quality.py` clean, and died **mid
  request** at 09:39:21 in a platform reset (`pcieport c0:03.1 [12] Timeout`); its log has **zero tracebacks, zero
  HIP errors and zero aborts in 8,584 lines**;
- the 0.32.0 launch before it died the same way at 09:09:32 (`pcieport c4:00.0 [ 7] BadDLLP`), also healthy to the
  last line.

So: **0.33.0 is not what is unstable here**, and a soak that a hard reset cuts in half proves nothing either way.
The soak and BetterBench rows stay open until the platform stops resetting, not because the version failed them.

### Upstream to watch (not in 0.33.0)

| what | state 2026-11-02 | why it matters here |
|---|---|---|
| `#62799` Triton skinny GEMM for 6-128 rows on RDNA4 | open, updated 2026-10-30 — **measured here, not worth carrying** (next section) | the only upstream item in this whole sweep with a two-digit R9700 win in it: the author measures **+28-29% output throughput with DSpark at conc 1-16 on one R9700** (gfx1201), 0.75-0.81x GEMM time over 36 shapes, because BF16 GEMMs with 6-128 rows fall from `wvSplitK` (600 GB/s) to hipBLASLt (420-470 GB/s) |
| `#57861` avoid RowWise FP8 scaled MM on RDNA4 | open, updated 2026-11-01 | vLLM 0.28+ picks RowWise on gfx1200/1201 where ChannelWise decodes faster |
| `#62718` split `skinny_gemms.cu` per kernel family | open, updated 2026-11-02 | ROCm build time (~58 s serialized); matters if we start source-building |
| `#64874` `Workspace is locked` assert in `rocm_fp8_paged_mqa_logits` (GLM-5.3-Flash, high decode concurrency) | open, filed 2026-10-31 | our `models/glm5next.py:270` reserves the decode logits through `current_workspace_manager()` — read this one before moving any GLM launch onto a new pin |
| `#63735` RDNA launch config for `kernel_paged_attention_2d`, `#63054` / `#64614` 64-bit row offsets in the Triton and RDNA-hybrid W4A16 GEMMs | merged, **main only** | RDNA4 fixes that will not reach a wheel until 0.34 |

### Measured, so we do not chase it: `#62799` is not worth carrying here

`tuning/skinny_probe.py` (new, 2026-11-02) graph-times the three per-rank BF16 decode GEMMs of Flash-Next at
TP4 across the row counts the PR targets — the router gate `[M,2560] x [512,2560]^T` (2.62 MB, **51 calls a
step**), the hyper-connection down `[336,10240]` (6.88 MB) and up `[10240,320]` (6.55 MB, 48 calls a step) — for
`torch.matmul`, `torch.mm(out_dtype=fp32)` and our own `r9k_router` kernel forced past its default `MAX_M`.

| shape | graph-replay floor on this box | hipBLASLt | our kernel | net of the 4.9 µs graph-launch |
|---|---|---|---|---|
| empty captured graph | **4.88 µs** | — | — | — |
| 64-element elementwise | 10.64 µs | — | — | ≈ 5.8 µs |
| 128x128 bf16 matmul | 12.12 µs | — | — | ≈ 7.2 µs |
| router `[512,2560]` at M=16 | — | 15.04 µs (bf16 out) / 17.60 µs (fp32 out **+ cast**, which is what the served op does) | 29.2 µs (2-3x worse — the note in PROGRESS: it re-reads activations per expert) | hipBLASLt ≈ 10.1, served path ≈ 12.7 |
| router at M=1..8 | — | 14.8-15.6 µs (168-177 GB/s) | **12.4-14.3 µs (183-212 GB/s)** | the default `MAX_M=8` is the measured crossover |
| HC down `[336,10240]` at M=1..8 | — | 24.2 µs (284 GB/s) | — | — |

Two things kill the idea:

1. **Our decode path is not on hipBLASLt where the PR says it should be.** The launcher serves with
   `R9K_HC_FP8_DECODE=1 R9K_HC_FP8_DECODE_MAXM=255` (`serve/native-flashnext.sh:21`): the HC down/up GEMMs run
   per-token fp8 on our WMMA kernels for every decode M up to 255, and `R9K_ROUTER_MAX_M` (default 8) puts the
   router gate on `r9k_router` for the small-M end (that default is **4 since 2026-11-08**, and the reason is a
   fault, not a measurement -- see below and [bug-router-gemm-oob.md](bug-router-gemm-oob.md)). The band the PR
   fills is mostly already filled here.
2. **What is left is latency-bound, not bandwidth-bound.** hipBLASLt reads the 2.6 MB router weight in 15.0 µs —
   a 128x128 matmul (12.1 µs) is 2.9 µs of it. The PR's 420-470 GB/s figure is for much larger weights; on our
   per-rank decode shapes the floor is the launch, so a 600 GB/s streaming kernel buys almost nothing: the whole
   recoverable envelope at M=16, taking the band floor as the best case, is `(17.6 - 12.1) us x 51 calls = 280 us`,
   i.e. **2.2% of a 13 ms step**, and that assumes a kernel as fast as a trivial matmul.

Two smaller findings worth keeping, both from the same probe:

- **`R9K_ROUTER_MAX_M` above 8 is a loss, and the kernel's own limit is 32.** At M=16/24/32 the r9k router costs
  29.2 / 41.1 / 43.7 µs against hipBLASLt's 14.6-14.9 — so raising the knob to the kernel's `r9k_router_max_m()`
  (32) would add ~1.5 ms a step. The default 8 was not timid, it was the measured *speed* crossover; the serving
  default is 4 now because the `MT=8` instantiation has faulted four times on this box
  ([bug-router-gemm-oob.md](bug-router-gemm-oob.md)) -- measured cost of that cap: ITL at conc 1/2/4 goes
  12.9/14.7/18.5 ms at cap 8 -> 13.0/15.1/18.4 ms at cap 4, and the 0.33 comparison arms have to carry the same
  cap on both sides.
- **The one candidate the probe exposes is the cast, not the GEMM.** Above `MAX_M` the served op is
  `torch.mm(out_dtype=fp32)` followed by a bf16 cast (`r9700_vllm/router.py:72`), and the cast measures
  **2.6 µs a call** (15.04 -> 17.60) = **130 us a step, ~1.0%**, plus 51 fewer launches. Handing the gate fp32
  instead (its softmax/top-k works in fp32 anyway) is a one-knob A/B, not a new kernel. It changes router rounding
  on near-ties, so it owes `tests/test_router_r9k.py`'s top-10 agreement and the GSM8K/needle gate — and by the
  bar this stack already uses for `moe/fold.py` (~0.1 ms of a 16.7 ms step "does not buy two patches"), ~0.13 ms
  buys one flag and no patch at all. **Not run yet.**

### Verdict and plan

**Not adopted: production stays on 0.32.** As of 2026-11-02 10:19 UTC the box was serving the standing 262k recipe
on `~/venv-r9700-033` (`vllm 0.33.0+rocm72`), and everything on the standing gate list that this platform lets us
run had come back level: integration, four kernel gates, `probe32` at three protocols, the retrieval ladder with its
needles, GSM8K + needle, `sanity_stress` at NSEQ+1 / 1.5x / 2x and with 24k prefills first, and a 25-minute
16-client soak with VRAM flat. Level, not better — and the one quality number it never produced is BetterBench,
because the box reset four more times that morning (see `notes/incident-2026-11-02/README.md`). Following a release
line that changes nothing measurable, on hardware that is the actual constraint, is not worth the second serving
line: the serve is back on `~/venv-r9700` (`vllm==0.32.0+rocm723`, pin `e4a5b553...`) with the same
262k/32/8192 recipe, and `compat/gate.py`'s TESTED lists are unchanged.

The 0.33 venv and caches stay in place as a ready-made re-measurement — no rebuild needed to re-check:

```bash
MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same \
  VENV=$HOME/venv-r9700-033 VLLM_CACHE_ROOT=$HOME/vllm-native-cache-033 \
  TRITON_CACHE_DIR=$HOME/vllm-native-triton-033 bash serve/native-flashnext.sh
```

What would change the decision, in order of likely value: `#62799` merged (the one item with a two-digit R9700
number in it — though our own decode band is latency-bound, section above, so expect less than the PR's +28-29%
here), `#57861` merged (RDNA4 FP8 kernel selection), and `#64874` (the DFlash `visual` patch) before any GLM launch
moves pin. If a 0.34 wheel is ever taken it inherits this note: the import audit is a `git diff` away, and the gate
order is `test_integration.py` -> the four kernel gates -> `probe32` -> ladder -> sanity -> BetterBench.

One operational consequence to keep in mind: **the 0.32 pin is no longer reinstallable from vLLM's wheel index.**
Per-commit indexes expire, and none of the six 0.32-era pins checked on 2026-11-02 (`26958d9e5`, `fe341a53d`,
`400461cad`, `79a8da9c4`, `0c3645128`, `5df63a72a`) is live. `~/venv-r9700` is therefore the artifact of record --
the thing to keep, not to rebuild. If it is ever lost, the closest live wheels are 0.33-era (measured level here,
but `compat/gate.py` would refuse to serve them until they are in the TESTED lists).

Steps, in order:

1. Fix the wheel URLs/flavor (done with this note).
2. Kernel gates that need VRAM, plus the 262k launch on `~/venv-r9700-033` with separate caches, measuring the two
   things 0.33.0 could change — `set_triton_ptr_range` on our fp8 KV path and the KV-cache token count. **Done**;
   both came out inside the launch-to-launch spread.
3. `sanity_stress` at NSEQ+1 / 1.5x / 2x, `soak` 1500 s, GSM8K, `bench/probe32.sh` bands. **Done**, table above.
   Not done: BetterBench, `bench/humaneval.py`, the 27B TP2 and GLM configs (that launch lives on the eight-card
   VM, and it owes `#64874` a read first).
4. When BetterBench is in: `0.33.0` into `compat/gate.py`'s TESTED lists, `amd-aiter 0.1.26` in the venv, the image
   `BASE` (the nightly line has no 0.33-era ROCm 10 image pinned yet — a separate question from the venv), the
   README/notes pin text and the release note.
5. Separately: `post_gemm2_hook` vs `moe/fold.py`, and `get_rdna_version()` replacing our arch probes. Both are
   plugin changes with their own A/B, not part of a pin bump.

---

## On this box: the comparison to run (open, 2026-11-08)

Nothing here needs a rebuild: `~/venv-r9700-033` (`vllm 0.33.0+rocm72`, torch/triton/aiter/openmpi and both
editable installs shared with the production venv, `libr9k.so` untouched) and its two caches were left in place on
2026-11-02 for exactly this. Run it when the box is idle and not in a reset storm -- Part 2's hardware note says
why, and [incident-2026-11-03/README.md](incident-2026-11-03/README.md) says the box was still dying at
02:50 on 11-03.

Gate order (the one Part 2 set, unchanged), each arm paired with a 0.32 launch of the same recipe:

```bash
# arm A -- production venv, the recipe this box serves today
MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same bash serve/native-flashnext.sh
# arm B -- the 0.33 venv, caches kept off the production ones
MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same \
  VENV=$HOME/venv-r9700-033 VLLM_CACHE_ROOT=$HOME/vllm-native-cache-033 \
  TRITON_CACHE_DIR=$HOME/vllm-native-triton-033 bash serve/native-flashnext.sh
# arm C -- the rocm100 wheel, the flavour the docker pin is built from (new: it needs its own venv)
mkdir -p $HOME/venv-r9700-033r && python3 -m venv $HOME/venv-r9700-033r   # or cp -a ~/venv-r9700-033 and swap the one wheel
$HOME/venv-r9700-033r/bin/python -m pip install --no-deps vllm==0.33.0+rocm100 \
  --extra-index-url https://wheels.vllm.ai/rocm/4ba787ec9959e566d3b587a796dff07673324da5/rocm100/
```

Arm C is worth the extra venv: `notes/native-4xr9700.md` flags the `rocm100` flavour -- the one whose **build SDK
matches the runtime the box actually loads**, and the one upstream's `v0.33.0-rocm100` image is built from -- as
never measured here. Both 0.33.0 wheels were live at `4ba787ec9` on 2026-11-08 (`+rocm72` manylinux_2_39,
`+rocm100` manylinux_2_34, `amd-aiter 0.1.26` beside each); `orjson` is a new hard dep of 0.33 and is already in
`~/venv-r9700-033`.

**Switching arms means stopping the serve, and that is the dangerous part, not the measuring.** Every arm change
goes through `bench/gate.sh` (verified stop, then it waits for the ranks' ~50 GiB of anon to be unmapped and fails if
the box has not given it back -- relaunching into the half-torn-down state is what produced the amdgpu SVM/GPU-fence
deadlock in [load-oom-svm-wedge.md](load-oom-svm-wedge.md)). Do not type the launcher's `pkill` patterns into a shell
by hand: they match the caller's own command line. Two arm switches per comparison, so the gate runs four times.

0. **Rebuild `libr9k.so` first -- it is older than the tree it now sits in.** The committed artifact is dated
   2026-11-05; the merge changes `kernels/r9k_dsv4_glue.hip` (upstream's rotation pre-pass rounds once, bit-equal to
   exllamav3). So `tests/test_dsv4_glue.py`, which now asserts bit-equality at every M, fails against the stale `.so`
   and the next launch would load the old kernel. It is a 4-12 minute hipcc compile (CPU-heavy), so it waits with the
   rest of this list:
   ```bash
   cd kernels && HIPCFLAGS=-I$HOME/opt/rocm-dev/opt/rocm/core-10.1/include ./build.sh && cp libr9k.so ../r9700_vllm/kernels/
   ```
   Nothing else in the merge touches a kernel source, so one rebuild closes it; `build.sh`'s export gate then runs the
   126-name list on the new artifact.
1. `tests/test_integration.py` (34 checks) + the four VRAM-hungry kernel gates -- 0.33 already passes these here;
   re-run only because the tree moved under them (the merge brings PR #2's launch-time collective calibration).
   **That calibration is the one confound in the pair.** The plugin tree is shared by both arms, so arm A runs
   `comm/calibrate.py` too even though the 0.32 numbers it is compared against were taken without it. On this box
   every card sits behind a switch, which is the regime the fixed cutoffs were measured on, so run the first launch
   of each arm with `R9K_ARN_CALIBRATE=0` and the paired `probe32` rows with it off; then one launch per arm with it
   on, and report what it moves here as its own row rather than folded into the version delta.
2. `bench/probe32.sh` at 32x300 / 32x1,200 / 16x300, twice per arm -- the paired 0.32 band is 1,576 / 1,816 / 1,137.
3. The retrieval ladder with needles, `bench/needle_long.py` -- 60k / 120k / 180k must stay at 8.6k / 8.0k / 7.4k tok/s
   and find the needle at every tier; this is the check that bounds `#60084`'s new `ptr_range` assertion.
4. `bench/quality.py 100 8` (GSM8K + needles), then **BetterBench**, the row Part 2 never produced and the only
   quality number that can veto the move on this box.
5. `bench/sanity_stress.py` at NSEQ+1 / 1.5x / 2x and `bench/soak.py --conc 16 --seconds 1500`, with the VRAM
   high-water read per card.

What then lands, in one commit: `qsa_kvfp8` and `moe_h2d_batch` gain `0.33.0` in `compat/gate.py` (or the reason
they do not), `notes/native-4xr9700.md` and the README's native recipe name the venv the box runs, and this section
closes with the two paired tables. If an arm is cut by a platform reset, it is re-run, not quoted from Part 1.
