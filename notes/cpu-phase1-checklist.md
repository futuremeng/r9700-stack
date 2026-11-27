# Phase 1 on the real box: perception layer bring-up checklist

Executable end to end, step by step, each step with the expected answer and the branch when it differs.
The plan it implements is `serve/cpu-sidecars.md`; the launcher is `serve/percept.sh`. This box: no root,
no docker, HF through hf-mirror, cmake in the repo venv — the checklist assumes all of that.

Fill in: date ____ , git HEAD ____ , engine recipe ____ (from `.r9700-state/launch.json`).

## 0 — Prereqs (5 min)

| # | check | command | expect |
|---|---|---|---|
| 0.1 | engine in serving, cards held | `bash serve/native-flashnext.sh` untouched; `curl -s localhost:8180/health` (or the gateway) | 200 |
| 0.2 | Phase-1 files present | `ls serve/percept.sh serve/cpu-sidecars.md && git log --oneline -1` | both exist; HEAD recorded above |
| 0.3 | sidecar toolchain | `command -v taskset ffmpeg; ls .venv/bin/cmake` | taskset + ffmpeg found, venv cmake exists |
| 0.4 | memory headroom | `awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo` | ≥ 18 GiB at steady state (the 18 GB this plan budgets from) |

ffmpeg missing → a static build or the distro package from whoever holds root; do NOT `pip install` an
ffmpeg wrapper into the venv as a substitute for the binary.

## 1 — Baseline, recorded before anything new runs

```
bash bench/probe32.sh | tee ~/logs/probe32-before.txt
awk '/pswpout/{print $2}' /proc/vmstat > ~/logs/pswpout-before.txt
awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo > ~/logs/memavail-before.txt
```

Expect: probe32 aggregate lands in the recorded spread (1,767–1,836 tok/s on this box, README table).
If it reads outside the spread, fix THAT first — a Phase-1 delta measured against a broken baseline is
not a measurement.

## 2 — Build sherpa-onnx (~10 min of build; do not overlap a cold start)

```
git clone --depth 1 https://github.com/k2-fsa/sherpa-onnx ~/sherpa-onnx
cmake -S ~/sherpa-onnx -B ~/sherpa-onnx/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
      -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
      -DSHERPA_ONNX_ENABLE_PYTHON=OFF
cmake --build ~/sherpa-onnx/build --target sherpa-onnx-offline-speaker-diarization -j 16
```

Diarization is ON by upstream default (checked in CMakeLists at v2.0.3 — no flag needed; it pulls
hclust-cpp). PORTAUDIO stays OFF because the box has no alsa-dev and sudo needs a password.
Expect: `build/bin/sherpa-onnx-offline-speaker-diarization --help` exits 0.

Failure branch — the build fetches onnxruntime from github: if the fetch fails on this network, download
the onnxruntime static tarball for linux-x64 by hand, unpack under `~/opt/onnxruntime`, and re-configure
with `-Donnxruntime_SOURCE_DIR=$HOME/opt/onnxruntime` (`SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE`
is ON, so the copy is picked up).

## 3 — Models: two verified repos, no guessing

The pair was pinned by checking the hf-mirror API 2026-11-27 — the diarization community bundle is
GATED pyannote, not this path; these two ungated sherpa exports are what the diarizer wants, and both
files resolve through the mirror with plain curl (verified sizes in parentheses):

```
mkdir -p ~/models/percept
curl -sL https://hf-mirror.com/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx \
     -o ~/models/percept/segmentation-3.0.onnx        # 6.1 MB
curl -sL https://hf-mirror.com/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx \
     -o ~/models/percept/campplus.onnx                # 27 MB
```

The names are the ones serve/percept.sh defaults to, so no native.env edit is needed unless the files
moved. If either resolve 404s, list the repo first (`curl -sL
https://hf-mirror.com/api/models/<repo>/tree/main`) and pin whatever SEG_MODEL/SPK_MODEL the listing
actually shows -- ls, don't assume.

## 4 — DRY, then first run

```
DRY=1 TASK=diarize bash serve/percept.sh
```
Expect: pinned command with both model paths that now exist, `cores: 48-63 (16 online) -- threads 16`,
no refusal, nothing started.

```
ffmpeg -i meeting.m4a -ar 16000 -ac 1 /tmp/meeting.wav
TASK=diarize INPUT=/tmp/meeting.wav bash serve/percept.sh
tail -20 ~/logs/percept-diarize.log
```

Expect: JSON with per-speaker segments and timestamps; wall time at least several times faster than the
audio's length (if it is SLOWER than real time, stop — a thread/pin mistake was made; re-check §4 of
serve/cpu-sidecars.md's ordering rules and the DRY line).

## 5 — Acceptance: the four numbers from serve/cpu-sidecars.md

```
bash bench/probe32.sh | tee ~/logs/probe32-after.txt
```
5.1 **aggregate within ±1 %** of probe32-before. Outside it: stop the sidecar, re-measure; if back
inside, the sidecar's core range is the cause — move 48-63 off whatever it overlaps (check
`taskset -pc <engine pids>` hasn't been pinned by hand onto the same range).

Soak 15 min with diarize in flight the whole time:

```
until [ ! -f /tmp/diarize-soak ]; do TASK=diarize INPUT=/tmp/soak.wav bash serve/percept.sh; done
```
and in another shell, 30 s apart:
```
awk '/pswpout/{print $2}' /proc/vmstat; awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo
```
5.2 **pswpout delta ≈ 0** while serving. Anything above the idle drift: something is writing anon —
re-read the mmap-only rule (embed.sh header; the 2026-11-01 incident).
5.3 **MemAvailable ≥ 12 GiB** throughout. Below it the Phase-3 daemon's unload threshold would be
firing — find what spent L3 before it stays true.
5.4 cold start next reboot unchanged vs the recorded load-time baseline. If the next cold start is
slower, L3 got spent: this checklist's §3 pulled 2-3 GiB of model into page cache while the engine
streamed 110 GiB — schedule Phase-1 model fetches BEFORE the engine's reload, not during, and say so
here in the record.

## 6 — Stop path

```
bash stop.sh --check
STOP=1 TASK=diarize bash serve/percept.sh
```
Expect: --check lists the percept tasks under step 3; STOP answers "no diarize run in flight" once the
soak loop is removed. `./stop.sh --embedding` must already cover Phase 1 — it does, via percept.sh's
enumerated tasks; nothing else to add.

## 7 — Record

Paste into this file: HEAD, the probe32 pair, the pswpout/MemAvailable triple, the diarization RT factor,
which model pair landed (§3), and one sentence per acceptance row under this line:

- date: ____  HEAD: ____
- 5.1 probe32 before/after: ____ / ____ tok/s
- 5.2 pswpout delta during soak: ____
- 5.3 MemAvailable min during soak: ____ GiB
- 5.4 cold-start after: ____ s vs baseline ____ s
- diarize RT factor: ____×
- model pair: community-1 / fallback pair — names: ____

## 7b — Monitor page (:8094)

The perception layer is CLI-shaped, so its Monitor reads the run journal instead of polling a port:

```
bash serve/percept-watch.sh            # page + state.json on 127.0.0.1:8094
STOP=1 bash serve/percept-watch.sh     # stop it
DRY=1  bash serve/percept-watch.sh     # print what would start, start nothing
```

On the real box the four cards must show reality: the binary path §2 built (not 未在 PATH), the §3
model files with real MB counts, 绑核 48-63 with 在线 16, and the guard card's MemAvailable vs
MEM_FLOOR. If native.env sets SHERPA_BIN/SEG_MODEL/SPK_MODEL the wrapper passes them through; a card
still showing a `$MODEL_DIR` literal means that file is not where the wrapper looks (`LOCAL_ENV`).
Optionally fold it into the unified Monitor page instead of its own tab — add to the gateway launch:

```
SIDECAR_STATE="$SIDECAR_STATE,/diarize=http://127.0.0.1:8094/state.json" bash serve/gateway.sh
```

`bash stop.sh --embedding` already stops it (percept-watch.sh is in the monitors list).

## Mac dry-run (dev checkout, arm64, 2026-11-27) — what transferred and what passed

The functional chain ran end-to-end on the Mac with the pip wheel behind a shim binary
(OfflineSpeakerDiarization API), the two §3 models, and 11.4 s of two-voice `say` audio:
`percept.sh` guards → launch → JSON segments `{0.03–4.79 spk0}, {5.26–11.10 spk1}` — 2 speakers,
boundary at the right 5-second switch, 1.27 s wall including model load (~9× realtime on 16
oversubscribed Mac threads; the 9654 number comes from §5). Two findings folded back into §4:

- With the default clustering (`num_clusters=-1, threshold=0.5`) the two clean TTS voices merged into
  one speaker; forcing `num_clusters=2` separated them at the right boundary. Real meeting audio with
  real voices will not need forcing — but a diarization that returns one speaker for an audio you know
  has two is a clustering question, not a pipeline bug. Try forced k before believing the model failed.
- The wheel ships no standalone CLI binary and no numpy; the §2 build from source is still the real-box
  path. The shim was test scaffolding for this checkout only.

## Known gaps this checklist does not paper over

- `TASK=align` has no CPU runtime pinned yet (`ALIGN_CMD` unset); its acceptance is its own task.
- sherpa-onnx is built `-j 16` on idle cores here; it is a one-time build cost — do it while the engine
  is at steady serving, never during a reload.
- The onnxruntime fetch step (§2 failure branch) has not been exercised on this network; if github is
  fully blocked the manual tarball path is the documented way.
