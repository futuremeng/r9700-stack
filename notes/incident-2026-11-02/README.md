# notes/incident-2026-11-02 — seven resets in one night; three of them idle, one of those with no job on the box at all

Same box as [incident-2026-11-01b](../incident-2026-11-01b/README.md) (`jiulu-9700`, kernel `7.0.0-42-generic`,
4× Radeon AI PRO R9700 `1002:7551`, bare metal), caught 2026-11-02 00:2x–00:5x UTC. **It did not stop there: four more deaths follow the same signature later the
same day (02:58, 06:30, 09:09, 09:39), one of them taking a healthy serve with it — see the last section.** **Observation only: nothing
on the box was changed for this incident** — no sysctl, no modprobe, no BIOS. The in-band IPMI readouts
(`sudo ipmitool -I open …`) were run on the host by the operator; the BMC web-UI material in
`bmc-power-events.txt` is quoted verbatim, and its record ids are corrected against the raw SEL in
`sel-decode.txt`.

The night of 11-01b ended with "two hard resets 32 minutes apart". It was not the end: the same firmware
signature appears **seven times between 20:30 and 00:25**, three of the seven boots were idle in the sample
window before they died, and one of those (death 7) had no job on the box at all.

## The seven deaths, and how the two bundles' boot numbers line up

Boot indices are relative to the boot each file was read from, and the two bundles read from different boots —
`boots.txt` numbers are its own, `incident-2026-11-01b` used another. The stable key is the boot **id**.

| death | boot id | first entry → last entry | life | what was on it | `-11-02` idx | `-11-01b` idx | firmware latch at the next boot |
|---|---|---|---|---|---|---|---|
| 1 | `aa661de3` | 11-01 03:27:55 → 20:30:41 | 17 h 03 m | serve up; `sa01` at 20:30:04 reads **97.86% idle** | -6 | -2 | yes (APIC `0x83`) |
| 2 | `9c62bb9e` | 11-01 20:36:38 → 21:02:15 | 25 m 37 s | serve up; a load had started at 21:01:2x, ~50 s before the death, so the 21:00 sample is idle too | -5 | -1 | yes |
| 3 | `6ae7b308` | 11-01 21:03:56 → 23:45:01 | 2 h 41 m | serving; **the boot 11-01b was captured from** | -4 | 0 | yes |
| 4 | `3b531c63` | 11-01 23:52:35 → 23:53:07 | **32 s** | nothing running, 32 s into a boot that was itself the recovery from death 3 | -3 | — | **none at all** |
| 5 | `b6aae6d2` | 11-01 23:59:36 → 00:01:37 | 2 m 01 s | recovery boot; no job recorded in this bundle | -2 | — | yes |
| 6 | `2989bceb` | 11-02 00:04:54 → 00:15:01 | 10 m 07 s | recovery boot; no job recorded in this bundle | -1 | — | yes |
| 7 | `75a86b4c` | 11-02 00:17:12 → 00:25:01 | 7 m 49 s | **idle**: `sshd` accept, one `crontab -l`, `update-notifier-download`, then a root `debian-sa1` cron; no vLLM rank, no HIP process | 0 | — | yes |

Earlier boots, for mapping only: `42f83cb5` `-15`/`-11` · `46015a3c` `-14`/`-10` · `b57181a2` `-13`/`-9` ·
`e7dd6b86` `-12`/`-8` · `01697656` `-11`/`-7` · `8a22f104` `-10`/`-6` · `d4275a7d` `-9`/`-5` ·
`52deb9f3` `-8`/`-4` · `ab5f1a30` `-7`/`-3`.

Two provenance notes that only show up when the tables are put side by side:

- `boots.txt` was run **by death 7** (its "this boot" block reads `up 8 min` at `00:25:10`, nine seconds after
  that boot's last journal line). Its own last row is the boot that ran the command. `idle-deaths.txt` says so
  in its header: the file was created empty at 00:25 in the boot that died at 00:25:01.
- The boot the later BMC readouts came from — `7e5bfe69`, 00:27:28 → past 00:50, up 22 min, load 0.1–0.3, no
  vLLM, four GPUs enumerating — is in neither table.
- Death 3 is the row `incident-2026-11-01b` calls boot `0` and whose last entry it recorded as `23:40:03`:
  that capture was taken 5 minutes before that boot itself died. The bundle was written while the incident was
  still running.

## Files

| file | what it answers |
|---|---|
| `boots.txt` | the boot list and `wtmp` under this bundle's numbering — note that no `shutdown` record closes deaths 3, 5 and 6 |
| `idle-deaths.txt` | the deaths with nothing running: death 7 in detail, and `sa01` as an independent witness that the box was idle in the sample windows before deaths 1 and 2 |
| `reset-reasons.txt` | what each following boot's firmware reported: reset reason, BERT, MCE |
| `bmc-power-events.txt` | the BMC web-UI paste, its OS-side counterpart, and the correction to 11-01b's dating (below) |
| `bmc-sensors-watchdog-psu.txt` | in-band `mc info` / `sel info` / `sdr elist` / watchdog / power: what it rules out and what it newly implicates |
| `sel-decode.txt` | the raw SEL mapped one-for-one onto the seven deaths, and the diagnosis it leaves |

## What this bundle establishes

- **The platform is resetting itself while powered, and the gaps say so.** Deaths 2, 6 and 7 are back on the
  BMC's own record **9 s, 41 s and 56 s** after their last journal line (death 7: power-on at 00:25:57, first
  journal of the recovery boot at 00:27:28 — 2 m 27 s including ~90 s of POST), with no power-off record and
  nobody logged in. A person cannot be that fast and that consistent. Death 3 is two power-on transitions 91 s
  apart with one OS boot after them: the platform re-tried POST on its own.
- **Death 4 is the odd one out and the only genuine power-off**: `0x12b4` `Asserted` at 23:54:57, 1 m 50 after
  boot `3b531c63` stopped logging, the only true power-off in the whole window and the only death whose
  recovery boot carries no reset-reason latch. Whether a chassis command with a username started it is the one
  human question left ("Still open" below); the other six stay platform-initiated either way.
- Not the power, not the watchdog, not a human: no mains/PSU event is logged once in 3639 SEL records, and the
  BMC watchdog is `Stopped` with action `No action` and expiration flags `None`. That puts the seven deaths
  inside the firmware's own verdict, `Previous system reset reason [0x88000800]: an uncorrected error caused a
  data fabric sync flood event`, and inside the BERT: `ProcessorError`, `cache error`, `Uncorrected`, socket 0,
  APIC `0x82` (6×) and `0x83` (1×) — six of seven with the identical MCE signature
  `Bank 5 aea0000000000108 / SYND 4d000000 / IPID 500b024029a00 / PPIN 2b61290dcb2002b`.
- **Load is not required to trigger it.** Death 7 died at idle — no vLLM rank, no HIP process, load average
  0.12, and its last kernel lines are an `igb` link-up plus one `hrtimer: interrupt took 27271 ns`: no AER, no
  MCE, no blocked task, no amdgpu line after enumeration — while the firmware still called it the same death.
  Death 4 died 32 s into a boot that was itself the recovery from death 3. That puts the fault domain outside
  anything the `notes/native-4xr9700.md` C22/C23 thread measures: the same signature appears at idle and at
  full model load, so host-side work in the load path cannot be it. What is still common to all seven is the
  box being up and running Linux on this platform.
- **The firmware cannot date a death, and 11-01b's numbers were read as if it could.** Every
  `PROCESSOR 2:a10f11 TIME <epoch>` in the latched deaths is 17 s before the *first journal line of the boot
  that recovered it*, 5/5: the record is written during the next boot's POST. Any "it hung N minutes before the
  reset" reading of `incident-2026-11-01b/mce-firmware-records.txt` is void; the BMC's power-on records are the
  only external clock that can date a death.
- **`PSU2` is installed, detected, and carrying nothing**: `Power In 16 W / Power Out 0 W`, `AC Lost: Presence
  detected`, `PWR Detect: Presence detected` — one supply (PSU1, ~208 W, of which `CPU_Power 96 W`) behind four
  cards. A real redundancy hole, and **not** the cause of these deaths: no power-off, AC-lost or `Pre-Init`
  record at any of them.
- **The SEL is 3639 entries at 100% used, `Overflow: false`, last delete 02/2025**, and at least one power-on
  record inside the incident window is already missing. The web UI's `ID: 4778..4794` are SEL record ids
  printed in **decimal** (`0x12aa` = 4778, `0x12ba` = 4794), which is why `sel get 0x4794` reports no such
  record. Read the SEL in hex ids, or save it.

## Still open

1. Decode the `Unknown | Asserted` OEM record: `sel get 0x12ba` and `sel get 0x12b4` (record type, generator
   id, event data bytes) — that says whether the BMC logs a restart cause at all.
2. The one human question: `0x12b4` at 23:54:57 (death 4). The UI's 操作日志 page will show whether a chassis
   power command with a username/source IP precedes it. The other six stay platform-initiated either way.
3. `sel save … && sel clear` before the next repro — with the operator's OK — or the next death loses records
   too. Not done here: it destroys evidence, and it is the operator's call.
4. Thermal visibility: `TR2`/`TR4` report `No Reading` and `TR3` reports `0 degrees C`, so the four `TR*`
   sensors that produced the 2025 flap era now see nothing — while the SEL is full of `TR2`/`TR3` 88–95 °C
   threshold crossings from when they did. `sdr get 'TR2 Temperature'` for the entity/labels.
5. The *configured* power-restore policy: `chassis policy list` prints the supported set only. `raw 0 7`
   (Get Chassis Power Configuration, low bits of byte 2) says which is in force — which decides whether
   death 7's 2 m 27 s self-recovery was policy or coincidence.

## Later the same day: four more deaths (02:58:40, 06:30:18, 09:09:32, 09:39:21 UTC), the last one taking a serve with it

Written 2026-11-02 ~09:50 UTC, reading the journal back from the boot that came up at 09:41:24. The bundle above
stopped at death 7 (00:25). The box went on: boots `274defed` (01:30:27 → 02:58:40), `3f605810` (03:00:42 →
06:30:18), `002dbde7` (06:35:06 → **09:09:32**) and `d9eca3ef` (09:11:32 → **09:39:21**), then `29ee487c` at
09:41:24.

By 11:20 UTC the day stands at **15 boots** (`journalctl --list-boots | grep -c 2026-11-02`) and the spacing has
collapsed to tens of minutes: `f1fe7747` ran 11:00:46 → 11:05:14, and the boot after it 11:07:37 → ~11:14. Two
relaunches of the standing 262k recipe (both back on `~/venv-r9700`, after the 0.33.0 trial had been dropped) were
cut inside that window before either had served a request — the first deaths today that a launch did not outlive. Nothing
about this changes the 0.32/0.33 finding, which was measured on launches that ran for tens of minutes; it does mean
any number taken between 10:30 and 11:20 has to be treated as taken on a box in a reset storm.

Two of those deaths had a vLLM launch on them, and in both cases the launch was healthy up to the millisecond the
platform went:

- **09:09:32** — the standing 262k launch on vLLM 0.32.0. Its own log ends on `GET /health` / `/api/status` 200s;
  the journal's last lines are `pcieport 0000:c4:00.0: [ 7] BadDLLP` (`aer_layer=Data Link Layer,
  aer_agent=Receiver ID`) plus a `{250}[Hardware Error]` block with `aer_uncor_status: 0x00000000`.
- **09:39:21** — the same recipe relaunched on **vLLM 0.33.0** for the pin gate ([../vllm-0.33.md](../vllm-0.33.md)),
  22 minutes into its life: `probe32`, the 60k/120k/180k retrieval ladder and `bench/quality.py` had already come
  back clean, and its log's last lines are six `POST /v1/chat/completions … 200 OK` with **zero tracebacks, zero
  HIP errors, zero aborts** in all 8,584 lines. The journal ends on `pcieport 0000:c0:03.1: [12] Timeout`.

Both dying boots carry the same link-layer storm, and it is concentrated on the same two ports every time —
`pcieport 0000:c4:00.0` 1,417 lines and `0000:c0:03.1` 505 in the 09:09 boot; 1,042 and 370 in the 09:39 boot,
with 10,855 and 8,421 `[Hardware Error]` journal lines respectively. That is the same pair the boot-by-port tally
in [../incident-2026-11-01b/aer-by-boot-port.txt](../incident-2026-11-01b/aer-by-boot-port.txt) was cut against.
Nothing on the box was changed for these four deaths either: observation only.

What it means for the work: a vLLM pin bump cannot be *gated* on this box while it behaves like this. The
remaining soak/sanity items for 0.33.0 are cheap and mechanical; it is the 25-minute soak and the 20-minute cold
load that the platform keeps cutting short, and a gate interrupted by a hard reset proves nothing either way.

## Why this is in the repo

It is the box state the Monitor and About pages are written against ([notes/webui.md](../webui.md)): a launch
that dies loudly is a software bug, and the pages inside the serve (`serve/plugin`) die with it. A box that
resets without ever writing a `shutdown` record, at idle, is not — and that is the case for keeping the
out-of-process sidecar on 8081, whose own log and whose card readings survive the engine, and for keeping a
watch on the BMC side rather than only on the OS side: the only clock that can date these deaths is the BMC's.

## The AER counters are now on the Monitor, and they are still moving

`amd-smi metric --pcie` cannot answer bandwidth on these cards — `BANDWIDTH`, `CURRENT_BANDWIDTH_SENT` /
`_RECEIVED`, `MAX_PACKET_SIZE` and `REPLAY_COUNT` all read `N/A` for all four GPUs — but the kernel's AER counters
read as an ordinary user with no `/dev/kfd`, which is what the hand-collected
[aer-by-boot-port.txt](../incident-2026-11-01b/aer-by-boot-port.txt) was made of. The Monitor now polls them
(`read_pcie` in `serve/webui.py`, the PCIe panel), so the evidence accrues with timestamps instead of being
grabbed by hand after the fact.

Read between 12:07 and 12:14 UTC, this boot (11:07):

| chain | correctable | detail | non-fatal / fatal |
|---|---|---|---|
| card2 `c6:00.0 → c5:00.0 → c4:00.0 → c0:03.1` | 28 → **31** | upstream port `c4:00.0` BadTLP 24 → 27; root port `c0:03.1` Timeout 2; endpoint and downstream port 1 each | 0 / 0 |
| card4 `06:00.0 → 05:00.0 → 04:00.0 → 00:03.1` | 5 | upstream port `04:00.0` BadTLP 3; endpoint and downstream port 1 each | 0 / 0 |
| card1 `c3:00.0 → …`, card3 `03:00.0 → …` | 0 | — | 0 / 0 |

Three new correctable errors on card2's upstream port in seven quiet minutes — the serve up, one stream finishing,
no reset — and every link still at its maximum 32.0 GT/s x16. `c4:00.0` is the same port named in this note as a
reset site, which is the reason to keep watching it by waveform rather than by snapshot.

Two reading cautions recorded so the numbers are not over-read: `NonFatalErr` inside `aer_dev_correctable` is a
bit of the correctable status set, not a non-fatal error (`TOTAL_ERR_NONFATAL` is the one that means that, and it
is 0 here); and the kernel zeroes these counters on link recovery, so the panel's waves are per-poll deltas that
treat a decrease as unknown, never as a negative count.

## And the request list now survives the death

Same gap on the request side: the Monitor's list lived in the serve's memory, so after each of the day's resets it
came back claiming nothing had run, and "what was it doing when it died" was answered from side logs. Finished
requests are now appended to a JSONL journal (`~/logs/requests-<port>.jsonl` by default; one fsync per finished
request, measured 1.98 ms, rows hold timings and token counts but never the text) and replayed when the pages come
back, each row labelled with the launch it came from. Details in [notes/webui.md](../webui.md).

## Two serve restarts, and what the counters did across them

Restarting the launch (12:50 and 12:55 UTC, to put the request journal and the new Monitor panels into the running
serve) turned out to be a clean observation, because the AER counters are boot-scoped and survive a serve restart:

| counter | 12:14 | 12:57 (after two restarts) |
|---|---|---|
| `c4:00.0` upstream port, card2's chain | BadTLP 27 | BadTLP **30**, BadDLLP **1** (first sighting), correctable total 30 |
| `04:00.0` upstream port, card4's chain | BadTLP 3 | BadTLP **6** |
| `c0:03.1` root port | Timeout 2 | Timeout **3** |
| card2 chain correctable | 31 | **36** |
| card4 chain correctable | 5 | **8** |
| every link | 32.0 GT/s x16 | 32.0 GT/s x16 — still at maximum, no width or speed fallback |
| `TOTAL_ERR_NONFATAL` / `FATAL` | 0 / 0 | 0 / 0 |

So: launching a serve is followed by correctable traffic on the switch upstream ports — five more on card2's
within ~40 minutes spanning two starts — without a single uncorrectable error and without the link renegotiating.
That is the shape of a link that is working but not clean. It is not yet evidence about the hard resets (no fatal
errors have ever been counted here), but it is now a continuous trace with timestamps rather than a hand-run
snapshot, which is what a correlation with a reset needs.

## What moves them: a start, and a long prefill — but not short traffic (13:26, corrected 14:20 UTC)

The first half, taken deliberately at 13:26: eight concurrent 400-token streams through the running serve,
sampling every chain's correctable total every second for the whole burst.

| | before | after | delta |
|---|---|---|---|
| card2 chain (`c6:00.0 → c5:00.0 → c4:00.0 → c0:03.1`) | 36 | 36 | **0** |
| card4 chain (`06:00.0 → 05:00.0 → 04:00.0 → 00:03.1`) | 8 | 8 | **0** |
| card1, card3 | 0 | 0 | 0 |

Over 90 consecutive one-second samples of an idle serve: also **0** nonzero deltas. The conclusion written at that
moment — "launches move them, traffic does not" — was **too strong, and real traffic falsified it inside an hour**.

### The real load, observed only (14:11–14:17)

Actual user concurrency arrived: 4–5 requests in flight, prompts 5k–170k tokens, 54–3,097 generated each. The
sidecar's event list captured **6 correctable events in 34 minutes**, every one `BadTLP` at the same two **switch
upstream ports** the restart-clustered ones used, and the journal made the correlation checkable — each event's
timestamp against the rows whose `started_at … started_at+wallclock` covers it:

| event | chain | in flight | of those, still prefilling | largest in-flight prompt |
|---|---|---:|---:|---:|
| 14:11:49 | card2 `c4:00.0` | 4 | 1 | 67,502 |
| 14:09:46 | card4 `04:00.0` | 4 | 1 | 144,375 |
| 13:53:55 | card4 `04:00.0` | 2 | 0 | 148,718 |
| 13:47:26 | card2 `c4:00.0` | 1 | 1 | 93,400 |
| 13:42:13 | card4 `04:00.0` | 2 | 1 | 133,551 |
| 13:37:42 | card4 `04:00.0` | 1 | 1 | 72,788 |

So: not "traffic" in general — a burst of short requests moved nothing — but **long-context traffic does move
them**, ~1 event per 5 minutes at 4–5 concurrent 60k–170k streams, and 5 of the 6 landed while a 72k–144k-token
prompt was being prefilled. That is the shape you would expect if the trigger is the large TLP/DMA load a big
prefill puts on the link rather than anything about starting a process. Card2's chain went 36 → 38 and card4's
8 → 12 across the hour.

Two more arrived while watching (14:30:39 and 14:32:11, both `04:00.0`, both with 5-7 requests in flight and the
pool over 60%), bringing it to **9 events in ~55 minutes of real load** — and both of those two landed on the
chain of **card4, the card that runs 10 °C hotter than the others on the sensor the Monitor was reading** (edge
78 °C against 66-69 °C, steady rather than climbing across 120 samples — per sensor, its junction is normal and
its **memory** is the hottest thing on the box; see [observe-2026-11-02.md](../observe-2026-11-02.md)). Not evidence of a causal chain, just two more facts about the same card sitting
next to each other.

Still true, and the part that matters for the reset question: **every event is correctable**. `TOTAL_ERR_NONFATAL`
and `TOTAL_ERR_FATAL` are 0 on every device, no link has ever renegotiated below **32.0 GT/s ×16** in ~90 minutes
of sampled history, and the box has been up since 11:07 (3 h 09 m through all of this) after a morning of 15
boots. Correctable-but-repeating on two specific upstream ports is a link that is working but not clean; it is
not yet an explanation of the hard resets.

Two consequences stand:

- The panel is drawn as rows plus an event list, not waveforms. ~1 event per 5 minutes is still an event log, not
  a curve. Rationale and the naming trap (`NonFatalErr` is a *correctable* counter) in [notes/webui.md](../webui.md).
- What counts as out of pattern is now sharper: a correctable during a long prefill is **expected**; what would
  matter is an **uncorrectable** counter, a link that renegotiates, a counter being **zeroed** under us, or an
  event with **no request in flight** while the cards are idle (the box has done all four of those things at a
  reset, and has never done them during serving).

## Continued: [incident-2026-11-03](../incident-2026-11-03/README.md)

The box died again at **02:50 UTC on 11-03**, under two minutes after its last journalled request — a
**242,882-token prompt**, the largest this box has ever completed — with five long requests in flight. No BMC read
for that death yet; the PSU and PDU questions left open above (§5c, §5d) are still the cheapest things to look at.

