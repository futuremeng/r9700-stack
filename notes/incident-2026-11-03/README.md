# notes/incident-2026-11-03 — the box died again at 02:50 UTC, within two minutes of serving the largest prompt this box has ever served

Same machine as [incident-2026-11-02](../incident-2026-11-02/README.md) (`jiulu-9700`, 4× R9700, bare metal).
**Observation only**: nothing on the box was changed, and nothing here was read from the BMC — this account cannot
open `/dev/ipmi0`, so the SEL and the PSU registers for this death are still unread.

## What is known without root

- `who -b` → **system boot 2026-11-03 02:50**. The previous boot (11-02 11:07) had run **15.7 h**, the longest
  stretch of that 15-boot day.
- At 02:52, `:8080` and `:8099` both refused connections and no vLLM process existed. **Nobody has brought the
  serve back up**; the box has been up and empty since.
- The request journal (`~/logs/requests-8080.jsonl`) survived, as it was built to: **1,435 rows** from the launch
  that started 11-02 11:2x under boot id `1793624132-16297`, one from the earlier attempt, and the file's last
  fsynced write is at **02:48**.

## What the journal says about the last minute of that launch

```
02:47:02  prompt   76,708  TTFT   8.9 s  wall  26.0 s   finished ~02:47:28
02:47:12  prompt  126,178  TTFT  15.7 s  wall  26.4 s   finished ~02:47:38
02:47:29  prompt   77,395  TTFT   9.1 s  wall  54.2 s   finished ~02:48:23
02:47:39  prompt  242,882  TTFT  35.3 s  wall  44.0 s   finished ~02:48:23
02:48:12  prompt   60,133  TTFT  21.8 s (12.3 s of it queue)  wall 10.8 s  finished ~02:48:23
```

Five real requests, all completed, all long — **prompts 60k–242k with four in flight at once** — the last writes
landing at 02:48:2x, and the box rebooting at ~02:50. So the death follows real long-context traffic by
**under two minutes**, on a box whose previous day's PCIe story is: correctable `BadTLP` events at the switch
upstream ports `c4:00.0` and `04:00.0`, **most of them during a long prefill**
([11-02 note, "What moves them"](../incident-2026-11-02/README.md)).

**242,882 prompt tokens is the largest request this box has ever completed** (previous record in the same journal:
240,067). That is a coincidence worth writing down and no more: the request finished cleanly, with its tokens
journalled, and whatever happened next is not visible from here.

## What this is not yet

Not a reproduction, and not evidence of a cause. What is missing is exactly what the last incident note asked for
and still has not got: the SEL for this boot (`sel elist`, and whether it filled again and dropped records), the
AER counters (`aer_dev_correctable` at the four upstream ports — they were **39 / 12 / 0 / 0** at the end of 11-02,
read through the Monitor's PCIe panel), and the PSU question left open there: **two 2600 W supplies, both on one
PDU, one of them outputting 0 W with an input-side reading of 0 V** (§5c, §5d). A `Redundancy Lost` record would
say whether the BMC considers that pair healthy; nothing in the old 3,639 entries said so either way.

## What the journal bought, in the meantime

The row redesign checked in the same hour (see [webui.md](webui.md), "What a request row leads with") was verified
against these exact rows — prefill and decode speed, the request's own token counts, `shared ×N`, and the corrected
wall-clock sum — rendered offline with `tools/monitor-fixture.py` while the box was down. That is the point of both:
the interesting rows are gone from the serve's memory a minute after they finish, and gone entirely when the box is.

## Open, in order

1. Bring the serve back or not: the standing recipe is
   `MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same bash serve/native-flashnext.sh`, and the journal means
   the Monitor would come back showing this launch's 1,435 rows. **Not done: the platform question is the open one,
   not the serve.**
2. Operator reads, when they get to the box: `ipmitool -I open sel elist` (and `sel save` + `sel clear` before the
   next one, as 11-02 §5 recommends — the SEL was at 0 bytes free), `ipmitool sdr elist | grep -i psu` under load,
   `ipmitool fru print` for the two supplies, and the physical PDU/cord question.
3. After a relaunch, the Monitor's PCIe panel gives timestamped AER events per chain, and
   `tools/watch-live.py` samples live traffic without adding any.

## Relaunched at 03:00, and what the counters said ten minutes later (the box had been idle for 10 min)

`MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same bash serve/native-flashnext.sh` — `/health` 200 by
~03:09, no request sent by me. The journal did what it was built for: the Monitor came back holding **100 rows
from the dead launch**, newest first, and the newest is the **242,882-token request** (`shared ×2.1`, first token
35.3 s). `live.preemptions_total` is present on this build (0), and `shared_decode` rides every row.

### The AER counters are per-boot, so yesterday's numbers are gone — and today's are worse than any launch on 11-02

The kernel zeroes these at enumeration, so "39 / 12 / 0 / 0" from the end of 11-02 is history, not a baseline.
Read at ~03:10, i.e. after the launch (110 GiB of weights, graph capture, warmup) and with **no user traffic yet**
(`since.requests` = 0, cards at 3% busy, 16-20 W, sclk 41 MHz):

| chain | correctable total | where | named | link |
|---|---:|---|---|---|
| **card2 `c6:00.0`** | **144** | `c4:00.0` upstream 80, `c0:03.1` root 62, endpoint + downstream 1 each | **BadTLP 77**, Timeout 65, NonFatalErr 1 | **16.0 GT/s ×16** at `c4:00.0` and `c0:03.1` |
| **card4 `06:00.0`** | 17 | `04:00.0` upstream 11, `00:03.1` root 4, endpoint + downstream 1 each | BadTLP 9, BadDLLP 1, Timeout 5 | 32.0 GT/s ×16 |
| card1 `c3:00.0` | 0 | — | — | 32.0 GT/s ×16 |
| card3 `03:00.0` | 0 | — | — | 32.0 GT/s ×16 |

**card2's switch uplink has renegotiated down to Gen4.** The endpoint (`c6:00.0`) and the switch downstream port
still read 32.0 GT/s; the two ends of the **switch-uplink** — `c4:00.0` and the root port `c0:03.1` — read 16.0.
So it is not the card's link that dropped, it is the hop that carries its traffic to the CPU, and it is the same
port that accounted for most of 11-02's events. Per the out-of-pattern list written on 11-02, a link renegotiated
below 32.0 GT/s ×16 **is** one of the things that was supposed to make us look — and this is the first time it has
been seen.

Still no claim of cause. Two things are observed together: a renegotiated uplink and 144 correctable events on
that chain, and both appeared during a launch rather than during idle — 45 s later, with the cards parked at 3%
busy, the totals had not moved (144 → 144, 17 → 17) and the sidecar logged no events.

### The panel's own semantics, fixed while reading this

`degraded` (link below its own maximum) used to colour a chain red whatever the box was doing, and both idle
states of this machine trip it: with **no context open** the GPUs sit in **D3hot and every chain reads
2.5 GT/s ×16** (measured 03:00, before the relaunch), and with **the serve up but nothing served** the links sit at
Gen4 — 16 GT/s on the c6 chain at 3% busy. So the page now says which of the three it is looking at:
`parked` (D3hot), `idle` (D0, nothing running — the alarm waits for work), or below-maximum **while work is
running**, which is the only case that turns to danger. Tests cover all three. The in-serve Monitor on `:8080`
predates this change and will keep colouring those chains red until the next restart; the `:8099` sidecar has it.

## Serve restarted at 03:21 to pick up the panel fix — and the counters said something new

Health 200 at **03:23:45** (~130 s, caches warm). New build confirmed live on `:8080`: `parked` on every chain, the
`idle` / `parked` / while-working wording, journal restored (**100 rows** from the dead launch, newest still the
242,882-token request), new boot id `1793676091-8252`, `preemptions_total` present, `shared_decode` on 100/100 rows.

**My mistake first:** I ran the relaunch with `STOP=1`, and `STOP=1` in this script means *stop and exit* — so the
box sat with no serve from **03:10 to 03:21** while I waited for a health that was never coming. The script now says
so out loud (`STOP=1 stops and exits: nothing was launched. Rerun the same command without STOP=1`). Ten minutes of
downtime, on me, and it is a serving box.

### A serve restart is a clean AER experiment, and it localizes the fault to one switch group

A serve restart does **not** re-train PCIe links and does **not** zero the counters (only boot enumeration does). So
the before/after is a clean delta for one launch of the model:

| chain | before (03:10) | after (03:23) | delta | link |
|---|---:|---:|---:|---|
| card4 `06:00.0 → 05:00.0 → 04:00.0 → **00:03.1**` | 17 | **28** | **+11** | 32.0 GT/s ×16 |
| card2 `c6:00.0 → c5:00.0 → c4:00.0 → **c0:03.1**` | 144 | 144 | 0 | **16.0 GT/s** at the uplink |
| card1 `c3:00.0 → … → **c0:01.1**` | 0 | 0 | 0 | 32.0 GT/s ×16 |
| card3 `03:00.0 → … → **00:01.1**` | 0 | 0 | 0 | 32.0 GT/s ×16 |

Two things fall out, and both are corrections to what I wrote above:

- **Every AER event this box has ever counted sits on the two chains whose root port is `xxx:03.1`** — `00:03.1`
  (card4) and `c0:03.1` (card2). The two chains under `00:01.1` and `c0:01.1` (card3, card1) have counted **zero**
  across 11-02's serving day, the reset storm, and both of today's launches. That is a hardware-localizing pattern
  worth handing to the operator, not a general "the PCIe is unhappy" statement.
- **The 144 on card2's chain came with the cold boot, not with a launch.** I wrote above that the events "accrued
  during the launch" — that reading is now split: this launch added **zero** there, while it added 11 on card4's
  chain. So `c4:00.0`'s 77 `BadTLP` are tied to the 02:50 boot and its link training (which is also what left that
  uplink at Gen4), not to model loading.

### A bug the restart exposed in my own change, and why the tests missed it

The `busyNow` the PCIe note uses was declared *below* the string that uses it. `const` in a temporal dead zone is a
`ReferenceError`, and the page's poll loop wraps its render in `try/catch`, so the whole thing surfaced as **the PCIe
rows and the request rows silently not rendering** — an empty section, not an error. String-presence tests passed
while the page was broken. Fixed three ways: the declaration moved above its use; a test now asserts the *order*
(not just the presence) of that declaration; and `tests/render-monitor.mjs` prints the page's notice bar and exits
non-zero when the page reported an error or rendered no rows, so a swallowed render error can never again look like
an empty panel. The live page now renders 4 PCIe rows — `16.0 GT/s ×16 (idle, max ×16)` in muted type, with the note
`links sag while nothing runs: that is power management, watch them under load` — and 100 request rows.

## 60 s of link-state sampling says the Gen4 uplink is not power management

Sampled `current_link_speed` / `current_link_width` every 2 s for 60 s with the serve up and nothing served
(cards at 3% busy, sclk 41 MHz, mclk 96 MHz):

| link | state over 60 s |
|---|---|
| `c6:00.0` endpoint (card2) | 32.0 GT/s ×16, one distinct value |
| `c5:00.0` switch downstream | 32.0 GT/s ×16 |
| **`c4:00.0` switch upstream** | **16.0 GT/s ×16** |
| **`c0:03.1` root port** | **16.0 GT/s ×16** |
| `04:00.0` switch upstream (card4, same topology) | 32.0 GT/s ×16 |

That kills the easy reading. A power-management sag would take the links down **together** (and the parked-box
case really does that: all four chains at 2.5 GT/s with no context open), and card4's chain has the same shape and
holds Gen5 at the same instant. This is one hop — `c4:00.0 ↔ c0:03.1` — pinned at half speed since the 02:50 boot,
which is also the hop whose upstream port carries 77 of the 144 `BadTLP` and the chain whose root port carries 65
`Timeout`. The panel therefore no longer explains it away: the muted label stays (`idle, max ×16`) but the box note
now reads *"a link is below its maximum with nothing running: parked chains always read that way · if it is still
short while cards work, that is a link fault"* — a test, not a verdict. The remaining open test is the loaded one:
whether that hop renegotiates to 32.0 GT/s when card2 actually works.

## The loaded test, as far as a serving box will allow: one request, 10 Hz around it

`tools/probe-load-link.py` — one chat request (59 prompt tokens, `max_tokens=1`), sysfs sampled at 10 Hz, bracketed
by a quiet second on each side. Sent at **03:33:54**, with `running 0 / waiting 0 / preemptions 0` confirmed first so
it did not contaminate anyone's concurrency. Engine-side wall **0.049 s**; HTTP round trip 0.09 s; the reply carried
no content because the single allowed token went to a reasoning token (`reasoning_tokens: 1`).

**This launch now contains exactly one synthetic row**, `chatcmpl-8fba340fe00c9784` (prompt 59, completion 1) at
03:33:54. Anyone averaging this launch's numbers should know it is mine.

| what | result |
|---|---|
| `c4:00.0` (switch upstream) | **16.0 GT/s ×16 in 21/21 samples** |
| `c0:03.1` (root port) | **16.0 GT/s ×16 in 21/21 samples** |
| the other 14 nodes on all four chains | 32.0 GT/s ×16 in 21/21 samples |
| AER correctable | **+0** on every node |
| `card1` `gpu_busy_percent` | 3 → 4 → **43** (the cards really stepped) |

What that supports: the two shorted nodes **did not attempt to renegotiate around activity**, and they stayed at half
speed while the cards demonstrably worked and while structurally identical nodes on the other chains held Gen5. What
it does **not** support: any conclusion that the hop is not marginal. A 59-token request moves a few kilobytes
host→device; inference streams its weights from HBM, so PCIe was barely asked for anything. The class of traffic that
did move counters on 11-02 is a **long-prompt prefill** (`c4:00.0` and `04:00.0` were exactly where those events
landed), and that probe costs ~20 s of box time plus one more journalled row — a decision for the operator, not for me
unilaterally.

# Correcting the premise of this whole file: the deaths DO leave a record, and it names a CPU core

I wrote above that this death had no BMC read because the account cannot open `/dev/ipmi0`. That was true and
irrelevant: **the record was sitting in a log this account can read.** The firmware writes one error record
into ACPI **BERT** when the crash happens, and the kernel replays it into the log at the start of the next boot.
`journalctl -k` needs no root here (measured). Everything below comes from that, and it is the strongest finding
of the entire incident.

## 18 of 18 boots in the journal carry one, and they are all the same fault

`journalctl --list-boots` holds 19 boots from 2026-11-01 20:36 to now. **Every boot after the first carries
exactly one `BERT: Total records found: 1` / `BERT: Error records from previous boot:`** — so the box has died
18 times in the two days the journal covers, not the "15 boots on 11-02" I had counted, and the deaths begin
**2026-11-01 20:36**, a day earlier than the incident note assumed.

Each record has the same shape (verbatim, the one that ended the 15.7 h launch at 02:50):

```text
BERT: Error records from previous boot:
[Hardware Error]: event severity: fatal
[Hardware Error]:  Error 0, type: fatal
[Hardware Error]:  fru_text: ProcessorError
mce: [Hardware Error]: CPU 17: Machine Check: 0 Bank 5: aea0000001000108
mce: [Hardware Error]: TSC 0
ADDR 142d6f3
MISC d0140ff600000000
PPIN 2b61290dcb2002b SYND 4d000000 IPID 500b024029a00
mce: [Hardware Error]: PROCESSOR 2:a10f11 TIME 1793674220 SOCKET 0 APIC 82 microcode a101158
```

- **`fru_text: ProcessorError`, `event severity: fatal`** — the firmware's own summary. It is a processor fault.
- **`Machine Check ... Bank 5`**, status `aea0000001000108` or `aea0000000000108`: MCi_STATUS bit 63 (VAL) set,
  **bit 61 (UC) set — uncorrectable**, ADDRV and MISCV set, PCC (bit 56) clear.
- **`IPID 500b024029a00`** resolves through the kernel's own table (`arch/x86/kernel/cpu/mce/amd.c`,
  `smca_hwid_mcatypes`: high half `0x000500b0` → hwid `0xB0`, mcatype `0x5`) to **`SMCA_EX`**, which the kernel
  names **`execution_unit`** — a core's execution logic. **Not** the memory controller (`0x96`), **not** PCIe
  (`0x46`), **not** NBIO/fabric (`0x18`/`0x80`).
- **`CPU 17` (APIC 82) or `CPU 113` (APIC 83)**, and sysfs says those two are the **SMT siblings of one physical
  core**: `/sys/devices/system/cpu/cpu17/topology/thread_siblings_list = 17,113`, `core_id = 65`, package 0.
  So 18 records do not name 18 victims or even two CPUs — they name **one core**, thread 0 or thread 1.
- Same `PPIN 2b61290dcb2002b` (one physical socket), same `microcode a101158`, CPU is **EPYC 9654** (96c/192t).
- **`TIME <epoch>` lands 17–18 s before the first kernel line of the next boot** in every single case — the BMC's
  reset-and-POST latency. That is what makes the record belong to *this* death rather than a stale one.

The 18 death times, from the records themselves: 2026-11-01 20:36:21, 21:03:39, 23:52:18, 2026-11-02 00:04:37,
00:16:55, 00:27:11, 01:07:19, 01:18:29, 01:30:09, 03:00:24, 06:34:49, 09:11:14, 09:41:07, 10:30:57, 10:36:54,
11:00:29, 11:07:20, and 2026-11-03 02:50:20. Gaps from 11 minutes to 5.3 hours.

## What this changes, and what it does not

**Changes:** the deaths are not silent, not unlogged, and not first-and-foremost a PCIe event. The AER correctables
and the Gen4 uplink on card2's chain are real and worth fixing, but they are a *chronic side condition* — the
machine-check record names the CPU's execution unit, 18 times out of 18, always the same core.

**Does not change, but sharpens:** the electrical question. A feed droop CAN produce a core-level uncorrectable
error, so the PSU/PDU gap still matters — but a droop does not pick the same two threads of the same core 18 times
while leaving 190 other threads alone. Core-locality points at that core (or its rail/CCX domain) first, and at the
platform second.

**What the microcode lookup says.** The box runs **0x0a101158**. Fetching `amd-ucode/microcode_amd_fam19h.bin`
from linux-firmware today, the Genoa (`19h`) revisions present upstream are **0x0a101148, 0x0a101153, 0x0a101158,
0x0a101248, 0x0a10124e, 0x0a101253** — so the box is **three revisions behind** what upstream carries. I could not
reach AMD's errata documents from this box (search engines unreachable; only direct fetches), so I am *not* claiming
one of those revisions fixes this. It is a cheap, in-scope thing for the operator to rule in or out.

## What I would do with this, in order

1. **Keep the records.** BERT holds exactly **one** record, so every death destroys the previous evidence, and the
   journal's own power-loss records are already full (`0 bytes free`). Install `rasdaemon` (it decodes and stores
   CPER/MCE records in a database), and `ipmitool sel save` + `sel clear` on the BMC side.
2. **BIOS + microcode update**, then re-observe. Also worth reading while there: AMD's Genoa errata list for
   execution-unit/EX MCE entries against `0x0a101253`.
3. **Try to reproduce it deliberately, cheaply, with no GPUs involved**: hammer that core alone —
   `taskset -c 17,113 <a busy loop>` for an hour, then with GPU load alongside. If the machine check comes back
   with the same IPID and the same core while the GPUs are idle, the platform question is answered in one evening.
4. **Interim containment for a serving box** (operator/root, and honest about its status as a workaround): disable
   that physical core in the BIOS (or offline CPU 17/113 — a core that is not executing cannot take an EX fault).
   Losing 2 of 192 threads costs this workload nothing. If the deaths then move to a different core, that is itself
   the answer: platform, not core.
5. Ask the operator for the BMC's own view now that we know what to ask for: `sel elist` around those 18 timestamps
   should contain processor MCE records, not power ones.

## The Monitor now says this out loud

`serve/webui.py` gained `read_death()` (the record for the boot before this one, 0.4 s, read once), `death_history()`
(one kernel-log read per boot the journal holds, ~0.3 s each, in a thread — **not** `-b all`, which measures 55 s
against this box's 517 MB journal), `smca_bank()` (the IPID → block-name lookup) and `core_of()` (logical CPU →
physical core and SMT siblings). The payload carries `last_death` and `deaths`; the Monitor prints a red line under
the journal line: what ended the previous boot, which block, which core, when, and how many such records the journal
still holds. Verified live: `12 records across the last 12 boots`, `execution unit ×12`, `core 65 ×12`,
`fatal ×12`. `tools/journal-pcie.py` is the same questions from a terminal, per boot, with the AER counts beside
them.

## Running the discriminating test: hammer core 65, GPUs idle (`tools/core-hammer.c`)

The cheapest experiment that can actually settle this is to ask **one core** for integer work and leave everything
else alone. `tools/core-hammer.c` does exactly that: it pins the whole process to a sibling pair (default
`17,113` = physical core 65) and runs eight independent integer chains — multiply, xor-shift, add, rotate,
compare-select — no FP, no allocation, no syscalls in the loop. Integer-only on purpose: the SMCA bank in every
record is `EX` (fixed-point execution), and each bank names its own block, so mixing in FP (`FP` bank) or memory
pressure (`UMC`/LLC) would blur which unit is being asked. Verified placement on this box: `affinity list: 17,113`
and `ps -L` shows the worker threads resident on 17 and 113.

Its heartbeat is **flushed and fsynced once a second** (`--log`), because the run's success condition is the box
resetting underneath it: the last fsynced line is then the whole record of how long it ran, on which CPUs, and how
far the arithmetic got (`~490 M iterations per second` across the pair at `-O2`). `--stop FILE` lets it be stopped
without `pkill`.

Three outcomes, each worth knowing:

| what happens | what it means |
|---|---|
| same machine check, `IPID ...029a00`, CPU 17 or 113 | core 65 (or its rail/CCX domain) is the fault: stop serving on this box until it is swapped or the core is disabled |
| a machine check naming a **different** core | the platform, not the core: voltage/cooling/microcode territory, and the PSU question goes back up the list |
| nothing after the full window | the hammer is not the trigger: what kills the box needs the load it has not been given yet (GPU + PCIe + CPU together), and the next phase is the same hammer *with* serving traffic |

Preconditions asked of the operator, in order: `ipmitool sel save` and then `ipmitool sel clear` — the SEL is
full (`0 bytes free`), so without clearing it a new fault may leave no BMC record at all. BERT still holds one
record regardless, and the Monitor now prints it, so the test is interpretable even if the BMC side is not.

## Is the operator's forced power cycle what wrote those records? (asked 2026-11-03 04:16)

A fair question, and it deserves the register-level answer rather than my inference. Two things had to be sorted:
what a forced power operation can and cannot write, and why the BMC shows nothing.

### The 18 records, field by field — this is the discriminator

Pulled per boot with the Monitor's own parser (`parse_death` over `journalctl -k -b <id>`):

| boot | death (UTC) | CPU / APIC | MCi_STATUS | ADDR |
|---|---|---|---|---|
| -18 | 11-01 20:36:21 | 113 / 83 | `aea0000001000108` | `1ffffffa2c9ce8b` |
| -17 | 11-01 21:03:39 | 17 / 82 | `aea0000000000108` | `7034db7a1efa` |
| -16 | 11-01 23:52:18 | 17 / 82 | `aea0000001000108` | `1ffffffa249ce8b` |
| -14 | 11-02 00:04:37 | 17 / 82 | `aea0000001000108` | `142d5d3` |
| -13 | 11-02 00:16:55 | 113 / 83 | `aea0000001000108` | `1ffffff8b09ce8b` |
| -12 | 11-02 00:27:11 | 17 / 82 | `aea0000000000108` | `7e46a5fa1efa` |
| -11 | 11-02 01:07:19 | 17 / 82 | `aea0000000000108` | — |
| -10 | 11-02 01:18:29 | 17 / 82 | `aea0000000000108` | — |
| -9 | 11-02 01:30:09 | 17 / 82 | `aea0000001000108` | `1ffffffb6a9ce8b` |
| -8 | 11-02 03:00:24 | 113 / 83 | `aea0000001000108` | — |
| -7 | 11-02 06:34:49 | 17 / 82 | `aea0000000000108` | — |
| -6 | 11-02 09:11:14 | 17 / 82 | `aea0000000000108` | — |
| -5 | 11-02 09:41:07 | 17 / 82 | `aea0000000000108` | — |
| -4 | 11-02 10:30:57 | 17 / 82 | `aea0000000000108` | — |
| -3 | 11-02 10:36:54 | 17 / 82 | `aea0000000000108` | `1654624` |
| -2 | 11-02 11:00:29 | 113 / 83 | `aea0000000000108` | `78d332da1efa` |
| -1 | 11-02 11:07:20 | 17 / 82 | `aea0000000000108` | `76961eba1efa` |
| 0 | 11-03 02:50:20 | 17 / 82 | `aea0000001000108` | `142d6f3` |

Constant across all 18: `MISC d0140ff600000000`, `PPIN 2b61290dcb2002b`, `SYND 4d000000`, bank 5,
`severity: fatal`, `fru_text: ProcessorError`, core 65. Alternating with the thread: IPID `...29a00` on CPU 17
(14 records) and `...29b00` on CPU 113 (4 records).

- **What a forced power operation cannot write.** A hard power cut clears the MCA MSRs at power-on reset, and the
  BIOS is not executing at the instant the button is held, so nothing is around to compose a CPER carrying valid
  `MCi_STATUS` (VAL=1, UC=1), `MCA_IPID`, `SYND` and `ADDR`. What a forced operation *does* leave is a different
  SEL event: chassis control / power button / reset.
- **The one mechanism that is genuinely possible** — and the honest half of the answer: a **warm** reset (BMC
  "reset", not a power cycle) leaves MCA MSRs intact, so banks left unconsumed by a previous fault *can* be logged
  by the next POST as "Error records from previous boot". If that were what we are seeing, the records would be
  **re-loggings of one fault** — therefore byte-identical.
- **They are not identical.** Every ADDR is different (`1ffffffa2c9ce8b`, `7034db7a1efa`, `142d5d3`, `1654624`,
  `142d6f3`, …), the failing **thread alternates** (IPID/APIC pair flips), and MCi_STATUS comes in two variants.
  A stale bank re-logged 18 times cannot change its own address 18 times. So the register content says 18 distinct
  faults on one core, whatever the power button was doing.
- **And a correction to my own reasoning:** I wrote that the record `TIME` sitting 17–18 s before the kernel's first
  line ties each record to its own death. That is too strong — a record written at the crash and a record written by
  POST of the next boot both land there. The `TIME` field is not the discriminator; the register content above is.

What would settle it outright is running anyway: `tools/core-hammer.c` loading **only** core 65 with the GPUs idle
and nobody touching the box. Started 04:00:56, so if a fatal record appears it can only be a fresh one.

### Why the BMC shows no hardware error events

Two measured facts, and together they are enough:

1. **The SEL is full.** The 11-02 capture read `0 bytes free` and the oldest entries already rolled. A full SEL
   cannot record new events, so "I see nothing" is expected.
2. **The firmware-first path to the OS is demonstrably alive right now.** This boot carries **42** APEI GHES
   records — all `severity: corrected`, `fru_text: PcieError` — timestamped 02:50:38, 02:50:47, 03:01:12, 03:01:19,
   03:01:24, 03:23:12, 03:23:29, 03:53:04, 04:06:07 …, naming `04:00.0` (AMD `1002:1478`, the on-card switch
   upstream port) and `00:03.1` (AMD `1022:14a5` root port). Whether those mirror into IPMI SEL is
   BIOS/vendor-dependent, and many AMD server BIOSes report to GHES/WHEA without adding SEL events at all.

Two of my own earlier statements are corrected by this: I had said the sysfs AER counters "did not move while idle".
Over a 1.4 h boot they move constantly — **42 corrected records**, including **03:53 and 04:06 with no request in
flight** — and the Monitor's own 1 Hz differ saw the 04:06:08 one (`04:00.0 BadTLP +1`, card4's chain 28 → 30).
My 45 s windows just happened to fall in the quiet parts. The PCIe error stream is continuous and idle-capable,
which strengthens the marginal-link reading and removes "traffic causes it" from the table as the whole story.

### What to ask the operator for, now that we know what to look for

```bash
sudo ipmitool sel info                       # "Use"/percent and overflow: is it still full?
sudo ipmitool sel elist | tail -40           # what it does hold
sudo ipmitool sel elist | grep -iE "chassis|power|reset"   # the times YOU touched it
sudo ipmitool sel save /root/sel-$(date -u +%F-T%H%MZ).txt && sudo ipmitool sel clear
sudo ipmitool sdr elist | grep -iE "psu|power|cpu|overflow"
sudo apt-get install -y rasdaemon && sudo systemctl enable --now rasdaemon   # decoded, kept: ras-mc-ctl --summary
```

The `grep -iE "chassis|power|reset"` line is the direct test of the hypothesis: those SEL records mark forced power
operations, and lining them up against the 18 death times above answers it without any interpretation from me.

## What the operator's IPMI run established (2026-11-03 04:26) — including one claim of mine that was wrong

### `sel info`: the BMC **was** recording at the moment of death, and the SEL was unreadable, so its contents are gone

```text
Entries : 3639   Free Space : 0 bytes   Percent Used : 100%   Overflow : false
Last Add Time : 11/03/2026 02:50:11 AM UTC
```

- **`Last Add Time 02:50:11`** is 9 s before the BERT record's `TIME 02:50:20` for the last death. So the BMC added
  a record at the death — we simply cannot see what kind it was: `sel elist` failed with
  `Get SEL Entry 0 command failed: Requested sensor, data, or record not found` and the `sel save` that ran before
  the `clear` therefore wrote nothing usable. **3,639 records are lost.** Noted here rather than hidden: the lesson
  is that a full SEL is not merely "no room for new events" but "the log may stop being readable at all", and
  `Percent Used 100%` with `Overflow: false` is the state to catch early.
- The forward path is now clean: the SEL is empty, so **the next crash's record will be enumerable** —
  `sudo ipmitool sel elist` should work, and it should be read before anything else fills it.

### PSU2 answered: energized, carrying nothing

`PSU2 Power In 16 W / Power Out 0 W` with `PSU1 In 320 W / Out 320 W`. 16 W on the input side means **AC is alive at
PSU2** — so it is not a dead supply and not an AC-less feed: standby-with-AC, exactly the discriminator asked for on
11-02. With both cords on one PDU, the practical redundancy is still absent, but the hardware at PSU2 is not broken.
Rails at that moment: `VCORE0 1.11 V`, `VCORE1 1.12 V`, `VSOC 0.98 V`, `VDDIO 1.14 V`, `CPU1 56 °C`,
`CPU_Power 136 W` (with the core hammer running). All nominal — and all second-cadence sensors, so none of them can
see a millisecond droop; that limitation from 5c still stands.

### My microcode claim was wrong, corrected

I wrote that the box is "three revisions behind what upstream carries". The box's own installed
`/lib/firmware/amd-ucode/microcode_amd_fam19h.bin` is **byte-for-byte the same size as the file I fetched from
linux-firmware and contains the same Genoa revisions**: `0x0a101148, 0x0a101153, 0x0a101158, 0x0a101248, 0x0a10124e,
0x0a101253, 0x0a108109`. What runs is **`0x0a101158` on all 192 threads**, with the kernel saying only
`Current revision: 0x0a101158 / Updated early from: 0x0a101158` — no refusal, no late-load line. So the newer blobs
are on disk but the loader does not select them for this CPUID (AMD gates patches by CPUID/patch-level, and the
BIOS applies its own first). **The lever is the vendor BIOS/AGESA, not `apt install amd64-microcode`.** That also
means the box is not obviously "out of date" on microcode: `0x0a101158` may be the newest patch that exists for this
part. Ask the vendor for the BIOS/AGESA note before treating this as an available fix.

### A locality fact that strengthens the one-core reading

Core 65's CCD (its L3 domain) is **CPUs `16-23,112-119`** — 8 cores, 32 MB L3. Across 18 records, the named thread is
always `17` or `113`: **never any of the six other physical cores on that same CCD**. A droop on the CCD's supply
would be expected to implicate CCD siblings at least sometimes; 18/18 landing on one physical core, alternating
between its two SMT threads, fits "that core" better than "that CCD's rail". Cheap next test, if we want it: hammer a
**sibling** core on the same CCD (say 18/114) — if it can be loaded for the same window with nothing logged, the
core-local reading gains again.

### rasdaemon installed (and why it will not show the 18 deaths)

`rasdaemon` is now installed and enabled. It records from now on — the 18 BERT records were replayed at 02:50:38,
before the daemon existed, so they are **not** in its database. Worth knowing for the next event:

```bash
sudo ras-mc-ctl --summary
sudo ras-mc-ctl --errors                       # decoded records; rasdaemon decodes AMD SMCA banks via extlog
sudo sqlite3 /var/lib/rasdaemon/ras_mc_event.db 'select timestamp,error_type,mcistatus,mciaddr from extlog;'
```

