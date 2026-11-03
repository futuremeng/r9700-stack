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

## 04:36 — the serve stopped answering, and that is when the Monitor's own design mattered

```text
04:36  curl http://127.0.0.1:8080/health -> 000 (connection refused); pgrep for `vllm serve` found nothing
04:41  same URL -> HTTP 200, and `ps -o lstart -p 8252` says the process started 03:21:14 -- same launch
       journal grew 1457 -> 1459 rows across the window, so requests were completing
```

Not resolved, and deliberately not explained: the launch survived, kept serving, and was unreachable on its
port for minutes. What is recorded is the design consequence — the in-serve Monitor is on the serve's port, so
it disappears exactly when a person needs it. The standalone sidecar (`:8099`) does not depend on the serve:
its cards, PCIe chains, slots, death record and (attached read-only) request journal all come from sysfs and
the kernel log. See `notes/webui.md` for the banner, the attach flag and the slot map.

Reseat target, from the firmware's slot table: **card2 = `0000:c6:00.0` = physical slot 6** — the chain with the
16.0 GT/s uplink (`c4:00.0 ↔ c0:03.1`) and 144 correctable events this boot. The other three sit in slots 1, 3
and 8. This is the chronic PCIe side condition, *not* the machine-check deaths (those name CPU core 65), and it
is worth fixing on its own merits: `tools/reseat-check.py --save` before power-down, `--diff` after boot.

## The core-65 hammer finished: 45 minutes, 1.32 trillion integer iterations, nothing logged

```text
1793681150 alive pid=13290 cpus=17,113 seconds_left=0 iters=1319559700000 check=bfb368dff735153a
```

`tools/core-hammer.c` ran the full 2,700 s pinned to **CPU 17/113 = physical core 65**, the core all 18
machine-check records name, and ended normally at 04:45:50. `journalctl -k -b 0 --since 04:00` holds **zero**
fatal records and zero machine-check lines; the box and the serve are fine.

What that buys, stated narrowly: **CPU-only integer load on the suspected core does not reproduce the fault in
45 minutes.** It does **not** clear the core. Every real death happened while the box was doing GPU work —
long-context prefill, four cards at their power cap, PCIe traffic — and during the hammer the cards idled at
a few percent. So the surviving discriminator still needs the platform load in it: the same hammer **while the
serve is carrying real long-prefill traffic** (or a combined CPU+GPU+PCIe load). Also worth remembering: the
heartbeat log is fsynced every second, so if a future run ends in a reset, the last line is the run record —
that is the whole reason the tool writes it.

## The slot-6 chain is not the only one counting: card4's chain is actively moving while the box is idle

```text
05:0x  slot 8 / card4 chain: 56 -> 59 -> 63 correctable since boot, and the panel's event list is live:
       04:51:38 04:00.0 BadTLP +1, 04:51:49 04:00.0 BadTLP +1, 04:52:06 04:00.0 BadTLP +1
       with busy% a few percent and no request in flight
```

So two chains have correctable traffic, and the more active one right now is **card4 (slot 8)** — the one whose
memory also runs hottest — not slot 6, which sits at 144 accumulated and quiet. Both are the chronic side
condition, not the machine-check deaths. Reseating slot 6 is still the action that targets the downgraded
uplink; it will not address a chain that trains at full speed and still racks up `BadTLP`.

## The two error chains are the riser pair, and the split is clean (2026-11-03 05:09)

The operator says slot 6 and slot 8 sit on one extension board. The counter layout agrees with that, and says
more than the totals do:

| chain | endpoint | downstream | **upstream port (on the board)** | **root port** | link |
|---|---|---|---|---|---|
| slot 6 / card2 | 1 (NonFatalErr) | 1 | **80 — BadTLP 77, Timeout 3** | **62 — Replay Timer Timeout** | **16.0 GT/s ×16** |
| slot 8 / card4 | 1 | 1 | **54 — BadTLP 50, BadDLLP 2, Timeout 2** | **7 — Timeout** | 32.0 GT/s ×16 |
| slot 3 / card1 | 0 | 0 | 0 | 0 | 32.0 GT/s ×16 |
| slot 1 / card3 | 0 | 0 | 0 | 0 | 32.0 GT/s ×16 |

- **Every counted event on this box lives in the two `:03.1` root-port chains** (`c0:03.1`, `00:03.1`); the two
  cards on `01.1` root ports read **zero**. If the board is fed by those two root ports, the grouping is exact.
- Bad packets are counted at the **switch upstream port** — the device on the board, on the link that crosses the
  connector — while the **root ports count replays** (`Timeout` 62 and 7). Same physical link, seen from both
  ends: one side receives malformed TLPs, the other is retransmitting.
- Rate, same boot, 19.7 min idle: **slot 8 +0.15/min, slot 6 +0/min**, slots 1/3 zero. So slot 6 accumulated its
  144 earlier (and keeps the downgraded link) but is currently quiet; slot 8 is the one still talking badly.

**Decision recorded: do not clear counters before the reseat.** AER counters are per boot and zeroed at
enumeration, and a reseat means a power cycle — the reboot is the clear. Clearing now would only destroy the
per-node signature above, which is the thing the post-reseat comparison needs. What was saved instead:
`~/logs/reseat-before-20261103T0505Z.json` (now including the named per-node split), plus `rasdaemon` recording
timestamped per-device events from now on, plus the Monitor's per-poll deltas and event timestamps.

## Serve stopped on purpose, 05:12 UTC — the first clean stop of this launch

```text
STOP=1 bash serve/native-flashnext.sh   ->  leftover vLLM processes after STOP: 0
:8080/health                            ->  connection refused (as intended)
four cards                              ->  busy 0%, VRAM 0.056 / 31.86 GiB each (released)
journal                                 ->  1,480 rows, unchanged by the stop
last request (seq 1480)                 ->  142,505 prompt tokens, 778 generated,
                                            first token 18.50 s, all in 23.28 s, completed
AER counters at the stop                ->  slot 6 chain 144 corr, slot 8 chain 63 corr,
                                            slots 1 and 3 zero
```

Nothing was in flight when it stopped (`running 0 / waiting 0`), so no request was cut. The launch that ran from
03:21:14 to 05:12:00 — 1 h 51 min, 1,480 requests, no death — is the longest this box has served since the
18 machine-check deaths started on 11-01, and it also ended without a crash for the first time in that stretch.

**What stayed up, and why that is the point.** The standalone Monitor on `:8099` never notices a serve death as
a blank page: it now reads `reachable: false / Connection refused`, prints the command that would bring this
launch back (`MAXLEN=262144 NSEQ=32 NBT=8192 PREFIX_CACHE=0 WEBUI=same bash serve/native-flashnext.sh`, taken
from the launch's own meta file), keeps the 100 attached journal rows, keeps the death record (execution unit
×12), and keeps reading 4 cards and 4 PCIe chains — all of it from sysfs and the kernel log.

**This is the right moment for the reseat**: the box is idle, no user is connected, and the next boot zeroes the
AER counters on its own — which is the "clear" we decided not to do by hand, so the post-reseat comparison starts
from a clean boot with the saved signature in `~/logs/reseat-before-20261103T0505Z.json` as the reference.


---

# The correction: the deaths are not the link. One core's L2 is producing fatal machine checks.

Written later the same day, after `./setup.sh` usage forced the question "how did that launch die?" and the
answer was not in the boot where the launch stopped. **This section supersedes the framing above**, which
attributed the deaths to the riser chains. The chains are a real and separate problem (§4) — they are not what
kills the box.

## The record lives in the NEXT boot

`journalctl --list-boots` on this box: **33 boots** between 2026-10-31 06:20 and 2026-11-03 06:30 — 32 boot
boundaries, of which **20 open with a replayed hardware-error record** carrying the crash's own `TIME`. That is
where a death's record is: the firmware keeps it (BERT/GHES) and hands it to the OS at the start of the boot
*after* the crash. Looking at the dying boot finds nothing, which is exactly how
[incident-2026-11-02](../incident-2026-11-02/idle-deaths.txt) came to conclude "no AER, no MCE, nothing in the
log". The other 13 boots left no record at all — and the firmware keeps only **one** (`BERT: Total records found:
1`), so the OS journal of the following boot is the only accumulating archive we have.

## The signature is identical 20 times

```
mce: [Hardware Error]: CPU {17|113}: Machine Check: 0 Bank 5: aea000000{0,1}000108
     TSC 0 ADDR …a1efa MISC d0140ff600000000 PPIN 2b61290dcb2002b
     SYND 4d000000 IPID 500b024029a00 (or …29b00)
     PROCESSOR 2:a10f11 SOCKET 0 APIC 0x82|0x83 microcode a101158
CPER: event severity: fatal | fru_text: ProcessorError | section_type: IA32/X64 processor error
      Error Structure Type: cache error | Uncorrected: true | Processor Context Corrupt: true
```

`MCA_STATUS 0xaea0000001000108` decoded: `VAL=1, OVER=0, UC=1` (uncorrectable), `PCC=1` (processor context
corrupt — not recoverable), `AddrV=1`, **MCACOD 0x0108 = architectural cache error, level 2 → the L2**,
MSCOD 0x0100 (or 0x0000 in the other variant). Bank 5 every time. `SYND` and `IPID` every time.

**cpu17 and cpu113 are the two SMT threads of one physical core**: `thread_siblings_list = 17,113`, `core_id 65`,
`physical_package_id 0`, and `cache/index2/shared_cpu_list = 17,113` on a 1024K L2. So 20 fatal machine checks
in 72 hours were reported by that core and by no other: **15 by cpu17, 5 by cpu113**. Two "different" CPUs was
one core all along — which is why `tools/journal-pcie.py` now prints the cpu number reduced to a core.

Independent of load, which is what separates it from everything else in this note: a boot lived **5 s**
(`-7`), another **12 s** (`-13`); boots with **zero** PCIe AER activity died the same way (`-7`, `-5`, `-13`);
a boot with 1783 AER events died the same way (`-11`). Idle deaths, boot-time deaths, prefill deaths — same bank,
same codes, same core.

## Why the BMC has nothing

The operator confirms the SEL has never carried an event of this class, and that is expected rather than
reassuring: these records reach the OS through **APEI/GHES + BERT** (`HEST corrected error threshold limit: 10`
is in boot `-2`'s log), and this firmware (ASUS **ESC4000A-E12**, board **K14PG-U12**, **BIOS 2501**, microcode
**0xa101158**) does not put GHES-delivered processor errors into the SEL. What the SEL's silence *does* rule out
is a power-side trigger: no power-event record for any of these deaths, so the PSU2-at-0-W redundancy problem
(incident-2026-11-02) is not the mechanism. In-band `ipmitool` still needs a password, so the SEL itself is read
by the operator, not from here.

## What was done, and what it predicts

At 06:28 UTC the operator took the two threads offline (`/proc/cpuinfo` 190 of 192 threads). Persisted so a
reset cannot quietly undo it: `host/offline-core.cpus` (the pairing and the reason), `host/r9700-offline-core.sh`
(refuses a pair that is not one core's SMT siblings; snapshots the sysfs topology first, because sysfs drops a
cpu's topology the moment it goes offline and the evidence would be lost), `host/r9700-offline-core.service`
(re-applies it at boot — there is no x86 cmdline that offlines a named core), and `setup.py` step 1 now reports
the state on every run, warning if the sick core is back in service.

The hypothesis this makes testable, and the only honest way to state it: **if the fatal MCA stops appearing in
the replayed records, the fault is confined to that core's L2 and the box is usable while it waits for an
RMA**; if deaths continue with a different bank or a different core, it is package-level and offlining bought
nothing. Neither is claimed here yet — the first data point is the next boot.

## §4 kept, because it is real but separate: the riser chains

Reading the counters properly (the tool's BDF regex used decimal buses, so **`0000:c4:00.0` — the busiest port
on this box — had never been counted**; fixed, `tests/test_journal_mca.py`) for boot `5ec7e479`:
`c4:00.0` BadTLP 49 + BadDLLP 1, root port `c0:03.1` Timeout 26 — kernel named lines 52 BadTLP / 28 Timeout —
and **every one of them inside 06:02–06:05**, the window in which `vllm serve` read 120.3 GB of weights out of
host RAM and pushed them across slot 6's riser. Idle afterwards: **+1 event in 5.7 min**. Each event is delivered
twice (`pcieport … aer_status:` from the port driver, `aer_cor_status:` inside a CPER block from firmware-first:
87 and 92 lines respectively), which is why block and event counts differ. sysfs per-boot totals for the boot:
slot 6 chain 83→85 correctable, slot 8 chain 6, slot 1 and slot 3 zero, **no uncorrectable on any link** — the
fatal errors are CPU-side, in a different reporting path.

Totals are per boot and zeroed at enumeration, so compare in-boot rates (`tools/reseat-check.py --save/--diff`);
`~/aer/watch-*.json`, snapshotted every 5 min by the `aer-watch` tmux session, is the running series.

# First data point: the deliberate reboot, 06:49:06 UTC

The units were installed and enabled before the reboot (verified: `is-enabled` = `enabled` for both, wants links
present). The reboot was graceful, and the protocol's clean branch is what happened.

**The boot-time workaround ran, on the real box, for the first time.** `journalctl -b -u r9700-offline-core.service`:
at **06:49:24 — 18 s after boot** — `cpu 17 113 = core 65 of package 0 (siblings 17,113, L2 1024K)`, `core 65
taken out of use: 190 of 192 threads online`. Before this run the pairing guard and the `echo 0` had literally
never executed (the manual 06:28 run took the already-offline early exit), so this was their first execution and
it happened before anything heavy started.

**No machine check.** Boot 0 presents **no BERT record at all** (`~/aer/mca-archive.jsonl` newest line: `mca
None, replayed_at [], bert None`): the 06:00:06 record was consumed by the graceful reboot. That is an
instrumentation gain by itself — from now on, a record whose TIME is recent cannot be confused with the old one.
`journal-pcie.py 1` over the whole boot including the model load: CPER severity `corrected` only, `fru_text
PcieError` only, and the single MCE line is `HEST corrected error threshold limit: 10` (the GHES threshold
notice, not a machine check).

**It survived the worst window.** The serve came up in **150 s** and generated a real completion afterwards. The
weight-load window — when 20 of 33 previous boots' faults were being produced, and the one this box dies in —
passed with the sick core out of service.

**The PCIe half, measured off counters that enumeration had zeroed** (`--save` at 06:54, `--diff` 3.1 min later,
during the load): slot 6 chain **+129 correctable in 3.1 min = 41.9/min**, root port `c0:03.1` Timeout 1 → 50,
`c4:00.0` BadTLP 67 + Timeout 7; **slots 1, 3 and 8: +0**. Same load correlation as the previous boot. Offlining
a CPU core changed nothing on the PCIe path — which is the negative control that says these really are two
independent faults, and it also means the riser problem is now the only PCIe problem left to chase, cleanly
measured from zero.

**What this does not establish.** One boot without a fatal MCA. The measured cadence before the change was 20
fatal events in 72 h (~one per 3.6 h), and this boot is ~8 minutes old, so the verdict is hours away, not here.
And the window the workaround cannot cover: **the 18 s between power-on and the unit** had all 192 threads
schedulable, and boot `-7` in the journal died 5 s after boot. A firmware/BIOS-level core disable is the durable
form of this workaround; worth putting to ASUS/AMD along with the MCACOD 0x0108 bank-5 evidence.

## A second symptom on the same chain: the slot-6 hop fell to 16.0 GT/s and stayed there

Found while verifying the relaunch at the full context (07:12-07:15 UTC, boot 06:49:06). `tools/reseat-check.py
--diff` against the 06:57 snapshot:

```
card2 0000:c0:03.1: 32.0 GT/s x16 -> 16.0 GT/s x16      (root port, max_link_speed 32.0)
card2 0000:c4:00.0: 32.0 GT/s x16 -> 16.0 GT/s x16      (switch upstream port, same hop)
card2 0000:c6:00.0: stays 32.0 GT/s x16                 (the card's own hop is fine)
slot 6 chain: 132 -> 144 correctable over 15.8 min, the burst inside the 140,401-token prefill
```

Three things matter here. **(1)** The kernel logged *nothing*: `journalctl -k -b 0 | grep -iE 'link|LnkSta|
downgraded'` is empty, so a link that drops a full generation is invisible in the journal and only sysfs polling
sees it -- this is why `aer-watch`'s 5-minute snapshots are worth keeping. **(2)** After the downgrade,
correctables stopped: `c4:00.0 TOTAL_ERR_COR` has been frozen at 84 across three samples 45 s apart while the
serve stays healthy, whereas the same counter moved +129 in 3.1 min during the weight load at Gen5. That is
correlation, not proof, but it points the same way as the BadTLP story: the hop is marginal at Gen5 and clean at
Gen4. **(3)** It is still the *same hop* as the BadTLP/Timeout counters -- root port `c0:03.1` to switch
upstream port `c4:00.0`, i.e. the slot-6 riser -- and it is still nothing to do with the CPU/L2 fault.

Open, and worth measuring before the next reseat: prefill/decode rates at Gen4 vs Gen5 on that card. Today's
140k-token prefill ran at ~7,600 prompt tok/s at Gen4; the note's Gen5-era numbers are in
[native-4xr9700.md](../native-4xr9700.md). Nothing retrained the link back up in the 20 minutes observed.

---

# The PCIe scope narrows: every correctable this box has ever recorded is on ONE expansion board

The operator states they did **not** reseat the GPU cards; what they pulled and refitted was the **expansion
board that carries slot 6 and slot 8**. That reorders the PCIe half of this note, because of what the counters
say across the whole journal history:

| chain | root port | slots on the same board? | correctables, boot `5ec7e479` (06:00-06:47) | correctables, boot `324e273b` (06:49-) |
|---|---|---|---|---|
| card2 / **slot 6** | `c0:03.1` (AMD 0x14a5) | yes | **87** (`c4:00.0` 59, `c0:03.1` 26, card-side ports 1 each) | **144** (`c4:00.0` 84 incl. BadTLP 77, `c0:03.1` 58 Timeout) |
| card4 / **slot 8** | `00:03.1` (AMD 0x14a5) | yes | **6** (`04:00.0` 3, `00:03.1` 1, card-side ports 1 each) | **0** |
| card1 / slot 3 | `c0:01.1` (AMD 0x14ab) | other path | 0 on all four ports | 0 |
| card3 / slot 1 | `00:01.1` (AMD 0x14ab) | other path | 0 on all four ports | 0 |

Two chains have ever produced a PCIe error on this box, and they are exactly the two slots on that one board.
Also worth keeping: the upstream/downstream pair in every chain is **on the GPU card** (AMD/ATI `Navi 10 XL
Upstream/Downstream Port`, devices `0x1478`/`0x1479`), so the only external link per chain is
root port ↔ card upstream port -- and that is the link that dropped to 16.0 GT/s.

**When the reseat happened, from the journal rather than from memory** (`tools/boot-gaps.py`, added with this
entry: it lists the gap between the last entry of one boot and the first of the next, which is how you tell
"this reboot had time for an open-chassis job" from "this was a 100-second reboot + POST"):

```
11-03  02:48:14 -> 02:50:38    2.4 min
11-03  05:16:17 -> 05:42:30   26.2 min   <-- the only window today long enough
11-03  05:45:19 -> 05:47:29    2.2 min
11-03  05:57:43 -> 06:00:23    2.7 min
11-03  06:47:44 -> 06:49:24    1.7 min   <-- the deliberate reboot; no physical work inside it
```

So the board work belongs to the 05:16-05:42 window (pending the operator's confirmation), and it was done with
power off: the GPU root ports `c0:03.1`/`00:03.1` have **no pciehp hot-plug controller**, so pulling that board
live would have taken both links down without any neat Surprise-Removal message -- and there is no such trace
anywhere: no link-down, no containment, no uncorrectable, all four cards enumerated, counters accumulating
normally. A powered removal would not look like that.

**What it did not do: fix slot 6.** The first enumeration after that window already errors (`ec0d0751`, 05:47:29:
`c4:00.0` BadTLP 1, `c0:03.1` Timeout 3), the next one reached 87 correctables, and in the newest boot slot 6
reached 144 **and lost its Gen5 hold** (16.0 GT/s, still there after 30+ min) while slot 8 -- same board, same
root-port device id, same card model -- is completely clean. That is a per-slot problem on that board, not a
board-wide one, and the trend is worse rather than better.

Remaining suspects, narrowed to that link and to one slot of that board: the board's slot-6 connector / its lane
routing / the power to that slot; card2's edge connector or its Gen5 receiver; the CPU PHY's equalization for
that port as configured by BIOS (ESC4000A-E12, BIOS 2501 -- `sudo lspci -vvv` `LnkCtl2` target speed would
separate "tried Gen5 and could not hold it" from "was told to run Gen4", not readable without root).
Asymmetric refitting of a two-slot board is a classic way to get exactly this (one end not fully engaged), so a
deliberate second refit -- both ends even, screws/standoffs, connector pins, board power connector -- is cheap
and worth doing.

**The decisive test needs no parts**: swap card2 and card4 **within that same board** (slot 6 <-> slot 8). If the
errors follow the card, card2's PHY/edge connector is the part to replace; if they stay with slot 6, the board is.
Swapping across boards would not answer it -- the boards differ.

None of this touches the fatal-MCA half of this note: the CPU/L2 deaths happened with this board in both its
pre- and post-refit states, on a different reporting path, and offlining core 65 changed nothing on the PCIe
side.

## Pre-registered before the swap: what each outcome will mean

The operator is about to power down and exchange the two GPUs **on the same board** -- slot 6 (`c6:00.0` chain)
and slot 8 (`06:00.0` chain). Identity is locked first, because PCI BDF follows the slot and only `ASIC_SERIAL`
follows the card (`tools/gpu-identity.py`, output kept in `~/aer/identity-pre-swap.txt`, 07:28:35):

```
0x2DCB548B0D0A5DFC  0000:03:00.0  slot 1   chain all 32.0 GT/s
0xD9B246D580E8CD73  0000:06:00.0  slot 8   chain all 32.0 GT/s     <-- the clean one
0x179155577B336A5A  0000:c3:00.0  slot 3   chain all 32.0 GT/s
0x51E10AFD4A21ADEE  0000:c6:00.0  slot 6   c4:00.0=16.0, c0:03.1=16.0   <-- the sick chain
```

Nothing about the board is touched during this operation -- the board was refitted already, and changing the
board and the cards in the same power-down would confound the one question the swap answers.

Declared in advance, so the result cannot be fitted afterwards:

- **errors follow the card** (after the swap the *slot 8* chain, carrying serial `0x51E1…ADEE`, accumulates
  BadTLP and/or drops to 16.0 GT/s, while slot 6 with serial `0xD9B2…CD73` stays clean) → the marginal Gen5 path
  is on that **card** (its edge connector or its Gen5 receiver). RMA the card, and the board is innocent.
- **errors stay with slot 6** (whatever card sits there errors, and the previously sick card is clean in slot 8)
  → the **board's slot-6 path** (connector, lane routing, power for that slot) or the CPU PHY/EQ for that port.
  RMA/replace the board, and the cards are innocent.
- **neither side errors after the swap** → the refit that happened on the way in fixed a contact problem, and the
  question is then "which of the two moves fixed it", which cannot be answered from this run -- note it as
  inconclusive rather than as a fix.
- **both sides error** → common-mode on that board or on the BIOS/PHY configuration, and the per-slot story dies.

Secondary readings that come free with the same power cycle: whether slot 6 enumerates at 32.0 GT/s cold (it did
at the 06:49 boot and fell during the 07:07-07:12 load window), and one more fatal-MCA data point with core 65
kept offline by the boot unit (the units are installed and were verified running: 18 s into boot 06:49:06).

## Settled by the `lspci -vvv` dump: nothing was configured down -- the link tried Gen5 and settled at Gen4

Read out of `~/lspci-vvv-1103-0740.txt` with `tools/lspci-links.py` (added with this entry -- the reason it
exists is that `lspci` puts a **TAB** after `LnkSta:`, so a grep written as `'LnkSta: '` matches nothing and the
one decisive line disappears from the output while looking perfectly reasonable).

| port | role | `LnkCtl2: Target Link Speed` | `LnkCap2: Supported Link Speeds` | `LnkSta` actual |
|---|---|---|---|---|
| `c0:03.1` | root port, slot 6 | **32GT/s** | 2.5-32GT/s, Retimer+ 2Retimers+ | **16GT/s** x16 |
| `c4:00.0` | card upstream port, slot 6 | **32GT/s** | 2.5-32GT/s, Retimer+ 2Retimers+ | **16GT/s (downgraded)** x16 |
| `c5:00.0` | card downstream port (the card-internal hop) | 32GT/s | 2.5-32GT/s | **32GT/s** x16 |
| `00:03.1` / `04:00.0` | the same two roles, slot 8 (control) | 32GT/s | 2.5-32GT/s | 32GT/s x16 |

Three conclusions, each of which kills a hypothesis:

1. **Not a configuration change.** Slot 6's root port asks for 32GT/s and advertises 2.5-32GT/s exactly like the
   three healthy chains. There is no "BIOS/OS set it to Gen4" story left to tell.
2. **Not a power-management artifact.** `LnkCtl: ASPM Disabled` on all four root ports, so this is not L1 exit or
   an autonomous power state; and the very next hop inside the same card negotiated 32GT/s -- the failure is
   **per hop**, not per device.
3. **The counters are a full account, not a survivor.** `DevCtl: CorrErr+ NonFatalErr+`, `CEMsk`/`UEMsk` every
   bit unmasked, `AERCap: ECRCGenEn+ ECRCChkEn+`, and `CESta`/`UESta` all clear at dump time. So correctables
   are reported and none are being hidden, ECRC is on in both directions, and "no uncorrectable on any link" is
   a real measurement rather than a masked one.

What the box *does* have that nobody has used: **`Lane Margining at the Receiver`** on both root ports and on the
card's upstream port, plus retimer presence detection (`Retimer+ 2Retimers+`) -- i.e. the standardized
per-lane eye-margin procedure exists exactly on the ports in question. It is invasive on a live link (it moves
the receiver's sampling point), so it is something to have AMD/ASUS run, or to run on a box that is not serving,
not something to drive with `setpci` mid-traffic. The Gen5 equalization extended-capability fields are **not**
exposed by these AMD parts (only `Physical Layer 16.0 GT/s` appears), so the EQ phase parameters cannot be read
from here -- put that to the vendor too; their tooling can.

That leaves the physical path (contact/depth/board/retimer) and the PHY/EQ parameters as the remaining
explanations, which is why the swap and the seating inspection are the next steps.

## The refit result, measured: seating restored Gen5, and changed nothing about the errors

08:17:12 boot. The operator opened the box and found the slot-6 card not fully seated; refitted it and powered
back on. No card swap was performed -- `tools/gpu-identity.py` confirms by `ASIC_SERIAL` that `0x51E10AFD4A21ADEE`
is still at `c6:00.0` and `0xD9B246D580E8CD73` still at `06:00.0` (`~/aer/identity-post-swap.txt`, 08:18:56). So
this is variant C of the plan above, and the card-vs-slot question is **still open**.

What the refit demonstrably fixed:

```
slot 6 chain after the refit, cold enumeration: c0:03.1 / c4:00.0 / c5:00.0 / c6:00.0 all 32.0/32.0 GT/s
(before: the external hop was 16.0 GT/s, and it had fallen there during the previous boot's load window)
```

What it did not fix -- the same measurement as before, the weight-load window, fresh counters, same launch shape
(`MAXLEN=262144 NSEQ=32 NBT=8192`, 120.3 GB of weights):

| boot | slot-6 external hop | load window | afterwards, idle |
|---|---|---|---|
| `324e273b` (before the refit) | 32 GT/s during the load, then **fell to 16 GT/s** | **+129 / 3.1 min = 41.9/min** | at Gen4: **0 events** across three 45 s samples, and counters stayed frozen for the rest of the boot |
| 08:17 boot (after the refit) | **32 GT/s held** (no fallback in the ~10 min observed) | **+153 / 3.4 min = 45.7/min** (`c4:00.0` BadTLP 86, `c0:03.1` Timeout 68) | ~+1 event / 2 min, still at Gen5 |

Other chains in the same window: slot 1 and slot 3 **+0**, slot 8 **+1**.

So the seating was real (it decided whether the port could hold Gen5 at all) and it is **not the cause of the
errors**. What the two boots now say together: that hop is **marginal specifically at Gen5** -- at 16 GT/s it was
silent for the remainder of a boot that kept serving, including a 140,401-token prefill; at 32 GT/s it produces
40-46 correctables per minute under DMA load and ~0.5/min idle, and it is the only chain on the machine that
does. Remaining suspects unchanged: the board's slot-6 path, this card's Gen5 receiver (the swap would separate
these two -- still not done), or PHY/retimer equalization for that port.

Also for the record, since a boot boundary is an MCA sample: this boot came up with **no replayed fatal record**
(second consecutive; the previous boot ended in a clean `poweroff`), with core 65 offline again automatically.

## The swap: the fault stayed with slot 6 -- and the card turned out to set the magnitude, not the existence

08:45:42 boot. The operator swapped cards. Which cards, read from `ASIC_SERIAL` rather than from memory
(`~/aer/identity-post-swap-0847.txt`, `tools/gpu-identity.py`):

```
0x51E10AFD4A21ADEE   was slot 6 (c6:00.0)  ->  now slot 3 (c3:00.0)      <- the card that had been sick
0x179155577B336A5A   was slot 3 (c3:00.0)  ->  now slot 6 (c6:00.0)      <- a card that had never errored
0xD9B246D580E8CD73   slot 8, unchanged      0x2DCB548B0D0A5DFC  slot 1, unchanged
```

Not the pair I had pre-registered (slot 6 <-> slot 8, same board), and that matters for one of the two readings
below, so it is stated where it can be misread later: slot 3 hangs off a **different** root port (device `0x14ab`,
not `0x14a5`) and a different board. Slot 8 would have been the tighter control.

Against the branches declared before the swap, the outcome is the second one -- **errors stay with slot 6**:

| measurement, same boot, same launch shape (`MAXLEN=262144 NSEQ=32 NBT=8192`, 120.3 GB of weights) | slot 6 chain | slot 3 chain (with the formerly-sick card) | slot 8 | slot 1 |
|---|---|---|---|---|
| enumeration (~2 min after power-on) | `c4:00.0` **BadTLP 1** | 0 | 0 | 0 |
| weight-load window (+3.0 min) | **+34 = 11.5/min** (BadTLP 23, root-port Timeout 13) | **0** | 0 | 0 |
| afterwards, 2.7 min idle at Gen5 | **0 new events** (23/13 frozen) | 0 | 0 | 0 |
| link | **32 GT/s held** | 32 GT/s | 32 | 32 |

A different card now sits in slot 6 and slot 6 still produces PCIe correctables that no other chain on the
machine produces. **That is the branch that says: the part to replace is the slot-6 path** -- the board's slot-6
connector/lanes/power for that slot, or the motherboard side for that port (`c0:03.1` PHY / equalization); the
card that had been sitting there is not the origin.

And the comparison across the two boots says something the plan had not anticipated -- **which card sits in slot 6
changes the magnitude**, with everything else held:

| boot 08:17 (card `0x51E1…ADEE` in slot 6) | boot 08:45 (card `0x1791…6A5A` in slot 6) |
|---|---|
| load window **+153 / 3.4 min = 45.7/min** | load window **+34 / 3.0 min = 11.5/min** |
| idle ~+1 event / 2 min at Gen5 | **0 events** over 2.7 min at Gen5 |
| Gen5 **fell to 16 GT/s** during the first load window | **Gen5 held** through load and 10+ min |

Two-factor reading, stated at the strength the data supports: the fault **requires** the slot-6 path (it appears
there with either card), and the card sitting there **modulates** it hard (4x the error rate, and whether the
link can hold Gen5 at all). A receiver at one end that is merely adequate plus a channel that is marginal at
Gen5 produces exactly that. So: RMA target = the slot-6 board path (and the motherboard-side alternative), with
card `0x51E10AFD4A21ADEE` noted as the weaker end rather than the culprit.

Cheap confirmation available if it is ever wanted: put the two cards back. If the rate returns to ~45/min and
Gen5 falls back to 16 GT/s again, the two-factor reading is confirmed; if it comes back at ~11/min, the
magnitude difference was not the card and this section over-reads a single sample each side.

Separately, and because a boot boundary is also an MCA sample: this boot again came with **no replayed fatal
record**, third in a row, with core 65 taken offline automatically. Total uptime observed with the core offline
so far is ~90 minutes across three boots, against a pre-change fatal cadence of about one per 3.6 h -- a
positive trend, not yet a verdict.

## Observation is now on a clock, not on whoever remembers

Two questions are open and both are answered by accumulated samples, so the accumulation was made automatic:

- **`tools/brief.py`** -- one line per call into `~/aer/series.log`: boot id, threads online, per-slot link speed
  of the weakest hop, per-slot correctable total and the rate against the previous sample, and serve health. It
  runs inside the `aer-watch` loop every 5 minutes alongside the JSON snapshot. Rates are per boot on purpose:
  the counters reset at enumeration and this box reboots itself, so a rate spanning boots would be fiction.
  Example (2026-11-03 09:06, after the swap):

  ```
  09:06:58  15d3  threads 190  slot1(03:00.0) 32GT cor=0  slot3(c3:00.0) 32GT cor=0
            slot6(c6:00.0) 32GT cor=42  0.00/min  slot8(06:00.0) 32GT cor=0  serve 200
  ```

- **`host/r9700-mca-archive.service`** now runs `tools/mca-hazard.py` at every boot as well, so the fatal-MCA
  verdict series lands in the journal without anyone being asked: cumulative clean uptime since the last fatal
  event and the probability that the silence is luck under the old cadence. (The unit in `/etc` predates that
  line; re-copy it before the next boot when convenient -- nothing is lost meanwhile, the numbers are one
  command away.)

What the current numbers say, at the moment this was written: slot 6 is the only erroring chain (42 correctables
this boot, 0/min at idle, Gen5 held), the other three are exactly zero, the serve is answering, core 65 is
offline, and the fatal-MCA clock stands at ~1.5 h of clean uptime -- which is `p=0.58` of being luck under the
pre-change cadence. The decision thresholds in this note are 7.1 h (p=0.05), 10.9 h (p=0.01), and ~24 h for the
"confined to that core" working conclusion.
