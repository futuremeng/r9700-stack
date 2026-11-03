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
