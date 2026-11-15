# Supervision: what a death leaves behind, and when a count becomes a claim

Written on 2026-11-15, the day the box had one production crash, five reproduction crashes of its own choosing,
and a 26-minute window in which the machine was on fire and nothing said so.

## 1. The watcher's job is a record before it is a rescue

`setup.py --watch` waits for `/health` to stop answering, then decides what to do. It has always been able to
restart; what it could not do until today was answer, hours later, **what it saw**. Two gaps had to be closed
before an automatic restart was safe to arm:

  * a crash that leaves the ranks holding the GPUs looked healthy. The API stopped answering while three
    `VLLM::Worker_TP*` processes sat on ~28 GiB each. `watch_dead()` asked "is the session gone, or are there no
    ranks?" and got "no" to both, so the launch recorded no death at all for 26 minutes. It now asks the third
    question — API silent, ranks still holding, a crash signature in the launch log — and calls that *wedged*.
  * a wedge needs the launcher's own stop. `pkill` would not have known which processes belonged to the launch, and
    the ranks ignore SIGTERM anyway; the recipe's `STOP=1` does.

`WATCH_RESTART=1` is the arm. `RESTART_MAX` per `RESTART_WINDOW_S` is the bound, and there is a second bound that
is not a count: a relaunch into a PCIe link running below its rated width (the check lives in
`notes/launch-methodology.md`) is a card already in a bad state, so the watcher stops and leaves the box up for
inspection. A restart that "fixes" a degrading GPU is how a repairable machine becomes an unexplained outage.

## 2. One ledger line per death — including the restarts that worked

`.r9700-state/faults.jsonl`. A line per death, appended and fsynced before any relaunch: the verdict and the
signature, uptime and launch, the front door's in-flight and gate counters, the ranks still holding, the driver's
page-fault events (this boot and inside the launch's own window), the per-card counts, the restart decision, and
whether a hardware alert had been raised.

That ordering is the honesty of the mechanism: **the evidence is written before the reset.** A system that
restarts itself does not get to claim afterwards that nothing happened; `tools/fault-report.py` counts the
long-term rate, the median uptime between deaths, and the shapes the driver reported — with years of hindsight,
the thing that makes "it came back by itself" an answer rather than a loss.

## 3. A count is not a claim; a claim names a card and an action

`journalctl -k` on this box holds 11 driver `[gfxhub] page fault` lines across **8 events** on **one PCI
function**, and the 11 lines are **two different faults** (`notes/bug-causal-conv1d-oob.md` §20): 6 with
`read / SQC (data) / ring 24` — the vLLM family being hunted — and 5 with `write / TCP / ring 40` from two bench
runs that are not this launch's problem at all. A page that shows "11" forever teaches the operator to ignore it.

So the escalation rule is not "the page is red". It is:

    HW_ALERT_N events on ONE card inside HW_ALERT_WINDOW_S  (defaults 3 / 7 days)

which on this box is true: 6 events on `c3:00.0` between 2026-11-14 20:59 and 2026-11-15 09:52. When it is true,
`raise_alert()` writes a standing `.r9700-state/ALERT.json`, appends `alerts.jsonl`, runs `ALERT_CMD` if the box
has a channel, and the Monitor's fault card changes from *counting* to *claiming*:

    Hardware: page faults      6 events on c3:00.0 in 7 d
    do: the fault follows the CARD, not the rank — swap it with a neighbour (or move it to another slot) and
        see whether the faults move with it; while unresolved, do not restart INTO it repeatedly

That claim is only worth making because the ledger keeps the evidence while the machine keeps serving. The action
it asks for is physical and has not been done yet; nothing in this tree can swap a card.

## 4. What "acceptable" meant on 2026-11-15, and what changed it

The measured rate, before the mitigation, was one death per 8,402 minutes of production (13.9 h). Availability
was then a function of how fast the death was *noticed*, not of the fault: attended, noticed in 3 minutes → 99.96%;
that afternoon's wedge, noticed in 26 minutes → 99.7%; unattended overnight → 94%. The last number is the one that
decided the work, and the fix for it was not a restart: it was (a) the ledger, (b) the alert, and (c) a page that
does not die with the engine — the Monitor was merged into the serve (`WEBUI=same`), so the page that reports a
wedged launch was itself wedged. The front door now serves pages from a standalone sidecar (`PAGES=:8181`, which
ATTACHES the engine's request journal read-only so the dead launch's rows stay on screen).

Two things are still missing to make the loop close: `ALERT_CMD` has no channel configured on this box (the alert
waits for someone to open the page), and the card swap that would separate "this silicon" from "this slot and
chain" is scheduled against nothing.
