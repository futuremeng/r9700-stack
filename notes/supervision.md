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

### Which port the watchdog asks, and what a stop costs in messages (2026-11-16)

The probe port is a claim, not a default. `--watch` used to read `os.environ.get("PORT", "8080")`, which on this box
is the **front door**: the gateway forwards `/health` to the engine, so the probe kept working — while the launch
record named a port nothing served, and `tools/recipe-check.py`, following that record, read the door's payload (a
sidecar's argv, a sidecar's env) and reported 20 "deviations" on an engine configured exactly as the recipe says.
Probing the door is also the wrong question for a breaker: a dead door would read as a dead engine, and a restart
would be spent on a healthy serve. `watch_port()` now takes the record's port first, the launcher's second, `8080`
last; `watch_loop` says the answer once at the claim — `probing :8180 for this launch (session …); restart on a known
crash signature: yes (up to 3 per 60 min)` — so "the watchdog called it dead" can be checked against what it
actually asked. `tests/test_setup.py` pins that the record's port answers `/api/status` with the serve's argv and is
not the gateway's.

One rough edge found while restarting on purpose, and left as a judgement rather than fixed quietly: a **deliberate**
`STOP=1` writes its ledger line honestly (`reason: the tmux session … is gone`, `sig: null`, `restart: False`), and
the same pass re-evaluates the standing hardware claim — which re-sends it. On 2026-11-16 a planned restart put a
second DingTalk message into the group for an unchanged claim (`alerts.jsonl` grew from 1 line to 2, `sent: true,
errcode: 0`). Either the claim should carry a fingerprint and stay quiet until the count or the card changes, or the
second message is the price of always re-checking on death. Worth deciding before the next planned restart, not
after.

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

The loop is closed on one axis and open on the other: the channel now exists (§5), so a standing claim reaches a
phone, and what remains unscheduled is the physical test — moving the card to separate "this silicon" from "this
slot and chain".
## 5. A claim that reaches nobody is a note to yourself (2026-11-15 evening)

§3 ended with a standing `ALERT.json` and a Monitor card that instructs. That is still a document. On 2026-11-15
the box knew, six times over, that `c3:00.0` keeps faulting — and the only way to find out was to already be
looking at the page. An alert that depends on someone opening a page is not an alert; it is a note that says
"somewhere there is something you should have been told".

The channel is a DingTalk robot (`serve/notify.py`), configured from the About page and sent by both programs that
can raise a claim: the watcher (at a death, before it relaunches) and `tools/fault-report.py --raise-if-due`
(hand-run, when the window already says what a crash would have said). One file, imported by both, because two
implementations of one POST is how a keyword filter ends up refusing the message while both programs believe they
sent it.

Four things this forced into the design, all of them the same failure seen from a different angle:

* **The POST returning 200 is not the delivery.** DingTalk answers a message that fails its security-keyword
  filter — or a dead token, or a rate limit — with `200` and a body of `{"errcode": 310000, ...}`. A notifier that
  looked at the status line would have recorded "sent" and moved on. The send is judged by the **errcode**, and the
  errcode is kept in the ledger line and in `ALERT.json.notify`, so three days later the honest answer to "did you
  get a message?" is in a file rather than in someone's memory.
* **The keyword `R9700` is in the title AND the body.** The robot's filter is a substring rule configured in the
  group; a message without it never reaches the group at all, and the only symptom is that JSON error above.
  Encoding it twice and in the builder (not in the caller's prose) is what makes the copy in `§3` and the copy on
  the phone the same message.
* **Configured is not working.** The page's card answers from `.r9700-state/notify-last.json` — the last attempt,
  including the robot's refusal — not from the presence of a channel file. `Save` says the file landed; `Send test`
  says the robot accepted, and a test message says in its own text that it is a test, so a channel check at an odd
  hour cannot be mistaken for a new incident.
* **A stored webhook is a bearer credential for a group.** `.r9700-state/notify.json` is written 0600, the page
  reads the channel **masked** (`host`, `path`, `token: withheld`, `@-lists counted`), the write endpoint wants a
  token that only a form loaded from this page has, and an empty webhook field means "the one already on the box" —
  because re-checking a channel must not require pasting the secret out of a chat window again.

The Monitor's fault card gained the delivery as part of its headline: `Hardware: page faults · pinged`, or
`· NOT pinged`, or `· no channel`. A red card that no one could have seen is the failure mode of every monitoring
system ever built, and the suffix is the only place it is admitted.

Still open, deliberately: the robot posts to a group, and a group is where a claim becomes a shared judgement
("is this worth swapping the card for?") rather than one person's backlog. If the alert stops being read, the fix
is the threshold (`HW_ALERT_N`), not the channel.
