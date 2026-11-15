#!/usr/bin/env python3
"""tools/fault-report.py -- the long view over the box's own fault record.

Three weeks from now the questions are not "is it up" but "how often does this die, does it come back by itself,
and is the thing that keeps dying the card or the code". Those answers live in files the watcher writes:

    .r9700-state/faults.jsonl   one line per death the watcher saw (restart followed or not)
    .r9700-state/alerts.jsonl   one line per hardware alert raised
    .r9700-state/ALERT.json     the alert that is standing right now, if any

This reads them and answers in order: the deaths with their signature, uptime and decision; what the kernel
driver currently reports for this boot (fault LINES and fault EVENTS, per PCI function, split by the shape the
driver printed -- read/write, UTCL2 client, ring, which is what separates this box's two fault families); and
the same per-card verdict the watcher escalates on. It reads nothing that needs the engine up, so it works during
the outage it is being run for.

Exit status: 0 with no standing alert, 2 with one -- so it can be dropped into a cron or a shell prompt and the
answer is still visible without reading it. `--raise-if-due` asserts the alert through the same function the
watcher calls, for the case where the condition already holds and waiting for the next crash to be told about it
is not a plan.

Run: python3 tools/fault-report.py [--state DIR] [--n N]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HAND = "tools/fault-report.py --raise-if-due (no death behind it: the window condition holds)"
sys.path.insert(0, str(ROOT))
import setup as S                                                # noqa: E402  (defs and constants only)


def rows(path):
    """JSONL that may be half-written: a line that does not parse is reported, never skipped silently, because
    the file being read is evidence and a truncated tail is a fact about the crash that truncated it."""
    out, bad = [], 0
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                bad += 1
    except OSError:
        return [], 0
    return out, bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(ROOT / ".r9700-state"))
    ap.add_argument("--n", type=int, default=12, help="how many recent deaths to list")
    ap.add_argument("--raise-if-due", action="store_true",
                    help="assert the alert through setup.py's own raise_alert if the per-card threshold holds "
                         "(the watcher raises it at a death; this raises it now, and says it was hand-run)")
    a = ap.parse_args(argv)
    state = Path(a.state)
    deaths, bad1 = rows(state / S.FAULT_LEDGER)
    alerts, bad2 = rows(state / S.ALERT_LEDGER)
    live = None
    try:
        live = json.loads((state / S.ALERT_FILE).read_text())
    except (OSError, ValueError):
        live = None

    print(f"== {state / S.FAULT_LEDGER}: {len(deaths)} death(s) recorded"
          + (f", {bad1} unparseable line(s)" if bad1 else ""))
    if deaths:
        again = sum(1 for d in deaths if d.get("restart"))
        print(f"   restarted: {again}   not restarted: {len(deaths) - again}")
        sig = Counter(d.get("sig") or "no signature" for d in deaths)
        for k, v in sig.most_common(5):
            print(f"   {v:3d} x {k}")
        ups = [d.get("uptime_s") for d in deaths if isinstance(d.get("uptime_s"), (int, float))]
        if ups:
            print(f"   uptime at death: min {min(ups)} s, median {sorted(ups)[len(ups) // 2]} s, max {max(ups)} s"
                  "   (a fault minutes into a launch and one six days in are different problems)")
        print(f"   last {min(a.n, len(deaths))}:")
        for d in deaths[-a.n:]:
            up = d.get("uptime_s")
            print(f"     {d.get('died_at') or d.get('at')}  {'restarted' if d.get('restart') else 'left down'}"
                  f"  up {up if up is not None else '?'}s  ranks left {d.get('ranks_left')}"
                  f"  {(d.get('sig') or '')[:34]}")
    print(f"== {state / S.ALERT_LEDGER}: {len(alerts)} hardware alert(s) raised"
          + (f", {bad2} unparseable line(s)" if bad2 else ""))
    for al in alerts[-3:]:
        print(f"   {al.get('what', '')}")
    print(f"== standing alert: {json.dumps(live)[:200] if live else 'none'}")

    ev = S.kernel_fault_events()
    if ev.get("error"):
        print(f"== kernel journal: CANNOT READ ({ev['error']}) -- this is not 'no faults'")
    else:
        print(f"== kernel journal, this boot: {ev['total']} fault line(s) over "
              f"{len({(e['at'], e['card']) for e in ev['events']})} event(s)")
        per = Counter()
        for e in ev["events"]:
            per[(e["card"], e.get("rw") or "?", e.get("client") or "?", e.get("ring") or "?")] += 1
        for (card, rw, client, ring), n in sorted(per.items(), key=lambda kv: -kv[1]):
            print(f"   {n:3d} line(s) {card}  {rw} / {client} / ring {ring}")
    v = S.hw_verdict(ev)
    if v and v.get("card"):
        print(f"== verdict: ESCALATED -- {v['count']} events on {v['card']} inside "
              f"{v['window_s'] // 86400} d ({v['since']} .. {v['until']})")
        print(f"   {S.alert_words(v)[1]}")
        if a.raise_if_due:
            # Raised through setup.py's own path, the same one the watcher uses at a death, so a hand-run call is
            # not a second, sloppier implementation of the claim. It says so in the row: no death signature, and
            # `raised_by` names how it happened -- an alert written by hand must stay distinguishable from one a
            # watcher raised on a crash, or the ledger stops being evidence.
            row = S.raise_alert(v, S.state_read("launch.json"), None)
            S.state_write(S.ALERT_FILE, dict(row, raised_by=HAND))
            print(f"   raised: {state / S.ALERT_FILE} and one line in {S.ALERT_LEDGER} "
                  "(its sig is null: no death behind this one)")
        return 2
    if v:
        print(f"== verdict: not escalated (worst card {max(v['cards'].values()) if v['cards'] else 0} event(s) "
              f"in {v['window_s'] // 86400} d; threshold {S.HW_ALERT_N})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
