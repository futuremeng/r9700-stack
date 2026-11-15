#!/usr/bin/env python3
"""The one place this box knows how to reach a human.

Two programs need to send an alert and neither should own the wire format. The alert is RAISED by the supervisor
(`setup.py`, at the moment a rank dies, possibly seconds before a relaunch) and it is CONFIGURED and TESTED from
the page (`serve/webui.py`, where an operator pastes a webhook into a form). Two implementations of one POST is how
one of them ends up sending a message the other would have rejected -- so both import this file, and the keyword
rule lives in exactly one place.

The channel is a DingTalk custom-robot webhook: POST JSON to `https://oapi.dingtalk.com/robot/send?access_token=…`
and the robot accepts the message only if it carries the robot's configured **security keyword** (this box's is
`R9700`). That filter answers with `errcode != 0` rather than an exception, and a security filter that silently
drops the alert is the same as no alert at all -- so `post()` reports the errcode, and the ledger records THAT,
never a bare "sent". Every message built here carries the keyword in the title AND the first line of the body.

The webhook URL is a credential. It is written mode 0600, and `mask()` is the only form anything may show: host
and path, never the token. `at_mobiles` are personal data, so they are reported as a COUNT, not as values.

Nothing here is DingTalk-agnostic on purpose: the payload shape (msgtype/markdown/at) belongs to that robot, and
`validate()` refuses a URL that is not a DingTalk robot rather than POSTing a secret somewhere that will ignore
it and answer "ok".
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# The robot's security keyword. A message without it is refused at the robot, not here, and the refusal is a JSON
# errcode that looks like a network hiccup unless someone made it a rule -- so the rule is here.
KEYWORD = (os.environ.get("ALERT_KEYWORD") or "R9700").strip()

# DingTalk truncates long markdown bodies; an alert whose "do" sentence got cut off is an alert that stopped being
# actionable, so the body is built to fit and says when it did not.
BODY_LIMIT = 1800


def validate(webhook):
    """(ok, why) for a webhook URL. A half-pasted URL out of a chat window is a common accident, and this box's
    notifier speaks DingTalk: anything else gets a refusal rather than a POST that answers "ok" and goes nowhere.

    Loopback is the one host exempt from the https rule. The rule exists so a bearer credential does not cross a
    wire in the clear, and a URL pointing at this machine crosses no wire -- and without the exemption the send
    path could not be tested at all without a network (the suite runs a fake robot on 127.0.0.1 and reads the body
    it was given).
    """
    u = None
    try:
        u = urllib.parse.urlsplit((webhook or "").strip())
    except ValueError as exc:
        return False, f"not a URL ({type(exc).__name__})"
    host = (u.hostname or "").lower()
    local = host in ("127.0.0.1", "localhost", "::1")
    if u.scheme != "https" and not (u.scheme == "http" and local):
        return False, (f"scheme is {u.scheme or 'nothing'}: that POSTs the webhook token in the clear"
                       if not local else f"host {host or '?'} is not a robot (this notifier speaks oapi.dingtalk.com)")
    if not local and not host.endswith("dingtalk.com"):
        return False, f"host {host or '?'} is not a DingTalk robot (this notifier speaks oapi.dingtalk.com)"
    if not (urllib.parse.parse_qs(u.query).get("access_token") or [""])[0]:
        return False, "no access_token in the query -- that is the credential, so paste the whole webhook URL"
    return True, ""


def load(path):
    """The channel as stored, or (None, why). Never an empty dict for "not configured": a file that exists but
    cannot be read is a different state from no file, and an operator who just saved a webhook deserves to be told
    the save did not land where the sender reads."""
    try:
        raw = Path(path).read_text()
    except FileNotFoundError:
        return None, None                                   # not configured: the normal state, not an error
    except OSError as exc:
        return None, f"cannot read {path}: {type(exc).__name__}: {exc}"
    try:
        cfg = json.loads(raw)
    except ValueError as exc:
        return None, f"{path} is not JSON ({exc}) -- it was written by the About page or by hand, and one of " \
                     f"those made a mistake"
    if not isinstance(cfg, dict) or not cfg.get("webhook"):
        return None, f"{path} has no webhook in it"
    return cfg, None


def save(path, cfg):
    """Write the channel 0600 and atomically. The file holds a bearer credential: anyone who can read it can post
    to the group, so it must not be world-readable even for the moment between truncate and write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)
    os.chmod(p, 0o600)                                      # umask does not get to decide about a credential
    return p.stat().st_mode & 0o777


def mask(cfg):
    """What a page may show about the channel: where it points, not what it costs to point there."""
    if not cfg:
        return None
    u = urllib.parse.urlsplit(str(cfg.get("webhook") or ""))
    out = {"host": u.netloc or "?", "path": u.path or "/", "token": "withheld",
           "keyword": str(cfg.get("keyword") or KEYWORD),
           "is_at_all": bool(cfg.get("is_at_all")),
           "at_mobiles_n": len(cfg.get("at_mobiles") or []), "at_users_n": len(cfg.get("at_users") or [])}
    for k in ("updated_at", "updated_by", "note"):
        if cfg.get(k) is not None:
            out[k] = cfg[k]
    return out


def build(row, cfg=None, test=False):
    """(title, text) for the robot: the alert as two sentences a person can act on, plus the evidence in lines.

    `row` is the ALERT.json shape from setup.py -- `what` (what was counted, and over what window), `do` (the
    physical action), the per-card counts, the fault families, and the launch it happened in. `test` marks a
    hand-sent message so a test never reads like a new incident at 03:00.
    """
    kw = str((cfg or {}).get("keyword") or KEYWORD)
    row = row or {}
    head = f"{kw} hardware alert: {row.get('count', '?')} page faults on 0000:{row.get('card', '?')}" \
        if not test else f"{kw} alert test (not a new fault)"
    title = head[:120]
    lines = [f"### {title}", ""]
    if row.get("what"):
        lines += [f"> what: {row['what']}", ""]
    if row.get("do"):
        lines += [f"> do: {row['do']}", ""]
    cards = row.get("cards") or {}
    if cards:
        lines.append("**per card** " + ", ".join(f"{c} ×{n}" for c, n in sorted(cards.items())))
    fam = row.get("families") or {}
    if fam:
        lines.append("**shapes** " + "; ".join(f"{k} ×{n}" for k, n in sorted(fam.items())))
    for label, key in (("session", "session"), ("recipe", "recipe"), ("config", "config"),
                       ("launch log", "log"), ("crash signature", "sig")):
        if row.get(key):
            lines.append(f"- {label}: {row[key]}")
    if row.get("since"):
        lines.append(f"- window: {row['since']} .. {row.get('until', '?')}")
    lines.append("- ledger: `.r9700-state/faults.jsonl` · live claim: `.r9700-state/ALERT.json`")
    text = "\n".join(str(x) for x in lines)
    tail = "\n\n…(truncated: the full alert is in .r9700-state/ALERT.json)"
    if len(text) > BODY_LIMIT:                              # truncated out loud, not silently
        text = text[:BODY_LIMIT - len(tail)] + tail
    return title, text


def post(cfg, row, opener=None, timeout=10.0, test=False):
    """Send it, and report what the ROBOT said.

    `errcode` is the answer that matters: HTTP 200 with `errcode: 310000` ("keywords not in content") is a failure
    to alert anyone, and a caller that treats a 200 as success would put a false line in the ledger. Returns a dict
    that is safe to append to a JSONL file: it never contains the webhook.
    """
    out = {"at": int(time.time()), "sent": False, "errcode": None, "errmsg": None,
           "error": None, "http": None}
    if not cfg or not cfg.get("webhook"):
        out["error"] = "no webhook configured (.r9700-state/notify.json, or the About page)"
        return out
    ok, why = validate(cfg["webhook"])
    if not ok:
        out["error"] = why
        return out
    title, text = build(row, cfg, test=test)
    body = {"msgtype": "markdown",
            "markdown": {"title": title, "text": text},
            "at": {"atMobiles": list(cfg.get("at_mobiles") or []), "atUserIds": list(cfg.get("at_users") or []),
                   "isAtAll": bool(cfg.get("is_at_all"))}}
    req = urllib.request.Request(
        cfg["webhook"], data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json; charset=utf-8",
                 "User-Agent": "r9700-stack/notify (hardware alert)"})
    try:
        open_ = opener or urllib.request.build_opener()
        with open_.open(req, timeout=timeout) as r:
            raw, out["http"] = r.read(), r.status
    except urllib.error.HTTPError as exc:                   # the robot answered, and did not like it
        raw = exc.read()[:400]
        out["http"], out["error"] = exc.code, f"HTTP {exc.code} from the robot"
    except (urllib.error.URLError, OSError, ValueError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        got = json.loads(raw.decode("utf-8", "replace") or "{}")
    except ValueError:
        out["error"] = (out["error"] or "the robot answered") + f" with a body that is not JSON: {raw[:120]!r}"
        return out
    out["errcode"] = got.get("errcode")
    out["errmsg"] = str(got.get("errmsg") or "")[:200]
    if out["errcode"] == 0 and not out["error"]:
        out["sent"] = True
    elif not out["error"]:
        # Named for what it means, because "errcode 310000" is not something the next reader should have to look
        # up while a card is still misbehaving.
        hint = {"310000": "the robot's security keyword is not in the message, or it is not "
                          f"{KEYWORD!r} -- set ALERT_KEYWORD or the keyword field on the About page",
                "130101": "the webhook token is not valid for that robot"}.get(str(out["errcode"]), "")
        out["error"] = f"the robot refused the message: errcode {out['errcode']} {out['errmsg']}" + \
                       (f" ({hint})" if hint else "")
    return out


def remember(path, res, masked=None):
    """Record the last send attempt where a page can read it.

    Because "is this channel working?" is not answered by the file being present: a webhook that the robot refuses
    (wrong keyword, dead token, blocked egress) is configured, saved, and useless. The last attempt -- with the
    robot's own errcode -- is the fact, and it is written by whichever process sent (the watcher or the page).
    Never contains the webhook; `masked` is what `mask()` already made safe.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"at": int(time.time()), "result": {k: res.get(k) for k in
                                             ("sent", "errcode", "errmsg", "error", "http", "at")}}
    if masked:
        row["channel"] = masked
    p.write_text(json.dumps(row, indent=1, sort_keys=True) + "\n")
    return row


def last(path):
    """The last recorded attempt, or None. A missing file means no attempt has been made, which is its own answer
    and is not the same as a successful one."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def main(argv):
    """`serve/notify.py [--test]` -- send what is standing in ALERT.json (or a synthetic test), and print what the
    robot said. Exit 0 only when the robot said errcode 0."""
    import sys
    here = Path(__file__).resolve().parent.parent / ".r9700-state"
    cfg, err = load(here / "notify.json")
    if err:
        print(f"channel: {err}")
        return 2
    if not cfg:
        print("channel: not configured -- open the About page and paste the webhook, or write "
              f"{here / 'notify.json'}")
        return 2
    print("channel:", json.dumps(mask(cfg), sort_keys=True))
    alert = None
    try:
        alert = json.loads((here / "ALERT.json").read_text())
    except (OSError, ValueError):
        pass
    row = alert or {"card": "unknown", "count": 0, "what": "no alert is standing: this is a channel check",
                    "do": "nothing to do -- the message arrived, which was the question"}
    res = post(cfg, row, test="--test" in argv)
    print("send:  ", json.dumps(res, sort_keys=True))
    if res["sent"]:
        print(f"the robot accepted the message (http {res['http']}, errcode 0)")
        return 0
    print(f"NOT delivered: {res['error']}")
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
