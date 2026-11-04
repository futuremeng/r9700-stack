#!/usr/bin/env python3
"""Is the live serve on THIS tree's production recipe, or on a config a test changed?

    tools/recipe-check.py            # asks the live serve on :8080
    tools/recipe-check.py URL        # any serve
    tools/recipe-check.py --json     # machine-readable verdict

Why this exists. Measuring things on this box needs config changes -- today it was `PREFIX_CACHE=1` to price the
prefix cache -- and a test config left running is worse than no measurement: every number the Monitor then shows
belongs to a machine that is not the one the recipe describes. The launcher already prints the exact command it
would run (`DRY=1`), and the serve publishes what it is ACTUALLY running (`GET /api/status`: the engine's own
`config` block plus the `launch` line the wrapper wrote). This tool puts those two side by side and answers one
question in one command: 改回来没有.

It does not keep its own copy of the defaults, because a copied default goes stale the day the script changes.
The production truth is obtained by asking the launcher itself, in its no-launch mode.
"""
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE.parent / "serve" / "native-flashnext.sh"

# The knobs that change what a measured number means. Deliberately not everything: `--host` or the model path
# being different is an operational fact, not a measurement-config fact, and this tool should stay boring enough
# to be run reflexively.
# (flag in the launcher's line, key in the recipe meta file, type, key in the engine's own config block or None).
# Which source answers which knob is not decoration: the engine publishes `cache_dtype` but NOT max_model_len or
# max_num_seqs (README records the trap where a requested context comes back silently smaller, and only the
# request-vs-usage comparison catches that), so the recipe file answers those and the engine answers what it
# really enabled. No default value appears in this file on purpose: production is whatever the launcher says.
TRUTH = {
    "max_model_len": ("--max-model-len", "maxlen", int, None),
    "max_num_seqs": ("--max-num-seqs", "nseq", int, None),
    "max_num_batched_tokens": ("--max-num-batched-tokens", "nbt", int, None),
    "kv_cache_dtype": ("--kv-cache-dtype", "kvdtype", str, "cache_dtype"),
}
BOOLS = {"enable_prefix_caching": ("--no-enable-prefix-caching", "prefix_cache")}
JSON_FLAGS = ("--speculative-config", "--compilation-config")

# The environment is part of the recipe too. Half of this stack's switches are `R9K_*` names that decide WHICH
# KERNEL RUNS -- they change what a measured number means without appearing anywhere in the engine's measured
# config. Measured on 2026-11-04: this tool answered "PRODUCTION recipe" for a launch whose gate kernel was
# switched by `R9K_ROUTER_MAX_M=4`, correctly and uselessly. So the launcher prints its env block
# (`ENVRECIPE=1`), records it in the recipe file as `env`, the serve publishes its own (whitelisted) environment
# in `/api/status`, and the three are compared here.
ENV_NS = re.compile(r"^(R9K_|R9700_)")
# ... except the Monitor's own plumbing. R9K_WEB_* and R9700_WEBUI_* decide where a page listens and which journal
# it appends to; they do not change what the engine measures, and the launcher exports them for the page after the
# recipe file is written, so comparing them would report a difference that is only the order of the script.
ENV_PAGE = re.compile(r"^(R9K_WEB_|R9700_WEBUI_)")
# Same rule the page uses: a name that looks like a credential is compared for presence only, and its value never
# reaches this tool's output.
SECRET = re.compile(r"KEY|TOKEN|SECRET|PASS|CRED|AUTH", re.I)
# Names the launcher either defaults or reads from the operator's shell. Asking for the PRODUCTION truth with any
# of them still set would make "production" depend on who typed the command -- the same reason this file keeps no
# copied defaults.
OPERATOR_KNOBS = ("MAXLEN", "NSEQ", "NBT", "UTIL", "KVDTYPE", "MTP", "PREFIX_CACHE", "CGSIZES", "EXTRA",
                  "EAGER", "AITER", "HWQ", "MWAITX", "OFFLOAD_GB", "ATTN", "REASONING", "TOOLCALL", "FORCE")


def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if not ENV_NS.match(k) and k not in OPERATOR_KNOBS}
    env.update(extra or {})
    return env


def production_args():
    """The command the launcher in THIS checkout would run, with no test env set.

    `FORCE=1` only to get past the "GPUs are busy" guard: DRY exits before launching anything. The parent
    environment supplies HOME (the launcher derives $VENV and $LOG from it), so the printed line is the real one.
    """
    r = subprocess.run(["bash", str(LAUNCHER)],
                       env=clean_env({"DRY": "1", "FORCE": "1"}),
                       capture_output=True, text=True, timeout=60)
    line = next((l for l in (r.stdout + r.stderr).splitlines() if l.startswith("launch:")), "")
    return line[len("launch:"):].strip()


def parse_args(cmd):
    """(--flag value) pairs and bare flags of a launch line.

    The launcher echoes its ARGS array unquoted, so a JSON value arrives spread over several words: those two
    flags are read brace-balanced off the raw line instead of token by token.
    """
    out, flags = {}, set()
    for flag in JSON_FLAGS:
        m = re.search(re.escape(flag) + r"\s+(\{.*?\})", cmd)
        if m:
            out[flag] = m.group(1)
    toks = cmd.split()
    i = 0
    while i < len(toks):
        tk = toks[i]
        if tk.startswith("--"):
            if tk in out:                                     # consumed above; skip its pieces
                i += 1
                while i < len(toks) and not toks[i].startswith("--"):
                    i += 1
                continue
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            if nxt.startswith("--") or nxt == "":
                flags.add(tk)
                i += 1
            else:
                out[tk] = nxt
                i += 2
        else:
            i += 1
    return out, flags


def compare(cmd, config, meta):
    """-> [{key, production, engine, meta}] for every measured-config knob that disagrees.

    Two sources on purpose: the engine reports what it ENDED UP with (README records the trap where a requested
    context comes back silently smaller), while the recipe file says what the operator asked for. A knob either of
    them disagrees with the launcher on is a deviation.
    """
    args, flags = parse_args(cmd)
    dev = []
    for key, (flag, metakey, cast, engkey) in TRUTH.items():
        want = cast(args[flag]) if flag in args else None
        got_eng = config.get(engkey) if engkey else None
        got_meta = meta.get(metakey)
        bad = want is not None and any(
            v is not None and cast(str(v)) != want for v in (got_eng, got_meta))
        if bad:
            dev.append({"key": key, "production": want, "engine": got_eng, "meta": got_meta})
    # Prefix caching is a flag of ABSENCE, which is exactly how a test config survives a revert.
    off = BOOLS["enable_prefix_caching"][0] in flags
    metakey = BOOLS["enable_prefix_caching"][1]
    want_pc = not off
    got_eng = str(config.get("enable_prefix_caching")).lower() in ("true", "1")
    got_meta = str(meta.get(metakey)) in ("1", "true")
    if got_eng != want_pc or got_meta != want_pc:
        dev.append({"key": "enable_prefix_caching", "production": want_pc,
                    "engine": got_eng, "meta": got_meta})
    mtp_prod = json.loads(args.get("--speculative-config", "{}")).get("num_speculative_tokens") if         "--speculative-config" in args else None
    if mtp_prod is not None and meta.get("mtp") is not None and str(mtp_prod) != str(meta.get("mtp")):
        dev.append({"key": "num_speculative_tokens", "production": mtp_prod, "engine": None, "meta": meta.get("mtp")})
    return dev


def production_env():
    """The environment the launcher in THIS checkout exports, as a dict -- operator shell knobs stripped."""
    r = subprocess.run(["bash", str(LAUNCHER)], env=clean_env({"DRY": "1", "FORCE": "1", "ENVRECIPE": "1"}),
                       capture_output=True, text=True, timeout=60)
    line = next((l for l in (r.stdout + r.stderr).splitlines() if l.startswith("env recipe:")), "")
    try:
        return json.loads(line[len("env recipe:"):])
    except ValueError:
        return None


def live_env(st):
    """The environment of the process that answered, as ({name: value, "~name": ""}, source).

    /proc first when this tool can read it -- the payload names the pid, and /proc shows every name, while the
    page's block is a whitelist and a whitelist can hide exactly the knob being looked for. The payload's own
    block is the fallback (a serve on another machine), and the source is printed either way so the reader knows
    how complete the answer is."""
    pid = (st.get("env") or {}).get("pid")
    if pid:
        try:
            raw = Path(f"/proc/{int(pid)}/environ").read_bytes()
        except (OSError, ValueError):
            raw = b""
        out = {}
        for kv in raw.split(b"\0"):
            if not kv:
                continue
            k, _, v = kv.decode("utf-8", "replace").partition("=")
            if not ENV_NS.match(k) or ENV_PAGE.match(k):
                continue
            if SECRET.search(k):
                out["~" + k] = ""
            else:
                out[k] = v
        if out:
            return out, f"/proc/{pid}/environ"
    vars_ = (st.get("env") or {}).get("vars")
    return (vars_, "the payload's whitelisted env block") if vars_ else (None, "nothing readable")


def compare_env(prod, live, meta):
    """Environment deviations: {key, production, engine, meta} per name that disagrees.

    `live` is the environment the SERVE publishes about itself (already whitelisted by the page; a name it treats
    as a credential arrives as `~NAME` with an empty value, and such a name is compared for presence only -- a
    gate that compared secret values would be a gate that leaks them into its own output). Anything in our own
    namespace that the recipe does not record is a deviation: an unrecorded knob is how a test config survives.
    """
    if prod is None:
        return [{"key": "env", "production": None, "engine": None, "meta": None}]
    if live is None:
        return [{"key": "environment", "production": f"{len(prod)} R9K_/R9700_ names",
                 "engine": "not published (this build's payload has no env block)", "meta": None}]
    masked = {k[1:] for k in live if k.startswith("~") and not ENV_PAGE.match(k[1:])}
    live = {k: v for k, v in live.items() if not k.startswith("~") and not ENV_PAGE.match(k)}
    meta = meta if isinstance(meta, dict) else {}
    # A credential is never printed from either side: the launcher's environment can hold one, and a report that
    # names its value would be a leak through the very tool that is supposed to be safe to run in front of others.
    prod = {k: ("set, value withheld" if SECRET.search(k) else v)
            for k, v in prod.items() if not ENV_PAGE.match(k)}
    dev = []
    for name in sorted(set(prod) | {k for k in live if ENV_NS.match(k)} | {k for k in masked if ENV_NS.match(k)}):
        want, got, m = prod.get(name), live.get(name), meta.get(name)
        if name not in prod:
            dev.append({"key": f"env:{name}", "production": None,
                        "engine": "set, value withheld" if name in masked else got,
                        "meta": "set" if name in meta else m})
        elif name not in live and name not in masked:
            dev.append({"key": f"env:{name}", "production": want, "engine": None, "meta": m})
        elif name in prod and name not in masked and str(prod[name]) != str(live[name]):
            dev.append({"key": f"env:{name}", "production": want, "engine": got, "meta": m})
        elif isinstance(meta, dict) and meta and name in meta and str(meta[name]) != str(live.get(name)):
            dev.append({"key": f"env:{name}", "production": want, "engine": got, "meta": m})
    return dev


def main(argv):
    as_json = "--json" in argv
    # A URL, or a saved /api/status JSON -- the file mode is what the test suite drives, so the comparison can be
    # checked against a fabricated test config without a live serve.
    arg = next((a for a in argv if a != "--json"), "http://127.0.0.1:8080")
    url = arg if arg.startswith("http") else None
    cmd = production_args()
    if not cmd:
        print("could not get the launch line from serve/native-flashnext.sh (DRY run failed)", file=sys.stderr)
        return 2
    try:
        if url:
            with urllib.request.urlopen(f"{url.rstrip('/')}/api/status", timeout=10) as f:
                st = json.load(f)
        else:
            st = json.loads(Path(arg).read_text())
    except (OSError, ValueError) as e:
        print(f"no status from {arg}: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    url = url or arg
    dev = compare(cmd, st.get("config") or {}, st.get("meta") or {})
    livev, env_src = live_env(st)
    env_dev = compare_env(production_env(), livev, (st.get("meta") or {}).get("env"))
    if as_json:
        print(json.dumps({"production_launch": cmd, "live_launch": st.get("launch"), "deviations": dev,
                          "env_deviations": env_dev}))
    elif not dev and not env_dev:
        print(f"PRODUCTION recipe ({url}): every measured knob and every R9K_/R9700_ env name matches this "
              f"tree's launcher.")
        print(f"  {cmd}")
        print(f"  env read from {env_src}")
    else:
        if dev:
            print(f"NOT the production recipe ({url}) -- {len(dev)} measured knob(s) differ:")
            for d in dev:
                print(f"  {d['key']}: this tree says {d['production']}, the live serve says engine={d['engine']} "
                      f"meta={d['meta']}")
        if env_dev:
            print(f"\n{len(env_dev)} environment knob(s) differ -- these change which kernel runs, and the engine's "
                  f"config does not show them (env read from {env_src}):")
            for d in env_dev:
                print(f"  {d['key']}: this tree says {d['production']}, the live serve says engine={d['engine']} "
                      f"meta={d['meta']}")
        print(f"\nrestore with:")
        print(f"  STOP=1 bash serve/native-flashnext.sh")
        print(f"  tmux new-session -d -s r9700-native-flashnext 'WEBUI=same LOG=<the launch's log> "
              f"bash $PWD/serve/native-flashnext.sh 2>&1 | tee -a <log>'")
        print(f"the live box was started as: {st.get('launch')}")
    return 1 if (dev or env_dev) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
