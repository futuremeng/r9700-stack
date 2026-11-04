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


def production_args():
    """The command the launcher in THIS checkout would run, with no test env set.

    `FORCE=1` only to get past the "GPUs are busy" guard: DRY exits before launching anything. The parent
    environment supplies HOME (the launcher derives $VENV and $LOG from it), so the printed line is the real one.
    """
    r = subprocess.run(["bash", str(LAUNCHER)],
                       env={**os.environ, "DRY": "1", "FORCE": "1"},
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
    if as_json:
        print(json.dumps({"production_launch": cmd, "live_launch": st.get("launch"), "deviations": dev}))
    elif not dev:
        print(f"PRODUCTION recipe ({url}): every measured knob matches this tree's launcher.")
        print(f"  {cmd}")
    else:
        print(f"NOT the production recipe ({url}) -- {len(dev)} measured knob(s) differ:")
        for d in dev:
            print(f"  {d['key']}: this tree says {d['production']}, the live serve says engine={d['engine']} "
                  f"meta={d['meta']}")
        print(f"\nrestore with:")
        print(f"  STOP=1 bash serve/native-flashnext.sh")
        print(f"  tmux new-session -d -s r9700-native-flashnext 'WEBUI=same LOG=<the launch's log> "
              f"bash $PWD/serve/native-flashnext.sh 2>&1 | tee -a <log>'")
        print(f"the live box was started as: {st.get('launch')}")
    return 1 if dev else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
