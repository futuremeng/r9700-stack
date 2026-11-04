#!/bin/sh
# Write the launch recipe the About page shows (serve/webui.py reads it through --meta).
# META=<file> in the environment, then KEY=VALUE pairs in the order you want them displayed. Nothing is
# inferred from the running engine: this is what the launcher was given, which is the only honest source.
META=${META:?usage: META=\$HOME/x.json bash serve/webmeta.sh key=value ...}
# A value that is a JSON object or array is stored as one, so the About page can render it as a row instead of as
# a wall of escaped quotes (the recipe's env block is an object: every R9K_/R9700_ name the launch exported).
# Scalars stay strings on purpose -- "32" as a string and 32 as a number read the same on the page, and the gate
# compares them with str(), so nothing gains from changing their type here.
python3 -c 'import json, sys


def val(s):
    try:
        v = json.loads(s)
    except ValueError:
        return s
    return v if isinstance(v, (dict, list)) else s


out = {}
for a in sys.argv[2:]:
    k, _, v = a.partition("=")
    out[k] = val(v)
open(sys.argv[1], "w").write(json.dumps(out, indent=1) + "\n")' "$META" "$@"
echo "$META"
