#!/bin/sh
# Write the launch recipe the About page shows (serve/webui.py reads it through --meta).
# META=<file> in the environment, then KEY=VALUE pairs in the order you want them displayed. Nothing is
# inferred from the running engine: this is what the launcher was given, which is the only honest source.
META=${META:?usage: META=\$HOME/x.json bash serve/webmeta.sh key=value ...}
python3 -c 'import json,sys; open(sys.argv[1],"w").write(json.dumps(dict(a.split("=",1) for a in sys.argv[2:]),indent=1)+"\n")' "$META" "$@"
echo "$META"
