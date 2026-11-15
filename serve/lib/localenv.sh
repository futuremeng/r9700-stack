# serve/lib/localenv.sh -- source this box's machine-local env file WITHOUT overriding what the caller exported.
#
# Every launcher used to say `set -a; . "$LOCAL_ENV"; set +a`, which makes the file an OVERRIDE rather than a
# default: `GATEWAY_PORT=8090 bash serve/gateway.sh` got 8080 back because the file says 8080, and serve/webui.sh's
# own documented usage line (`PORT=8100 bash serve/webui.sh  # a second launch on 8100 -> monitor on 8101`) silently
# watched 8180 on this box. It is also the order setup.py does NOT read: `launcher_var` asks the environment first,
# then the file, then the script's default, which is precisely how a tool and a launcher come to disagree about what
# a launch MEANS (notes/launch-methodology.md §10 is the whole argument for the file existing).
#
# So: a box fact is still a box fact -- it is just a default. Everything the caller already exported survives,
# untouched, and the file fills only the names nobody typed. `set -a` is kept, because the file's facts belong to
# the child processes too, not only to the script that sourced it.
: "${LOCAL_ENV:=$HOME/.config/r9700-stack/native.env}"
if [ -f "$LOCAL_ENV" ]; then
  _r9k_keep=()
  for _r9k_n in $(sed -n -E 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*/\2/p' \
                  "$LOCAL_ENV" | sort -u); do
    if [ -n "${!_r9k_n+set}" ]; then _r9k_keep+=("$_r9k_n=${!_r9k_n}"); fi
  done
  set -a; . "$LOCAL_ENV"; set +a
  for _r9k_kv in ${_r9k_keep[@]+"${_r9k_keep[@]}"}; do
    export "${_r9k_kv%%=*}=${_r9k_kv#*=}"
  done
  if [ ${#_r9k_keep[@]} -gt 0 ]; then
    echo "local env: $LOCAL_ENV (${#_r9k_keep[@]} name(s) already in the environment left alone: ${_r9k_keep[*]%%=*})"
  else
    echo "local env: $LOCAL_ENV"
  fi
fi
