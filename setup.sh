#!/bin/sh
# r9700-stack on Linux: the first run brings the box up (image, build repo, checks) and can start a launch;
# later runs just check it and start one. Needs an amdgpu host with the R9700s passed through with their full
# 32 GB BAR0 (host/README.md) and docker. Nothing big is installed by this file.
#
# A Python (3.10+, with a venv that has pip) is made in .venv and holds NO torch: the serving stack lives in the
# image (docker/Dockerfile pip-installs the plugin inside it). The host Python is for the tests that need no
# GPU, for bench/, for serve/webui.py (stdlib only) and for this script.
cd "$(dirname "$0")" || exit 1
# Python 3.10+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (that is the
# separate python3-venv package), and a venv made without it has no pip
ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
# a .venv from an earlier run that failed half-way has a python but no pip: start it again
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3 python; do
    if command -v $c >/dev/null 2>&1 && ok_py $c; then PY=$c; break; fi
  done
  if [ -z "$PY" ]; then
    # only reached when the distro's python cannot make a usable venv: this asks for sudo, and says so
    echo "Python 3.10+ with a pip-capable venv is needed; installing python3-venv/python3-pip (sudo asks) ..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y python3 python3-pip
    fi
    for c in python3 python; do
      if command -v $c >/dev/null 2>&1 && ok_py $c; then PY=$c; break; fi
    done
  fi
  if [ -z "$PY" ]; then
    echo "No usable Python found. On Debian/Ubuntu: sudo apt install python3-venv python3-pip, then run"
    echo "./setup.sh again."
    exit 1
  fi
  $PY -m venv .venv || { rm -rf .venv; echo "could not create .venv: sudo apt install python3-venv"; exit 1; }
fi
exec .venv/bin/python setup.py "$@"
