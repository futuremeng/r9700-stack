#!/usr/bin/env python3
"""The VLLM_PLUGINS value for a merged monitor: ours ADDED, nothing already running REMOVED.

Why this file exists. `VLLM_PLUGINS` is not an "also load these" list -- for the groups other than
endpoint plugins it is a STRICT ALLOWLIST (vllm/plugins/__init__.py: `load_plugins_by_group` loads
everything when `VLLM_PLUGINS` is None, and only the named ones when it is set). An endpoint plugin,
meanwhile, is loaded ONLY when named. So the obvious line

    VLLM_PLUGINS=r9700_webui

turns "load every discovered plugin" into "load exactly this one" -- which silently disables this repo's
own `r9700` plugin, the one that teaches compressed-tensors the MXFP4-pack-quantized scheme. It cost a
four-minute launch to find out: the engine died with
`NotImplementedError: No compressed-tensors compatible scheme was found for layer_name=...in_proj_qkvz`.

So the allowlist is built from what is actually discoverable, never typed by hand.

  usage: webui-plugins.py [--require r9700] [--quiet]
"""

import sys
from importlib.metadata import entry_points

GROUPS = ("vllm.general_plugins", "vllm.stat_logger_plugins", "vllm.endpoint_plugins",
          "vllm.v1.sample logits processors", "vllm.platform_plugins")
OURS = "r9700_webui"


def allowlist(existing=(), require=()):
    """Everything vLLM would load anyway, plus ours, plus whatever the caller insists on."""
    have = set(existing)
    for group in GROUPS:
        try:
            have.update(e.name for e in entry_points(group=group))
        except Exception:                     # a malformed distribution must not break a launch
            pass
    have.add(OURS)
    have.update(require)
    return ",".join(sorted(have))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    require = []
    quiet = "--quiet" in argv
    if "--require" in argv:
        require = [argv[argv.index("--require") + 1]]
    found = set()
    for group in GROUPS:
        try:
            found.update(e.name for e in entry_points(group=group))
        except Exception:
            pass
    out = allowlist([p for p in __import__("os").environ.get("VLLM_PLUGINS", "").split(",") if p], require)
    print(out)
    if not quiet:
        print(f"# VLLM_PLUGINS={out}", file=sys.stderr)
        for name in require:                     # naming a plugin that is not installed would silently load
            if name not in found:                # nothing: the launch would run without the repo's kernels
                print(f"WARN plugin {name!r} is not discoverable in this interpreter: the launch will run "
                      f"without it (is the repo installed? pip install -e .)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
