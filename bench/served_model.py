"""Ask the serve what to call it. No bench in this tree names a model in its own source.

Why this exists: the box switched from Qwen3.8 to SuperVL on 2026-11-18, and a dozen bench tools kept posting
`"model": "Qwen3.8"` to a serve that had stopped answering that name. A stale name in a source file fails at request
time with a 404 that reads like a server bug (`The model `Qwen3.8` does not exist.` -- vLLM does not list what it
does serve), and the tool that noticed was a bench, not the serve. The serve has always published its names on
`/v1/models`; this reads them.

Resolution order:

  MODEL=<name>                       explicit override: A/B against a second engine, a LoRA, a parked checkpoint
  otherwise the FIRST id on /v1/models

The first id is the canonical one by construction, not by luck: vLLM builds `base_model_paths` in the order
`--served-model-name` is given, and its serving layer answers `base_model_paths[0].name` as *the* model name, so
whatever name the launcher puts first is the box's name and any later names would be aliases. This box serves exactly
one name, **SuperVL**, and deliberately no `Qwen3.8` alias: a client still sending the old string believes it is
talking to a model this box does not run, which is the same lie the standing-launch work of 2026-11-18 was meant to
end (notes/nemotron-super-vl.md).

resolve() takes either the root (`http://localhost:8080`) or a chat URL (`.../v1/chat/completions`), because the
benches carry both shapes.
"""

import json
import os
import urllib.error
import urllib.request

_TAILS = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/models")


def root_of(base: str) -> str:
    """http://host:port from http://host:port/v1/chat/completions (idempotent)."""
    b = (base or "").rstrip("/")
    for t in _TAILS:
        if b.endswith(t):
            b = b[: -len(t)]
            break
    return b.rstrip("/")


def served_ids(base: str, timeout: int = 10):
    """Every model id this serve advertises, in the serve's own order."""
    root = root_of(base)
    try:
        with urllib.request.urlopen(root + "/v1/models", timeout=timeout) as r:
            data = json.loads(r.read()).get("data", [])
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise SystemExit(f"{root}/v1/models is not answering ({type(e).__name__}: {e}) -- refusing to post a model "
                         f"name this process did not read from the serve. MODEL=<name> overrides the lookup.")
    return [d.get("id") for d in data if isinstance(d, dict) and d.get("id")]


def resolve(base: str, timeout: int = 10) -> str:
    """The model name to ask this serve for. Fails loudly rather than guessing a name."""
    env = os.environ.get("MODEL")
    if env:
        return env
    ids = served_ids(base, timeout=timeout)
    if not ids:
        raise SystemExit(f"{root_of(base)}/v1/models answered with no model ids -- MODEL=<name> overrides the lookup")
    return ids[0]


def served_card(base: str, timeout: int = 10):
    """(id, checkpoint path) of the first model the serve advertises.

    The path matters because a tool that counts tokens has to count them against the checkpoint that is actually
    loaded: an id from one model and a tokenizer from another mis-counts every tier without an error anywhere. vLLM's
    ModelCard carries `root` for exactly this, so it is read rather than guessed.
    """
    root = root_of(base)
    try:
        with urllib.request.urlopen(root + "/v1/models", timeout=timeout) as r:
            data = json.loads(r.read()).get("data", [])
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise SystemExit(f"{root}/v1/models is not answering ({type(e).__name__}: {e}) -- refusing to guess a "
                         f"checkpoint path.")
    for d in data:
        if isinstance(d, dict) and d.get("id"):
            return d["id"], d.get("root") or ""
    raise SystemExit(f"{root}/v1/models answered with no model cards")
