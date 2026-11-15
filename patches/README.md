# patches/

Upstream fixes this box needs **before** upstream ships them to the artifact it actually runs.

That is only ever the native path, and only because that artifact is frozen: `~/venv-r9700` holds a pinned
vLLM `0.32.0+rocm…` wheel from a per-commit index that has since expired, so it cannot be re-fetched, upgraded,
or rebuilt from a recipe (see `setup.py`'s own warning in `native()` — *"the venv you have IS the artifact of
record — keep it"*). An upstream fix therefore arrives as a diff against the installed tree, and `setup.sh`
reports its state as part of the box walk, next to the version numbers, because a patched wheel and a stock one
are different artifacts and the difference only surfaces at 05:00 while a fault is being attributed.

## What is here

| File | Upstream | Installed file | Marker that proves it is applied |
| --- | --- | --- | --- |
| `vllm-0.32.0-61174-clear-stale-block-table-rows.patch` | vllm-project/vllm#61174, 2026-10-17 | `vllm/v1/worker/gpu/block_table.py` | `_num_dirty_rows` |
| `vllm-0.32.0-block_table.py.unpatched` | — | the file as the wheel shipped it | keep for a clean revert |

`setup.py`'s `VPATCHES` is the list the box walk checks; a new patch goes in both places.

## Apply, verify, revert

```bash
SP=$(~/venv-r9700/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
patch -p1 -d "$SP" < patches/vllm-0.32.0-61174-clear-stale-block-table-rows.patch

# the test upstream wrote for it, run against THIS venv (7 cases; needs a GPU, tiny allocations)
LD_LIBRARY_PATH=$HOME/opt/openmpi/usr/lib/x86_64-linux-gnu \
  ~/venv-r9700/bin/python -m pytest tests/test_vllm_stale_block_table_rows.py -q

# revert
patch -R -p1 -d "$SP" < patches/vllm-0.32.0-61174-clear-stale-block-table-rows.patch
```

Measured on this box, 2026-11-15: **7 failed** against the wheel as installed, **7 passed** after the patch.

How to tell whether a given launch carries the patches at all, without comparing mtimes by hand: the engine
says so at load, one line in its launch log --

```
grep -m1 "venv patch state" ~/logs/setup-<that launch>.log
r9700 webui loaded in vllm 0.32.0 (this process) -- venv patch state: #61174 present
```

That line comes from the plugin (`serve/plugin/r9700_webui/__init__.py`, `patch_text()`), and `tests/test_webui.py`
pins that its list matches `setup.py`'s `VPATCHES` marker for marker. (It first shipped mid-hunt, so launches before
2026-11-15 09:45 have no such line; for those, the mtime/start-time argument in `notes/…-oob.md` §14 is the proof.)

What it is **not**: the fix for the mixed-step device faults of `notes/bug-causal-conv1d-oob.md`. A soak hit the
identical fault (`[gfxhub]` read at `SQC (data)`, same card) **250 s** into a launch that carried this patch,
against 431 s and 536 s without it — faster, if anything (§14). It is kept because it fixes a real correctness
bug that this box drives straight into: block ids read out of rows a previous, larger batch left behind.
That test file is upstream's own (`tests/v1/worker/test_gpu_block_table.py` at commit `61bb5908`), copied here
with one deliberate change — upstream's `skipif(not current_platform.is_cuda())` skips it on every ROCm build,
so the regression test for a bug in the mamba state path never runs on the platform where that bug bites.

## Two things that are easy to get wrong

- **A patch to the venv does not touch a running engine.** The process is the artifact; compare the launch's
  start time with the patch's mtime (`stat -c %y patches/<name>` and `.r9700-state/launch.json`'s `started`)
  before concluding that the box "has" the fix.
- **A clean soak is not proof, and a patched launch is not a fixed launch.** The fault family of
  `notes/bug-causal-conv1d-oob.md` reproduces in ~4-9 minutes of `tools/soak-mixed-step.py`; #61174 did not stop
  it, so "the box is patched" must never be reported as "the fault is understood". Measure it, then say which of
  the two you have.
