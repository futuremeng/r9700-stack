#!/bin/bash
# PREFETCH A/B IS A NO-OP ON THIS BOX -- kept only as a record of why. Two independent reasons:
#
# 1. The knobs do nothing without the strategy. vLLM (model_executor/model_loader/weight_utils.py:1115) does
#      should_prefetch = safetensors_load_strategy == "prefetch"
#      if safetensors_load_strategy is None: should_prefetch = is_net_fs and fits_in_ram
#    so on EXT4 with no --safetensors-load-strategy the prefetch path is never taken and
#    --safetensors-prefetch-num-threads / --safetensors-prefetch-block-size tune a code path that does not run.
#    Confirmed in the launch log: "Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized
#    network FS (NFS/Lustre) and the checkpoint size (110.60 GiB) exceeds 90% of available RAM (40.74 GiB)."
#
# 2. Forcing it (--safetensors-load-strategy=prefetch) stages shards ahead of the iteration into host RAM.
#    The checkpoint is 110.6 GiB, the box has 93.8 GiB, and the 4 ranks already peak at ~54 GiB of RSS between
#    them; vLLM's own warning says "This may cause out-of-memory errors". Do not run it here.
#
# What the two launches DID measure (the only usable numbers, same config, no flag change):
#   01:32:54 run=1 load_s=208.4 experts_place_s=137.6   (page cache cold-ish)
#   01:35:39 run=2 load_s=103.2 experts_place_s= 34.5   (page cache warm)
# i.e. 2.02x of the cold start is page-cache state, and it is worth more than any knob vLLM exposes here.
# Use bench/coldstart_ab.sh instead: it measures the axis that actually exists, with a memory gate.
#
# The run that produced those two lines is also what killed the box -- see notes/load-oom-svm-wedge.md:
# it launched the next 4-rank load while the previous one's memory had not been reclaimed, memory exhausted,
# kswapd0 wedged on the amdgpu SVM mmu-notifier mutex, and the whole machine went into 122 s+ blocked-task
# territory. This script must never be run without the gate coldstart_ab.sh has.
cd /home/jiulu/r9700-stack || exit 1
echo "disabled: the prefetch knobs are no-ops without --safetensors-load-strategy=prefetch, and forcing that"
echo "OOMs a 93.8 GiB box with a 110.6 GiB checkpoint. See bench/coldstart_ab.sh."
exit 2
