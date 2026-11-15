# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The regression tests vLLM wrote for upstream #61174 ("Clear stale shared block-table rows", 2026-10-17),
carried here so this box can ask, at any time, whether the venv it is about to serve with still has the bug.

Why they are in this repository: the native venv is a pinned 0.32.0 wheel that CANNOT be re-fetched (see
setup.py's own warning), so an upstream fix has to be applied as a patch -- and a patch nobody can verify is a
patch that silently disappears at the next rebuild. Run this against the unpatched wheel and it fails 7/7 on
2026-11-15; run it against the patched one and it passes 7/7. That pair, on this card, is why
patches/vllm-0.32.0-61174-clear-stale-block-table-rows.patch exists.

    LD_LIBRARY_PATH=$HOME/opt/openmpi/usr/lib/x86_64-linux-gnu \
      ~/venv-r9700/bin/python -m pytest tests/test_vllm_stale_block_table_rows.py -q

Upstream's own copy marks these `skipif(not current_platform.is_cuda())`, which on a ROCm build SKIPS them --
the test for a block-table bug in the mamba state path never runs on the platform where the bug was found. The
marker is dropped here, and `vllm.platforms` is not imported for that reason. Needs a GPU (tiny allocations;
it has run beside a live launch).
"""

import pytest
import torch

from vllm.v1.worker.gpu.block_table import BlockTables

@pytest.fixture
def populated_block_tables():
    tables = BlockTables([16, 32], 8, 64, [3, 5], torch.device("cuda"))
    for req_index in range(8):
        tables.append_block_ids(
            req_index, ([req_index + 1] * 3, [req_index + 1] * 5), overwrite=True
        )
    tables.apply_staged_writes()
    tables.gather_block_tables(torch.arange(8, device="cuda"), num_reqs_padded=8)
    return tables


@pytest.mark.parametrize("previous_dummy", [False, True])
@pytest.mark.parametrize("next_dummy", [False, True])
def test_shared_block_tables_clear_rows_from_larger_batch(
    populated_block_tables, previous_dummy, next_dummy
):
    """A smaller target batch must leave no stale rows for larger draft graphs."""
    tables = populated_block_tables
    if previous_dummy:
        for table in tables.get_dummy_block_tables(8):
            table.fill_(99)  # Profiling supplies valid context/state slots.
    ptrs = [table.data_ptr() for table in tables.input_block_tables]
    idx = torch.tensor([7, 2, 0, 5, 1], device="cuda")
    for num_reqs in (3, 5, 0):
        if next_dummy:
            out = tables.get_dummy_block_tables(num_reqs)
            num_rows = num_reqs
        else:
            num_rows = num_reqs + 1
            out = tables.gather_block_tables(idx[:num_reqs], num_rows)
        for table, view, ptr in zip(tables.input_block_tables, out, ptrs):
            assert view.shape == (num_rows, table.shape[1])
            assert table.data_ptr() == ptr
            if num_rows:
                assert view.data_ptr() == ptr
            assert torch.count_nonzero(table[num_reqs:]).item() == 0
            if next_dummy:
                assert torch.count_nonzero(view).item() == 0
                view.fill_(99)
            else:
                expected = (idx[:num_reqs, None] + 1).expand(-1, table.shape[1])
                torch.testing.assert_close(view[:num_reqs], expected.to(torch.int32))


def test_external_block_table_output_preserves_shared_rows(populated_block_tables):
    """PCP-local gathers must neither clear shared rows nor reset their tracking."""
    tables = populated_block_tables
    previous = [table.clone() for table in tables.input_block_tables]
    out = tuple(torch.full_like(table[:3], -1) for table in previous)
    out_ptrs = torch.tensor(
        [table.data_ptr() for table in out], dtype=torch.uint64, device="cuda"
    )
    idx = torch.tensor([7, 2], device="cuda")
    result = tables.gather_block_tables(idx, 3, out=out, out_ptrs=out_ptrs)
    for table, before, view in zip(tables.input_block_tables, previous, result):
        torch.testing.assert_close(table, before)
        assert view[:, 0].tolist() == [8, 3, 0]
    tables.gather_block_tables(idx[:1], num_reqs_padded=1)
    for table in tables.input_block_tables:
        assert torch.count_nonzero(table[1:]).item() == 0


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_larger_draft_batch_uses_null_kpool_slots(
    populated_block_tables, use_cuda_graph
):
    """Draft padding must not resolve to another request's physical KPool slots."""
    from vllm.v1.attention.backends.mla.indexer import compute_kpool_tail_slot_mapping

    tables = populated_block_tables
    idx = torch.tensor([2, 0, 4, 1, 3], device="cuda")
    query_start_loc = torch.arange(9, dtype=torch.int32, device="cuda").clamp_max_(5)
    slot_mapping = torch.full((8,), -1, dtype=torch.int64, device="cuda")
    positions = torch.arange(8, dtype=torch.int64, device="cuda")

    def compute_slots():
        return compute_kpool_tail_slot_mapping(
            slot_mapping,
            tables.input_block_tables[0],
            query_start_loc,
            positions,
            num_actual_tokens=8,
            num_reqs=8,
            kpool=8,
        )

    if use_cuda_graph:
        compute_slots()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            slots = compute_slots()
    tables.gather_block_tables(idx, num_reqs_padded=5)
    if use_cuda_graph:
        graph.replay()
    else:
        slots = compute_slots()
    assert slots.tolist() == [24, 9, 42, 19, 36, -1, -1, -1]
