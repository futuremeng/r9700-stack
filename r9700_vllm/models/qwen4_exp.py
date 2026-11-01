"""Qwen4Exp (Qwen3.8-Flash-Next) model classes for gfx1201, registered over the stock architectures through
``ModelRegistry.register_model`` (the documented way to override a built-in model). Each is a thin subclass of
the stock ROCm class; forward/compute graphs are stock.

What the subclasses add, all around construction and weight loading:
  * checkpoint compat for tcclaviger's quantizations: ``*.weight_scale_inv`` -> ``*.weight_scale`` (DeepSeek-
    style block-fp8 scale names); drop the fork's fp8-attention q/k/v scales (no stock slot, bf16 KV); MTP: drop
    the fork-private 4-bit draft head (``mtp.lm_head.weight_q4/_scale/_zero``) and, with R9K_MTP_MLP=mxfp4
    (default), re-quantize the MTP experts' fp8 block-128 weights to MXFP4 on load (640-wide experts cannot
    shard by 128 at TP2; quant/ct.py drops that CT group). Draft-only: affects MTP acceptance, never output.
  * construction scope: int6 PLE table (ple/int6.py) when the checkpoint's PLE shards are int6, and exact-size
    pinning for vLLM's UVA expert offload (utils/hostmem.py).
  * optional quantized LM heads, R9K_TARGET_LMHEAD / R9K_DRAFT_LMHEAD = fp8|mxfp4 (models/lm_heads.py).
  * our QSA sparse attention (attn/qsa.py, kernels/r9k_qsa.hip) bound on every full-attention layer's impl
    after construction; R9K_QSA=stock keeps vLLM's Triton kernel.

Unavoidable internals touched (both scoped to ``__init__`` and restored): the ``PLEVocabParallelEmbedding`` name
in ``vllm.models.qwen4_exp.amd.ple_layer`` (stock constructs it without a quant config or class hook) and
``torch.Tensor.pin_memory``. Upstream candidates: an embedding-class hook / quant_config for the PLE table, and
exact-size pinning in the UVA offloader.
"""
from __future__ import annotations

import contextlib
import os
import re
import time

import torch

from vllm.logger import init_logger
from vllm.models.qwen4_exp.amd import ple_layer as _ple_layer
from vllm.models.qwen4_exp.amd.model import Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration
from vllm.models.qwen4_exp.amd.mtp import Qwen4ExpMTP

from . import lm_heads
from ..quant.nvfp4 import quantize_mxfp4_search
from ..utils import prof, prestage
from ..attn import qsa as _qsa
from ..ple import short_conv as _ple_conv
from .. import hc as _hc
from .. import router as _router
from ..moe import shared as _shared
from ..moe import route as _route
from ..moe import fold as _fold
from ..comm import pipe as _pipe
from ..compat import memsnap as _memsnap

logger = init_logger("vllm." + __name__)

_DROP_MTP_HEAD = re.compile(r"(^|\.)mtp\.lm_head\.weight_(q4|scale|zero)$")
_DROP_ATTN_SCALE = re.compile(r"\.self_attn\.(attn\.)?[qkv]_scale$")     # bf16 KV: all unused
_MTP_MLP_FP8 = re.compile(r"(^|\.)mtp\.layers\.\d+\.mlp\.(experts\.\d+|shared_expert)\.(gate|up|down)_proj\."
                          r"weight(_scale_inv|_scale)?$")
_MTP_EXPERTS_FUSED = re.compile(r"(^|\.)mtp\.layers\.\d+\.mlp\.experts\.(gate_up|down)_proj$")


def rename_scale(name: str) -> str:
    """``weight_scale_inv`` (dequant multiplier, same values/shape as CT's ``weight_scale``) -> CT's name."""
    return name[: -len("_inv")] if name.endswith(".weight_scale_inv") else name


def fp8_block_to_mxfp4(base: str, w: torch.Tensor, scale: torch.Tensor, block: int = 128):
    """fp8 [N, K] + block scales [ceil(N/128), ceil(K/128)] -> CT MXFP4 (``weight_packed`` [N, K/2] u8,
    ``weight_scale`` [N, K/32] E8M0 u8) under ``base``."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wf = w.to(dev).float()
    N, K = wf.shape
    s = scale.to(dev).float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    packed, e8m0 = lm_heads.quantize_mxfp4((wf * s).to(torch.bfloat16))
    yield f"{base}.weight_packed", packed.cpu()
    yield f"{base}.weight_scale", e8m0.cpu()


def target_weights(weights):
    """Target-model checkpoint stream: drop unused attention scales, rename block-fp8 scales; log the failing
    tensor if loading raises (stock only says which parameter).

    With R9K_LOAD_PROF=1 it also times the two halves of the load: pulling a tensor out of the safetensors
    shards (the `for` step) against what the consumer does with it -- the consumer is vLLM's weight_loader, and
    for this checkpoint that is where the plugin's fold repack and the int6 PLE copy happen. Measured floor on
    this box: the raw 110.6 GiB read is ~19 s at 5.8 GB/s, a dummy (no-content) load is 8.8 s, and the real
    load is 143.8 s, so the ~125 s in between is what this is for."""
    last = [None]

    def gen():
        pull = place = 0.0
        seen: dict[str, list] = {}
        if _PROF:
            prof.snap("target start")
            prof.start_stacks()
            prof.install_copy_probe()     # must be installed before any weight is placed
        from ..moe import h2dbatch
        h2dbatch.install()                # batch stock's per-expert H2D copies (off unless R9K_MOE_BATCH_H2D=1)
        t = time.perf_counter()
        tstream = t                            # stream clock, for reporting when each group arrives
        for n, w in weights:
            now = time.perf_counter(); pull += now - t
            if _DROP_ATTN_SCALE.search(n):
                t = now
                continue
            n = rename_scale(n)
            last[0] = n
            if _PROF:
                prof.set_current(n)
            yield n, w
            now = time.perf_counter(); place += now - t      # weight_loader: fold repack, int6 PLE copy, ... 
            if _PROF:
                g = seen.setdefault(_group(n), [0, 0.0, 0, now - tstream, 0.0])
                g[0] += 1
                g[1] += w.numel() * w.element_size()
                g[2] += now - t
                g[4] = now - tstream
            t = now
        h2dbatch.flush_all()              # every staged byte has to reach the device before the stream ends
        from ..utils import hostmem          # late pinning: register the PLE tables now, not at construction
        hostmem.finalize_late_pins()
        if _PROF:
            prof.dump_stacks("target pass")      # which functions, not just which tensor group
            prof.report_copies("target pass")
            prof.stop_stacks()
            parts = ", ".join(f"{k} {v[0]}t/{v[1] / 2**30:.1f}GiB/{v[2]:.1f}s" for k, v in sorted(seen.items()))
            logger.info("r9700 load profile (target): pull %.1f s, place %.1f s | %s", pull, place, parts)
            for k, v in sorted(seen.items(), key=lambda kv: -kv[1][2])[:4]:
                logger.info("r9700 load profile: %7.1f s on %s (%s tensors, %.1f GiB checkpoint bytes a rank)",
                            v[2], k, v[0], v[1] / 2**30)
            for k, v in sorted(seen.items(), key=lambda kv: kv[1][3]):
                logger.info("r9700 load profile: %-8s first tensor at +%5.1f s of the stream, last at +%5.1f s",
                            k, v[3], v[4])
            prof.snap("target end")     # majflt here = bytes the target pass took from disk rather than cache
        # E1: the draft pass is about to re-read its bf16 MTP weights, which four ranks' anon (~50 GiB, 9.7 GiB a
        # rank of it pinned PLE table) have in general just evicted. Queue the readahead while the target pass is
        # still finishing, once per rank; the page cache is shared, so three of the four are no-ops.
        prestage.prestage()
    return gen(), last


_PROF = os.environ.get("R9K_LOAD_PROF", "") == "1"
_GROUP = [(re.compile(r"(^|\.)mtp\."), "mtp"),
          (re.compile(r"ngram|(^|\.)ple\."), "ple"),
          (re.compile(r"\.experts\."), "experts"),
          (re.compile(r"hyper|(^|\.)hc\."), "hc"),
          (re.compile(r"lm_head"), "head")]


def _group(name: str) -> str:
    for rx, g in _GROUP:
        if rx.search(name):
            return g
    return "other"


def mtp_weights(weights, mtp_mxfp4: bool):
    dropped, pend = [], {}
    requant = 0.0
    rq = [0.0, 0.0, 0.0, 0.0]      # _PROF only: [to-GPU (pageable H2D from the mmap'd pages), search, cat+to-CPU, calls]
    n_t, n_b, t_all = 0, 0, time.perf_counter()
    for name, w in weights:
        if _DROP_MTP_HEAD.search(name):
            dropped.append(name)
            continue
        if _DROP_ATTN_SCALE.search(name):
            continue
        if _PROF:
            n_t += 1
            n_b += w.numel() * w.element_size()
        # HF rev 6fe3e746 ships the MTP MLP in bf16 (fused experts) instead of fp8
        if mtp_mxfp4 and (_MTP_EXPERTS_FUSED.search(name) or (_MTP_MLP_FP8.search(name) and name.endswith(".weight"))) \
                and w.dtype == torch.bfloat16:
            # MSE-searched block exponents (quant/nvfp4.py): 3-5% lower error than round-to-nearest on bf16 MLP weights.
            # The chunk row count is the load-time VRAM peak: each chunk goes to fp32 on the device, and the search
            # holds a couple of fp32 copies of it (2^18 rows x 2560 cols = 2.68 GiB each). Lower it on a card that
            # has no room for that (4x R9700 TP4 with the 512-expert Flash-Next rev: 5 GiB of headroom left).
            rows = int(os.environ.get("R9K_MTP_QUANT_ROWS", 1 << 18))
            t0 = time.perf_counter()
            if _PROF:
                q = []
                for x in w.reshape(-1, w.shape[-1]).split(rows):
                    torch.cuda.synchronize()      # attribution: the search is async, so wall time needs a barrier
                    ta = time.perf_counter()
                    xc = x.cuda().float()
                    torch.cuda.synchronize()
                    tb = time.perf_counter()
                    q.append(quantize_mxfp4_search(xc))
                    torch.cuda.synchronize()
                    tc = time.perf_counter()
                    rq[0] += tb - ta
                    rq[1] += tc - tb
                    del xc
                pk = torch.cat([p for p, _ in q])
                sc = torch.cat([s for _, s in q])
                rq[2] += time.perf_counter() - tc
                rq[3] += 1
                yield f"{name}_packed", pk.cpu().reshape(*w.shape[:-1], -1)
                yield f"{name}_scale", sc.cpu().reshape(*w.shape[:-1], -1)
                requant += time.perf_counter() - t0
                continue
            q = [quantize_mxfp4_search(x.cuda().float()) for x in w.reshape(-1, w.shape[-1]).split(rows)]
            requant += time.perf_counter() - t0
            yield f"{name}_packed", torch.cat([p for p, _ in q]).cpu().reshape(*w.shape[:-1], -1)
            yield f"{name}_scale", torch.cat([s for _, s in q]).cpu().reshape(*w.shape[:-1], -1)
            continue
        if mtp_mxfp4 and _MTP_MLP_FP8.search(name):
            base = name.rsplit(".", 1)[0]
            kind = "scale" if name.endswith(("weight_scale_inv", "weight_scale")) else "w"
            pend.setdefault(base, {})[kind] = w
            if len(pend[base]) == 2:
                d = pend.pop(base)
                yield from fp8_block_to_mxfp4(base, d["w"], d["scale"])
            continue
        yield rename_scale(name), w
    if pend:
        logger.warning("r9700: unpaired MTP fp8 tensors left: %s", list(pend)[:4])
    if _PROF:
        logger.info("r9700 load profile (draft): bf16->MXFP4 requant %.1f s (%s tensors, %.1f GiB a rank, "
                    "%.1f s wall)", requant, n_t, n_b / 2**30, time.perf_counter() - t_all)
        logger.info("r9700 load profile (draft requant): to-GPU %.1f s, search %.1f s, cat %.1f s over %d calls "
                    "(a lone quantize_mxfp4_search costs 0.39 s of launch + 4.3 GiB/s; the to-CPU transfer lands in pull, not "
                    "here)", *rq[:3], int(rq[3]))
        prof.snap("draft end")          # majflt delta vs "target end" = the duplicate read, measured
    if dropped:
        logger.info_once("r9700: ignored fork-private MTP draft head tensors: %s", ", ".join(dropped))


@contextlib.contextmanager
def construction_scope():
    """Stock construction with the int6 PLE embedding class (int6 checkpoints only) and exact pinning.

    The PLE layer instantiates its table through a name in its own module: `PLEVocabParallelEmbedding` up to the
    2026-09 vLLM, `Qwen4ExpPLEDeviceEmbedding` / `Qwen4ExpPLEPinnedHostEmbedding` (chosen by engram_config) from
    2026-10. Whichever is there is swapped for the int6 class (built on the device base: our table is a UVA view
    of pinned host rows either way) for the duration of the construction, then restored."""
    from ..ple.int6 import checkpoint_ple_format, make_int6_embedding_cls
    from ..utils.hostmem import exact_pinning
    names = [n for n in ("Qwen4ExpPLEDeviceEmbedding", "Qwen4ExpPLEPinnedHostEmbedding", "PLEVocabParallelEmbedding")
             if getattr(_ple_layer, n, None) is not None]
    swap = bool(names) and checkpoint_ple_format() == "int6"
    if checkpoint_ple_format() == "int6" and not names:
        raise RuntimeError("r9700: vLLM's ple_layer exposes none of the PLE embedding classes this hook knows "
                           "(PLEVocabParallelEmbedding, Qwen4ExpPLEDeviceEmbedding); int6 PLE needs a new "
                           "construction hook for this vLLM version")
    saved = {n: getattr(_ple_layer, n) for n in names}
    if swap:
        base = saved.get("Qwen4ExpPLEDeviceEmbedding") or saved.get("PLEVocabParallelEmbedding")
        cls = make_int6_embedding_cls(base)
        for n in names:
            setattr(_ple_layer, n, cls)
    try:
        with exact_pinning():
            yield
    finally:
        if swap:
            for n, c in saved.items():
                setattr(_ple_layer, n, c)


def _head_fmt(env: str) -> str | None:
    fmt = os.environ.get(env, "").lower()
    return fmt if fmt in ("fp8", "mxfp4") else None


def _make_shadow(owner, head, env: str, what: str) -> None:
    fmt = _head_fmt(env)
    if fmt and isinstance(getattr(head, "weight", None), torch.Tensor) and head.weight.device.type == "cuda":
        # plain attribute (not a registered submodule): the shadow shares lm_head's weight
        object.__setattr__(owner, "_r9k_head", lm_heads.shadow(head, fmt, what))


def _head(owner):
    head = owner.__dict__.get("_r9k_head")
    return owner.lm_head if head is None else head


def _load_target(self, super_load, weights):
    gen, last = target_weights(weights)
    try:
        return super_load(gen)
    except Exception as e:
        logger.error("r9700: weight load failed at checkpoint tensor %s: %s", last[0], e)
        raise


class R9kQwen4ExpForConditionalGeneration(Qwen4ExpForConditionalGeneration):
    # one GDN state page per request under speculative decoding (models/gdn.py): vLLM sizes the padded mamba page
    # from these before any layer exists
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from .gdn import onepage_state_shape_from_config
        return onepage_state_shape_from_config(Qwen4ExpForConditionalGeneration, vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from .gdn import onepage_state_dtype_from_config
        return onepage_state_dtype_from_config(Qwen4ExpForConditionalGeneration, vllm_config)

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config):      # the hook vLLM prefers when a model defines it
        from .gdn import onepage_specs_from_config
        return onepage_specs_from_config(Qwen4ExpForConditionalGeneration, vllm_config)


    def __init__(self, *, vllm_config, prefix: str = "model") -> None:
        with construction_scope():
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        _qsa.install(self)
        _ple_conv.install(self)
        _hc.install()
        _router.install(self)
        _hc.install_mix(self)
        _shared.install(self)
        _route.install(self)
        _fold.install(self)
        _pipe.install(self)
        _memsnap.install(self)

    def load_weights(self, weights):
        loaded = _load_target(self, super().load_weights, weights)
        _hc.quantize_fp8(self)                         # fp8 hc weights need the loaded checkpoint
        _make_shadow(self, self.language_model.lm_head, "R9K_TARGET_LMHEAD", "target")
        return loaded

    def compute_logits(self, hidden_states):
        head = self.__dict__.get("_r9k_head")
        if head is None:
            return super().compute_logits(hidden_states)
        return self.language_model.logits_processor(head, hidden_states)


class R9kQwen4ExpForCausalLM(Qwen4ExpForCausalLM):
    # one GDN state page per request under speculative decoding (models/gdn.py): vLLM sizes the padded mamba page
    # from these before any layer exists
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from .gdn import onepage_state_shape_from_config
        return onepage_state_shape_from_config(Qwen4ExpForCausalLM, vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from .gdn import onepage_state_dtype_from_config
        return onepage_state_dtype_from_config(Qwen4ExpForCausalLM, vllm_config)

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config):      # the hook vLLM prefers when a model defines it
        from .gdn import onepage_specs_from_config
        return onepage_specs_from_config(Qwen4ExpForCausalLM, vllm_config)


    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        with construction_scope():
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        _qsa.install(self)
        _ple_conv.install(self)
        _hc.install()
        _router.install(self)
        _hc.install_mix(self)
        _shared.install(self)
        _route.install(self)
        _fold.install(self)
        _pipe.install(self)
        _memsnap.install(self)

    def load_weights(self, weights):
        loaded = _load_target(self, super().load_weights, weights)
        _hc.quantize_fp8(self)                         # fp8 hc weights need the loaded checkpoint
        _make_shadow(self, self.lm_head, "R9K_TARGET_LMHEAD", "target")
        return loaded

    def compute_logits(self, hidden_states):
        return self.logits_processor(_head(self), hidden_states)


class R9kQwen4ExpMTP(Qwen4ExpMTP):

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        with construction_scope():
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        _qsa.install(self)
        _ple_conv.install(self)
        _hc.install()
        _router.install(self)
        _hc.install_mix(self)
        _shared.install(self)
        _route.install(self)
        _fold.install(self)
        _pipe.install(self)
        _memsnap.install(self)

    def load_weights(self, weights):
        loaded = super().load_weights(mtp_weights(weights, os.environ.get("R9K_MTP_MLP", "mxfp4") == "mxfp4"))
        _hc.quantize_fp8(self)                         # fp8 hc weights need the loaded checkpoint
        _make_shadow(self, self.lm_head, "R9K_DRAFT_LMHEAD", "MTP draft")
        return loaded

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        return self.logits_processor(_head(self), hidden_states)


ARCHS = {
    "Qwen4ExpForConditionalGeneration": "r9700_vllm.models.qwen4_exp:R9kQwen4ExpForConditionalGeneration",
    "Qwen4ExpForCausalLM": "r9700_vllm.models.qwen4_exp:R9kQwen4ExpForCausalLM",
    "Qwen4ExpMTP": "r9700_vllm.models.qwen4_exp:R9kQwen4ExpMTP",
}
