"""fp8 (e4m3) main KV cache for Qwen4Exp QSA on RDNA4: open stock's bf16-only guards, keep the scales honest.

What is already fp8-capable upstream, and what is not:

* **The writer is ready.** The fused QSA prepare Triton kernel stores the main K/V through ``_to_dst_dtype()``,
  which converts to ``tl.float8e4nv`` when the destination cache is fp8, and it passes the descales as
  ``attn._k_scale_float`` / ``attn._v_scale_float`` (vllm ``models/qwen4_exp/amd/indexer_qsa.py:323-326``).
* **The reader is not** (on ROCm): three guards say so -- ``models/qwen4_exp/amd/qsa.py:190`` (layer:
  ``cache_config.cache_dtype not in ("auto", "bfloat16")``), ``:273`` (layer: ``kv_cache_torch_dtype !=
  torch.bfloat16``) and ``:113`` (impl: ``self.kv_cache_dtype not in ("auto", "bfloat16")``). Upstream's *NVIDIA*
  reader (``models/qwen4_exp/nvidia/qsa.py:306``) already reads fp8 with those same per-layer scales; our reader is
  ``attn/qsa.py`` -> ``kernels/r9k_qsa.hip`` ``r9k_qsa_attn_f8``.

With ``R9K_QSA_KVF8=1`` and ``--kv-cache-dtype fp8_e4m3`` the layer is constructed with ``cache_dtype`` temporarily
reported as ``"bfloat16"`` -- which also keeps the *side* caches (the indexer's raw and compressed keys) bf16, as
intended, and lets the impl's own guard pass -- and afterwards the layer's ``kv_cache_dtype`` /
``kv_cache_torch_dtype`` are set to fp8, which is exactly what ``get_kv_cache_spec()`` reads. vLLM then allocates the
main pool as e4m3 and the hybrid page math doubles the attention block size (448 -> 896 tokens) to keep the attention
page at least as large as the mamba page. Both ends use the same two host floats, so the descale cannot drift: the
writer quantizes by ``layer._k_scale``, our attention kernel descales by ``layer._k_scale_float``
(``attn/qsa.py:kv_descale``), read once per layer, no per-call host sync.

Scales: ``set_default_quant_scales`` registers them at 1.0. e4m3's largest finite magnitude is 448 and the
checkpoint's own KV calibration (``kv_calibration_report_mtp.json``: K ``observed_max`` 90.5, V 11.06, divisor 200)
says nothing near the range limit, so at scale 1.0 nothing clips and the scale is a precision/headroom choice, not a
correctness one. The MTP layer ships real scales (``model-kvscales.safetensors``) and vLLM will load them into that
layer; the 12 main layers run at the 1.0 default until we calibrate them.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)

_PATCHED = False


def patch() -> bool:
    """Install the construction window. Returns False (self-disabling) when off or when the target is absent."""
    global _PATCHED
    if os.environ.get("R9K_QSA_KVF8", "0") != "1" or _PATCHED:
        return False
    if not hasattr(torch, "float8_e4m3fn"):
        logger.warning("r9700: R9K_QSA_KVF8 set but this torch has no float8_e4m3fn; ignoring")
        return False
    try:
        from vllm.models.qwen4_exp.amd.qsa import Qwen4ExpQSAAttention as Layer
    except Exception as e:                                            # a vLLM that moved the model: off, not fatal
        logger.warning("r9700: QSA fp8 KV patch not installed (%s)", e)
        return False
    from . import gate                                               # the version gate this patch never called: a pin
    gate.check("qsa_kvfp8")                                          # bump re-tests it, see compat/gate.py
    orig = Layer.__init__

    def __init__(self, *args, **kwargs):
        vc = kwargs.get("vllm_config")
        cc = getattr(vc, "cache_config", None) if vc is not None else None
        asked = str(getattr(cc, "cache_dtype", "auto")) if cc is not None else "auto"
        if not asked.startswith("fp8"):
            return orig(self, *args, **kwargs)
        prev = cc.cache_dtype
        cc.cache_dtype = "bfloat16"                                   # the guards compare against this
        try:
            orig(self, *args, **kwargs)                               # side caches bf16, impl guard passes
        finally:
            cc.cache_dtype = prev
        self.kv_cache_dtype = prev
        self.kv_cache_torch_dtype = torch.float8_e4m3fn               # get_kv_cache_spec() reads these
        impl = getattr(self, "impl", None)
        if impl is not None:
            impl.kv_cache_dtype = prev
        logger.info("r9700: QSA main KV cache as %s (k_scale %s, v_scale %s)", prev,
                    getattr(self, "_k_scale_float", "?"), getattr(self, "_v_scale_float", "?"))

    Layer.__init__ = __init__
    Layer._r9k_kvfp8 = True
    _PATCHED = True
    return True
