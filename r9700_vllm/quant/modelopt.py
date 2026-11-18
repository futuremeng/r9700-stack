"""modelopt_mixed checkpoints (NVIDIA's own container for the same NVFP4 bits) on gfx1201.

Nemotron-H MoE checkpoints from nvidia/ carry `quant_method: modelopt_mixed`: a per-layer algorithm map
(`hf_quant_config.json`, 41,156 entries on Super VL 120B: 41,043 NVFP4 routed-expert matrices + 113 FP8), so
`ModelOptMixedPrecisionConfig` resolves each layer to its own recipe. The NVFP4 bits are exactly the ones
quant/nvfp4.py already converts to MXFP4 for libr9k -- stock ROCm has only a per-call dequant emulation for
NVFP4 linears and no NVFP4 MoE backend at all, which is what Super VL decodes at on 4xR9700 (8.9 tok/s).

Only the container differs, and all three differences are bookkeeping:

  * the packed e2m1 lives in `w13_weight` / `w2_weight`, not CT's `w13_weight_packed` / `w2_weight_packed`;
  * the per-16 scales are float8_e4m3fn *tensors* (CT keeps raw u8 bits) -- the conversion wants `.float()`
    of them, which is the same numbers, so no repacking;
  * the fp32 global scale is `w*_weight_scale_2` and ModelOpt MULTIPLIES it: w = e2m1 * s16 * scale_2, where
    CT stores a divisor (w = e2m1 * s16 / global). So this module hands the conversion `row_div = 1/scale_2`,
    and libr9k's per-row multiplier comes out as `1/div = scale_2` -- both paths stay exact.

`R9kModelOptMixedConfig` is registered under the stock name ("modelopt_mixed") with vLLM's
`register_quantization_config`, and changes exactly one thing: an NVFP4 `RoutedExperts` layer is handed to
`R9kMxfp4MoEMethod`'s machinery (libr9k grouped MXFP4xFP8 + expert cache) with the weights converted at load,
instead of stock's backend search, which on ROCm lands on emulation for every one of the 512-expert layers.
Everything else -- the FP8 dense linears, the excluded (bf16) projections, the KV-cache recipe -- is stock.

R9K_NVFP4_MOE=native keeps the checkpoint bits and serves the experts through libr9k's grouped NVFP4 kernel
(VRAM-resident experts only; it costs the e4m3 scales' VRAM and prefill speed -- see quant/nvfp4.py).
R9K_DISABLE=nvfp4 leaves these layers on stock emulation.

Not yet: libr9k's grouped experts assumed a gated MLP -- w13's rows are the gate|up halves and the epilogue is
`silu(g)*u` (kernels/r9k_moe_mxfp4a8.hip). Nemotron-H's MoE is `activation_without_mul` with mlp_hidden_act=relu2,
so `r9k_relu2_quant_fp8` (same file) now does relu^2 + the per-row fp8 quant over a single-half GEMM1 output, and
R9700Mxfp4Experts advertises `_supports_no_act_and_mul()` plus `MoEActivation.RELU2_NO_MUL` when the loaded
libr9k.so carries that symbol. Tile configs (R9K_MOE_CFG1/2) are still keyed on the gated shapes.
"""
from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4FusedMoE,
)

from ..kernels import moe as K
from ..moe import cache as C
from ..moe import prep
from ..moe.experts import r9k_available
from .ct import R9kMxfp4MoEMethod, _cached, _off
from .nvfp4 import MOE_MODE, nvfp4_to_mxfp4, prep_native_moe

logger = init_logger("vllm." + __name__)


def make_moe_method_cls(mxfp4_method_cls):
    """modelopt's NVFP4 expert layout at create time, MXFP4 in place after loading.

    Subclasses the plugin's MXFP4 MoE method (not ModelOpt's), so `create_weights` is the only place stock
    ModelOpt code runs: we need its parameter names/shapes/weight loaders, because that is what the
    checkpoint's `*.experts.<e>.{up,down}_proj.{weight,weight_scale,weight_scale_2,input_scale}` keys land in.
    """

    class R9kModelOptNvfp4MoEMethod(mxfp4_method_cls):

        def __init__(self, moe, nvfp4_config):
            super().__init__(moe)
            self.quant_config = nvfp4_config        # create_weights reads group_size / is_checkpoint_nvfp4_*
            self.use_global_sf = False              # one scalar input scale per expert, as the checkpoint has

        def uses_weight_scale_2_pattern(self) -> bool:
            """FP4 convention: the per-tensor global scale is `weight_scale_2`, and `weight_scale` holds the block
            scales. RoutedExperts' weight loader routes on exactly this (`uses_weight_scale_2_pattern`) -- our base
            is the CT MXFP4 method, which answers False and would try to load [N, K/16] e4m3 block scales as
            scalars (RuntimeError: shape '[]' is invalid for input of size 172032)."""
            return True

        def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                           **extra_weight_attrs):
            ModelOptNvFp4FusedMoE.create_weights(self, layer, num_experts, hidden_size,
                                                 intermediate_size_per_partition, params_dtype, **extra_weight_attrs)

        def _row_div(self, layer, name: str, rows: int) -> torch.Tensor:
            """[E, N] fp32 divisor for the conversion: ModelOpt's scale_2 multiplies, the helper divides."""
            g = getattr(layer, name).data.float()                     # [E, shards] (w13) or [E] (w2)
            if g.ndim == 1:
                return (1.0 / g)[:, None].expand(-1, rows).contiguous()
            return (1.0 / g).repeat_interleave(rows // g.shape[1], dim=1).contiguous()

        @staticmethod
        def _convert(packed_p, scale_p, row_div):
            """Per expert: requant in place into the packed storage, e8m0 [E, N, K/32] u8 returned."""
            E = packed_p.shape[0]
            s_out = torch.empty((E, packed_p.shape[1], packed_p.shape[2] * 2 // 32), dtype=torch.uint8,
                                device=scale_p.device)
            for e in range(E):
                p, s = nvfp4_to_mxfp4(packed_p.data[e], scale_p.data[e], row_div[e])
                packed_p.data[e].copy_(p)
                s_out[e].copy_(s)
            return s_out

        def process_weights_after_loading(self, layer) -> None:
            for pre in ("w13", "w2"):                               # modelopt names -> what the prep expects
                w = getattr(layer, f"{pre}_weight")
                setattr(layer, f"{pre}_weight_packed", prep.as_param(w.data, w))
                delattr(layer, f"{pre}_weight")
            div13 = self._row_div(layer, "w13_weight_scale_2", layer.w13_weight_packed.shape[1])
            div2 = self._row_div(layer, "w2_weight_scale_2", layer.w2_weight_packed.shape[1])
            resident = not (C.is_host(layer.w13_weight_packed) or C.is_host(layer.w2_weight_packed))
            native = MOE_MODE == "native" and resident
            if native:
                prep_native_moe(layer, div13, div2)                 # exact bits; e4m3 scales .view(u8) inside prep
                logger.info_once("r9700: modelopt NVFP4 MoE experts served natively by libr9k (exact checkpoint "
                                 "bits; R9K_NVFP4_MOE=mxfp4 converts at load instead)")
                self._install_kernel(layer)                         # no expert cache: native experts are VRAM-resident
            else:
                if MOE_MODE == "native":
                    logger.info_once("r9700: modelopt NVFP4 MoE experts are host-offloaded: converting to MXFP4 "
                                     "(the expert cache is MXFP4-only)")
                s13 = self._convert(layer.w13_weight_packed, layer.w13_weight_scale, div13)
                s2 = self._convert(layer.w2_weight_packed, layer.w2_weight_scale, div2)
                layer.w13_weight_scale = Parameter(s13, requires_grad=False)
                layer.w2_weight_scale = Parameter(s2, requires_grad=False)
                for n in ("w13_weight_scale_2", "w2_weight_scale_2", "w13_input_scale", "w2_input_scale"):
                    if hasattr(layer, n):
                        delattr(layer, n)
                logger.info_once("r9700: modelopt NVFP4 MoE experts converted to MXFP4 at load -> libr9k "
                                 "(e2m1 * s16 * scale_2 -> e8m0 per 32; R9K_DISABLE=nvfp4 keeps stock emulation)")
                super().process_weights_after_loading(layer)

    return R9kModelOptNvfp4MoEMethod


@register_quantization_config("modelopt_mixed")
class R9kModelOptMixedConfig(ModelOptMixedPrecisionConfig):
    """Stock modelopt_mixed, except NVFP4 routed experts -> libr9k (converted to MXFP4 at load)."""

    def get_quant_method(self, layer, prefix):
        if r9k_available() and not _off("nvfp4") and isinstance(layer, RoutedExperts) \
                and self._resolve_quant_algo(prefix) == "NVFP4":
            # libr9k serves two MoE shapes: the gated one (w13 rows = gate|up halves, epilogue silu(g)*u) and,
            # since the relu^2 act+quant kernel landed, the ungated one Nemotron-H uses (`activation_without_mul`,
            # mlp_hidden_act=relu2: one `up` matrix, up_proj [2688,512] packed / down_proj [1024,1344] around the
            # 1024-wide MoE latent). Anything else goes back to stock -- on ROCm that means the per-call NVFP4
            # emulation, but it loads. R9K_MOE_* tile configs are keyed on the gated shapes for now.
            act = getattr(layer.moe_config, "activation", None)
            served = (layer.moe_config.is_act_and_mul and act is MoEActivation.SILU) \
                or (act is MoEActivation.RELU2_NO_MUL and K.relu2_quant_supported())
            if not served:
                logger.info_once("r9700: modelopt NVFP4 MoE left on stock: libr9k's grouped MXFP4 experts do not "
                                 "serve this activation (gated=%s activation=%s relu2 kernel=%s)",
                                 layer.moe_config.is_act_and_mul, getattr(act, "value", act), K.relu2_quant_supported())
                return super().get_quant_method(layer, prefix)
            return _cached(make_moe_method_cls, R9kMxfp4MoEMethod)(layer.moe_config, self.nvfp4_config)
        return super().get_quant_method(layer, prefix)
