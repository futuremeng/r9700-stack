#!/usr/bin/env python3
"""bench/mm_encoder.py -- run the checkpoint's OWN vision tower outside the serve, and hand the LM its embeddings.

Why (the direction the box is taking): Strata's "mmproj" is not a better encoder -- its GGUF metadata says
`clip.projector_type=qwen3vl_merger`, 27 blocks, hidden 1152, patch 16, merge 2, mean/std 0.5, projection_dim 2560,
which is the same tower this checkpoint carries as 333 BF16 `model.visual.*` tensors and which vLLM instantiates at
`vllm/models/qwen4_exp/amd/model.py:871` (`self.visual = Qwen3_VisionTransformer(...)`). So what a separate encoder
can change is **what the tower is fed and how the pixels are cut** -- which is worth measuring, because the engine's
own preprocessing is the one part of the image path we cannot otherwise A/B.

This script is the encoder side: load the tower (CPU, bf16, the checkpoint's own weights), preprocess one image two
ways, and print what the LM would receive:

  hf     the HuggingFace fast image processor at the launch's band -- the path the serve uses today, the CONTROL
  mtmd   a llama.cpp-style cut: bicubic to multiples of patch*merge inside OUR band, mean/std 0.5, the frame
         duplicated across the temporal axis, flattened in HF's (t, h, w, C, temporal, p, p) order

Run: LD_LIBRARY_PATH=/home/jiulu/opt/openmpi/usr/lib/x86_64-linux-gnu \
     ~/venv-r9700/bin/python3 bench/mm_encoder.py --image /path/to.png [--band 1048576 4194304]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

CKPT = os.environ.get("MODEL", os.path.expanduser("~/models/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ"))


def visual_state_dict():
    """The checkpoint's vision tensors, name-mapped onto HF's `blocks.N.*` / `patch_embed.*` / `merger.*`."""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    out, prefix = {}, "model.visual."
    for shard in sorted({v for k, v in idx.items() if k.startswith(prefix)}):
        with safe_open(os.path.join(CKPT, shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    out[k[len(prefix):]] = f.get_tensor(k)
    return out


def mtmd_cut(img, min_px, max_px, patch, merge, temporal=2):
    """A llama.cpp/mtmd-style resize + normalize + patchify. Returns ([rows, C, temporal, patch, patch], (1, hp, wp))
    in the layout the tower consumes -- rows are ordered t, h, w and the same frame fills the temporal axis."""
    grid = patch * merge
    w, h = img.size
    f = 1.0
    while w * h * f * f < min_px or w * h * f * f > max_px:
        f *= np.sqrt((min_px if w * h * f * f < min_px else max_px) / (w * h * f * f))
    h2 = max(grid, int(round(h * f / grid)) * grid)
    w2 = max(grid, int(round(w * f / grid)) * grid)
    im = img.convert("RGB").resize((w2, h2), 3)                       # 3 = BICUBIC, the filter mtmd uses
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5                                               # the checkpoint's image_mean / image_std
    hp, wp = h2 // patch, w2 // patch
    t5 = x.reshape(hp, patch, wp, patch, 3).transpose(0, 2, 4, 1, 3)  # (hp, wp, C, patch, patch)
    tile = np.broadcast_to(t5[:, :, :, None, :, :], (hp, wp, 3, temporal, patch, patch))
    pix = torch.from_numpy(np.ascontiguousarray(tile.reshape(hp * wp, 3, temporal, patch, patch)))
    return pix, (1, hp, wp)


def hf_cut(img, band):
    """The control: the checkpoint's own HF fast image processor, at the launch's pixel band."""
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(CKPT)
    errs = []
    for kw in ({"size": {"shortest_edge": band[0], "longest_edge": band[1]}},
               {"min_pixels": band[0], "max_pixels": band[1]}):
        try:
            out = proc(images=img, return_tensors="pt", **kw)
            return out["pixel_values"], out["image_grid_thw"][0].tolist()
        except Exception as e:
            errs.append(f"{list(kw)}: {type(e).__name__} {e}")
    raise SystemExit("the HF processor took neither band form: " + " | ".join(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--band", type=int, nargs=2, default=[1048576, 4194304])
    ap.add_argument("--tag", default="mmenc")
    a = ap.parse_args()
    from PIL import Image
    img = Image.open(a.image)
    cfg = json.load(open(os.path.join(CKPT, "config.json")))["vision_config"]
    print(f"image {a.image} {img.size} band {a.band} ckpt {os.path.basename(CKPT)}")

    sd = visual_state_dict()
    print(f"vision tensors {len(sd)} dtypes {sorted({str(v.dtype) for v in sd.values()})}")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionConfig as VC
        net = Qwen3VLVisionModel(VC(**{k: v for k, v in cfg.items() if k != "model_type"}))
    except Exception as e:
        print("config ctor path failed:", type(e).__name__, e)
        net = Qwen3VLVisionModel._from_config(Qwen3VLVisionModel.config_class(**cfg))
    torch.set_grad_enabled(False)
    net = net.to(dtype=torch.bfloat16).eval()
    have = net.state_dict()
    miss = sorted(set(sd) - set(have))
    print(f"loaded {len(sd) - len(miss)}/{len(sd)}" + (f"  MISSING {miss[:6]}" if miss else ""))
    net.load_state_dict({k: v for k, v in sd.items() if k in have}, strict=False)

    for how in ("hf", "mtmd"):
        t0 = time.time()
        if how == "hf":
            pix, grid = hf_cut(img, a.band)
        else:
            pix, grid = mtmd_cut(img, a.band[0], a.band[1], cfg["patch_size"], cfg["spatial_merge_size"])
        pix = pix.to(dtype=torch.bfloat16)
        thw = torch.tensor([grid], dtype=torch.int64)
        emb = net(pix, grid_thw=thw)
        merge = cfg["spatial_merge_size"]
        n = grid[0] * (grid[1] // merge) * (grid[2] // merge)
        print(f"  {how:5} grid {grid} rows {pix.shape[0]} -> embeds {tuple(emb.shape)} (expect {n} rows) "
              f"mean {float(emb.float().mean()):+.4f} std {float(emb.float().std()):.4f}  {time.time()-t0:5.1f} s")
        path = f"/tmp/{a.tag}-{how}.pt"
        torch.save({"image_embeds": emb.cpu(), "image_grid_thw": thw.cpu()}, path)
        print(f"       wrote {path}")


if __name__ == "__main__":
    main()
