# Credits

Almost everything here is original work, licensed Apache-2.0 (see [LICENSE](LICENSE)). A few pieces are other
people's and are acknowledged below; [NOTICE](NOTICE) is the authoritative statement of what the licence does
and does not cover.

## Third-party components

| What | Source | Used in | License |
|---|---|---|---|
| Device-side expert LRU cache kernels (vendored unmodified) | davetha -- https://github.com/davetha/r9700-lru-expert-cache (3743f13) | `kernels/third_party/davetha/` | **Apache-2.0** -- its own LICENSE and NOTICE are preserved in that directory |
| EXL3 trellis decode (lane plan, codebook pair decoder) and the RDNA "tiles" GEMV dot core, 128-point Hadamard warp shuffle (copied, adapted: integer K only, per-row offsets) | turboderp -- https://github.com/turboderp-org/exllamav3 (v1.6.6) | `kernels/r9k_moe_exl3.hip`, `kernels/r9k_exl3_gemv.hip` (sections marked EXLLAMAV3) | **MIT** |
| Chat template `qwen-fixed-v22.3.jinja` (copied verbatim) | GGZ14/vllm-mxfp4 -- https://github.com/GGZ14/vllm-mxfp4 (92eed82) | `serve/templates/` | used with the author's permission, with credit; no upstream LICENSE file |
| Outfit variable font (two woff2 subsets, copied verbatim) | The Outfit Project Authors -- https://github.com/Outfitio/Outfit-Fonts | `serve/web/fonts/`, and the README title `docs/r9700-stack-lockup-{dark,light}.svg` as path outlines written by `docs/lockup.py` | **SIL Open Font License 1.1** -- `serve/web/fonts/OFL.txt`; the lockup ships outlines only, no font file, which is what the OFL asks for |
| Monitor / About page design (tokens, components, layout) | Strata -- https://github.com/Niko1221/Strata | `serve/web/tokens.css`, `serve/web/web.css`, `serve/web/monitor.*`, `serve/web/about.*`, and the README's centred title lockup (the shape of its hero: mark, outlined word, dark/light pair; drawn here from our own face and tokens) | **MIT** -- ported and rewritten for vLLM's counters; the parts a vLLM launch cannot answer are left out rather than guessed |

The chat template is worth about +14% speculative-decoding acceptance over the checkpoint's own template. It is
the one file in this repository not covered by our Apache-2.0 grant: permission was given for **this** project,
so it does not automatically pass to forks.

## Upstream, used unmodified and not redistributed

vLLM, ROCm, PyTorch and Triton (Apache-2.0 / MIT / BSD). Nothing from them is vendored here; the whole point of
this project is that it runs as a plugin on stock builds.

## Model checkpoints

Qwen3.8-27B-NVFP4, Qwen3.8-Flash-Next and the DFlash2 drafter are third-party weights under their own model-card
terms. They are not part of this repository.

---

Provenance of the kernels themselves -- including which constants are generated from published format
specifications rather than authored, and how each replacement was written -- is recorded in
[notes/independence.md](notes/independence.md). `tools/gen_kmag.py --check` regenerates the folded-unpack table
from the OCP e2m1 and e4m3 specifications and verifies it against the checked-in copy.
