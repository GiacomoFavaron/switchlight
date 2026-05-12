# Session Summary

## Goal
Build the inverse rendering frontend (Milestone A1) for the SwitchLight portrait relighting pipeline: a CLI that takes any portrait image and produces a complete "buffer bundle" (surface normals, albedo, foreground mask, plus constant material properties) ready to feed into downstream rendering and training.

## Completed
End-to-end inverse rendering frontend working on M1 Max (MPS backend):

- **Foreground matting** via `rembg` (u2net_human_seg model, tuned for portraits)
- **Surface normals** via DSINE v02 (CVPR 2024 weights from HuggingFace `camenduru/DSINE`)
  - Custom loader bypasses their CUDA-hardcoded hubconf
  - Mirror-padding wrapper for tight portrait crops (improves quality around hair/shoulders)
  - DSINE's canonical test-time pipeline: ImageNet normalize → pad to 32-multiples → forward with default 60° FoV intrinsics → crop back
- **Albedo** via Ordinal Shading / Intrinsic v2 colorful pipeline (Careaga & Aksoy)
  - Uses `load_models('v2')` for the 5-stage colorful decomposition
  - Output key: `'hr_alb'` (high-resolution albedo)
- **CLI** `scripts/extract_buffers.py` runs all three, saves a single `.pt` bundle, optionally writes debug PNGs
- **Visual validation** on a real portrait: normals show correct blue/purple front-facing with clean cyan/green falloff at the sides; albedo is well-decomposed (lighting removed, skin/hair/clothing colors preserved)

Performance (M1 Max, MPS, 768×768):
- First run: ~5 minutes (downloads ~3GB of model weights)
- Subsequent runs: ~22 seconds total per image (matting 2s, normals 2s, albedo 18s)

## Technical Decisions

**Buffer bundle contract** (the contract between modules — DO NOT diverge):
```python
{
    'image':     Tensor[3,H,W] float32, linear RGB [0,1],
    'normal':    Tensor[3,H,W] float32, camera space, unit vectors [-1,1], +Z toward camera,
    'albedo':    Tensor[3,H,W] float32, linear RGB [0,1],
    'roughness': Tensor[1,H,W] float32, [0.05, 1.0] (currently constant 0.5),
    'specular':  Tensor[1,H,W] float32, [0,1] (currently constant 0.04 = dielectric F0),
    'mask':      Tensor[1,H,W] float32, [0,1], foreground = 1,
    'hdri_path': str | None,
    'meta':      dict,
}
```

**Color space:** the entire pipeline operates in **linear RGB**. sRGB ↔ linear conversion happens only at file I/O boundaries (`inverse/io_utils.py`). Inputs from JPEG/PNG are sRGB and get linearized on load; outputs to PNG get re-encoded to sRGB on save.

**Normal convention:** camera space, Z **toward** the camera (positive Z = surface facing the lens). This was validated empirically — DSINE's native output already uses this convention, no flip needed.

**Resolutions:** training 384×384, inference/demo 768×768, smoke tests 256×256.

**Environment:** Python 3.10, managed by `uv`. Project installed editable. DSINE and Intrinsic are cloned into `third_party/` (gitignored) and used either via PEP 420 namespace packages (DSINE) or as a properly installed package (Intrinsic, which pulls in `chrislib` and `altered_midas` transitively).

**Hyperparameters for DSINE_v02** (from `projects/dsine/experiments/exp001_cvpr2024/dsine.txt` + defaults in `projects/dsine/config.py`):
```python
Namespace(
    NNET_architecture="v02", NNET_encoder_B=5, NNET_decoder_NF=2048,
    NNET_decoder_BN=False, NNET_decoder_down=8, NNET_learned_upsampling=True,
    NNET_output_dim=3, NNET_feature_dim=64, NNET_hidden_dim=64,
    NRN_prop_ps=5, NRN_num_iter_train=5, NRN_num_iter_test=5, NRN_ray_relu=True,
)
```

## Files Created/Changed

```
switchlight/
├── pyproject.toml                       # uv project, Python 3.10, all deps incl. geffnet
├── README.md                            # setup instructions (now with 3-step install)
├── .gitignore                           # excludes third_party/, *.pt, outputs/, portrait.jpg
├── inverse/
│   ├── __init__.py
│   ├── io_utils.py                      # sRGB↔linear, device selection, load/save helpers
│   ├── matting.py                       # rembg wrapper (u2net_human_seg)
│   ├── dsine_wrapper.py                 # DSINE v02 loader + estimate_normals()
│   └── intrinsic_wrapper.py             # Ordinal Shading v2 loader + estimate_albedo()
├── scripts/
│   ├── extract_buffers.py               # main CLI
│   └── setup_third_party.sh             # clones DSINE and Intrinsic
├── tests/
│   └── test_inverse_smoke.py            # standalone smoke test for I/O + matting
├── render/  (empty, A2)
├── refine/  (empty, B1)
├── data/blender/  (empty, C1)
├── eval/   (empty, C3)
└── configs/  (empty)
```

## Blockers
None for A1. Repo is on `main`, working tree clean, two commits ahead of initial scaffolding.

Forward-looking risks (not blocking right now):
- Ordinal Shading is slow (~18s/image) because its 5-stage models reload internally per call. If A2 + UNet inference latency becomes a problem, this is the first thing to cache properly.
- Roughness and specular are currently constants. Good enough for skin (which is mostly diffuse with dielectric F0=0.04) but limits material realism. Could refine if time allows.

## Next Steps

**Immediate (Jack):** Start A2 — the Cook-Torrance forward renderer in PyTorch. Files: `render/cook_torrance.py`, `render/hdri.py`. The renderer needs to take a buffer bundle + an HDRI and produce a rendered image. Validation step (the critical checkpoint of A2): output must visually match Blender's Cycles reference render when fed the same GT buffers + HDRI.

**Hrithik (parallel, not blocked by A1 anymore):** Start C1 — Blender scene + 50-frame smoke render. Critical contract: your output bundles MUST match the schema above (same keys, same tensor shapes, same dtypes, same color space). Jack's `extract_buffers.py` is the reference implementation of the bundle format; load any output `.pt` to see the structure.

**Alex (parallel, not blocked):** Start B1 — UNet architecture + training loop scaffolding. The dataset loader will consume bundles in the format above (specifically the `image`, `normal`, `albedo`, `mask` keys plus a `rendered_input` key that Jack will pre-render once the Cook-Torrance renderer exists).

**Coordination point:** When Hrithik's first 50-frame smoke render is ready AND Jack's Cook-Torrance has a draft running, the team needs to validate together that the renderer's output on Blender GT buffers visually matches Blender's reference image under the same HDRI. This is the load-bearing checkpoint of the whole project — if these two don't agree, downstream training is wasted compute. Allocate a paired debugging session for this.
