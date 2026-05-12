# Session Summary

## Goal
Complete Milestone A2 (the physically-based renderer) and validate it against Hrithik's Blender ground truth. The validation step was the load-bearing checkpoint of the whole project: if our renderer's output agrees with Blender's reference under the same buffers + HDRI, the physics path is verified end-to-end and downstream training has a known-correct baseline to learn from.

## Completed

### A1 — Inverse rendering frontend (still locked, from session 1)
Unchanged. Bundle extraction continues to work on real portraits.

### A2 — Cook-Torrance + IBL renderer
All 7 sub-steps validated:

1. HDRI load + tonemap (`render/hdri.py::load_hdri`)
2. Latlong ↔ cubemap conversion (round-trip near-identical)
3. Diffuse irradiance prefilter (Frisvad tangent frame, Hammersley + firefly clamp)
4. Specular GGX prefilter mip chain (Karis split-sum, mip-biased sampling)
5. BRDF integration LUT (vectorized, LearnOpenGL canonical form)
6. Cook-Torrance shading validated on synthetic sphere grid (textbook PBR result)
7. Cook-Torrance shading validated on real portraits (jack, hrithik)

### A2 validation against Blender GT (THIS SESSION)

Hrithik's C1 landed in parallel: `data/blender/smoke_test/frame_0000.pt` is a Blender-rendered character with all GT buffers + beauty render packed into the shared bundle contract. We ran our renderer on those buffers + the same HDRI and compared output.

**Result: MAE = 0.041 in linear RGB over the foreground.**

Visual agreement is strong:
- Same lighting direction (upper-right, matching the umbrella in the studio HDRI)
- Same brightness pattern (forehead/cheeks/shoulders bright; chest/lower-body darker)
- Same overall color tone
- Costume materials read correctly (red armor reads red, teal sleeves read teal)

Residual error concentrated where expected:
- **Skin** — looks slightly translucent/glossy (no subsurface scattering in our model; this is precisely the gap the refinement UNet (B2) is meant to fill)
- **Specular highlights on armor** — softened vs. Blender's path-traced reference (consequence of split-sum approximation)
- **Hair** — slightly over-shiny (no anisotropic BRDF)
- **Constant material assumption** — Hrithik's bundle ships constant roughness=0.5 and F0=0.04; Blender uses per-pixel material maps from the Principled BSDF. This explains most of the per-region disagreement.

All three error sources are well-understood limitations of physics-only Cook-Torrance, and all three are directly addressable by the refinement UNet which is Alex's next milestone.

### Supporting infrastructure (this session)

- `render/cook_torrance.py` — forward shader (split-sum IBL)
- `render/cache.py` — disk cache for prefiltered HDRIs (SHA1-keyed, ~3 min per HDRI once, instant after)
- `scripts/relight.py` — end-to-end relighting CLI (single + batch modes, debug-grid output)
- `scripts/compare_to_blender.py` — validation script that produces the renderer-vs-Blender triptych and MAE

## Technical Decisions

### Validation methodology
The validation does **exposure-matching** before computing MAE — our renderer's absolute brightness scale is arbitrary (depends on HDRI intensity and exposure choice), so a meaningful comparison normalizes mean foreground luminance to match Blender's reference before taking the per-pixel difference. This is standard practice in IBL papers.

### Path remapping in bundles
Initial bundle had absolute paths baked in (`/Users/hrithikg/...`); Hrithik re-packed with repo-relative paths after the issue was raised. **Going forward: all bundle paths are relative to repo root.**

### Skin appearance — known limitation
Our Cook-Torrance output makes skin look slightly translucent/dry compared to Blender's reference. The cause is that Cook-Torrance models only surface reflection; real skin gets ~30% of its perceived appearance from subsurface scattering (light entering, scattering 1-2mm inside, exiting nearby). We chose not to model SSS because:
1. SSS adds substantial complexity (multiple-bounce simulation or learned approximation)
2. SwitchLight's own approach is to let a refinement UNet learn this residual rather than model it explicitly
3. The MAE = 0.041 result is acceptable as physics-only baseline; the UNet's job is to close the remaining gap

### What we are NOT trying to fix at the renderer level
- Subsurface scattering / skin "alive" look → deferred to UNet (B2)
- Sharp specular highlights on glossy materials → consequence of split-sum, accepted
- Anisotropic hair BRDF → not modeled, accepted
- Per-material reflectance variation → future work (would need real material maps from Blender)

## Onboarding (unchanged)

```bash
source .venv/bin/activate
uv pip install -e ".[train,dev]"
bash scripts/setup_third_party.sh
uv pip install ./third_party/Intrinsic
```

To run the full renderer-validation pipeline end-to-end:

```bash
# Relight a real photo under any HDRI
python scripts/relight.py \
    --input <portrait.jpg> --hdri data/hdris/<file>.hdr \
    --output outputs/relit.png --save-debug-grid

# Validate against Blender GT
python scripts/compare_to_blender.py \
    --bundle data/blender/smoke_test/frame_0000.pt
```

## Files Created/Changed (this session)

```
switchlight/
├── render/
│   ├── cook_torrance.py        # NEW — forward shader
│   ├── cache.py                # NEW — prefilter disk cache
│   └── hdri.py                 # extended with prefilter functions
└── scripts/
    ├── relight.py              # NEW — main relighting entry point
    ├── compare_to_blender.py   # NEW — validation against Blender GT
    ├── sweep_materials.py      # NEW — (optional) material constant sweep
    ├── test_cubemap.py         # NEW
    ├── test_diffuse_prefilter.py  # NEW
    ├── test_specular_prefilter.py # NEW
    ├── test_brdf_lut.py        # NEW (saves data/brdf_lut.pt)
    ├── test_sphere_render.py   # NEW
    └── visualize_hdri.py       # NEW
```

## Blockers
None. A1 + A2 + Hrithik's C1 (validated) are all green. The pipeline is fully connected end-to-end.

## Next Steps

**Jack (immediate):** A3 — polish the inference pipeline.
- Currently `scripts/relight.py` works but pays the buffer-extraction cost on every `--input` run. For batch processing over the team's photos, we'll want a `--batch-bundle-dir` that consumes pre-extracted bundles.
- Generate hero figures for the report: 3 team members × 3+ HDRIs = relighting matrix.
- Then A4 (UNet integration) once Alex's B2 produces a checkpoint.

**Hrithik (immediate):** Scale C1 → C2.
- The bundle format is now battle-tested. Next is scaling to ~30 single-HDRI bundles using a varied HDRI set (warm indoor, cool office, outdoor, sunset, neon, etc.)
- Optional but high-value: export real per-pixel `roughness` and `specular` maps from Blender's Principled BSDF, not constants. This would tighten our renderer-vs-Blender MAE further and give Alex's UNet a richer training signal.

**Alex (immediate, unblocked):** B1 — UNet architecture + training loop.
- Data loader contract is locked (see below). Use `frame_0000.pt` as the validation single-frame to develop against.
- The UNet input: concat of (rendered_input, albedo, normal) → 9 channels in. Output: 3-channel residual added to rendered_input.
- "Rendered input" comes from running our renderer on each bundle's buffers. This is the same pipeline `compare_to_blender.py` exercises — see that script for the canonical pattern.

## Critical contract (locked)

```python
torch.save({
    'image':     Tensor[3,H,W] float32 linear RGB [0,1],
    'normal':    Tensor[3,H,W] float32 camera-space [-1,1], +Z toward camera,
    'albedo':    Tensor[3,H,W] float32 linear RGB [0,1],
    'roughness': Tensor[1,H,W] float32 [0.05, 1.0],
    'specular':  Tensor[1,H,W] float32 [0, 1],
    'mask':      Tensor[1,H,W] float32 [0, 1],
    'hdri_path': str (relative to repo root),
    'meta':      dict,
}, 'frame_NNNN.pt')
```

## Headline result for the report

> "We implemented a from-scratch physics-based portrait renderer using split-sum IBL with GGX importance-sampled specular prefiltering and image-based diffuse irradiance. Validation against Blender Cycles reference renders (same buffers, same HDRI, same camera) yields MAE = 0.041 in linear RGB over foreground pixels. Residual error is concentrated in regions where Cook-Torrance is known to be insufficient — subsurface-scattering skin, sharp specular highlights, and anisotropic hair — and is addressed by the refinement UNet in the downstream stage."

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

## Onboarding (for teammates after pulling A1)

After `git pull`, run from the repo root:

```bash
source .venv/bin/activate
uv pip install -e ".[train,dev]"          # picks up new deps like geffnet
bash scripts/setup_third_party.sh         # clones DSINE + Intrinsic into third_party/
uv pip install ./third_party/Intrinsic    # pulls in chrislib, altered_midas transitively
```

If you don't have a `.venv` yet (first time setting up):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv if missing
uv venv --python 3.10
source .venv/bin/activate
# then run the three commands above
```

Verify the inverse rendering frontend works end-to-end:

```bash
python scripts/extract_buffers.py \
    --input <some_portrait.jpg> \
    --output bundle.pt \
    --visualize outputs/debug/ \
    --size 768
```

First run downloads ~3GB of weights (DSINE + Intrinsic v2 stages + an EfficientNet backbone). Cached after that. Subsequent runs ~22s per image on M1 Max.

Inspect `outputs/debug/02_normal.png` (should be blue/purple on front-facing surfaces) and `outputs/debug/03_albedo.png` (should look flat-lit with skin/clothing colors preserved) to confirm everything's working.

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