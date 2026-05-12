# Session Summary — A2 validated + A3 complete (Alex unblocked for B1)

## What just happened (the short version)

The renderer (A2) is validated against Blender ground truth: **MAE = 0.041** in linear RGB over the foreground. The batch pipeline (A3) is built and tested. Alex is fully unblocked to start B1 (the UNet) tomorrow against `data/blender/smoke_test_augmented/frame_0000.pt` as a single-frame target — the rest of the dataset will appear in `data/blender/dataset_augmented/` as Hrithik ships C2 and Jack runs the batch render on each new wave.

---

## For Alex — start here

### What to do tomorrow

1. **Pull and set up:**
   ```bash
   git pull --rebase
   source .venv/bin/activate
   uv pip install -e ".[train,dev]"
   ```

2. **Verify the training target loads correctly:**
   ```bash
   python -c "
   import torch
   b = torch.load('data/blender/smoke_test_augmented/frame_0000.pt', weights_only=False)
   print('Keys:', list(b.keys()))
   for k in ['rendered_input', 'image', 'normal', 'albedo', 'mask']:
       print(f'  {k}: shape={tuple(b[k].shape)}  range=[{b[k].min():.3f}, {b[k].max():.3f}]')
   "
   ```

   You should see `rendered_input` exists, is `[3, 768, 768]` float32, and its range goes a bit above 1.0 (this is intentional — see below).

3. **Start B1** — architecture, dataloader, training loop. Use the single bundle as a one-shot overfitting target while iterating on the architecture. When Hrithik's C2 dataset lands and Jack's batch script processes it, you swap your dataloader's `--bundle-dir` and you're training on real data.

### The training contract (locked)

Each augmented bundle has these keys. Use the augmented ones in `data/blender/dataset_augmented/` (or `smoke_test_augmented/` for now), NOT the raw bundles from `data/blender/dataset/`:

```python
{
    # === Targets ===
    'image':           [3, H, W] float32, linear RGB [0,1]    # Blender's GT render (your training target)

    # === UNet inputs (concat these → 9-channel input) ===
    'rendered_input':  [3, H, W] float32, LINEAR HDR (can exceed 1.0)   # our renderer's output, exposure-matched to 'image'
    'albedo':          [3, H, W] float32, linear RGB [0,1]
    'normal':          [3, H, W] float32, camera-space [-1,1], +Z toward camera

    # === Conditioning / loss masking ===
    'mask':            [1, H, W] float32, [0,1]   # foreground = 1, mask your loss with this

    # === Metadata / not for training ===
    'roughness':       [1, H, W] float32, [0.05, 1.0]   # currently constant 0.5
    'specular':        [1, H, W] float32, [0, 1]        # currently constant 0.04
    'hdri_path':       str (repo-relative)
    'meta':            dict
    'render_hdri_path': str   # the HDRI used to produce rendered_input
    'render_mae':      float  # the renderer's MAE on this bundle (for diagnostics)
}
```

### UNet recipe (Jack's recommended starting point — feel free to override)

- **Input:** concat `[rendered_input, albedo, normal]` → 9 channels
- **Output:** 3-channel residual `r`
- **Final prediction:** `final = rendered_input + r`
- **Loss:** L1 between `final` and `image`, foreground-masked
  ```python
  loss = ((final - image).abs() * mask).sum() / (mask.sum() * 3).clamp_min(1)
  ```

### Three things to know before you start

**1. `rendered_input` is unclamped HDR — it can exceed 1.0.** That's intentional; Blender's PNG-stored `image` is clamped at 1.0 but our renderer preserves the full dynamic range. Your network should learn to compress those highlights as part of its task. **Don't put a sigmoid on the output**, or you'll lose that information.

**2. Foreground only.** Background pixels are zero on both sides. Mask your loss; don't let the network waste capacity learning "predict zero outside the silhouette."

**3. Start tiny.** Overfit a single image first. If the architecture can't memorize one frame on its own, no point training on 30. The augmented smoke-test bundle is your friend.

### Useful reference code

If you want to see how `rendered_input` was computed (so the inputs are not a black box), look at:
- `scripts/batch_render.py` — produces augmented bundles
- `scripts/compare_to_blender.py` — single-bundle validation pattern
- `render/cook_torrance.py::cook_torrance_shade` — the renderer itself

You don't need to touch any of these. They're informational.

### Tooling note: skimage SSIM

`eval/visualize.py` and `eval/metrics.py` (your earlier work) — when you wire up evaluation, use `from skimage.metrics import structural_similarity` rather than rolling SSIM by hand. And `make_demo_grid.py` is scaffolding only; we'll replace it with the real demo flow once your UNet has a checkpoint.

---

## For Hrithik — C2 plan (locked, no changes from your last message)

You're on:
1. Pick first HDRI from the stress-test list (strong directional / soft diffuse / colored / HDR peaks / mixed multi-source)
2. Render + pack `frame_0001.pt` with repo-relative HDRI path
3. Run `python scripts/compare_to_blender.py --bundle data/blender/dataset/frame_0001.pt` to validate the contract still works
4. If green, continue rendering the rest
5. Same roughness=0.5 / F0=0.04 constants as `frame_0000.pt` (Alex needs consistent material assumptions across the dataset)

When each batch of bundles lands, ping Jack and he'll run `scripts/batch_render.py` over your `data/blender/dataset/` to produce the augmented versions Alex trains on.

Forward-looking but not blocking: per-pixel roughness/F0 maps from Blender's Principled BSDF would tighten the renderer-vs-GT MAE further and give the UNet a richer signal. Optional; only after C2 is stable.

---

## For Jack — A4 + report production

- **A4 (UNet integration)** — small wire-up task once Alex ships a B2 checkpoint. Mostly: load checkpoint in `relight.py`, run as post-process behind a `--use-refinement` flag.
- **Hero figures** — 3 team members × multiple HDRIs relighting matrix. Can start anytime.
- **Report draft** — main lift on Jack while Alex trains.

---

## Headline result for the report (also for Alex's training motivation)

> "We implemented a from-scratch physics-based portrait renderer using split-sum IBL with GGX importance-sampled specular prefiltering and image-based diffuse irradiance. Validation against Blender Cycles reference renders (same buffers, same HDRI, same camera) yields MAE = 0.041 in linear RGB over foreground pixels. Residual error is concentrated in regions where Cook-Torrance is known to be insufficient — subsurface-scattering skin, sharp specular highlights, and anisotropic hair — and is addressed by the refinement UNet in the downstream stage."

That last clause is what Alex's B2 is for. The MAE Alex needs to beat — significantly — is 0.041. (Realistically the UNet should also dramatically improve perceptual quality even if MAE only drops moderately; SSIM and LPIPS are better proxies for what we care about.)

---

## A2 + A3 — what got built

### A2: Cook-Torrance + IBL renderer (validated)
- Full IBL prefiltering: HDRI load, latlong↔cubemap, diffuse irradiance, specular GGX mip chain (Karis split-sum with mip-biased sampling), BRDF integration LUT
- Cook-Torrance shading with orthographic V=+Z, split-sum specular term
- Validated visually on synthetic spheres + real portraits + against Blender GT

### A3: Batch dataset processing
- `scripts/batch_render.py` processes a directory of bundles under one HDRI / multiple HDRIs / the bundle's own embedded HDRI
- Validates contract before rendering (catches malformed bundles cheaply)
- Saves augmented bundles with `rendered_input`, `render_hdri_path`, `render_mae` added
- Resume-friendly (skips existing outputs unless `--force`)
- Exposure-matches our renderer output to Blender's reference so the UNet residual stays bounded across the dataset

### Validation methodology
Foreground-masked MAE in linear RGB after exposure matching. Standard practice in IBL papers — our renderer's absolute brightness scale is arbitrary so meaningful comparison normalizes mean foreground luminance first.

### Known limitations (deferred to UNet)
- No subsurface scattering — skin looks slightly translucent
- Split-sum approximation softens sharp specular highlights
- No anisotropic BRDF for hair
- Constant per-pixel roughness/F0 (vs. Blender's per-material BSDF maps)

All four are what the UNet's residual is meant to absorb.

---

## Files in the repo (post-session-3)

```
switchlight/
├── inverse/                  # A1 — buffer extraction (from session 1, unchanged)
│   ├── io_utils.py
│   ├── matting.py
│   ├── dsine_wrapper.py
│   └── intrinsic_wrapper.py
├── render/                   # A2 + A3 — renderer + cache
│   ├── hdri.py
│   ├── cook_torrance.py      # cook_torrance_shade + normalize_exposure + compute_mae
│   └── cache.py              # disk cache for prefiltered HDRIs
├── scripts/
│   ├── extract_buffers.py    # A1 entry point (image → bundle)
│   ├── relight.py            # interactive relighting (one image, one HDRI)
│   ├── compare_to_blender.py # single-bundle validation against Blender GT
│   ├── batch_render.py       # A3 — dataset-scale rendering for training
│   ├── sweep_materials.py    # optional: material constant search
│   ├── test_*.py             # per-step renderer validators
│   └── setup_third_party.sh
├── data/
│   ├── blender/
│   │   ├── smoke_test/                 # raw bundles from Hrithik (C1)
│   │   ├── smoke_test_augmented/       # with rendered_input added (training-ready)
│   │   ├── dataset/                    # C2 raw bundles will land here
│   │   ├── dataset_augmented/          # C2 + Jack's batch render will land here
│   │   └── hdri/                       # HDRIs Hrithik used in Blender
│   ├── hdris/                          # general HDRI library
│   └── brdf_lut.pt                     # cached BRDF integration LUT
├── cache/hdri/                         # gitignored — per-HDRI prefilter cache
├── eval/                               # Alex's eval scripts
├── refine/                             # Alex's UNet code goes here
└── outputs/                            # gitignored — debug images, validation triptychs
```

## Onboarding (any new machine, any teammate)

```bash
# Clone, set up
git clone <repo>
cd switchlight
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[train,dev]"
bash scripts/setup_third_party.sh
uv pip install ./third_party/Intrinsic

# Smoke test (extract buffers from an image)
python scripts/extract_buffers.py --input <portrait.jpg> --output bundle.pt

# Smoke test (relight under an HDRI)
python scripts/relight.py --bundle bundle.pt --hdri data/blender/hdri/studio_small_03_2k.hdr \
    --output outputs/relit.png --save-debug-grid

# Smoke test (Alex's path — load training-ready bundle)
python -c "import torch; b = torch.load('data/blender/smoke_test_augmented/frame_0000.pt', weights_only=False); print(list(b.keys()))"
```

## Blockers
None. A1 + A2 + A3 + C1 are green. B1 (Alex) and C2 (Hrithik) are both unblocked and parallel.

---

# Previous session — A1 (Jack, session 1, for reference)

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

Performance (M1 Max, MPS, 768×768):
- First run: ~5 minutes (downloads ~3GB of model weights)
- Subsequent runs: ~22 seconds total per image (matting 2s, normals 2s, albedo 18s)

## Technical Decisions

**Color space:** linear RGB throughout. sRGB ↔ linear only at file I/O boundaries.

**Normal convention:** camera space, Z toward the camera. Validated empirically — DSINE's native output already uses this, no flip needed.

**Resolutions:** training 384×384, inference 768×768, smoke tests 256×256.

**Hyperparameters for DSINE_v02** (from `projects/dsine/experiments/exp001_cvpr2024/dsine.txt` + defaults in `projects/dsine/config.py`):
```python
Namespace(
    NNET_architecture="v02", NNET_encoder_B=5, NNET_decoder_NF=2048,
    NNET_decoder_BN=False, NNET_decoder_down=8, NNET_learned_upsampling=True,
    NNET_output_dim=3, NNET_feature_dim=64, NNET_hidden_dim=64,
    NRN_prop_ps=5, NRN_num_iter_train=5, NRN_num_iter_test=5, NRN_ray_relu=True,
)
```

**Performance note:** Ordinal Shading is slow (~18s/image) because its 5-stage models reload internally per call. If inference latency becomes a problem, this is the first thing to cache properly.