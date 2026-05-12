# SwitchLight

SwitchLight-inspired portrait relighting pipeline. COMP5405 Digital Media Computing 2026S1 group project.

## Team

- Jack (Giacomo Favaron) — Rendering & Pipeline
- Alex — Training & Refinement UNet
- Hrithik — Blender Data & Evaluation

## Setup

We use [uv](https://github.com/astral-sh/uv) for environment and dependency management.

### One-time install of uv

macOS / Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Set up the project

```bash
git clone https://github.com/<your-org>/switchlight.git
cd switchlight
uv venv --python 3.10
source .venv/bin/activate    # macOS/Linux
uv pip install -e ".[train,dev]"
```

That's it. `uv pip install -e .` installs the project in editable mode along with all dependencies. The `[train,dev]` extras pull in things you only need for training and development.

### Verify install

```bash
python -c "import torch; print('torch', torch.__version__); print('mps available:', torch.backends.mps.is_available())"
```

On M1 Max you should see MPS available. On Colab you'll see CUDA available.

## Repo structure

```
switchlight/
├── inverse/          # Pretrained model wrappers (DSINE, Ordinal Shading, matting)
├── render/           # Cook-Torrance + HDRI utilities
├── refine/           # Refinement UNet + training
├── data/             # Blender pipeline + (gitignored) datasets
├── eval/             # Metrics + figures
├── scripts/          # CLI entry points
├── tests/            # Sanity tests
├── configs/          # YAML configs for training
└── third_party/      # Cloned external repos (DSINE, Intrinsic) — gitignored
```

## Pipeline overview

```
input.jpg ──► extract_buffers ──► (normal, albedo, roughness, specular, mask)
                                          │
                                          ▼
              prefiltered HDRI ──► Cook-Torrance renderer
                                          │
                                          ▼
                                  rendered output
                                          │
                                          ▼
                              [optional] refinement UNet
                                          │
                                          ▼
                                  final relit image
```

## Quick start (once A1 lands)

```bash
# Extract buffers from a portrait
python scripts/extract_buffers.py --input photo.jpg --output bundle.pt

# (later, once renderer exists) Relight with a target HDRI
python scripts/relight.py --input photo.jpg --hdri studio.hdr --output relit.png
```

## See also

- `00_OVERVIEW.md` — shared decisions, tensor contracts, dependency graph
- `01_JACK.md` / `02_ALEX.md` / `03_HRITHIK.md` — per-person detailed plans
