# SwitchLight

SwitchLight-inspired portrait relighting pipeline. COMP5405 Digital Media Computing 2026S1 group project.

## Team

- 560472101  — Rendering & Pipeline
- 560401374 — Training & Refinement UNet
- 550615169 — Blender Data & Evaluation

## Setup

We use [uv](https://github.com/astral-sh/uv) for environment and dependency management.

### One-time install of uv

macOS / Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Set up the project

```bash
git clone git@github.com:GiacomoFavaron/switchlight.git
cd switchlight
uv venv --python 3.10
source .venv/bin/activate    # macOS/Linux

# 1. Install project + declared dependencies
uv pip install -e ".[train,dev]"

# 2. Clone third-party model repos (DSINE, Intrinsic)
bash scripts/setup_third_party.sh

# 3. Install the Intrinsic package (pulls in chrislib, altered_midas, etc.)
uv pip install ./third_party/Intrinsic
```

**First run note:** the inverse rendering frontend downloads ~3GB of model weights from HuggingFace and GitHub the first time it runs. Subsequent runs are fast (~22s/image at 768×768 on M1 Max).

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

## Quick start
```

# Extract buffers from a portrait
python scripts/extract_buffers.py --input photo.jpg --output bundle.pt

# Relight with a target HDRI
python scripts/relight.py --input photo.jpg --hdri data/blender/hdri/courtyard_2k.hdr --output relit.png
```