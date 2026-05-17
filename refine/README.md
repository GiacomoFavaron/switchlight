# Refinement UNet

This directory contains the learned refinement experiment for the SwitchLight-inspired
pipeline. The final demo ships the physics-only renderer because the refinement UNet did
not improve visual quality on the available five-bundle synthetic dataset.

## Contents

- `unet.py`: a small 4-level residual U-Net with roughly 929K trainable parameters.
  It takes 9 input channels (`rendered_input`, albedo, normal) and predicts a 3-channel
  residual that is added to the physics render.
- `dataset.py`: dataset loader for Blender `.pt` bundles, including augmented bundles
  with a `rendered_input` key from the Cook-Torrance renderer.
- `losses.py`: foreground-masked L1 loss plus foreground-masked VGG16 perceptual loss
  on `relu2_2` and `relu3_3`.

## Training Runs

B2 was trained on Colab using the 5 augmented Blender bundles in
`data/blender/dataset_augmented` at 384x384 resolution. The initial run used
`vgg_weight=0.5` for 10,000 steps and produced strong color/checkerboard artifacts.
A conservative retry used `configs/b2_retry_vgg01.yaml` with `vgg_weight=0.1` for
1,000 steps. The retry reduced the most extreme color shift but still did not beat
the physics input visually, especially on `frame_0004`, where the renderer was already
close to Blender ground truth.

Drive artifacts:

- Initial B2 checkpoint: `Drive/switchlight/checkpoints/b2/step_010000.pt`
- Initial B2 grids: `Drive/switchlight/results/b2/`
- Retry checkpoint: `Drive/switchlight/checkpoints/b2_retry_vgg01/step_001000.pt`
- Retry log: `Drive/switchlight/logs/b2_retry_vgg01_train.jsonl`
- Retry grids: `Drive/switchlight/results/b2_retry_vgg01/`

## Outcome

The UNet work is kept as a documented negative result. With only five training bundles,
the model overfit and learned visible artifacts rather than a robust renderer correction.
The final project should use the Cook-Torrance physics renderer directly, while the
refinement code remains available for reproducibility and future work with a larger,
more diverse synthetic dataset.
