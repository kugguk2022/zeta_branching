# Next Steps

## 1. Make runs reproducible
- Add fixed random seeds for NumPy and PyTorch in training scripts.
- Store training configuration metadata (epochs, batch size, grid sizes) next to each model checkpoint.

## 2. Improve model/version management
- Rename model/data artifacts with clear version tags (for example: `surrogate_v3_depth4.pt`).
- Keep a small changelog file in `models/` describing what changed between versions.

## 3. Add quick evaluation metrics
- Save numeric validation metrics (median/mean/max error) into a CSV or JSON file in `outputs/`.
- Track inference speed for ground truth vs surrogate over fixed test grids.

## 4. Add a simple automation entrypoint
- Create a `run_all.ps1` (or `Makefile` equivalent) for:
  - generating baseline plots
  - training/loading surrogate
  - exporting validation and sweep outputs

## 5. Add dependency pinning
- Create a `requirements.txt` with pinned versions (`numpy`, `matplotlib`, `torch`, `mpmath`, optional `imageio`, `Pillow`).
- Include short setup instructions for CPU and CUDA environments.

## 6. Add lightweight tests
- Add unit tests for:
  - branch-cut continuity behavior
  - surrogate input/output shape and dtype
  - path handling (artifacts always written to `models/` or `outputs/`)

## 7. Optional project polish
- Add a small `outputs/README.md` index describing each generated image/video.
- Consider a notebook for interactive parameter exploration once the core scripts stabilize.
