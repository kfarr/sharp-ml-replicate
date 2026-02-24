# Sharp ML Replicate - Cog Deployment

## Project Overview
Cog deployment of Apple's [SHARP model](https://github.com/apple/ml-sharp) — single image to 3D Gaussian splats. Deployed at https://replicate.com/kfarr/sharp-ml (T4 GPU).

## Files
- `cog.yaml` — Build config (CUDA 12.1, Python 3.11, dependencies, ml-sharp clone)
- `predict.py` — Cog predictor with GPU-optimized postprocessing

## Build & Push

### Prerequisites
Log in to Replicate (get CLI token from https://replicate.com/auth/token):
```bash
cog login --token-stdin
# paste token, press Enter, then Ctrl+D
```

### Push
```bash
cog push r8.im/kfarr/sharp-ml
```

## Build Issues Encountered & Fixes

### 1. `.python-version` conflict (FIXED)
Apple's ml-sharp repo contains `.python-version` specifying Python 3.13, but the Cog base image uses Python 3.11 via pyenv. Fixed by adding `rm -f /opt/ml-sharp/.python-version` before `pip install -e .` in cog.yaml run steps.

### 2. Authentication for push
`REPLICATE_API_TOKEN` env var is NOT sufficient for `cog push`. You must run `cog login --token-stdin` with a CLI auth token from https://replicate.com/auth/token (different from the API token at replicate.com/account/api-tokens).

## Known Risks (from initial task, not yet encountered)
- **gsplat** — CUDA C++ extension compiled from source. Currently resolves to gsplat 1.5.3 with torch 2.10.0. If it fails, try pinning `torch==2.1.0` or `gsplat==1.0.0`.
- **open3d** — NOT imported in predict.py (only used upstream for visualization). ~500MB. Safe to remove from cog.yaml if image size is a concern.
- **moviepy==1.0.3** — NOT imported in predict.py. Safe to remove.
- **e3nn** — IS used by ml-sharp internals. Keep this.

## Architecture Notes
- `predict.py` includes GPU-optimized postprocessing (quaternion conversion, SVD decomposition) replacing upstream's slow CPU-based scipy code
- Model weights (~700MB) are downloaded at runtime in `setup()` from Apple's CDN, not bundled in the image
- Internal resolution is fixed at 1536x1536
- Output is a .ply file containing 3D Gaussian splats
- First run has ~2 min cold start (model download + warmup); subsequent runs are fast

## Dependencies (torch version)
The `torch>=2.0.0` constraint resolves to torch 2.10.0 (latest). If CUDA compatibility issues arise, pin to a specific version.
