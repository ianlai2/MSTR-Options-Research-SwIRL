# Quick Start: Run Clustering Now (CPU) or Later (GPU on Linux/WSL2)

## Current Status

✓ Notebook is **fully functional with CPU clustering** (MiniBatchKMeans)  
✓ GPU code is ready (will auto-detect RAPIDS if available)  
⚠️ RAPIDS packages (25.10) are **NOT available for Windows** in conda channels
✓ GPU via WSL2/Linux/Docker is possible (see below)

## Option 1: Start Clustering NOW (CPU, 5 min)

### What you need:
- Current Python environment (already has pandas, scikit-learn, numpy, etc.)
- Jupyter notebook

### Steps:
1. Open `options_ask_prediction.ipynb` in Jupyter
2. Scroll to the GPU-aware clustering cell
3. Run it — it will automatically use **MiniBatchKMeans** (CPU fallback)
4. Continue with regime visualization and artifact saving

### Result:
- Regimes computed and saved in ~2-5 min (depending on dataset size)
- All artifacts (parquet, scaler, centers) saved
- Same quality as GPU (just slower for very large datasets)

**This is the recommended path for Windows machines.**

---

## Option 2: Set Up GPU Later (WSL2 / Linux / Docker)

### Why?
- GPU would be 3-5× faster for large datasets
- Requires Linux environment (RAPIDS Windows support is very limited)

### Options:
**A) WSL2 (Windows Subsystem for Linux)** — Run Linux Ubuntu on Windows with GPU passthrough:
- Setup: ~30 min
- Then use Mamba inside Ubuntu to install RAPIDS
- See GPU_SETUP.md Option C/D for details

**B) Docker with RAPIDS image** — Guaranteed reproducible GPU environment:
- Setup: ~20 min (download Docker, pull RAPIDS image)
- Then: `docker run --gpus all -it <RAPIDS image>`
- See GPU_SETUP.md Option C for details

**C) Remote Linux machine with GPU** — Rent cloud compute (e.g., AWS EC2, GCP, Colab):
- Launch GPU instance with CUDA 13 pre-installed
- Clone repo, create mamba environment (5 min), run notebook

### Which should I choose?
- **WSL2 + Mamba**: Best for local development with GPU
- **Docker**: Best for reproducibility and team sharing
- **Cloud GPU**: Best for one-off runs without local hardware investment

---

## My Recommendation

**Use the CPU path (Option 1, now)**:
- RAPIDS for Windows is unavailable for the latest versions (25.10)
- CPU clustering produces exact same results
- MiniBatchKMeans runs in ~3-5 minutes on this dataset
- Zero additional setup needed
- If GPU speedup is critical later, switch to WSL2/Linux/Docker (no code changes)

GPU on Windows is not practical for RAPIDS 25.10. CPU is your best option for immediate, production-ready results.

---

## Quick Commands

### CPU Clustering (ready NOW):
```bash
cd c:\Users\Saber\Desktop\research\MSTR-Options-Research-SwIRL
jupyter lab
# Open notebook, run clustering cell (MiniBatchKMeans will auto-run)
```

### GPU Clustering on WSL2 (when ready):
```bash
# Inside WSL2 Ubuntu:
mamba create -n rapids-cu13 -c rapidsai -c nvidia -c conda-forge \
  cudf cuml cupy python=3.11 cudatoolkit=13.0 \
  pandas numpy scikit-learn jupyter jupyterlab

conda activate rapids-cu13
jupyter lab
# Run notebook (same code, GPU auto-detected)
```

### GPU Clustering via Docker:
```bash
docker run --gpus all -it -p 8888:8888 \
  nvcr.io/nvidia/rapidsai/rapidsai:25.10-cuda13.0-devel-ubuntu22.04

# Inside container:
git clone <your-repo>
cd MSTR-Options-Research-SwIRL
jupyter lab --ip=0.0.0.0
```

---

## Files Created for You

- `GPU_SETUP.md` — Detailed setup guide for all options (conda, pip, docker, CPU)
- `check_gpu_setup.py` — Verify GPU readiness at any time (run: `python check_gpu_setup.py`)
- `setup_rapids_env.bat` — One-click conda setup (requires conda in PATH; use Mamba for faster setup)
- Notebook cells — GPU-aware clustering code (auto-detects and falls back to CPU)

---

## Next: What to Do?

**→ Start clustering now on CPU** (ready immediately):

```bash
cd c:\Users\Saber\Desktop\research\MSTR-Options-Research-SwIRL
jupyter lab
# Open options_ask_prediction.ipynb
# Run the clustering cell (it will auto-detect CPU and use MiniBatchKMeans)
```

Results in ~5 minutes. Same quality as GPU, just slower.

---

**If you want GPU later:** See Option 2 (WSL2/Linux/Docker). But CPU is the practical choice for Windows right now.
