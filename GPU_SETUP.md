# GPU/RAPIDS Setup Guide for Options Clustering

This guide explains how to set up GPU acceleration (RAPIDS cuDF/cuML) for this project on both local and remote machines.

## Status

- **Current Setup**: Windows machine with NVIDIA RTX 4060 Laptop (driver 573.09, CUDA 13 compatible)
- **Current Environment**: CPU-only fallback (MiniBatchKMeans) active in notebook
- **⚠️ IMPORTANT**: RAPIDS Windows support is limited. Most RAPIDS packages (25.10) are **not available** for Windows.
- **Recommendation**: 
  - For **immediate results on Windows**: Use CPU clustering (fully functional, ready now)
  - For **GPU acceleration**: Use WSL2/Linux or Docker (see Option C/D below)

## Option A: Conda/Mamba Setup (Recommended)

### Prerequisites
- **Conda** or **Mamba** installed (see [https://conda.io/projects/conda/en/latest/](https://conda.io/) or [https://mamba.readthedocs.io/](https://mamba.readthedocs.io/))
- **CUDA Toolkit 13.0+** or compatible driver (you have driver 573.09, compatible with CUDA 13)
- At least 8 GB GPU memory (RTX 4060 has 8 GB)

### Steps

1. **Download & Install Miniconda** (if you don't have conda/mamba):
   - Windows: [https://docs.conda.io/projects/miniconda/en/latest/](https://docs.conda.io/projects/miniconda/en/latest/)
   - Extract and run the installer; during setup, choose to initialize conda for PowerShell.

2. **Activate a new shell** and create a RAPIDS environment:

   ```bash
   conda create -n rapids-cu13 -c rapidsai -c nvidia -c conda-forge \
     cudf=25.10 cuml=25.10 cupy=12.0 \
     python=3.11 cudatoolkit=13.0 \
     pandas numpy scikit-learn matplotlib seaborn \
     jupyter jupyterlab pyarrow joblib
   ```

   (For CUDA 12, replace `cudf=25.10 cuml=25.10 cupy=12.0 cudatoolkit=13.0` with `cudf=25.10 cuml=25.10 cupy=11.0 cudatoolkit=12.0`)

3. **Activate the environment**:
   ```bash
   conda activate rapids-cu13
   ```

4. **Install additional packages**:
   ```bash
   pip install --upgrade umap-learn pynndescent
   ```

5. **Launch Jupyter from this environment**:
   ```bash
   jupyter lab
   ```
   Open `options_ask_prediction.ipynb` and run the clustering cells. The GPU-aware clustering code will automatically detect cuML/cuDF and use GPU acceleration.

### Verify GPU setup

Run the **"Environment & GPU / RAPIDS detection helper"** cell in the notebook. You should see:
- `cudf version: 25.10.x`
- `cuml version: 25.10.x`
- GPU device count > 0 (if cupy is installed)

---

## Option B: Pip Installation (Windows, Experimental)

RAPIDS pip packages for Windows are limited and often conflict. If you must use pip:

```bash
pip install --upgrade \
  --extra-index-url=https://pypi.nvidia.com \
  "cudf-cu13==25.10.*" \
  "cuml-cu13==25.10.*" \
  "cugraph-cu13==25.10.*" \
  "cupy-cu13==12.*"
```

**Caveat**: This may fail on Windows due to dependency conflicts. Conda/mamba is strongly recommended.

---

## Option C: Docker (Linux + GPU passthrough)

If you're on Windows and want a guaranteed reproducible GPU environment:

1. Install **Docker Desktop for Windows** with WSL2 backend and GPU support ([https://docs.nvidia.com/cuda/wsl-user-guide/](https://docs.nvidia.com/cuda/wsl-user-guide/))

2. Use the official RAPIDS Docker image:
   ```bash
   docker run --gpus all -it -p 8888:8888 \
     nvcr.io/nvidia/rapidsai/rapidsai:25.10-cuda12.0-devel-ubuntu22.04
   ```
   (Replace `cuda12.0` with `cuda13.0` if needed)

3. Inside the container, clone your repo and run the notebook.

---

## Option D: Current Fallback (CPU, No GPU)

If GPU setup is not immediately available, the notebook **automatically falls back to MiniBatchKMeans** (CPU). This is slower but fully functional:

- Clustering still works with the same API.
- Final cluster labels are saved to `df_clustering_processed_with_regimes.parquet`.
- Cluster centers are saved as NumPy arrays in `kmeans_centers.npy` (portable and reproducible).
- Later, when you set up GPU, simply re-run the clustering cell in the GPU environment — it will detect cuML and use GPU acceleration without code changes.

---

## Reproducibility on Another Machine

Once you have results from this notebook:

1. **Save these artifacts** (done automatically by persistence cells):
   - `df_clustering_processed_with_regimes.parquet` (or `.csv`) — the full feature + regime dataset
   - `clustering_scaler.joblib` — fitted StandardScaler for feature normalization
   - `kmeans_centers.npy` — cluster centers (k, n_features)
   - `regime_summary_stats.csv` — per-regime statistics

2. **On a new machine**, to reproduce cluster assignments for new data:
   ```python
   import numpy as np
   import joblib
   from sklearn.preprocessing import StandardScaler
   
   # Load artifacts
   scaler = joblib.load('clustering_scaler.joblib')
   centers = np.load('kmeans_centers.npy')
   
   # Prepare new data: X_new (n_samples, n_features)
   X_new_scaled = scaler.transform(X_new)
   
   # Assign to nearest cluster center (works on CPU)
   from scipy.spatial.distance import cdist
   distances = cdist(X_new_scaled, centers, metric='euclidean')
   cluster_labels = distances.argmin(axis=1)
   ```

3. **If you want GPU re-training** on new data:
   - Set up conda RAPIDS environment (steps above).
   - Load `df_clustering_processed_with_regimes.parquet`.
   - Run the GPU clustering cell — it will automatically detect cuML and train on GPU.

---

## Quick Conda Cheat Sheet

```bash
# List all environments
conda env list

# Activate an environment
conda activate rapids-cu13

# Install additional packages (while activated)
pip install some_package

# Deactivate
conda deactivate

# Remove environment
conda env remove -n rapids-cu13

# Export environment to file (for sharing/reproducibility)
conda env export > rapids-cu13.yml

# Create environment from file on another machine
conda env create -f rapids-cu13.yml
```

---

## Troubleshooting

### "No module named 'cudf'"
- You may be running the notebook in a different Python environment than where RAPIDS is installed.
- If using conda: activate the correct environment (`conda activate rapids-cu13`) before launching Jupyter.
- If using pip: reinstall with `pip install --upgrade cudf-cu13 cuml-cu13` and restart the kernel.

### RAPIDS installation hangs or fails
- Check internet connection and PyPI/conda-forge mirrors are accessible.
- Try `conda update -n base conda` to update conda itself.
- For Windows, use conda/mamba — pip installation of RAPIDS on Windows is unreliable.

### GPU not detected ("nvidia-smi" works but cupy shows device count 0)
- Verify CUDA Toolkit 13.0 is installed and PATH includes CUDA bin directory.
- Check `nvidia-smi` output (you should see your GPU listed).
- Reinstall cupy: `conda install -c conda-forge cupy-cuda-13`.

### Performance is slow on GPU
- Ensure you're running in the GPU environment (activate conda env).
- Check GPU usage: open a separate terminal and run `nvidia-smi` in a loop.
- For small datasets (< 100k rows), GPU overhead may exceed speedup; CPU MiniBatchKMeans can be faster.

---

## References

- RAPIDS Official Docs: [https://rapids.ai/](https://rapids.ai/)
- RAPIDS Start Guide: [https://rapids.ai/start.html](https://rapids.ai/start.html)
- cuDF API: [https://docs.rapids.ai/api/cudf/stable/](https://docs.rapids.ai/api/cudf/stable/)
- cuML API: [https://docs.rapids.ai/api/cuml/stable/](https://docs.rapids.ai/api/cuml/stable/)
- NVIDIA CUDA Toolkit: [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)

---

## Summary

| Method | Setup Time | Reliability (Windows) | Performance | Recommendation |
|--------|------------|----------------------|-------------|-----------------|
| **Conda** (Option A) | 15-20 min | Excellent | GPU (Fast) | ✓ **Best choice** |
| **Pip** (Option B) | 5-10 min | Poor | GPU (Fast if works) | ⚠ Last resort |
| **Docker** (Option C) | 10-15 min | Excellent | GPU (Fast) | ✓ Good alternative |
| **CPU Fallback** (Option D) | 0 min | N/A | CPU (Slow for large data) | Default for now |

---

## Next Steps

1. If you have conda/mamba installed: run the conda setup (Option A) and update your notebook Python kernel to point to the `rapids-cu13` environment.
2. If you don't have conda: download Miniconda from the link above and install it.
3. Once set up, re-run the notebook from the GPU environment.
4. The clustering cell will automatically detect cuML and switch to GPU mode.

For questions or issues, refer to the troubleshooting section above or check the RAPIDS documentation links.
