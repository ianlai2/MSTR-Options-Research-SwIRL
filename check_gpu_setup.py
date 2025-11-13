#!/usr/bin/env python
"""
GPU Setup Verification & Troubleshooting Script

Usage:
    python check_gpu_setup.py

This script checks for CUDA, RAPIDS (cuDF/cuML), and other GPU-related packages.
Run this in the environment where you plan to run the notebook.
"""

import sys
import importlib
import subprocess
import platform

def get_version(pkg_name):
    """Safely get package version."""
    try:
        m = importlib.import_module(pkg_name)
        return getattr(m, '__version__', 'unknown version')
    except Exception as e:
        return f"Not installed ({e})"

def run_cmd(cmd, description):
    """Run shell command and print output."""
    try:
        result = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT)
        print(f"✓ {description}")
        print(f"  {result.strip()}\n")
        return True
    except Exception as e:
        print(f"✗ {description}")
        print(f"  Error: {e}\n")
        return False

def main():
    print("=" * 60)
    print("  GPU Setup Verification for Options Clustering")
    print("=" * 60)
    print()
    
    # System info
    print("System Information:")
    print(f"  Python: {sys.version}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Executable: {sys.executable}\n")
    
    # Check for CUDA
    print("Checking CUDA / GPU Setup:")
    cuda_available = run_cmd(['nvidia-smi', '--query-gpu=driver_version,name,memory.total', '--format=csv,noheader'],
                              'nvidia-smi (GPU driver info)')
    
    # Check Python packages
    print("Checking Python Packages:")
    
    packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'sklearn': 'CPU Machine Learning (scikit-learn)',
        'pyarrow': 'Parquet I/O',
        'cudf': 'GPU DataFrame (RAPIDS)',
        'cuml': 'GPU Machine Learning (RAPIDS)',
        'cupy': 'GPU NumPy-like arrays',
        'umap': 'Nonlinear dimensionality reduction',
        'pynndescent': 'Fast nearest neighbors',
    }
    
    rapids_available = False
    
    for pkg, description in packages.items():
        version = get_version(pkg)
        status = "✓" if "Not installed" not in version else "✗"
        print(f"  {status} {pkg:15s} ({description:40s}): {version}")
        if pkg in ['cudf', 'cuml'] and 'Not installed' not in version:
            rapids_available = True
    
    print()
    
    # Check cupy GPU details
    print("GPU Details (via cupy, if available):")
    try:
        import cupy as cp
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            print(f"  ✓ GPU device count: {device_count}")
            for i in range(device_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
                print(f"    Device {i}: {name}")
        except Exception as e:
            print(f"  ✗ Could not query GPU details: {e}")
    except ImportError:
        print(f"  ℹ cupy not installed (GPU array support unavailable)\n")
    
    # Summary & Recommendations
    print("=" * 60)
    print("Summary & Recommendations:")
    print("=" * 60)
    
    if rapids_available:
        print("✓ RAPIDS (cuDF/cuML) is installed.")
        print("  → GPU acceleration is ENABLED in the notebook clustering code.")
        print("  → Proceed with running the notebook.")
    else:
        print("✗ RAPIDS (cuDF/cuML) is NOT installed.")
        print("  → Clustering will fall back to CPU (MiniBatchKMeans).")
        print("\n  To enable GPU acceleration:")
        print("    1. Install Miniconda: https://docs.conda.io/projects/miniconda/")
        print("    2. Run: setup_rapids_env.bat (Windows) or follow GPU_SETUP.md")
        print("    3. Re-launch Jupyter from the 'rapids-cu13' environment")
        print("    4. Re-run the notebook clustering cells")
    
    print()
    if cuda_available:
        print("✓ NVIDIA GPU and driver detected.")
    else:
        print("⚠ GPU driver not detected (nvidia-smi not found).")
        print("  → Check NVIDIA drivers are installed: https://nvidia.com/drivers")
    
    print()
    print("For more details, see: GPU_SETUP.md")
    print()

if __name__ == '__main__':
    main()
