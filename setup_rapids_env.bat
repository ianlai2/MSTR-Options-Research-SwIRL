@echo off
REM Setup RAPIDS conda environment for Options Clustering Project
REM This script creates a dedicated 'rapids-cu13' environment with GPU acceleration
REM
REM Prerequisites: Miniconda/Anaconda installed and conda available in PATH
REM
REM Usage: Double-click this file or run: setup_rapids_env.bat

echo.
echo ========================================
echo  RAPIDS Environment Setup (CUDA 13)
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: conda not found in PATH
    echo Please install Miniconda/Anaconda first:
    echo   https://docs.conda.io/projects/miniconda/en/latest/
    echo.
    pause
    exit /b 1
)

echo Conda found. Proceeding with environment creation...
echo.

REM Create environment
echo Creating conda environment 'rapids-cu13'...
call conda create -n rapids-cu13 -c rapidsai -c nvidia -c conda-forge ^
  cudf=25.10 cuml=25.10 cupy=12.0 ^
  python=3.11 cudatoolkit=13.0 ^
  pandas numpy scikit-learn matplotlib seaborn ^
  jupyter jupyterlab pyarrow joblib -y

if errorlevel 1 (
    echo.
    echo ERROR: conda create failed. Check the output above.
    pause
    exit /b 1
)

echo.
echo Environment created successfully!
echo.

REM Install additional packages
echo Installing additional packages (umap-learn, pynndescent)...
call conda run -n rapids-cu13 pip install --upgrade umap-learn pynndescent

if errorlevel 1 (
    echo WARNING: pip install had issues, but environment may still be usable.
    echo Try manually: conda activate rapids-cu13 ^&^& pip install umap-learn pynndescent
)

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Open a new PowerShell or Command Prompt
echo   2. Run: conda activate rapids-cu13
echo   3. Launch Jupyter: jupyter lab
echo   4. Open options_ask_prediction.ipynb
echo   5. Run the clustering cells (GPU will be auto-detected)
echo.
echo To verify GPU setup, run the "Environment & GPU detection" cell in the notebook.
echo.
pause
