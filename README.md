# MSTR-Options-Research-SwIRL

Research toolkit for options ask price prediction, market regime clustering, and returns distribution analysis.

## Quick Start
Clone the repo, create a `.env` (machine-specific paths), and install Python requirements as needed.

## Returns Distribution CLI (`returns_dist_fit_v2.py`)
Pipeline:
1. Clean raw price strings
2. Compute returns
3. Fit KDE (FFT Gaussian)
4. KS scan of continuous infinite-support SciPy distributions (+ KDE CDF)
5. Plot best parametric vs KDE (skipped if KDE best)
6. Persist plot and KS scan CSV

Basic usage:
```bash
python returns_dist_fit_v2.py --input path/to/master_stock.csv --ticker TSM --kde-bw ISJ --alpha 0.01
```
Using config:
```bash
python returns_dist_fit_v2.py --config returns_dist_fit_v2_config.json
```
If `output` not supplied: auto names `<TICKER>_returns_fit.png` and `<TICKER>_returns_fit_ks_scan.csv`.

Full documentation: `docs/returns_dist_fit_v2.md`

Generate PDF (requires `fpdf`):
```bash
python generate_returns_dist_fit_pdf.py
```
Output: `docs/returns_dist_fit_v2_docs.pdf`

## Market Regime Clustering
Notebook `options_ask_prediction.ipynb` builds engineered features, scales per symbol, optimizes K via cross-validated silhouette/inertia, and persists a pipeline with KMeans + scaling.

## Environment Note
Create `.env` for local path overrides (e.g. Dropbox mount points).

## License
See `LICENSE`.

