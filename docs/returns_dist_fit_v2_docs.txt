# returns_dist_fit_v2 Documentation

Comprehensive reference for the returns distribution fitting pipeline.

## Overview
`returns_dist_fit_v2.py` performs a quantitative workflow for a single ticker's return series:
1. Ingest & clean raw price strings (parentheses negatives, commas, currency symbols).
2. Compute lagged returns and drop missing rows.
3. Fit a Gaussian-kernel FFT KDE (non-parametric density).
4. Fit and evaluate all (or user-specified) SciPy continuous distributions with infinite support via KS test.
5. Add KDE itself to KS comparison (empirical CDF vs sample).
6. Select best distribution by highest KS p-value.
7. Plot best parametric PDF vs KDE (skipped if KDE itself wins).
8. Persist plot and KS scan results.

## CLI Usage
You can run via either a JSON config file or command line flags (flags override config values):

```bash
python returns_dist_fit_v2.py --config returns_dist_fit_v2_config.json
```

Or explicitly:
```bash
python returns_dist_fit_v2.py \
  --input path/to/master_stock.csv \
  --ticker TSM \
  --kde-bw ISJ \
  --alpha 0.01 \
  --distributions "t,nct,gennorm,laplace,skewnorm" \
  --start-date 2024-01-01 \
  --end-date 2024-06-30
```

If `--output` (or `output` in config) is omitted, the script auto-generates `<TICKER>_returns_fit.png`.

## Configuration File Example
```json
{
  "input": "C:/data/master_stock.csv",
  "output": null,
  "kde_bw": "ISJ",
  "ticker": "NVDA",
  "start_date": "2024-01-01",
  "end_date": "2024-06-30",
  "date_col": "Date",
  "group_col": "Tic",
  "price_col": "Price",
  "alpha": 0.01,
  "distributions": ["t", "nct", "gennorm", "laplace", "skewnorm"],
  "quiet": false
}
```

## Parameter Reference
| Parameter | Description | Default |
|-----------|-------------|---------|
| `input` | Path to CSV/Parquet data file | REQUIRED |
| `output` | Output plot path (PNG). If null uses `<ticker>_returns_fit.png` | null |
| `ticker` | Symbol to filter (column `group_col`) | None |
| `group_col` | Ticker column name | `Tic` |
| `price_col` | Price column name (cleaned) | `Price` |
| `date_col` | Date column for filtering | `Date` |
| `start_date` / `end_date` | Inclusive date range filters | None |
| `kde_bw` | KDE bandwidth method or numeric value | `ISJ` |
| `alpha` | KS significance threshold | 0.01 |
| `distributions` | Comma string or JSON list of SciPy distribution names | all infinite-support continuous |
| `quiet` | Suppress progress output | false |

## Return Values (from `analyze_returns`)
```python
{
  'returns': np.ndarray,          # cleaned return series
  'kde': FFTKDE,                  # fitted KDE object
  'ks_scan': pd.DataFrame,        # sorted KS results (p_value desc)
  'best_distribution': str,       # name of best distribution (or 'KDE')
  'best_params': tuple|None       # fitted params if parametric
}
```

## KS Scan Output Columns
| Column | Meaning |
|--------|---------|
| `distribution` | SciPy distribution name or `KDE` |
| `D_stat` | KS test statistic |
| `p_value` | KS p-value |
| `decision_alpha` | Reject / Fail to reject at `alpha` |
| `alpha` | Threshold used |
| `parameters` | Fitted params tuple (parametric) or KDE bandwidth string |

## Function Reference
### `coerce_price_column(df, price_col='Price')`
Normalizes price column strings to numeric. Handles commas, currency symbols, plus signs, parentheses for negatives. Returns a copy with coerced column.

### `calculate_returns(df, group_col='Tic', price_col='Price', verbose=True)`
Adds lagged price (`Price_Lag1`) and percentage returns (`Returns`) grouped by ticker. Returns modified DataFrame.

### `fit_kde(data, bw='ISJ', verbose=True)`
Fits FFT-based Gaussian KDE to return series using KDEpy. Bandwidth may be a method name or numeric value.

### `get_infinite_support_continuous_distributions()`
Enumerates SciPy continuous distributions whose support is (-inf, +inf).

### `find_dist(dist_list, sample, alpha=0.01)`
Fits each distribution in `dist_list` to `sample`. Performs KS test and builds a results DataFrame sorted by `p_value` descending.

### `analyze_returns(...)`
Orchestrates entire pipeline. See Return Values section.

### `plot_side_by_side(data, dist_name, params, kde, ticker, output_path)`
Generates dual subplot: Histogram + parametric PDF, Histogram + KDE curve. Skipped if KDE is best distribution.

## Decision Logic for Best Distribution
1. Parametric distributions fitted & tested via KS.
2. KDE empirical CDF tested and appended.
3. Highest KS p-value row selected. If row is KDE distribution, plotting is skipped.

## Example Programmatic Use
```python
import pandas as pd
from returns_dist_fit_v2 import analyze_returns

df = pd.read_csv('master_stock.csv')
results = analyze_returns(df, ticker='TSM', start_date='2024-05-01', end_date='2024-05-31')
print('Best:', results['best_distribution'], results['best_params'])
print(results['ks_scan'].head())
```

## Adding New Distributions
Pass a custom list via CLI: `--distributions "norm,t,skewnorm,gennorm"` or JSON list in config. Ensure names map to attributes in `scipy.stats`.

## Performance Notes
- For large datasets (> millions of rows), filter early by `ticker` and date range to reduce memory.
- KDE bandwidth `ISJ` generally robust; supply numeric bandwidth for fine tuning if needed.
- KS tests run sequentially; limit distribution list to reduce runtime.

## Exit Codes
- `0` success
- `1` missing input file or no distributions fitted

## Generating PDF Docs
After this markdown file exists:
```bash
python generate_returns_dist_fit_pdf.py
```
Produces `docs/returns_dist_fit_v2_docs.pdf` (requires `fpdf` library). Install if missing:
```bash
pip install fpdf
```

## Troubleshooting
| Issue | Resolution |
|-------|------------|
| `ImportError: KDEpy not installed` | `pip install KDEpy` |
| Empty KS scan | Narrow data (maybe all NaNs); verify price coercion succeeded |
| All distributions rejected | Data may violate continuous assumptions; widen time window |
| PNG not created | Ensure ticker provided when `output` is null |

## License
Refer to repository LICENSE.

---
Generated on demand. Keep synchronized with code changes.
