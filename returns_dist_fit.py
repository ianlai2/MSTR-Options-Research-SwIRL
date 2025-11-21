"""returns_dist_fit_v2

Pipeline (requested order):
1. Ingest data & clean price column
2. Calculate returns (drop NaNs)
3. Fit KDE to returns series
4. Find best continuous distribution (infinite support) using KS tests
5. Generate side-by-side plot (best parametric vs KDE)
6. Return best fit parametric distribution name & parameters + KS scan table

Config/CLI parameters supported:
- input: path to CSV/Parquet data
- output: path for final image (CSV summaries derived from stem)
- kde_bw: bandwidth method for FFTKDE (default ISJ)
- ticker: filter DataFrame by ticker symbol (column default 'Tic')
- start_date, end_date: inclusive date range filters (on date_col)
- date_col: name of date column (default 'Date')
- group_col: ticker column (default 'Tic')
- price_col: price column (default 'Price')
- distributions: comma separated list of scipy.stats distribution names (default: all infinite support continuous)
- alpha: KS test significance threshold (default 0.01)
- quiet: suppress verbose output

Output files (if output specified):
- <output_image> (PNG) side-by-side plot
- <output_image_stem>_ks_scan.csv full KS results

Requires: pandas, numpy, scipy, matplotlib, tqdm, KDEpy (for KDE).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
try:
    from KDEpy import FFTKDE
except ImportError:
    FFTKDE = None
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Cleaning utilities
# ----------------------------------------------------------------------------

def coerce_price_column(df: pd.DataFrame, price_col: str = 'Price') -> pd.DataFrame:
    """Coerce a raw price column containing messy string representations into numeric floats.

    This function attempts to locate a usable price column, applying a series of
    regex‐based normalizations (handling parentheses for negatives, removing
    commas, currency symbols, stray quotes and any non numeric characters) and
    then converts the cleaned strings to numeric values.

    If the provided `price_col` is not present, several common fallbacks are
    searched. The original DataFrame is never modified in place.

    Parameters
    ----------
    df : pd.DataFrame
        Source data containing at least one price column candidate.
    price_col : str, default 'Price'
        Preferred price column name. Fallbacks are tried if absent.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with the resolved price column coerced
        to numeric (invalid parses become NaN).

    Raises
    ------
    KeyError
        If no suitable price column can be found.

    Notes
    -----
    The cleaning intentionally strips anything not part of a standard float
    representation (digits, period, minus sign). Parenthetical negatives are
    converted ("(123.4)" -> "-123.4").
    """
    target = df.copy()
    if price_col not in target.columns:
        for alt in ['Price', 'prices_clean', 'Stock Price']:
            if alt in target.columns:
                price_col = alt
                break
    if price_col not in target.columns:
        raise KeyError("No price-like column found for cleaning.")
    raw = target[price_col].astype(str).str.strip()
    cleaned = (raw
               .str.replace(r'\(([^)]+)\)', r'-\1', regex=True)
               .str.replace("'", '', regex=False)
               .str.replace(',', '', regex=False)
               .str.replace(r'[\$+]', '', regex=True)
               .str.replace(r'[^0-9.\-]', '', regex=True))
    target[price_col] = pd.to_numeric(cleaned, errors='coerce')
    return target

# ----------------------------------------------------------------------------
# Returns calculation
# ----------------------------------------------------------------------------

def calculate_returns(df: pd.DataFrame, group_col: str = 'Tic', price_col: str = 'Price', verbose: bool = True) -> pd.DataFrame:
    """Calculate simple returns for each group (ticker) and append columns.

    Adds two columns:
    - `Price_Lag1`: Previous price within each group.
    - `Returns`: (Price - Price_Lag1) / Price_Lag1.

    Parameters
    ----------
    df : pd.DataFrame
        Input price time series with at least group and price columns.
    group_col : str, default 'Tic'
        Column identifying securities/instruments (grouping key).
    price_col : str, default 'Price'
        Column containing numeric prices.
    verbose : bool, default True
        If True, shows a tqdm progress bar for the two operations.

    Returns
    -------
    pd.DataFrame
        Copy of input with `Price_Lag1` and `Returns` appended (may contain
        NaNs for the first observation per group).

    Raises
    ------
    KeyError
        If either the group or price column is missing.
    """
    out = df.copy()
    if price_col not in out.columns:
        raise KeyError(f"Price column '{price_col}' not found.")
    if group_col not in out.columns:
        raise KeyError(f"Group column '{group_col}' not found.")
    if verbose:
        with tqdm(total=2, desc="Calculating returns", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            out['Price_Lag1'] = out.groupby(group_col)[price_col].shift(1)
            pbar.update(1)
            out['Returns'] = (out[price_col] - out['Price_Lag1']) / out['Price_Lag1']
            pbar.update(1)
    else:
        out['Price_Lag1'] = out.groupby(group_col)[price_col].shift(1)
        out['Returns'] = (out[price_col] - out['Price_Lag1']) / out['Price_Lag1']
    return out

# ----------------------------------------------------------------------------
# Distribution scanning
# ----------------------------------------------------------------------------

def get_infinite_support_continuous_distributions() -> List[str]:
    """Enumerate SciPy continuous distributions with infinite real support.

    Searches the ``scipy.stats`` namespace for instances of ``rv_continuous``
    whose lower bound ``a`` is ``-inf`` and upper bound ``b`` is ``inf``.

    Returns
    -------
    list of str
        Distribution names suitable for fitting to unbounded real data.

    Notes
    -----
    This intentionally excludes bounded, semi-bounded and discrete
    distributions. The list can vary between SciPy versions.
    """
    names = []
    for d in dir(stats):
        obj = getattr(stats, d)
        if isinstance(obj, stats.rv_continuous):
            if obj.a == -np.inf and obj.b == np.inf:
                names.append(obj.name)
    return names

def find_dist(dist_list: List[str], sample: np.ndarray, alpha: float = 0.01) -> pd.DataFrame:
    """Fit and KS-test a collection of distributions against a sample.

    For each distribution name provided, ``scipy.stats`` is used to ``fit``
    parameters via MLE, then a one-sample Kolmogorov–Smirnov test is applied.
    Results are collected, sorted by descending p-value and returned.

    Parameters
    ----------
    dist_list : list of str
        Distribution names accessible as ``getattr(stats, name)``.
    sample : np.ndarray
        1D array of sample observations.
    alpha : float, default 0.01
        Significance level used to derive the textual decision column.

    Returns
    -------
    pd.DataFrame
        Columns: ``distribution``, ``D_stat``, ``p_value``, ``decision_alpha``,
        ``alpha``, ``parameters``. Sorted by ``p_value`` descending. May be
        empty if all fits fail.

    Notes
    -----
    - Exceptions during fitting or testing are swallowed (row skipped).
    - Parameter tuples follow SciPy's (shape..., loc, scale) convention.
    - Use caution interpreting p-values with large samples (may reject slight
      deviations).
    """
    rows = []
    for dist_name in dist_list:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(sample)
            ks_stat, ks_p = stats.kstest(sample, dist_name, args=params)
            rows.append({
                'distribution': dist_name,
                'D_stat': ks_stat,
                'p_value': ks_p,
                'decision_alpha': 'Fail to reject' if ks_p > alpha else 'Reject',
                'alpha': alpha,
                'parameters': params
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=['distribution','D_stat','p_value','decision_alpha','alpha','parameters'])
    return pd.DataFrame(rows).sort_values(by='p_value', ascending=False).reset_index(drop=True)

# ----------------------------------------------------------------------------
# KDE fitting
# ----------------------------------------------------------------------------

def fit_kde(data: np.ndarray, bw: str = 'ISJ', verbose: bool = True) -> Any:
    """Fit a Gaussian FFT-based KDE to 1D data using KDEpy.

    Parameters
    ----------
    data : np.ndarray
        1D array of observations.
    bw : str or float, default 'ISJ'
        Bandwidth specification passed to ``FFTKDE`` (e.g., 'ISJ', 'silverman').
    verbose : bool, default True
        If True, displays a progress bar for the fitting step.

    Returns
    -------
    KDEpy.FFTKDE
        Fitted KDE object. Use ``evaluate()`` to obtain support and density.

    Raises
    ------
    ImportError
        If KDEpy is not installed.
    """
    if FFTKDE is None:
        raise ImportError("KDEpy not installed. Install with: pip install KDEpy")
    if verbose:
        with tqdm(total=1, desc="Fitting KDE", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            kde = FFTKDE(kernel='gaussian', bw=bw).fit(data.reshape(-1, 1))
            pbar.update(1)
    else:
        kde = FFTKDE(kernel='gaussian', bw=bw).fit(data.reshape(-1, 1))
    return kde

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def plot_side_by_side(data: np.ndarray, dist_name: str, params: tuple, kde: Any, ticker: str, output_path: Optional[str]) -> None:
    """Generate a side-by-side parametric vs KDE returns distribution plot.

    Left panel: Histogram with overlaid parametric PDF for the best fitted
    distribution. Right panel: Histogram with overlaid KDE density curve.

    Parameters
    ----------
    data : np.ndarray
        1D returns series.
    dist_name : str
        Name of best parametric distribution.
    params : tuple
        Fitted parameter tuple for the distribution (shape..., loc, scale).
    kde : Any
        Fitted KDE object supporting ``evaluate()``.
    ticker : str
        Label used in the plot title.
    output_path : str or None
        If provided, path to save PNG (created/overwritten). If None, shows plot.

    Returns
    -------
    None

    Notes
    -----
    Uses Freedman–Diaconis rule ('fd') for histogram bin sizing. Figure is
    closed after saving/showing to avoid memory accumulation in batch runs.
    """
    x = np.linspace(np.min(data), np.max(data), 500)
    dist = getattr(stats, dist_name)
    pdf_vals = dist.pdf(x, *params)
    bins = 'fd'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # Parametric
    ax1.hist(data, bins=bins, density=True, alpha=0.6, color='steelblue', label='Histogram')
    ax1.plot(x, pdf_vals, 'r--', lw=2, label=f"{dist_name} PDF")
    ax1.set_title(f'Best Parametric: {dist_name}')
    ax1.set_xlabel('Returns'); ax1.set_ylabel('Density'); ax1.legend(); ax1.grid(True, alpha=0.3)
    # KDE
    support, density = kde.evaluate()
    ax2.hist(data, bins=bins, density=True, alpha=0.6, color='steelblue', label='Histogram')
    ax2.plot(support, density, lw=2, color='darkgreen', label='KDE')
    ax2.set_title('Non-Parametric: KDE')
    ax2.set_xlabel('Returns'); ax2.set_ylabel('Density'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.suptitle(f'{ticker} Returns Fit ({len(data):,} observations)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
    else:
        plt.show()
    plt.close(fig)

# ----------------------------------------------------------------------------
# Main analysis function
# ----------------------------------------------------------------------------

def analyze_returns(df: pd.DataFrame,
                    output_path: Optional[str] = None,
                    kde_bw: str = 'ISJ',
                    group_col: str = 'Tic',
                    price_col: str = 'Price',
                    ticker: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    date_col: str = 'Date',
                    distributions: Optional[List[str]] = None,
                    alpha: float = 0.01,
                    verbose: bool = True) -> dict:
    """End-to-end returns analysis pipeline producing fits, plot and summary.

    Steps:
    1. Clean price column.
    2. Optional ticker/date filtering.
    3. Compute lagged prices and simple returns; drop NaNs.
    4. Fit KDE to returns.
    5. Determine parametric distribution list (all infinite support if not provided).
    6. KS scan for each parametric distribution.
    7. Perform KS test for KDE via empirical CDF integration.
    8. Rank all (parametric + KDE) by p-value, choose best.
    9. Plot parametric vs KDE (if best is parametric) and persist KS scan CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing at least price, group, and optional date columns.
    output_path : str, optional
        Path for output plot. If None and `ticker` provided, auto-names ``<ticker>_returns_fit.png``.
    kde_bw : str, default 'ISJ'
        Bandwidth specification for FFTKDE.
    group_col : str, default 'Tic'
        Column identifying instrument/group for lagged return calculation.
    price_col : str, default 'Price'
        Column holding prices to clean and use for returns.
    ticker : str, optional
        If set, filters DataFrame to matching group before analysis.
    start_date, end_date : str, optional
        ISO date strings delimiting inclusive date filter on `date_col`.
    date_col : str, default 'Date'
        Date column name (parsed to datetime if filtering applied).
    distributions : list of str, optional
        User-specified distribution names to scan. If None, all infinite support continuous.
    alpha : float, default 0.01
        Significance threshold for KS decision labeling.
    verbose : bool, default True
        If True, prints progress messages.

    Returns
    -------
    dict
        Keys:
        - ``returns``: 1D numpy array of cleaned returns.
        - ``kde``: fitted KDE object.
        - ``ks_scan``: DataFrame of KS results including KDE row.
        - ``best_distribution``: name of highest p-value entry.
        - ``best_params``: parameter tuple (None if best is KDE).

    Raises
    ------
    RuntimeError
        If no parametric distributions fit successfully.

    Notes
    -----
    KDE KS test constructs an empirical CDF by cumulative trapezoid-like
    integration over evaluated support. Plot skipped when KDE is top performer.
    """
    if verbose:
        print(f"Starting with {len(df):,} rows")

    # 1. Clean prices
    df = coerce_price_column(df, price_col)

    # Ticker filter
    if ticker:
        before = len(df)
        df = df[df[group_col] == ticker].copy()
        if verbose:
            print(f"Ticker filter {ticker}: {before:,} -> {len(df):,}")

    # Date filtering
    if date_col in df.columns and (start_date or end_date):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format='mixed')
        before = len(df)
        if start_date:
            df = df[df[date_col] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df[date_col] <= pd.to_datetime(end_date)]
        if verbose:
            print(f"Date filter: {before:,} -> {len(df):,}")

    # 2. Calculate returns
    if verbose: print("Calculating returns...")
    df = calculate_returns(df, group_col=group_col, price_col=price_col, verbose=verbose)
    before = len(df)
    df = df.dropna(subset=['Price_Lag1', 'Returns'])
    if verbose:
        dropped = before - len(df)
        print(f"Dropped {dropped:,} rows with NaNs in returns ({dropped/before*100:.2f}%)")

    returns = df['Returns'].to_numpy()

    # 3. Fit KDE
    if verbose: print(f"Fitting KDE (bw={kde_bw})...")
    kde = fit_kde(returns, bw=kde_bw, verbose=verbose)
    if verbose:
        print(f"KDE bandwidth: {kde.bw:.6f}")

    # 4. Determine distributions list
    if distributions is None:
        distributions = get_infinite_support_continuous_distributions()
        if verbose:
            print(f"Scanning {len(distributions)} infinite-support continuous distributions")
    else:
        if verbose:
            print(f"Scanning user-specified {len(distributions)} distributions")

    # 5. KS scan (parametric distributions)
    if verbose: print("Running KS scan...")
    ks_df = find_dist(distributions, returns, alpha=alpha)
    if ks_df.empty:
        raise RuntimeError("No distributions successfully fitted.")
    
    # 5b. KS test for KDE (using empirical CDF)
    if verbose: print("KS test for KDE...")
    def kde_cdf(x):
        support, density = kde.evaluate()
        # Integrate density up to x using cumulative sum
        dx = np.diff(support)
        cumulative = np.cumsum(density[:-1] * dx)
        cumulative = np.concatenate([[0], cumulative])
        return np.interp(x, support, cumulative)
    
    ks_kde_stat, ks_kde_p = stats.kstest(returns, kde_cdf)
    kde_row = pd.DataFrame([{
        'distribution': 'KDE',
        'D_stat': ks_kde_stat,
        'p_value': ks_kde_p,
        'decision_alpha': 'Fail to reject' if ks_kde_p > alpha else 'Reject',
        'alpha': alpha,
        'parameters': f'bw={kde.bw:.6f}'
    }])
    
    # Concatenate and re-sort by p_value
    ks_df = pd.concat([ks_df, kde_row], ignore_index=True)
    ks_df = ks_df.sort_values(by='p_value', ascending=False).reset_index(drop=True)
    
    # Identify best distribution (highest p-value)
    best = ks_df.iloc[0]
    best_dist = best['distribution']
    best_params = best['parameters'] if best_dist != 'KDE' else None
    if verbose:
        print("Best fit distribution:")
        print(best)

    # 6. Plot
    if not output_path and ticker:
        output_path = f"{ticker}_returns_fit.png"
    
    # Only plot if best is not KDE (parametric distribution)
    if best_dist != 'KDE':
        plot_side_by_side(returns, best_dist, tuple(best_params), kde, ticker or 'TICKER', output_path)
    else:
        if verbose:
            print(f"Best fit is KDE (non-parametric), skipping parametric vs KDE plot.")

    # 7. Persist KS scan
    if output_path:
        scan_path = Path(output_path).with_suffix('').as_posix() + '_ks_scan.csv'
        ks_df.to_csv(scan_path, index=False)
        if verbose:
            print(f"Saved KS scan: {scan_path}")

    return {
        'returns': returns,
        'kde': kde,
        'ks_scan': ks_df,
        'best_distribution': best_dist,
        'best_params': best_params
    }

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a JSON configuration file.

    Parameters
    ----------
    path : str
        Path to JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    with open(path, 'r') as f:
        return json.load(f)

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for returns distribution fitting script.

    Returns
    -------
    argparse.Namespace
        Namespace containing all parsed CLI parameters.
    """
    p = argparse.ArgumentParser(description="Returns distribution fitting (KDE + best parametric)")
    p.add_argument('--config', '-c', help='Path to JSON config file')
    p.add_argument('--input', '-i', help='Path to CSV/Parquet data')
    p.add_argument('--output', '-o', help='Output image path (PNG)')
    p.add_argument('--kde-bw', default='ISJ', help='KDE bandwidth method (default ISJ)')
    p.add_argument('--group-col', default='Tic', help='Ticker column (default Tic)')
    p.add_argument('--price-col', default='Price', help='Price column (default Price)')
    p.add_argument('--ticker', help='Ticker symbol filter')
    p.add_argument('--start-date', help='Start date YYYY-MM-DD')
    p.add_argument('--end-date', help='End date YYYY-MM-DD')
    p.add_argument('--date-col', default='Date', help='Date column name (default Date)')
    p.add_argument('--distributions', help='Comma separated list of distributions to test')
    p.add_argument('--alpha', type=float, default=0.01, help='Alpha for KS test decisions (default 0.01)')
    p.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose output')
    return p.parse_args()

def main() -> None:
    """CLI entry point.

    Merges configuration file values with CLI overrides, loads data, executes
    the analysis pipeline, and prints summary statistics and top KS scan rows.
    """
    args = parse_args()
    config = load_config(args.config) if args.config else {}

    # Merge precedence: CLI overrides config
    input_path = args.input or config.get('input')
    output_path = args.output or config.get('output')
    kde_bw = args.kde_bw if args.kde_bw != 'ISJ' else config.get('kde_bw', 'ISJ')
    group_col = args.group_col if args.group_col != 'Tic' else config.get('group_col', 'Tic')
    price_col = args.price_col if args.price_col != 'Price' else config.get('price_col', 'Price')
    ticker = args.ticker or config.get('ticker')
    start_date = args.start_date or config.get('start_date')
    end_date = args.end_date or config.get('end_date')
    date_col = args.date_col if args.date_col != 'Date' else config.get('date_col', 'Date')
    alpha = args.alpha if args.alpha != 0.01 else config.get('alpha', 0.01)
    quiet = args.quiet or config.get('quiet', False)
    distributions_arg = args.distributions or config.get('distributions')
    distributions = None
    if distributions_arg:
        if isinstance(distributions_arg, list):
            distributions = distributions_arg
        else:
            distributions = [d.strip() for d in distributions_arg.split(',') if d.strip()]

    if not input_path:
        print('Error: input path required (use --input or config file).')
        return

    # Load data
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    if not quiet:
        print(f"Loaded {input_path}: {len(df):,} rows")

    results = analyze_returns(df=df,
                              output_path=output_path,
                              kde_bw=kde_bw,
                              group_col=group_col,
                              price_col=price_col,
                              ticker=ticker,
                              start_date=start_date,
                              end_date=end_date,
                              date_col=date_col,
                              distributions=distributions,
                              alpha=alpha,
                              verbose=not quiet)

    if not quiet:
        print('\nReturns summary:')
        print(pd.Series(results['returns']).describe())
        print('\nBest distribution:')
        print(results['best_distribution'])
        print('Parameters:', results['best_params'])
        print('\nTop KS scan head:')
        print(results['ks_scan'].head())

if __name__ == '__main__':
    main()
