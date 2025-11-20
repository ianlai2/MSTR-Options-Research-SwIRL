"""Options returns analysis with parametric and non-parametric distribution fitting.

Execution Flow:
    1. Read dataframe (parquet/csv)
    2. Calculate returns series grouped by symbol
    3. Drop all missing values
    4. Fit KDE (non-parametric) and Non-Central t (parametric) to returns
    5. Output side-by-side comparison plots

Main Function:
    analyze_returns_and_plot(df, output_path=None, bins=150, kde_bw='silverman',
                             group_col='Symbol', price_col='Stock Price')
        Complete pipeline from dataframe to side-by-side plots.

Core Functions:
    calculate_returns(df, group_col, price_col) -> pd.DataFrame
        Add lagged price and returns columns.
    
    parametric_fit(data) -> dict
        Fit non-central t-distribution, return parameters.
    
    nonparametric_fit(data, bw) -> tuple
        Fit KDE, return (support, density).
    
    plot_side_by_side_fits(data, nct_params, kde_support, kde_density, bins, output_path)
        Create side-by-side comparison plot.

CLI Usage:
    python returns_analysis.py --input MSTR_Options_cleaned.parquet
    python returns_analysis.py --input data.csv --output results.png --bins 200

Dependencies:
    pandas, numpy, scipy, matplotlib, statsmodels, (optional) pyarrow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from KDEpy import FFTKDE
from tqdm import tqdm


# ============================================================================
# CORE PIPELINE FUNCTIONS
# ============================================================================

def calculate_returns(df: pd.DataFrame, group_col: str = 'Symbol', 
                     price_col: str = 'Stock Price', verbose: bool = True) -> pd.DataFrame:
    """Calculate returns using lagged stock prices grouped by symbol.
    
    Args:
        df: DataFrame with stock price data
        group_col: Column to group by (default 'Symbol')
        price_col: Column containing stock prices
        verbose: Show progress bar
    
    Returns:
        DataFrame with added columns: 'Stock.Price_Lag1' and 'Returns'
    """
    df = df.copy()
    
    if verbose:
        with tqdm(total=2, desc="Calculating returns", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            df['Stock.Price_Lag1'] = df.groupby(group_col)[price_col].shift(1)
            pbar.update(1)
            df['Returns'] = (df[price_col] - df['Stock.Price_Lag1']) / df['Stock.Price_Lag1']
            pbar.update(1)
    else:
        df['Stock.Price_Lag1'] = df.groupby(group_col)[price_col].shift(1)
        df['Returns'] = (df[price_col] - df['Stock.Price_Lag1']) / df['Stock.Price_Lag1']
    
    return df


def _get_ticker(df: pd.DataFrame) -> str:
    """Helper function to extract ticker symbol from DataFrame if needed."""
    return df['Symbol'].iloc[0][:4]



def parametric_fit(data: pd.Series | np.ndarray, verbose: bool = True) -> dict:
    """Fit non-central t-distribution to data.
    
    The non-central t-distribution models both heavy tails (df) 
    and skewness (nc parameter).
    
    Args:
        data: Input data for fitting
        verbose: Show progress bar
    
    Returns:
        Dictionary with fitted parameters: df, nc, loc, scale
    """
    if verbose:
        pbar = tqdm(total=100, desc="Fitting NCT distribution", 
                   bar_format='{l_bar}{bar}| {n_fmt}%')
        
        # Custom optimizer that updates progress bar
        def optimizer_with_progress(func, x0, args=(), disp=0):
            iteration_count = {'count': 0}
            
            def callback(xk):
                iteration_count['count'] += 1
                # Update progress bar (estimate ~100 iterations max)
                progress = min(iteration_count['count'], 100)
                pbar.n = progress
                pbar.refresh()
            
            result = minimize(func, x0, args=args, method='Nelder-Mead',
                            callback=callback,
                            options={'disp': disp, 'maxiter': 1000})
            
            if result.success:
                pbar.n = 100
                pbar.refresh()
                pbar.close()
                return result.x
            else:
                pbar.close()
                raise RuntimeError('NCT optimization failed')
        
        params = stats.nct.fit(data, optimizer=optimizer_with_progress)
    else:
        params = stats.nct.fit(data)
    
    df_est, nc_est, loc_est, scale_est = params
    
    return {
        'df': df_est,
        'nc': nc_est,
        'loc': loc_est,
        'scale': scale_est,
        'params': params
    }


def nonparametric_fit(data: pd.Series | np.ndarray, bw: str = 'ISJ', verbose: bool = True) -> FFTKDE:
    """Fit Kernel Density Estimator using FFT method.
    
    Args:
        data: Input data for KDE
        bw: Bandwidth selection method (default 'ISJ')
        verbose: Show progress bar
    
    Returns:
        Fitted FFTKDE object 
    """
    if verbose:
        with tqdm(total=1, desc="Fitting KDE", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            kde = FFTKDE(kernel='gaussian', bw=bw).fit(data.to_numpy().reshape(-1, 1))
            pbar.update(1)
    else:
        kde = FFTKDE(kernel='gaussian', bw=bw).fit(data.to_numpy().reshape(-1, 1))
    
    return kde 



def plot_side_by_side_fits(df: pd.DataFrame, data: pd.Series | np.ndarray, nct_params: dict, 
                           kde: FFTKDE, 
                           output_path: str = None, 
                           figsize: tuple = (16, 6)) -> None:
    """Plot parametric and non-parametric fits in side-by-side subplots.
    
    Args:
        data: Returns data
        nct_params: Non-central t distribution parameters from parametric_fit()
        kde: Fitted KDE object from nonparametric_fit()
        bins: Number of histogram bins
        output_path: If provided, save figure to this path instead of showing
        figsize: Figure size as (width, height) tuple
    """
    x = np.linspace(np.min(data), np.max(data), 500)
    pdf_values = stats.nct.pdf(x, nct_params['df'], nct_params['nc'], 
                               nct_params['loc'], nct_params['scale'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Parametric (NCT)
    bins = int(2 * (len(data) ** (1/3))) 
    ax1.hist(data, bins=bins, density=True, alpha=0.6, color='steelblue', label='Data Histogram')
    ax1.plot(x, pdf_values, 'r--', lw=2.5,
             label=f"NCT Fit (df={nct_params['df']:.1f}, nc={nct_params['nc']:.2f})")
    ax1.set_title('Parametric Fit: Non-Central t-Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Non-parametric (KDE)
    kde_support, kde_density = kde.evaluate()
    ax2.hist(data, bins=bins, density=True, alpha=0.6, color='steelblue', label='Data Histogram')
    ax2.plot(kde_support, kde_density, lw=2.5, color='darkgreen', label='KDE Fit')
    ax2.set_title('Non-Parametric Fit: Kernel Density Estimation', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Returns')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{_get_ticker(df)} Returns Distribution Fitting ({len(data):,} observations)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def analyze_returns_and_plot(df: pd.DataFrame, 
                            output_path: str = None,
                            kde_bw: str = 'ISJ',
                            group_col: str = 'Symbol',
                            price_col: str = 'Stock Price',
                            verbose: bool = True) -> dict:
    """Complete pipeline: calculate returns, fit models, and generate side-by-side plots.
    
    Args:
        df: Input DataFrame with stock price data
        output_path: Path to save figure (if None, displays plot)
        bins: Number of histogram bins
        kde_bw: KDE bandwidth method (default 'ISJ')
        group_col: Column to group by for returns calculation
        price_col: Price column name
        verbose: Print progress information
    
    Returns:
        Dictionary with returns series, NCT params, and KDE fit
    """
    if verbose:
        print(f"Starting analysis on DataFrame with {len(df):,} rows...")
    
    # Step 1: Calculate returns
    if verbose:
        print("Step 1: Calculating returns...")
    df = calculate_returns(df, group_col=group_col, price_col=price_col, verbose=verbose)
    
    # Step 2: Drop missing values
    if verbose:
        print("Step 2: Dropping missing values...")
    original_len = len(df)
    df = df.dropna(subset=['Stock.Price_Lag1', 'Returns'])
    if verbose:
        print(f"  Dropped {original_len - len(df):,} rows ({(original_len - len(df))/original_len*100:.2f}%)")
        print(f"  Remaining: {len(df):,} observations")
    
    returns = df['Returns']
    
    # Step 3: Fit parametric model (Non-Central t)
    if verbose:
        print("Step 3: Fitting parametric model (Non-Central t-distribution)...")
    nct_params = parametric_fit(returns, verbose=verbose)
    if verbose:
        print(f"\n  Parametric Fit Parameters:")
        print(f"  {'='*60}")
        print(f"  Degrees of Freedom (df): {nct_params['df']:.4f}")
        print(f"  Non-centrality (nc):     {nct_params['nc']:.4f}")
        print(f"  Location (loc):          {nct_params['loc']:.6f}")
        print(f"  Scale:                   {nct_params['scale']:.6f}")
        print(f"  {'='*60}\n")
    
    # Step 4: Fit non-parametric model (KDE)
    if verbose:
        print(f"Step 4: Fitting non-parametric model (KDE with {kde_bw} bandwidth)...")
    kde = nonparametric_fit(returns, bw=kde_bw, verbose=verbose)
    kde_support, kde_density = kde.evaluate()   
    if verbose:
        print(f"\n  Non-Parametric Fit Parameters:")
        print(f"  {'='*60}")
        print(f"  Bandwidth method:        {kde_bw}")
        print(f"  Bandwidth value:         {kde.bw:.6f}")
        print(f"  Support points:          {len(kde_support)}")
        print(f"  Support range:           [{kde_support.min():.6f}, {kde_support.max():.6f}]")
        print(f"  {'='*60}\n")
    
    # Step 5: Generate side-by-side plots
    if verbose:
        print("Step 5: Generating side-by-side comparison plots...")
        with tqdm(total=1, desc="Generating plots", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            plot_side_by_side_fits(df, returns, nct_params, kde, 
                                   output_path=output_path)
            pbar.update(1)
    else:
        plot_side_by_side_fits(df, returns, nct_params, kde, 
                               output_path=output_path)
    
    if verbose:
        print("\nAnalysis complete!")
    
    return {
        'returns': returns,
        'nct_params': nct_params,
        'kde': kde
    }


# ============================================================================
# REMOVED/SIMPLIFIED SECTIONS
# ============================================================================
# The following functions are kept for backward compatibility but not used in main pipeline

def plot_kde_overlay(data: pd.Series | np.ndarray, kde: FFTKDE, bins: int = 150) -> None:
    """Plot histogram with KDE overlay (non-parametric fit)."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, density=True, alpha=0.5, color='steelblue', label='Data Histogram')
    plt.plot(kde.support, kde.density, lw=2, color='darkgreen', label='FFT-KDE')
    plt.title(f'Non-Parametric Fit: KDE on {len(data):,} points')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_nct_overlay(data: pd.Series | np.ndarray, nct_params: dict, bins: int = 150) -> None:
    """Plot histogram with fitted non-central t distribution overlay (parametric fit)."""
    x = np.linspace(np.min(data), np.max(data), 500)
    pdf_values = stats.nct.pdf(x, nct_params['df'], nct_params['nc'], 
                               nct_params['loc'], nct_params['scale'])
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='steelblue', label='Data Histogram')
    plt.plot(x, pdf_values, 'r--', lw=2,
             label=f"Skew-t (nct) Fit\n(df={nct_params['df']:.1f}, skew={nct_params['nc']:.1f})")
    plt.title('Parametric Fit: Non-Central t-Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_combined_fit(data: pd.Series | np.ndarray, nct_params: dict, 
                     kde: FFTKDE, 
                     bins: int = 150) -> None:
    """Plot histogram with both parametric (NCT) and non-parametric (KDE) overlays."""
    x = np.linspace(np.min(data), np.max(data), 500)
    pdf_values = stats.nct.pdf(x, nct_params['df'], nct_params['nc'], 
                               nct_params['loc'], nct_params['scale'])
    kde_support, kde_density = kde.evaluate()   
    plt.figure(figsize=(12, 7))
    plt.hist(data, bins=bins, density=True, alpha=0.5, color='steelblue', label='Data Histogram')
    plt.plot(x, pdf_values, 'r--', lw=2.5,
             label=f"Parametric: NCT (df={nct_params['df']:.1f}, nc={nct_params['nc']:.2f})")
    plt.plot(kde_support, kde_density, lw=2.5, color='darkgreen', label='Non-Parametric: KDE')
    plt.title(f'{_get_ticker(data)} Returns Distribution Fits ({len(data):,} observations)')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# ============================================================================
# CLI / MAIN
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
    
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze options returns: calculate returns, fit KDE and NCT, output side-by-side plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config File Format (JSON):
{
    "input": "path/to/data.parquet",
    "output": "path/to/output.png",
    "bins": 150,
    "kde_bw": "silverman",
    "group_col": "Symbol",
    "price_col": "Stock Price",
    "quiet": false
}

Example:
  python returns_analysis.py --config config.json
  python returns_analysis.py --input data.parquet --output plot.png
        """
    )
    parser.add_argument('--config', '-c', help='Path to JSON configuration file')
    parser.add_argument('--input', '-i', help='Path to cleaned parquet/csv file')
    parser.add_argument('--output', '-o', help='Output path for side-by-side plot image', default=None)
    parser.add_argument('--bins', type=int, default=150, help='Number of histogram bins (default: 150)')
    parser.add_argument('--kde-bw', default='silverman', help='KDE bandwidth method (default: silverman)')
    parser.add_argument('--group-col', default='Symbol', help='Column for grouping (default: Symbol)')
    parser.add_argument('--price-col', default='Stock Price', help='Price column (default: Stock Price)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Load config file if provided and merge with command-line args (CLI takes precedence)
    config = {}
    if args.config:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}\n")
    
    # Merge config with args (command-line arguments override config file)
    input_path = args.input if args.input else config.get('input')
    output_path = args.output if args.output else config.get('output')
    bins = args.bins if args.bins != 150 else config.get('bins', 150)
    kde_bw = args.kde_bw if args.kde_bw != 'silverman' else config.get('kde_bw', 'silverman')
    group_col = args.group_col if args.group_col != 'Symbol' else config.get('group_col', 'Symbol')
    price_col = args.price_col if args.price_col != 'Stock Price' else config.get('price_col', 'Stock Price')
    quiet = args.quiet or config.get('quiet', False)
    
    # Load data
    if input_path:
        if input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
        if not quiet:
            print(f"Loaded {input_path}: {len(df):,} rows\n")
    else:
        print("Error: No input data file specified. Use --input or provide in config file.")
        return
    
    # Run complete pipeline
    results = analyze_returns_and_plot(
        df=df,
        output_path=output_path,
        kde_bw=kde_bw,
        group_col=group_col,
        price_col=price_col,
        verbose=not quiet
    )
    
    if not quiet:
        print(f"\nReturns statistics:")
        print(results['returns'].describe())


if __name__ == '__main__':
    main()
