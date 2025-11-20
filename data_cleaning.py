"""Data cleaning utilities for MSTR options dataset.

Functions provided:
    load_data(path: str | None = None, env_var: str = 'DATA_PATH') -> pd.DataFrame
        Load raw data from a CSV path or environment variable.

    filter_trading_days(df: pd.DataFrame, calendar: str = 'NYSE') -> pd.DataFrame
        Return a copy of the dataframe restricted to exchange trading days.

    validate_stock_price_encoding(df: pd.DataFrame, column: str = 'Stock Price') -> dict
        Detect rows where the price column is an empty array or empty string.

    export_cleaned(df: pd.DataFrame, parquet_name: str, csv_name: str) -> str
        Attempt Parquet export (pyarrow) with CSV fallback; return output path.

    query_by_strike_expiry(df: pd.DataFrame, strike: float, expiration: str,
                           strike_col: str = 'Strike Price',
                           expiry_col: str = 'Expiration Date') -> pd.DataFrame
        Convenience query replicating notebook example.

    group_symbol_head(df: pd.DataFrame, symbol_col: str = 'Symbol', n: int = 5) -> pd.DataFrame
        Return first n rows per symbol group.

CLI Usage (updated):
    # Export CSV (default)
    python data_cleaning.py --input path/to/file.csv --output cleaned_options

    # Explicit CSV filename
    python data_cleaning.py --input path/to/file.csv --output cleaned_options.csv

    # Export Parquet
    python data_cleaning.py --input path/to/file.csv --output cleaned_options --parquet

    # Query strike + expiry after cleaning
    python data_cleaning.py -i raw.csv --strike 80 --expiry 2027-12-17 --group-head 3

    # Use JSON config (keys mirror CLI dest names; CLI overrides config)
    python data_cleaning.py --config config.json

JSON Config Example (config.json):
{
  "input": "path/to/file.csv",
  "output": "cleaned_options",
  "parquet": true,
  "calendar": "NYSE",
  "show_missing": true,
  "validate_price": true,
  "strike": 80,
  "expiry": "2027-12-17",
  "group_head": 3
}

Dependencies:
    pandas, numpy, pandas_market_calendars, (optional) pyarrow

Note: This module avoids implicit reliance on environment variables except
      where explicitly requested by passing None for path.
"""

from __future__ import annotations

import os
import argparse
from typing import Any, Dict

import numpy as np
import pandas as pd
import json

def _load_json_config(path: str) -> Dict[str, Any]:
    """Load JSON config file returning a dict. Empty file yields empty dict.

    Provides clear errors if file unreadable or top-level is not an object.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        if os.path.getsize(path) == 0:
            return {}
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON structure must be an object/dict.")
    return data

def load_data(path: str | None = None, env_var: str = 'DATA_PATH') -> pd.DataFrame:
    """Load raw CSV data.

    If `path` is None, attempts to read from environment variable `env_var`.
    Raises FileNotFoundError if resolved path does not exist.
    """
    if path is None:
        path = os.getenv(env_var)
        if not path:
            raise FileNotFoundError(f"Environment variable {env_var} not set or empty.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)

def _ensure_datetime(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])

def filter_trading_days(df: pd.DataFrame, calendar: str = 'NYSE', timestamp_col: str = 'Timestamp',
                        expiry_col: str = 'Expiration Date') -> pd.DataFrame:
    """Restrict dataframe to exchange trading days using pandas_market_calendars.

    Converts timestamp/expiry columns to datetime first. Returns a copy.
    """
    import pandas_market_calendars as mcal  # local import to keep base load light
    _ensure_datetime(df, [timestamp_col, expiry_col])
    start = df[timestamp_col].dt.date.min()
    end = df[timestamp_col].dt.date.max()
    cal = mcal.get_calendar(calendar)
    schedule = cal.schedule(start_date=start, end_date=end)
    trading_days = pd.to_datetime(schedule.index).date
    mask = df[timestamp_col].dt.date.isin(trading_days)
    return df.loc[mask].copy()

def validate_stock_price_encoding(df: pd.DataFrame, column: str = 'Stock Price') -> Dict[str, Any]:
    """Return summary of rows where `column` is an empty array/tuple/ndarray or empty string."""
    def is_empty_array(x: Any) -> bool:
        return isinstance(x, (list, tuple, np.ndarray)) and len(x) == 0
    def is_empty_string(x: Any) -> bool:
        return isinstance(x, str) and x.strip() == ''
    empty_array_mask = df[column].apply(is_empty_array)
    empty_string_mask = df[column].apply(is_empty_string)
    combined_mask = empty_array_mask | empty_string_mask
    return {
        'empty_array_count': int(empty_array_mask.sum()),
        'empty_string_count': int(empty_string_mask.sum()),
        'total_problematic_rows': int(combined_mask.sum()),
        'empty_array_examples': df.loc[empty_array_mask].head(10),
        'empty_string_examples': df.loc[empty_string_mask].head(10)
    }

def export_cleaned(df: pd.DataFrame, parquet_name: str, csv_name: str) -> str:
    """Export cleaned dataframe to Parquet (preferred) with CSV fallback.

    Returns path used.
    """
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(parquet_name, engine='pyarrow')
        return parquet_name
    except Exception:
        df.to_csv(csv_name, index=False)
        return csv_name

def query_by_strike_expiry(df: pd.DataFrame, strike: float, expiration: str,
                           strike_col: str = 'Strike Price',
                           expiry_col: str = 'Expiration Date') -> pd.DataFrame:
    """Replicates notebook query for a specific strike and expiration date (string parse)."""
    _ensure_datetime(df, [expiry_col])
    # Accept various date formats; convert input expiration to date
    target_date = pd.to_datetime(expiration).date()
    return df[(df[strike_col] == strike) & (df[expiry_col].dt.date == target_date)]

def group_symbol_head(df: pd.DataFrame, symbol_col: str = 'Symbol', n: int = 5) -> pd.DataFrame:
    """Return first n rows of each symbol group."""
    return df.groupby(symbol_col).head(n)

def _missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    ms = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Percentage Missing': (df.isnull().sum() / len(df)) * 100
    }).sort_values('Missing Values', ascending=False)
    return ms

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean options data and export.")
    parser.add_argument('--input', '-i', help='Path to raw CSV (defaults to env DATA_PATH)', default=None)
    parser.add_argument('--output', '-o', help='Base output filename (with or without extension). If omitted, uses "cleaned_options".', default=None)
    parser.add_argument('--parquet', action='store_true', help='Export Parquet (default CSV)')
    parser.add_argument('--calendar', default='NYSE', help='Market calendar code (default NYSE)')
    parser.add_argument('--show-missing', action='store_true', help='Print missing value summary')
    parser.add_argument('--validate-price', action='store_true', help='Validate Stock Price encoding')
    parser.add_argument('--strike', type=float, help='Optional strike for query example')
    parser.add_argument('--expiry', help='Optional expiration date for query example (e.g. 2027-12-17)')
    parser.add_argument('--group-head', type=int, default=0, help='If >0, show head rows per symbol')
    parser.add_argument('--config', '-c', help='Path to JSON config providing defaults. CLI flags override config.', default=None)
    args = parser.parse_args()

    # If config provided, merge values for any argument still at an "unset" state.
    # Unset heuristic: value is None, False, or 0 (for group_head). CLI precedence retained.
    if args.config:
        try:
            cfg = _load_json_config(args.config)
        except Exception as e:
            raise SystemExit(f"Failed to load config: {e}")
        accepted = {'input', 'output', 'parquet', 'calendar', 'show_missing', 'validate_price', 'strike', 'expiry', 'group_head'}
        applied = {}
        for k, v in cfg.items():
            if k not in accepted:
                continue
            current = getattr(args, k)
            if current in (None, False, 0):
                setattr(args, k, v)
                applied[k] = v
        print("Loaded JSON config:", args.config)
        if applied:
            print("Applied config defaults for:", ", ".join(f"{k}={applied[k]}" for k in sorted(applied)))
        else:
            print("Config provided but all values overridden by explicit CLI flags.")

    print("="*80)
    print("OPTIONS DATA CLEANING PIPELINE")
    print("="*80)
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    df_raw = load_data(args.input)
    print(f"  Loaded {len(df_raw):,} rows from: {args.input or os.getenv('DATA_PATH')}")
    
    # Step 2: Filter to trading days
    print(f"\nStep 2: Filtering to {args.calendar} trading days...")
    df_filtered = filter_trading_days(df_raw, calendar=args.calendar)
    rows_removed = len(df_raw) - len(df_filtered)
    print(f"  Removed {rows_removed:,} non-trading day rows ({rows_removed/len(df_raw)*100:.2f}%)")
    print(f"  Remaining: {len(df_filtered):,} rows")

    if args.show_missing:
        print("\nStep 3: Analyzing missing values...")
        ms = _missing_summary(df_filtered)
        print(ms)

    if args.validate_price:
        print("\nValidating Stock Price encoding...")
        validation = validate_stock_price_encoding(df_filtered)
        print({k: v for k, v in validation.items() if not k.endswith('_examples')})

    # Step 4: Export
    print("\nStep 4: Exporting cleaned data...")
    base = args.output or 'cleaned_options'
    # Normalize base (strip existing extensions for consistent dual naming)
    base_root, base_ext = os.path.splitext(base)
    if base_ext.lower() in {'.csv', '.parquet'}:
        base = base_root  # we'll append extension below

    parquet_name = f"{base}.parquet"
    csv_name = f"{base}.csv"

    if args.parquet:
        try:
            import pyarrow  # noqa: F401
            df_filtered.to_parquet(parquet_name, engine='pyarrow')
            output_path = parquet_name
            fmt = 'parquet'
        except Exception as e:
            df_filtered.to_csv(csv_name, index=False)
            output_path = csv_name
            fmt = f"csv (parquet failed: {e})"
    else:
        df_filtered.to_csv(csv_name, index=False)
        output_path = csv_name
        fmt = 'csv'

    print(f"  ✓ Export format: {fmt}")
    print(f"  ✓ Exported to: {output_path}")
    print(f"  ✓ Final dataset: {len(df_filtered):,} rows")

    if args.strike is not None and args.expiry:
        print(f"\nQuerying strike={args.strike}, expiry={args.expiry}...")
        q = query_by_strike_expiry(df_filtered, args.strike, args.expiry)
        print(f"  Found {len(q)} matching rows:")
        print(q.head())

    if args.group_head > 0:
        print(f"\nShowing first {args.group_head} rows per symbol...")
        gh = group_symbol_head(df_filtered, n=args.group_head)
        print(gh.head(args.group_head * 5))
    
    print("\n" + "="*80)
    print("DATA CLEANING COMPLETE")
    print("="*80)

if __name__ == '__main__':  # pragma: no cover
    main()
