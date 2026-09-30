"""
Data Pipeline Module
CSE-481 Engineering Economics Project
Handles market data ingestion from Yahoo Finance, data validation, and caching.
"""

from typing import Dict, List, Optional
import os
from pathlib import Path
import pandas as pd
import yfinance as yf

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    WARMUP_START_DATE,
    ANALYSIS_END_DATE,
    MAX_ALLOWED_MISSING_RATIO,
)
from src.universe import (
    BIST_30_UNIVERSE,
    BENCHMARK_SYMBOL,
    get_all_tickers,
    get_ticker_to_yahoo_map,
)


def fetch_single_ticker(
    yahoo_symbol: str,
    start_date: str = WARMUP_START_DATE,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Downloads historical daily OHLCV data from Yahoo Finance for a given symbol.
    """
    print(f"Downloading {yahoo_symbol} from {start_date} to {end_date or 'today'}...")
    df = yf.download(
        tickers=yahoo_symbol,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise ValueError(f"No data returned from Yahoo Finance for {yahoo_symbol}")
    
    # Flatten MultiIndex columns if present (e.g. from newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()
    return df


def save_ticker_data(ticker: str, df: pd.DataFrame, is_raw: bool = True) -> Path:
    """Saves DataFrame as both parquet (fast) and csv (inspectable)."""
    target_dir = RAW_DATA_DIR if is_raw else PROCESSED_DATA_DIR
    parquet_path = target_dir / f"{ticker}.parquet"
    csv_path = target_dir / f"{ticker}.csv"
    
    df.to_parquet(parquet_path)
    df.to_csv(csv_path)
    return parquet_path


def load_ticker_data(
    ticker: str,
    is_raw: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Loads ticker data from local cache (parquet). Falls back to CSV if parquet missing."""
    target_dir = RAW_DATA_DIR if is_raw else PROCESSED_DATA_DIR
    parquet_path = target_dir / f"{ticker}.parquet"
    csv_path = target_dir / f"{ticker}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    else:
        raise FileNotFoundError(f"No cached data found for ticker {ticker} in {target_dir}")

    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    return df


def clean_and_preprocess_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans OHLCV data:
    - Verifies required columns (Open, High, Low, Close, Adj Close, Volume)
    - Fills forward single missing business day prices
    - Calculates daily returns, log returns, and trading volume in TRY (Turnover)
    """
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    cleaned = df.copy()
    
    # Handle Adj Close fallback
    if "Adj Close" not in cleaned.columns:
        cleaned["Adj Close"] = cleaned["Close"]

    # Sort and forward-fill weekend/small gaps
    cleaned = cleaned.sort_index()
    cleaned = cleaned.ffill()

    # Feature additions
    cleaned["Daily_Return"] = cleaned["Adj Close"].pct_change()
    cleaned["Turnover"] = cleaned["Close"] * cleaned["Volume"]
    
    return cleaned


def download_entire_universe(
    start_date: str = WARMUP_START_DATE,
    end_date: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Downloads and caches data for all 30 BIST stocks plus the XU100 benchmark.
    """
    symbol_map = get_ticker_to_yahoo_map()
    all_data = {}

    # 1. Download benchmark first
    bench_file = RAW_DATA_DIR / "XU100.parquet"
    if not bench_file.exists() or force_refresh:
        bench_df = fetch_single_ticker(BENCHMARK_SYMBOL, start_date, end_date)
        save_ticker_data("XU100", bench_df, is_raw=True)
        cleaned_bench = clean_and_preprocess_ohlcv(bench_df)
        save_ticker_data("XU100", cleaned_bench, is_raw=False)
        all_data["XU100"] = cleaned_bench
    else:
        all_data["XU100"] = load_ticker_data("XU100", is_raw=False)

    # 2. Download 30 stocks
    for ticker, yahoo_symbol in symbol_map.items():
        raw_file = RAW_DATA_DIR / f"{ticker}.parquet"
        if not raw_file.exists() or force_refresh:
            try:
                raw_df = fetch_single_ticker(yahoo_symbol, start_date, end_date)
                save_ticker_data(ticker, raw_df, is_raw=True)
                cleaned_df = clean_and_preprocess_ohlcv(raw_df)
                save_ticker_data(ticker, cleaned_df, is_raw=False)
                all_data[ticker] = cleaned_df
            except Exception as e:
                print(f"Error fetching {ticker} ({yahoo_symbol}): {e}")
        else:
            all_data[ticker] = load_ticker_data(ticker, is_raw=False)

    print(f"Successfully processed {len(all_data)} series (30 stocks + Benchmark).")
    return all_data


if __name__ == "__main__":
    download_entire_universe()
