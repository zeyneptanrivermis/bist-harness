"""
Mandatory Research Scenario 1: Sector Relative-Value / Catch-Up Opportunity
CSE-481 Engineering Economics Project

Goal:
Identify stocks that are materially lagging their own sector while the sector is strong,
then test whether selected laggards historically tend to catch up over 5, 10, 20 trading days.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.universe import get_sector_map, get_sectors, get_stocks_in_sector


def calculate_sector_returns(
    price_dict: Dict[str, pd.DataFrame],
    window: int = 20,
    price_col: str = "Adj Close",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes rolling returns for each stock and the median rolling return for each sector.
    Returns:
    - stock_returns_df: columns are tickers, values are rolling N-day returns
    - sector_median_returns_df: columns are sectors, values are median rolling returns
    """
    returns_series = {}
    for ticker, df in price_dict.items():
        if ticker != "XU100" and price_col in df.columns:
            returns_series[ticker] = df[price_col].pct_change(window)

    stock_returns_df = pd.DataFrame(returns_series).dropna(how="all")

    sector_map = get_sector_map()
    sectors = get_sectors()
    sector_medians = {}

    for sector in sectors:
        tickers = get_stocks_in_sector(sector)
        valid_tickers = [t for t in tickers if t in stock_returns_df.columns]
        if valid_tickers:
            sector_medians[sector] = stock_returns_df[valid_tickers].median(axis=1)

    sector_median_returns_df = pd.DataFrame(sector_medians)
    return stock_returns_df, sector_median_returns_df


def calculate_sector_lag_scores(
    price_dict: Dict[str, pd.DataFrame],
    window: int = 20,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    Sector Lag Score = Stock Return - Sector Median Return over N-day window.
    Negative lag score means the stock is underperforming its sector (laggard).
    """
    stock_returns, sector_medians = calculate_sector_returns(price_dict, window, price_col)
    sector_map = get_sector_map()

    lag_scores = pd.DataFrame(index=stock_returns.index)

    for ticker in stock_returns.columns:
        sector = sector_map.get(ticker)
        if sector and sector in sector_medians.columns:
            lag_scores[ticker] = stock_returns[ticker] - sector_medians[sector]

    return lag_scores


def rank_sector_laggards(
    date: Optional[str] = None,
    price_dict: Optional[Dict[str, pd.DataFrame]] = None,
    lookback_window: int = 20,
) -> pd.DataFrame:
    """
    Syllabus Section 32 function: Ranks all stocks inside each sector by their lag score on a given date.
    If date is None, uses the latest available date.
    """
    if price_dict is None:
        from src.data_pipeline import load_ticker_data
        from src.universe import get_all_tickers
        tickers = get_all_tickers()
        price_dict = {t: load_ticker_data(t) for t in tickers}

    stock_returns, sector_medians = calculate_sector_returns(price_dict, lookback_window)
    lag_scores = calculate_sector_lag_scores(price_dict, lookback_window)
    sector_map = get_sector_map()

    target_date = pd.to_datetime(date) if date else lag_scores.index[-1]
    if target_date not in lag_scores.index:
        # Pick the nearest previous date
        valid_dates = lag_scores.index[lag_scores.index <= target_date]
        if len(valid_dates) == 0:
            raise ValueError(f"No available data on or before date {date}")
        target_date = valid_dates[-1]

    records = []
    for ticker in lag_scores.columns:
        sector = sector_map.get(ticker, "Unknown")
        stock_ret = stock_returns.loc[target_date, ticker] if ticker in stock_returns.columns else np.nan
        sec_ret = sector_medians.loc[target_date, sector] if (sector in sector_medians.columns and target_date in sector_medians.index) else np.nan
        lag = lag_scores.loc[target_date, ticker] if target_date in lag_scores.index else np.nan

        records.append({
            "Date": target_date,
            "Ticker": ticker,
            "Sector": sector,
            "Stock_Return": stock_ret,
            "Sector_Median_Return": sec_ret,
            "Lag_Score": lag,
        })

    rank_df = pd.DataFrame(records)
    # Rank within sector (ascending lag score: 1 = most lagging)
    rank_df["Rank_In_Sector"] = rank_df.groupby("Sector")["Lag_Score"].rank(ascending=True, method="min")
    rank_df = rank_df.sort_values(by=["Sector", "Lag_Score"])
    return rank_df.reset_index(drop=True)


def evaluate_laggard_catchup(
    price_dict: Dict[str, pd.DataFrame],
    lookback_window: int = 20,
    forward_horizons: List[int] = [5, 10, 20],
    min_lag_threshold: float = -0.03,            # At least -3% behind sector
    min_sector_return: float = 0.05,             # Sector is strong (+5% or higher)
    signal_shift: int = 1,                       # Strict t+1 execution to eliminate look-ahead bias
) -> pd.DataFrame:
    """
    Backtests the catch-up hypothesis across the universe:
    When a stock is lagging a strong sector (lag_score < min_lag_threshold and sector_return >= min_sector_return),
    does it subsequently outperform its sector peers over forward horizons?

    signal_shift=1 ensures that entry occurs at t+1 (after the day t signal close is observed).
    """
    stock_returns, sector_medians = calculate_sector_returns(price_dict, lookback_window)
    lag_scores = calculate_sector_lag_scores(price_dict, lookback_window)
    sector_map = get_sector_map()

    events = []
    dates = lag_scores.index

    for date in dates:
        for ticker in lag_scores.columns:
            sector = sector_map.get(ticker)
            if not sector or sector not in sector_medians.columns:
                continue

            sec_ret = sector_medians.loc[date, sector]
            lag_score = lag_scores.loc[date, ticker]

            # Condition: Strong sector and substantial lag
            if sec_ret >= min_sector_return and lag_score <= min_lag_threshold:
                event_record = {
                    "Date": date,
                    "Ticker": ticker,
                    "Sector": sector,
                    "Sector_Return_Lookback": sec_ret,
                    "Stock_Return_Lookback": stock_returns.loc[date, ticker],
                    "Lag_Score": lag_score,
                }

                df = price_dict[ticker]
                locs = df.index.get_indexer([date])
                if len(locs) > 0 and locs[0] != -1:
                    idx = locs[0]
                    # Entry index respects signal_shift (t+1)
                    entry_idx = idx + signal_shift
                    if entry_idx < len(df):
                        entry_price = df["Adj Close"].iloc[entry_idx]
                        event_record["Entry_Date"] = df.index[entry_idx]
                        event_record["Entry_Price"] = entry_price

                        for h in forward_horizons:
                            exit_idx = entry_idx + h
                            if exit_idx < len(df):
                                future_stock_ret = (df["Adj Close"].iloc[exit_idx] / entry_price) - 1.0

                                # Forward return of other peers in sector
                                sec_tickers = [t for t in get_stocks_in_sector(sector) if t in price_dict and t != ticker]
                                if not sec_tickers:
                                    sec_tickers = [t for t in get_stocks_in_sector(sector) if t in price_dict]

                                peer_rets = []
                                for pt in sec_tickers:
                                    pdf = price_dict[pt]
                                    plocs = pdf.index.get_indexer([df.index[entry_idx]])
                                    if len(plocs) > 0 and plocs[0] != -1 and plocs[0] + h < len(pdf):
                                        p_entry = pdf["Adj Close"].iloc[plocs[0]]
                                        p_exit = pdf["Adj Close"].iloc[plocs[0] + h]
                                        peer_rets.append((p_exit / p_entry) - 1.0)

                                sec_fwd_median = np.median(peer_rets) if peer_rets else 0.0
                                outperformance = future_stock_ret - sec_fwd_median

                                event_record[f"Fwd_{h}d_Stock_Return"] = future_stock_ret
                                event_record[f"Fwd_{h}d_Sector_Return"] = sec_fwd_median
                                event_record[f"Fwd_{h}d_Outperformance"] = outperformance
                                event_record[f"Fwd_{h}d_CatchUp_Hit"] = 1 if outperformance > 0 else 0
                            else:
                                event_record[f"Fwd_{h}d_Outperformance"] = np.nan
                                event_record[f"Fwd_{h}d_CatchUp_Hit"] = np.nan

                events.append(event_record)

    return pd.DataFrame(events)

