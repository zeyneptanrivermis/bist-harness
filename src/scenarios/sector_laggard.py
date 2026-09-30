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


def evaluate_laggard_catchup(
    price_dict: Dict[str, pd.DataFrame],
    lookback_window: int = 20,
    forward_horizons: List[int] = [5, 10, 20],
    laggard_percentile_threshold: float = 0.25, # bottom quartile
    min_sector_return: float = 0.05,            # Sector is "strong" (at least +5% over lookback)
) -> pd.DataFrame:
    """
    Backtests the catch-up hypothesis across the universe:
    When a stock is lagging a strong sector (lag_score < 0 and sector_return > min_sector_return),
    does it subsequently outperform its sector median over forward horizons?
    """
    stock_returns, sector_medians = calculate_sector_returns(price_dict, lookback_window)
    lag_scores = calculate_sector_lag_scores(price_dict, lookback_window)
    sector_map = get_sector_map()

    events = []

    dates = lag_scores.index
    for i, date in enumerate(dates):
        for ticker in lag_scores.columns:
            sector = sector_map.get(ticker)
            if not sector or sector not in sector_medians.columns:
                continue

            sec_ret = sector_medians.loc[date, sector]
            lag_score = lag_scores.loc[date, ticker]

            # Condition: Sector is strong and stock is a laggard
            if sec_ret >= min_sector_return and lag_score < -0.03: # at least 3% behind
                event_record = {
                    "Date": date,
                    "Ticker": ticker,
                    "Sector": sector,
                    "Sector_Return_Lookback": sec_ret,
                    "Stock_Return_Lookback": stock_returns.loc[date, ticker],
                    "Lag_Score": lag_score,
                }

                # Forward outperformance
                df = price_dict[ticker]
                locs = df.index.get_indexer([date])
                if len(locs) > 0 and locs[0] != -1:
                    idx = locs[0]
                    for h in forward_horizons:
                        if idx + h < len(df):
                            future_stock_ret = (df["Adj Close"].iloc[idx + h] / df["Adj Close"].iloc[idx]) - 1.0
                            # Sector forward return
                            sec_tickers = [t for t in get_stocks_in_sector(sector) if t in price_dict]
                            peer_rets = []
                            for pt in sec_tickers:
                                pdf = price_dict[pt]
                                plocs = pdf.index.get_indexer([date])
                                if len(plocs) > 0 and plocs[0] != -1 and plocs[0] + h < len(pdf):
                                    peer_rets.append((pdf["Adj Close"].iloc[plocs[0] + h] / pdf["Adj Close"].iloc[plocs[0]]) - 1.0)
                            
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
