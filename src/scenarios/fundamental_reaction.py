"""
Mandatory Research Scenario 4: Quarterly Fundamentals and Price Reaction (PEAD)
CSE-481 Engineering Economics Project

Goal:
Connect point-in-time quarterly fundamental changes (Revenue, EBITDA, Net Income, Margins, Leverage)
with official KAP disclosure timestamps. Measure post-disclosure price reaction and Cumulative
Abnormal Return (CAR) relative to the BIST 100 benchmark (XU100) across 1-day, 5-day, and 20-day horizons.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.fundamentals import get_fundamental_metrics
from src.data_pipeline import get_stock_history, get_index_history
from src.config import DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE_RATE


def calculate_quarterly_changes(fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Syllabus Section 32 function:
    Calculates Quarter-over-Quarter (QoQ) and Year-over-Year (YoY) changes
    for financial metrics.
    """
    df = fund_df.copy().sort_values(by="kap_disclosure_date").reset_index(drop=True)

    # QoQ growth
    df["Revenue_Growth_QoQ"] = df["revenue"].pct_change(1)
    df["EBITDA_Growth_QoQ"] = df["ebitda"].pct_change(1)
    df["Net_Income_Growth_QoQ"] = df["net_income"].pct_change(1)
    df["EBITDA_Margin_Change_QoQ"] = df["ebitda_margin"].diff(1)
    df["ROE_Change_QoQ"] = df["roe"].diff(1)

    # YoY growth (4 quarters back)
    df["Revenue_Growth_YoY"] = df["revenue"].pct_change(4)
    df["EBITDA_Growth_YoY"] = df["ebitda"].pct_change(4)
    df["Net_Income_Growth_YoY"] = df["net_income"].pct_change(4)
    df["EBITDA_Margin_Change_YoY"] = df["ebitda_margin"].diff(4)

    # Fundamental classification
    # Strong improvement: Both Revenue and EBITDA growing > 10% QoQ
    conditions = [
        (df["EBITDA_Growth_QoQ"] >= 0.10) & (df["Revenue_Growth_QoQ"] >= 0.05),
        (df["EBITDA_Growth_QoQ"] < 0.0) | (df["Net_Income_Growth_QoQ"] < 0.0),
    ]
    choices = ["STRONG_IMPROVEMENT", "DETERIORATING"]
    df["Fundamental_Regime"] = np.select(conditions, choices, default="STABLE_GROWTH")

    return df


def align_financial_disclosure_dates(
    ticker: str,
    price_df: Optional[pd.DataFrame] = None,
    market_df: Optional[pd.DataFrame] = None,
    forward_horizons: List[int] = [1, 5, 20],
    signal_shift: int = 1,  # Strict t+1 execution (disclosed after close -> traded next day)
) -> pd.DataFrame:
    """
    Syllabus Section 32 function:
    Aligns each quarterly financial disclosure with market prices starting from
    the next trading session (t+1) to strictly eliminate look-ahead bias.
    Computes Cumulative Abnormal Returns (CAR) versus XU100.
    """
    fund_df = get_fundamental_metrics(ticker)
    fund_metrics = calculate_quarterly_changes(fund_df)

    if price_df is None:
        price_df = get_stock_history(ticker)
    if market_df is None:
        market_df = get_index_history()

    price_df = price_df.sort_index()
    market_df = market_df.sort_index()

    events = []
    roundtrip_cost = 2.0 * (DEFAULT_COMMISSION_RATE + DEFAULT_SLIPPAGE_RATE)

    for _, row in fund_metrics.iterrows():
        kap_date = pd.to_datetime(row["kap_disclosure_date"])
        period = row["period"]

        # Find the market day corresponding to disclosure or immediately following
        valid_trading_dates = price_df.index[price_df.index >= kap_date]
        if len(valid_trading_dates) == 0:
            continue

        announcement_date = valid_trading_dates[0]
        announcement_idx = price_df.index.get_loc(announcement_date)

        # Apply signal_shift (t+1 entry to ensure public availability before trade)
        entry_idx = announcement_idx + signal_shift
        if entry_idx >= len(price_df):
            continue

        entry_date = price_df.index[entry_idx]
        entry_price = price_df["Adj Close"].iloc[entry_idx]

        # Market entry price
        m_entry_locs = market_df.index.get_indexer([entry_date])
        if len(m_entry_locs) == 0 or m_entry_locs[0] == -1:
            continue
        m_entry_idx = m_entry_locs[0]
        m_entry_price = market_df["Adj Close"].iloc[m_entry_idx]

        record: Dict[str, Any] = {
            "Ticker": ticker,
            "Period": period,
            "KAP_Disclosure_Date": kap_date.strftime("%Y-%m-%d"),
            "Market_Announcement_Date": announcement_date.strftime("%Y-%m-%d"),
            "Entry_Date": entry_date.strftime("%Y-%m-%d"),
            "Entry_Price": entry_price,
            "Revenue": row["revenue"],
            "EBITDA": row["ebitda"],
            "Net_Income": row["net_income"],
            "EBITDA_Margin": row["ebitda_margin"],
            "ROE": row["roe"],
            "Revenue_Growth_QoQ": row["Revenue_Growth_QoQ"],
            "EBITDA_Growth_QoQ": row["EBITDA_Growth_QoQ"],
            "Net_Income_Growth_QoQ": row["Net_Income_Growth_QoQ"],
            "Fundamental_Regime": row["Fundamental_Regime"],
        }

        # Measure forward post-earnings returns and abnormal returns (CAR)
        for h in forward_horizons:
            exit_idx = entry_idx + h
            m_exit_idx = m_entry_idx + h

            if exit_idx < len(price_df) and m_exit_idx < len(market_df):
                exit_price = price_df["Adj Close"].iloc[exit_idx]
                stock_ret = (exit_price / entry_price) - 1.0

                m_exit_price = market_df["Adj Close"].iloc[m_exit_idx]
                bench_ret = (m_exit_price / m_entry_price) - 1.0

                # Cumulative Abnormal Return (CAR) = Stock Return - Benchmark Return
                car = stock_ret - bench_ret
                net_stock_ret = stock_ret - roundtrip_cost

                record[f"Fwd_{h}d_Stock_Return"] = stock_ret
                record[f"Fwd_{h}d_Benchmark_Return"] = bench_ret
                record[f"Fwd_{h}d_CAR"] = car
                record[f"Fwd_{h}d_Outperformed_Benchmark"] = int(car > 0)
            else:
                record[f"Fwd_{h}d_Stock_Return"] = np.nan
                record[f"Fwd_{h}d_Benchmark_Return"] = np.nan
                record[f"Fwd_{h}d_CAR"] = np.nan
                record[f"Fwd_{h}d_Outperformed_Benchmark"] = np.nan

        events.append(record)

    return pd.DataFrame(events)


def evaluate_fundamental_reaction(
    tickers: List[str] = ["TUPRS", "THYAO", "ASELS", "BIMAS", "FROTO"],
    horizons: List[int] = [1, 5, 20],
) -> pd.DataFrame:
    """
    Syllabus Section 32 function:
    Evaluates Post-Earnings Announcement Drift (PEAD) across multiple tickers.
    """
    all_events = []
    for ticker in tickers:
        try:
            ev_df = align_financial_disclosure_dates(ticker=ticker, forward_horizons=horizons)
            all_events.append(ev_df)
        except Exception as e:
            print(f"Skipping {ticker}: {e}")

    if not all_events:
        return pd.DataFrame()

    combined = pd.concat(all_events, ignore_index=True)
    return combined
