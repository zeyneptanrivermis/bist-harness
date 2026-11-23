"""Previewable strategy space (proposal section 29): variants A-E judged by the same backtest.

A technical only        Supertrend up and MACD histogram > 0
B + sector relative     A and the stock is not a deep laggard against its peer median
C + fundamentals        B and the latest disclosed quarter did not worsen margin or leverage
D + macro               C and USD/TRY has not jumped over the last 20 trading days
E + news                D and no negative news item for the ticker in the last few trading days

Every signal at date t uses only data published by t; ``run_backtest`` then shifts it one day (t+1).
A layer with no data for the ticker is a pass-through and is named in ``flags`` so a variant is never
presented as "fundamental" when no fundamentals existed.
These are educational backtest labels, not trading advice.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.context.news import get_news_events
from src.data_pipeline import load_ticker_data
from src.fundamentals import get_fundamental_metrics
from src.indicators import calculate_all_indicators
from src.macro import load_macro_series
from src.mcp_tools.serialization import json_safe
from src.universe import get_peer_group

VARIANTS = ("A", "B", "C", "D", "E")
LAG_LIMIT = 0.05      # B: blocked when 20-day return trails the peer median by more than 5 points
FX_STRESS = 0.08      # D: blocked when USD/TRY rose more than 8% in 20 trading days
NEWS_WINDOW = 5       # E: trading days a negative news item keeps the stock blocked
MIN_TRADES = 10       # below this a result is anecdotal and flagged
METRICS = ("Total_Return", "Annualized_Return_CAGR", "Sharpe_Ratio", "Max_Drawdown", "Win_Rate", "Total_Trades", "Alpha_vs_Benchmark")


def _sector_ok(close: pd.Series, ticker: str, peers: Optional[Dict[str, pd.DataFrame]], flags: list) -> pd.Series:
    peers = {t: f["Adj Close"] for t, f in (peers or {}).items() if t != ticker}
    if not peers:
        flags.append("no_peer_group")
        return pd.Series(True, index=close.index)
    peer_median = pd.DataFrame(peers).reindex(close.index).ffill().pct_change(20).median(axis=1)
    return (close.pct_change(20) - peer_median) >= -LAG_LIMIT  # NaN (warm-up) compares False


def _fundamental_ok(index: pd.DatetimeIndex, fundamentals: Optional[pd.DataFrame], flags: list) -> pd.Series:
    if fundamentals is None or len(fundamentals) < 2:
        flags.append("no_fundamentals")
        return pd.Series(True, index=index)
    f = fundamentals.sort_values("kap_disclosure_date")
    leverage = f["net_debt"] / f["ebitda"]
    improving = ((f["ebitda_margin"] >= f["ebitda_margin"].shift()) & (leverage <= leverage.shift())).iloc[1:]
    by_disclosure = pd.Series(improving.values, index=pd.DatetimeIndex(f["kap_disclosure_date"].iloc[1:]))
    known = by_disclosure.reindex(index.union(by_disclosure.index)).ffill().reindex(index)  # last disclosed quarter by day t
    if known.isna().all():
        flags.append("no_fundamentals")
    return known.fillna(True).astype(bool)  # before the first disclosure: nothing to judge


def _macro_ok(index: pd.DatetimeIndex, fx: Optional[pd.Series], flags: list) -> pd.Series:
    if fx is None or fx.empty:
        flags.append("no_macro")
        return pd.Series(True, index=index)
    level = fx.reindex(fx.index.union(index)).ffill().reindex(index)  # fx is indexed by publication date
    return (level.pct_change(20) <= FX_STRESS) | level.pct_change(20).isna()


def _news_ok(index: pd.DatetimeIndex, news: Optional[pd.DataFrame], ticker: str, flags: list) -> pd.Series:
    own = news[news["ticker"] == ticker] if news is not None and not news.empty else pd.DataFrame()
    if own.empty:
        flags.append("no_news_records")
        return pd.Series(True, index=index)
    hit = pd.Series(0, index=index)
    for day in own.loc[own["stance"] == "negative", "effective_date"]:
        pos = index.searchsorted(day)  # first trading day on or after the effective date
        if pos < len(index):
            hit.iloc[pos] = 1
    return hit.rolling(NEWS_WINDOW, min_periods=1).max() == 0


def generate_strategy_variants(ticker: str, price: pd.DataFrame, peers: Optional[Dict[str, pd.DataFrame]] = None,
                               fundamentals: Optional[pd.DataFrame] = None, fx: Optional[pd.Series] = None,
                               news: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, list]:
    """Return ``(signals, flags)``: columns A-E of 0/1 flags indexed like ``price``, nested by construction."""
    flags: list = []
    enriched = calculate_all_indicators(price)
    a = (enriched["Supertrend_Direction"] == 1) & (enriched["MACD_Hist"] > 0)
    b = a & _sector_ok(price["Adj Close"], ticker, peers, flags)
    c = b & _fundamental_ok(price.index, fundamentals, flags)
    d = c & _macro_ok(price.index, fx, flags)
    e = d & _news_ok(price.index, news, ticker, flags)
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d, "E": e}).fillna(False).astype(int), flags


def _inputs(ticker: str, as_of: str) -> Dict[str, Any]:
    cut = pd.Timestamp(as_of)
    price, market = (load_ticker_data(t).loc[lambda f: f.index <= cut] for t in (ticker, "XU100"))
    peers = {t: load_ticker_data(t).loc[lambda f: f.index <= cut] for t in get_peer_group(ticker)}
    try:
        fundamentals = get_fundamental_metrics(ticker)
        fundamentals = fundamentals[fundamentals["kap_disclosure_date"] <= cut]
    except ValueError:
        fundamentals = None
    fx = load_macro_series("USDTRY")
    return {"price": price, "market": market, "peers": peers, "fundamentals": fundamentals,
            "fx": fx["value"].loc[:cut] if fx is not None else None, "news": get_news_events(as_of=as_of)}


def _score(signal: pd.Series, price: pd.DataFrame, market: pd.DataFrame) -> Dict[str, Any]:
    result = run_backtest(signal, price, benchmark_df=market)
    row = {k: result.metrics.get(k) for k in METRICS}
    row["Buy_and_Hold_Return"] = float(result.buy_and_hold_equity.iloc[-1] / result.buy_and_hold_equity.iloc[0] - 1)
    row["Signal_Days"] = int(signal.sum())
    if row["Total_Trades"] < MIN_TRADES:
        row["warning"] = f"only {row['Total_Trades']} trades: any return is anecdotal"
    return row


def compare_strategy_variants(ticker: str, as_of: str, oos_start: Optional[str] = None, **inputs: Any) -> Dict[str, Any]:
    """Backtest A-E with identical costs, t+1 timing and benchmarks; optionally report an out-of-sample slice.

    ``inputs`` may override ``price, market, peers, fundamentals, fx, news`` (used by tests); otherwise the
    local cache is read and cut at ``as_of``.
    """
    data = {"peers": None, "fundamentals": None, "fx": None, "news": None} if {"price", "market"} <= inputs.keys() else _inputs(ticker, as_of)
    data.update(inputs)
    price, market = data["price"], data["market"]
    signals, flags = generate_strategy_variants(ticker, price, data["peers"], data["fundamentals"], data["fx"], data["news"])
    out: Dict[str, Any] = {"ticker": ticker, "as_of": as_of, "flags": sorted(set(flags)), "variants": {}}
    for name in VARIANTS:
        out["variants"][name] = _score(signals[name], price, market)
    if oos_start:
        cut = pd.Timestamp(oos_start)
        out["out_of_sample_from"] = oos_start
        for name in VARIANTS:
            out["variants"][name]["out_of_sample"] = _score(signals[name].loc[cut:], price.loc[cut:], market)
    out["note"] = "Educational backtest labels. Prediction and backtest performance are not future guarantees."
    return json_safe(out)
