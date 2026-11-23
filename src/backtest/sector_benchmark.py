"""Equal-weight sector peer benchmark (spec section 18: compare with a sector benchmark where possible)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.universe import get_sector_map


def benchmark_peers(ticker: str) -> list[str]:
    """Peers sharing the first word of the sector label ("Retail Trade" and "Retail / Apparel" are one retail group).

    shortcut: first-word grouping is a coursework approximation, replace with a real sector index when licensed data exists.
    """
    sectors = get_sector_map()
    group = sectors.get(ticker, "").split(" ")[0]
    return [t for t, sector in sectors.items() if group and sector.split(" ")[0] == group and t != ticker]


def build_sector_benchmark(ticker: str, price_dict: Dict[str, pd.DataFrame], as_of: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Daily-rebalanced equal-weight index of the ticker's sector peers (the ticker itself is excluded).

    Returns a frame with an ``Adj Close`` column, or None when no peer has data, e.g. a sector with a single stock.
    """
    returns = []
    for peer in benchmark_peers(ticker):
        frame = price_dict.get(peer)
        if frame is None or frame.empty:
            continue
        if as_of is not None:
            frame = frame.loc[frame.index <= pd.Timestamp(as_of)]
        returns.append(frame["Adj Close"].pct_change().rename(peer))
    if not returns:
        return None
    mean_ret = pd.concat(returns, axis=1).mean(axis=1).dropna()
    return pd.DataFrame({"Adj Close": (1.0 + mean_ret).cumprod()})


def sector_benchmark_metrics(equity_curve: pd.Series, sector_df: Optional[pd.DataFrame], peer_count: int = 0) -> Dict[str, Any]:
    """Strategy return minus sector-benchmark return over the same dates."""
    if sector_df is None or sector_df.empty:
        return {"Sector_Benchmark": "unavailable (no sector peers with data)"}
    aligned = sector_df["Adj Close"].reindex(equity_curve.index).ffill().dropna()
    if len(aligned) < 2:
        return {"Sector_Benchmark": "unavailable (no overlapping dates)"}
    strategy = float(equity_curve.loc[aligned.index[-1]] / equity_curve.loc[aligned.index[0]] - 1.0)
    sector = float(aligned.iloc[-1] / aligned.iloc[0] - 1.0)
    return {"Sector_Benchmark_Return": sector, "Alpha_vs_Sector": strategy - sector}
