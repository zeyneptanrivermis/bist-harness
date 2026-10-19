"""Thin, deterministic MCP wrappers. They never make trading recommendations."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

from src.backtest.engine import calculate_performance, run_backtest
from src.data_pipeline import get_index_history, get_stock_history, load_ticker_data
from src.fundamentals import get_fundamental_metrics
from src.indicators import calculate_all_indicators, find_support_resistance
from src.mcp_tools.models import ToolResponse
from src.mcp_tools.serialization import frame_payload, json_safe
from src.harness.quality import check_data_quality
from src.scenarios.fundamental_reaction import align_financial_disclosure_dates, evaluate_fundamental_reaction
from src.scenarios.sector_laggard import evaluate_laggard_catchup, rank_sector_laggards
from src.scenarios.technical_reversals import compare_single_vs_combined_reversals, detect_reversal_events
from src.scenarios.weekday_patterns import detect_multiday_patterns, test_weekday_pattern
from src.universe import get_peer_group, get_universe_df


def _cut_as_of(frame: pd.DataFrame, as_of_date: Optional[str]) -> pd.DataFrame:
    if as_of_date is None:
        return frame.copy()
    return frame.loc[frame.index <= pd.Timestamp(as_of_date)].copy()


def _local_history(ticker: str, start_date: Optional[str], end_date: Optional[str], allow_download: bool) -> pd.DataFrame:
    """Use cache by default so an MCP call cannot silently change a replay."""
    if allow_download:
        return get_stock_history(ticker, start_date=start_date, end_date=end_date)
    return load_ticker_data(ticker, start_date=start_date, end_date=end_date)


def get_universe_tool() -> ToolResponse:
    return ToolResponse(tool_name="get_universe", data=frame_payload(get_universe_df()))


def get_peer_group_tool(ticker: str) -> ToolResponse:
    return ToolResponse(tool_name="get_peer_group", data={"ticker": ticker, "peers": get_peer_group(ticker)})


def get_stock_history_tool(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    as_of_date: Optional[str] = None,
    allow_download: bool = False,
) -> ToolResponse:
    frame = _local_history(ticker, start_date, end_date, allow_download)
    frame = _cut_as_of(frame, as_of_date)
    return ToolResponse(
        tool_name="get_stock_history",
        as_of_date=as_of_date,
        source="network_or_cache" if allow_download else "local_cache",
        data=frame_payload(frame),
    )


def get_index_history_tool(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    as_of_date: Optional[str] = None,
    allow_download: bool = False,
) -> ToolResponse:
    if allow_download:
        frame = get_index_history(start_date=start_date, end_date=end_date)
    else:
        frame = load_ticker_data("XU100", start_date=start_date, end_date=end_date)
    return ToolResponse(
        tool_name="get_index_history",
        as_of_date=as_of_date,
        source="network_or_cache" if allow_download else "local_cache",
        data=frame_payload(_cut_as_of(frame, as_of_date)),
    )


def get_fundamental_metrics_tool(ticker: str, as_of_date: Optional[str] = None) -> ToolResponse:
    frame = get_fundamental_metrics(ticker)
    if as_of_date:
        frame = frame[frame["kap_disclosure_date"] <= pd.Timestamp(as_of_date)].copy()
    return ToolResponse(
        tool_name="get_fundamental_metrics",
        as_of_date=as_of_date,
        source="curated_local_dataset",
        status="warning",
        message="Curated records require source identifiers before being treated as externally verified KAP evidence.",
        data=frame_payload(frame),
    )


def calculate_indicators_tool(ticker: str, as_of_date: str) -> ToolResponse:
    frame = calculate_all_indicators(_cut_as_of(_local_history(ticker, None, None, False), as_of_date))
    return ToolResponse(tool_name="calculate_indicators", as_of_date=as_of_date, data=frame_payload(frame.tail(100)))


def find_support_resistance_tool(ticker: str, as_of_date: str, window: int = 20) -> ToolResponse:
    frame = _cut_as_of(_local_history(ticker, None, None, False), as_of_date)
    result = find_support_resistance(frame, window=window)
    return ToolResponse(tool_name="find_support_resistance", as_of_date=as_of_date, data=frame_payload(result))


def check_data_quality_tool(ticker: str, as_of_date: str) -> ToolResponse:
    """Expose the mandatory gate as a first-class MCP tool."""
    report = check_data_quality(_local_history(ticker, None, None, False), as_of_date, metadata={"signal_shift": 1})
    status = "ok" if report.status == "PASS" else "warning"
    return ToolResponse(
        tool_name="check_data_quality",
        as_of_date=as_of_date,
        status=status,
        data=report.to_dict(),
    )


def test_weekday_pattern_tool(ticker: str, as_of_date: str) -> ToolResponse:
    result = test_weekday_pattern(_cut_as_of(_local_history(ticker, None, None, False), as_of_date))
    return ToolResponse(tool_name="test_weekday_pattern", as_of_date=as_of_date, data=frame_payload(result))


def detect_multiday_patterns_tool(ticker: str, as_of_date: str, pattern_type: str = "consecutive_down") -> ToolResponse:
    frame = _cut_as_of(_local_history(ticker, None, None, False), as_of_date)
    result = detect_multiday_patterns(frame, pattern_type=pattern_type)
    return ToolResponse(tool_name="detect_multiday_patterns", as_of_date=as_of_date, data=frame_payload(result))


def detect_reversal_events_tool(ticker: str, as_of_date: str) -> ToolResponse:
    price = _cut_as_of(_local_history(ticker, None, None, False), as_of_date)
    market = _cut_as_of(load_ticker_data("XU100"), as_of_date)
    return ToolResponse(tool_name="detect_reversal_events", as_of_date=as_of_date, data=frame_payload(detect_reversal_events(price, market_df=market)))


def compare_reversals_tool(ticker: str, as_of_date: str) -> ToolResponse:
    events = detect_reversal_events(_cut_as_of(_local_history(ticker, None, None, False), as_of_date))
    return ToolResponse(tool_name="compare_single_vs_combined_reversals", as_of_date=as_of_date, data=frame_payload(compare_single_vs_combined_reversals(events)))


def _cached_universe_data(tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
    return {ticker: load_ticker_data(ticker) for ticker in tickers}


def rank_sector_laggards_tool(as_of_date: str) -> ToolResponse:
    tickers = get_universe_df()["ticker"].tolist()
    result = rank_sector_laggards(date=as_of_date, price_dict=_cached_universe_data(tickers))
    return ToolResponse(tool_name="rank_sector_laggards", as_of_date=as_of_date, data=frame_payload(result))


def evaluate_laggard_catchup_tool(as_of_date: str) -> ToolResponse:
    tickers = get_universe_df()["ticker"].tolist()
    cached = {ticker: _cut_as_of(frame, as_of_date) for ticker, frame in _cached_universe_data(tickers).items()}
    return ToolResponse(tool_name="evaluate_laggard_catchup", as_of_date=as_of_date, data=frame_payload(evaluate_laggard_catchup(cached)))


def align_fundamental_disclosures_tool(ticker: str, as_of_date: str) -> ToolResponse:
    price = _cut_as_of(_local_history(ticker, None, None, False), as_of_date)
    market = _cut_as_of(load_ticker_data("XU100"), as_of_date)
    result = align_financial_disclosure_dates(ticker, price_df=price, market_df=market)
    return ToolResponse(tool_name="align_financial_disclosure_dates", as_of_date=as_of_date, data=frame_payload(result))


def evaluate_fundamental_reaction_tool(tickers: list[str]) -> ToolResponse:
    return ToolResponse(tool_name="evaluate_fundamental_reaction", source="curated_local_dataset", data=frame_payload(evaluate_fundamental_reaction(tickers=tickers)))


def run_backtest_tool(ticker: str, as_of_date: str, signals: Optional[list[int]] = None) -> ToolResponse:
    price = _cut_as_of(_local_history(ticker, None, None, False), as_of_date)
    market = _cut_as_of(load_ticker_data("XU100"), as_of_date)
    if signals is None:
        signal_series = (calculate_all_indicators(price)["Supertrend_Direction"] == 1).astype(int)
    else:
        if len(signals) != len(price):
            raise ValueError("signals length must equal the cached price history length after as_of filtering")
        signal_series = pd.Series(signals, index=price.index)
    result = run_backtest(signal_series, price, benchmark_df=market)
    return ToolResponse(tool_name="run_backtest", as_of_date=as_of_date, data={"metrics": json_safe(result.metrics), "trade_count": len(result.trades)})


def calculate_performance_tool(equity_values: list[float], dates: list[str]) -> ToolResponse:
    if len(equity_values) != len(dates):
        raise ValueError("equity_values and dates must have equal length")
    equity = pd.Series(equity_values, index=pd.to_datetime(dates))
    return ToolResponse(tool_name="calculate_performance", data={"metrics": json_safe(calculate_performance(equity))})
