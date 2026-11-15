"""A small, cutoff-safe router for local model tool-call demonstrations."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Callable

from src.mcp_tools.serialization import json_safe
from src.mcp_tools.tools import (
    align_fundamental_disclosures_tool,
    calculate_indicators_tool,
    check_data_quality_tool,
    detect_reversal_events_tool,
    get_stock_history_tool,
    get_fundamental_metrics_tool,
    rank_sector_laggards_tool,
    run_backtest_tool,
    test_weekday_pattern_tool,
)
from src.universe import get_universe_df


class ToolRequestError(ValueError):
    """Raised when an untrusted model response is not an allowed tool request."""


class SafeProjectToolRouter:
    """Execute only named, local, cutoff-aware project wrappers.

    This does not replace HarnessSession, which remains the authoritative
    workflow for completed analyses.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {
            "get_stock_history": get_stock_history_tool,
            "check_data_quality": check_data_quality_tool,
            "calculate_indicators": calculate_indicators_tool,
            "detect_reversal_events": detect_reversal_events_tool,
            "test_weekday_pattern": test_weekday_pattern_tool,
            "rank_sector_laggards": rank_sector_laggards_tool,
            "align_fundamental_disclosures": align_fundamental_disclosures_tool,
            "run_backtest": run_backtest_tool,
            "get_fundamental_metrics": get_fundamental_metrics_tool,
        }
        self._argument_names = {
            "get_stock_history": {"ticker", "as_of_date"},
            "check_data_quality": {"ticker", "as_of_date"},
            "calculate_indicators": {"ticker", "as_of_date"},
            "detect_reversal_events": {"ticker", "as_of_date"},
            "test_weekday_pattern": {"ticker", "as_of_date"},
            "rank_sector_laggards": {"as_of_date"},
            "align_fundamental_disclosures": {"ticker", "as_of_date"},
            "run_backtest": {"ticker", "as_of_date"},
            "get_fundamental_metrics": {"ticker", "as_of_date"},
        }
        self._allowed_tickers = frozenset(get_universe_df()["ticker"].tolist())

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def execute_json(self, response_text: str, *, as_of_date: str) -> dict[str, Any]:
        """Parse a strict JSON call and prevent a model from advancing the cutoff."""
        try:
            request = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ToolRequestError("Model response is not a single JSON tool request.") from exc
        if not isinstance(request, dict) or set(request) != {"tool_name", "arguments"}:
            raise ToolRequestError("Tool request must contain only tool_name and arguments.")
        tool_name = request["tool_name"]
        arguments = request["arguments"]
        if tool_name not in self._tools or not isinstance(arguments, dict):
            raise ToolRequestError("Tool is not approved or arguments are malformed.")
        if set(arguments) - self._argument_names[tool_name]:
            raise ToolRequestError("Tool request included an argument outside the approved schema.")
        ticker = arguments.get("ticker")
        if ticker is not None and ticker not in self._allowed_tickers:
            raise ToolRequestError("Ticker is not present in the fixed project universe.")
        supplied_cutoff = arguments.get("as_of_date", as_of_date)
        if supplied_cutoff != as_of_date or date.fromisoformat(supplied_cutoff) > date.fromisoformat(as_of_date):
            raise ToolRequestError("Tool requests may not change or advance the harness cutoff date.")
        arguments["as_of_date"] = as_of_date
        if tool_name == "rank_sector_laggards":
            arguments = {"as_of_date": as_of_date}
        response = self._tools[tool_name](**arguments)
        return json_safe(response.model_dump())
