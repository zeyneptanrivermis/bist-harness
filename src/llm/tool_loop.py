"""Cutoff-safe router for local-model tool plans.

The model returns one JSON plan: {"tool_calls": [{"tool_name": ..., "arguments": {...}}, ...]}.
Every call is checked against a fixed schema before anything runs: approved tool name, approved
argument names and types, tickers from the fixed universe, ISO dates that never pass the analysis
date. The analysis date is injected by the harness; the model cannot change or advance it.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Callable

from src.mcp_tools import research as R
from src.universe import get_universe_df

MAX_CALLS = 4
MAX_PEERS = 12

# tool -> (function(as_of, **arguments), allowed argument names, alternatives: the call must contain every argument of at least one set)
_TOOLS: dict[str, tuple[Callable[..., dict[str, Any]], set[str], tuple[set[str], ...]]] = {
    "get_stock_history": (lambda a, **k: R.get_stock_history_summary(k.pop("ticker"), a, **k), {"ticker", "start_date", "end_date"}, ({"ticker"},)),
    "calculate_indicators": (lambda a, **k: R.calculate_indicators_summary(k.pop("ticker"), a, **k), {"ticker", "start_date", "end_date"}, ({"ticker"},)),
    "detect_reversal_events": (lambda a, **k: R.detect_reversal_events_summary(k.pop("ticker"), a, **k), {"ticker", "event", "start_date", "end_date"}, ({"ticker"},)),
    "rank_sector_relative_strength": (lambda a, **k: R.rank_sector_relative_strength_summary(a, **k), {"tickers", "sector", "horizon_days", "start_date", "end_date"}, ({"tickers"}, {"sector"})),
    "test_weekday_pattern": (lambda a, **k: R.test_weekday_pattern_summary(k.pop("ticker"), a, **k), {"ticker", "pattern", "start_date", "end_date", "oos_start_date"}, ({"ticker"},)),
    "get_fundamental_metrics": (lambda a, **k: R.get_fundamental_metrics_summary(k.pop("ticker"), a), {"ticker"}, ({"ticker"},)),
    "calculate_fundamental_change": (lambda a, **k: R.calculate_fundamental_change_summary(k.pop("ticker"), a, **k), {"ticker", "start_date", "end_date"}, ({"ticker"},)),
    "get_macro_context": (lambda a, **k: R.get_macro_context_summary(a, **k), {"ticker", "start_date", "end_date"}, ()),
    "run_backtest": (lambda a, **k: R.run_backtest_summary(k.pop("ticker"), a, **k), {"ticker", "start_date", "end_date"}, ({"ticker"},)),
    "check_data_quality": (lambda a, **k: R.check_data_quality_summary(k.pop("ticker"), a), {"ticker"}, ({"ticker"},)),
}
_DATE_ARGS = ("start_date", "end_date", "oos_start_date")


class ToolRequestError(ValueError):
    """Raised when an untrusted model response is not an allowed tool plan."""


def _iso(value: Any, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ToolRequestError(f"{name} must be an ISO date (YYYY-MM-DD).") from exc


class SafeProjectToolRouter:
    def __init__(self) -> None:
        self._allowed_tickers = frozenset(get_universe_df()["ticker"].tolist())

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return tuple(_TOOLS)

    def validate_call(self, request: Any, *, as_of_date: str) -> tuple[str, dict[str, Any]]:
        """Check one call; returns (tool_name, clean arguments). Runs nothing."""
        cutoff = _iso(as_of_date, "as_of_date")
        if not isinstance(request, dict) or set(request) != {"tool_name", "arguments"}:
            raise ToolRequestError("A tool call must contain only tool_name and arguments.")
        name, args = request["tool_name"], request["arguments"]
        if name not in _TOOLS or not isinstance(args, dict):
            raise ToolRequestError("Tool is not approved or arguments are malformed.")
        _, allowed, required = _TOOLS[name]
        args = dict(args)
        if args.pop("as_of_date", as_of_date) != as_of_date:  # the cutoff is the harness's, not the model's
            raise ToolRequestError("Tool requests may not change or advance the harness cutoff date.")
        if set(args) - allowed:
            raise ToolRequestError("Tool request included an argument outside the approved schema.")
        if required and not any(group <= set(args) for group in required):
            raise ToolRequestError(f"{name} needs one of: {', '.join('+'.join(sorted(g)) for g in required)}.")
        tickers = ([args["ticker"]] if "ticker" in args else []) + (args["tickers"] if isinstance(args.get("tickers"), list) else [])
        if "tickers" in args and (not isinstance(args["tickers"], list) or not 2 <= len(args["tickers"]) <= MAX_PEERS):
            raise ToolRequestError(f"tickers must be a list of 2 to {MAX_PEERS} symbols.")
        if any(t not in self._allowed_tickers for t in tickers):
            raise ToolRequestError("Ticker is not present in the fixed project universe.")
        dates = {k: _iso(args[k], k) for k in _DATE_ARGS if k in args}
        if any(d > cutoff for d in dates.values()):
            raise ToolRequestError("Tool requests may not use dates after the analysis date.")
        if "start_date" in dates and "end_date" in dates and dates["start_date"] > dates["end_date"]:
            raise ToolRequestError("start_date is after end_date.")
        if "event" in args and args["event"] not in R.EVENT_FAMILIES or "pattern" in args and args["pattern"] not in R.WEEKLY_PATTERNS:
            raise ToolRequestError("Unknown event or pattern name.")
        if "horizon_days" in args and not (isinstance(args["horizon_days"], int) and 1 <= args["horizon_days"] <= 252):
            raise ToolRequestError("horizon_days must be an integer between 1 and 252.")
        if "sector" in args and not (isinstance(args["sector"], str) and 0 < len(args["sector"]) <= 40):
            raise ToolRequestError("sector must be a short text.")
        return name, args

    def validate_plan(self, response_text: str, *, as_of_date: str) -> list[tuple[str, dict[str, Any]]]:
        """Parse a strict JSON plan of 1 to MAX_CALLS tool calls."""
        try:
            plan = json.loads(response_text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ToolRequestError("Model response is not a single JSON tool plan.") from exc
        calls = plan.get("tool_calls") if isinstance(plan, dict) and set(plan) == {"tool_calls"} else None
        if not isinstance(calls, list) or not 1 <= len(calls) <= MAX_CALLS:
            raise ToolRequestError(f"A plan must be {{\"tool_calls\": [1 to {MAX_CALLS} calls]}}.")
        return [self.validate_call(call, as_of_date=as_of_date) for call in calls]

    def execute(self, tool_name: str, arguments: dict[str, Any], *, as_of_date: str) -> dict[str, Any]:
        """Run one already validated call. Missing data is returned as available=False, never raised as a crash."""
        try:
            return _TOOLS[tool_name][0](as_of_date, **dict(arguments))
        except FileNotFoundError as exc:
            return R._unavailable(tool_name, as_of_date, f"Local data is missing: {exc}")
