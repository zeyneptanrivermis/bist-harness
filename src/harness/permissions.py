"""Which deterministic tools each harness state may call (spec section 23: a state exposes only a subset)."""

from __future__ import annotations

STATE_TOOLS: dict[str, frozenset[str]] = {
    "CHECK_DATA_QUALITY": frozenset({"check_data_quality"}),
    # Scenario tools chosen by the validated plan. run_backtest may appear in a plan, so it is allowed here too.
    "RUN_SCENARIO": frozenset({
        "get_stock_history", "get_index_history", "calculate_indicators", "find_support_resistance",
        "detect_reversal_events", "rank_sector_relative_strength", "test_weekday_pattern",
        "get_fundamental_metrics", "calculate_fundamental_change", "get_macro_context", "run_backtest",
    }),
    "LOAD_FUNDAMENTALS": frozenset({"get_fundamental_metrics", "calculate_fundamental_change"}),
    "LOAD_MACRO_CONTEXT": frozenset({"get_macro_context"}),
    "LOAD_CONTEXT": frozenset({"get_news_events"}),
    "PREVIEW_STRATEGIES": frozenset({"compare_strategy_variants"}),
    "RUN_BACKTEST": frozenset({"run_backtest"}),
}


def assert_tool_allowed(state: str, tool_name: str) -> None:
    """Raise PermissionError when a tool is called from a state that does not expose it."""
    if tool_name not in STATE_TOOLS.get(state, frozenset()):
        raise PermissionError(f"Tool '{tool_name}' is not permitted in state {state}.")
