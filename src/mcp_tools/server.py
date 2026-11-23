"""FastMCP server factory kept separate from the deterministic tool implementations."""

from __future__ import annotations

from typing import Callable

from src.mcp_tools import tools


def build_mcp_server():
    """Build the server lazily so importing core analysis never depends on FastMCP.

    FastMCP is an optional runtime boundary: tests call the wrappers directly, while
    an MCP client calls this factory and receives the same bounded contracts.
    """
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("FastMCP is required to run the MCP server. Install requirements.txt first.") from exc

    mcp = FastMCP("BIST Financial Analytics Harness")
    exposed: list[Callable] = [
        tools.get_universe_tool,
        tools.get_peer_group_tool,
        tools.get_stock_history_tool,
        tools.get_index_history_tool,
        tools.get_fundamental_metrics_tool,
        tools.calculate_indicators_tool,
        tools.find_support_resistance_tool,
        tools.check_data_quality_tool,
        tools.test_weekday_pattern_tool,
        tools.detect_multiday_patterns_tool,
        tools.detect_reversal_events_tool,
        tools.compare_reversals_tool,
        tools.rank_sector_laggards_tool,
        tools.evaluate_laggard_catchup_tool,
        tools.align_fundamental_disclosures_tool,
        tools.evaluate_fundamental_reaction_tool,
        tools.run_backtest_tool,
        tools.calculate_performance_tool,
        tools.get_news_events_tool,
        tools.compare_strategy_variants_tool,
    ]
    for function in exposed:
        mcp.tool(name=function.__name__.removesuffix("_tool"))(function)
    return mcp


if __name__ == "__main__":  # pragma: no cover - manual server entry point
    build_mcp_server().run()
