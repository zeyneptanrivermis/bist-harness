"""Capability boundaries that distinguish experiment versions A through E."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

from src.experiments.models import ExperimentVersion


TOOL_CATALOG = frozenset(
    {
        "get_universe",
        "get_peer_group",
        "get_stock_history",
        "get_index_history",
        "get_fundamental_metrics",
        "calculate_indicators",
        "find_support_resistance",
        "check_data_quality",
        "test_weekday_pattern",
        "detect_multiday_patterns",
        "detect_reversal_events",
        "compare_reversals",
        "rank_sector_laggards",
        "evaluate_laggard_catchup",
        "align_fundamental_disclosures",
        "evaluate_fundamental_reaction",
        "run_backtest",
        "calculate_performance",
    }
)

REQUIRED_STATE_PREFIX = (
    "START",
    "SELECT_UNIVERSE",
    "LOAD_DATA",
    "CHECK_DATA_QUALITY",
)


@dataclass(frozen=True)
class VersionPolicy:
    version: ExperimentVersion
    allows_tools: bool
    requires_state: bool
    requires_quality_gate: bool
    requires_evidence: bool
    requires_replay: bool
    requires_human_review: bool
    allowed_tools: FrozenSet[str]


def policy_for(version: ExperimentVersion) -> VersionPolicy:
    if version is ExperimentVersion.RAW_LLM:
        return VersionPolicy(version, False, False, False, False, False, False, frozenset())
    if version is ExperimentVersion.TOOLS:
        return VersionPolicy(version, True, False, False, False, False, False, TOOL_CATALOG)
    if version is ExperimentVersion.STATEFUL:
        return VersionPolicy(version, True, True, False, False, False, False, TOOL_CATALOG)
    if version is ExperimentVersion.EVIDENCE:
        return VersionPolicy(version, True, True, True, True, False, False, TOOL_CATALOG)
    return VersionPolicy(version, True, True, True, True, True, True, TOOL_CATALOG)
