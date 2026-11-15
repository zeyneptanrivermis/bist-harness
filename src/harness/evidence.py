"""Build a small immutable evidence set from verified deterministic outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

from src.config import ALLOWED_ACTIONS


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    claim: str
    source_tool: str
    as_of_date: str
    values: Dict[str, Any]


def build_evidence(
    *,
    as_of_date: str,
    quality_report: Dict[str, Any],
    scenario: str,
    scenario_summary: Dict[str, Any],
    backtest_metrics: Dict[str, Any],
) -> List[EvidenceItem]:
    """Only committed, tool-produced facts are admitted to the evidence bundle."""
    evidence = [
        EvidenceItem(
            "E1",
            "Data quality gate completed before calculation.",
            "check_data_quality",
            as_of_date,
            {"status": quality_report["status"], "fingerprint": quality_report.get("data_fingerprint")},
        ),
        EvidenceItem(
            "E2",
            f"{scenario} was calculated using data available at the cutoff.",
            "scenario_engine",
            as_of_date,
            scenario_summary,
        ),
    ]
    if backtest_metrics:
        evidence.append(EvidenceItem("E3", "A t+1, cost-aware backtest was calculated.", "run_backtest", as_of_date, backtest_metrics))
    return evidence


def select_action(scenario: str, scenario_summary: Dict[str, Any]) -> str:
    """Map evidence to the finite educational action space; never emit buy/sell."""
    if scenario == "technical_reversal" and scenario_summary.get("event_count", 0) > 0:
        action = "TECHNICAL_REVERSAL_CANDIDATE"
    elif scenario == "fundamental_reaction" and scenario_summary.get("strong_improvement_count", 0) > 0:
        action = "FUNDAMENTAL_IMPROVEMENT_CANDIDATE"
    elif scenario == "sector_laggard" and scenario_summary.get("rank_in_sector") == 1 and scenario_summary.get("Lag_Score", 0) < 0:  # single-stock sectors tie at 0
        action = "POTENTIAL_CATCH_UP_CANDIDATE"
    elif scenario == "weekday_pattern":
        action = "INVESTIGATE"
    else:
        action = "WATCH"
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Action {action} is outside the permitted educational action space.")
    return action


def evidence_to_dict(evidence: Iterable[EvidenceItem]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in evidence]
