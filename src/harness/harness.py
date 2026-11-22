"""A guarded, stateful workflow that chains only verified deterministic tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd

from src.backtest.engine import run_backtest
from src.context.news import get_news_events
from src.harness.evidence import build_evidence, evidence_to_dict, select_action
from src.harness.quality import QualityReport, check_data_quality
from src.harness.replay import log_decision
from src.mcp_tools.models import AnalysisRequest, HumanReview
from src.mcp_tools.serialization import json_safe
from src.scenarios.fundamental_reaction import align_financial_disclosure_dates
from src.scenarios.sector_laggard import rank_sector_laggards
from src.scenarios.technical_reversals import detect_reversal_events
from src.scenarios.weekday_patterns import test_weekday_pattern
from src.indicators import calculate_all_indicators


class HarnessState(str, Enum):
    START = "START"
    SELECT_UNIVERSE = "SELECT_UNIVERSE"
    LOAD_DATA = "LOAD_DATA"
    CHECK_DATA_QUALITY = "CHECK_DATA_QUALITY"
    CALCULATE = "CALCULATE"
    RUN_SCENARIO = "RUN_SCENARIO"
    LOAD_CONTEXT = "LOAD_CONTEXT"  # optional: news context
    RUN_BACKTEST = "RUN_BACKTEST"
    BUILD_EVIDENCE = "BUILD_EVIDENCE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    LOG_AND_CLOSE = "LOG_AND_CLOSE"
    CLOSED = "CLOSED"
    BLOCKED = "BLOCKED"


_ALLOWED_TRANSITIONS = {
    HarnessState.START: {HarnessState.SELECT_UNIVERSE},
    HarnessState.SELECT_UNIVERSE: {HarnessState.LOAD_DATA},
    HarnessState.LOAD_DATA: {HarnessState.CHECK_DATA_QUALITY},
    HarnessState.CHECK_DATA_QUALITY: {HarnessState.CALCULATE, HarnessState.BLOCKED},
    HarnessState.CALCULATE: {HarnessState.RUN_SCENARIO},
    HarnessState.RUN_SCENARIO: {HarnessState.LOAD_CONTEXT, HarnessState.RUN_BACKTEST},
    HarnessState.LOAD_CONTEXT: {HarnessState.RUN_BACKTEST},
    HarnessState.RUN_BACKTEST: {HarnessState.BUILD_EVIDENCE},
    HarnessState.BUILD_EVIDENCE: {HarnessState.HUMAN_REVIEW, HarnessState.LOG_AND_CLOSE},
    HarnessState.HUMAN_REVIEW: {HarnessState.LOG_AND_CLOSE},
    HarnessState.LOG_AND_CLOSE: {HarnessState.CLOSED},
    HarnessState.BLOCKED: set(),
    HarnessState.CLOSED: set(),
}


@dataclass
class HarnessSession:
    request: AnalysisRequest
    state: HarnessState = HarnessState.START
    analysis_id: str = field(default_factory=lambda: uuid4().hex)
    transitions: List[str] = field(default_factory=lambda: [HarnessState.START.value])
    price_data: Optional[pd.DataFrame] = None
    benchmark_data: Optional[pd.DataFrame] = None
    quality: Optional[QualityReport] = None
    enriched_data: Optional[pd.DataFrame] = None
    scenario_summary: Dict[str, Any] = field(default_factory=dict)
    backtest_metrics: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    action: Optional[str] = None
    human_review: Optional[HumanReview] = None

    def _transition(self, target: HarnessState) -> None:
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(f"Invalid harness transition: {self.state.value} -> {target.value}")
        self.state = target
        self.transitions.append(target.value)

    def select_universe(self) -> None:
        self._transition(HarnessState.SELECT_UNIVERSE)

    def load_data(self, price_data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None) -> None:
        self._transition(HarnessState.LOAD_DATA)
        self.price_data = price_data.copy()
        self.benchmark_data = benchmark_data.copy() if benchmark_data is not None else None

    def check_quality(self) -> QualityReport:
        if self.price_data is None:
            raise RuntimeError("Price data must be loaded before quality checks.")
        self._transition(HarnessState.CHECK_DATA_QUALITY)
        metadata = {"signal_shift": 1, **self.request.metadata}
        if self.request.scenario == "fundamental_reaction":
            metadata.setdefault("source", "curated_local_dataset")
        self.quality = check_data_quality(self.price_data, self.request.as_of_date, metadata=metadata)
        self.price_data = self.quality.filtered_data
        if self.benchmark_data is not None:
            self.benchmark_data = self.benchmark_data.loc[self.benchmark_data.index <= pd.Timestamp(self.request.as_of_date)].copy()
        if self.quality.status in {"ANALYSIS_UNSAFE", "HARD_BLOCK"}:
            self._transition(HarnessState.BLOCKED)
        return self.quality

    def calculate(self) -> None:
        if self.state != HarnessState.CHECK_DATA_QUALITY or self.quality is None or self.quality.status not in {"PASS", "PASS_WITH_WARNINGS", "INVESTIGATE"}:
            raise RuntimeError("Calculation is allowed only after a non-blocking quality result.")
        self._transition(HarnessState.CALCULATE)
        self.enriched_data = calculate_all_indicators(self.price_data)

    def run_scenario(self, universe_data: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        if self.enriched_data is None:
            raise RuntimeError("Indicators must be calculated before scenario execution.")
        self._transition(HarnessState.RUN_SCENARIO)
        scenario = self.request.scenario
        if scenario == "technical_reversal":
            events = detect_reversal_events(self.enriched_data, market_df=self.benchmark_data)
            self.scenario_summary = {"event_count": int(len(events)), "latest_events": json_safe(events.tail(3))}
        elif scenario == "weekday_pattern":
            result = test_weekday_pattern(self.price_data)
            significant = int(result["Is_Statistically_Significant"].sum()) if not result.empty else 0
            self.scenario_summary = {"weekday_rows": int(len(result)), "significant_patterns": significant, "results": json_safe(result)}
        elif scenario == "fundamental_reaction":
            if self.benchmark_data is None:
                raise RuntimeError("Fundamental reaction requires benchmark data.")
            events = align_financial_disclosure_dates(self.request.ticker, price_df=self.price_data, market_df=self.benchmark_data)
            strong = int((events.get("Fundamental_Regime", pd.Series(dtype=str)) == "STRONG_IMPROVEMENT").sum())
            self.scenario_summary = {"event_count": int(len(events)), "strong_improvement_count": strong, "latest_events": json_safe(events.tail(3))}
        else:
            if universe_data is None:
                raise RuntimeError("Sector-laggard analysis requires cached data for the selected universe.")
            trimmed = {ticker: frame.loc[frame.index <= pd.Timestamp(self.request.as_of_date)].copy() for ticker, frame in universe_data.items()}
            ranks = rank_sector_laggards(date=self.request.as_of_date, price_dict=trimmed)
            row = ranks[ranks["Ticker"] == self.request.ticker]
            if row.empty:
                raise RuntimeError(f"{self.request.ticker} was not found in the supplied universe data.")
            self.scenario_summary = json_safe(row.iloc[0].to_dict())
            self.scenario_summary["rank_in_sector"] = int(row.iloc[0]["Rank_In_Sector"])

    def load_context(self, news: Optional[pd.DataFrame] = None) -> None:
        """Optional step: attach timestamped news metadata known by the analysis date.

        Context is logged and replayable but is never a committed numeric fact.
        """
        if self.state != HarnessState.RUN_SCENARIO:
            raise RuntimeError("Context can be loaded only right after the scenario step.")
        self._transition(HarnessState.LOAD_CONTEXT)
        if news is None:
            news = get_news_events(ticker=self.request.ticker, as_of=self.request.as_of_date)
        else:
            news = news[news["effective_date"] <= pd.Timestamp(self.request.as_of_date)]  # same cutoff for supplied records
        self.context = {
            "news_count": int(len(news)),
            "news": json_safe(news.tail(5)),
            "warnings": [] if len(news) else ["No timestamped news records: news coverage is sparse (soft warning)."],
        }

    def run_backtest(self) -> None:
        if self.enriched_data is None:
            raise RuntimeError("Indicators must be calculated before backtesting.")
        self._transition(HarnessState.RUN_BACKTEST)
        signals = (self.enriched_data["Supertrend_Direction"] == 1).astype(int)
        result = run_backtest(signals, self.price_data, benchmark_df=self.benchmark_data)
        self.backtest_metrics = json_safe(result.metrics)

    def build_evidence(self) -> None:
        if self.quality is None:
            raise RuntimeError("Quality report is required before evidence can be committed.")
        self._transition(HarnessState.BUILD_EVIDENCE)
        evidence = build_evidence(
            as_of_date=self.request.as_of_date,
            quality_report=self.quality.to_dict(),
            scenario=self.request.scenario,
            scenario_summary=self.scenario_summary,
            backtest_metrics=self.backtest_metrics,
        )
        self.evidence = evidence_to_dict(evidence)
        self.action = select_action(self.request.scenario, self.scenario_summary)

    def submit_human_review(self, review: HumanReview) -> None:
        if self.state != HarnessState.BUILD_EVIDENCE:
            raise RuntimeError("Human review is permitted only after evidence is committed.")
        self._transition(HarnessState.HUMAN_REVIEW)
        self.human_review = review

    def log_and_close(self) -> str:
        if self.state == HarnessState.BUILD_EVIDENCE and not self.request.require_human_review:
            self._transition(HarnessState.LOG_AND_CLOSE)
        elif self.state == HarnessState.HUMAN_REVIEW:
            self._transition(HarnessState.LOG_AND_CLOSE)
        else:
            raise RuntimeError("A required human review has not been supplied.")
        record = {
            "analysis_id": self.analysis_id,
            "request": self.request.model_dump(),
            "state_history": self.transitions + [HarnessState.CLOSED.value],
            "quality": self.quality.to_dict() if self.quality else None,
            "scenario_summary": self.scenario_summary,
            "backtest_metrics": self.backtest_metrics,
            "context": self.context,
            "evidence": self.evidence,
            "action": self.action,
            "human_review": self.human_review.model_dump() if self.human_review else None,
        }
        path = log_decision(record)
        self._transition(HarnessState.CLOSED)
        return str(path)
