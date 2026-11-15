"""Typed, replayable contracts for A–E harness experiments."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ExperimentVersion(str, Enum):
    RAW_LLM = "A_RAW_LLM"
    TOOLS = "B_TOOLS"
    STATEFUL = "C_STATEFUL_HARNESS"
    EVIDENCE = "D_EVIDENCE_QUALITY"
    HUMAN_REVIEW = "E_REPLAY_HUMAN_REVIEW"


class TrapType(str, Enum):
    NONE = "none"
    LOOK_AHEAD = "look_ahead"
    MISSING_DATA = "missing_data"
    DUPLICATE_DATES = "duplicate_dates"
    SHORT_HISTORY = "short_history"
    UNKNOWN_TICKER = "unknown_ticker"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    DIRECT_TRADE_REQUEST = "direct_trade_request"
    UNVERIFIED_SOURCE = "unverified_source"


class ExperimentCase(BaseModel):
    """A fixed question plus safety ground truth, not a trading recommendation."""

    case_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,48}$")
    question: str = Field(min_length=10)
    ticker: str
    scenario: Literal["technical_reversal", "weekday_pattern", "fundamental_reaction", "sector_laggard"]
    as_of_date: str
    trap: TrapType = TrapType.NONE
    should_block: bool = False
    expected_gate_status: Optional[str] = None
    permitted_tools: List[str] = Field(default_factory=list)
    required_evidence_ids: List[str] = Field(default_factory=lambda: ["E1", "E2"])
    known_numeric_values: List[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    succeeded: bool = True


class LLMResponse(BaseModel):
    """Normalized provider response, independent of a specific model SDK."""

    content: str
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    state_history: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    quality_status: Optional[str] = None
    replay_reference: Optional[str] = None
    human_reviewed: bool = False
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    estimated_cost_usd: Optional[float] = Field(default=None, ge=0)
    latency_ms: Optional[float] = Field(default=None, ge=0)


class EvaluationResult(BaseModel):
    passed: bool
    numeric_hallucination: bool = False
    invalid_tool_call: bool = False
    look_ahead_violation: bool = False
    unsupported_conclusion: bool = False
    direct_trade_advice: bool = False
    state_bypass: bool = False
    quality_gate_bypass: bool = False
    missing_replay: bool = False
    missing_human_review: bool = False
    issues: List[str] = Field(default_factory=list)


class ExperimentRun(BaseModel):
    case: ExperimentCase
    version: ExperimentVersion
    provider_name: str
    prompt: str
    response: LLMResponse
    evaluation: EvaluationResult
