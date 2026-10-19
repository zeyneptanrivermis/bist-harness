"""JSON-safe request and response contracts used by MCP tools."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ToolResponse(BaseModel):
    """Every tool exposes provenance and a JSON-compatible payload."""

    tool_name: str
    status: Literal["ok", "warning", "error"] = "ok"
    message: Optional[str] = None
    as_of_date: Optional[str] = None
    source: str = "local_cache"
    data: Dict[str, Any] = Field(default_factory=dict)


class AnalysisRequest(BaseModel):
    """The bounded request accepted by the stateful harness."""

    ticker: str = Field(min_length=1, max_length=12)
    scenario: Literal[
        "technical_reversal",
        "weekday_pattern",
        "fundamental_reaction",
        "sector_laggard",
    ]
    as_of_date: str
    require_human_review: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HumanReview(BaseModel):
    decision: Literal["APPROVED", "REJECTED"]
    reviewer: str = Field(min_length=1, max_length=120)
    rationale: str = Field(min_length=3, max_length=2_000)
