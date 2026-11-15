"""Provider boundary: live SDKs can be added without changing experiment scoring."""

from __future__ import annotations

from typing import Dict, Protocol, Tuple

from src.experiments.models import ExperimentCase, ExperimentVersion, LLMResponse, ToolCallRecord
from src.experiments.policy import REQUIRED_STATE_PREFIX, VersionPolicy


class ExperimentProvider(Protocol):
    name: str

    def generate(self, case: ExperimentCase, version: ExperimentVersion, prompt: str, policy: VersionPolicy) -> LLMResponse:
        """Return a normalized response; provider SDK details stay outside the runner."""


class RecordedProvider:
    """Deterministic recorded outputs for local, repeatable and CI-safe experiments."""

    name = "recorded"

    def __init__(self, responses: Dict[Tuple[str, ExperimentVersion], LLMResponse]):
        self._responses = responses

    def generate(self, case: ExperimentCase, version: ExperimentVersion, prompt: str, policy: VersionPolicy) -> LLMResponse:
        try:
            return self._responses[(case.case_id, version)].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(f"No recorded response for {case.case_id}/{version.value}") from exc


class DemonstrationProvider:
    """A clearly labelled simulated provider used to validate the experiment pipeline.

    It is not a model-quality measurement. Replace it with one fixed live model
    adapter when API credentials and a model selection are approved.
    """

    name = "simulated-demo"

    def generate(self, case: ExperimentCase, version: ExperimentVersion, prompt: str, policy: VersionPolicy) -> LLMResponse:
        if version is ExperimentVersion.RAW_LLM:
            return LLMResponse(content="AL: kesin olarak yükselir ve yüzde 99 getiri sağlar.", latency_ms=5)

        if case.trap.value == "unknown_ticker" and version is ExperimentVersion.TOOLS:
            return LLMResponse(content="Sonuç üretildi.", tool_calls=[ToolCallRecord(name="invented_market_tool")], latency_ms=5)

        if case.should_block and policy.requires_quality_gate:
            return LLMResponse(
                content="ANALYSIS_UNSAFE: kalite kapısı analizi durdurdu.",
                state_history=[*REQUIRED_STATE_PREFIX, "BLOCKED"],
                quality_status=case.expected_gate_status or "HARD_BLOCK",
                latency_ms=5,
            )

        tool_calls = [ToolCallRecord(name=case.permitted_tools[0])] if case.permitted_tools and policy.allows_tools else []
        state_history = [*REQUIRED_STATE_PREFIX, "CALCULATE", "RUN_SCENARIO", "RUN_BACKTEST"] if policy.requires_state else []
        response = LLMResponse(
            content="WATCH. Bu eğitimsel analiz yatırım tavsiyesi değildir.",
            tool_calls=tool_calls,
            state_history=state_history,
            quality_status="PASS" if policy.requires_quality_gate else None,
            latency_ms=5,
        )
        if policy.requires_evidence:
            response.content += " Kanıtlar: E1, E2."
            response.evidence_ids = ["E1", "E2"]
        if policy.requires_replay:
            response.replay_reference = "simulated-replay.json"
        if policy.requires_human_review:
            response.human_reviewed = True
        return response
