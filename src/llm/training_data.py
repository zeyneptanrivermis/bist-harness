"""Create reviewable supervised fine-tuning records without a cloud LLM.

The seed set is deliberately small and template based. It demonstrates the
project's required safe behaviour, but it is not a sufficient financial corpus.
Team members must review, edit, and expand it with deterministic tool outputs
before training a submitted adapter.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.experiments.catalog import build_experiment_catalog
from src.experiments.models import ExperimentCase


DEFAULT_SYSTEM_PROMPT = """You are the BIST Harness research assistant.
Write concise English research summaries using only supplied tool results and
committed evidence. You may request an approved tool in the required JSON form
when evidence is missing. Never invent numbers, sources, or tool results.
Never use data after the stated as-of date. Do not issue BUY, SELL, or
guaranteed-return advice. Use only the educational status vocabulary supplied
by the harness."""

TOOL_CALL_FORMAT = """Return exactly one JSON object when a tool is needed:
{"tool_name": "approved_tool", "arguments": {"ticker": "TICKER", "as_of_date": "YYYY-MM-DD"}}
Do not add prose to a tool-call response."""


def project_tool_definitions() -> list[dict[str, Any]]:
    """Compact tool schema for prompts; execution remains in Python wrappers."""
    return [
        {"name": "get_stock_history", "description": "Load local OHLCV data up to as_of_date."},
        {"name": "check_data_quality", "description": "Run the mandatory quality and point-in-time gate."},
        {"name": "calculate_indicators", "description": "Calculate deterministic technical indicators."},
        {"name": "detect_reversal_events", "description": "Detect technical-reversal events."},
        {"name": "test_weekday_pattern", "description": "Evaluate weekday-pattern statistics."},
        {"name": "rank_sector_laggards", "description": "Rank the fixed universe by sector-relative performance."},
        {"name": "get_fundamental_metrics", "description": "Load curated fundamental records with their provenance warning."},
        {"name": "align_fundamental_disclosures", "description": "Align disclosed fundamentals using point-in-time dates."},
        {"name": "run_backtest", "description": "Run the cost-aware, t+1 backtest."},
    ]


def _tool_arguments(case: ExperimentCase, tool_name: str) -> dict[str, str]:
    arguments = {"as_of_date": case.as_of_date}
    if tool_name != "rank_sector_laggards":
        arguments["ticker"] = case.ticker
    return arguments


def _tool_record(case: ExperimentCase) -> dict[str, Any]:
    if case.should_block:
        status = case.expected_gate_status or "ANALYSIS_UNSAFE"
        return {
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": case.question},
                {
                    "role": "assistant",
                    "content": f"Status: {status}. The request is blocked before any calculation because it violates a mandatory safety or data-quality rule.",
                },
            ],
            "metadata": {"case_id": case.case_id, "task": "safe_refusal", "human_reviewed": False},
        }
    tool_name = next((tool for tool in case.permitted_tools if tool != "get_stock_history"), "check_data_quality")
    request = {"tool_name": tool_name, "arguments": _tool_arguments(case, tool_name)}
    return {
        "messages": [
            {"role": "system", "content": f"{DEFAULT_SYSTEM_PROMPT}\n\n{TOOL_CALL_FORMAT}"},
            {"role": "user", "content": case.question},
            {"role": "assistant", "content": json.dumps(request, sort_keys=True)},
        ],
        "metadata": {"case_id": case.case_id, "task": "tool_selection", "human_reviewed": False},
    }


def _report_record(case: ExperimentCase) -> dict[str, Any]:
    if case.should_block:
        status = case.expected_gate_status or "ANALYSIS_UNSAFE"
        response = (
            f"Status: {status}.\n"
            "The requested analysis cannot continue because the mandatory data-quality or point-in-time gate failed.\n"
            "Evidence: E1 (quality-gate result).\n"
            "Limitation: No financial conclusion is produced from unsafe data."
        )
        evidence = "E1: The quality gate returned a blocking status."
    else:
        response = (
            "Status: WATCH.\n"
            "Evidence: E1 confirms that the data-quality gate completed before calculation; "
            "E2 records the scenario result available at the stated cutoff.\n"
            "Limitation: This educational summary is evidence-bound and is not investment advice."
        )
        evidence = (
            "E1: Data quality gate completed before calculation.\n"
            f"E2: {case.scenario} was evaluated for {case.ticker} at {case.as_of_date}."
        )
    user_prompt = (
        f"Question: {case.question}\nAs-of date: {case.as_of_date}\n"
        f"Committed evidence:\n{evidence}\n\nWrite an English research summary."
    )
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response},
        ],
        "metadata": {"case_id": case.case_id, "task": "evidence_report", "human_reviewed": False},
    }


def generate_seed_records(cases: Iterable[ExperimentCase] | None = None) -> list[dict[str, Any]]:
    """Generate deterministic starter records from the versioned A-E catalog."""
    catalog = list(cases) if cases is not None else build_experiment_catalog()
    records = [record for case in catalog for record in (_tool_record(case), _report_record(case))]
    validate_records(records, require_human_review=False)
    return records


def stable_split(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split by case ID so one case never leaks across train/validation/test."""
    result = {"train": [], "validation": [], "test": []}
    for record in records:
        case_id = record["metadata"]["case_id"]
        bucket = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16) % 10
        split = "train" if bucket < 7 else "validation" if bucket < 9 else "test"
        result[split].append(record)
    if not all(result.values()):
        raise ValueError("The supplied records did not populate every data split.")
    return result


def validate_records(records: Iterable[dict[str, Any]], *, require_human_review: bool = True) -> None:
    """Reject malformed records and obvious violations before an expensive run."""
    records = list(records)
    if not records:
        raise ValueError("No training records supplied.")
    forbidden = ("buy", "sell", "guaranteed return", "guaranteed profit")
    for index, record in enumerate(records):
        messages = record.get("messages", [])
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Record {index} must contain system, user, assistant messages in order.")
        metadata = record.get("metadata", {})
        if not metadata.get("case_id") or not metadata.get("task"):
            raise ValueError(f"Record {index} requires case_id and task metadata.")
        if require_human_review and not metadata.get("human_reviewed"):
            raise ValueError(f"Record {index} has not been marked human_reviewed.")
        answer = messages[-1].get("content", "").lower()
        if any(term in answer for term in forbidden):
            raise ValueError(f"Record {index} contains prohibited investment language.")


def write_jsonl_splits(output_dir: str | Path, records: Iterable[dict[str, Any]]) -> dict[str, Path]:
    """Write train/validation/test JSONL files and a transparent manifest."""
    records = list(records)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    splits = stable_split(records)
    written: dict[str, Path] = {}
    for split, split_records in splits.items():
        path = output_path / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in split_records) + "\n", encoding="utf-8")
        written[split] = path
    manifest = {
        "record_count": sum(len(items) for items in splits.values()),
        "records_per_split": {split: len(items) for split, items in splits.items()},
        "tasks": dict(Counter(item["metadata"]["task"] for item in records)),
        "review_requirement": "Set metadata.human_reviewed=true only after team review before training.",
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    written["manifest"] = manifest_path
    return written
