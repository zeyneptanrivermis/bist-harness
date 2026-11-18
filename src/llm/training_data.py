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

from src.config import ALLOWED_ACTIONS
from src.experiments.catalog import build_experiment_catalog
from src.experiments.models import ExperimentCase
from src.universe import get_all_tickers


DEFAULT_SYSTEM_PROMPT = """You are the BIST Harness research assistant.
Write concise English research summaries using only supplied tool results and
committed evidence. You may request an approved tool in the required JSON form
when evidence is missing. Never invent numbers, sources, or tool results.
Never use data after the stated as-of date. Do not issue BUY, SELL, or
guaranteed-return advice. Use only the educational status vocabulary supplied
by the harness."""

TOOL_CALL_FORMAT = """When tools are needed, return exactly one JSON object and nothing else:
{"tool_calls": [{"tool_name": "<tool>", "arguments": {...}}]}
Use 1 to 4 calls. Dates are ISO (YYYY-MM-DD). Never request data after the analysis date. If the request cannot be answered safely, reply in prose that starts with "Status: INVESTIGATE." or "Status: ANALYSIS_UNSAFE."."""


# The model's final conclusion must come from this finite set (gate statuses such as HARD_BLOCK are not conclusions).
STATUSES = frozenset(ALLOWED_ACTIONS)

def tool_prompt(question: str, as_of_date: str | None) -> str:
    """User turn for the tool-selection step; identical in training and in the agent loop."""
    return f"Question: {question}" + (f"\nAs-of date: {as_of_date}" if as_of_date else "")


def report_prompt(question: str, as_of_date: str, evidence: list[dict[str, Any]]) -> str:
    """User turn for the evidence-report step; identical in training and in the agent loop."""
    return (
        f"Question: {question}\n"
        f"As-of date: {as_of_date}\n"
        f"Committed evidence (JSON):\n{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}\n\n"
        "Write a concise English research summary. Cite evidence IDs, do not add facts, and do not provide BUY/SELL advice."
    )


def tool_system_prompt() -> str:
    tools = "\n".join(f"- {t['name']}({t['arguments']}): {t['description']}" for t in project_tool_definitions())
    return f"{DEFAULT_SYSTEM_PROMPT}\n\n{TOOL_CALL_FORMAT}\n\nTools:\n{tools}"


TOOL_SPECS = [
    ("get_stock_history", "ticker, start_date, end_date", "Price history summary."),
    ("calculate_indicators", "ticker, start_date, end_date", "Latest SMA/RSI/Bollinger/KAMA/Supertrend/Ichimoku state."),
    ("detect_reversal_events", "ticker, event (BB_LOWER_TOUCH, BB_MIDDLE_TEST, BB_UPPER_TOUCH, RSI_OVERSOLD, RSI_OVERBOUGHT, SUPERTREND_BULLISH_FLIP, SUPERTREND_BEARISH_FLIP, KAMA_CROSS_UP, ICHIMOKU_CLOUD_BREAKOUT), start_date, end_date", "Event counts, forward returns, single vs combined conditions."),
    ("rank_sector_relative_strength", "tickers (list of 2+) or sector, horizon_days, start_date, end_date", "Peer lag gaps and historical catch-up hit rates."),
    ("test_weekday_pattern", "ticker, pattern (weekday, mon_tue_wed_down, consecutive_down, consecutive_up, large_down_day), start_date, end_date, oos_start_date", "Pattern statistics in-sample and out-of-sample."),
    ("get_fundamental_metrics", "ticker", "Point-in-time quarterly metrics."),
    ("calculate_fundamental_change", "ticker, start_date, end_date", "Quarterly changes and post-disclosure abnormal returns."),
    ("get_macro_context", "start_date, end_date, ticker (optional)", "USD/TRY, EUR/TRY, policy rate, CPI, Fed funds and FX link to a stock."),
    ("run_backtest", "ticker, start_date, end_date", "Cost-aware t+1 backtest."),
    ("check_data_quality", "ticker", "Mandatory quality gate."),
]


def project_tool_definitions() -> list[dict[str, str]]:
    """Compact tool schema shown to the model; execution stays in the Python router."""
    return [{"name": name, "arguments": args, "description": description} for name, args, description in TOOL_SPECS]


def refusal_text(gate_status: str) -> str:
    return (
        "Status: ANALYSIS_UNSAFE. "
        f"The request is blocked before any calculation because it violates a mandatory safety or data-quality rule (gate: {gate_status})."
    )


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
                {"role": "system", "content": tool_system_prompt()},
                {"role": "user", "content": tool_prompt(case.question, case.as_of_date)},
                {"role": "assistant", "content": refusal_text(status)},
            ],
            "metadata": {"case_id": case.case_id, "task": "safe_refusal", "human_reviewed": False},
        }
    tool_name = next((tool for tool in case.permitted_tools if tool != "get_stock_history"), "check_data_quality")
    request = {"tool_name": tool_name, "arguments": _tool_arguments(case, tool_name)}
    return {
        "messages": [
            {"role": "system", "content": tool_system_prompt()},
            {"role": "user", "content": tool_prompt(case.question, case.as_of_date)},
            {"role": "assistant", "content": json.dumps(request, sort_keys=True)},
        ],
        "metadata": {"case_id": case.case_id, "task": "tool_selection", "human_reviewed": False},
    }


def _report_record(case: ExperimentCase) -> dict[str, Any]:
    if case.should_block:
        status = case.expected_gate_status or "ANALYSIS_UNSAFE"
        response = (
            "Status: ANALYSIS_UNSAFE.\n"
            f"The requested analysis cannot continue because the mandatory data-quality or point-in-time gate returned {status}.\n"
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
