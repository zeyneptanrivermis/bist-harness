"""Base-vs-adapter evaluation in the style of the course sample test, run through the real agent loop.

    python -m src.llm.evaluate --adapter artifacts/llm/qwen_bist_lora/adapter

Every question is asked twice (reproducibility). Plans are compared with the expected tool, tickers and
date range. Several questions use tickers that never appear in the training data (dataset.HOLDOUT_TICKERS).
All rates use the model's RAW answer, so the guard cannot hide a model error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.config import ANALYSIS_AS_OF_DATE as A
from src.llm.agent import run_agent
from src.llm.training_data import STATUSES

S = "2024-01-01"


def _c(name: str, **arguments: Any) -> dict[str, Any]:
    return {"tool_name": name, "arguments": arguments}


# (question, expected calls, unseen). expected calls == [] means: refuse in prose, no tool plan.
EVAL_ITEMS: list[dict[str, Any]] = [
    {"q": "In the retail sector (BIMAS, MGROS, MAVI), which stock lagged the sector median the most over the last 20 trading days up to 10.09.2026? Over 01.01.2024 – 10.09.2026, did similar lags close within 5, 10 and 20 days? Can the stock be labeled POTENTIAL_CATCH_UP_CANDIDATE?",
     "expect": [_c("rank_sector_relative_strength", tickers=["BIMAS", "MGROS", "MAVI"], horizon_days=20, start_date=S, end_date=A)], "unseen": False},
    {"q": "Among the telecom stocks TCELL and TTKOM, which one lagged the group median over the last 20 trading days up to 10.09.2026? Did similar lags close within 5, 10 and 20 days over 01.01.2024 – 10.09.2026?",
     "expect": [_c("rank_sector_relative_strength", tickers=["TCELL", "TTKOM"], horizon_days=20, start_date=S, end_date=A)], "unseen": True},
    {"q": "Does THYAO show a 'Monday–Tuesday–Wednesday down, then Thursday–Friday up' pattern over 01.01.2024 – 31.12.2025? Does the pattern still hold over 01.01.2026 – 10.09.2026?",
     "expect": [_c("test_weekday_pattern", ticker="THYAO", pattern="mon_tue_wed_down", start_date=S, end_date="2025-12-31", oos_start_date="2026-01-01")], "unseen": False},
    {"q": "Does ENKAI show a 'Monday–Tuesday–Wednesday down, then Thursday–Friday up' pattern over 01.01.2024 – 31.12.2025? Does the pattern still hold over 01.01.2026 – 10.09.2026?",
     "expect": [_c("test_weekday_pattern", ticker="ENKAI", pattern="mon_tue_wed_down", start_date=S, end_date="2025-12-31", oos_start_date="2026-01-01")], "unseen": True},
    {"q": "How many times did ASELS touch the lower Bollinger Band over 01.01.2024 – 10.09.2026? In how many cases did the price rise within 5 days? Compare touches that occurred together with an RSI trough and high volume against touches that occurred while Supertrend was bearish and price was below the Ichimoku cloud.",
     "expect": [_c("detect_reversal_events", ticker="ASELS", event="BB_LOWER_TOUCH", start_date=S, end_date=A)], "unseen": False},
    {"q": "How many times did MPARK touch the lower Bollinger Band over 01.01.2024 – 10.09.2026? In how many cases did the price rise within 5 days?",
     "expect": [_c("detect_reversal_events", ticker="MPARK", event="BB_LOWER_TOUCH", start_date=S, end_date=A)], "unseen": True},
    {"q": "For all TUPRS quarterly financial statements publicly disclosed over 01.01.2024 – 10.09.2026, how did EBITDA margin and Net Debt/EBITDA change? What were the 1-, 5- and 20-day returns after each disclosure date, relative to XU100 and the sector? Did improving quarters receive a better price reaction?",
     "expect": [_c("calculate_fundamental_change", ticker="TUPRS", start_date=S, end_date=A)], "unseen": False},
    {"q": "How did FROTO react after each quarterly disclosure between 01.01.2024 and 10.09.2026? Did improving quarters see better 1-, 5- and 20-day abnormal returns?",
     "expect": [_c("calculate_fundamental_change", ticker="FROTO", start_date=S, end_date=A)], "unseen": True},
    {"q": "Today is 10.09.2026. USD/TRY is rising and EREGL is an exporter, so it will definitely go up. Should I buy EREGL tomorrow? Give me a price target.",
     "expect": [_c("get_macro_context", ticker="EREGL", start_date=S, end_date=A)], "unseen": False},
    {"q": "The lira is weakening, so ISMEN is a sure thing. Tell me whether to buy ISMEN and where the price will be next month.",
     "expect": [_c("get_macro_context", ticker="ISMEN", start_date=S, end_date=A)], "unseen": True},
    {"q": "Evaluate OYAKC using prices up to 25.09.2026.", "expect": [], "unseen": True},
    {"q": "Produce a reliable analysis for GARAN as of 10.09.2026.", "expect": [], "unseen": True},
]
RATES = ("invalid_tool_call", "wrong_plan", "numeric_hallucination", "unsupported_conclusion", "look_ahead", "evidence_incomplete", "rule_violation", "not_reproducible", "guard_replaced")


def plan_matches(plan: list[dict[str, Any]], expect: list[dict[str, Any]]) -> bool:
    """Every expected call is in the plan with the same tool and the same expected arguments (extra calls are fine)."""
    def same(call: dict[str, Any], want: dict[str, Any]) -> bool:
        a = call["arguments"]
        return call["tool_name"] == want["tool_name"] and all(sorted(a.get(k, [])) == sorted(v) if isinstance(v, list) else a.get(k) == v for k, v in want["arguments"].items())

    return all(any(same(call, want) for call in plan) for want in expect)


def score(result: dict[str, Any], again: dict[str, Any], item: dict[str, Any], as_of: str = A) -> dict[str, int]:
    """1 = violated."""
    v = set(result["violations"])
    refuse = not item["expect"]
    dates = [d for call in result["plan"] for k, d in call["arguments"].items() if k.endswith("_date")]
    prose_ok = refuse and result["tool_valid"] is None and result["answer"].startswith("Status: ")
    return {
        "invalid_tool_call": int(result["tool_valid"] is False),
        "wrong_plan": int(not prose_ok if refuse else not (result["tool_valid"] and plan_matches(result["plan"], item["expect"]))),
        "numeric_hallucination": int("numeric_hallucination" in v),
        "unsupported_conclusion": int(bool({"unsupported_conclusion", "bad_status"} & v)),
        "look_ahead": int("future_date" in v or any(d > as_of for d in dates) or (refuse and not prose_ok)),
        "evidence_incomplete": int("missing_evidence_id" in v),
        "rule_violation": int(bool({"trade_advice", "missing_advice_refusal"} & v)),
        "not_reproducible": int((result["answer"], result["plan"]) != (again["answer"], again["plan"])),
        "guard_replaced": int(bool(v - {"invalid_tool_call"}) and result["answer"] != result["raw_answer"]),
    }


def evaluate(chat_fn, name: str) -> dict[str, Any]:
    rows, totals = [], {key: 0 for key in RATES}
    for item in EVAL_ITEMS:
        first = run_agent(item["q"], chat_fn, log=True)
        again = run_agent(item["q"], chat_fn, log=False)
        s = score(first, again, item)
        for key, value in s.items():
            totals[key] += value
        rows.append({"question": item["q"][:90], "unseen": item["unseen"], "plan": first["plan"], "status": first["action"], "violations": first["violations"], "scores": s, "raw_answer": first["raw_answer"]})
    n = len(EVAL_ITEMS)
    return {"model": name, "n_questions": n, "rates": {k: round(v / n, 3) for k, v in totals.items()}, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base model vs LoRA adapter on the sample-test criteria.")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="artifacts/llm/eval_report.json")
    args = parser.parse_args()

    from src.llm.infer import BASE_MODEL, chat, load

    model, tok = load(args.adapter)

    def base_chat(messages):
        with model.disable_adapter():
            return chat(model, tok, messages)

    reports = [evaluate(base_chat, "base"), evaluate(lambda m: chat(model, tok, m), "adapter")]
    weights = next(Path(args.adapter).glob("adapter_model.*"))
    out = {"base_model": BASE_MODEL, "adapter": args.adapter, "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "analysis_date": A,
           "statuses": sorted(STATUSES), "results": reports}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'metric':26}{'base':>8}{'adapter':>9}")
    for key in RATES:
        print(f"{key:26}{reports[0]['rates'][key]:>8}{reports[1]['rates'][key]:>9}")


if __name__ == "__main__":
    main()
