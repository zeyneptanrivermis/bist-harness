"""End-to-end local loop: question -> JSON tool plan -> router -> harness states -> evidence -> guarded summary.

    from src.llm.agent import run_agent, local_chat_fn
    run_agent("Does THYAO show a Monday-Wednesday down pattern ...?", local_chat_fn("artifacts/llm/qwen_bist_lora/adapter"))

The model only (1) plans which tools to call and (2) words the summary. The harness (code) runs the
mandatory states in order, runs the quality gate and the backtest itself, picks the conclusion from
the tools' evidence, and replaces a model summary that breaks a rule with the deterministic draft.
The raw model answer is always kept so an evaluation sees the model's real behaviour.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable

from src.config import ANALYSIS_AS_OF_DATE
from src.harness.replay import log_decision
from src.llm.infer import first_json
from src.llm.summaries import advice_requested, draft_summary
from src.llm.tool_loop import SafeProjectToolRouter, ToolRequestError
from src.llm.training_data import DEFAULT_SYSTEM_PROMPT, STATUSES, refusal_text, report_prompt, tool_prompt, tool_system_prompt
from src.mcp_tools import research as R

ChatFn = Callable[[list[dict[str, str]]], str]
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_STATUS = re.compile(r"Status: ([A-Z_]+)")
_ADVICE = re.compile(r"\b(buy|sell)\b|guaranteed", re.IGNORECASE)
_PRIMARY = ("rank_sector_relative_strength", "test_weekday_pattern", "detect_reversal_events", "calculate_fundamental_change", "get_macro_context")
_BLOCKING = {"HARD_BLOCK", "ANALYSIS_UNSAFE"}


def local_chat_fn(adapter_dir: str = "", model_name: str | None = None, max_new_tokens: int = 900) -> ChatFn:
    """Chat function backed by a local Hugging Face model (+ LoRA adapter). Needs the model on disk."""
    from src.llm.infer import BASE_MODEL, chat, load

    model, tokenizer = load(adapter_dir, model_name or BASE_MODEL)
    return lambda messages: chat(model, tokenizer, messages, max_new_tokens=max_new_tokens)


def ollama_chat_fn(model: str = "bist-qwen", host: str = "http://localhost:11434", max_new_tokens: int = 900) -> ChatFn:
    """Chat function for a model served by a local Ollama daemon (localhost only, no cloud API)."""
    from urllib.request import Request, urlopen

    def chat(messages: list[dict[str, str]]) -> str:
        body = json.dumps({"model": model, "messages": messages, "stream": False, "options": {"temperature": 0, "num_predict": max_new_tokens}}).encode()
        with urlopen(Request(f"{host}/api/chat", body, {"Content-Type": "application/json"}), timeout=300) as response:
            text = json.loads(response.read())["message"]["content"]
            return re.sub(r"^\s*<think>\s*</think>\s*", "", text).strip()  # Qwen3 non-thinking mode writes an empty think block

    return chat


def _numeric_forms(value: float) -> set[str]:
    forms = {f"{value:.{d}f}" for d in (0, 1, 2, 3, 4)} | {f"{value * 100:.{d}f}" for d in (0, 1, 2)}
    return forms | {str(value)} | {f.lstrip("-") for f in forms}


def _evidence_numbers(node: Any) -> set[str]:
    if isinstance(node, bool) or node is None:
        return set()
    if isinstance(node, (int, float)):
        return _numeric_forms(float(node))
    if isinstance(node, dict):
        return set().union(*(_evidence_numbers(v) for v in node.values())) if node else set()
    if isinstance(node, list):
        return set().union(*(_evidence_numbers(v) for v in node)) if node else set()
    return {n.lstrip("-") for n in _NUMBER.findall(str(node))}  # strings (periods, dates, warnings) carry digits


def unsupported_numbers(answer: str, evidence: list[dict[str, Any]], question: str = "") -> list[str]:
    """Numbers in the answer that trace neither to the evidence nor to the user's own question (E-ids and ISO dates ignored).

    shortcut: integers 0-20 pass unchecked (template words like "5-day", "t+1"); upgrade by tagging them.
    """
    allowed = _evidence_numbers(evidence) | {n.lstrip("-") for n in _NUMBER.findall(question)}
    text = re.sub(r"\bE\d+\b|XU100", " ", _DATE.sub(" ", answer))
    return [n for n in _NUMBER.findall(text) if n.lstrip("-") not in allowed and not (n.isdigit() and int(n) <= 20)]


def check_answer(answer: str, evidence: list[dict[str, Any]], expected_status: str | None, as_of_date: str, question: str = "") -> list[str]:
    """Rule violations of one model summary (empty list = acceptable)."""
    found = []
    match = _STATUS.match(answer)
    if not match or match.group(1) not in STATUSES:
        found.append("bad_status")
    elif expected_status and match.group(1) != expected_status:
        found.append("unsupported_conclusion")
    if _ADVICE.search(answer):
        found.append("trade_advice")
    if any(d > as_of_date for d in _DATE.findall(answer)):
        found.append("future_date")
    if any(not re.search(rf"\b{item['evidence_id']}\b", answer) for item in evidence):
        found.append("missing_evidence_id")
    if unsupported_numbers(answer, evidence, question):
        found.append("numeric_hallucination")
    if advice_requested(question) and "cannot give investment advice" not in answer:
        found.append("missing_advice_refusal")
    return found


def _plan_tickers(plan: list[tuple[str, dict[str, Any]]]) -> list[str]:
    seen: list[str] = []
    for _, args in plan:
        names = ([args["ticker"]] if "ticker" in args else []) + list(args.get("tickers", [])) + (R.resolve_peers(None, args.get("sector")) if args.get("sector") else [])
        seen += [t for t in names if t not in seen]
    return seen


def _item(n: int, values: dict[str, Any], as_of: str) -> dict[str, Any]:
    return {"evidence_id": f"E{n}", "claim": f"{values['tool']} output at the cutoff.", "source_tool": values["tool"], "as_of_date": as_of, "values": values}


def _expected_status(evidence: list[dict[str, Any]]) -> str:
    values = [e["values"] for e in evidence]
    if any(v.get("suggested_status") == "ANALYSIS_UNSAFE" for v in values):
        return "ANALYSIS_UNSAFE"
    if any(not v.get("available", True) for v in values):
        return "INVESTIGATE"
    primary = next((v for v in values if v["tool"] in _PRIMARY), None) or next((v for v in reversed(values) if v["tool"] != "run_backtest"), values[-1])
    return primary.get("suggested_status", "INVESTIGATE")


def _primary_ticker(plan: list[tuple[str, dict[str, Any]]], outputs: list[dict[str, Any]]) -> str | None:
    for (_, args), out in zip(plan, outputs):
        if args.get("ticker"):
            return args["ticker"]
        if out.get("most_lagging"):
            return out["most_lagging"]
    return None


def execute_plan(plan: list[tuple[str, dict[str, Any]]], as_of: str, router: SafeProjectToolRouter) -> dict[str, Any]:
    """Run the harness states for a validated plan: gate first, then the tools, then the mandatory backtest."""
    states = ["START", "SELECT_UNIVERSE", "LOAD_DATA", "CHECK_DATA_QUALITY"]
    evidence: list[dict[str, Any]] = []
    tickers = _plan_tickers(plan)
    if tickers:  # the mandatory gate runs in code, before any calculation, for every stock in the plan
        gates = {t: router.execute("check_data_quality", {"ticker": t}, as_of_date=as_of) for t in tickers}
        status = {t: g.get("gate_status", "ANALYSIS_UNSAFE") for t, g in gates.items()}
        blocking = [s for s in status.values() if s in _BLOCKING]
        if blocking or any(not g["available"] for g in gates.values()):
            return {"states": [*states, "BLOCKED"], "evidence": [], "expected": "ANALYSIS_UNSAFE", "blocked_answer": refusal_text(blocking[0] if blocking else "NO_DATA")}
        evidence.append(_item(1, {"tool": "check_data_quality", "available": True, "gate": status, "warnings": []}, as_of))
    states += ["CALCULATE", "RUN_SCENARIO"]
    outputs = [router.execute(name, args, as_of_date=as_of) for name, args in plan]
    evidence += [_item(len(evidence) + i + 1, out, as_of) for i, out in enumerate(outputs)]
    ticker = _primary_ticker(plan, outputs)
    first = next((a for _, a in plan if "start_date" in a), {})
    window = {k: v for k, v in first.items() if k in ("start_date", "end_date")}
    names = {n for n, _ in plan}
    if ticker and "rank_sector_relative_strength" in names and "get_macro_context" not in names:  # a lag is only structural vs temporary with macro context
        states.append("LOAD_CONTEXT")
        evidence.append(_item(len(evidence) + 1, router.execute("get_macro_context", {"ticker": ticker, **window}, as_of_date=as_of), as_of))
    states.append("RUN_BACKTEST")
    if ticker and "run_backtest" not in names:  # the backtest state is mandatory, so the harness runs it if the plan did not
        evidence.append(_item(len(evidence) + 1, router.execute("run_backtest", {"ticker": ticker, **window}, as_of_date=as_of), as_of))
    return {"states": [*states, "BUILD_EVIDENCE", "LOG_AND_CLOSE"], "evidence": evidence, "expected": _expected_status(evidence), "blocked_answer": None}


def run_agent(question: str, chat_fn: ChatFn, *, as_of_date: str | None = None, router: SafeProjectToolRouter | None = None, log: bool = True) -> dict[str, Any]:
    """Answer one question. The returned dict is the full audit trail."""
    router = router or SafeProjectToolRouter()
    as_of = as_of_date or ANALYSIS_AS_OF_DATE
    result: dict[str, Any] = {"question": question, "as_of_date": as_of, "tool_call": None, "tool_valid": None, "plan": [], "state_history": ["START"],
                              "action": None, "evidence": [], "raw_answer": None, "answer": None, "violations": [], "replay_log": None}

    reply = first_json(chat_fn([{"role": "system", "content": tool_system_prompt()}, {"role": "user", "content": tool_prompt(question, as_of)}]))
    result["tool_call"] = reply
    try:
        plan = router.validate_plan(reply, as_of_date=as_of)
    except ToolRequestError as exc:
        result["raw_answer"] = reply
        status = _STATUS.match(reply)
        if status and status.group(1) in STATUSES and not _ADVICE.search(reply):
            result["answer"] = reply  # a prose refusal is allowed and is not a tool call
            result["action"] = status.group(1)
        else:
            result["tool_valid"] = False
            result["action"] = "ANALYSIS_UNSAFE"  # the router blocked the plan, which is what the answer says
            result["violations"] = ["invalid_tool_call"]
            result["answer"] = f"{refusal_text('INVALID_TOOL_REQUEST')} ({exc})"
        return result
    result["tool_valid"] = True
    result["plan"] = [{"tool_name": n, "arguments": a} for n, a in plan]
    run = execute_plan(plan, as_of, router)
    result["state_history"] = run["states"]
    if run["blocked_answer"]:
        result.update(action="ANALYSIS_UNSAFE", answer=run["blocked_answer"])
        return result
    evidence, expected, states = run["evidence"], run["expected"], run["states"]
    result.update(evidence=evidence, action=expected)

    raw = chat_fn([{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": report_prompt(question, as_of, evidence)}])
    result["raw_answer"] = raw
    result["violations"] = check_answer(raw, evidence, expected, as_of, question)
    result["answer"] = raw if not result["violations"] else draft_summary(expected, evidence, question)
    if log:
        key = hashlib.sha256(json.dumps([question, as_of, result["plan"]], sort_keys=True).encode()).hexdigest()[:16]  # same question -> same log id
        result["replay_log"] = str(log_decision({"analysis_id": key, "request": {"question": question, "as_of_date": as_of}, "state_history": states, "plan": result["plan"],
                                                 "evidence": evidence, "action": expected, "answer": result["answer"],
                                                 "raw_answer": raw, "violations": result["violations"]}))
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ask the local BIST harness model a question (no cloud API).")
    parser.add_argument("question", nargs="?", help="omit for an interactive session (the model is loaded once)")
    parser.add_argument("--as-of", default=ANALYSIS_AS_OF_DATE, help="analysis date; nothing after it may be used")
    parser.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    parser.add_argument("--adapter", default="", help="hf backend: LoRA adapter directory")
    parser.add_argument("--ollama-model", default="bist-qwen")
    parser.add_argument("--json", action="store_true", help="print the full audit trail")
    args = parser.parse_args()
    chat_fn = local_chat_fn(args.adapter) if args.backend == "hf" else ollama_chat_fn(args.ollama_model)
    show = lambda r: print(json.dumps(r, indent=2, ensure_ascii=False, default=str) if args.json else r["answer"])
    if args.question:
        show(run_agent(args.question, chat_fn, as_of_date=args.as_of))
        return
    print(f"Analysis date {args.as_of}. Ask a question; an empty line quits.")
    while question := input("> ").strip():
        show(run_agent(question, chat_fn, as_of_date=args.as_of))


if __name__ == "__main__":
    main()
