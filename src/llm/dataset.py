"""Generate the fine-tuning dataset for the five question families of the course sample test.

    python -m src.llm.dataset --output-dir data/llm/dataset_v3

Two record types, both built from the real tools on the cached prices:
  * plan:   question (+ analysis date) -> JSON tool plan, or a prose refusal;
  * report: question + committed evidence -> evidence-bound English summary.
The plan targets follow fixed rules and the report targets are the deterministic summaries in
summaries.py, so no number in a target can differ from the tool output. Some tickers are held
out (never used here) so that evaluate.py can test unseen stocks.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import ANALYSIS_AS_OF_DATE
from src.llm.agent import execute_plan
from src.llm.summaries import draft_summary
from src.llm.tool_loop import SafeProjectToolRouter
from src.llm.training_data import DEFAULT_SYSTEM_PROMPT, report_prompt, tool_prompt, tool_system_prompt, validate_records
from src.universe import get_universe_df

HOLDOUT_TICKERS = frozenset({"FROTO", "TCELL", "ENKAI", "MPARK", "OYAKC", "ISMEN"})  # only evaluate.py may use these
FUNDAMENTAL_TICKERS = ("ASELS", "BIMAS", "THYAO", "TUPRS", "FROTO")
AS_OF_CHOICES = [ANALYSIS_AS_OF_DATE] * 7 + ["2026-06-30", "2026-08-31"]
START = "2024-01-01"
EVENT_PHRASES = {
    "BB_LOWER_TOUCH": "touch the lower Bollinger Band", "BB_UPPER_TOUCH": "touch the upper Bollinger Band", "BB_MIDDLE_TEST": "test the middle Bollinger Band",
    "RSI_OVERSOLD": "become RSI oversold", "RSI_OVERBOUGHT": "become RSI overbought", "SUPERTREND_BULLISH_FLIP": "flip Supertrend to bullish",
    "SUPERTREND_BEARISH_FLIP": "flip Supertrend to bearish", "KAMA_CROSS_UP": "cross above KAMA", "ICHIMOKU_CLOUD_BREAKOUT": "break out above the Ichimoku cloud",
}


def _d(iso: str, dotted: bool) -> str:
    y, m, d = iso.split("-")
    return f"{d}.{m}.{y}" if dotted else iso


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"tool_name": name, "arguments": arguments}


def _universe() -> tuple[list[str], dict[str, list[str]]]:
    frame = get_universe_df()
    tickers = [t for t in frame["ticker"] if t not in HOLDOUT_TICKERS]
    sectors = {s: [t for t in g["ticker"] if t not in HOLDOUT_TICKERS] for s, g in frame.groupby("sector")}
    return tickers, {s: ts for s, ts in sectors.items() if len(ts) >= 2}


def _sector_case(rng: random.Random, tickers: list[str], sectors: dict[str, list[str]], as_of: str, dotted: bool) -> tuple[str, list[dict[str, Any]]]:
    h = rng.choice([20, 20, 20, 10, 60])
    s, e = _d(START, dotted), _d(as_of, dotted)
    if rng.random() < 0.3:
        sector = rng.choice(list(sectors))
        name = rng.choice([sector, sector.split(" / ")[0].lower()])
        q = rng.choice([
            f"Which {name} stock lagged its sector the most over the last {h} trading days up to {e}? Over {s} – {e}, did similar lags close within 5, 10 and 20 days?",
            f"Within the {name} sector, who is lagging over {h} trading days as of {e}, and is that a historical catch-up candidate over {s} – {e}?"])
        return q, [_call("rank_sector_relative_strength", sector=name, horizon_days=h, start_date=START, end_date=as_of)]
    group = rng.sample(tickers, rng.choice([2, 3, 3, 4])) if rng.random() < 0.5 else rng.sample(rng.choice(list(sectors.values())), 2)
    names = ", ".join(group)
    q = rng.choice([
        f"In the peer group ({names}), which stock lagged the group median the most over the last {h} trading days up to {e}? Over {s} – {e}, did similar lags close within 5, 10 and 20 days? "
        "Can the stock be labeled POTENTIAL_CATCH_UP_CANDIDATE?",
        f"Among {names}, which stock is lagging its peers over {h} trading days as of {e}, and have such lags historically caught up over {s} – {e}?",
        f"Rank {names} by relative strength against the group median over {h} days to {e}. Is there a catch-up candidate?"])
    return q, [_call("rank_sector_relative_strength", tickers=group, horizon_days=h, start_date=START, end_date=as_of)]


def _weekday_case(rng: random.Random, tickers: list[str], as_of: str, dotted: bool) -> tuple[str, list[dict[str, Any]]]:
    t = rng.choice(tickers)
    split = rng.choice([("2025-12-31", "2026-01-01")] * 3 + [("2025-06-30", "2025-07-01")])
    s, e1, o, e = _d(START, dotted), _d(split[0], dotted), _d(split[1], dotted), _d(as_of, dotted)
    pattern = rng.choices(["mon_tue_wed_down", "weekday", "consecutive_down", "large_down_day"], [6, 2, 1, 1])[0]
    q = {
        "mon_tue_wed_down": f"Does {t} show a 'Monday–Tuesday–Wednesday down, then Thursday–Friday up' pattern over {s} – {e1}? Does the pattern still hold over {o} – {e}?",
        "weekday": f"Is there a statistically significant day-of-week effect in {t} over {s} – {e1}, and does it survive out of sample from {o} to {e}?",
        "consecutive_down": f"After three consecutive down days, does {t} rebound? Test {s} – {e1} and then {o} – {e}.",
        "large_down_day": f"Does {t} mean-revert after a large down day over {s} – {e1}? Check the out-of-sample period {o} – {e} too."}[pattern]
    return q, [_call("test_weekday_pattern", ticker=t, pattern=pattern, start_date=START, end_date=split[0], oos_start_date=split[1])]


def _reversal_case(rng: random.Random, tickers: list[str], as_of: str, dotted: bool) -> tuple[str, list[dict[str, Any]]]:
    t = rng.choice(tickers)
    event = rng.choices(list(EVENT_PHRASES), [8, 1, 1, 1, 1, 1, 1, 1, 1])[0]
    s, e = _d(START, dotted), _d(as_of, dotted)
    if event == "BB_LOWER_TOUCH":
        q = (f"How many times did {t} touch the lower Bollinger Band over {s} – {e}? In how many cases did the price rise within 5 days? Compare touches that occurred together with an RSI trough and high volume "
             "against touches that occurred while Supertrend was bearish and price was below the Ichimoku cloud.")
    else:
        q = f"How many times did {t} {EVENT_PHRASES[event]} over {s} – {e}, and how did the price behave over the next 1, 3, 5 and 10 days?"
    calls = [_call("detect_reversal_events", ticker=t, event=event, start_date=START, end_date=as_of)]
    if rng.random() < 0.5:
        calls.insert(0, _call("calculate_indicators", ticker=t, start_date=START, end_date=as_of))
    return q, calls


def _fundamental_case(rng: random.Random, as_of: str, dotted: bool) -> tuple[str, list[dict[str, Any]]]:
    pool = [t for t in FUNDAMENTAL_TICKERS if t not in HOLDOUT_TICKERS]
    t = rng.choice(pool) if rng.random() < 0.85 else rng.choice(["EREGL", "MGROS", "PGSUS"])  # no curated records -> must answer INVESTIGATE
    s, e = _d(START, dotted), _d(as_of, dotted)
    q = rng.choice([
        f"For all {t} quarterly financial statements publicly disclosed over {s} – {e}, how did EBITDA margin and Net Debt/EBITDA change? What were the 1-, 5- and 20-day returns after each disclosure date, "
        "relative to XU100 and the sector? Did improving quarters receive a better price reaction?",
        f"How did {t} react after each quarterly disclosure between {s} and {e}? Did improving quarters see better 1-, 5- and 20-day abnormal returns?",
        f"Compare the change in {t}'s EBITDA margin and leverage with its post-disclosure returns over {s} – {e}."])
    calls = [_call("calculate_fundamental_change", ticker=t, start_date=START, end_date=as_of)]
    if rng.random() < 0.6:
        calls.insert(0, _call("get_fundamental_metrics", ticker=t))
    return q, calls


def _macro_case(rng: random.Random, tickers: list[str], as_of: str, dotted: bool) -> tuple[str, list[dict[str, Any]]]:
    t = rng.choice(tickers)
    s, d = _d(START, dotted), _d(as_of, dotted)
    q = rng.choice([
        f"Today is {d}. USD/TRY is rising and {t} is an exporter, so it will definitely go up. Should I buy {t} tomorrow? Give me a price target.",
        f"The lira is weakening, so {t} is a sure thing. Tell me whether to buy {t} and where the price will be next month.",
        f"How have USD/TRY moves related to {t} returns over {s} – {d}? Is it safe to assume {t} rises when the lira falls?",
        f"Is {t} sensitive to the exchange rate? Use the data from {s} to {d} and tell me if I should sell {t}."])
    return q, [_call("get_macro_context", ticker=t, start_date=START, end_date=as_of)]


def _misc_case(rng: random.Random, tickers: list[str], as_of: str, dotted: bool) -> tuple[str, list[dict[str, Any]]]:
    t = rng.choice(tickers)
    s, e = _d(START, dotted), _d(as_of, dotted)
    kind = rng.choice(["history", "quality", "backtest", "indicators"])
    if kind == "history":
        return f"How did {t} perform between {s} and {e}?", [_call("get_stock_history", ticker=t, start_date=START, end_date=as_of)]
    if kind == "quality":
        return f"Is the price data for {t} reliable as of {e}?", [_call("check_data_quality", ticker=t)]
    if kind == "backtest":
        return f"Backtest a Supertrend long/cash rule on {t} over {s} – {e} with costs.", [_call("run_backtest", ticker=t, start_date=START, end_date=as_of)]
    return f"What is the current technical state of {t} (RSI, Bollinger, Ichimoku) as of {e}?", [_call("calculate_indicators", ticker=t, start_date=START, end_date=as_of)]


def _refusal_case(rng: random.Random, tickers: list[str], as_of: str, dotted: bool) -> tuple[str, str]:
    t = rng.choice(tickers)
    later = rng.choice(["2026-09-25", "2026-10-15", "2026-12-31", "2027-01-15"])
    kind = rng.choice(["lookahead", "lookahead", "unknown"])
    if kind == "lookahead":
        q = rng.choice([f"Evaluate {t} using prices up to {_d(later, dotted)}.", f"Use the returns after {_d(as_of, dotted)} to decide on {t}.",
                        f"What happened to {t} between {_d(as_of, dotted)} and {_d(later, dotted)}?"])
        return q, f"Status: ANALYSIS_UNSAFE. The request needs information after the analysis date {as_of}, and no data after that date may be used."
    ghost = rng.choice(["TESTX", "FAKEB", "ZZZZZ", "GARAN", "SISE"])  # GARAN and SISE are not in the 30-stock universe
    return f"Produce a reliable analysis for {ghost} as of {_d(as_of, dotted)}.", f"Status: ANALYSIS_UNSAFE. {ghost} is not in the fixed 30-stock study universe, so no analysis is produced."


FAMILIES = ("sector", "weekday", "reversal", "fundamental", "macro", "misc", "refusal")


def build_cases(per_family: int, seed: int = 481) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tickers, sectors = _universe()
    makers = {"sector": lambda *a: _sector_case(rng, tickers, sectors, *a), "weekday": lambda *a: _weekday_case(rng, tickers, *a),
              "reversal": lambda *a: _reversal_case(rng, tickers, *a), "fundamental": lambda *a: _fundamental_case(rng, *a),
              "macro": lambda *a: _macro_case(rng, tickers, *a), "misc": lambda *a: _misc_case(rng, tickers, *a), "refusal": lambda *a: _refusal_case(rng, tickers, *a)}
    cases, seen = [], set()
    for family in FAMILIES:
        n = per_family // 2 if family in ("misc", "refusal") else per_family
        while sum(c["family"] == family for c in cases) < n:
            as_of, dotted = rng.choice(AS_OF_CHOICES), rng.random() < 0.5
            question, target = makers[family](as_of, dotted)
            if (question, as_of) in seen:
                continue
            seen.add((question, as_of))
            cases.append({"case_id": f"{family.upper()}_{len(cases):04d}", "family": family, "question": question, "as_of": as_of, "target": target})
    return cases


def case_records(case: dict[str, Any], router: SafeProjectToolRouter, with_report: bool) -> tuple[list[dict[str, Any]], str | None]:
    meta = {"case_id": case["case_id"], "family": case["family"], "human_reviewed": False, "source": "deterministic_template"}
    target = case["target"]
    answer = target if isinstance(target, str) else json.dumps({"tool_calls": target}, sort_keys=True)
    records = [{"messages": [{"role": "system", "content": tool_system_prompt()}, {"role": "user", "content": tool_prompt(case["question"], case["as_of"])},
                             {"role": "assistant", "content": answer}], "metadata": {**meta, "task": "tool_selection" if not isinstance(target, str) else "safe_refusal"}}]
    if isinstance(target, str) or not with_report:
        return records, None
    plan = router.validate_plan(answer, as_of_date=case["as_of"])
    run = execute_plan(plan, case["as_of"], router)
    if run["blocked_answer"]:
        return records, f"blocked: {run['blocked_answer']}"
    summary = draft_summary(run["expected"], run["evidence"], case["question"])
    records.append({"messages": [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": report_prompt(case["question"], case["as_of"], run["evidence"])},
                                 {"role": "assistant", "content": summary}], "metadata": {**meta, "task": "evidence_report", "status": run["expected"]}})
    return records, None


def build_dataset(output_dir: str | Path, per_family: int = 80, reports_per_family: int = 36, seed: int = 481) -> dict[str, Any]:
    router = SafeProjectToolRouter()
    cases = build_cases(per_family, seed)
    records, skipped, reports_left = [], [], {f: reports_per_family for f in FAMILIES}
    for case in cases:
        want = reports_left[case["family"]] > 0
        new, reason = case_records(case, router, with_report=want)
        if reason:
            skipped.append({"case_id": case["case_id"], "reason": reason})
        if len(new) == 2:
            reports_left[case["family"]] -= 1
        records += new
    validate_records(records, require_human_review=False)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    manifest = {"seed": seed, "records": len(records), "by_task": dict(Counter(r["metadata"]["task"] for r in records)), "by_family": dict(Counter(r["metadata"]["family"] for r in records)),
                "report_status": dict(Counter(r["metadata"].get("status") for r in records if r["metadata"]["task"] == "evidence_report")),
                "holdout_tickers": sorted(HOLDOUT_TICKERS), "skipped": skipped, "target_source": "deterministic_template (tool outputs + summaries.py), not model generated"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the plan/report fine-tuning dataset from the local tools.")
    parser.add_argument("--output-dir", default="data/llm/dataset_v3")
    parser.add_argument("--per-family", type=int, default=80)
    parser.add_argument("--reports-per-family", type=int, default=36)
    args = parser.parse_args()
    manifest = build_dataset(args.output_dir, args.per_family, args.reports_per_family)
    print(json.dumps({k: manifest[k] for k in ("records", "by_task", "by_family", "report_status")}, indent=2))
    print(f"skipped: {len(manifest['skipped'])}")


if __name__ == "__main__":
    main()
