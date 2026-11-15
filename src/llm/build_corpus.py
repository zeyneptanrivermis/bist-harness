"""Build human-reviewable local-LLM candidates from deterministic project runs.

Example (first data acquisition):
    python -m src.llm.build_corpus --download --dates-per-ticker 5

The command downloads public OHLCV data only when ``--download`` is supplied.
It never calls a cloud text-generation service and it does not mark any record
as ready for training. A team member must review each English draft and change
``human_reviewed`` to true in a separate reviewed JSONL file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from src.config import ANALYSIS_END_DATE, ANALYSIS_START_DATE, DATA_DIR, MIN_REQUIRED_HISTORY_DAYS
from src.data_pipeline import (
    clean_and_preprocess_ohlcv,
    fetch_single_ticker,
    get_index_history,
    get_stock_history,
    load_ticker_data,
    save_ticker_data,
)
from src.fundamentals import CURATED_FUNDAMENTAL_RECORDS
from src.harness.harness import HarnessSession
from src.mcp_tools.models import AnalysisRequest
from src.mcp_tools.serialization import json_safe
from src.universe import BENCHMARK_SYMBOL, get_all_tickers, get_ticker_to_yahoo_map
from src.llm.training_data import DEFAULT_SYSTEM_PROMPT


SUPPORTED_SCENARIOS = (
    "technical_reversal",
    "weekday_pattern",
    "sector_laggard",
    "fundamental_reaction",
)
FUNDAMENTAL_TICKERS = frozenset(record["ticker"] for record in CURATED_FUNDAMENTAL_RECORDS)


@dataclass(frozen=True)
class CorpusBuildResult:
    records: list[dict[str, Any]]
    skipped: list[dict[str, str]]
    output_dir: Path


def select_common_cutoffs(price_data: dict[str, pd.DataFrame], count: int) -> list[str]:
    """Choose evenly spaced, shared trading cutoffs with indicator warm-up history."""
    frames = [frame for ticker, frame in price_data.items() if ticker != "XU100" and not frame.empty]
    if not frames:
        raise ValueError("No stock price data is available.")
    common = frames[0].index
    for frame in frames[1:]:
        common = common.intersection(frame.index)
    common = common[(common >= pd.Timestamp(ANALYSIS_START_DATE)) & (common <= pd.Timestamp(ANALYSIS_END_DATE))]
    if len(common) <= MIN_REQUIRED_HISTORY_DAYS:
        raise ValueError("Not enough common trading dates after the analysis start date.")
    # Candidate dates must retain enough history from each raw frame.
    valid = [date for date in common if all((frame.index <= date).sum() >= MIN_REQUIRED_HISTORY_DAYS for frame in frames)]
    if len(valid) < count:
        raise ValueError(f"Only {len(valid)} valid common cutoff dates are available; requested {count}.")
    positions = sorted({round(index * (len(valid) - 1) / (count - 1)) for index in range(count)}) if count > 1 else [len(valid) - 1]
    return [pd.Timestamp(valid[position]).strftime("%Y-%m-%d") for position in positions]


def _load_cached_universe(tickers: Iterable[str]) -> dict[str, pd.DataFrame]:
    data = {ticker: load_ticker_data(ticker) for ticker in tickers}
    data["XU100"] = load_ticker_data("XU100")
    return data


def _ensure_market_cache(tickers: Sequence[str], *, includes_sector_scenario: bool, force_refresh: bool) -> None:
    """Acquire only the data needed for this run; sector ranking needs all stocks."""
    required_tickers = get_all_tickers() if includes_sector_scenario else list(tickers)
    if not force_refresh:
        get_index_history()
        for ticker in required_tickers:
            get_stock_history(ticker)
        return

    raw_benchmark = fetch_single_ticker(BENCHMARK_SYMBOL)
    save_ticker_data("XU100", raw_benchmark, is_raw=True)
    save_ticker_data("XU100", clean_and_preprocess_ohlcv(raw_benchmark), is_raw=False)
    symbol_map = get_ticker_to_yahoo_map()
    for ticker in required_tickers:
        raw_frame = fetch_single_ticker(symbol_map[ticker])
        save_ticker_data(ticker, raw_frame, is_raw=True)
        save_ticker_data(ticker, clean_and_preprocess_ohlcv(raw_frame), is_raw=False)


def _question(ticker: str, scenario: str, as_of_date: str) -> str:
    questions = {
        "technical_reversal": f"Assess technical-reversal evidence for {ticker} as of {as_of_date}.",
        "weekday_pattern": f"Assess whether weekday effects for {ticker} remain stable after costs as of {as_of_date}.",
        "sector_laggard": f"Assess {ticker}'s sector-relative laggard evidence as of {as_of_date}.",
        "fundamental_reaction": f"Assess the point-in-time fundamental-reaction evidence for {ticker} as of {as_of_date}.",
    }
    return questions[scenario]


def _pct(value: Any) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value * 100:.1f}%"


def _fact_line(item: dict[str, Any]) -> str:
    """Turn one evidence item into a sentence containing only its own numbers."""
    eid, v = item["evidence_id"], item["values"]
    if item["source_tool"] == "check_data_quality":
        return f"{eid}: The data quality gate returned {v['status']}, so the analysis may proceed with caveats."
    if item["source_tool"] == "run_backtest":
        return (
            f"{eid}: The t+1, cost-aware backtest made {v['Total_Trades']} trades with total return {_pct(v['Total_Return'])} "
            f"versus a benchmark return of {_pct(v['Benchmark_Total_Return'])} (alpha {_pct(v['Alpha_vs_Benchmark'])}), "
            f"win rate {_pct(v['Win_Rate'])}, maximum drawdown {_pct(v['Max_Drawdown'])}, Sharpe ratio {v['Sharpe_Ratio']:.2f}."
        )
    if "results" in v:  # weekday_pattern
        sig = [r["Day_Name"] for r in v["results"] if r["Is_Statistically_Significant"]]
        best = max(v["results"], key=lambda r: r["Net_Mean_Return"])
        sig_text = f"Statistically significant days: {', '.join(sig)}." if sig else "No weekday is statistically significant."
        return (
            f"{eid}: {sig_text} The best weekday after costs is {best['Day_Name']} with a mean net return of "
            f"{_pct(best['Net_Mean_Return'])} over {best['Sample_Count']} samples (win rate {_pct(best['Win_Rate'])}, "
            f"p-value vs zero {best['P_Value_vs_Zero']:.3f})."
        )
    if "Lag_Score" in v:  # sector_laggard
        return (
            f"{eid}: {v['Ticker']} ranks {v['rank_in_sector']} in {v['Sector']} with a lag score of {v['Lag_Score']:.2f}; "
            f"its return is {_pct(v['Stock_Return'])} against a sector median of {_pct(v['Sector_Median_Return'])}."
        )
    events = v.get("latest_events") or []
    n = v.get("event_count", len(events))
    text = f"{eid}: {n} event{'s were' if n != 1 else ' was'} detected before the cutoff."
    if events and "Event_Name" in events[0]:  # technical_reversal
        last = events[-1]
        done = [e for e in events if e.get("Fwd_1d_Net_Return") is not None]
        text += f" The latest is {last['Event_Name']} ({last['Event_Direction_Bias']} bias) on {last['Event_Date'][:10]} with RSI {last['RSI_14']:.1f}."
        if done:
            text += f" Among the last {len(done)} events with a known 1-day outcome, {sum(1 for e in done if e['Fwd_1d_Hit'] == 1)} hit; mean net 1-day return was {_pct(sum(e['Fwd_1d_Net_Return'] for e in done) / len(done))}."
    elif events:  # fundamental_reaction
        last = events[-1]
        text += (
            f" The latest is {last['Period']} (disclosed {last['KAP_Disclosure_Date']}, "
            f"regime {last['Fundamental_Regime']}): 5-day cumulative abnormal return {_pct(last['Fwd_5d_CAR'])}, "
            f"20-day {_pct(last['Fwd_20d_CAR'])}; {v.get('strong_improvement_count', 0)} strong-improvement events."
        )
    return text


def _draft_summary(action: str, evidence: list[dict[str, Any]]) -> str:
    evidence_ids = ", ".join(item["evidence_id"] for item in evidence)
    facts = "\n".join(_fact_line(item) for item in evidence)
    return (
        f"Status: {action}.\n"
        f"{facts}\n"
        f"Evidence IDs: {evidence_ids}.\n"
        "Limitation: This is an educational, evidence-bound research summary; it is not investment advice."
    )


def _review_flags(action: str | None, quality: str | None, evidence: list[dict[str, Any]]) -> list[str]:
    """Hints for human reviewers; they never change the draft."""
    flags = []
    if quality == "INVESTIGATE":
        flags.append("quality gate INVESTIGATE: the summary must state the data caveat")
    for item in evidence:
        v = item["values"]
        if "Lag_Score" in v and action == "POTENTIAL_CATCH_UP_CANDIDATE" and v["Lag_Score"] == 0:
            flags.append("catch-up status but lag score is 0: verify the status or reject the row")
        if item["source_tool"] == "run_backtest" and v["Alpha_vs_Benchmark"] < 0:
            flags.append("backtest trails the benchmark: do not imply it supports the signal")
    return flags


def _candidate_record(session: HarnessSession) -> dict[str, Any]:
    evidence = json_safe(session.evidence)
    question = _question(session.request.ticker, session.request.scenario, session.request.as_of_date)
    user_prompt = (
        f"Question: {question}\n"
        f"As-of date: {session.request.as_of_date}\n"
        f"Committed evidence (JSON):\n{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}\n\n"
        "Write a concise English research summary. Cite evidence IDs, do not add facts, and do not provide BUY/SELL advice."
    )
    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": _draft_summary(session.action or "WATCH", evidence)},
        ],
        "metadata": {
            "case_id": f"{session.request.ticker}_{session.request.scenario}_{session.request.as_of_date.replace('-', '')}",
            "task": "evidence_report",
            "human_reviewed": False,
            "ticker": session.request.ticker,
            "scenario": session.request.scenario,
            "as_of_date": session.request.as_of_date,
            "quality_status": session.quality.status if session.quality else None,
            "action": session.action,
            "source": "local_deterministic_harness",
            "auto_flags": _review_flags(session.action, session.quality.status if session.quality else None, evidence),
        },
    }


def _run_case(
    *, ticker: str, scenario: str, as_of_date: str, price_data: dict[str, pd.DataFrame]
) -> tuple[dict[str, Any] | None, str | None]:
    request = AnalysisRequest(ticker=ticker, scenario=scenario, as_of_date=as_of_date, require_human_review=False)
    session = HarnessSession(request)
    session.select_universe()
    session.load_data(price_data[ticker], price_data["XU100"])
    quality = session.check_quality()
    if quality.status in {"ANALYSIS_UNSAFE", "HARD_BLOCK"}:
        return None, f"quality gate returned {quality.status}"
    try:
        session.calculate()
        session.run_scenario(universe_data={key: value for key, value in price_data.items() if key != "XU100"})
        session.run_backtest()
        session.build_evidence()
    except Exception as exc:  # The manifest preserves failures for team review.
        return None, f"{type(exc).__name__}: {exc}"
    return _candidate_record(session), None


def build_corpus(
    *,
    price_data: dict[str, pd.DataFrame],
    tickers: Sequence[str],
    scenarios: Sequence[str],
    as_of_dates: Sequence[str],
    output_dir: str | Path,
    reviewers: Sequence[str] = (),
) -> CorpusBuildResult:
    """Run requested deterministic cases and save unreviewed JSONL/CSV candidates."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for as_of_date in as_of_dates:
        for ticker in tickers:
            for scenario in scenarios:
                if scenario == "fundamental_reaction" and ticker not in FUNDAMENTAL_TICKERS:
                    skipped.append({"ticker": ticker, "scenario": scenario, "as_of_date": as_of_date, "reason": "no curated fundamental records"})
                    continue
                record, reason = _run_case(ticker=ticker, scenario=scenario, as_of_date=as_of_date, price_data=price_data)
                if record is None:
                    skipped.append({"ticker": ticker, "scenario": scenario, "as_of_date": as_of_date, "reason": reason or "unknown"})
                else:
                    records.append(record)

    jsonl_path = output_path / "candidates.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""), encoding="utf-8")
    # reviewed_summary starts as the draft so reviewers edit rather than write; human_reviewed stays false.
    review_rows = [
        {
            **{k: v for k, v in record["metadata"].items() if k != "auto_flags"},
            "assigned_to": reviewers[i % len(reviewers)] if reviewers else "",
            "user_prompt": record["messages"][1]["content"],
            "draft_summary": record["messages"][2]["content"],
            "reviewed_summary": record["messages"][2]["content"],
            "reviewer": "",
            "review_notes": "; ".join(record["metadata"].get("auto_flags", [])),
        }
        for i, record in enumerate(records)
    ]
    pd.DataFrame(review_rows).to_csv(output_path / "candidates_review.csv", index=False)
    manifest = {
        "candidate_count": len(records),
        "skipped_count": len(skipped),
        "as_of_dates": list(as_of_dates),
        "tickers": list(tickers),
        "scenarios": list(scenarios),
        "review_requirement": "All candidates are unreviewed. Copy approved, edited records to reviewed_records.jsonl with human_reviewed=true.",
        "skipped": skipped,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return CorpusBuildResult(records=records, skipped=skipped, output_dir=output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create human-reviewable local-LLM corpus candidates from deterministic harness runs.")
    parser.add_argument("--download", action="store_true", help="Download/cache public OHLCV data before building. Required for an empty data directory.")
    parser.add_argument("--force-refresh", action="store_true", help="Refresh already cached OHLCV data; requires --download.")
    parser.add_argument("--output-dir", default=str(DATA_DIR / "llm" / "candidates"), help="Ignored corpus-output directory.")
    parser.add_argument("--dates-per-ticker", type=int, default=3, help="Number of shared, evenly spaced trading cutoffs.")
    parser.add_argument("--as-of-dates", nargs="*", help="Explicit YYYY-MM-DD cutoffs; overrides automatic selection.")
    parser.add_argument("--tickers", nargs="*", default=get_all_tickers(), help="Subset of fixed-universe symbols to include.")
    parser.add_argument("--reviewers", nargs="*", default=[], help="Group member names; rows are assigned round-robin in an assigned_to column.")
    parser.add_argument("--scenarios", nargs="*", default=list(SUPPORTED_SCENARIOS), choices=SUPPORTED_SCENARIOS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.force_refresh and not args.download:
        raise SystemExit("--force-refresh requires --download.")
    unknown = sorted(set(args.tickers) - set(get_all_tickers()))
    if unknown:
        raise SystemExit(f"Unknown ticker(s): {', '.join(unknown)}")
    data_tickers = get_all_tickers() if "sector_laggard" in args.scenarios else args.tickers
    if args.download:
        _ensure_market_cache(args.tickers, includes_sector_scenario="sector_laggard" in args.scenarios, force_refresh=args.force_refresh)
    try:
        price_data = _load_cached_universe(data_tickers)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\nRun again with --download to acquire the required public OHLCV cache.") from exc
    as_of_dates = args.as_of_dates or select_common_cutoffs(price_data, args.dates_per_ticker)
    result = build_corpus(
        price_data=price_data,
        tickers=args.tickers,
        scenarios=args.scenarios,
        as_of_dates=as_of_dates,
        output_dir=args.output_dir,
        reviewers=args.reviewers,
    )
    print(f"Created {len(result.records)} unreviewed candidates in {result.output_dir}")
    print(f"Skipped {len(result.skipped)} cases; inspect manifest.json before review.")


if __name__ == "__main__":
    main()
