"""Run the approved tools once and save their raw output, so every number in reports/*.md can be re-checked.

    python scripts/make_report_evidence.py        (writes reports/evidence/*.json)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ANALYSIS_AS_OF_DATE as A
from src.llm.tool_loop import SafeProjectToolRouter

OUT = Path("reports/evidence")
CALLS = {
    "sector_retail": {"tool_name": "rank_sector_relative_strength", "arguments": {"tickers": ["BIMAS", "MGROS", "MAVI"], "horizon_days": 20, "start_date": "2024-01-01", "end_date": A}},
    "sector_banking": {"tool_name": "rank_sector_relative_strength", "arguments": {"sector": "Banking", "horizon_days": 20, "start_date": "2024-01-01", "end_date": A}},
    "weekday_thyao": {"tool_name": "test_weekday_pattern", "arguments": {"ticker": "THYAO", "start_date": "2024-01-01", "end_date": "2025-12-31", "oos_start_date": "2026-01-01"}},
    "weekday_pattern_thyao": {"tool_name": "test_weekday_pattern", "arguments": {"ticker": "THYAO", "pattern": "mon_tue_wed_down", "start_date": "2024-01-01", "end_date": "2025-12-31", "oos_start_date": "2026-01-01"}},
    "reversal_asels": {"tool_name": "detect_reversal_events", "arguments": {"ticker": "ASELS", "start_date": "2024-01-01", "end_date": A}},
    "fundamental_metrics_tuprs": {"tool_name": "get_fundamental_metrics", "arguments": {"ticker": "TUPRS"}},
    "fundamental_change_tuprs": {"tool_name": "calculate_fundamental_change", "arguments": {"ticker": "TUPRS", "start_date": "2024-01-01", "end_date": A}},
    "macro_eregl": {"tool_name": "get_macro_context", "arguments": {"ticker": "EREGL", "start_date": "2024-01-01", "end_date": A}},
    "backtest_asels": {"tool_name": "run_backtest", "arguments": {"ticker": "ASELS", "start_date": "2024-01-01", "end_date": A}},
}


def write_new_family_evidence() -> None:
    """Event counts and regime splits for the families added after the first report (ASELS, cut at the analysis date)."""
    from src.data_pipeline import load_ticker_data
    from src.scenarios.technical_reversals import detect_reversal_events, summarize_reversals_by_regime

    cut = lambda f: f.loc[f.index <= A]
    events = detect_reversal_events(cut(load_ticker_data("ASELS")), market_df=cut(load_ticker_data("XU100")))
    summary = summarize_reversals_by_regime(events, target_horizon=5)
    payload = {"as_of_date": A, "ticker": "ASELS", "event_counts": events["Event_Name"].value_counts().to_dict(),
               "by_regime_5d": summary.to_dict(orient="records")}
    (OUT / "reversal_families_asels.json").write_text(json.dumps(payload, default=str, ensure_ascii=False, indent=1), encoding="utf-8")
    print("reversal_families_asels ok")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    router = SafeProjectToolRouter()
    for name, call in CALLS.items():
        try:
            tool, args = router.validate_call(call, as_of_date=A)
            result = router.execute(tool, args, as_of_date=A)
        except Exception as exc:  # keep the failure in the evidence instead of hiding it
            result = {"error": f"{type(exc).__name__}: {exc}"}
        (OUT / f"{name}.json").write_text(json.dumps({"call": call, "as_of_date": A, "result": result}, default=str, ensure_ascii=False, indent=1), encoding="utf-8")
        print(name, "ERROR" if "error" in result else "ok", len(json.dumps(result, default=str)))
    write_new_family_evidence()


if __name__ == "__main__":
    main()
