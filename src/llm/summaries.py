"""Deterministic English summaries of tool evidence.

They are (1) the training targets for the report task, (2) the fallback shown when the model's
own summary breaks a rule, and (3) the reference the guard compares against. They only restate
numbers that are in the evidence.
"""

from __future__ import annotations

import re
from typing import Any

LIMITATION = "Limitation: This is an educational, evidence-bound summary; it is not investment advice."
_ADVICE_ASK = re.compile(r"\b(buy|sell|price target|target price|should i|definitely|certainly|guaranteed|will go up|will rise)\b", re.IGNORECASE)


def advice_requested(question: str) -> bool:
    return bool(_ADVICE_ASK.search(question))


def _p(v: Any) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _n(v: Any, d: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def _warn(values: dict[str, Any]) -> str:
    return "".join(f" Warning: {w}" for w in values.get("warnings", []))


def fact_line(item: dict[str, Any]) -> str:
    eid, v = item["evidence_id"], item["values"]
    tool = v.get("tool")
    if not v.get("available", True):
        return f"{eid}: {tool} returned no usable data.{_warn(v)}"
    if tool == "check_data_quality":
        gate = ", ".join(f"{t} {s}" for t, s in v["gate"].items()) if "gate" in v else f"{v['ticker']} {v['gate_status']}"
        return f"{eid}: The data-quality gate ran before any calculation and returned {gate}."
    if tool == "rank_sector_relative_strength":
        rank = ", ".join(f"{r['ticker']} {_p(r['return'])} (gap {_p(r['lag_gap'])})" for r in v["ranking"])
        h = v["historical_catch_up"]
        line = (f"{eid}: Over the last {v['horizon_days']} trading days to {v['window_end']} the peer median return is {_p(v['peer_median_return'])}; {rank}. "
                f"The most lagging stock is {v['most_lagging']}. Since {v['period']['start']}, lags of at least {_p(v['lag_threshold'])} closed within 5, 10 and 20 days in "
                f"{_p(h['5d']['hit_rate'])}, {_p(h['10d']['hit_rate'])} and {_p(h['20d']['hit_rate'])} of cases ({h['10d']['events']} events, {h['10d']['independent_episodes']} independent episodes); "
                f"{_p(h['10d']['false_positive_rate'])} kept underperforming at 10 days (median outperformance {_p(h['10d']['median_outperformance'])}).")
        return line + _warn(v)
    if tool == "test_weekday_pattern":
        def block(label: str, p: dict[str, Any]) -> str:
            if "days" in p:
                best = max(p["days"], key=lambda d: d["mean"])
                return f"{label}: best weekday {best['day']} mean {_p(best['mean'])} (p {_n(best['p_value'], 3)}), unconditional mean {_p(p['unconditional_mean'])}."
            if "thu_fri_both_up" in p:
                if not p["occurrences"]:
                    return f"{label}: the pattern never occurred."
                return (f"{label}: {p['occurrences']} occurrences in {p['weeks']} weeks; Thursday and Friday were both up in {p['thu_fri_both_up']}; mean Thursday {_p(p['thu']['mean'])}, "
                        f"mean Friday {_p(p['fri']['mean'])} against {_p(p['unconditional_thu_mean'])} and {_p(p['unconditional_fri_mean'])} in other weeks (p {_n(p['p_value_vs_other_weeks'], 3)}).")
            return f"{label}: {p['occurrences']} occurrences."
        text = f"{eid}: " + block(f"In-sample {v['period']['start']} to {v['period']['end']}", v["in_sample"])
        if "out_of_sample" in v:
            text += " " + block(f"Out-of-sample from {v['out_of_sample']['period']['start']}", v["out_of_sample"])
        m = v["multiple_testing"]
        return text + f" Multiple-testing risk: {m['tests_run']} test(s) run, Bonferroni alpha {_n(m['bonferroni_alpha'], 3)}." + _warn(v)
    if tool == "detect_reversal_events":
        b = v["by_horizon"]
        combos = "; ".join(f"{c['condition'].split('. ', 1)[1]}: {c['n']} cases, 5-day hit rate {_p(c['hit_rate_5d'])}" for c in v["combined_conditions"] if c["n"])
        return (f"{eid}: {v['event_count']} {v['event']} events from {v['period']['start']} to {v['period']['end']}. Mean returns after 1, 3, 5 and 10 days: "
                f"{_p(b['1d']['mean'])}, {_p(b['3d']['mean'])}, {_p(b['5d']['mean'])}, {_p(b['10d']['mean'])}; 5-day hit rate {_p(b['5d']['hit_rate'])}. {combos}." + _warn(v))
    if tool == "calculate_fundamental_change":
        first, last = v["quarters"][0], v["quarters"][-1]
        g = v["improving_vs_other"]

        def cars(label: str) -> str:
            return f"{label}: {_p(g[label]['mean_car_xu100_1d'])}, {_p(g[label]['mean_car_xu100_5d'])}, {_p(g[label]['mean_car_xu100_20d'])} after {g[label]['quarters']} quarters"

        return (f"{eid}: {len(v['quarters'])} disclosed quarters. EBITDA margin moved from {_p(first['ebitda_margin'])} ({first['period']}) to {_p(last['ebitda_margin'])} ({last['period']}, disclosed {last['disclosed']}); "
                f"net debt to annualized EBITDA from {_n(first['net_debt_to_annualized_ebitda'])} to {_n(last['net_debt_to_annualized_ebitda'])}. Latest quarter returns after 1 and 5 days: {_p(last['ret_1d'])} and {_p(last['ret_5d'])} "
                f"(abnormal versus XU100 {_p(last['car_xu100_1d'])} and {_p(last['car_xu100_5d'])}); its 20-day return is {_p(last['ret_20d']) if last['ret_20d'] is not None else 'not complete yet'}. "
                f"Mean abnormal return versus XU100 after 1, 5 and 20 days: {cars('improving')}; {cars('other')}." + _warn(v))
    if tool == "get_fundamental_metrics":
        last = v["quarters"][-1]
        return f"{eid}: Latest public quarter {last['period']} (disclosed {last['disclosed']}): EBITDA margin {_p(last['ebitda_margin'])}, net debt to annualized EBITDA {_n(last['net_debt_to_annualized_ebitda'])}." + _warn(v)
    if tool == "get_macro_context":
        series = "; ".join(f"{k} last {_n(s['last'], 4)} ({_p(s['change_pct'])} over the period, published {s['last_published']})" for k, s in v["series"].items()) or "no macro series were available"
        text = f"{eid}: Macro context to {v['period']['end']}: {series}."
        if "fx_relationship" in v:
            f = v["fx_relationship"]
            text += (f" For {f['ticker']} over {f['weeks']} weeks, weekly correlation with USD/TRY is {_n(f['correlation_weekly'])}; mean weekly return is {_p(f['mean_stock_return_when_usdtry_up'])} "
                     f"in {f['weeks_up']} weeks when USD/TRY rose and {_p(f['mean_stock_return_when_usdtry_down'])} in {f['weeks_down']} weeks when it did not.")
        return text + _warn(v)
    if tool == "run_backtest":
        m = v["metrics"]
        return (f"{eid}: The t+1, cost-aware {v['strategy']} backtest for {v['ticker']} made {_n(m['Total_Trades'], 0)} trades with total return {_p(m['Total_Return'])} versus a benchmark return of "
                f"{_p(m['Benchmark_Total_Return'])}, Sharpe ratio {_n(m['Sharpe_Ratio'])}, maximum drawdown {_p(m['Max_Drawdown'])}.")
    if tool == "calculate_indicators":
        x = v["latest"]
        return f"{eid}: On {v['latest_date']} {v['ticker']} closed at {_n(x['Close'])} with RSI {_n(x['RSI_14'], 1)}, lower Bollinger band {_n(x['BB_Lower'])} and Ichimoku state {x['Ichimoku_State']}."
    if tool == "get_stock_history":
        return f"{eid}: {v['ticker']} has {v['rows']} sessions from {v['first_date']} to {v['last_date']}, last close {_n(v['last_close'])}, period return {_p(v['period_return'])}."
    return f"{eid}: {tool} evidence is available."


def draft_summary(status: str, evidence: list[dict[str, Any]], question: str = "") -> str:
    lines = "\n".join(fact_line(item) for item in evidence)
    advice = ("I cannot give investment advice, a trading directive or a price target. Any claim that the price will definitely move is unverified; only the evidence above supports conclusions.\n"
              if advice_requested(question) else "")
    ids = ", ".join(item["evidence_id"] for item in evidence)
    return f"Status: {status}.\n{lines}\n{advice}Evidence IDs: {ids}.\n{LIMITATION}"
