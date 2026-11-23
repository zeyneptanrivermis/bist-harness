"""Date-range research tools named as in the course tool table (Section 22).

Each function returns a small JSON-safe dict (numbers, sample sizes, warnings and a
``suggested_status`` from the finite action space). They never read data after ``as_of``
and never raise for missing data: they return ``available: False`` so the agent can answer
INVESTIGATE instead of crashing. Forward windows that are not complete by the end date are
excluded and counted.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.engine import run_backtest
from src.data_pipeline import load_ticker_data
from src.fundamentals import get_fundamental_metrics
from src.harness.quality import check_data_quality
from src.indicators import calculate_all_indicators
from src.macro import ALL_SERIES, FX_SYMBOLS, load_macro_series
from src.mcp_tools.serialization import json_safe
from src.scenarios.fundamental_reaction import align_financial_disclosure_dates, calculate_quarterly_changes
from src.scenarios.technical_reversals import compare_single_vs_combined_reversals, detect_reversal_events
from src.universe import get_universe_df
from src.config import DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE_RATE

ROUNDTRIP_COST = 2.0 * (DEFAULT_COMMISSION_RATE + DEFAULT_SLIPPAGE_RATE)
WEEK_SEQUENCES = 32  # the Mon-Wed-down pattern is one of 2^5 possible Mon-Fri up/down sequences; correct for all of them

SMALL_SAMPLE = 30  # below this many events a hit rate is anecdotal
MIN_EPISODES = 10  # below this many independent episodes nothing may be labelled a candidate
EVENT_FAMILIES = ("BB_LOWER_TOUCH", "BB_MIDDLE_TEST", "BB_UPPER_TOUCH", "RSI_OVERSOLD", "RSI_OVERBOUGHT",
                  "SUPERTREND_BULLISH_FLIP", "SUPERTREND_BEARISH_FLIP", "KAMA_CROSS_UP", "ICHIMOKU_CLOUD_BREAKOUT")
WEEKLY_PATTERNS = ("weekday", "mon_tue_wed_down", "consecutive_down", "consecutive_up", "large_down_day")


def _r(value: Any, digits: int = 4) -> Any:
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else round(float(value), digits)


def _end(as_of: str, end_date: Optional[str]) -> str:
    return min(end_date or as_of, as_of)


def _px(ticker: str, end: str) -> pd.DataFrame:
    frame = load_ticker_data(ticker)
    return frame.loc[frame.index <= pd.Timestamp(end)]


def _base(tool: str, as_of: str, start: Optional[str], end: str, **extra: Any) -> dict[str, Any]:
    return {"tool": tool, "available": True, "as_of_date": as_of, "period": {"start": start, "end": end}, "warnings": [], **extra}


def _unavailable(tool: str, as_of: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"tool": tool, "available": False, "as_of_date": as_of, "warnings": [reason], "suggested_status": "INVESTIGATE", **extra}


def _stat(values: pd.Series) -> dict[str, Any]:
    values = values.dropna()
    return {"n": int(len(values)), "mean": _r(values.mean()) if len(values) else None, "median": _r(values.median()) if len(values) else None}


# --- reversal events (Scenario 3) -------------------------------------------------------------
def detect_reversal_events_summary(ticker: str, as_of: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                                   event: str = "BB_LOWER_TOUCH") -> dict[str, Any]:
    end = _end(as_of, end_date)
    price = _px(ticker, end)
    events = detect_reversal_events(price, market_df=_px("XU100", end))
    if start_date:
        events = events[events["Event_Date"] >= pd.Timestamp(start_date)]
    base = events[events["Event_Name"] == event]
    out = _base("detect_reversal_events", as_of, start_date, end, ticker=ticker, event=event, event_count=int(len(base)))
    complete = base.dropna(subset=["Fwd_5d_Hit"])
    out["excluded_incomplete_5d"] = int(len(base) - len(complete))
    if out["excluded_incomplete_5d"]:
        out["warnings"].append(f"{out['excluded_incomplete_5d']} touch(es) excluded: the 5-day forward window is not complete by {end}.")
    if len(base) < SMALL_SAMPLE:
        out["warnings"].append(f"Small sample: {len(base)} events (fewer than {SMALL_SAMPLE}).")
    out["by_horizon"] = {f"{h}d": {**_stat(base[f"Fwd_{h}d_Return"]), "hit_rate": _r(base[f"Fwd_{h}d_Hit"].dropna().mean()) if base[f"Fwd_{h}d_Hit"].notna().any() else None}
                         for h in (1, 3, 5, 10)}
    comparison = compare_single_vs_combined_reversals(events, base_event=event, target_horizon=5)
    out["combined_conditions"] = [{"condition": r["Condition"], "n": int(r["Sample_Count"]), "hit_rate_5d": _r(r["Win_Rate"]), "mean_5d": _r(r["Mean_Return"])}
                                  for r in json_safe(comparison)] if not comparison.empty else []
    thin = [f"{c['condition'].split('. ', 1)[1]} (n={c['n']})" for c in out["combined_conditions"] if c["n"] < MIN_EPISODES]
    if thin:
        out["warnings"].append(f"Small sample: fewer than {MIN_EPISODES} events in " + "; ".join(thin) + ". Their hit rates are anecdotal.")
    hit5 = out["by_horizon"]["5d"]["hit_rate"]
    out["suggested_status"] = "INVESTIGATE" if len(complete) < 10 else "TECHNICAL_REVERSAL_CANDIDATE" if (hit5 or 0) >= 0.6 else "WATCH"
    return out


def calculate_indicators_summary(ticker: str, as_of: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict[str, Any]:
    end = _end(as_of, end_date)
    last = calculate_all_indicators(_px(ticker, end)).iloc[-1]
    keys = ["Close", "RSI_14", "BB_Lower", "BB_Middle", "BB_Upper", "KAMA_10", "Supertrend_Direction", "Ichimoku_State", "KAMA_Slope_State"]
    out = _base("calculate_indicators", as_of, start_date, end, ticker=ticker, latest_date=str(last.name.date()),
                latest={k: (_r(last[k]) if isinstance(last[k], (float, np.floating)) else json_safe(last[k])) for k in keys})
    out["suggested_status"] = "WATCH"
    return out


# --- weekday / multi-day patterns (Scenario 2) ---------------------------------------------------
def _weekday_period(df: pd.DataFrame, pattern: str) -> dict[str, Any]:
    ret = df["Adj Close"].pct_change().dropna()
    if pattern == "weekday":
        from src.scenarios.weekday_patterns import test_weekday_pattern

        table = test_weekday_pattern(df)
        return {"days": [{"day": r["Day_Name"], "n": int(r["Sample_Count"]), "mean": _r(r["Mean_Return"]), "median": _r(r["Median_Return"]),
                          "p_value": _r(r["P_Value_vs_Zero"]), "net_mean": _r(r["Net_Mean_Return"])} for r in json_safe(table)],
                "unconditional_mean": _r(ret.mean()), "tests_run": 5}
    if pattern == "mon_tue_wed_down":
        week = ret.groupby([ret.index.isocalendar().year, ret.index.isocalendar().week])
        rows = []
        for _, w in week:
            by_day = {d.dayofweek: v for d, v in w.items()}
            if all(k in by_day for k in range(5)):
                rows.append({"cond": all(by_day[k] < 0 for k in (0, 1, 2)), "thu": by_day[3], "fri": by_day[4]})
        t = pd.DataFrame(rows)
        if t.empty or t["cond"].sum() == 0:
            return {"occurrences": 0, "weeks": int(len(t)), "tests_run": WEEK_SEQUENCES}
        hit, rest = t[t["cond"]], t[~t["cond"]]
        both_up = int(((hit["thu"] > 0) & (hit["fri"] > 0)).sum())
        p = stats.ttest_ind(hit["thu"] + hit["fri"], rest["thu"] + rest["fri"], equal_var=False).pvalue if len(hit) >= 5 and len(rest) > 1 else None
        # buy at Wednesday's close (when the pattern is known), sell at Friday's close: one round trip of commission + slippage
        wed_fri = (1 + hit["thu"]) * (1 + hit["fri"]) - 1
        return {"occurrences": int(len(hit)), "weeks": int(len(t)), "thu_fri_both_up": both_up, "thu": _stat(hit["thu"]), "fri": _stat(hit["fri"]),
                "unconditional_thu_mean": _r(rest["thu"].mean()), "unconditional_fri_mean": _r(rest["fri"].mean()),
                "wed_fri_mean": _r(wed_fri.mean()), "wed_fri_net_mean": _r(wed_fri.mean() - ROUNDTRIP_COST), "roundtrip_cost": ROUNDTRIP_COST,
                "p_value_vs_other_weeks": _r(p) if p is not None else None, "tests_run": WEEK_SEQUENCES}
    from src.scenarios.weekday_patterns import detect_multiday_patterns, summarize_pattern_performance

    events = detect_multiday_patterns(df, pattern_type=pattern, forward_horizons=[1, 2, 5])
    summary = summarize_pattern_performance(events, [1, 2, 5]) if not events.empty else pd.DataFrame()
    return {"occurrences": int(len(events)), "horizons": [{"h": int(r["Horizon_Days"]), "n": int(r["Total_Events"]), "mean": _r(r["Mean_Return"]),
            "median": _r(r["Median_Return"]), "net_mean": _r(r["Net_Mean_Return"]), "p_value": _r(r["P_Value"])} for r in json_safe(summary)], "tests_run": 1}


def _pattern_significant(period: dict[str, Any], alpha: float) -> Optional[bool]:
    if "days" in period:
        return any(d["p_value"] is not None and d["p_value"] < alpha for d in period["days"])
    p = period.get("p_value_vs_other_weeks")
    if p is None and period.get("horizons"):
        p = min((h["p_value"] for h in period["horizons"] if h["p_value"] is not None), default=None)
    return None if p is None else p < alpha


def test_weekday_pattern_summary(ticker: str, as_of: str, pattern: str = "weekday", start_date: Optional[str] = None,
                                 end_date: Optional[str] = None, oos_start_date: Optional[str] = None) -> dict[str, Any]:
    end = _end(as_of, end_date)
    frame = _px(ticker, as_of)
    in_end = end if not oos_start_date else min(end, str((pd.Timestamp(oos_start_date) - pd.Timedelta(days=1)).date()))
    sample = frame.loc[(frame.index >= pd.Timestamp(start_date or "1900-01-01")) & (frame.index <= pd.Timestamp(in_end))]
    out = _base("test_weekday_pattern", as_of, start_date, in_end, ticker=ticker, pattern=pattern, in_sample=_weekday_period(sample, pattern))
    tests = out["in_sample"]["tests_run"]
    alpha = 0.05 / tests
    out["multiple_testing"] = {"tests_run": tests, "bonferroni_alpha": _r(alpha), "note": "Searching many weekday/pattern combinations produces false positives by chance."}
    sig_in = _pattern_significant(out["in_sample"], alpha)
    sig_out = None
    if oos_start_date:
        oos = frame.loc[frame.index >= pd.Timestamp(oos_start_date)]
        out["out_of_sample"] = {"period": {"start": oos_start_date, "end": as_of}, **_weekday_period(oos, pattern)}
        sig_out = _pattern_significant(out["out_of_sample"], alpha)
    n_in = out["in_sample"].get("occurrences", 0)
    if n_in and n_in < 10:
        out["warnings"].append(f"Small sample: the pattern occurred only {n_in} times in-sample.")
    if oos_start_date and sig_in and not sig_out:
        out["warnings"].append("The in-sample pattern is not significant out of sample: it is unstable.")
    out["suggested_status"] = "INVESTIGATE" if (n_in and n_in < 10) else "REJECT_SIGNAL" if (not sig_in or (oos_start_date and not sig_out)) else "INVESTIGATE"
    return out


# --- sector relative strength (Scenario 1) --------------------------------------------------------
def resolve_peers(tickers: Optional[list[str]], sector: Optional[str]) -> list[str]:
    universe = get_universe_df()
    if tickers:
        return [t for t in tickers if t in set(universe["ticker"])]
    key = (sector or "").lower().rstrip("s")
    return universe[universe["sector"].str.lower().str.contains(key, regex=False)]["ticker"].tolist() if key else []


def rank_sector_relative_strength_summary(as_of: str, tickers: Optional[list[str]] = None, sector: Optional[str] = None, horizon_days: int = 20,
                                          start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict[str, Any]:
    end = _end(as_of, end_date)
    peers = resolve_peers(tickers, sector)
    if len(peers) < 2:
        return _unavailable("rank_sector_relative_strength", as_of, f"Need at least 2 peers in the universe; got {peers}.", peers=peers)
    px = pd.DataFrame({t: _px(t, end)["Adj Close"] for t in peers}).dropna()
    ret = px.pct_change(horizon_days)
    lag = ret.sub(ret.median(axis=1), axis=0)
    now = lag.iloc[-1].sort_values()
    worst = now.index[0]
    out = _base("rank_sector_relative_strength", as_of, start_date, end, peers=peers, horizon_days=horizon_days, window_end=str(px.index[-1].date()),
                peer_median_return=_r(ret.iloc[-1].median()),
                ranking=[{"ticker": t, "return": _r(ret.iloc[-1][t]), "lag_gap": _r(v), "rank": i + 1} for i, (t, v) in enumerate(now.items())], most_lagging=worst)
    thr = min(0.03, abs(now.iloc[0]) / 2)  # "similar" lag: at least half as deep as the current one, never shallower than 3%
    start = pd.Timestamp(start_date or px.index[0])
    out["lag_threshold"] = _r(thr)
    stats_by_h: dict[str, Any] = {}
    for h in (5, 10, 20):
        fwd = px.shift(-(1 + h)) / px.shift(-1) - 1  # entry at t+1, exit h sessions later; NaN while the window is incomplete
        hits, episodes, flagged = [], 0, 0
        for t in peers:
            mask = (lag[t] <= -thr) & (lag.index >= start)
            outperf = fwd[t] - fwd.drop(columns=[t]).median(axis=1)
            flagged += int(mask.sum())
            hits.append(outperf[mask].dropna())
            last_i = -10**9
            for i in np.flatnonzero(mask.to_numpy() & outperf.notna().to_numpy()):
                if i - last_i > 20:
                    episodes += 1
                    last_i = i
        all_hits = pd.concat(hits) if hits else pd.Series(dtype=float)
        n = int(len(all_hits))
        stats_by_h[f"{h}d"] = {"events": n, "excluded_incomplete": flagged - n, "independent_episodes": episodes,
                               "hit_rate": _r((all_hits > 0).mean()) if n else None, "false_positive_rate": _r((all_hits <= 0).mean()) if n else None,
                               "median_outperformance": _r(all_hits.median()) if n else None}
    out["historical_catch_up"] = stats_by_h
    has_fund = worst in set(_fund_tickers())
    out["most_lagging_has_fundamental_records"] = has_fund
    if not has_fund:
        out["warnings"].append(f"No fundamental records for {worst}: a structural lag cannot be separated from a temporary one.")
    out["warnings"].append("Macro evidence is not part of this tool; the get_macro_context evidence shows whether FX or rates may explain the lag.")
    h10 = stats_by_h["10d"]
    if h10["independent_episodes"] < MIN_EPISODES:
        out["warnings"].append(f"Small sample: {h10['independent_episodes']} independent episodes.")
    ok = now.iloc[0] <= -0.03 and (h10["hit_rate"] or 0) >= 0.55 and h10["independent_episodes"] >= MIN_EPISODES and has_fund
    out["suggested_status"] = "POTENTIAL_CATCH_UP_CANDIDATE" if ok else "INVESTIGATE" if (h10["independent_episodes"] < MIN_EPISODES or not has_fund) else "WATCH"
    return out


def _fund_tickers() -> list[str]:
    from src.llm.build_corpus import FUNDAMENTAL_TICKERS

    return list(FUNDAMENTAL_TICKERS)


# --- fundamentals (Scenario 4) -----------------------------------------------------------------
def _funds(ticker: str, as_of: str) -> Optional[pd.DataFrame]:
    try:
        frame = get_fundamental_metrics(ticker)
    except ValueError:
        return None
    frame = frame[frame["kap_disclosure_date"] <= pd.Timestamp(as_of)].copy()  # public by the analysis date
    frame["net_debt_to_annualized_ebitda"] = frame["net_debt"] / (4 * frame["ebitda"])
    return frame


def get_fundamental_metrics_summary(ticker: str, as_of: str) -> dict[str, Any]:
    frame = _funds(ticker, as_of)
    if frame is None or frame.empty:
        return _unavailable("get_fundamental_metrics", as_of, f"No point-in-time fundamental records for {ticker}.", ticker=ticker)
    out = _base("get_fundamental_metrics", as_of, None, as_of, ticker=ticker, quarters=[
        {"period": r["period"], "disclosed": str(r["kap_disclosure_date"].date()), "ebitda_margin": _r(r["ebitda_margin"]), "net_debt_to_annualized_ebitda": _r(r["net_debt_to_annualized_ebitda"], 2),
         "roe": _r(r["roe"])} for _, r in frame.iterrows()])
    out["warnings"].append("Curated dataset: source identifiers are required before treating it as verified KAP evidence.")
    out["suggested_status"] = "WATCH"
    return out


def _sector_excess(ticker: str, entry: pd.Timestamp, h: int, end: str) -> Optional[float]:
    universe = get_universe_df().set_index("ticker")["sector"]
    peers = [t for t in universe[universe == universe.get(ticker)].index if t != ticker]
    rets = []
    for p in peers:
        px = _px(p, end)["Adj Close"]
        i = px.index.get_indexer([entry])[0]
        if i != -1 and i + h < len(px):
            rets.append(px.iloc[i + h] / px.iloc[i] - 1)
    return float(np.median(rets)) if rets else None


def calculate_fundamental_change_summary(ticker: str, as_of: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict[str, Any]:
    end = _end(as_of, end_date)
    frame = _funds(ticker, end)
    if frame is None or len(frame) < 2:
        return _unavailable("calculate_fundamental_change", as_of, f"Fewer than 2 point-in-time quarters for {ticker}.", ticker=ticker)
    changes = calculate_quarterly_changes(frame.drop(columns=["net_debt_to_annualized_ebitda"])).merge(frame[["period", "net_debt_to_annualized_ebitda"]], on="period")
    changes["nd_change"] = changes["net_debt_to_annualized_ebitda"].diff()
    price, market = _px(ticker, end), _px("XU100", end)
    events = align_financial_disclosure_dates(ticker, price_df=price, market_df=market).set_index("Period")
    rows, incomplete = [], []
    for _, c in changes.iterrows():
        if start_date and c["kap_disclosure_date"] < pd.Timestamp(start_date):
            continue
        e = events.loc[c["period"]] if c["period"] in events.index else None
        row = {"period": c["period"], "disclosed": str(c["kap_disclosure_date"].date()), "ebitda_margin": _r(c["ebitda_margin"]),
               "ebitda_margin_change_qoq": _r(c["EBITDA_Margin_Change_QoQ"]), "net_debt_to_annualized_ebitda": _r(c["net_debt_to_annualized_ebitda"], 2),
               "nd_ebitda_change_qoq": _r(c["nd_change"], 2), "improving": bool(c["EBITDA_Margin_Change_QoQ"] > 0 and c["nd_change"] < 0)}
        for h in (1, 5, 20):
            car, sector_excess = (e[f"Fwd_{h}d_CAR"] if e is not None else np.nan), None
            row[f"ret_{h}d"] = _r(e[f"Fwd_{h}d_Stock_Return"]) if e is not None else None
            row[f"car_xu100_{h}d"] = _r(car)
            if e is not None and pd.notna(car):
                s = _sector_excess(ticker, pd.Timestamp(e["Entry_Date"]), h, end)
                row[f"car_sector_{h}d"] = None if s is None else _r(e[f"Fwd_{h}d_Stock_Return"] - s)
            else:
                row[f"car_sector_{h}d"] = None
                incomplete.append(f"{c['period']}:{h}d")
        rows.append(row)
    out = _base("calculate_fundamental_change", as_of, start_date, end, ticker=ticker, quarters=rows, incomplete_windows=incomplete)
    if incomplete:
        out["warnings"].append(f"Incomplete post-disclosure windows by {end} (excluded from averages): {', '.join(incomplete)}.")
    if all(r["car_sector_5d"] is None for r in rows):
        out["warnings"].append("Sector-basket comparison unavailable: no sector peers with data.")
    out["warnings"].append("Association only: no causal claim that better earnings raise the price.")
    groups = {}
    for label, flag in (("improving", True), ("other", False)):
        sel = [r for r in rows if r["improving"] is flag]
        groups[label] = {"quarters": len(sel), **{f"mean_car_xu100_{h}d": _r(np.mean([r[f"car_xu100_{h}d"] for r in sel if r[f"car_xu100_{h}d"] is not None])) if any(r[f"car_xu100_{h}d"] is not None for r in sel) else None for h in (1, 5, 20)}}
    out["improving_vs_other"] = groups
    out["suggested_status"] = "INVESTIGATE" if (incomplete or len(rows) < 8) else "WATCH"
    return out


# --- macro (Section 15) --------------------------------------------------------------------------
def get_macro_context_summary(as_of: str, start_date: Optional[str] = None, end_date: Optional[str] = None, ticker: Optional[str] = None) -> dict[str, Any]:
    end = _end(as_of, end_date)
    out = _base("get_macro_context", as_of, start_date, end, series={}, unavailable_series=[])
    for name in ALL_SERIES:
        series = load_macro_series(name)
        if series is None:
            out["unavailable_series"].append(name)
            continue
        seen = series.loc[(series.index <= pd.Timestamp(end)) & (series.index >= pd.Timestamp(start_date or "1900-01-01"))]  # by publication date
        if seen.empty:
            out["unavailable_series"].append(name)
            continue
        out["series"][name] = {"observations": int(len(seen)), "first": _r(seen["value"].iloc[0], 4), "last": _r(seen["value"].iloc[-1], 4),
                               "change_pct": _r(seen["value"].iloc[-1] / seen["value"].iloc[0] - 1), "last_published": str(seen.index[-1].date())}
    if out["unavailable_series"]:
        out["warnings"].append(f"No data for: {', '.join(out['unavailable_series'])}. Do not state values for these series.")
    if ticker and "USDTRY" in out["series"]:
        fx = load_macro_series("USDTRY")["value"].loc[: pd.Timestamp(end)]
        stock = _px(ticker, end)["Adj Close"]
        weekly = pd.DataFrame({"fx": fx.resample("W-FRI").last().pct_change(), "stock": stock.resample("W-FRI").last().pct_change()}).dropna()
        weekly = weekly.loc[weekly.index >= pd.Timestamp(start_date or "1900-01-01")]
        if len(weekly) >= 20:
            up = weekly[weekly["fx"] > 0]["stock"]
            down = weekly[weekly["fx"] <= 0]["stock"]
            out["fx_relationship"] = {"ticker": ticker, "weeks": int(len(weekly)), "correlation_weekly": _r(weekly["fx"].corr(weekly["stock"])),
                                      "mean_stock_return_when_usdtry_up": _r(up.mean()), "weeks_up": int(len(up)),
                                      "mean_stock_return_when_usdtry_down": _r(down.mean()), "weeks_down": int(len(down))}
        else:
            out["warnings"].append("Too few weeks to relate USD/TRY to the stock.")
    out["suggested_status"] = "INVESTIGATE"  # an FX story is a hypothesis, never confirmation
    return out


# --- backtest / history / quality ---------------------------------------------------------------
def run_backtest_summary(ticker: str, as_of: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict[str, Any]:
    end = _end(as_of, end_date)
    price = _px(ticker, end)
    if start_date:
        price = price.loc[price.index >= pd.Timestamp(start_date)]
    market = _px("XU100", end)
    enriched = calculate_all_indicators(_px(ticker, end))
    signals = (enriched["Supertrend_Direction"] == 1).astype(int).loc[price.index]
    result = run_backtest(signals, price, benchmark_df=market)
    keys = ("Total_Return", "Benchmark_Total_Return", "Alpha_vs_Benchmark", "Sharpe_Ratio", "Max_Drawdown", "Win_Rate", "Total_Trades")
    out = _base("run_backtest", as_of, start_date, end, ticker=ticker, strategy="supertrend_long_cash_t_plus_1", metrics={k: _r(result.metrics.get(k)) for k in keys})
    out["suggested_status"] = "WATCH"
    return out


def get_stock_history_summary(ticker: str, as_of: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict[str, Any]:
    end = _end(as_of, end_date)
    price = _px(ticker, end)
    if start_date:
        price = price.loc[price.index >= pd.Timestamp(start_date)]
    out = _base("get_stock_history", as_of, start_date, end, ticker=ticker, rows=int(len(price)), first_date=str(price.index[0].date()), last_date=str(price.index[-1].date()),
                last_close=_r(price["Close"].iloc[-1]), period_return=_r(price["Adj Close"].iloc[-1] / price["Adj Close"].iloc[0] - 1))
    out["suggested_status"] = "WATCH"
    return out


def check_data_quality_summary(ticker: str, as_of: str) -> dict[str, Any]:
    report = check_data_quality(load_ticker_data(ticker), as_of, metadata={"signal_shift": 1})
    out = _base("check_data_quality", as_of, None, as_of, ticker=ticker, gate_status=report.status, rows_evaluated=report.rows_evaluated,
                issue_codes=[i.code for i in report.issues], fingerprint=report.data_fingerprint)
    out["suggested_status"] = "ANALYSIS_UNSAFE" if report.status in {"HARD_BLOCK", "ANALYSIS_UNSAFE"} else "WATCH"
    return out
