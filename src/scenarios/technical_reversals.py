"""
Mandatory Research Scenario 3: Technical Reversal Patterns
CSE-481 Engineering Economics Project

Goal:
Treat technical events (Bollinger Band touches, RSI extrema, KAMA crosses,
Supertrend flips, Ichimoku breakouts) as timestamped research events.
Compare single-indicator events against combined multi-indicator conditions
(e.g., Lower Bollinger Touch + RSI Trough + Volume Surge + Flattening KAMA).
Measure reversal strength and win rates across 1d, 3d, 5d, 10d horizons with
strict t+1 signal execution.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE_RATE
from src.indicators import calculate_all_indicators


MANDATORY_EVENT_FAMILIES = [
    "BB_LOWER_TOUCH",
    "BB_UPPER_TOUCH",
    "BB_MIDDLE_TEST",
    "RSI_OVERSOLD",
    "RSI_OVERBOUGHT",
    "SUPERTREND_BULLISH_FLIP",
    "SUPERTREND_BEARISH_FLIP",
    "KAMA_CROSS_UP",
    "ICHIMOKU_CLOUD_BREAKOUT",
    "RSI_LOCAL_TROUGH",
    "RSI_LOCAL_PEAK",
    "KAMA_CROSS_DOWN",
    "KAMA_SLOPE_TURN_UP",
    "KAMA_SLOPE_TURN_DOWN",
    "ICHIMOKU_CLOUD_REJECTION_UP",
    "ICHIMOKU_CLOUD_REJECTION_DOWN",
    "ICHIMOKU_RETURN_INTO_CLOUD",
    "SUPPORT_TOUCH",
    "RESISTANCE_TOUCH",
]

SR_WINDOW = 20
SR_TOLERANCE = 0.005  # a touch is a low/high within 0.5% of the prior-window level


def detect_reversal_events(
    df: pd.DataFrame,
    event_families: Optional[List[str]] = None,
    forward_horizons: List[int] = [1, 3, 5, 10],
    market_df: Optional[pd.DataFrame] = None,
    signal_shift: int = 1,  # Strict t+1 execution
) -> pd.DataFrame:
    """
    Syllabus Section 32 function:
    Scans the enriched price series for timestamped technical events.
    Records market state at the time of each event and tracks forward returns.
    """
    # Ensure indicators are calculated
    if "RSI_14" not in df.columns or "BB_Lower" not in df.columns:
        enriched = calculate_all_indicators(df)
    else:
        enriched = df.copy()

    enriched = enriched.sort_index()
    n = len(enriched)
    dates = enriched.index
    families = event_families or MANDATORY_EVENT_FAMILIES

    # Add 20-day Volume SMA for volume regime confirmation
    vol_sma = enriched["Volume"].rolling(window=20).mean()
    vol_ratio = enriched["Volume"] / vol_sma.replace(0, np.nan)

    # Market regime if provided
    mkt_regime_series = None
    if market_df is not None:
        mkt_ret_20d = market_df["Adj Close"].pct_change(20)
        mkt_regime_series = np.where(mkt_ret_20d >= 0, "BULL_MARKET", "BEAR_MARKET")
        mkt_regime_df = pd.Series(mkt_regime_series, index=market_df.index)

    # Support/resistance use the PRIOR window only (shift(1)) so the level is known at t.
    sr_support = enriched["Low"].rolling(SR_WINDOW).min().shift(1)
    sr_resistance = enriched["High"].rolling(SR_WINDOW).max().shift(1)

    # Volatility regime: 20d realized vol vs its own expanding median (point in time).
    realized_vol = enriched["Close"].pct_change().rolling(20).std()
    vol_regime_series = pd.Series(
        np.where(realized_vol >= realized_vol.expanding(min_periods=40).median(), "HIGH_VOL", "LOW_VOL"),
        index=enriched.index,
    ).where(realized_vol.notna(), "UNKNOWN")

    events_list = []
    roundtrip_cost = 2.0 * (DEFAULT_COMMISSION_RATE + DEFAULT_SLIPPAGE_RATE)

    for i in range(1, n):
        curr_date = dates[i]
        prev_date = dates[i - 1]

        close = enriched["Close"].iloc[i]
        prev_close = enriched["Close"].iloc[i - 1]
        low = enriched["Low"].iloc[i]
        high = enriched["High"].iloc[i]
        prev_low = enriched["Low"].iloc[i - 1]
        prev_high = enriched["High"].iloc[i - 1]

        bb_lower = enriched["BB_Lower"].iloc[i]
        bb_upper = enriched["BB_Upper"].iloc[i]
        bb_mid = enriched["BB_Middle"].iloc[i]

        rsi = enriched["RSI_14"].iloc[i]
        prev_rsi = enriched["RSI_14"].iloc[i - 1]

        st_dir = enriched["Supertrend_Direction"].iloc[i]
        prev_st_dir = enriched["Supertrend_Direction"].iloc[i - 1]

        kama = enriched["KAMA_10"].iloc[i]
        prev_kama = enriched["KAMA_10"].iloc[i - 1]
        kama_slope_state = enriched.get("KAMA_Slope_State", pd.Series(["UNKNOWN"] * n, index=dates)).iloc[i]

        ichi_state = enriched.get("Ichimoku_State", pd.Series(["UNKNOWN"] * n, index=dates)).iloc[i]
        prev_ichi_state = enriched.get("Ichimoku_State", pd.Series(["UNKNOWN"] * n, index=dates)).iloc[i - 1]

        curr_vol_ratio = vol_ratio.iloc[i] if not np.isnan(vol_ratio.iloc[i]) else 1.0

        # Event conditions check
        active_events = []

        # 1. Bollinger Lower Touch followed by bounce/touch
        if "BB_LOWER_TOUCH" in families and (low <= bb_lower or close <= bb_lower):
            active_events.append(("BB_LOWER_TOUCH", "BULLISH"))

        # 2. Bollinger Upper Touch
        if "BB_UPPER_TOUCH" in families and (high >= bb_upper or close >= bb_upper):
            active_events.append(("BB_UPPER_TOUCH", "BEARISH"))

        # 3. Bollinger Middle Test
        if "BB_MIDDLE_TEST" in families and (low <= bb_mid and high >= bb_mid):
            active_events.append(("BB_MIDDLE_TEST", "NEUTRAL"))

        # 4. RSI Oversold (<= 30)
        if "RSI_OVERSOLD" in families and rsi <= 30.0:
            active_events.append(("RSI_OVERSOLD", "BULLISH"))

        # 5. RSI Overbought (>= 70)
        if "RSI_OVERBOUGHT" in families and rsi >= 70.0:
            active_events.append(("RSI_OVERBOUGHT", "BEARISH"))

        # 6. Supertrend Bullish Flip (-1 -> +1)
        if "SUPERTREND_BULLISH_FLIP" in families and prev_st_dir == -1 and st_dir == 1:
            active_events.append(("SUPERTREND_BULLISH_FLIP", "BULLISH"))

        # 7. Supertrend Bearish Flip (+1 -> -1)
        if "SUPERTREND_BEARISH_FLIP" in families and prev_st_dir == 1 and st_dir == -1:
            active_events.append(("SUPERTREND_BEARISH_FLIP", "BEARISH"))

        # 8. KAMA Cross Up (Price crosses above KAMA)
        if "KAMA_CROSS_UP" in families and prev_close <= prev_kama and close > kama:
            active_events.append(("KAMA_CROSS_UP", "BULLISH"))

        # 9. Ichimoku Cloud Breakout
        if "ICHIMOKU_CLOUD_BREAKOUT" in families and prev_ichi_state != "ABOVE_CLOUD" and ichi_state == "ABOVE_CLOUD":
            active_events.append(("ICHIMOKU_CLOUD_BREAKOUT", "BULLISH"))

        # 10. RSI local trough / peak, confirmed one bar later (no look-ahead)
        if i >= 2:
            prev2_rsi = enriched["RSI_14"].iloc[i - 2]
            if "RSI_LOCAL_TROUGH" in families and prev_rsi < prev2_rsi and prev_rsi < rsi and prev_rsi <= 40.0:
                active_events.append(("RSI_LOCAL_TROUGH", "BULLISH"))
            if "RSI_LOCAL_PEAK" in families and prev_rsi > prev2_rsi and prev_rsi > rsi and prev_rsi >= 60.0:
                active_events.append(("RSI_LOCAL_PEAK", "BEARISH"))

        # 11. KAMA cross down and slope turns
        if "KAMA_CROSS_DOWN" in families and prev_close >= prev_kama and close < kama:
            active_events.append(("KAMA_CROSS_DOWN", "BEARISH"))
        prev_kama_state = enriched.get("KAMA_Slope_State", pd.Series(["UNKNOWN"] * n, index=dates)).iloc[i - 1]
        if "KAMA_SLOPE_TURN_UP" in families and kama_slope_state == "RISING" and prev_kama_state != "RISING":
            active_events.append(("KAMA_SLOPE_TURN_UP", "BULLISH"))
        if "KAMA_SLOPE_TURN_DOWN" in families and kama_slope_state == "FALLING" and prev_kama_state != "FALLING":
            active_events.append(("KAMA_SLOPE_TURN_DOWN", "BEARISH"))

        # 12. Ichimoku rejection (price tests the cloud edge but closes back outside) and return into cloud
        cloud_top = enriched["Cloud_Top"].iloc[i] if "Cloud_Top" in enriched else np.nan
        cloud_bottom = enriched["Cloud_Bottom"].iloc[i] if "Cloud_Bottom" in enriched else np.nan
        if "ICHIMOKU_CLOUD_REJECTION_UP" in families and ichi_state == "ABOVE_CLOUD" and low <= cloud_top * (1 + SR_TOLERANCE):
            active_events.append(("ICHIMOKU_CLOUD_REJECTION_UP", "BULLISH"))
        if "ICHIMOKU_CLOUD_REJECTION_DOWN" in families and ichi_state == "BELOW_CLOUD" and high >= cloud_bottom * (1 - SR_TOLERANCE):
            active_events.append(("ICHIMOKU_CLOUD_REJECTION_DOWN", "BEARISH"))
        if "ICHIMOKU_RETURN_INTO_CLOUD" in families and ichi_state == "INSIDE_CLOUD" and prev_ichi_state in {"ABOVE_CLOUD", "BELOW_CLOUD"}:
            active_events.append(("ICHIMOKU_RETURN_INTO_CLOUD", "NEUTRAL"))

        # 13. Support / resistance touches (levels from the prior window only)
        sup, res = sr_support.iloc[i], sr_resistance.iloc[i]
        if "SUPPORT_TOUCH" in families and not np.isnan(sup) and low <= sup * (1 + SR_TOLERANCE):
            active_events.append(("SUPPORT_TOUCH", "BULLISH"))
        if "RESISTANCE_TOUCH" in families and not np.isnan(res) and high >= res * (1 - SR_TOLERANCE):
            active_events.append(("RESISTANCE_TOUCH", "BEARISH"))

        if not active_events:
            continue

        # Look up market regime on event date
        mkt_reg = "UNKNOWN"
        if mkt_regime_series is not None and curr_date in mkt_regime_df.index:
            mkt_reg = mkt_regime_df.loc[curr_date]

        entry_idx = i + signal_shift
        if entry_idx >= n:
            continue

        entry_date = dates[entry_idx]
        entry_price = enriched["Adj Close"].iloc[entry_idx]

        for ev_name, ev_type in active_events:
            record: Dict[str, Any] = {
                "Event_Date": curr_date,
                "Event_Name": ev_name,
                "Event_Direction_Bias": ev_type,
                "Close_Price": close,
                "RSI_14": rsi,
                "Supertrend_Direction": st_dir,
                "KAMA_Slope_State": kama_slope_state,
                "Ichimoku_State": ichi_state,
                "Volume_Ratio": curr_vol_ratio,
                "Volume_Regime": "HIGH" if curr_vol_ratio >= 1.2 else ("LOW" if curr_vol_ratio < 0.8 else "NORMAL"),
                "Market_Regime": mkt_reg,
                "Volatility_Regime": vol_regime_series.iloc[i],
                "Entry_Date": entry_date,
                "Entry_Price": entry_price,
            }

            # Forward returns
            for h in forward_horizons:
                exit_idx = entry_idx + h
                if exit_idx < n:
                    exit_price = enriched["Adj Close"].iloc[exit_idx]
                    fwd_ret = (exit_price / entry_price) - 1.0
                    net_ret = fwd_ret - roundtrip_cost
                    # Success criteria: positive return for bullish, negative for bearish
                    if ev_type == "BULLISH":
                        hit = int(fwd_ret > 0)
                    elif ev_type == "BEARISH":
                        hit = int(fwd_ret < 0)
                    else:
                        hit = int(abs(fwd_ret) > 0.01)

                    record[f"Fwd_{h}d_Return"] = fwd_ret
                    # Reversal strength: return in the direction the event predicts
                    record[f"Reversal_Strength_{h}d"] = -fwd_ret if ev_type == "BEARISH" else fwd_ret
                    record[f"Fwd_{h}d_Net_Return"] = net_ret
                    record[f"Fwd_{h}d_Hit"] = hit
                else:
                    record[f"Fwd_{h}d_Return"] = np.nan
                    record[f"Reversal_Strength_{h}d"] = np.nan
                    record[f"Fwd_{h}d_Net_Return"] = np.nan
                    record[f"Fwd_{h}d_Hit"] = np.nan

            events_list.append(record)

    return pd.DataFrame(events_list)


def compare_single_vs_combined_reversals(
    events_df: pd.DataFrame,
    base_event: str = "BB_LOWER_TOUCH",
    target_horizon: int = 5,
) -> pd.DataFrame:
    """
    Syllabus Section 13 & 32 requirement:
    Compares the historical success of a single indicator event against
    multi-indicator combined conditions (e.g. Aselsan example from syllabus).
    """
    if events_df.empty:
        return pd.DataFrame()

    hit_col = f"Fwd_{target_horizon}d_Hit"
    ret_col = f"Fwd_{target_horizon}d_Return"
    net_col = f"Fwd_{target_horizon}d_Net_Return"

    base_sub = events_df[events_df["Event_Name"] == base_event].copy()
    if base_sub.empty:
        return pd.DataFrame()

    subsets = {
        "1. Single Event (Baseline)": base_sub,
        "2. Combined: Event + RSI <= 35": base_sub[base_sub["RSI_14"] <= 35],
        "3. Combined: Event + Volume Surge (>1.2x)": base_sub[base_sub["Volume_Ratio"] >= 1.2],
        "4. Combined: Event + KAMA Not Falling": base_sub[base_sub["KAMA_Slope_State"].isin(["RISING", "FLATTENING"])],
        "5. Full Ideal Bundle: Event + RSI<=35 + VolSurge + KAMA Stable": base_sub[
            (base_sub["RSI_14"] <= 35)
            & (base_sub["Volume_Ratio"] >= 1.2)
            & (base_sub["KAMA_Slope_State"].isin(["RISING", "FLATTENING"]))
        ],
        "6. High Risk Flaw: Event but Bearish Supertrend & Below Cloud": base_sub[
            (base_sub["Supertrend_Direction"] == -1) & (base_sub["Ichimoku_State"] == "BELOW_CLOUD")
        ],
        "7. Combined: Event + RSI<=35 + Volume Surge (>1.2x)": base_sub[(base_sub["RSI_14"] <= 35) & (base_sub["Volume_Ratio"] >= 1.2)],
    }

    comparison = []
    for label, subset in subsets.items():
        valid = subset.dropna(subset=[hit_col])
        count = len(valid)
        if count == 0:
            comparison.append({
                "Condition": label,
                "Sample_Count": 0,
                "Win_Rate": np.nan,
                "Mean_Return": np.nan,
                "Median_Return": np.nan,
                "Net_Mean_Return": np.nan,
            })
            continue

        win_rate = valid[hit_col].mean()
        mean_ret = valid[ret_col].mean()
        med_ret = valid[ret_col].median()
        net_ret = valid[net_col].mean()

        comparison.append({
            "Condition": label,
            "Sample_Count": count,
            "Win_Rate": win_rate,
            "Mean_Return": mean_ret,
            "Median_Return": med_ret,
            "Net_Mean_Return": net_ret,
        })

    return pd.DataFrame(comparison)


def summarize_reversals_by_regime(events_df: pd.DataFrame, target_horizon: int = 5) -> pd.DataFrame:
    """Spec section 13: bounce rate and strength per event, split by market and volatility regime."""
    if events_df.empty:
        return pd.DataFrame()
    hit, strength = f"Fwd_{target_horizon}d_Hit", f"Reversal_Strength_{target_horizon}d"
    valid = events_df.dropna(subset=[hit])
    grouped = valid.groupby(["Event_Name", "Market_Regime", "Volatility_Regime"])
    return grouped.agg(
        Sample_Count=(hit, "size"),
        Win_Rate=(hit, "mean"),
        Mean_Strength=(strength, "mean"),
        Median_Strength=(strength, "median"),
    ).reset_index()
