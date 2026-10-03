"""
Mandatory Research Scenario 2: Weekday and Multi-Day Patterns
CSE-481 Engineering Economics Project

Goal:
Test whether individual stocks or sectors exhibit recurring day-of-week or
multi-day sequences (e.g., Monday weakness / Friday rally, 3 consecutive down
days followed by a rebound), while controlling for transaction costs, market
regimes, and multiple-testing risk.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DEFAULT_COMMISSION_RATE, DEFAULT_SLIPPAGE_RATE


WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def test_weekday_pattern(
    df: pd.DataFrame,
    return_col: str = "Daily_Return",
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
) -> pd.DataFrame:
    """
    Syllabus Section 32 function:
    Analyzes return distributions across days of the week (Monday=0 to Friday=4).
    Calculates:
    - Count of trading days
    - Mean return & Median return
    - Standard deviation & Annualized Sharpe estimate
    - Win rate (% positive days)
    - T-statistic & p-value versus zero and versus unconditional distribution
    - Net average return after roundtrip costs (2 * (commission + slippage))
    """
    data = df.copy()
    if return_col not in data.columns:
        if "Adj Close" in data.columns:
            data[return_col] = data["Adj Close"].pct_change()
        else:
            raise KeyError(f"Return column {return_col} not found in dataframe.")

    data = data.dropna(subset=[return_col])
    data["DayOfWeek"] = data.index.dayofweek  # 0=Monday, 4=Friday
    
    # Filter only Monday-Friday (exclude any unexpected Sunday/Saturday trading)
    data = data[data["DayOfWeek"].isin(range(5))]

    unconditional_returns = data[return_col].values
    unconditional_mean = np.mean(unconditional_returns)
    roundtrip_cost = 2.0 * (commission_rate + slippage_rate)

    results = []
    for day_idx in range(5):
        day_name = WEEKDAY_NAMES[day_idx]
        day_rets = data[data["DayOfWeek"] == day_idx][return_col].values
        n = len(day_rets)
        if n < 5:
            continue

        mean_ret = np.mean(day_rets)
        median_ret = np.median(day_rets)
        std_ret = np.std(day_rets, ddof=1) if n > 1 else 0.0
        win_rate = np.mean(day_rets > 0)
        
        # T-test comparing day return against zero
        t_stat, p_val = stats.ttest_1samp(day_rets, 0.0) if std_ret > 0 else (0.0, 1.0)
        
        # T-test comparing day return against other days
        other_rets = data[data["DayOfWeek"] != day_idx][return_col].values
        t_stat_rel, p_val_rel = stats.ttest_ind(day_rets, other_rets, equal_var=False) if (len(other_rets) > 1 and std_ret > 0) else (0.0, 1.0)

        # Net return after transaction costs
        net_mean_ret = mean_ret - roundtrip_cost

        results.append({
            "Day_Index": day_idx,
            "Day_Name": day_name,
            "Sample_Count": n,
            "Mean_Return": mean_ret,
            "Median_Return": median_ret,
            "Std_Dev": std_ret,
            "Win_Rate": win_rate,
            "T_Stat_vs_Zero": t_stat,
            "P_Value_vs_Zero": p_val,
            "P_Value_vs_Others": p_val_rel,
            "Net_Mean_Return": net_mean_ret,
            "Is_Statistically_Significant": bool(p_val < 0.05),
        })

    return pd.DataFrame(results)


def detect_multiday_patterns(
    df: pd.DataFrame,
    pattern_type: str = "consecutive_down",
    n_days: int = 3,
    forward_horizons: List[int] = [1, 2, 5],
    market_df: Optional[pd.DataFrame] = None,
    signal_shift: int = 1,  # Strict t+1 execution to eliminate look-ahead bias
) -> pd.DataFrame:
    """
    Syllabus Section 32 function:
    Detects multi-day sequence patterns and measures subsequent forward returns.
    
    Supported pattern_types:
    - 'consecutive_down': n_days consecutive negative daily returns
    - 'consecutive_up': n_days consecutive positive daily returns
    - 'large_down_day': single day drop > 2.0% (mean reversion candidate)
    - 'mon_tue_wed_down': Mon, Tue, Wed all negative, measuring Thu & Fri returns
    """
    data = df.copy().sort_index()
    if "Daily_Return" not in data.columns:
        data["Daily_Return"] = data["Adj Close"].pct_change()
    
    data = data.dropna(subset=["Daily_Return"])
    rets = data["Daily_Return"].values
    dates = data.index
    n = len(data)

    event_indices = []

    if pattern_type == "consecutive_down":
        for i in range(n_days - 1, n):
            # Check if previous n_days returns were all strictly negative
            window = rets[i - n_days + 1 : i + 1]
            if np.all(window < 0):
                event_indices.append(i)

    elif pattern_type == "consecutive_up":
        for i in range(n_days - 1, n):
            window = rets[i - n_days + 1 : i + 1]
            if np.all(window > 0):
                event_indices.append(i)

    elif pattern_type == "large_down_day":
        for i in range(n):
            if rets[i] <= -0.02:  # at least -2% drop
                event_indices.append(i)

    elif pattern_type == "mon_tue_wed_down":
        # Specifically test the textbook scenario: Mon, Tue, Wed down
        for i in range(2, n):
            curr_date = dates[i]
            if curr_date.dayofweek == 2:  # Wednesday
                d0 = dates[i - 2]
                d1 = dates[i - 1]
                # Verify they are Mon(0), Tue(1), Wed(2) of the same week
                if d0.dayofweek == 0 and d1.dayofweek == 1:
                    if rets[i - 2] < 0 and rets[i - 1] < 0 and rets[i] < 0:
                        event_indices.append(i)
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")

    records = []
    roundtrip_cost = 2.0 * (DEFAULT_COMMISSION_RATE + DEFAULT_SLIPPAGE_RATE)

    for idx in event_indices:
        event_date = dates[idx]
        entry_idx = idx + signal_shift

        if entry_idx >= n:
            continue

        entry_date = dates[entry_idx]
        entry_price = data["Adj Close"].iloc[entry_idx]

        rec: Dict[str, Any] = {
            "Event_Date": event_date,
            "Pattern_Type": pattern_type,
            "Entry_Date": entry_date,
            "Entry_Price": entry_price,
            "Event_Day_Return": rets[idx],
        }

        # Market regime at event date (if provided)
        if market_df is not None:
            if "Daily_Return" not in market_df.columns:
                market_df["Daily_Return"] = market_df["Adj Close"].pct_change()
            m_locs = market_df.index.get_indexer([event_date])
            if len(m_locs) > 0 and m_locs[0] != -1:
                m_ret = market_df["Daily_Return"].iloc[m_locs[0]]
                rec["Market_Return_Event_Day"] = m_ret
                rec["Market_Regime"] = "BULL" if m_ret > 0 else "BEAR"

        # Forward return measurement
        for h in forward_horizons:
            exit_idx = entry_idx + h
            if exit_idx < n:
                exit_price = data["Adj Close"].iloc[exit_idx]
                fwd_ret = (exit_price / entry_price) - 1.0
                net_ret = fwd_ret - roundtrip_cost
                rec[f"Fwd_{h}d_Return"] = fwd_ret
                rec[f"Fwd_{h}d_Net_Return"] = net_ret
                rec[f"Fwd_{h}d_Positive"] = int(fwd_ret > 0)
            else:
                rec[f"Fwd_{h}d_Return"] = np.nan
                rec[f"Fwd_{h}d_Net_Return"] = np.nan
                rec[f"Fwd_{h}d_Positive"] = np.nan

        records.append(rec)

    return pd.DataFrame(records)


def summarize_pattern_performance(events_df: pd.DataFrame, horizons: List[int] = [1, 2, 5]) -> pd.DataFrame:
    """
    Summarizes win rate, mean return, median return, and net return across horizons.
    """
    if events_df.empty:
        return pd.DataFrame()

    summary = []
    for h in horizons:
        col = f"Fwd_{h}d_Return"
        net_col = f"Fwd_{h}d_Net_Return"
        pos_col = f"Fwd_{h}d_Positive"
        if col in events_df.columns:
            valid = events_df.dropna(subset=[col])
            n = len(valid)
            if n > 0:
                mean_r = valid[col].mean()
                median_r = valid[col].median()
                win_rate = valid[pos_col].mean()
                net_mean_r = valid[net_col].mean()
                t_stat, p_val = stats.ttest_1samp(valid[col], 0.0) if valid[col].std() > 0 else (0.0, 1.0)
                summary.append({
                    "Horizon_Days": h,
                    "Total_Events": n,
                    "Mean_Return": mean_r,
                    "Median_Return": median_r,
                    "Win_Rate": win_rate,
                    "Net_Mean_Return": net_mean_r,
                    "T_Stat": t_stat,
                    "P_Value": p_val,
                    "Is_Profitable_After_Costs": bool(net_mean_r > 0 and p_val < 0.05),
                })

    return pd.DataFrame(summary)
