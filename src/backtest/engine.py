"""
Backtesting and Verification Engine
CSE-481 Engineering Economics Project

Core Features:
1. Strict t+1 signal execution (prevents look-ahead bias)
2. Realistic BIST fee model (broker commission + slippage)
3. Quantitative performance statistics (CAGR, Sharpe, Sortino, Max Drawdown, Win Rate)
4. Benchmark comparison (XU100 Index and Asset Buy-and-Hold)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_COMMISSION_RATE,
    DEFAULT_SLIPPAGE_RATE,
    SIGNAL_SHIFT_DAYS,
)


@dataclass
class TradeRecord:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    holding_days: int
    gross_return: float
    net_return: float
    pnl_try: float


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    drawdown_series: pd.Series
    trades: List[TradeRecord]
    metrics: Dict[str, Any]
    benchmark_equity: Optional[pd.Series] = None
    buy_and_hold_equity: Optional[pd.Series] = None

    def summary(self) -> pd.DataFrame:
        """Returns key performance metrics formatted as a pandas DataFrame."""
        return pd.DataFrame([self.metrics]).T.rename(columns={0: "Value"})


def calculate_performance(
    equity: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
    risk_free_rate: float = 0.40,  # 40% annual risk-free benchmark for Turkey 2025-2026
    trades: Optional[List[TradeRecord]] = None,
) -> Dict[str, Any]:
    """
    Syllabus Section 32 function:
    Calculates quantitative metrics for a given portfolio equity curve.
    """
    if len(equity) < 2:
        return {}

    daily_returns = equity.pct_change().dropna()
    total_days = (equity.index[-1] - equity.index[0]).days
    trading_days = len(equity)

    initial_val = equity.iloc[0]
    final_val = equity.iloc[-1]
    total_return = (final_val / initial_val) - 1.0

    # Annualization factor (typically 252 trading days)
    years = total_days / 365.25 if total_days > 0 else trading_days / 252.0
    cagr = (final_val / initial_val) ** (1.0 / years) - 1.0 if years > 0 and final_val > 0 else np.nan

    # Volatility
    daily_std = daily_returns.std()
    annualized_vol = daily_std * np.sqrt(252) if daily_std > 0 else 0.0

    # Sharpe Ratio
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252) - 1.0
    excess_returns = daily_returns - daily_rf
    sharpe = (excess_returns.mean() / daily_std) * np.sqrt(252) if daily_std > 0 else np.nan

    # Downside Volatility & Sortino Ratio
    downside_returns = daily_returns[daily_returns < daily_rf] - daily_rf
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0.0
    sortino = (
        (cagr - risk_free_rate) / downside_std
        if (downside_std > 0 and not np.isnan(cagr))
        else np.nan
    )

    # Maximum Drawdown
    cummax = equity.cummax()
    drawdowns = (equity - cummax) / cummax
    max_drawdown = drawdowns.min()  # Negative value, e.g. -0.15 for -15%

    # Drawdown Duration
    is_zero = drawdowns == 0
    dd_groups = (~is_zero).astype(int).groupby(is_zero.cumsum()).cumsum()
    max_dd_duration_days = int(dd_groups.max()) if len(dd_groups) > 0 else 0

    # Calmar Ratio
    calmar = (cagr / abs(max_drawdown)) if abs(max_drawdown) > 0 and not np.isnan(cagr) else np.nan

    # Trade statistics
    num_trades = len(trades) if trades is not None else 0
    win_rate = np.nan
    profit_factor = np.nan
    avg_trade_ret = np.nan

    if trades and len(trades) > 0:
        net_returns = [t.net_return for t in trades]
        pnls = [t.pnl_try for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(trades)
        avg_trade_ret = np.mean(net_returns)
        total_win = sum(wins)
        total_loss = abs(sum(losses))
        profit_factor = (total_win / total_loss) if total_loss > 0 else (np.inf if total_win > 0 else 1.0)

    metrics = {
        "Initial_Capital": initial_val,
        "Final_Capital": final_val,
        "Total_Return": total_return,
        "Annualized_Return_CAGR": cagr,
        "Annualized_Volatility": annualized_vol,
        "Sharpe_Ratio": sharpe,
        "Sortino_Ratio": sortino,
        "Max_Drawdown": max_drawdown,
        "Max_Drawdown_Duration_Bars": max_dd_duration_days,
        "Calmar_Ratio": calmar,
        "Total_Trades": num_trades,
        "Win_Rate": win_rate,
        "Profit_Factor": profit_factor,
        "Avg_Trade_Return": avg_trade_ret,
    }

    # Benchmark comparison
    if benchmark_equity is not None and len(benchmark_equity) == len(equity):
        bench_ret = (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) - 1.0
        metrics["Benchmark_Total_Return"] = bench_ret
        metrics["Alpha_vs_Benchmark"] = total_return - bench_ret

    return metrics


def run_backtest(
    signals: pd.Series,
    price_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    signal_shift: int = SIGNAL_SHIFT_DAYS,
    risk_free_rate: float = 0.40,
) -> BacktestResult:
    """
    Syllabus Section 32 function:
    Simulates a long/cash backtest given a time series of trading signals.
    
    Signal Convention:
    +1: LONG position (100% equity allocated)
     0: CASH position (0% equity allocated)
    -1: EXIT / CASH
    
    Execution timing:
    When signal at date t is +1, entry happens at date t + signal_shift (t+1 Open or Close).
    """
    # Align dates
    common_dates = signals.dropna().index.intersection(price_df.index)
    if len(common_dates) < 5:
        raise ValueError("Insufficient overlapping dates between signals and price series.")

    data = price_df.loc[common_dates].copy().sort_index()
    sig = signals.loc[common_dates].copy().sort_index()

    # Shift signals by signal_shift to enforce strict t+1 execution (No Look-Ahead Bias)
    execution_signals = sig.shift(signal_shift).fillna(0)
    
    close_prices = data["Adj Close"].values
    dates = data.index
    n = len(data)

    cash = initial_capital
    shares = 0.0
    position = 0  # 0: Cash, 1: Long
    entry_price = 0.0
    entry_date = None

    equity_values = np.zeros(n)
    trades: List[TradeRecord] = []

    cost_per_trade = commission_rate + slippage_rate

    for i in range(n):
        curr_price = close_prices[i]
        target_sig = execution_signals.iloc[i]
        curr_date = dates[i]

        # Check for Entry: Target signal > 0 and currently in cash
        if target_sig > 0 and position == 0:
            effective_entry_price = curr_price * (1.0 + slippage_rate)
            # Reserve commission
            investable_cash = cash / (1.0 + commission_rate)
            shares = investable_cash / effective_entry_price
            cash = 0.0
            position = 1
            entry_price = effective_entry_price
            entry_date = curr_date

        # Check for Exit: Target signal <= 0 and currently holding stock
        elif target_sig <= 0 and position == 1:
            effective_exit_price = curr_price * (1.0 - slippage_rate)
            gross_proceeds = shares * effective_exit_price
            net_proceeds = gross_proceeds * (1.0 - commission_rate)
            
            gross_ret = (effective_exit_price / entry_price) - 1.0
            pnl = net_proceeds - (shares * entry_price)
            net_ret = pnl / (shares * entry_price)
            holding_bars = i - data.index.get_loc(entry_date)

            trades.append(
                TradeRecord(
                    entry_date=entry_date,
                    exit_date=curr_date,
                    entry_price=entry_price,
                    exit_price=effective_exit_price,
                    shares=shares,
                    holding_days=holding_bars,
                    gross_return=gross_ret,
                    net_return=net_ret,
                    pnl_try=pnl,
                )
            )
            cash = net_proceeds
            shares = 0.0
            position = 0
            entry_price = 0.0
            entry_date = None

        # Current total equity mark-to-market
        if position == 1:
            current_portfolio_value = shares * curr_price
        else:
            current_portfolio_value = cash

        equity_values[i] = current_portfolio_value

    equity_series = pd.Series(equity_values, index=dates, name="Strategy_Equity")
    daily_returns = equity_series.pct_change().fillna(0)
    cummax = equity_series.cummax()
    drawdown_series = (equity_series - cummax) / cummax

    # Benchmark: Buy and Hold of the underlying stock
    bh_equity = (close_prices / close_prices[0]) * initial_capital
    bh_series = pd.Series(bh_equity, index=dates, name="Buy_and_Hold_Equity")

    # Benchmark: XU100 Index (if provided)
    bench_series = None
    if benchmark_df is not None:
        b_aligned = benchmark_df.reindex(dates).ffill()
        if not b_aligned.empty and "Adj Close" in b_aligned:
            b_close = b_aligned["Adj Close"].values
            bench_equity = (b_close / b_close[0]) * initial_capital
            bench_series = pd.Series(bench_equity, index=dates, name="Benchmark_XU100_Equity")

    metrics = calculate_performance(
        equity=equity_series,
        benchmark_equity=bench_series if bench_series is not None else bh_series,
        risk_free_rate=risk_free_rate,
        trades=trades,
    )

    return BacktestResult(
        equity_curve=equity_series,
        daily_returns=daily_returns,
        drawdown_series=drawdown_series,
        trades=trades,
        metrics=metrics,
        benchmark_equity=bench_series,
        buy_and_hold_equity=bh_series,
    )
