"""
Deterministic Technical Indicators Module
CSE-481 Engineering Economics Project
Pure mathematical/deterministic calculation of:
- Trend: SMA, EMA, KAMA
- Momentum: RSI, MACD
- Volatility: Bollinger Bands, ATR
- Trend State: Supertrend
- Cloud / Structure: Ichimoku Cloud
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def calculate_sma(series: pd.Series, window: int = 20) -> pd.Series:
    """Calculates Simple Moving Average."""
    return series.rolling(window=window).mean()


def calculate_ema(series: pd.Series, span: int = 20) -> pd.Series:
    """Calculates Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_kama(
    series: pd.Series,
    period: int = 10,
    fast_span: int = 2,
    slow_span: int = 30,
) -> pd.Series:
    """
    Kaufman's Adaptive Moving Average (KAMA).
    Adapts smoothing speed to market volatility/noise.
    """
    change = (series - series.shift(period)).abs()
    volatility = (series - series.shift(1)).abs().rolling(period).sum()

    # Avoid division by zero
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)

    fast_sc = 2.0 / (fast_span + 1.0)
    slow_sc = 2.0 / (slow_span + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = np.zeros(len(series))
    kama[:] = np.nan

    first_valid_idx = period
    if len(series) > first_valid_idx:
        kama[first_valid_idx] = series.iloc[first_valid_idx]
        for i in range(first_valid_idx + 1, len(series)):
            kama[i] = kama[i - 1] + sc.iloc[i] * (series.iloc[i] - kama[i - 1])

    return pd.Series(kama, index=series.index, name=f"KAMA_{period}")


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) using Wilder's exponential smoothing.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0).rename(f"RSI_{period}")


def calculate_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).
    Returns DataFrame with MACD, MACD_Signal, and MACD_Hist.
    """
    fast_ema = calculate_ema(series, span=fast_period)
    slow_ema = calculate_ema(series, span=slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, span=signal_period)
    hist = macd_line - signal_line

    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACD_Signal": signal_line,
            "MACD_Hist": hist,
        },
        index=series.index,
    )


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr.rename(f"ATR_{period}")


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands: Middle, Upper, Lower, %B, and Bandwidth.
    """
    mid = calculate_sma(series, window=window)
    std = series.rolling(window=window).std()

    upper = mid + (num_std * std)
    lower = mid - (num_std * std)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)

    return pd.DataFrame(
        {
            "BB_Middle": mid,
            "BB_Upper": upper,
            "BB_Lower": lower,
            "BB_Bandwidth": bandwidth,
            "BB_PctB": pct_b,
        },
        index=series.index,
    )


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Supertrend Indicator.
    Returns:
    - Supertrend line
    - Supertrend_Direction: +1 (Bullish), -1 (Bearish)
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(df)

    atr = calculate_atr(df, period=period).values
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    trend = np.zeros(n)
    supertrend = np.zeros(n)

    # Initialize
    trend[0] = 1
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = final_lower[0]

    for i in range(1, n):
        # Final Upper Band
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final Lower Band
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Trend direction determination
        if trend[i - 1] == 1:
            if close[i] < final_lower[i]:
                trend[i] = -1
                supertrend[i] = final_upper[i]
            else:
                trend[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                trend[i] = 1
                supertrend[i] = final_lower[i]
            else:
                trend[i] = -1
                supertrend[i] = final_upper[i]

    return pd.DataFrame(
        {
            "Supertrend": supertrend,
            "Supertrend_Direction": trend,  # +1 Bull, -1 Bear
        },
        index=df.index,
    )


def calculate_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> pd.DataFrame:
    """
    Ichimoku Kinko Hyo (Ichimoku Cloud).
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Tenkan-sen (Conversion Line)
    tenkan_sen = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2

    # Kijun-sen (Base Line)
    kijun_sen = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2

    # Senkou Span A (Leading Span A) - plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)

    # Senkou Span B (Leading Span B) - plotted 26 periods ahead
    senkou_span_b = (
        (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2
    ).shift(kijun_period)

    # Chikou Span (Lagging Span) - close plotted 26 periods behind
    chikou_span = close.shift(-kijun_period)

    cloud_top = pd.concat([senkou_span_a, senkou_span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([senkou_span_a, senkou_span_b], axis=1).min(axis=1)

    # State relative to cloud
    state = np.where(
        close > cloud_top,
        "ABOVE_CLOUD",
        np.where(close < cloud_bottom, "BELOW_CLOUD", "INSIDE_CLOUD"),
    )

    return pd.DataFrame(
        {
            "Tenkan_Sen": tenkan_sen,
            "Kijun_Sen": kijun_sen,
            "Senkou_Span_A": senkou_span_a,
            "Senkou_Span_B": senkou_span_b,
            "Chikou_Span": chikou_span,
            "Cloud_Top": cloud_top,
            "Cloud_Bottom": cloud_bottom,
            "Ichimoku_State": state,
        },
        index=df.index,
    )


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches an OHLCV dataframe with all required deterministic indicators.
    """
    result = df.copy()
    close = result["Close"]

    # Trend
    result["SMA_20"] = calculate_sma(close, 20)
    result["SMA_50"] = calculate_sma(close, 50)
    result["SMA_200"] = calculate_sma(close, 200)
    result["EMA_20"] = calculate_ema(close, 20)
    result["KAMA_10"] = calculate_kama(close, period=10)
    result["KAMA_Slope"] = result["KAMA_10"].diff()

    # Momentum
    result["RSI_14"] = calculate_rsi(close, 14)
    macd_df = calculate_macd(close)
    result = pd.concat([result, macd_df], axis=1)

    # Volatility
    result["ATR_14"] = calculate_atr(result, 14)
    bb_df = calculate_bollinger_bands(close, window=20, num_std=2.0)
    result = pd.concat([result, bb_df], axis=1)

    # Trend State (Supertrend)
    st_df = calculate_supertrend(result, period=10, multiplier=3.0)
    result = pd.concat([result, st_df], axis=1)

    # Cloud Structure (Ichimoku)
    ichi_df = calculate_ichimoku(result)
    result = pd.concat([result, ichi_df], axis=1)

    return result
