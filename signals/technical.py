"""
Technical signal scoring — returns float 0.0-1.0 per sub-signal.
Composite technical score = weighted average of all sub-signals.
"""
import numpy as np
import pandas as pd
import ta


def _safe(series: pd.Series) -> pd.Series:
    return series.fillna(0)


def score_golden_cross(df: pd.DataFrame) -> float:
    """50 SMA crosses above 200 SMA. Higher score if recent."""
    if len(df) < 200:
        return 0.0
    close = df["Close"].squeeze()
    sma50  = ta.trend.sma_indicator(close, window=50)
    sma200 = ta.trend.sma_indicator(close, window=200)
    diff = sma50 - sma200
    if diff.iloc[-1] <= 0:
        return 0.0
    # Check when the cross happened
    cross_days_ago = None
    for i in range(1, min(60, len(diff))):
        if diff.iloc[-1-i] <= 0:
            cross_days_ago = i
            break
    if cross_days_ago is None:
        return 0.5  # been above a long time
    if cross_days_ago <= 10:
        return 1.0
    if cross_days_ago <= 30:
        return 0.85
    return 0.65


def score_price_vs_200sma(df: pd.DataFrame) -> float:
    """Price reclaiming 200 SMA after being below it."""
    if len(df) < 200:
        return 0.0
    close = df["Close"].squeeze()
    sma200 = ta.trend.sma_indicator(close, window=200)
    if close.iloc[-1] <= sma200.iloc[-1]:
        return 0.0
    # Check if it recently crossed
    for i in range(1, min(10, len(close))):
        if close.iloc[-1-i] <= sma200.iloc[-1-i]:
            return 1.0  # fresh reclaim
    return 0.5  # already above for a while


def score_bounce_off_50sma(df: pd.DataFrame) -> float:
    """Price bouncing off 50 SMA as support, multiple touches = more conviction."""
    if len(df) < 60:
        return 0.0
    close = df["Close"].squeeze()
    low   = df["Low"].squeeze()
    sma50 = ta.trend.sma_indicator(close, window=50)
    recent = slice(-60, None)
    bounces = 0
    for i in range(-59, -1):
        pct_from_50 = (low.iloc[i] - sma50.iloc[i]) / sma50.iloc[i]
        # Touched within 1% of 50 SMA then closed above it
        if -0.01 <= pct_from_50 <= 0.005 and close.iloc[i] > sma50.iloc[i]:
            bounces += 1
    # Currently holding above 50 SMA
    if close.iloc[-1] < sma50.iloc[-1]:
        return 0.0
    if bounces >= 3:
        return 1.0
    if bounces == 2:
        return 0.75
    if bounces == 1:
        return 0.5
    return 0.25


def score_ma_stacking(df: pd.DataFrame) -> float:
    """All MAs trending up and stacked: 9 > 21 > 50 > 200."""
    if len(df) < 200:
        return 0.0
    close = df["Close"].squeeze()
    ma9   = ta.trend.sma_indicator(close, window=9).iloc[-1]
    ma21  = ta.trend.sma_indicator(close, window=21).iloc[-1]
    ma50  = ta.trend.sma_indicator(close, window=50).iloc[-1]
    ma200 = ta.trend.sma_indicator(close, window=200).iloc[-1]
    stacked = sum([
        ma9 > ma21,
        ma21 > ma50,
        ma50 > ma200,
        close.iloc[-1] > ma9,
    ])
    return stacked / 4.0


def score_macd(df: pd.DataFrame) -> float:
    """MACD crossing above signal line with positive histogram."""
    if len(df) < 35:
        return 0.0
    close = df["Close"].squeeze()
    macd_diff = ta.trend.macd_diff(close)
    macd_line = ta.trend.macd(close)
    sig_line  = ta.trend.macd_signal(close)

    if macd_diff.iloc[-1] <= 0:
        return 0.0
    # Bullish cross: histogram turned positive recently?
    cross_days_ago = None
    for i in range(1, min(20, len(macd_diff))):
        if macd_diff.iloc[-1-i] <= 0:
            cross_days_ago = i
            break
    histogram_growing = macd_diff.iloc[-1] > macd_diff.iloc[-2]
    base = 0.3 if cross_days_ago is None else min(1.0, 0.5 + (20 - cross_days_ago) / 20)
    return min(1.0, base + (0.2 if histogram_growing else 0))


def score_rsi(df: pd.DataFrame) -> float:
    """RSI recovering from oversold (<30) back above 50."""
    if len(df) < 20:
        return 0.0
    close = df["Close"].squeeze()
    rsi = ta.momentum.rsi(close, window=14)
    current_rsi = rsi.iloc[-1]

    # Check if RSI was oversold in last 20 days
    was_oversold = any(rsi.iloc[-20:] < 30)

    if current_rsi < 30:
        return 0.1  # currently oversold
    if current_rsi >= 50 and was_oversold:
        return 1.0  # recovered from oversold — ideal
    if current_rsi >= 60:
        return 0.65
    if current_rsi >= 50:
        return 0.45
    if current_rsi >= 40:
        return 0.25
    return 0.1


def score_stochastic(df: pd.DataFrame) -> float:
    """Stochastic crossing upward from oversold zone (<20)."""
    if len(df) < 20:
        return 0.0
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    close = df["Close"].squeeze()
    stoch_k = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
    stoch_d = ta.momentum.stoch_signal(high, low, close, window=14, smooth_window=3)

    k = stoch_k.iloc[-1]
    d = stoch_d.iloc[-1]
    prev_k = stoch_k.iloc[-2]
    prev_d = stoch_d.iloc[-2]

    was_oversold = any(stoch_k.iloc[-10:] < 20)
    cross_up = (prev_k < prev_d) and (k > d)

    if k < 20:
        return 0.1
    if was_oversold and k > 20 and cross_up:
        return 1.0
    if was_oversold and k > 20:
        return 0.75
    if cross_up and k < 50:
        return 0.5
    if k > 50:
        return 0.35
    return 0.2


def score_roc(df: pd.DataFrame) -> float:
    """Rate of Change turning positive after negative reading."""
    if len(df) < 20:
        return 0.0
    close = df["Close"].squeeze()
    roc = ta.momentum.roc(close, window=12)
    current = roc.iloc[-1]
    was_negative = any(roc.iloc[-15:] < 0)

    if current > 0 and was_negative:
        return min(1.0, 0.5 + current / 20)
    if current > 5:
        return 0.6
    if current > 0:
        return 0.4
    return 0.0


def score_volume_accumulation(df: pd.DataFrame) -> float:
    """More volume on up days vs down days over last 20 sessions."""
    if len(df) < 20:
        return 0.0
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    recent_close  = close.iloc[-20:]
    recent_volume = volume.iloc[-20:]
    up_vol   = recent_volume[recent_close > recent_close.shift(1)].sum()
    down_vol = recent_volume[recent_close < recent_close.shift(1)].sum()
    if up_vol + down_vol == 0:
        return 0.5
    ratio = up_vol / (up_vol + down_vol)
    return float(np.clip(ratio, 0, 1))


def score_volume_breakout(df: pd.DataFrame) -> float:
    """Volume spike (>1.5x avg) on a price breakout above recent resistance."""
    if len(df) < 50:
        return 0.0
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    avg_vol = volume.iloc[-50:-1].mean()
    recent_high = close.iloc[-50:-1].max()

    breakout = close.iloc[-1] > recent_high
    vol_spike = volume.iloc[-1] > avg_vol * 1.5

    if breakout and vol_spike:
        ratio = float(volume.iloc[-1] / avg_vol)
        return min(1.0, 0.5 + (ratio - 1.5) / 3)
    if breakout:
        return 0.4
    if vol_spike:
        return 0.2
    return 0.0


def score_obv(df: pd.DataFrame) -> float:
    """OBV making new highs while price consolidates = institutional accumulation."""
    if len(df) < 60:
        return 0.0
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    obv = ta.volume.on_balance_volume(close, volume)

    obv_high = obv.iloc[-60:].max()
    obv_current = obv.iloc[-1]
    price_high = close.iloc[-60:].max()
    price_current = close.iloc[-1]

    obv_near_high = obv_current >= obv_high * 0.97
    price_below_high = price_current < price_high * 0.95  # price consolidating

    if obv_near_high and price_below_high:
        return 1.0  # classic accumulation
    if obv_near_high:
        return 0.7
    if obv_current > obv.iloc[-30]:
        return 0.4
    return 0.1


def score_higher_highs_lows(df: pd.DataFrame) -> float:
    """Higher highs and higher lows forming after a downtrend."""
    if len(df) < 40:
        return 0.0
    close = df["Close"].squeeze()
    # Compare two 20-day halves
    first_half  = close.iloc[-40:-20]
    second_half = close.iloc[-20:]
    hh = second_half.max() > first_half.max()
    hl = second_half.min() > first_half.min()
    # Check prior downtrend (price was lower 60 days ago)
    prior_downtrend = len(df) >= 60 and close.iloc[-60] > close.iloc[-40]

    if hh and hl and prior_downtrend:
        return 1.0
    if hh and hl:
        return 0.7
    if hh or hl:
        return 0.4
    return 0.0


def score_breakout_resistance(df: pd.DataFrame) -> float:
    """Breakout above a resistance level tested multiple times."""
    if len(df) < 90:
        return 0.0
    close = df["Close"].squeeze()
    # Resistance = highest close in prior 60-day window (before last 5 days)
    resistance = close.iloc[-90:-5].max()
    current = close.iloc[-1]

    if current > resistance:
        # How many times was resistance tested?
        near_resistance = ((close.iloc[-90:-5] >= resistance * 0.98) &
                           (close.iloc[-90:-5] <= resistance * 1.02))
        touches = near_resistance.sum()
        if touches >= 3:
            return 1.0
        if touches == 2:
            return 0.75
        return 0.5
    return 0.0


def score_price_coiling(df: pd.DataFrame) -> float:
    """Tight price range / low ATR = coiling before breakout."""
    if len(df) < 40:
        return 0.0
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    close = df["Close"].squeeze()
    atr = ta.volatility.average_true_range(high, low, close, window=14)
    # Compare recent ATR to 60-day average ATR
    if len(atr.dropna()) < 30:
        return 0.0
    recent_atr = atr.iloc[-5:].mean()
    avg_atr    = atr.iloc[-30:].mean()
    if avg_atr == 0:
        return 0.0
    compression = recent_atr / avg_atr
    # Lower compression = tighter coil = higher score
    if compression < 0.5:
        return 1.0
    if compression < 0.7:
        return 0.75
    if compression < 0.85:
        return 0.5
    return 0.1


def score_fibonacci(df: pd.DataFrame) -> float:
    """Price holding at 38.2% or 61.8% Fibonacci retracement."""
    if len(df) < 90:
        return 0.0
    close = df["Close"].squeeze()
    high = close.iloc[-90:].max()
    low  = close.iloc[-90:].min()
    span = high - low
    if span == 0:
        return 0.0
    current = close.iloc[-1]
    fib_382 = high - 0.382 * span
    fib_618 = high - 0.618 * span

    tol = span * 0.025  # 2.5% tolerance
    at_382 = abs(current - fib_382) <= tol
    at_618 = abs(current - fib_618) <= tol

    if at_618:
        return 1.0  # deeper retrace hold = stronger
    if at_382:
        return 0.75
    return 0.0


def compute_technical_score(df: pd.DataFrame) -> dict:
    """Return individual signal scores and weighted composite."""
    signals = {
        "Golden Cross":          (score_golden_cross(df),       0.10),
        "Price vs 200 SMA":      (score_price_vs_200sma(df),    0.08),
        "Bounce off 50 SMA":     (score_bounce_off_50sma(df),   0.06),
        "MA Stacking":           (score_ma_stacking(df),        0.08),
        "MACD Crossover":        (score_macd(df),               0.10),
        "RSI Recovery":          (score_rsi(df),                0.09),
        "Stochastic":            (score_stochastic(df),         0.06),
        "Rate of Change":        (score_roc(df),                0.05),
        "Volume Accumulation":   (score_volume_accumulation(df),0.09),
        "Volume Breakout":       (score_volume_breakout(df),    0.10),
        "OBV Accumulation":      (score_obv(df),                0.07),
        "Higher Highs/Lows":     (score_higher_highs_lows(df),  0.04),
        "Resistance Breakout":   (score_breakout_resistance(df),0.04),
        "Price Coiling":         (score_price_coiling(df),      0.02),
        "Fibonacci Support":     (score_fibonacci(df),          0.02),
    }
    total_weight = sum(w for _, w in signals.values())
    composite = sum(s * w for s, w in signals.values()) / total_weight
    return {
        "composite": round(composite, 4),
        "signals": {k: round(v[0], 4) for k, v in signals.items()},
    }
