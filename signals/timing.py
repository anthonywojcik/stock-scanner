"""
Timing & setup quality signals.
Earnings catalysts, seasonality, post-earnings dip, consolidation age.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def score_earnings_catalyst(info: dict) -> float:
    """Earnings date upcoming within 30 days = pre-earnings opportunity."""
    try:
        earnings_date = None
        raw = info.get("earningsTimestamp") or info.get("earningsDate")
        if raw:
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
            if isinstance(raw, (int, float)):
                earnings_date = datetime.fromtimestamp(raw)
            elif isinstance(raw, datetime):
                earnings_date = raw

        if earnings_date is None:
            return 0.3

        days_away = (earnings_date - datetime.now()).days
        if 5 <= days_away <= 14:
            return 1.0   # imminent catalyst
        if 14 < days_away <= 30:
            return 0.8
        if 0 <= days_away < 5:
            return 0.6   # too close — binary event risk
        if days_away < 0:
            return 0.2   # just passed
        return 0.3
    except Exception:
        return 0.3


def score_seasonality(info: dict) -> float:
    """Simple month-based seasonality proxy."""
    month = datetime.now().month
    sector = (info.get("sector") or "").lower()

    # Semis / tech: historically strong Q4 (Oct-Dec) and Q1
    if "technology" in sector or "semiconductor" in sector:
        if month in [10, 11, 12, 1, 2]:
            return 0.85
        if month in [3, 4]:
            return 0.6
        return 0.4

    # Energy: strong Q4 / winter
    if "energy" in sector:
        if month in [10, 11, 12]:
            return 0.8
        if month in [6, 7, 8]:
            return 0.3
        return 0.5

    # Retail/Consumer: strong pre-holiday (Oct-Dec)
    if "consumer" in sector:
        if month in [10, 11, 12]:
            return 0.8
        if month in [1, 2]:
            return 0.35
        return 0.5

    # Healthcare: relatively stable seasonality
    if "health" in sector:
        return 0.55

    # Default: Jan effect (small caps), summer doldrums in Aug
    if month == 1:
        return 0.7
    if month in [7, 8]:
        return 0.35
    return 0.5


def score_post_earnings_dip(df: pd.DataFrame | None, info: dict) -> float:
    """
    Stock dropped 5%+ on strong earnings = best entry.
    Strong earnings proxy: earnings growth > 0.
    """
    if df is None or df.empty or len(df) < 10:
        return 0.3

    close = df["Close"].squeeze()
    # Check for a significant dip in the last 10 trading days
    recent_low  = float(close.iloc[-10:].min())
    recent_high = float(close.iloc[-10:].max())
    current     = float(close.iloc[-1])

    dip_pct = (recent_high - recent_low) / recent_high

    # Was the dip significant (5%+)?
    if dip_pct < 0.05:
        return 0.2  # no meaningful dip

    # Is the stock now recovering (close > recent_low by 2%+)?
    recovering = current > recent_low * 1.02

    # Were earnings good? Use earningsGrowth or positive ROE
    earnings_good = (
        (info.get("earningsGrowth") or 0) > 0 or
        (info.get("returnOnEquity") or 0) > 0.10
    )

    if dip_pct >= 0.08 and recovering and earnings_good:
        return 1.0   # classic dip-on-good-earnings setup
    if dip_pct >= 0.05 and recovering:
        return 0.7
    if dip_pct >= 0.05 and not recovering:
        return 0.4   # still falling — wait
    return 0.2


def score_consolidation_age(df: pd.DataFrame | None) -> float:
    """
    Stock that consolidated 6-12 months after a big run is primed.
    Proxy: price within 15% of 52-week high but low recent ATR.
    """
    if df is None or df.empty or len(df) < 60:
        return 0.3

    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    high_52w = float(close.max())
    current  = float(close.iloc[-1])
    pct_from_high = (high_52w - current) / high_52w

    # ATR-based volatility in recent vs 6-month avg
    import ta
    atr = ta.volatility.average_true_range(high, low, close, window=14)
    if len(atr.dropna()) < 30:
        return 0.3
    recent_atr = float(atr.iloc[-10:].mean())
    avg_atr    = float(atr.iloc[-60:].mean())
    compression = recent_atr / avg_atr if avg_atr > 0 else 1.0

    if pct_from_high < 0.15 and compression < 0.7:
        return 1.0   # near highs + compressed = coiling for breakout
    if pct_from_high < 0.15:
        return 0.65
    if pct_from_high < 0.30 and compression < 0.7:
        return 0.5
    if pct_from_high > 0.50:
        return 0.2   # far from highs — needs more repair
    return 0.35


def compute_timing_score(df: pd.DataFrame | None, info: dict) -> dict:
    signals = {
        "Earnings Catalyst":      (score_earnings_catalyst(info),          0.30),
        "Seasonality":            (score_seasonality(info),                0.20),
        "Post-Earnings Dip":      (score_post_earnings_dip(df, info),      0.30),
        "Consolidation Setup":    (score_consolidation_age(df),            0.20),
    }
    total_weight = sum(w for _, w in signals.values())
    composite = sum(s * w for s, w in signals.values()) / total_weight
    return {
        "composite": round(composite, 4),
        "signals": {k: round(v[0], 4) for k, v in signals.items()},
    }
