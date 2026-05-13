"""
Macro & sector environment signal scoring.
Answers: is the broader environment favorable for a rally?
"""
import numpy as np
import pandas as pd


def score_sector_rotation(sector_perf: dict, ticker_sector: str) -> float:
    """Money flowing INTO the stock's sector."""
    if not sector_perf or not ticker_sector:
        return 0.3

    # Normalize sector name
    sector = ticker_sector
    matched = None
    for key in sector_perf:
        if key.lower() in sector.lower() or sector.lower() in key.lower():
            matched = key
            break

    if matched is None:
        # Rank all sectors by 1-month return; if no match, return neutral
        returns = {k: v.get("return_1m", 0) for k, v in sector_perf.items()}
        if not returns:
            return 0.3
        avg = np.mean(list(returns.values()))
        return 0.3

    sector_data = sector_perf[matched]
    ret_1m = sector_data.get("return_1m", 0)
    ret_3m = sector_data.get("return_3m", 0)

    # Rank vs all sectors
    all_1m = sorted([v.get("return_1m", 0) for v in sector_perf.values()], reverse=True)
    rank = all_1m.index(ret_1m) + 1 if ret_1m in all_1m else len(all_1m) // 2
    total = len(all_1m)
    rank_score = 1.0 - (rank - 1) / max(total - 1, 1)  # 1.0 = top sector, 0.0 = bottom

    return_score = min(1.0, max(0.0, (ret_1m + 10) / 20))  # -10% to +10% mapped 0-1
    return round((rank_score * 0.6 + return_score * 0.4), 4)


def score_spy_trend(spy_df: pd.DataFrame | None) -> float:
    """S&P 500 in uptrend = favorable. Downtrend = fight the tape risk."""
    if spy_df is None or spy_df.empty:
        return 0.5
    close = spy_df["Close"].squeeze()
    if len(close) < 50:
        return 0.5
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if len(close) >= 200 else sma50

    above_50  = close.iloc[-1] > sma50.iloc[-1]
    above_200 = close.iloc[-1] > sma200.iloc[-1]
    golden    = sma50.iloc[-1] > sma200.iloc[-1] if len(close) >= 200 else above_50

    # Momentum: 20-day return
    ret_20d = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0

    score = 0.0
    score += 0.35 if above_50  else 0
    score += 0.30 if above_200 else 0
    score += 0.20 if golden     else 0
    score += min(0.15, max(0, ret_20d * 3))
    return round(score, 4)


def score_vix(vix_df: pd.DataFrame | None) -> float:
    """
    VIX elevated then rapidly drops = risk-on flood.
    Low and stable VIX = complacency / mild positive.
    """
    if vix_df is None or vix_df.empty:
        return 0.5
    close = vix_df["Close"].squeeze()
    if len(close) < 10:
        return 0.5

    current  = float(close.iloc[-1])
    peak_30d = float(close.iloc[-30:].max()) if len(close) >= 30 else current
    avg_5d   = float(close.iloc[-5:].mean())

    # VIX dropping from spike = risk-on
    spike_then_drop = peak_30d > 25 and current < peak_30d * 0.75
    if spike_then_drop:
        return 1.0

    if current < 15:
        return 0.65   # low vol = calm market
    if current < 20:
        return 0.75
    if current < 25:
        return 0.5
    if current < 35:
        return 0.3   # elevated fear
    return 0.1        # extreme fear


def score_dollar_trend(dollar_df: pd.DataFrame | None, info: dict) -> float:
    """
    Weak dollar favors international-revenue companies and commodities.
    Strong dollar = headwind for global earners.
    """
    if dollar_df is None or dollar_df.empty:
        return 0.5

    close = dollar_df["Close"].squeeze()
    if len(close) < 20:
        return 0.5

    # Dollar 20-day return
    ret = float(close.iloc[-1] / close.iloc[-20] - 1)
    # Is this company internationally exposed?
    intl_revenue_pct = info.get("revenueGrowth")  # crude proxy — adjust if FMP data available
    country = info.get("country", "US")
    is_intl = country != "United States" or (info.get("totalRevenue", 0) or 0) > 10e9

    dollar_falling = ret < -0.01
    dollar_rising  = ret > 0.01

    if is_intl:
        if dollar_falling:
            return 0.85
        if dollar_rising:
            return 0.25
        return 0.5
    else:
        # Domestic focus — dollar matters less
        return 0.5


def score_market_breadth(sector_perf: dict) -> float:
    """
    Broad participation (most sectors positive) = healthy bull.
    Narrow rally (only 1-2 sectors up) = caution.
    """
    if not sector_perf:
        return 0.4
    positive = sum(1 for v in sector_perf.values() if v.get("return_1m", 0) > 0)
    total = len(sector_perf)
    if total == 0:
        return 0.4
    breadth = positive / total
    if breadth > 0.80:
        return 1.0
    if breadth > 0.60:
        return 0.75
    if breadth > 0.40:
        return 0.5
    if breadth > 0.20:
        return 0.25
    return 0.05


def compute_macro_score(spy_df: pd.DataFrame | None,
                         vix_df: pd.DataFrame | None,
                         dollar_df: pd.DataFrame | None,
                         sector_perf: dict,
                         info: dict) -> dict:
    ticker_sector = info.get("sector", "")
    signals = {
        "Sector Rotation":     (score_sector_rotation(sector_perf, ticker_sector), 0.25),
        "S&P 500 Trend":       (score_spy_trend(spy_df),                           0.30),
        "VIX Environment":     (score_vix(vix_df),                                 0.20),
        "Dollar Trend":        (score_dollar_trend(dollar_df, info),               0.10),
        "Market Breadth":      (score_market_breadth(sector_perf),                 0.15),
    }
    total_weight = sum(w for _, w in signals.values())
    composite = sum(s * w for s, w in signals.values()) / total_weight
    return {
        "composite": round(composite, 4),
        "signals": {k: round(v[0], 4) for k, v in signals.items()},
    }
