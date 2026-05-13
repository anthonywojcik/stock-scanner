"""
Sentiment & positioning signals.
Put/call ratio, short interest, analyst consensus, media contrarian.
"""
import numpy as np
import pandas as pd


def score_put_call_ratio(pcr: float | None) -> float:
    """
    High put/call = excessive pessimism = contrarian buy.
    Normal range 0.6-1.0. >1.2 = extreme fear.
    """
    if pcr is None:
        return 0.4
    if pcr > 1.3:
        return 1.0   # extreme pessimism = contrarian buy signal
    if pcr > 1.1:
        return 0.8
    if pcr > 0.9:
        return 0.6
    if pcr > 0.7:
        return 0.45
    return 0.3       # low put/call = complacency


def score_short_interest_sentiment(info: dict, finviz: dict) -> float:
    """
    High short interest = fuel for squeeze.
    Combined with positive catalyst = violent upside.
    (Counterpart to fundamental score which penalizes high short.)
    """
    short_pct = info.get("shortPercentOfFloat")
    fv_short = finviz.get("Short Float")
    if fv_short and "%" in str(fv_short):
        try:
            short_pct = float(str(fv_short).replace("%", "").strip()) / 100
        except (ValueError, TypeError):
            pass

    if short_pct is None:
        return 0.3

    if short_pct > 0.30:
        return 1.0   # massive short interest = massive squeeze potential
    if short_pct > 0.20:
        return 0.85
    if short_pct > 0.10:
        return 0.60
    if short_pct > 0.05:
        return 0.35
    return 0.15


def score_analyst_consensus_contrarian(finviz: dict, upgrades: pd.DataFrame | None) -> float:
    """
    When most analysts have Sell/Hold but fundamentals are improving,
    any upgrade triggers outsized moves. Contrarian signal.
    """
    rec = finviz.get("Recom.")
    if rec:
        try:
            rec_val = float(rec)
            # 1=Strong Buy, 3=Hold, 5=Strong Sell
            # Contrarian: bearish consensus (3.5+) with improving fundamentals
            if rec_val >= 3.5:
                return 0.9   # bears are wrong setup
            if rec_val >= 3.0:
                return 0.65
            if rec_val >= 2.5:
                return 0.45
            if rec_val >= 2.0:
                return 0.3
            return 0.2       # universally loved = less squeeze potential
        except (ValueError, TypeError):
            pass

    if upgrades is not None and not upgrades.empty:
        try:
            grade_col = None
            for col in upgrades.columns:
                if "grade" in col.lower() or "to" in col.lower():
                    grade_col = col
                    break
            if grade_col:
                text = " ".join(upgrades[grade_col].head(10).astype(str).tolist()).lower()
                bear_count = sum(1 for w in ["sell", "underperform", "underweight", "hold"] if w in text)
                bull_count = sum(1 for w in ["buy", "outperform", "overweight", "strong buy"] if w in text)
                if bear_count > bull_count:
                    return 0.8   # bears dominant — contrarian setup
                if bull_count > bear_count * 2:
                    return 0.2   # over-loved
                return 0.45
        except Exception:
            pass
    return 0.35


def score_media_sentiment(info: dict, finviz: dict) -> float:
    """
    Contrarian neglect proxy: under-followed stocks with improving fundamentals
    get re-rated violently when discovered. Large caps get neutral (already known).
    """
    avg_vol = info.get("averageVolume") or info.get("averageVolume10days") or 0
    mkt_cap = info.get("marketCap") or 0

    # Large/mega caps are widely covered — neutral, not penalized
    if mkt_cap > 50e9 or avg_vol > 20e6:
        return 0.50
    if avg_vol > 5e6:
        return 0.55
    if avg_vol > 1e6:
        return 0.65  # mid-cap: less media coverage = more re-rating potential
    return 0.80      # small cap: ignored = contrarian opportunity


def compute_sentiment_score(pcr: float | None,
                             info: dict,
                             finviz: dict,
                             upgrades: pd.DataFrame | None) -> dict:
    signals = {
        "Put/Call Ratio":         (score_put_call_ratio(pcr),                              0.25),
        "Short Interest Fuel":    (score_short_interest_sentiment(info, finviz),           0.30),
        "Contrarian Consensus":   (score_analyst_consensus_contrarian(finviz, upgrades),   0.25),
        "Media Neglect Factor":   (score_media_sentiment(info, finviz),                    0.20),
    }
    total_weight = sum(w for _, w in signals.values())
    composite = sum(s * w for s, w in signals.values()) / total_weight
    return {
        "composite": round(composite, 4),
        "signals": {k: round(v[0], 4) for k, v in signals.items()},
    }
