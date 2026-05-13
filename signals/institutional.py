"""
Institutional signal scoring.
Uses yfinance institutional holders, analyst upgrades/downgrades, and Finviz data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def score_institutional_ownership(holders: pd.DataFrame | None, info: dict) -> float:
    """High and growing institutional ownership."""
    # Check yfinance info fields
    inst_pct = info.get("heldPercentInstitutions")
    if inst_pct is not None:
        if inst_pct > 0.80:
            return 1.0
        if inst_pct > 0.60:
            return 0.75
        if inst_pct > 0.40:
            return 0.5
        if inst_pct > 0.20:
            return 0.3
        return 0.1

    if holders is None or holders.empty:
        return 0.3
    # Check if top holders increased positions recently
    try:
        if "Shares" in holders.columns and "% Out" in holders.columns:
            total_pct = holders["% Out"].sum()
            if total_pct > 0.70:
                return 0.85
            if total_pct > 0.50:
                return 0.65
            if total_pct > 0.30:
                return 0.45
        return 0.3
    except Exception:
        return 0.3


def score_analyst_upgrades(upgrades: pd.DataFrame | None, finviz: dict) -> float:
    """Recent analyst upgrades and raised price targets."""
    score = 0.3

    # Use Finviz analyst recommendation (1=Strong Buy, 5=Strong Sell)
    rec = finviz.get("Recom.")
    if rec:
        try:
            rec_val = float(rec)
            if rec_val <= 1.5:
                score = max(score, 1.0)
            elif rec_val <= 2.0:
                score = max(score, 0.8)
            elif rec_val <= 2.5:
                score = max(score, 0.6)
            elif rec_val <= 3.0:
                score = max(score, 0.4)
            else:
                score = max(score, 0.1)
        except (ValueError, TypeError):
            pass

    if upgrades is None or upgrades.empty:
        return score

    try:
        # Look at last 90 days of analyst actions
        cutoff = datetime.now() - timedelta(days=90)
        if upgrades.index.tz is not None:
            cutoff = pd.Timestamp(cutoff, tz=upgrades.index.tz)
        else:
            cutoff = pd.Timestamp(cutoff)

        recent = upgrades[upgrades.index >= cutoff] if not upgrades.empty else upgrades.head(10)
        if recent.empty:
            recent = upgrades.head(10)

        upgrade_keywords = ["Upgrade", "Initiated", "Raised", "Buy", "Outperform", "Overweight"]
        downgrade_keywords = ["Downgrade", "Sell", "Underperform", "Underweight", "Lowered"]

        grade_col = None
        for col in recent.columns:
            if "grade" in col.lower() or "action" in col.lower() or "to" in col.lower():
                grade_col = col
                break

        if grade_col:
            text = recent[grade_col].str.cat(sep=" ")
            upgrades_count   = sum(1 for k in upgrade_keywords   if k.lower() in text.lower())
            downgrades_count = sum(1 for k in downgrade_keywords if k.lower() in text.lower())

            if upgrades_count > downgrades_count:
                analyst_score = min(1.0, 0.5 + 0.1 * upgrades_count)
                score = max(score, analyst_score)
            elif downgrades_count > upgrades_count:
                score = min(score, 0.3)
    except Exception:
        pass

    return round(score, 4)


def score_short_squeeze_setup(info: dict, finviz: dict) -> float:
    """High short interest + positive catalyst = squeeze potential."""
    short_pct = info.get("shortPercentOfFloat")

    # Try Finviz for more accurate short float
    fv_short = finviz.get("Short Float")
    if fv_short and "%" in str(fv_short):
        try:
            short_pct = float(str(fv_short).replace("%", "").strip()) / 100
        except (ValueError, TypeError):
            pass

    if short_pct is None:
        return 0.2

    if short_pct > 0.30:
        return 1.0   # extreme short — massive squeeze fuel
    if short_pct > 0.20:
        return 0.85
    if short_pct > 0.10:
        return 0.55
    if short_pct > 0.05:
        return 0.3
    return 0.1  # barely shorted


def score_price_target_upside(info: dict, finviz: dict) -> float:
    """Analyst price target significantly above current price."""
    current = info.get("currentPrice") or info.get("regularMarketPrice")
    target  = info.get("targetMeanPrice")

    # Try Finviz target price
    fv_target = finviz.get("Target Price")
    if fv_target and current:
        try:
            fv_val = float(str(fv_target).replace("$", "").replace(",", "").strip())
            if fv_val > 0:
                target = fv_val
        except (ValueError, TypeError):
            pass

    if not current or not target or current <= 0:
        return 0.3

    upside = (target - current) / current
    if upside > 0.50:
        return 1.0
    if upside > 0.30:
        return 0.85
    if upside > 0.15:
        return 0.65
    if upside > 0.05:
        return 0.45
    if upside > 0:
        return 0.3
    return 0.1  # downside to target


def score_index_inclusion_risk(info: dict) -> float:
    """
    Proxy for index inclusion potential: small-to-mid cap with strong
    performance and growing institutional interest tends to get added.
    """
    market_cap = info.get("marketCap") or 0
    inst_pct   = info.get("heldPercentInstitutions") or 0

    # Large caps already in major indices — less upside from inclusion
    if market_cap > 50e9:
        return 0.3
    if market_cap > 10e9 and inst_pct > 0.5:
        return 0.7
    if market_cap > 2e9 and inst_pct > 0.3:
        return 0.85
    if market_cap > 0.5e9:
        return 0.6
    return 0.3


def compute_institutional_score(holders: pd.DataFrame | None,
                                 upgrades: pd.DataFrame | None,
                                 info: dict,
                                 finviz: dict) -> dict:
    signals = {
        "Institutional Ownership": (score_institutional_ownership(holders, info),    0.25),
        "Analyst Upgrades":        (score_analyst_upgrades(upgrades, finviz),        0.30),
        "Short Squeeze Setup":     (score_short_squeeze_setup(info, finviz),         0.20),
        "Price Target Upside":     (score_price_target_upside(info, finviz),         0.15),
        "Index Inclusion Proxy":   (score_index_inclusion_risk(info),                0.10),
    }
    total_weight = sum(w for _, w in signals.values())
    composite = sum(s * w for s, w in signals.values()) / total_weight
    return {
        "composite": round(composite, 4),
        "signals": {k: round(v[0], 4) for k, v in signals.items()},
    }
