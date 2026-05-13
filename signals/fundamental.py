"""
Fundamental signal scoring — returns float 0.0-1.0 per sub-signal.
Uses yfinance info dict and earnings history.
"""
import numpy as np
import pandas as pd


def score_eps_revenue_beat(info: dict, earnings_hist: pd.DataFrame | None) -> float:
    """Beat on both EPS and revenue."""
    if earnings_hist is not None and not earnings_hist.empty:
        try:
            # earnings_history has epsEstimate, epsActual columns
            hist = earnings_hist.sort_index(ascending=False)
            if "epsEstimate" in hist.columns and "epsActual" in hist.columns:
                last = hist.dropna(subset=["epsEstimate", "epsActual"]).head(4)
                if len(last) == 0:
                    pass
                else:
                    beats = (last["epsActual"] > last["epsEstimate"]).sum()
                    if beats >= 3:
                        return 1.0
                    if beats == 2:
                        return 0.65
                    if beats == 1:
                        return 0.35
                    return 0.0
        except Exception:
            pass
    # Fallback to yfinance info fields
    surprise = info.get("earningsSurprisePercent") or info.get("earningsQuarterlyGrowth")
    if surprise is not None:
        if surprise > 0.10:
            return 0.9
        if surprise > 0.05:
            return 0.7
        if surprise > 0:
            return 0.5
        return 0.1
    return 0.3  # neutral when no data


def score_guidance(info: dict) -> float:
    """Revenue guidance — raised is bullish."""
    # yfinance doesn't surface guidance directly; proxy via revenue growth trend
    rev_growth = info.get("revenueGrowth")
    if rev_growth is None:
        return 0.3
    if rev_growth > 0.25:
        return 1.0
    if rev_growth > 0.15:
        return 0.8
    if rev_growth > 0.05:
        return 0.55
    if rev_growth > 0:
        return 0.35
    return 0.0


def score_gross_margin(info: dict) -> float:
    """Expanding gross margins are a quality signal."""
    gm = info.get("grossMargins")
    if gm is None:
        return 0.3
    if gm > 0.60:
        return 1.0
    if gm > 0.40:
        return 0.75
    if gm > 0.20:
        return 0.5
    if gm > 0:
        return 0.25
    return 0.0


def score_revenue_acceleration(info: dict) -> float:
    """Revenue growth rate increasing (acceleration)."""
    qtrly = info.get("revenueGrowth")
    yearly = info.get("earningsGrowth")
    if qtrly is None:
        return 0.3
    # Simple proxy: quarterly growth vs yearly — acceleration if qtrly > yearly
    if yearly is not None and qtrly > yearly and qtrly > 0:
        return min(1.0, 0.5 + qtrly)
    if qtrly > 0.20:
        return 0.8
    if qtrly > 0.10:
        return 0.6
    if qtrly > 0:
        return 0.4
    return 0.1


def score_earnings_surprise(info: dict, earnings_hist: pd.DataFrame | None) -> float:
    """Consistently beating estimates by wide margins = sandbagging."""
    if earnings_hist is not None and not earnings_hist.empty:
        try:
            if "epsEstimate" in earnings_hist.columns and "epsActual" in earnings_hist.columns:
                h = earnings_hist.dropna(subset=["epsEstimate", "epsActual"]).head(4)
                if len(h) > 0:
                    # Only calculate where estimate != 0
                    h = h[h["epsEstimate"] != 0].copy()
                    if len(h) > 0:
                        h["surprise_pct"] = (h["epsActual"] - h["epsEstimate"]) / h["epsEstimate"].abs()
                        avg_surprise = h["surprise_pct"].mean()
                        if avg_surprise > 0.15:
                            return 1.0
                        if avg_surprise > 0.05:
                            return 0.7
                        if avg_surprise > 0:
                            return 0.45
                        return 0.1
        except Exception:
            pass
    return 0.3


def score_cash_runway(info: dict) -> float:
    """Sufficient cash, low debt burden."""
    free_cashflow = info.get("freeCashflow") or 0
    total_cash    = info.get("totalCash") or 0
    total_debt    = info.get("totalDebt") or 0
    market_cap    = info.get("marketCap") or 1

    if total_cash == 0:
        return 0.3
    cash_to_debt = total_cash / max(total_debt, 1)
    cash_to_mcap = total_cash / market_cap

    score = 0.0
    if cash_to_debt > 1.5:
        score += 0.5
    elif cash_to_debt > 0.5:
        score += 0.3
    if free_cashflow > 0:
        score += 0.3
    if cash_to_mcap > 0.1:
        score += 0.2
    return min(1.0, score)


def score_debt_to_cashflow(info: dict) -> float:
    """Manageable debt relative to operating cash flow."""
    op_cf     = info.get("operatingCashflow") or 0
    total_debt = info.get("totalDebt") or 0
    if op_cf <= 0:
        return 0.2
    ratio = total_debt / op_cf
    if ratio < 1:
        return 1.0
    if ratio < 2:
        return 0.75
    if ratio < 4:
        return 0.5
    if ratio < 8:
        return 0.25
    return 0.0


def score_insider_buying(insider_tx: pd.DataFrame | None) -> float:
    """Executives buying shares on the open market (not options)."""
    if insider_tx is None or insider_tx.empty:
        return 0.3
    try:
        recent = insider_tx.head(20)
        buy_col = None
        for col in recent.columns:
            if "transaction" in col.lower() or "type" in col.lower():
                buy_col = col
                break
        if buy_col is None:
            return 0.3
        buys = recent[recent[buy_col].str.contains("Buy|Purchase", case=False, na=False)]
        sales = recent[recent[buy_col].str.contains("Sale|Sell", case=False, na=False)]
        if len(buys) == 0 and len(sales) == 0:
            return 0.3
        ratio = len(buys) / max(1, len(buys) + len(sales))
        if ratio > 0.7:
            return 1.0
        if ratio > 0.4:
            return 0.6
        return max(0.0, ratio)
    except Exception:
        return 0.3



def compute_fundamental_score(info: dict, earnings_hist: pd.DataFrame | None,
                               insider_tx: pd.DataFrame | None) -> dict:
    signals = {
        "EPS & Revenue Beat":    (score_eps_revenue_beat(info, earnings_hist),     0.18),
        "Revenue Guidance":      (score_guidance(info),                            0.12),
        "Gross Margin Quality":  (score_gross_margin(info),                        0.10),
        "Revenue Acceleration":  (score_revenue_acceleration(info),                0.12),
        "Earnings Surprise":     (score_earnings_surprise(info, earnings_hist),    0.15),
        "Cash Runway":           (score_cash_runway(info),                         0.12),
        "Debt/Cash Flow":        (score_debt_to_cashflow(info),                    0.10),
        "Insider Buying":        (score_insider_buying(insider_tx),                0.11),
    }
    total_weight = sum(w for _, w in signals.values())
    composite = sum(s * w for s, w in signals.values()) / total_weight
    return {
        "composite": round(composite, 4),
        "signals": {k: round(v[0], 4) for k, v in signals.items()},
    }
