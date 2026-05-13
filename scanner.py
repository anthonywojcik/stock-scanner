"""
Composite scorer and stock scanner.
Orchestrates all signal modules and returns a ranked list of stocks.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from config import SCORE_WEIGHTS, BUY_SIGNAL_THRESHOLD

from data.fetcher import (
    get_price_data, get_fundamentals, get_earnings_history,
    get_institutional_holders, get_insider_transactions,
    get_upgrades_downgrades, get_finviz_data,
    get_vix_data, get_spy_data, get_sector_performance,
    get_dollar_index, get_pcr_data,
)
from signals.technical    import compute_technical_score
from signals.fundamental  import compute_fundamental_score
from signals.institutional import compute_institutional_score
from signals.macro        import compute_macro_score
from signals.sentiment    import compute_sentiment_score
from signals.timing       import compute_timing_score


def score_stock(ticker: str,
                spy_df=None,
                vix_df=None,
                dollar_df=None,
                sector_perf=None,
                pcr=None) -> dict | None:
    """
    Compute the full rally confidence score for a single ticker.
    Returns a result dict or None if data is unavailable.
    """
    # --- Price data ---
    df = get_price_data(ticker, period="1y")
    if df is None or df.empty:
        return None

    # --- Fundamental & holder data ---
    info         = get_fundamentals(ticker)
    earnings     = get_earnings_history(ticker)
    holders      = get_institutional_holders(ticker)
    insider_tx   = get_insider_transactions(ticker)
    upgrades     = get_upgrades_downgrades(ticker)
    finviz       = get_finviz_data(ticker)

    # --- Signal scores ---
    tech   = compute_technical_score(df)
    fund   = compute_fundamental_score(info, earnings, insider_tx)
    inst   = compute_institutional_score(holders, upgrades, info, finviz)
    macro  = compute_macro_score(spy_df, vix_df, dollar_df, sector_perf or {}, info)
    senti  = compute_sentiment_score(pcr, info, finviz, upgrades)
    timing = compute_timing_score(df, info)

    # --- Composite score (0-100) ---
    composite = (
        tech["composite"]   * SCORE_WEIGHTS["technical"]     +
        fund["composite"]   * SCORE_WEIGHTS["fundamental"]   +
        inst["composite"]   * SCORE_WEIGHTS["institutional"] +
        macro["composite"]  * SCORE_WEIGHTS["macro"]         +
        senti["composite"]  * SCORE_WEIGHTS["sentiment"]     +
        timing["composite"] * SCORE_WEIGHTS["timing"]
    ) * 100

    # --- Extract key display metrics ---
    price   = info.get("currentPrice") or info.get("regularMarketPrice")
    if price is None and not df.empty:
        price = float(df["Close"].iloc[-1])

    change_1d = None
    if len(df) >= 2:
        change_1d = float((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100)

    change_1m = None
    if len(df) >= 22:
        change_1m = float((df["Close"].iloc[-1] / df["Close"].iloc[-22] - 1) * 100)

    return {
        "ticker":       ticker,
        "name":         info.get("shortName") or info.get("longName") or ticker,
        "sector":       info.get("sector", "Unknown"),
        "industry":     info.get("industry", "Unknown"),
        "market_cap":   info.get("marketCap"),
        "price":        round(price, 2) if price else None,
        "change_1d":    round(change_1d, 2) if change_1d is not None else None,
        "change_1m":    round(change_1m, 2) if change_1m is not None else None,
        "composite":    round(composite, 1),
        "is_buy":       composite >= BUY_SIGNAL_THRESHOLD,
        "categories": {
            "Technical":     round(tech["composite"]   * 100, 1),
            "Fundamental":   round(fund["composite"]   * 100, 1),
            "Institutional": round(inst["composite"]   * 100, 1),
            "Macro":         round(macro["composite"]  * 100, 1),
            "Sentiment":     round(senti["composite"]  * 100, 1),
            "Timing":        round(timing["composite"] * 100, 1),
        },
        "signals": {
            "technical":     tech["signals"],
            "fundamental":   fund["signals"],
            "institutional": inst["signals"],
            "macro":         macro["signals"],
            "sentiment":     senti["signals"],
            "timing":        timing["signals"],
        },
        "meta": {
            "pe_ratio":       info.get("trailingPE"),
            "forward_pe":     info.get("forwardPE"),
            "market_cap_raw": info.get("marketCap"),
            "short_float":    info.get("shortPercentOfFloat"),
            "analyst_target": info.get("targetMeanPrice"),
            "beta":           info.get("beta"),
            "52w_high":       info.get("fiftyTwoWeekHigh"),
            "52w_low":        info.get("fiftyTwoWeekLow"),
        },
        "price_df": df,
    }


def scan_universe(tickers: list,
                  progress_callback=None,
                  sector_filter: str = "All",
                  min_market_cap: float = 0,
                  max_tickers: int = 500) -> list[dict]:
    """
    Scan a list of tickers. Returns sorted list of result dicts.
    progress_callback(i, total, ticker) called for each stock.
    """
    # Fetch shared macro data once
    spy_df     = get_spy_data(period="1y")
    vix_df     = get_vix_data(period="6mo")
    dollar_df  = get_dollar_index(period="3mo")
    sector_perf = get_sector_performance()
    pcr        = get_pcr_data()

    results = []
    tickers = tickers[:max_tickers]

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, len(tickers), ticker)
        try:
            result = score_stock(
                ticker,
                spy_df=spy_df,
                vix_df=vix_df,
                dollar_df=dollar_df,
                sector_perf=sector_perf,
                pcr=pcr,
            )
            if result is None:
                continue
            # Apply filters
            if sector_filter != "All" and result["sector"] != sector_filter:
                continue
            if min_market_cap > 0 and (result["market_cap"] or 0) < min_market_cap:
                continue
            results.append(result)
        except Exception:
            continue

    results.sort(key=lambda x: x["composite"], reverse=True)
    return results


def two_pass_scan(
    universe: "dict[str, str] | list[str]",
    progress_callback=None,
    deep_n: int = 30,
) -> list[dict]:
    """
    Fast prescriptive scan optimised for the auto-scout feature.

    `universe` may be either:
      - a dict  {ticker: source_label}  (from build_dynamic_universe)
      - a plain list of tickers

    Pass 1 (fast): batch-downloads all price data, scores every ticker on
    technical + macro signals only — pure math, no per-stock API calls.
    Keeps the top `deep_n` candidates.

    Pass 2 (deep): runs the full 41-signal analysis on those candidates.

    progress_callback(i, total, ticker, phase) — phase is 1 or 2.
    """
    from data.fetcher import get_price_data_batch
    from signals.technical import compute_technical_score

    # Normalise universe to (ticker_list, source_map)
    if isinstance(universe, dict):
        source_map: dict[str, str] = universe
        tickers: list[str] = list(universe.keys())
    else:
        source_map = {}
        tickers = list(universe)

    # ── Shared macro data (fetched once) ─────────────────────────────────────
    spy_df      = get_spy_data(period="1y")
    vix_df      = get_vix_data(period="6mo")
    dollar_df   = get_dollar_index(period="3mo")
    sector_perf = get_sector_performance()
    pcr         = get_pcr_data()

    # ── Pass 1: technical filter ──────────────────────────────────────────────
    if progress_callback:
        progress_callback(0, len(tickers), "Downloading all price data…", 1)

    price_data = get_price_data_batch(tickers, period="1y")

    pass1 = []
    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i + 1, len(tickers), ticker, 1)
        df = price_data.get(ticker)
        if df is None or df.empty or len(df) < 60:
            continue
        try:
            tech  = compute_technical_score(df)
            quick = tech["composite"]
            pass1.append({"ticker": ticker, "quick": quick})
        except Exception:
            continue

    pass1.sort(key=lambda x: x["quick"], reverse=True)
    candidates = [r["ticker"] for r in pass1[:deep_n]]

    # ── Pass 2: full deep analysis ────────────────────────────────────────────
    full_results = []
    for i, ticker in enumerate(candidates):
        if progress_callback:
            progress_callback(i + 1, len(candidates), ticker, 2)
        try:
            result = score_stock(
                ticker,
                spy_df=spy_df, vix_df=vix_df,
                dollar_df=dollar_df, sector_perf=sector_perf, pcr=pcr,
            )
            if result:
                result["source"] = source_map.get(ticker, "Auto Scout")
                full_results.append(result)
        except Exception:
            continue

    full_results.sort(key=lambda x: x["composite"], reverse=True)
    return full_results


def get_macro_snapshot() -> dict:
    """Return current macro environment data for the dashboard."""
    spy_df      = get_spy_data(period="6mo")
    vix_df      = get_vix_data(period="6mo")
    dollar_df   = get_dollar_index(period="3mo")
    sector_perf = get_sector_performance()
    pcr         = get_pcr_data()

    vix_current = None
    vix_change  = None
    if vix_df is not None and not vix_df.empty:
        close = vix_df["Close"].squeeze()
        vix_current = round(float(close.iloc[-1]), 2)
        if len(close) >= 5:
            vix_change = round(float(close.iloc[-1] - close.iloc[-5]), 2)

    spy_ret_1m = None
    if spy_df is not None and not spy_df.empty:
        close = spy_df["Close"].squeeze()
        if len(close) >= 22:
            spy_ret_1m = round(float((close.iloc[-1] / close.iloc[-22] - 1) * 100), 2)

    return {
        "vix_current":  vix_current,
        "vix_change":   vix_change,
        "spy_ret_1m":   spy_ret_1m,
        "sector_perf":  sector_perf,
        "pcr":          pcr,
        "vix_df":       vix_df,
        "spy_df":       spy_df,
    }
