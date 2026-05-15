"""
Dynamic universe builder.

Assembles the auto-scout stock list fresh before each scan from 4 layers:

  Layer 1 — AI/Tech/Infra Anchors   (config.AI_TECH_ANCHORS, always included)
  Layer 2 — High Short Interest      (Finviz screener: short float > threshold)
  Layer 3 — Momentum Leaders         (Finviz screener: near 52-week high, above MAs)
  Layer 4 — S&P 500 Top Performers   (top N by 1-month return via yfinance)
  Layer 5 — User Watchlist           (tickers the user adds manually)

Each ticker carries a source label so the UI can show where it came from.
"""
import re
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from bs4 import BeautifulSoup

from config import (
    AI_TECH_ANCHORS,
    DYNAMIC_SHORT_INTEREST_MIN_PCT,
    DYNAMIC_SHORT_INTEREST_N,
    DYNAMIC_MOMENTUM_N,
    DYNAMIC_SP500_TOP_N,
    FALLBACK_TICKERS,
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def _scrape_finviz_screener(url: str, max_n: int) -> list[str]:
    """
    Pull ticker symbols from a Finviz screener URL.
    Looks for <a> tags whose href points to a quote page.
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        soup = BeautifulSoup(resp.text, "lxml")
        tickers = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "quote.ashx?t=" in href:
                raw = href.split("quote.ashx?t=")[1].split("&")[0].strip().upper()
                if _TICKER_RE.match(raw) and raw not in tickers:
                    tickers.append(raw)
            if len(tickers) >= max_n:
                break
        return tickers
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def get_high_short_interest_stocks() -> list[str]:
    """
    Finviz screen: short float > threshold, market cap mid+ (>$2B).
    Orders by short float descending — highest squeeze fuel first.
    """
    pct = DYNAMIC_SHORT_INTEREST_MIN_PCT
    url = (
        f"https://finviz.com/screener.ashx"
        f"?v=111&f=sh_short_o{pct},cap_midover&o=-short"
    )
    results = _scrape_finviz_screener(url, DYNAMIC_SHORT_INTEREST_N)
    return results or []


@st.cache_data(ttl=3600, show_spinner=False)
def get_momentum_stocks() -> list[str]:
    """
    Finviz screen: price within 5% of 52-week high, above 50-day SMA,
    average volume > 500K. Orders by % change descending.
    These are the stocks already in motion — institutions are buying.
    """
    url = (
        "https://finviz.com/screener.ashx"
        "?v=111&f=ta_highlow52w_nh,ta_sma50_pa,cap_midover,sh_avgvol_o500&o=-change"
    )
    results = _scrape_finviz_screener(url, DYNAMIC_MOMENTUM_N)
    return results or []


@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_top_performers() -> list[str]:
    """
    Pull the S&P 500 list, batch-download 1-month price data,
    return the top N by return. Captures sector rotation winners.
    """
    try:
        sp500_df = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )[0]
        all_tickers = (
            sp500_df["Symbol"]
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        # Sample 150 to keep the download fast
        sample = all_tickers[:150]
        raw = yf.download(
            sample, period="1mo", auto_adjust=True,
            progress=False, threads=True,
        )
        if raw.empty:
            return []

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        returns: dict[str, float] = {}
        for t in close.columns:
            col = close[t].dropna()
            if len(col) >= 5:
                returns[t] = float((col.iloc[-1] / col.iloc[0] - 1) * 100)

        top = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in top[: DYNAMIC_SP500_TOP_N]]
    except Exception:
        return []


def build_dynamic_universe(user_tickers: list[str] | None = None) -> dict[str, str]:
    """
    Assemble the full auto-scout universe.

    Returns a dict  {ticker: source_label}  so callers know where each
    stock came from.  Tickers added by multiple layers keep the first
    (highest-priority) label.
    """
    universe: dict[str, str] = {}

    # Layer 1 — always-on AI/Tech/Infra anchors
    for t in AI_TECH_ANCHORS:
        universe[t] = "AI/Tech Core"

    # Layer 2 — high short interest (squeeze fuel)
    for t in get_high_short_interest_stocks():
        if t not in universe:
            universe[t] = "High Short Interest"

    # Layer 3 — momentum leaders (near 52-week highs)
    for t in get_momentum_stocks():
        if t not in universe:
            universe[t] = "Momentum Leader"

    # Layer 4 — S&P 500 top performers (sector rotation)
    for t in get_sp500_top_performers():
        if t not in universe:
            universe[t] = "S&P Top Performer"

    # Layer 5 — user watchlist
    for raw in (user_tickers or []):
        t = raw.strip().upper()
        if t and _TICKER_RE.match(t):
            universe[t] = "Your Watchlist"

    return universe


def universe_summary(universe: dict[str, str]) -> dict[str, int]:
    """Count tickers per source layer."""
    counts: dict[str, int] = {}
    for src in universe.values():
        counts[src] = counts.get(src, 0) + 1
    return counts
