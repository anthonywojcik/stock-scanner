import time
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from bs4 import BeautifulSoup
from config import FALLBACK_TICKERS, SECTOR_ETFS


@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception:
        return FALLBACK_TICKERS


@st.cache_data(ttl=3600)
def get_price_data(ticker: str, period: str = "1y") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty or len(df) < 60:
            return None
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_price_data_batch(tickers: list, period: str = "1y") -> dict:
    """Batch download — returns {ticker: DataFrame}."""
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
        result = {}
        for t in tickers:
            try:
                if len(tickers) == 1:
                    df = raw.copy()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.droplevel(1)
                else:
                    df = raw[t].copy()
                df.dropna(how="all", inplace=True)
                if not df.empty and len(df) >= 60:
                    result[t] = df
            except Exception:
                pass
        return result
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def get_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def get_earnings_history(ticker: str) -> pd.DataFrame | None:
    try:
        t = yf.Ticker(ticker)
        hist = t.earnings_history
        if hist is None or hist.empty:
            return None
        return hist
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_institutional_holders(ticker: str) -> pd.DataFrame | None:
    try:
        t = yf.Ticker(ticker)
        holders = t.institutional_holders
        if holders is None or holders.empty:
            return None
        return holders
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_insider_transactions(ticker: str) -> pd.DataFrame | None:
    try:
        t = yf.Ticker(ticker)
        insiders = t.insider_transactions
        if insiders is None or insiders.empty:
            return None
        return insiders
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_upgrades_downgrades(ticker: str) -> pd.DataFrame | None:
    try:
        t = yf.Ticker(ticker)
        ud = t.upgrades_downgrades
        if ud is None or ud.empty:
            return None
        return ud.head(20)
    except Exception:
        return None


@st.cache_data(ttl=86400)
def get_finviz_data(ticker: str) -> dict:
    """Scrape Finviz stats for short interest, analyst rec, price target."""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}&ty=c&ta=1&p=d"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
        result = {}
        for td in soup.find_all("td", class_="snapshot-td2"):
            label_td = td.find_previous_sibling("td", class_="snapshot-td2-cp")
            if label_td:
                result[label_td.get_text(strip=True)] = td.get_text(strip=True)
        return result
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def get_vix_data(period: str = "6mo") -> pd.DataFrame | None:
    try:
        df = yf.download("^VIX", period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_spy_data(period: str = "6mo") -> pd.DataFrame | None:
    try:
        df = yf.download("SPY", period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_sector_performance() -> dict:
    """Return 1-month and 3-month returns for each sector ETF."""
    results = {}
    for sector, etf in SECTOR_ETFS.items():
        try:
            df = yf.download(etf, period="3mo", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 5:
                continue
            close = df["Close"].dropna()
            ret_1m = float((close.iloc[-1] / close.iloc[max(0, len(close)-22)] - 1) * 100)
            ret_3m = float((close.iloc[-1] / close.iloc[0] - 1) * 100)
            results[sector] = {"etf": etf, "return_1m": ret_1m, "return_3m": ret_3m}
        except Exception:
            pass
    return results


@st.cache_data(ttl=3600)
def get_dollar_index(period: str = "3mo") -> pd.DataFrame | None:
    try:
        df = yf.download("DX-Y.NYB", period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_pcr_data() -> float | None:
    """Fetch put/call ratio from CBOE."""
    try:
        url = "https://www.cboe.com/us/options/market_statistics/daily/"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if cells and "total" in cells[0].get_text(strip=True).lower():
                for i, cell in enumerate(cells):
                    if "put" in cell.get_text(strip=True).lower() and i + 1 < len(cells):
                        try:
                            return float(cells[i + 1].get_text(strip=True))
                        except Exception:
                            pass
        return None
    except Exception:
        return None


def get_ticker_info_safe(ticker: str) -> dict:
    """Return yfinance info without crashing."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}
