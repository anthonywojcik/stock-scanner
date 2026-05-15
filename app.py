# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import hmac
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from config import BUY_SIGNAL_THRESHOLD, SCORE_WEIGHTS
from data.fetcher import get_sp500_tickers
from data.universe import build_dynamic_universe, universe_summary
from scanner import scan_universe, score_stock, get_macro_snapshot, two_pass_scan

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rally Scout",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tab navigation helper ─────────────────────────────────────────────────────
def navigate_to_tab(label: str):
    js = f"""
    <script>
    (function() {{
        function clickTab() {{
            var tabs = window.parent.document.querySelectorAll('button[role="tab"]');
            for (var t of tabs) {{
                if (t.innerText.trim().indexOf({repr(label)}) !== -1) {{ t.click(); return; }}
            }}
        }}
        setTimeout(clickTab, 120);
    }})();
    </script>
    """
    components.html(js, height=0, scrolling=False)


# ── Design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Layout */
.block-container { padding-top: 1rem !important; max-width: 1400px; }
section[data-testid="stSidebar"] {
    background: #080c14;
    border-right: 1px solid rgba(255,255,255,0.04);
}

/* Cards */
.rs-card-1 {
    background: linear-gradient(135deg, #071710 0%, #0a1d13 100%);
    border: 1.5px solid #00d084;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: 0 0 32px rgba(0,208,132,0.08), 0 2px 12px rgba(0,0,0,0.5);
}
.rs-card-2 {
    background: #0e1520;
    border: 1px solid #1a2d3e;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.rs-card-3 {
    background: #0b1119;
    border: 1px solid #16243a;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}

/* Action badges */
.badge { display:inline-flex; align-items:center; padding:3px 11px; border-radius:6px; font-weight:700; font-size:0.78em; letter-spacing:0.4px; white-space:nowrap; }
.badge-now   { background:#00d084; color:#000; }
.badge-buy   { background:#059669; color:#fff; }
.badge-scale { background:#0284c7; color:#fff; }
.badge-watch { background:#d97706; color:#fff; }
.badge-hold  { background:#1e2a3a; color:#64748b; border:1px solid #607d96; }

/* Signal rows */
.sig-row  { display:flex; align-items:center; margin-bottom:6px; gap:8px; }
.sig-icon { font-size:0.82em; width:16px; flex-shrink:0; }
.sig-text { flex:1; font-size:0.79em; color:#a0b4c8; line-height:1.35; }
.sig-track { width:68px; background:#1a2335; border-radius:2px; height:4px; flex-shrink:0; }
.sig-fill  { height:4px; border-radius:2px; }
.sig-pct   { font-size:0.69em; width:24px; text-align:right; flex-shrink:0; }

/* Info boxes */
.box-bull  { background:rgba(0,208,132,0.07); border-left:3px solid #00d084; padding:14px 18px; border-radius:0 8px 8px 0; margin-bottom:10px; }
.box-bear  { background:rgba(244,63,94,0.07);  border-left:3px solid #f43f5e;  padding:14px 18px; border-radius:0 8px 8px 0; margin-bottom:10px; }
.box-action { background:rgba(0,208,132,0.06); border:1px solid rgba(0,208,132,0.20); border-radius:10px; padding:14px 18px; margin:12px 0; }
.box-trade  { background:#0b1520; border:1px solid #1a3050; border-radius:10px; padding:16px 20px; margin:12px 0; }
.box-risk   { background:rgba(251,146,60,0.06); border:1px solid rgba(251,146,60,0.20); border-radius:8px; padding:10px 14px; margin-top:8px; }

/* Category panels */
.cat-panel { background:#0e1520; border:1px solid #1a2840; border-radius:10px; padding:14px 16px; margin-bottom:14px; }
.cat-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #1a2840; }

/* Pills */
.pill { display:inline-flex; align-items:center; gap:3px; padding:2px 9px; border-radius:10px; font-size:0.69em; white-space:nowrap; }
.pill-source { background:#111d2e; color:#7e9ab8; border:1px solid #1a2840; }
.pill-cat { background:#111d2e; border-radius:5px; padding:2px 8px; font-size:0.71em; margin-right:3px; }

/* Environment banner */
.env-banner { border-radius:12px; padding:18px 22px; margin-bottom:20px; }

/* Metrics */
div[data-testid="metric-container"] {
    background:#0e1520;
    border:1px solid #1a2840;
    border-radius:8px;
    padding:10px 14px;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size:1.25em !important; }

/* Sidebar branding */
.sb-brand { padding:2px 0 14px; }
.sb-brand-title { font-size:1.15em; font-weight:800; color:#e2e8f0; letter-spacing:-0.5px; }
.sb-brand-sub { font-size:0.72em; color:#8aabca; margin-top:1px; }

/* Footer */
.rs-footer { color:#607d96; font-size:0.71em; line-height:1.7; margin-top:36px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.04); }

/* Scan status pill */
.scan-pill { background:#0a1a0f; border:1px solid #1a3a22; border-radius:8px; padding:10px 14px; margin-bottom:12px; }
.scan-pill-label { font-size:0.69em; color:#4ade80; font-weight:700; letter-spacing:0.6px; }
.scan-pill-sub { font-size:0.78em; color:#8aabca; margin-top:2px; }

/* ── Mobile responsive ────────────────────────────────────────────────────── */
@media screen and (max-width: 768px) {
    /* Container */
    .block-container {
        padding-left: 0.55rem !important;
        padding-right: 0.55rem !important;
        padding-top: 0.35rem !important;
        max-width: 100% !important;
    }

    /* Touch-friendly buttons — Apple HIG minimum 44px */
    .stButton > button {
        min-height: 48px !important;
        font-size: 0.95em !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        letter-spacing: 0.2px !important;
    }

    /* Cards — tighter side padding */
    .rs-card-1 { padding: 14px 12px !important; margin-bottom: 14px !important; }
    .rs-card-2 { padding: 13px 12px !important; margin-bottom: 11px !important; }
    .rs-card-3 { padding: 10px 10px !important; margin-bottom:  8px !important; }

    /* Metric tiles — more breathable on 2-col vs 3-col desktop */
    div[data-testid="metric-container"] {
        padding: 9px 11px !important;
        border-radius: 8px !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.15em !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.69em !important;
        letter-spacing: 0.3px !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.73em !important;
    }

    /* Signal rows — readable body text */
    .sig-track { width: 44px !important; }
    .sig-text  { font-size: 0.82em !important; line-height: 1.4 !important; }
    .sig-pct   { font-size: 0.72em !important; }

    /* Trade plan — single column + larger type */
    .box-trade > div[style*="grid"] {
        grid-template-columns: 1fr !important;
        gap: 12px !important;
    }

    /* Box padding — tighter but not cramped */
    .box-action { padding: 12px 13px !important; }
    .box-trade  { padding: 13px 13px !important; }
    .box-bull,
    .box-bear   { padding: 12px 13px !important; }
    .box-risk   { padding: 9px 12px !important;  }

    /* Larger trade-plan label and value text */
    .box-trade [style*="0.69em"] { font-size: 0.73em !important; }
    .box-trade [style*="0.82em"] { font-size: 0.87em !important; }

    /* Conviction card "WHY" body text */
    .box-action [style*="0.81em"] { font-size: 0.87em !important; }

    /* Category pills — compact */
    .pill     { font-size: 0.63em !important; padding: 2px 7px !important; }
    .pill-cat { font-size: 0.68em !important; padding: 2px 6px !important; }

    /* Category signal panels */
    .cat-panel  { padding: 10px 11px !important; margin-bottom: 10px !important; }
    .cat-header { margin-bottom: 8px !important; padding-bottom: 6px !important; }

    /* Env banner — slightly tighter */
    .env-banner { padding: 14px 15px !important; margin-bottom: 14px !important; }

    /* Footer — hide on mobile (not useful) */
    .rs-footer { display: none !important; }
}
</style>
""", unsafe_allow_html=True)


# ── Password gate ─────────────────────────────────────────────────────────────
def _check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True

    col_l, col_m, col_r = st.columns([1, 1.4, 1])
    with col_m:
        st.markdown(
            "<div style='text-align:center;padding:40px 0 24px'>"
            "<div style='font-size:2em;font-weight:800;color:#e2e8f0;letter-spacing:-0.5px'>📈 Rally Scout</div>"
            "<div style='color:#7092b0;font-size:0.88em;margin-top:6px'>Sign in to continue</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        pwd = st.text_input("Password", type="password", placeholder="Enter password",
                            label_visibility="collapsed")
        if st.button("Sign in", type="primary", use_container_width=True):
            expected = st.secrets.get("password", "").strip()
            if not expected:
                st.error("App secret not configured. Add `password = \"yourpassword\"` in Streamlit Cloud → App Settings → Secrets.")
            elif hmac.compare_digest(pwd.strip().encode(), expected.encode()):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False

if not _check_password():
    st.stop()


# ── Translation tables ────────────────────────────────────────────────────────
SIGNAL_PLAIN = {
    "Golden Cross":           "50-day avg crossed ABOVE 200-day — textbook bull trend signal",
    "Price vs 200 SMA":       "Price above the long-term 200-day average",
    "Bounce off 50 SMA":      "50-day average acting as a confirmed support floor",
    "MA Stacking":            "All moving averages aligned bullishly (9 > 21 > 50 > 200)",
    "MACD Crossover":         "Short-term momentum flipping positive",
    "RSI Recovery":           "Momentum bounced from oversold territory back above 50",
    "Stochastic":             "Oscillator crossed up from an oversold low",
    "Rate of Change":         "Price rate-of-change turned positive after a negative reading",
    "Volume Accumulation":    "More buying volume than selling across last 20 sessions",
    "Volume Breakout":        "Broke out on unusually high volume — institutional conviction",
    "OBV Accumulation":       "On-Balance Volume rising while price consolidates (smart money buying)",
    "Higher Highs/Lows":      "Making higher highs and higher lows — uptrend structure forming",
    "Resistance Breakout":    "Broke above a key resistance level tested multiple times",
    "Price Coiling":          "Tight price range — potential energy building for a breakout",
    "Fibonacci Support":      "Pulled back to a key Fibonacci level and held",
    "EPS & Revenue Beat":     "Beat Wall Street on BOTH earnings AND revenue last quarter",
    "Revenue Guidance":       "Forward revenue growth trajectory is accelerating",
    "Gross Margin Quality":   "Profit margins are healthy and expanding",
    "Revenue Acceleration":   "Revenue growth is speeding up quarter over quarter",
    "Earnings Surprise":      "Consistently beats estimates by wide margins (sandbagging pattern)",
    "Cash Runway":            "Strong cash position — no dilutive fundraising risk",
    "Debt/Cash Flow":         "Debt is manageable relative to cash generation",
    "Insider Buying":         "Executives buying shares with their own money",
    "Institutional Ownership":"Large funds hold a significant stake",
    "Analyst Upgrades":       "Recent analyst upgrades with raised price targets",
    "Short Squeeze Setup":    "High short interest = forced buyers on any positive catalyst",
    "Price Target Upside":    "Analyst price targets imply meaningful upside from here",
    "Index Inclusion Proxy":  "Candidate for index addition (triggers passive fund buying)",
    "Sector Rotation":        "Money actively flowing INTO this sector right now",
    "S&P 500 Trend":          "Broad market is in an uptrend — tailwind for all longs",
    "VIX Environment":        "Fear index declining — risk-on environment",
    "Dollar Trend":           "Dollar trend is favorable for this company's earnings",
    "Market Breadth":         "Rally is broad-based, not concentrated in one sector",
    "Put/Call Ratio":         "Options traders extremely bearish — contrarian buy setup",
    "Short Interest Fuel":    "High short interest amplifies upside on any catalyst",
    "Contrarian Consensus":   "Analysts mostly bearish — next upgrade triggers an outsized move",
    "Media Neglect Factor":   "Stock is under the radar — nobody left to sell",
    "Earnings Catalyst":      "Earnings report approaching — a hard catalyst to force repricing",
    "Seasonality":            "Historically strong time of year for this sector",
    "Post-Earnings Dip":      "Dropped on strong earnings — a gift entry point",
    "Consolidation Setup":    "Long consolidation after a big run — coiled for the next leg",
}

REASON_MAP = {
    "Golden Cross":        "The 50-day just crossed above the 200-day — a textbook bull signal that major fund algorithms are programmed to follow.",
    "Volume Breakout":     "Broke above resistance on the highest volume in months. When institutions move, volume is the fingerprint.",
    "OBV Accumulation":    "On-Balance Volume hitting new highs while price consolidates — smart money is building a position before the move is visible.",
    "RSI Recovery":        "RSI bounced from oversold back above 50. Momentum has definitively shifted from sellers to buyers.",
    "Post-Earnings Dip":   "Sold off on a strong earnings beat — profit-taking on good news is one of the best risk/reward entries in the market.",
    "Short Squeeze Setup": "Extreme short interest is a loaded spring. One positive catalyst forces short sellers to buy, amplifying every point of upside.",
    "Insider Buying":      "Executives bought shares with their own money. The people closest to the business are voting with capital.",
    "EPS & Revenue Beat":  "Beat on BOTH earnings AND revenue — not just one line. This is the gold standard of earnings quality.",
    "Analyst Upgrades":    "Recent analyst upgrades with raised targets. New institutional money flows in with a 2-4 week lag after upgrades.",
    "Price Coiling":       "Price compressing into a tight range — low volatility before high volatility. The breakout is being built.",
    "MA Stacking":         "All moving averages in perfect bullish order. The trend is clean and confirmed across every timeframe.",
    "Earnings Catalyst":   "A hard earnings catalyst is approaching. The market must reprice this stock imminently.",
    "Sector Rotation":     "Money is rotating INTO this sector. You have a macro tailwind, not a headwind.",
    "VIX Environment":     "VIX dropping from a spike — fear fading fast, risk-on capital flooding back into the market.",
    "Resistance Breakout": "Broke above a level that held multiple times. Every seller at this level is now underwater and out of the way.",
    "Higher Highs/Lows":   "Higher highs and higher lows after a downtrend — the trend structure has definitively reversed.",
    "Fibonacci Support":   "Held the 61.8% Fibonacci retracement — where institutional buyers defend hardest.",
    "Contrarian Consensus":"Most analysts are bearish while data is improving. When consensus flips, the repricing is violent.",
    "Revenue Acceleration":"Revenue growth accelerating quarter over quarter. The business is gaining momentum.",
    "Consolidation Setup": "Long consolidation after a big move resets sentiment. Stock coiled for the next leg.",
    "Volume Accumulation": "Systematic buying volume over selling across 20 sessions. Institutions accumulating quietly.",
    "Price vs 200 SMA":    "Just reclaimed the 200-day average — institutional algorithms switch from sell to buy mode.",
    "Bounce off 50 SMA":   "50-day acting as a reliable support floor, confirmed by multiple touches.",
    "MACD Crossover":      "MACD crossed above signal with histogram turning positive — momentum is shifting.",
}

CAVEAT_MAP = {
    "Technical":     "Chart structure not yet ideal — consider scaling in rather than a full position.",
    "Fundamental":   "Fundamentals softer than ideal — this leans on technical/sentiment more than business quality.",
    "Institutional": "No major institutional catalyst yet — relies on organic price action without a big fund trigger.",
    "Macro":         "Macro environment not ideal — this is a stock-specific trade; size accordingly.",
    "Sentiment":     "Sentiment not contrarian enough for a squeeze — upside may be more measured.",
    "Timing":        "No near-term catalyst visible — this may take time to play out; be patient.",
}

CAT_ICONS = {
    "Technical": "📊", "Fundamental": "📋", "Institutional": "🏛️",
    "Macro": "🌊", "Sentiment": "🧠", "Timing": "⏰",
}

_SOURCE_ICONS = {
    "AI/Tech Core":        "🤖",
    "High Short Interest": "🔥",
    "Momentum Leader":     "📈",
    "S&P Top Performer":   "🏆",
    "Your Watchlist":      "⭐",
    "Your Portfolio":      "💼",
}


# ── Core helpers ──────────────────────────────────────────────────────────────
def score_color(s: float) -> str:
    if s >= 75: return "#00d084"
    if s >= 60: return "#00b070"
    if s >= 45: return "#f59e0b"
    if s >= 30: return "#64748b"
    return "#f43f5e"

def sig_emoji(s: float) -> str:
    if s >= 0.75: return "✅"
    if s >= 0.50: return "🟡"
    if s >= 0.30: return "⚠️"
    return "🔴"

def fmt_cap(mc) -> str:
    if not mc: return "N/A"
    if mc >= 1e12: return f"${mc/1e12:.2f}T"
    if mc >= 1e9:  return f"${mc/1e9:.1f}B"
    return f"${mc/1e6:.0f}M"

def action_label(score: float) -> tuple[str, str]:
    if score >= 80: return "BUY NOW",  "badge-now"
    if score >= 72: return "BUY",      "badge-buy"
    if score >= 65: return "SCALE IN", "badge-scale"
    if score >= 55: return "WATCH",    "badge-watch"
    return "MONITOR", "badge-hold"

def action_badge(score: float) -> str:
    lbl, css = action_label(score)
    return f"<span class='badge {css}'>{lbl}</span>"

def position_size_label(score: float) -> str:
    if score >= 80: return "Full position"
    if score >= 72: return "⅔ position"
    if score >= 65: return "½ starter"
    return "Watch only"

def trade_plan(result: dict) -> dict:
    score  = result["composite"]
    price  = result.get("price") or 0
    meta   = result.get("meta", {})
    target = meta.get("analyst_target")
    low52  = meta.get("52w_low")
    high52 = meta.get("52w_high")

    if score >= 80:   entry_note = "Enter at current price or on any 3-5% pullback."
    elif score >= 72: entry_note = "Enter at current price or scale in over 2 tranches on dips."
    elif score >= 65: entry_note = "Scale in over 3 entries; start with half, add on each confirmation."
    else:             entry_note = "Do not commit capital yet. Watch for score to cross 65."

    stop = None
    stop_pct = None
    if price > 0:
        stop_10 = price * 0.90
        stop = max(low52 or 0, stop_10)
        stop_pct = (price - stop) / price * 100

    target_note = ""
    if not target and price:
        target = price * 1.20
        target_note = "est."
    else:
        target_note = "analyst"

    upside_pct = ((target / price) - 1) * 100 if price and target else None

    rr = None
    if price and stop and target:
        risk   = price - stop
        reward = target - price
        if risk > 0:
            rr = reward / risk

    return {
        "entry_note":  entry_note,
        "pos_size":    position_size_label(score),
        "stop":        stop,
        "stop_pct":    stop_pct,
        "target":      target,
        "target_note": target_note,
        "upside_pct":  upside_pct,
        "rr":          rr,
    }

_TICKER_VALID = __import__("re").compile(r"^[A-Z]{1,5}(-[A-Z]{1,2})?$")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_prices(tickers: tuple) -> dict[str, float]:
    """Batch-download current closing prices for a tuple of tickers. Cached 5 min."""
    if not tickers:
        return {}
    try:
        raw = yf.download(list(tickers), period="2d", auto_adjust=True,
                          progress=False, threads=True)
        if raw.empty:
            return {}
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        prices: dict[str, float] = {}
        for t in tickers:
            col = close[t].dropna() if t in close.columns else pd.Series(dtype=float)
            if col.empty and len(tickers) == 1:
                col = close.squeeze().dropna()
            if not col.empty:
                prices[t] = round(float(col.iloc[-1]), 4)
        return prices
    except Exception:
        return {}

def parse_yahoo_portfolio(df: pd.DataFrame) -> list[dict] | None:
    """
    Parse a Yahoo Finance portfolio export CSV into a list of holding dicts.
    Returns None if the CSV doesn't contain holdings data (e.g. a watchlist export).
    Multiple lots of the same ticker are aggregated with weighted-average cost basis.
    """
    df.columns = [str(c).strip() for c in df.columns]
    if "Symbol" not in df.columns or "Quantity" not in df.columns:
        return None

    def _f(val) -> float:
        """Parse a numeric cell; returns 0.0 for blanks, non-numeric, or NaN."""
        try:
            f = float(str(val).replace(",", "").replace("$", "").strip())
            return 0.0 if f != f else f  # f != f is True only for NaN
        except (ValueError, TypeError):
            return 0.0

    lots: list[dict] = []
    for _, row in df.iterrows():
        ticker = str(row.get("Symbol", "")).strip().upper()
        # Skip cash rows ($$CASH_TX etc.), header echoes, and non-ticker symbols
        if not ticker or not _TICKER_VALID.match(ticker):
            continue
        qty = _f(row.get("Quantity", 0))
        if qty <= 0:
            continue
        lots.append({
            "ticker":            ticker,
            "shares":            qty,
            "cost_basis":        _f(row.get("Purchase Price", 0)),
            "current_price_csv": _f(row.get("Current Price", row.get("Last Price", 0))),
        })

    if not lots:
        return None

    # Aggregate multiple lots — weighted-average cost basis
    agg: dict = {}
    for h in lots:
        t = h["ticker"]
        if t not in agg:
            agg[t] = {"shares": 0.0, "cost_total": 0.0, "current_price_csv": 0.0}
        agg[t]["shares"]        += h["shares"]
        agg[t]["cost_total"]    += h["shares"] * h["cost_basis"]
        agg[t]["current_price_csv"] = h["current_price_csv"]

    return sorted(
        [
            {
                "ticker":            t,
                "shares":            d["shares"],
                "cost_basis":        d["cost_total"] / d["shares"] if d["shares"] else 0.0,
                "current_price_csv": d["current_price_csv"],
            }
            for t, d in agg.items()
        ],
        key=lambda x: x["ticker"],
    )

def portfolio_action(score: float | None, pnl_pct: float) -> tuple[str, str]:
    """Return (label, badge_css) recommendation for a held position."""
    if score is None:
        return "UNSCORED", "badge-hold"
    if score >= 80 and pnl_pct < 15:
        return "ADD", "badge-now"
    if score >= 72 and pnl_pct < 35:
        return "BUY MORE", "badge-buy"
    if score >= 65:
        return "HOLD", "badge-scale"
    if score < 55 and pnl_pct > 20:
        return "TRIM", "badge-watch"
    if score < 50 and pnl_pct < -10:
        return "CUT", "badge-hold"
    if score < 40:
        return "EXIT", "badge-hold"
    return "HOLD", "badge-scale"

def make_action_statement(result: dict) -> tuple[str, str, str]:
    score = result["composite"]
    sigs  = result["signals"]
    cats  = result["categories"]

    all_sigs: dict = {}
    for k in sigs:
        all_sigs.update(sigs[k])

    top_sig  = max(all_sigs, key=all_sigs.get) if all_sigs else None
    weak_cat = min(cats.items(), key=lambda x: x[1])

    primary = REASON_MAP.get(top_sig, SIGNAL_PLAIN.get(top_sig, "Multiple signals aligned across categories."))
    caveat  = CAVEAT_MAP.get(weak_cat[0], "") if weak_cat[1] < 48 else ""
    return action_badge(score), primary, caveat

def render_signal_row(name: str, score: float):
    plain = SIGNAL_PLAIN.get(name, name)
    icon  = sig_emoji(score)
    pct   = int(score * 100)
    color = score_color(pct)
    st.markdown(
        f"<div class='sig-row'>"
        f"<span class='sig-icon'>{icon}</span>"
        f"<span class='sig-text'>{plain}</span>"
        f"<div class='sig-track'><div class='sig-fill' style='width:{pct}%;background:{color}'></div></div>"
        f"<span class='sig-pct' style='color:{color}'>{pct}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

def generate_thesis(result: dict) -> tuple[list[str], list[str]]:
    reasons, risks = [], []
    sigs = result["signals"]
    cats = result["categories"]
    meta = result["meta"]
    t = sigs["technical"]; f = sigs["fundamental"]
    i = sigs["institutional"]; m = sigs["macro"]
    s = sigs["sentiment"]; tm = sigs["timing"]

    if t.get("Golden Cross", 0) > 0.8:
        reasons.append("Golden Cross active — 50-day crossed above 200-day. Major fund algorithms are programmed to buy this signal.")
    elif t.get("MA Stacking", 0) > 0.75:
        reasons.append("All moving averages stacked in perfect bullish order. The trend is confirmed on every timeframe simultaneously.")
    if t.get("Volume Breakout", 0) > 0.7:
        reasons.append("Broke above resistance on high volume — institutions are behind this move, not retail speculation.")
    if t.get("OBV Accumulation", 0) > 0.75:
        reasons.append("On-Balance Volume making new highs while price consolidates — smart money is quietly building before the move is visible.")
    if t.get("RSI Recovery", 0) > 0.85:
        reasons.append("RSI bounced from oversold back above 50 — momentum has definitively flipped from sellers to buyers.")
    if t.get("Price Coiling", 0) > 0.75:
        reasons.append("Price compressing in a tight range — low volatility precedes high volatility. The breakout is being built.")
    if t.get("Fibonacci Support", 0) > 0.9:
        reasons.append("Held the 61.8% Fibonacci retracement. Institutional buyers defend this level hard — they stepped in here.")
    if f.get("EPS & Revenue Beat", 0) > 0.8:
        reasons.append("Beat on BOTH earnings AND revenue last quarter. One-sided beats are table stakes; this is the gold standard.")
    if f.get("Revenue Acceleration", 0) > 0.75:
        reasons.append("Revenue growth accelerating quarter over quarter — the business is gaining momentum, not decelerating.")
    if f.get("Insider Buying", 0) > 0.75:
        reasons.append("Executives buying shares with their own money — the strongest internal vote of confidence a company can send.")
    if f.get("Earnings Surprise", 0) > 0.85:
        reasons.append("Consistently beating estimates by wide margins — management is sandbagging guidance. This pattern repeats reliably.")
    if i.get("Short Squeeze Setup", 0) > 0.8:
        reasons.append("High short interest is a loaded spring. Any positive catalyst forces short sellers to buy, amplifying every point of upside.")
    if i.get("Analyst Upgrades", 0) > 0.75:
        reasons.append("Recent analyst upgrades with raised price targets. New institutional capital follows upgrades with a 2-4 week lag.")
    if i.get("Price Target Upside", 0) > 0.8:
        reasons.append("Analyst consensus implies significant upside — Wall Street sees more room to run from here.")
    if m.get("Sector Rotation", 0) > 0.75:
        reasons.append("Money actively rotating INTO this sector. The macro tailwind is real and measurable in sector ETF flows.")
    if m.get("VIX Environment", 0) > 0.85:
        reasons.append("VIX dropping from a spike — fear fading fast, risk-on capital flooding back into equities.")
    if s.get("Contrarian Consensus", 0) > 0.8:
        reasons.append("Most analysts still bearish. Lopsided bearish consensus means any upgrade triggers an outsized repricing.")
    if s.get("Put/Call Ratio", 0) > 0.75:
        reasons.append("Options market showing extreme pessimism — historically this level of fear precedes strong reversals.")
    if tm.get("Post-Earnings Dip", 0) > 0.85:
        reasons.append("Sold off on strong earnings — profit-taking on good news creates one of the best risk/reward entries available.")
    if tm.get("Earnings Catalyst", 0) > 0.85:
        reasons.append("Earnings approaching — a hard catalyst forces the market to reprice this stock in the near term.")

    if cats["Technical"] < 40:
        risks.append("Technical structure is still broken — price and momentum are not yet supporting a buy entry.")
    if t.get("Price vs 200 SMA", 0) < 0.3:
        risks.append("Trading BELOW the 200-day average — historically the highest-risk zone. The trend is still structurally down.")
    if cats["Fundamental"] < 40:
        risks.append("Fundamentals are underwhelming — earnings quality, margins, or growth aren't confirming the bull case.")
    if f.get("Insider Buying", 0) < 0.2:
        risks.append("Insiders are selling, not buying — the people who know the business best are reducing exposure.")
    if cats["Macro"] < 40:
        risks.append("Macro environment is unfavorable — buying into a declining market or out-of-favor sector fights the tape.")
    if meta.get("beta") and meta["beta"] > 2.0:
        risks.append(f"High beta ({meta['beta']:.1f}x) — this stock moves 2x the market in both directions. Size the position accordingly.")
    if f.get("Cash Runway", 0) < 0.3:
        risks.append("Weak cash position — elevated risk of a dilutive capital raise that would pressure share price.")
    if t.get("Volume Breakout", 0) < 0.2 and t.get("Volume Accumulation", 0) < 0.35:
        risks.append("Volume is not confirming the price action — moves on thin volume frequently reverse without warning.")
    if not result["is_buy"]:
        risks.append(f"Overall Rally Score of {result['composite']:.0f} is below the {BUY_SIGNAL_THRESHOLD} buy threshold — more signal confirmation needed.")

    if not reasons:
        reasons.append("No high-conviction bullish signals detected. Monitor for improving conditions.")
    if not risks:
        risks.append("No major red flags identified. Standard position sizing applies.")

    return reasons[:6], risks[:4]


# ── Chart builders ────────────────────────────────────────────────────────────
def build_price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(),   close=close, name=ticker,
        increasing_line_color="#00d084", decreasing_line_color="#f43f5e",
        increasing_fillcolor="#00d084",  decreasing_fillcolor="#f43f5e",
    ), row=1, col=1)
    for window, color, name in [(20,"#a78bfa","20d"), (50,"#38bdf8","50d"), (200,"#fb923c","200d")]:
        if len(close) >= window:
            fig.add_trace(go.Scatter(
                x=df.index, y=close.rolling(window).mean(),
                name=name, line=dict(color=color, width=1.2), opacity=0.8,
            ), row=1, col=1)
    bar_colors = ["#00d084" if c >= o else "#f43f5e"
                  for c, o in zip(df["Close"].squeeze(), df["Open"].squeeze())]
    fig.add_trace(go.Bar(x=df.index, y=volume, name="Volume",
                          marker_color=bar_colors, opacity=0.45), row=2, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1520", plot_bgcolor="#0e1520",
        height=440, showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig.update_yaxes(gridcolor="#1a2535", zeroline=False)
    fig.update_xaxes(gridcolor="#1a2535")
    return fig

def build_radar(categories: dict) -> go.Figure:
    labels = list(categories.keys())
    values = list(categories.values()) + [list(categories.values())[0]]
    labels = labels + [labels[0]]
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=labels, fill="toself",
        fillcolor="rgba(0,208,132,0.10)",
        line=dict(color="#00d084", width=2),
        hovertemplate="%{theta}: %{r:.0f}/100<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0e1520",
            radialaxis=dict(visible=True, range=[0,100], gridcolor="#1a2840",
                            tickfont=dict(size=8), tickcolor="#607d96"),
            angularaxis=dict(gridcolor="#1a2840"),
        ),
        template="plotly_dark", paper_bgcolor="#0b1119",
        height=270, margin=dict(l=10,r=10,t=10,b=10), showlegend=False,
    )
    return fig

def build_sector_chart(sector_perf: dict) -> go.Figure:
    if not sector_perf:
        return go.Figure()
    df = pd.DataFrame([{"Sector": k, "1M": v["return_1m"]} for k, v in sector_perf.items()])
    df = df.sort_values("1M", ascending=True)
    colors = [score_color(50 + v * 2) for v in df["1M"]]
    fig = go.Figure(go.Bar(
        y=df["Sector"], x=df["1M"], orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in df["1M"]], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1520", plot_bgcolor="#0e1520",
        height=340, margin=dict(l=0,r=55,t=5,b=0), showlegend=False,
        xaxis=dict(title="1-Month Return (%)", gridcolor="#1a2535"),
        yaxis=dict(gridcolor="#1a2535"),
    )
    return fig


# ── Conviction card ───────────────────────────────────────────────────────────
def render_conviction_card(r: dict, tier: int):
    score  = r["composite"]
    color  = score_color(score)
    badge_html, primary, caveat = make_action_statement(r)
    price  = r.get("price")
    price_str = f"${price:,.2f}" if price else "N/A"
    chg1d  = r.get("change_1d")
    chg1m  = r.get("change_1m")
    chg_color = "#00d084" if (chg1d or 0) >= 0 else "#f43f5e"
    chg_arrow = "▲" if (chg1d or 0) >= 0 else "▼"
    chg1d_str = f"{chg_arrow} {abs(chg1d):.1f}% today" if chg1d is not None else ""
    chg1m_str = f"{chg1m:+.1f}% (1M)" if chg1m is not None else ""
    cap_str   = fmt_cap(r.get("market_cap"))
    css = {1: "rs-card-1", 2: "rs-card-2", 3: "rs-card-3"}[tier]

    source   = r.get("source", "")
    src_icon = _SOURCE_ICONS.get(source, "•")
    src_pill = (f"<span class='pill pill-source'>{src_icon} {source}</span>") if source else ""

    cat_pills = "".join(
        f"<span class='pill-cat' style='color:{score_color(v)}'>"
        f"{CAT_ICONS.get(k,'')} {k[:4]} {v:.0f}</span>"
        for k, v in r["categories"].items()
    )

    pos = position_size_label(score)

    st.markdown(f"""
<div class='{css}'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:12px'>
    <div style='flex:1;min-width:0'>
      <div style='display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin-bottom:3px'>
        <span style='font-size:1.2em;font-weight:800;color:#f1f5f9;letter-spacing:-0.3px'>{r["ticker"]}</span>
        {badge_html}
        {src_pill}
      </div>
      <div style='color:#7092b0;font-size:0.75em;margin-bottom:7px'>{r.get("name","")[:32]}  ·  {r.get("sector","")}</div>
      <div style='margin-bottom:8px;line-height:1.9'>{cat_pills}</div>
    </div>
    <div style='text-align:right;flex-shrink:0'>
      <div style='font-size:2em;font-weight:800;color:{color};line-height:1'>{score:.0f}</div>
      <div style='font-size:0.62em;color:#607d96;letter-spacing:0.4px'>/100</div>
      <div style='font-size:0.79em;margin-top:6px;white-space:nowrap'>
        <span style='color:#e2e8f0'>{price_str}</span>
        <span style='color:{chg_color};margin-left:5px'>{chg1d_str}</span>
      </div>
      <div style='font-size:0.69em;color:#7092b0;margin-top:1px'>{chg1m_str}</div>
      <div style='font-size:0.67em;color:#607d96;margin-top:1px'>{cap_str}</div>
    </div>
  </div>

  <div class='box-action'>
    <div style='font-size:0.67em;color:#4ade80;font-weight:700;letter-spacing:0.8px;margin-bottom:4px'>WHY THIS RALLIES</div>
    <div style='color:#d1fae5;font-size:0.81em;line-height:1.5'>{primary}</div>
  </div>

  {"<div class='box-risk'><span style='color:#fbbf24;font-size:0.68em;font-weight:700;letter-spacing:0.5px'>RISK  ·  </span><span style='color:#fde68a;font-size:0.75em'>" + caveat + "</span></div>" if caveat else ""}

  <div style='font-size:0.69em;color:#607d96;margin-top:8px'>
    Size: <span style='color:#7092b0;font-weight:600'>{pos}</span>
  </div>
</div>""", unsafe_allow_html=True)

    if st.button("Full Analysis →", key=f"dd_{r['ticker']}_{tier}", use_container_width=True):
        st.session_state.selected_result      = r
        st.session_state.navigate_to_analysis = True


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class='sb-brand'>
        <div class='sb-brand-title'>Rally Scout</div>
        <div class='sb-brand-sub'>AI-powered signal engine</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Scan status
    if st.session_state.get("auto_scan_complete"):
        ts      = st.session_state.get("auto_scan_time")
        age     = int((datetime.now() - ts).total_seconds() / 60) if ts else 0
        results = st.session_state.get("auto_results", [])
        buys    = [r for r in results if r["is_buy"]]
        st.markdown(
            f"<div class='scan-pill'>"
            f"<div class='scan-pill-label'>✓ SCAN COMPLETE</div>"
            f"<div class='scan-pill-sub'>{len(results)} analyzed · {len(buys)} buy signals · {age}m ago</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        univ = st.session_state.get("last_universe", {})
        if univ:
            for src, cnt in universe_summary(univ).items():
                icon = _SOURCE_ICONS.get(src, "•")
                st.caption(f"{icon} **{src}**: {cnt}")
        if st.button("🔄 Re-Scan Now", use_container_width=True):
            st.session_state.auto_scan_complete = False
            st.session_state.auto_results = []
            st.rerun()
    else:
        st.info("Scanning on launch…")

    st.divider()

    # Portfolio import — Yahoo Finance CSV
    st.markdown("**💼 My Portfolio**")
    st.caption("Yahoo Finance → Portfolio → top-right Export button → upload here.")
    uploaded_pf = st.file_uploader(
        "Upload Yahoo Finance export (.csv)",
        type="csv",
        label_visibility="collapsed",
    )
    if uploaded_pf is not None:
        pf_file_id = f"{uploaded_pf.name}:{uploaded_pf.size}"
        if st.session_state.get("_portfolio_file_id") != pf_file_id:
            try:
                df_pf = pd.read_csv(uploaded_pf)
                holdings = parse_yahoo_portfolio(df_pf)
                if holdings is None:
                    st.error(
                        "This looks like a Watchlist export (no quantity/price data). "
                        "Export from your **Portfolio** page instead."
                    )
                else:
                    st.session_state.portfolio         = holdings
                    st.session_state._portfolio_file_id = pf_file_id
                    st.session_state.auto_scan_complete = False
                    st.session_state.auto_results       = []
                    st.success(f"Loaded {len(holdings)} positions from {uploaded_pf.name}")
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")

    portfolio = st.session_state.get("portfolio", [])
    if portfolio:
        tickers_str = ", ".join(h["ticker"] for h in portfolio)
        st.caption(f"{len(portfolio)} positions: {tickers_str[:80]}{'…' if len(tickers_str) > 80 else ''}")
        if st.button("🗑 Clear Portfolio", use_container_width=True):
            st.session_state.portfolio          = []
            st.session_state._portfolio_file_id = None
            st.session_state.auto_scan_complete = False
            st.session_state.auto_results       = []
            st.rerun()

    st.divider()

    # Watchlist
    st.markdown("**⭐ Watchlist**")
    st.caption("Extra tickers to always include in every scan.")
    watchlist_raw = st.text_area(
        "Tickers",
        value=st.session_state.get("user_watchlist_raw", ""),
        height=64,
        placeholder="CRWV, IREN, APLD",
        label_visibility="collapsed",
    )
    if watchlist_raw != st.session_state.get("user_watchlist_raw", ""):
        st.session_state.user_watchlist_raw = watchlist_raw
        st.session_state.auto_scan_complete = False
        st.session_state.auto_results       = []

    st.divider()

    # Custom scan settings
    st.markdown("**Custom Scan Settings**")
    universe_choice = st.selectbox(
        "Universe",
        ["Dynamic (auto-scout)", "S&P 500", "Custom Tickers"],
        label_visibility="collapsed",
    )
    if universe_choice == "Custom Tickers":
        custom_raw     = st.text_area("Tickers", value="AAPL, NVDA, MSFT, TSLA, AMD",
                                       height=64, label_visibility="collapsed")
        custom_tickers = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]
    else:
        custom_tickers = []

    sector_filter = st.selectbox("Sector Filter", [
        "All","Technology","Healthcare","Financials","Consumer Disc.",
        "Industrials","Energy","Materials","Utilities","Real Estate",
        "Comm. Services","Consumer Staples",
    ])
    cap_opts = {"Any":0,"Mega (>$200B)":200e9,"Large (>$10B)":10e9,
                "Mid (>$2B)":2e9,"Small (>$300M)":300e6}
    min_cap  = cap_opts[st.selectbox("Min Market Cap", list(cap_opts.keys()))]
    max_scan = st.number_input("Max Stocks", 5, 500, 75, 5)
    run_custom = st.button("Run Custom Scan →", type="primary", use_container_width=True)

    st.divider()

    # Score legend
    st.markdown("**Score Guide**")
    for score_min, label, css, desc in [
        (80, "80+  BUY NOW",   "#00d084", "All signals aligned"),
        (72, "72+  BUY",       "#059669", "High conviction"),
        (65, "65+  SCALE IN",  "#0284c7", "Build in tranches"),
        (55, "55+  WATCH",     "#d97706", "Improving, not confirmed"),
        (0,  "<55  MONITOR",   "#607d96", "Too early"),
    ]:
        lbl_parts = label.split(None, 1)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
            f"<div style='width:8px;height:8px;border-radius:50%;background:{css};flex-shrink:0'></div>"
            f"<span style='font-size:0.74em;color:#7092b0'>"
            f"<b style='color:{css}'>{lbl_parts[0]}</b>&nbsp;{lbl_parts[1]}&nbsp;—&nbsp;{desc}"
            f"</span></div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# Mobile detection — no extra packages; User-Agent from request headers
try:
    _ua = st.context.headers.get("User-Agent", "")
except Exception:
    _ua = ""
is_mobile = any(k in _ua for k in ("iPhone", "Android", "Mobile", "iPad"))

if is_mobile:
    st.markdown(
        "<h1 style='font-size:1.35em;font-weight:800;letter-spacing:-0.3px;margin-bottom:1px'>📈 Rally Scout</h1>"
        "<p style='color:#7092b0;font-size:0.80em;margin-bottom:0'>41 signals · Know exactly what to buy.</p>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<h1 style='font-size:1.8em;font-weight:800;letter-spacing:-0.5px;margin-bottom:2px'>📈 Rally Scout</h1>"
        "<p style='color:#7092b0;font-size:0.88em;margin-bottom:0'>Dynamic universe · 41 signals · Tells you exactly what to buy and why.</p>",
        unsafe_allow_html=True,
    )

_tab_labels = (
    ["🏆 Picks", "💼 Portfolio", "📊 Analysis", "🌊 Market", "🔍 Scan"]
    if is_mobile else
    ["🏆 Top Picks", "💼 Portfolio", "📊 Deep Dive", "🌊 Market", "🔍 Custom Scan"]
)
tab_scout, tab_portfolio, tab_analysis, tab_market, tab_custom = st.tabs(_tab_labels)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TOP PICKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_scout:

    if not st.session_state.get("auto_scan_complete", False):
        st.markdown("### Scanning the market for highest-conviction setups…")
        st.caption("Pass 1 — Technical screen across all stocks. Pass 2 — Full 41-signal analysis on top candidates.")
        phase_label = st.empty()
        prog        = st.progress(0)
        ticker_disp = st.empty()

        def auto_progress(i, total, ticker, phase=1):
            pct = int(i / max(total, 1) * 100)
            if phase == 1:
                phase_label.markdown("**Phase 1 of 2 — Technical Screen:** Filtering for the best chart setups…")
                prog.progress(max(2, pct // 2))
            else:
                phase_label.markdown("**Phase 2 of 2 — Deep Analysis:** Running all 41 signals on top candidates…")
                prog.progress(50 + max(1, pct // 2))
            ticker_disp.caption(f"Analyzing **{ticker}**")

        user_tickers = [t.strip().upper() for t in
                        st.session_state.get("user_watchlist_raw", "").split(",") if t.strip()]
        pf_tickers = [h["ticker"] for h in st.session_state.get("portfolio", [])]
        universe = build_dynamic_universe(user_tickers=user_tickers)
        # Layer 6 — portfolio tickers (applied here to avoid signature dependency)
        for _t in pf_tickers:
            if _t:
                universe[_t] = "Your Portfolio"
        st.session_state.last_universe = universe

        results = two_pass_scan(universe, progress_callback=auto_progress, deep_n=30,
                                force_tickers=pf_tickers)
        prog.progress(100)
        ticker_disp.empty()
        phase_label.empty()

        st.session_state.auto_results       = results
        st.session_state.auto_scan_complete = True
        st.session_state.auto_scan_time     = datetime.now()
        st.rerun()

    results = st.session_state.get("auto_results", [])

    if not results:
        st.warning("No results yet. Click **Re-Scan Now** in the sidebar.")
    else:
        buys  = [r for r in results if r["is_buy"]]
        tier1 = [r for r in results if r["composite"] >= 80]
        tier2 = [r for r in results if 65 <= r["composite"] < 80]
        tier3 = [r for r in results if 50 <= r["composite"] < 65]

        # Summary metrics
        sector_counts: dict = {}
        for r in buys:
            sector_counts[r["sector"]] = sector_counts.get(r["sector"], 0) + 1

        if is_mobile:
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)
            c1.metric("Analyzed", len(results))
            c2.metric("Buy Signals", len(buys))
            c3.metric("Top Score",
                      f"{results[0]['composite']:.0f}" if results else "—",
                      results[0]["ticker"] if results else "")
            if buys:
                c4.metric("Avg Score", f"{np.mean([r['composite'] for r in buys]):.1f}")
            if sector_counts:
                hot = max(sector_counts, key=sector_counts.get)
                st.markdown(
                    f"<div style='font-size:0.78em;color:#7092b0;margin-top:2px;padding:6px 2px'>"
                    f"🔥 Hot sector: <b style='color:#e2e8f0'>{hot}</b>"
                    f" <span style='color:#607d96'>· {sector_counts[hot]} signals</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Stocks Analyzed", len(results))
            c2.metric("Buy Signals", len(buys))
            c3.metric("Top Score",
                      f"{results[0]['composite']:.0f}" if results else "—",
                      results[0]["ticker"] if results else "")
            if buys:
                c4.metric("Avg Buy Score", f"{np.mean([r['composite'] for r in buys]):.1f}")
            if sector_counts:
                hot = max(sector_counts, key=sector_counts.get)
                c5.metric("Hot Sector", hot[:16], f"{sector_counts[hot]} signals")

        st.divider()

        # Tier 1
        if tier1:
            st.markdown("## 🏆 Highest Conviction — Score 80+")
            st.caption("All major signal categories firing simultaneously. The strongest setups currently in the market.")
            if is_mobile:
                for r in tier1:
                    render_conviction_card(r, tier=1)
            else:
                cols1 = st.columns(2)
                for i, r in enumerate(tier1):
                    with cols1[i % 2]:
                        render_conviction_card(r, tier=1)
        else:
            st.markdown("## 🏆 Highest Conviction — Score 80+")
            st.info("No stocks currently score 80+. The setups below are the strongest available today.")

        # Tier 2
        if tier2:
            st.markdown("## ⚡ Strong Buy — Score 65–79")
            st.caption("Clear buy signal with most categories confirming. Scale into these over 2-3 entries.")
            if is_mobile:
                for r in tier2:
                    render_conviction_card(r, tier=2)
            else:
                cols2 = st.columns(2)
                for i, r in enumerate(tier2):
                    with cols2[i % 2]:
                        render_conviction_card(r, tier=2)

        # Tier 3
        if tier3:
            with st.expander(f"👀 Building Momentum — Score 50–64  ({len(tier3)} stocks)"):
                st.caption("Not confirmed buys yet, but signals are improving. Watch for score to cross 65.")
                for r in tier3:
                    score     = r["composite"]
                    color     = score_color(score)
                    price_str = f"${r['price']:,.2f}" if r["price"] else "N/A"
                    chg_color = "#00d084" if (r.get("change_1d") or 0) >= 0 else "#f43f5e"
                    all_s: dict = {}
                    for k in r["signals"]:
                        all_s.update(r["signals"][k])
                    top2 = sorted(all_s.items(), key=lambda x: x[1], reverse=True)[:2]
                    top2_text = "  ·  ".join(
                        SIGNAL_PLAIN.get(n, n).split("—")[0].strip()[:38] for n, _ in top2
                    )
                    st.markdown(f"""
<div class='rs-card-3' style='margin-bottom:8px'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <div>
      <span style='font-weight:700;font-size:1.05em;color:#e2e8f0'>{r["ticker"]}</span>
      <span style='color:#607d96;font-size:0.78em;margin-left:8px'>{r["name"][:28]} · {r.get("sector","")}</span>
    </div>
    <div style='text-align:right'>
      <span style='color:{color};font-weight:700;font-size:1.1em'>{score:.0f}</span>
      <span style='color:#607d96;font-size:0.72em'>/100</span>
      <span style='color:#7092b0;font-size:0.80em;margin-left:10px'>{price_str}</span>
    </div>
  </div>
  <div style='color:#607d96;font-size:0.73em;margin-top:5px'>Emerging: {top2_text}</div>
</div>""", unsafe_allow_html=True)
                    if st.button("Analyze →", key=f"t3_{r['ticker']}", use_container_width=is_mobile):
                        st.session_state.selected_result      = r
                        st.session_state.navigate_to_analysis = True

        if not tier1 and not tier2 and not tier3:
            st.warning("No setups above 50. Market may be in a risk-off environment. Check the Market tab.")

        # Score distribution — desktop only (too dense on small screens)
        if not is_mobile:
            st.divider()
            st.markdown("#### Score Distribution — All Analyzed Stocks")
            scores_list  = [r["composite"] for r in results]
            tickers_list = [r["ticker"]    for r in results]
            fig_dist = go.Figure(go.Bar(
                x=tickers_list, y=scores_list,
                marker_color=[score_color(s) for s in scores_list],
                text=[f"{s:.0f}" for s in scores_list],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>",
            ))
            fig_dist.add_hline(y=BUY_SIGNAL_THRESHOLD, line_dash="dash",
                               line_color="#f59e0b", annotation_text=f"Buy threshold ({BUY_SIGNAL_THRESHOLD})")
            fig_dist.add_hline(y=80, line_dash="dash",
                               line_color="#00d084", annotation_text="Highest conviction (80)")
            fig_dist.update_layout(
                template="plotly_dark", paper_bgcolor="#0b1119", plot_bgcolor="#0b1119",
                height=300, margin=dict(l=0,r=0,t=24,b=0),
                yaxis=dict(range=[0,105], gridcolor="#1a2535"),
                xaxis=dict(gridcolor="#1a2535"),
                showlegend=False,
            )
            st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:
    portfolio = st.session_state.get("portfolio", [])

    if not portfolio:
        st.markdown("### 💼 My Portfolio")
        st.info(
            "Upload your Yahoo Finance portfolio export to see every holding scored, "
            "ranked, and given a specific action (Add / Hold / Trim / Cut).\n\n"
            "**How to export:** Yahoo Finance → Finance → Portfolios → select your portfolio "
            "→ click the **Export** icon (top right of the holdings table) → upload the CSV in the sidebar."
        )
    else:
        # Build a lookup: ticker → scored result (auto-scan + any manual Deep Dive analyses)
        auto_results = st.session_state.get("auto_results", [])
        scored_map: dict[str, dict] = {r["ticker"]: r for r in auto_results}
        scored_map.update(st.session_state.get("manual_scores", {}))

        unscored = [h["ticker"] for h in portfolio if h["ticker"] not in scored_map]
        if unscored:
            st.warning(
                f"{len(unscored)} holding(s) haven't been scanned yet: "
                f"**{', '.join(unscored[:8])}{'…' if len(unscored) > 8 else ''}**. "
                "Click **Re-Scan Now** in the sidebar to score them."
            )

        # ── Batch-fetch fresh prices for every portfolio ticker ────────────────
        all_pf_tickers = tuple(h["ticker"] for h in portfolio)
        batch_prices = fetch_live_prices(all_pf_tickers)

        def _live(h: dict, r: dict | None) -> float:
            """Live price priority: batch yfinance → scan result → CSV snapshot."""
            for candidate in (
                batch_prices.get(h["ticker"]),
                r["price"] if r and r.get("price") else None,
                h["current_price_csv"],
            ):
                try:
                    f = float(candidate or 0)
                    if f > 0 and f == f:   # f != f catches NaN
                        return f
                except (TypeError, ValueError):
                    pass
            return 0.0

        # ── Portfolio summary ──────────────────────────────────────────────────
        total_cost  = 0.0
        total_value = 0.0
        for h in portfolio:
            r = scored_map.get(h["ticker"])
            live_price = _live(h, r)
            total_cost  += h["shares"] * h["cost_basis"]
            total_value += h["shares"] * live_price

        total_pnl     = total_value - total_cost
        total_pnl_pct = (total_value / total_cost - 1) * 100 if total_cost > 0 else 0.0
        pnl_color     = "#00d084" if total_pnl >= 0 else "#f43f5e"
        n_buy         = sum(1 for h in portfolio
                            if scored_map.get(h["ticker"], {}).get("is_buy"))

        st.markdown("### 💼 Portfolio Overview")
        if is_mobile:
            sm1, sm2 = st.columns(2)
            sm3, sm4 = st.columns(2)
        else:
            sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Positions", len(portfolio))
        sm2.metric("Buy Signals", f"{n_buy} / {len(portfolio)}")
        sm3.metric("Total Value", f"${total_value:,.0f}")
        sm4.metric(
            "Unrealized P&L",
            f"${total_pnl:+,.0f}",
            delta=f"{total_pnl_pct:+.1f}%",
        )

        st.divider()

        # ── Position cards — sorted: ADD first, EXIT last ──────────────────────
        def _sort_key(h):
            r     = scored_map.get(h["ticker"])
            score = r["composite"] if r else -1
            live  = _live(h, r)
            pnl_p = (live / h["cost_basis"] - 1) * 100 if h["cost_basis"] > 0 and live > 0 else 0
            lbl, _ = portfolio_action(score if r else None, pnl_p)
            order  = {"ADD": 0, "BUY MORE": 1, "HOLD": 2, "TRIM": 3, "CUT": 4, "EXIT": 5, "UNSCORED": 6}
            return order.get(lbl, 7)

        sorted_portfolio = sorted(portfolio, key=_sort_key)

        for h in sorted_portfolio:
            r          = scored_map.get(h["ticker"])
            score      = r["composite"] if r else None
            sc         = score_color(score) if score is not None else "#607d96"
            live_price = _live(h, r)
            cost_basis = h["cost_basis"]
            shares     = h["shares"]
            pnl_dollar = shares * (live_price - cost_basis)
            pnl_pct    = (live_price / cost_basis - 1) * 100 if cost_basis > 0 else 0.0
            pos_value  = shares * live_price
            pnl_color_h = "#00d084" if pnl_dollar >= 0 else "#f43f5e"
            pnl_sign   = "+" if pnl_dollar >= 0 else "−"

            act_lbl, act_css = portfolio_action(score, pnl_pct)
            name   = r.get("name", "")[:28]   if r else h["ticker"]
            sector = r.get("sector", "")       if r else ""

            # Category score pills
            if r:
                cat_pills = "".join(
                    f"<span class='pill-cat' style='color:{score_color(v)}'>"
                    f"{CAT_ICONS.get(k,'')} {k[:4]} {v:.0f}</span>"
                    for k, v in r["categories"].items()
                )
            else:
                cat_pills = "<span style='color:#607d96;font-size:0.75em'>Not yet scored — re-scan to analyze</span>"

            tier = 1 if (score or 0) >= 80 else (2 if (score or 0) >= 65 else 3)
            css  = {1: "rs-card-1", 2: "rs-card-2", 3: "rs-card-3"}[tier]

            # Build score block separately to avoid complex f-string conditionals
            if score is not None:
                score_block = (
                    f"<div style='font-size:2em;font-weight:800;color:{sc};line-height:1'>{score:.0f}</div>"
                    f"<div style='font-size:0.61em;color:#607d96'>/100</div>"
                )
            else:
                score_block = "<div style='font-size:0.80em;color:#607d96;padding-top:4px'>—<br><span style='font-size:0.75em'>unscored</span></div>"

            # Show "no price" note when price data is missing
            if live_price > 0:
                detail_line = (
                    f"{shares:.2f} sh &nbsp;·&nbsp; cost ${cost_basis:.2f}"
                    f" &nbsp;·&nbsp; live ${live_price:.2f}"
                    f" &nbsp;·&nbsp; <span style='color:#7092b0'>${pos_value:,.0f} value</span>"
                )
            else:
                detail_line = f"{shares:.2f} sh &nbsp;·&nbsp; cost ${cost_basis:.2f} &nbsp;·&nbsp; <span style='color:#607d96'>price unavailable — re-scan</span>"

            st.markdown(
                f"<div class='{css}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:10px'>"
                f"<div style='flex:1;min-width:0'>"
                f"<div style='display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin-bottom:3px'>"
                f"<span style='font-size:1.2em;font-weight:800;color:#f1f5f9'>{h['ticker']}</span>"
                f"<span class='badge {act_css}'>{act_lbl}</span>"
                f"</div>"
                f"<div style='color:#7092b0;font-size:0.74em;margin-bottom:9px'>{name}{' · ' + sector if sector else ''}</div>"
                f"<div style='background:#0b1119;border-radius:7px;padding:9px 11px;margin-bottom:9px'>"
                f"<div style='display:flex;align-items:baseline;gap:8px;flex-wrap:wrap'>"
                f"<span style='color:{pnl_color_h};font-size:1.05em;font-weight:700'>{pnl_sign}${abs(pnl_dollar):,.0f}</span>"
                f"<span style='color:{pnl_color_h};font-size:0.82em'>({pnl_pct:+.1f}%)</span>"
                f"<span style='color:#607d96;font-size:0.70em'>unrealized</span>"
                f"</div>"
                f"<div style='font-size:0.72em;color:#607d96;margin-top:5px'>{detail_line}</div>"
                f"</div>"
                f"<div style='margin-bottom:6px;line-height:2.0'>{cat_pills}</div>"
                f"</div>"
                f"<div style='text-align:right;flex-shrink:0'>{score_block}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if r and st.button("Full Analysis →", key=f"pf_{h['ticker']}", use_container_width=True):
                st.session_state.selected_result      = r
                st.session_state.navigate_to_analysis = True


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    result = st.session_state.get("selected_result")

    if is_mobile:
        direct_ticker = st.text_input("Analyze any ticker",
                                       placeholder="e.g. NVDA, AAPL, CRWV",
                                       label_visibility="collapsed")
        analyze_btn = st.button("Analyze →", type="primary", use_container_width=True)
    else:
        col_in, col_btn = st.columns([4, 1])
        with col_in:
            direct_ticker = st.text_input("Analyze any ticker",
                                           placeholder="e.g. NVDA, AAPL, CRWV",
                                           label_visibility="collapsed")
        with col_btn:
            st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
            analyze_btn = st.button("Analyze →", type="primary", use_container_width=True)

    if analyze_btn and direct_ticker.strip():
        ticker_upper = direct_ticker.strip().upper()
        with st.spinner(f"Running 41 signals on {ticker_upper}…"):
            snap   = get_macro_snapshot()
            result = score_stock(
                ticker_upper,
                spy_df=snap["spy_df"], vix_df=snap["vix_df"],
                sector_perf=snap["sector_perf"], pcr=snap["pcr"],
            )
            if result:
                st.session_state.selected_result = result
                # Also cache so the Portfolio tab can see it
                ms = st.session_state.get("manual_scores", {})
                ms[result["ticker"]] = result
                st.session_state.manual_scores = ms
            else:
                st.error(f"Could not fetch data for **{ticker_upper}**. Check the ticker symbol and try again.")
                result = None

    if result is None:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.info("Click **Full Analysis →** on any stock card in Top Picks, or type a ticker above.")
    else:
        ticker = result["ticker"]
        score  = result["composite"]
        color  = score_color(score)
        badge_html, primary, caveat = make_action_statement(result)
        reasons, risks = generate_thesis(result)
        meta  = result["meta"]
        plan  = trade_plan(result)

        # ── Header ──
        if is_mobile:
            st.markdown(
                f"<div style='margin-bottom:2px'>"
                f"<span style='font-size:1.65em;font-weight:800;color:#f1f5f9;letter-spacing:-0.4px'>{ticker}</span>"
                f"&nbsp;&nbsp;{badge_html}"
                f"</div>"
                f"<div style='color:#7e9ab8;font-size:0.79em;margin-bottom:4px'>{result.get('name','')}</div>"
                f"<div style='color:#607d96;font-size:0.75em;margin-bottom:14px'>"
                f"{result.get('sector','Unknown')} · {result.get('industry','Unknown')}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-bottom:4px'>"
                f"<span style='font-size:2em;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px'>{ticker}</span>"
                f"<span style='color:#64748b;font-size:1em'>{result.get('name','')}</span>"
                f"{badge_html}"
                f"</div>"
                f"<div style='color:#7092b0;font-size:0.83em;margin-bottom:18px'>"
                f"{result.get('sector','Unknown')} · {result.get('industry','Unknown')}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── KPI strip ──
        price  = result.get("price")
        chg1d  = result.get("change_1d")
        chg1m  = result.get("change_1m")
        _score_tile = (
            f"<div style='background:#0e1520;border:1px solid {color}40;border-radius:10px;"
            f"padding:14px 18px;text-align:center'>"
            f"<div style='font-size:3em;font-weight:800;color:{color};line-height:1'>{score:.0f}</div>"
            f"<div style='color:#607d96;font-size:0.72em;letter-spacing:0.5px;margin-top:2px'>RALLY SCORE / 100</div>"
            f"</div>"
        )
        if is_mobile:
            # Score: full-width prominent block
            st.markdown(_score_tile, unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            # Price + 1-Month side by side
            mA, mB = st.columns(2)
            mA.metric("Price", f"${price:,.2f}" if price else "N/A",
                      delta=f"{chg1d:+.1f}% today" if chg1d is not None else None)
            mB.metric("1-Month", f"{chg1m:+.1f}%" if chg1m is not None else "N/A")
            # Market Cap + Analyst Target side by side
            mC, mD = st.columns(2)
            mC.metric("Market Cap", fmt_cap(result.get("market_cap")))
            mD.metric(
                "Target",
                f"${meta['analyst_target']:.2f}" if meta.get("analyst_target") else "N/A",
                delta=(f"{(meta['analyst_target']/price-1)*100:+.1f}% upside"
                       if meta.get("analyst_target") and price else None),
            )
        else:
            h1, h2, h3, h4, h5 = st.columns([1.4,1,1,1,1])
            with h1:
                st.markdown(_score_tile, unsafe_allow_html=True)
            h2.metric("Price", f"${price:,.2f}" if price else "N/A",
                      delta=f"{chg1d:+.1f}% today" if chg1d is not None else None)
            h3.metric("1-Month Return", f"{chg1m:+.1f}%" if chg1m is not None else "N/A")
            h4.metric("Market Cap", fmt_cap(result.get("market_cap")))
            h5.metric(
                "Analyst Target",
                f"${meta['analyst_target']:.2f}" if meta.get("analyst_target") else "N/A",
                delta=(f"{(meta['analyst_target']/price-1)*100:+.1f}% upside"
                       if meta.get("analyst_target") and price else None),
            )

        st.divider()

        # ── Recommendation box ──
        caveat_line = (
            f"<div style='margin-top:8px;padding-top:8px;border-top:1px solid rgba(251,146,60,0.2)'>"
            f"<span style='color:#fbbf24;font-size:0.72em;font-weight:700'>⚠ RISK WATCH · </span>"
            f"<span style='color:#fde68a;font-size:0.80em'>{caveat}</span></div>"
        ) if caveat else ""

        st.markdown(
            f"<div class='box-action'>"
            f"<div style='font-size:0.69em;color:#4ade80;font-weight:700;letter-spacing:0.8px;margin-bottom:6px'>RECOMMENDATION</div>"
            f"<div style='color:#d1fae5;font-size:0.90em;line-height:1.6'>{primary}</div>"
            f"{caveat_line}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Trade plan ──
        stop_str   = f"${plan['stop']:,.2f} ({plan['stop_pct']:.1f}% below)" if plan["stop"] else "N/A"
        target_str = (f"${plan['target']:,.2f} +{plan['upside_pct']:.0f}% ({plan['target_note']})"
                      if plan["target"] else "N/A")
        rr_str     = f"{plan['rr']:.1f} : 1" if plan["rr"] else "N/A"

        st.markdown(
            f"<div class='box-trade'>"
            f"<div style='font-size:0.69em;color:#38bdf8;font-weight:700;letter-spacing:0.8px;margin-bottom:10px'>TRADE PLAN</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px 24px'>"
            f"<div>"
            f"  <div style='font-size:0.69em;color:#607d96;letter-spacing:0.4px'>ENTRY</div>"
            f"  <div style='color:#94a3b8;font-size:0.82em;margin-top:2px'>{plan['entry_note']}</div>"
            f"</div>"
            f"<div>"
            f"  <div style='font-size:0.69em;color:#607d96;letter-spacing:0.4px'>POSITION SIZE</div>"
            f"  <div style='color:#94a3b8;font-size:0.82em;margin-top:2px'>{plan['pos_size']}</div>"
            f"</div>"
            f"<div>"
            f"  <div style='font-size:0.69em;color:#607d96;letter-spacing:0.4px'>STOP LOSS</div>"
            f"  <div style='color:#f43f5e;font-size:0.82em;font-weight:600;margin-top:2px'>{stop_str}</div>"
            f"</div>"
            f"<div>"
            f"  <div style='font-size:0.69em;color:#607d96;letter-spacing:0.4px'>PRICE TARGET</div>"
            f"  <div style='color:#00d084;font-size:0.82em;font-weight:600;margin-top:2px'>{target_str}</div>"
            f"</div>"
            f"<div>"
            f"  <div style='font-size:0.69em;color:#607d96;letter-spacing:0.4px'>RISK / REWARD</div>"
            f"  <div style='color:#94a3b8;font-size:0.82em;margin-top:2px'>{rr_str}</div>"
            f"</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Bull thesis / risk factors ──
        if is_mobile:
            t_col = st.container()
            r_col = st.container()
        else:
            t_col, r_col = st.columns(2)
        with t_col:
            st.markdown("#### 🟢 Why This Could Rally")
            html = "".join(f"<li style='margin-bottom:9px;color:#d1fae5;font-size:0.86em'>{r}</li>" for r in reasons)
            st.markdown(f"<div class='box-bull'><ul style='margin:0;padding-left:18px'>{html}</ul></div>",
                        unsafe_allow_html=True)
        with r_col:
            st.markdown("#### 🔴 Risks & Weaknesses")
            html = "".join(f"<li style='margin-bottom:9px;color:#fecaca;font-size:0.86em'>{r}</li>" for r in risks)
            st.markdown(f"<div class='box-bear'><ul style='margin:0;padding-left:18px'>{html}</ul></div>",
                        unsafe_allow_html=True)

        st.divider()

        # ── Chart + radar ──
        if is_mobile:
            ch_col = st.container()
            ra_col = st.container()
        else:
            ch_col, ra_col = st.columns([3, 1])
        with ch_col:
            df_p = result.get("price_df")
            if df_p is not None and not df_p.empty:
                fig_p = build_price_chart(df_p, ticker)
                if is_mobile:
                    fig_p.update_layout(height=300)
                st.plotly_chart(fig_p, use_container_width=True)
        with ra_col:
            st.markdown("**Category Scores**")
            if is_mobile:
                # 2-column grid of score tiles — radar is unreadable at mobile width
                cats = list(result["categories"].items())
                for row_start in range(0, len(cats), 2):
                    pair = cats[row_start:row_start + 2]
                    cols_c = st.columns(len(pair))
                    for ci, (cat, val) in enumerate(pair):
                        c = score_color(val)
                        cols_c[ci].markdown(
                            f"<div style='background:#0e1520;border:1px solid #1a2840;"
                            f"border-radius:8px;padding:9px 10px;text-align:center;margin-bottom:8px'>"
                            f"<div style='font-size:1.3em;font-weight:800;color:{c};line-height:1.1'>{val:.0f}</div>"
                            f"<div style='font-size:0.62em;color:#7e9ab8;margin-top:3px;letter-spacing:0.3px'>"
                            f"{CAT_ICONS.get(cat,'')} {cat.upper()}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.plotly_chart(build_radar(result["categories"]), use_container_width=True)
                for cat, val in result["categories"].items():
                    c = score_color(val)
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;font-size:0.80em;margin-bottom:4px'>"
                        f"<span style='color:#7e9ab8'>{CAT_ICONS.get(cat,'')} {cat}</span>"
                        f"<span style='color:{c};font-weight:700'>{val:.0f}/100</span></div>",
                        unsafe_allow_html=True,
                    )

        st.divider()

        # ── Signal breakdown ──
        st.markdown("### Every Signal — Plain English")
        st.caption("✅ Strong  🟡 Moderate  ⚠️ Weak  🔴 Bearish")

        cat_order = [
            ("technical",     "Technical",     "📊 Technical — Price & Volume"),
            ("fundamental",   "Fundamental",   "📋 Fundamental — Business Quality"),
            ("institutional", "Institutional", "🏛️ Institutional — Smart Money"),
            ("macro",         "Macro",         "🌊 Macro — Market Environment"),
            ("sentiment",     "Sentiment",     "🧠 Sentiment — Positioning"),
            ("timing",        "Timing",        "⏰ Timing — Entry Quality"),
        ]
        if is_mobile:
            left_col = st.container()
            right_col = left_col  # single stacked column on mobile
        else:
            left_col, right_col = st.columns(2)
        for idx, (key, cat_name, title) in enumerate(cat_order):
            col = left_col if idx % 2 == 0 else right_col
            with col:
                cat_score = result["categories"][cat_name]
                cat_color = score_color(cat_score)
                st.markdown(
                    f"<div class='cat-panel'>"
                    f"<div class='cat-header'>"
                    f"<b style='font-size:0.88em;color:#e2e8f0'>{title}</b>"
                    f"<span style='color:{cat_color};font-weight:700;font-size:0.88em'>{cat_score:.0f}/100</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                for sig_name, sig_val in result["signals"][key].items():
                    render_signal_row(sig_name, sig_val)
                st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        # ── Key numbers ──
        st.markdown("### Key Numbers")
        if is_mobile:
            km1, km2 = st.columns(2)
            km3, km4 = st.columns(2)
            km5, km6 = st.columns(2)
        else:
            km1, km2, km3, km4, km5, km6 = st.columns(6)
        km1.metric("P/E (TTM)",   f"{meta['pe_ratio']:.1f}x"   if meta.get("pe_ratio")   else "N/A")
        km2.metric("Forward P/E", f"{meta['forward_pe']:.1f}x" if meta.get("forward_pe") else "N/A")
        km3.metric("Beta",        f"{meta['beta']:.2f}"         if meta.get("beta")        else "N/A",
                   help="A beta of 2.0 means this stock moves roughly 2x the market in both directions.")
        km4.metric("52W High",    f"${meta['52w_high']:.2f}"   if meta.get("52w_high")   else "N/A")
        km5.metric("52W Low",     f"${meta['52w_low']:.2f}"    if meta.get("52w_low")    else "N/A")
        km6.metric("Short Float",
                   f"{meta['short_float']*100:.1f}%" if meta.get("short_float") else "N/A",
                   help="Short float above 20% signals high squeeze potential on any positive catalyst.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MARKET
# ══════════════════════════════════════════════════════════════════════════════
with tab_market:
    st.markdown("### 🌊 Market Environment")
    st.caption("Macro context determines whether to be aggressive, selective, or defensive on every trade.")

    with st.spinner("Loading market data…"):
        snap = get_macro_snapshot()

    vix      = snap["vix_current"]
    vix_chg  = snap["vix_change"]
    spy_ret  = snap["spy_ret_1m"]
    sect_prf = snap["sector_perf"]
    pcr      = snap["pcr"]

    if vix is not None and spy_ret is not None:
        if vix < 18 and spy_ret > 3:
            ec,ei,et = "#00d084","🟢","RISK-ON"
            ed = "Market is calm and trending upward. Be aggressive with your highest-conviction longs. Full positions are appropriate."
        elif vix > 28 or spy_ret < -5:
            ec,ei,et = "#f43f5e","🔴","RISK-OFF"
            ed = "Fear is elevated or the market is declining. Only trade the strongest setups and keep positions at half-size or less."
        else:
            ec,ei,et = "#f59e0b","🟡","MIXED"
            ed = "Neither clearly bullish nor clearly bearish. Focus on stock-specific setups that can outperform on their own fundamentals."
    else:
        ec,ei,et,ed = "#64748b","🔵","LOADING","Fetching data…"

    st.markdown(
        f"<div class='env-banner' style='background:#0e1520;border:1.5px solid {ec}40'>"
        f"<span style='font-size:1.35em;font-weight:800;color:{ec}'>{ei} {et}</span><br>"
        f"<span style='color:#94a3b8;font-size:0.88em;line-height:1.6'>{ed}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if is_mobile:
        mc1, mc2 = st.columns(2)
        mc3, mc4 = st.columns(2)
    else:
        mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("VIX Fear Index", f"{vix:.1f}" if vix else "N/A",
               delta=f"{vix_chg:+.1f} vs 5d ago" if vix_chg is not None else None,
               delta_color="inverse")
    mc1.caption("< 18 = calm · 18–28 = caution · > 28 = fear")
    mc2.metric("S&P 500 (1-Month)", f"{spy_ret:+.1f}%" if spy_ret is not None else "N/A")
    mc2.caption("> +3% = tailwind · < -5% = fight the tape")
    mc3.metric("Put/Call Ratio", f"{pcr:.2f}" if pcr else "N/A")
    mc3.caption("> 1.2 = extreme fear = contrarian buy signal")
    if sect_prf:
        best = max(sect_prf.items(), key=lambda x: x[1]["return_1m"])
        mc4.metric("Hottest Sector", best[0], f"{best[1]['return_1m']:+.1f}% (1M)")
        mc4.caption("Rotate long bias into the sector receiving inflows")

    st.divider()

    if is_mobile:
        s_col = st.container()
        c_col = st.container()
    else:
        s_col, c_col = st.columns([1.2, 1])
    with s_col:
        st.markdown("#### Sector Rotation — Where Is the Money Going?")
        st.caption("Green bars = inflows. Bias your longs toward those sectors. Avoid longs in red sectors.")
        if sect_prf:
            st.plotly_chart(build_sector_chart(sect_prf), use_container_width=True)
    with c_col:
        st.markdown("#### VIX (Fear) — 6 Months")
        st.caption("A spike followed by a fast drop = best time to be long. Flat and low = calm conditions.")
        vix_df = snap["vix_df"]
        if vix_df is not None:
            cls = vix_df["Close"].squeeze()
            fv = go.Figure()
            fv.add_trace(go.Scatter(
                x=vix_df.index, y=cls, fill="tozeroy",
                fillcolor="rgba(244,63,94,0.08)", line=dict(color="#f43f5e", width=1.5),
            ))
            fv.add_hline(y=20, line_dash="dash", line_color="#f59e0b", annotation_text="Caution (20)")
            fv.add_hline(y=30, line_dash="dash", line_color="#f43f5e",  annotation_text="Fear (30)")
            fv.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1520", plot_bgcolor="#0e1520",
                height=160 if is_mobile else 200,
                margin=dict(l=0,r=0,t=5,b=0), showlegend=False,
                yaxis=dict(gridcolor="#1a2535"), xaxis=dict(gridcolor="#1a2535"),
            )
            st.plotly_chart(fv, use_container_width=True)

        st.markdown("#### S&P 500 — 6 Months")
        spy_df = snap["spy_df"]
        if spy_df is not None:
            cls = spy_df["Close"].squeeze()
            fs = go.Figure()
            fs.add_trace(go.Scatter(x=spy_df.index, y=cls, name="S&P 500",
                                     line=dict(color="#00d084", width=1.5)))
            fs.add_trace(go.Scatter(x=spy_df.index, y=cls.rolling(50).mean(),
                                     name="50d avg", line=dict(color="#38bdf8", width=1, dash="dash")))
            fs.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1520", plot_bgcolor="#0e1520",
                height=155 if is_mobile else 190,
                margin=dict(l=0,r=0,t=5,b=0),
                legend=dict(orientation="h", y=1.15, font=dict(size=9)),
                yaxis=dict(gridcolor="#1a2535"), xaxis=dict(gridcolor="#1a2535"),
            )
            st.plotly_chart(fs, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOM SCAN
# ══════════════════════════════════════════════════════════════════════════════
with tab_custom:
    st.markdown("### 🔍 Custom Scan")
    st.caption("Run a full 41-signal analysis on any universe with your own filters.")

    if "custom_results" not in st.session_state:
        st.session_state.custom_results = []

    if run_custom:
        if universe_choice == "S&P 500":
            scan_tickers = get_sp500_tickers()
        elif universe_choice == "Custom Tickers":
            scan_tickers = custom_tickers
        else:
            scan_tickers = list(build_dynamic_universe().keys())

        pb2 = st.progress(0, text="Starting…")
        stxt2 = st.empty()

        def custom_progress(i, total, ticker):
            pb2.progress(int(i / max(total, 1) * 100),
                         text=f"Scanning {ticker} ({i + 1}/{total})")
            stxt2.caption(f"Analyzing: **{ticker}**")

        custom_results = scan_universe(
            scan_tickers,
            progress_callback=custom_progress,
            sector_filter=sector_filter,
            min_market_cap=min_cap,
            max_tickers=int(max_scan),
        )
        pb2.progress(100, text="Complete.")
        stxt2.empty()
        st.session_state.custom_results = custom_results

    custom_results = st.session_state.get("custom_results", [])

    if not custom_results:
        st.info("Configure your scan in the sidebar and click **Run Custom Scan →**.")
    else:
        buys_c = [r for r in custom_results if r["is_buy"]]
        if is_mobile:
            ca, cb = st.columns(2)
            cc, cd = st.columns(2)
        else:
            ca, cb, cc, cd = st.columns(4)
        ca.metric("Scanned",     len(custom_results))
        cb.metric("Buy Signals", len(buys_c))
        cc.metric("Avg Score",   f"{np.mean([r['composite'] for r in custom_results]):.1f}")
        cd.metric("Top Score",   f"{custom_results[0]['composite']:.1f}", custom_results[0]["ticker"])
        st.divider()

        rows = []
        for r in custom_results:
            lbl, _ = action_label(r["composite"])
            rows.append({
                "Ticker":        r["ticker"],
                "Company":       r["name"][:24],
                "Sector":        r.get("sector", ""),
                "Price":         f"${r['price']:,.2f}" if r["price"] else "N/A",
                "1D%":           f"{r['change_1d']:+.1f}%" if r["change_1d"] is not None else "N/A",
                "Score":         r["composite"],
                "Action":        lbl,
                "Technical":     r["categories"]["Technical"],
                "Fundamental":   r["categories"]["Fundamental"],
                "Institutional": r["categories"]["Institutional"],
                "Macro":         r["categories"]["Macro"],
            })
        df_c = pd.DataFrame(rows)

        def _cs(v):
            return f"color:{score_color(v)};font-weight:700" if isinstance(v, (int, float)) else ""

        styled = (
            df_c.style
            .map(_cs, subset=["Score","Technical","Fundamental","Institutional","Macro"])
            .format({"Score":"{:.1f}","Technical":"{:.1f}","Fundamental":"{:.1f}",
                     "Institutional":"{:.1f}","Macro":"{:.1f}"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=480)

        st.divider()
        sel = st.selectbox("Deep-dive into →", [r["ticker"] for r in custom_results],
                            key="custom_sel", label_visibility="visible")
        if st.button("Full Analysis →", type="primary"):
            st.session_state.selected_result      = next(r for r in custom_results if r["ticker"] == sel)
            st.session_state.navigate_to_analysis = True


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='rs-footer'>
<b>Disclaimer</b> · Rally Scout is an informational tool and does not constitute financial advice, a solicitation,
or a recommendation to buy or sell any security. All signals are generated algorithmically from public market data.
Past signal performance does not guarantee future results. Always conduct your own due diligence and consult a
licensed financial advisor before making investment decisions. Market data is sourced from Yahoo Finance and Finviz
and may be delayed by 15 minutes or more.
</div>
""", unsafe_allow_html=True)


# ── Tab navigation — fires after all tabs are rendered ────────────────────────
if st.session_state.get("navigate_to_analysis"):
    st.session_state.navigate_to_analysis = False
    navigate_to_tab("Analysis" if is_mobile else "Deep Dive")
