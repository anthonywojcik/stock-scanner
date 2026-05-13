# ── AI / Tech / Infrastructure anchors ───────────────────────────────────────
# Always included in every auto-scout scan regardless of dynamic layer results.
# Focused on the dominant themes: AI, semiconductors, cloud infrastructure,
# AI power/data-center infrastructure, and the key software platforms running on top.
AI_TECH_ANCHORS = [
    # Semiconductors — the picks-and-shovels of AI
    "NVDA", "AMD", "AVGO", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ARM",
    "SMCI", "ASML", "TSM", "INTC", "ON",
    # Mega-cap tech platforms
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "ORCL", "CRM", "ADBE", "NOW",
    # AI-native software
    "PLTR", "CRWD", "NET", "DDOG", "SNOW", "ZS", "MDB", "HUBS", "GTLB", "S",
    # AI power & data-center infrastructure (AI needs electricity and real estate)
    "VST", "CEG", "NEE", "EQIX", "DLR", "OKLO", "SMR",
    # Financials levered to tech/growth
    "GS", "V", "MA", "COIN", "MSTR",
    # High-beta growth with AI exposure
    "UBER", "ABNB", "SPOT", "RBLX", "HOOD", "SOFI",
]

# ── Dynamic universe layer settings ──────────────────────────────────────────
DYNAMIC_SHORT_INTEREST_MIN_PCT = 10   # Min short float % to qualify for squeeze layer
DYNAMIC_SHORT_INTEREST_N       = 25   # Max stocks from this layer
DYNAMIC_MOMENTUM_N             = 25   # Max stocks from momentum screen
DYNAMIC_SP500_TOP_N            = 20   # Max stocks from S&P 500 top performers

# ── Scoring ───────────────────────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "technical":     0.25,
    "fundamental":   0.25,
    "institutional": 0.20,
    "macro":         0.15,
    "sentiment":     0.10,
    "timing":        0.05,
}

BUY_SIGNAL_THRESHOLD = 65

# ── Optional free FRED API key ────────────────────────────────────────────────
# Get one at https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = ""

# ── Sector ETFs for rotation analysis ────────────────────────────────────────
SECTOR_ETFS = {
    "Technology":       "XLK",
    "Healthcare":       "XLV",
    "Financials":       "XLF",
    "Consumer Disc.":   "XLY",
    "Industrials":      "XLI",
    "Energy":           "XLE",
    "Materials":        "XLB",
    "Utilities":        "XLU",
    "Real Estate":      "XLRE",
    "Comm. Services":   "XLC",
    "Consumer Staples": "XLP",
}

FALLBACK_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","V",
    "UNH","XOM","LLY","WMT","MA","AVGO","HD","CVX","MRK","ABBV",
    "AMD","NFLX","ORCL","MCD","COST","CRM","ADBE","BAC","GS","PLTR",
]
