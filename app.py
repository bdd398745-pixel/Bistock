# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# ==================== Page Config ====================
st.set_page_config(
    page_title="India Intraday Screener — Pro (SMA/RSI/MACD + VWAP + ORB)",
    page_icon="⚡",
    layout="wide",
)

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")
TODAY_IST = datetime.now(IST).date()

# ==================== Constants ======================
NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
MARKET_OPEN_MIN = 9*60 + 15
MARKET_CLOSE_MIN = 15*60 + 30

# ==================== Utilities ======================
def _to_yahoo_symbol(symbol: str, exchange: str = "NS") -> str:
    s = symbol.strip().upper()
    return f"{s}.NS" if exchange == "NS" else (f"{s}.BO" if exchange == "BO" else s)

@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_nse_universe() -> pd.DataFrame:
    """Fetch NSE Equity symbols from archive CSV. Returns SYMBOL, YF_TICKER."""
    try:
        headers = {"User-Agent":"Mozilla/5.0"}
        resp = requests.get(NSE_EQUITY_LIST_URL, headers=headers, timeout=25)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if "SYMBOL" not in df.columns:
            raise ValueError("SYMBOL column not found")
        df["SYMBOL"] = df["SYMBOL"].astype(str)
        df["YF_TICKER"] = df["SYMBOL"].apply(lambda s: _to_yahoo_symbol(s, "NS"))
        df = df.drop_duplicates(subset=["YF_TICKER"]).reset_index(drop=True)
        return df[["SYMBOL","YF_TICKER"]]
    except Exception as e:
        st.warning(f"Could not fetch NSE equity list automatically: {e}")
        return pd.DataFrame({
            "SYMBOL":["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","BHARTIARTL","KOTAKBANK","LT","ITC"],
            "YF_TICKER":["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","ITC.NS"]
        })

def max_allowed_period(interval: str) -> str:
    """
    Yahoo Finance intraday limits (conservative):
      - '1m' <= 7d
      - 2m/5m/15m/30m <= 30d
      - 60m/90m/1h <= 180d
      - else 365d
    """
    i = interval.lower()
    if i == "1m":
        return "7d"
    if i in {"2m","5m","15m","30m"}:
        return "30d"
    if i in {"60m","90m","1h"}:
        return "180d"
    return "365d"

def clamp_period_days(request_days: int, interval: str) -> int:
    mx = max_allowed_period(interval)
    mx_days = int(mx.replace("d",""))
    return min(request_days, mx_days)

@st.cache_data(ttl=60*15, show_spinner=False)
def yf_download_batch(tickers, period: str = "30d", interval: str = "5m"):
    """Batch download with chunking; returns MultiIndex columns (Ticker, Field)."""
    if not tickers:
        return pd.DataFrame()
    chunk_size = 250
    chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]
    frames = []
    for ch in chunks:
        data = yf.download(
            tickers=ch,
            period=period,
            interval=interval,
            group_by='ticker',
            auto_adjust=True,
            threads=True,
            progress=False
        )
        if data.empty:
            continue
        if not isinstance(data.columns, pd.MultiIndex):
            data = pd.concat({ch[0]: data}, axis=1)
        frames.append(data)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined

# ==================== Indicators ======================
def _ensure_long(df_multi: pd.DataFrame) -> pd.DataFrame:
    """Convert MultiIndex columns to long (Date index, columns=OHLCV + Ticker col)."""
    df_multi = df_multi.swaplevel(axis=1).sort_index(axis=1)
    frames = []
    for t in df_multi.columns.get_level_values(0).unique():
        sub = df_multi[t].copy()
        sub["Ticker"] = t
        frames.append(sub)
    long = pd.concat(frames, axis=0)
    long = long.dropna(subset=["Close"], how="any")
    return long

def add_time_cols_ist(g: pd.DataFrame) -> pd.DataFrame:
    idx = g.index
    if idx.tz is None:
        idx = idx.tz_localize(UTC)
    ist_idx = idx.tz_convert(IST)
    g = g.copy()
    g.index = ist_idx
    g["SessionDate"] = g.index.date
    g["MinutesSinceOpen"] = ((g.index.hour*60 + g.index.minute) - MARKET_OPEN_MIN).astype(int)
    g["MinutesToClose"] = MARKET_CLOSE_MIN - (g.index.hour*60 + g.index.minute)
    return g

def compute_indicators_intraday(long_df: pd.DataFrame) -> pd.DataFrame:
    def calc_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_index()
        g = add_time_cols_ist(g)

        # SMAs
        g["SMA20"] = g["Close"].rolling(20, min_periods=20).mean()
        g["SMA50"] = g["Close"].rolling(50, min_periods=50).mean()

        # RSI14
        delta = g["Close"].diff()
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(up, index=g.index).rolling(14).mean()
        roll_down = pd.Series(down, index=g.index).rolling(14).mean()
        rs = roll_up / (roll_down + 1e-9)
        g["RSI14"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = g["Close"].ewm(span=12, adjust=False).mean()
        ema26 = g["Close"].ewm(span=26, adjust=False).mean()
        g["MACD"] = ema12 - ema26
        g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
        g["MACD_Hist"] = g["MACD"] - g["MACD_Signal"]

        # Returns & vol proxy
        g["Ret_3"] = g["Close"].pct_change(3)
        g["Ret_5"] = g["Close"].pct_change(5)
        g["VolProxy"] = g["Close"].pct_change().rolling(240).std()

        # Typical price and VWAP (session)
        tp = (g["High"] + g["Low"] + g["Close"]) / 3.0
        g["_TPxV"] = tp * g["Volume"]
        g["VWAP"] = g.groupby("SessionDate").apply(
            lambda df: (df["_TPxV"].cumsum() / (df["Volume"].replace(0, np.nan).cumsum())).values
        ).reset_index(level=0, drop=True)
        g["VWAP_Slope3"] = g["VWAP"].diff(3)
        g.drop(columns=["_TPxV"], inplace=True)

        # ATR(14) using intraday True Range
        tr1 = g["High"] - g["Low"]
        tr2 = (g["High"] - g["Close"].shift()).abs()
        tr3 = (g["Low"] - g["Close"].shift()).abs()
        g["TR"] = np.nanmax(np.vstack([tr1.values, tr2.values, tr3.values]), axis=0)
        g["ATR14"] = pd.Series(g["TR"], index=g.index).rolling(14, min_periods=14).mean()

        return g

    out = long_df.groupby("Ticker", group_keys=False).apply(calc_group)
    return out

def base_signal_rule(row: pd.Series) -> str:
    s20, s50 = row.get("SMA20"), row.get("SMA50")
    rsi = row.get("RSI14")
    macd, sig = row.get("MACD"), row.get("MACD_Signal")
    if pd.notna(s20) and pd.notna(s50) and pd.notna(rsi) and pd.notna(macd) and pd.notna(sig):
        buy = (s20 > s50) and (40 <= rsi <= 70) and (macd > sig)
        sell = (s20 < s50) and ((rsi > 70) or (macd < sig))
        if buy: return "BUY"
        if sell: return "SELL"
    return "HOLD"

def intraday_score(row: pd.Series) -> float:
    close = row.get("Close", np.nan)
    s20, s50 = row.get("SMA20", np.nan), row.get("SMA50", np.nan)
    rsi = row.get("RSI14", np.nan)
    macd_hist = row.get("MACD_Hist", np.nan)
    volp = row.get("VolProxy", np.nan)
    ret3 = row.get("Ret_3", np.nan)
    vwap = row.get("VWAP", np.nan)
    vsl = row.get("VWAP_Slope3", 0.0)

    sma_comp = 0.0 if not np.isfinite(close*s20*s50) else (s20 - s50) / max(close, 1e-9)
    rsi_comp = 0.0 if not np.isfinite(rsi) else np.clip((rsi - 50.0)/20.0, -1.5, 1.5)
    macd_comp = 0.0
    denom = volp if (np.isfinite(volp) and volp > 1e-6) else 0.02
    if np.isfinite(macd_hist):
        macd_comp = np.clip(macd_hist/denom, -3.0, 3.0)
    mom_comp = 0.0 if not np.isfinite(ret3) else np.clip(ret3, -0.10, 0.10)
    vwap_comp = 0.0
    if np.isfinite(vwap) and np.isfinite(close):
        vwap_comp = np.clip((close - vwap) / max(close, 1e-9), -0.02, 0.02)
        slope_bonus = 0.002 if (np.isfinite(vsl) and vsl > 0) else (-0.002 if np.isfinite(vsl) and vsl < 0 else 0.0)
        vwap_comp += slope_bonus

    w_sma, w_rsi, w_macd, w_mom, w_vwap = 0.25, 0.20, 0.25, 0.15, 0.15
    score = w_sma*sma_comp + w_rsi*rsi_comp + w_macd*macd_comp + w_mom*mom_comp + w_vwap*vwap_comp
    return float(score)

# ==================== ORB & PDH/PDL ====================
def compute_or_levels(df: pd.DataFrame, orb_minutes: int) -> pd.DataFrame:
    """Compute Opening Range High/Low for each session."""
    def do(g: pd.DataFrame) -> pd.Series:
        open_window = g[(g["MinutesSinceOpen"] >= 0) & (g["MinutesSinceOpen"] < orb_minutes)]
        if open_window.empty:
            return pd.Series({"ORH": np.nan, "ORL": np.nan})
        return pd.Series({"ORH": open_window["High"].max(), "ORL": open_window["Low"].min()})
    lev = df.groupby(["Ticker","SessionDate"]).apply(do).reset_index()
    return lev

def add_pdh_pdl(daily_long: pd.DataFrame) -> pd.DataFrame:
    """Get previous day high/low per ticker."""
    daily_long = daily_long.sort_index()
    prev = daily_long.groupby("Ticker").apply(lambda g: g.tail(2)).reset_index()
    # The second-to-last row is previous day
    def prev_day_hl(g):
        if len(g) < 2:
            return pd.Series({"PDH": np.nan, "PDL": np.nan})
        prev_row = g.iloc[-2]
        return pd.Series({"PDH": prev_row["High"], "PDL": prev_row["Low"]})
    out = prev.groupby("Ticker").apply(prev_day_hl).reset_index()
    return out

# ==================== Sidebar Controls ====================
st.sidebar.title("⚙️ Intraday Screener — Pro Settings")

universe_source = st.sidebar.selectbox(
    "Universe",
    ["NSE (auto)", "Upload CSV (SYMBOL or YF_TICKER)", "Manual tickers"],
)

interval = st.sidebar.selectbox(
    "Interval",
    ["5m","15m","30m","1h","1d"], index=0,
    help="For intraday, 5m/15m/30m/1h. 1d for EOD checks."
)

history_days_req = st.sidebar.slider("History (days)", 5, 180, 30, step=5)
history_days = clamp_period_days(history_days_req, interval)

# Auto-refresh during market hours
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Auto-Refresh")
auto_refresh = st.sidebar.checkbox("Enable (market hours)", value=True)
refresh_secs = st.sidebar.slider("Refresh every (secs)", 60, 900, 300, step=30)

# Liquidity prefilter (daily)
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Liquidity Prefilter (daily)")
min_price = st.sidebar.number_input("Min Price (₹)", min_value=1.0, value=50.0, step=1.0)
max_price = st.sidebar.number_input("Max Price (₹)", min_value=10.0, value=5000.0, step=10.0)
min_avg_turn_cr = st.sidebar.number_input("Min 20‑D Avg Turnover (₹ Cr)", min_value=0.0, value=5.0, step=0.5)

# Pro Mode gates
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Pro Filters")
pro_mode = st.sidebar.checkbox("Pro Mode (VWAP, ORB, session windows)", value=True)
skip_open_mins = st.sidebar.number_input("Skip first N mins", min_value=0, max_value=60, value=10, step=5)
skip_close_mins = st.sidebar.number_input("Skip last N mins", min_value=0, max_value=60, value=10, step=5)
orb_minutes = st.sidebar.slider("Opening Range minutes", 5, 60, 30, step=5)
require_or_breakout = st.sidebar.checkbox("Require ORB breakout (for signals)", value=False)
require_vwap_align = st.sidebar.checkbox("Require VWAP alignment (BUY Close>VWAP; SELL Close<VWAP)", value=True)

# Intraday turnover filter
min_intra_turn_cr = st.sidebar.number_input("Min avg intraday turnover (₹ Cr) — last 60 bars", min_value=0.0, value=1.0, step=0.5)

# HTF confirmation
st.sidebar.markdown("---")
st.sidebar.subheader("🕐 1h Trend Confirmation")
htf_confirm = st.sidebar.checkbox("Confirm with 1h SMA trend", value=True)
htf_check_top_k = st.sidebar.slider("Confirm top K candidates", 50, 400, 200, step=50)

# Position sizing
st.sidebar.markdown("---")
st.sidebar.subheader("💰 Position Size Helper (hypothetical)")
capital = st.sidebar.number_input("Capital (₹)", min_value=10000.0, value=200000.0, step=5000.0)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 2.0, 0.5, step=0.1)
fees_bps = st.sidebar.slider("Fees + Brokerage + Slippage (bps one-way)", 0, 50, 10, step=1)
atr_mult = st.sidebar.slider("ATR stop multiple (x)", 0.5, 3.0, 1.0, step=0.1)

st.sidebar.markdown("---")
run = st.sidebar.button("🔁 Run Intraday Scan")

# ==================== Universe Build ====================
if universe_source == "NSE (auto)":
    uni_df = fetch_nse_universe()
    yf_tickers = uni_df["YF_TICKER"].tolist()
elif universe_source == "Upload CSV (SYMBOL or YF_TICKER)":
    file = st.sidebar.file_uploader("Upload CSV with SYMBOL or YF_TICKER", type=["csv"])
    yf_tickers = []
    if file:
        tmp = pd.read_csv(file)
        if "YF_TICKER" in tmp.columns:
            yf_tickers = tmp["YF_TICKER"].dropna().astype(str).tolist()
        elif "SYMBOL" in tmp.columns:
            yf_tickers = tmp["SYMBOL"].dropna().astype(str).apply(lambda s: _to_yahoo_symbol(s, "NS")).tolist()
        else:
            st.error("CSV must contain SYMBOL or YF_TICKER column.")
else:
    manual = st.sidebar.text_area("Enter comma-separated Yahoo tickers (e.g., RELIANCE.NS, TCS.NS)")
    yf_tickers = [t.strip() for t in manual.split(",") if t.strip()]

# ==================== Header ====================
st.title("⚡ India Intraday Screener — Pro")
st.caption("Intraday ranking with liquidity prefilter, VWAP & ORB alignment, optional 1h confirmation. Data: Yahoo Finance (may be delayed). Not advice.")

# Auto-refresh only during market hours
now_ist = datetime.now(IST)
mins_today = now_ist.hour*60 + now_ist.minute
if auto_refresh and (MARKET_OPEN_MIN <= mins_today <= MARKET_CLOSE_MIN):
    st_autorefresh(interval=int(refresh_secs*1000), key="auto_refresh")

if not run:
    st.info("Configure settings and click **Run Intraday Scan**.")
    st.stop()

if not yf_tickers:
    st.warning("Universe is empty. Provide tickers from the sidebar.")
    st.stop()

# ==================== Prefilter by Liquidity (Daily) ====================
with st.spinner("Prefiltering by daily liquidity…"):
    daily = yf_download_batch(yf_tickers, period="90d", interval="1d")
if daily.empty:
    st.error("Could not fetch daily data for prefilter. Try smaller universe.")
    st.stop()

daily_long = _ensure_long(daily)

def calc_daily_liq(g):
    g = g.sort_index()
    g["Turnover"] = g["Close"] * g["Volume"]  # in ₹
    avg_turn = g["Turnover"].tail(20).mean() if len(g) >= 1 else np.nan
    last_close = g["Close"].iloc[-1]
    return pd.Series({"AvgTurnCr": (avg_turn or 0.0)/1e7, "LastClose": last_close})

liq = daily_long.groupby("Ticker").apply(calc_daily_liq).reset_index()
liq = liq[(liq["LastClose"]>=min_price) & (liq["LastClose"]<=max_price) & (liq["AvgTurnCr"]>=min_avg_turn_cr)]
liquid_tickers = liq["Ticker"].tolist()
st.caption(f"Liquidity filter kept {len(liquid_tickers)} / {len(yf_tickers)} tickers (Min ₹{min_price}-{max_price}, AvgTurn ≥ {min_avg_turn_cr} Cr)")

if len(liquid_tickers) == 0:
    st.error("No tickers passed the liquidity filter. Relax filters and retry.")
    st.stop()

# ==================== Intraday Download ====================
period = f"{history_days}d"
with st.spinner(f"Downloading intraday ({interval}, {period}) for {len(liquid_tickers)} tickers…"):
    intra = yf_download_batch(liquid_tickers, period=period, interval=interval)

if intra.empty:
    st.error("Intraday download returned empty. Try fewer tickers, shorter history, or different interval.")
    st.stop()

# ==================== Indicator Computation ====================
with st.spinner("Computing indicators, VWAP, ORB & ATR…"):
    long_df = _ensure_long(intra)
    full = compute_indicators_intraday(long_df)

# Session window trimming
if pro_mode:
    full = full[(full["MinutesSinceOpen"] >= skip_open_mins) & (full["MinutesToClose"] >= skip_close_mins)]

# Opening Range levels
or_levels = compute_or_levels(full, orb_minutes)
full = full.merge(or_levels, on=["Ticker","SessionDate"], how="left")

# ==================== Previous Day HL (optional gating) ====================
with st.spinner("Fetching previous day high/low for gating…"):
    prev_daily = yf_download_batch(liquid_tickers, period="15d", interval="1d")
prev_long = _ensure_long(prev_daily) if not prev_daily.empty else pd.DataFrame()
pdh_pdl = add_pdh_pdl(prev_long) if not prev_long.empty else pd.DataFrame({"Ticker": liquid_tickers, "PDH": np.nan, "PDL": np.nan})

# Keep most recent bar per ticker
latest = full.groupby("Ticker").tail(1).reset_index()
latest = latest.merge(pdh_pdl, on="Ticker", how="left")

# ==================== Intraday turnover filter (post-download) ====================
post_turn = full.groupby("Ticker").apply(lambda g: (g["Close"]*g["Volume"]).tail(60).mean()/1e7).reset_index()
post_turn.columns = ["Ticker","AvgIntraTurnCr"]
latest = latest.merge(post_turn, on="Ticker", how="left")
latest = latest[latest["AvgIntraTurnCr"] >= min_intra_turn_cr]

if latest.empty:
    st.error("No tickers passed the intraday turnover filter. Relax filters and retry.")
    st.stop()

# ==================== Signals & Scoring ====================
latest["BaseSignal"] = latest.apply(base_signal_rule, axis=1)
latest["Score"] = latest.apply(intraday_score, axis=1)

# VWAP alignment gate
if require_vwap_align:
    def eligible_vwap(row):
        if row["BaseSignal"] == "BUY":
            return pd.notna(row["VWAP"]) and (row["Close"] > row["VWAP"]) and (row.get("VWAP_Slope3",0) >= 0)
        if row["BaseSignal"] == "SELL":
            return pd.notna(row["VWAP"]) and (row["Close"] < row["VWAP"]) and (row.get("VWAP_Slope3",0) <= 0)
        return False
    latest["EligibleVWAP"] = latest.apply(eligible_vwap, axis=1)
else:
    latest["EligibleVWAP"] = latest["BaseSignal"].isin(["BUY","SELL"])

# ORB gate
if require_or_breakout:
    def eligible_orb(row):
        orh, orl, close = row.get("ORH"), row.get("ORL"), row.get("Close")
        if row["BaseSignal"] == "BUY" and pd.notna(orh):
            return close > orh
        if row["BaseSignal"] == "SELL" and pd.notna(orl):
            return close < orl
        return False
    latest["EligibleORB"] = latest.apply(eligible_orb, axis=1)
else:
    latest["EligibleORB"] = True

# 1h HTF confirmation
if htf_confirm:
    candidates = latest[latest["EligibleVWAP"] & latest["EligibleORB"]].copy()
    candidates = candidates.reindex(candidates["Score"].abs().sort_values(ascending=False).head(htf_check_top_k).index)
    htf_tickers = candidates["Ticker"].tolist()
    with st.spinner(f"Fetching 1h confirmation for top {len(htf_tickers)} candidates…"):
        htf = yf_download_batch(htf_tickers, period="60d", interval="1h")
    if not htf.empty:
        htf_long = _ensure_long(htf)
        def htf_trend(g):
            g = g.sort_index()
            g["SMA20"] = g["Close"].rolling(20, min_periods=20).mean()
            g["SMA50"] = g["Close"].rolling(50, min_periods=50).mean()
            last = g.tail(1)
            if last.empty:
                return pd.Series({"HTF_Trend": "NA"})
            s20 = last["SMA20"].values[0]
            s50 = last["SMA50"].values[0]
            if pd.notna(s20) and pd.notna(s50):
                if s20 > s50: return pd.Series({"HTF_Trend": "UP"})
                if s20 < s50: return pd.Series({"HTF_Trend": "DOWN"})
            return pd.Series({"HTF_Trend": "FLAT"})
        conf = htf_long.groupby("Ticker").apply(htf_trend).reset_index()
        latest = latest.merge(conf, on="Ticker", how="left")
        def confirm_mask(row):
            if row["BaseSignal"] == "BUY":
                return row.get("HTF_Trend","NA") == "UP"
            if row["BaseSignal"] == "SELL":
                return row.get("HTF_Trend","NA") == "DOWN"
            return False
        latest["HTF_OK"] = latest.apply(confirm_mask, axis=1)
    else:
        latest["HTF_Trend"] = "NA"
        latest["HTF_OK"] = False
else:
    latest["HTF_Trend"] = "NA"
    latest["HTF_OK"] = True

# Final signal
latest["Signal"] = "HOLD"
mask_buy = (latest["BaseSignal"]=="BUY") & latest["EligibleVWAP"] & latest["EligibleORB"] & latest["HTF_OK"]
mask_sell = (latest["BaseSignal"]=="SELL") & latest["EligibleVWAP"] & latest["EligibleORB"] & latest["HTF_OK"]
latest.loc[mask_buy, "Signal"] = "BUY"
latest.loc[mask_sell, "Signal"] = "SELL"

# ==================== Rank & Display ====================
disp_cols = [
    "Ticker","Close","SMA20","SMA50","RSI14","MACD","MACD_Signal","MACD_Hist",
    "VWAP","ORH","ORL","Ret_3","ATR14","Score","AvgIntraTurnCr","BaseSignal","Signal","HTF_Trend"
]
view = latest[disp_cols].copy()

buys = view[view["Signal"]=="BUY"].sort_values("Score", ascending=False).head( st.session_state.get('top_n', 5) )
sells = view[view["Signal"]=="SELL"].sort_values("Score", ascending=True).head( st.session_state.get('top_n', 5) )

col1, col2 = st.columns(2)
with col1:
    st.subheader(f"Top BUY (intraday ranked)")
    st.dataframe(buys, use_container_width=True)
with col2:
    st.subheader(f"Top SELL (intraday ranked)")
    st.dataframe(sells, use_container_width=True)

# Download
csv_data = view.sort_values("Ticker").to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Latest Signals (CSV)",
    data=csv_data,
    file_name=f"intraday_signals_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
)

# ==================== Position Size Helper ====================
st.markdown("---")
st.subheader("Position Size Helper (Hypothetical)")

def suggest_position(entry, atr, atr_mult, capital, risk_pct, fees_bps):
    """Risk-based size using ATR multiple as stop distance."""
    if not (np.isfinite(entry) and np.isfinite(atr) and entry>0 and atr>0):
        return 0, 0.0, 0.0
    stop_distance = atr_mult * atr
    fees = (fees_bps/10000.0) * entry  # one-way
    total_cost = stop_distance + (2*fees)
    if total_cost <= 0:
        return 0, 0.0, 0.0
    risk_amt = capital * (risk_pct/100.0)
    qty = int(risk_amt / total_cost)
    est_loss = qty * total_cost
    stop_price_long = entry - stop_distance
    return max(qty,0), est_loss, stop_price_long

st.caption("Pick a ticker; stop distance = ATR×multiple. This is **not advice**.")
tick_for_size = st.selectbox("Ticker", options=view["Ticker"].tolist(), index=0)
sel = view[view["Ticker"]==tick_for_size].iloc[0]
entry_price = float(sel["Close"])
atr_val = float(sel["ATR14"]) if pd.notna(sel["ATR14"]) else entry_price*0.01
qty, est_loss, stop_price = suggest_position(entry_price, atr_val, atr_mult, capital, risk_pct, fees_bps)
c1,c2,c3 = st.columns(3)
with c1: st.metric("Entry (₹)", f"{entry_price:.2f}")
with c2: st.metric("Suggested Qty", f"{qty}")
with c3: st.metric("Est. Max Loss (₹)", f"{est_loss:.0f}")
st.caption(f"ATR(14): {atr_val:.2f} | Suggested Stop (Long): ₹{stop_price:.2f}")

# ==================== Charting ====================
st.markdown("---")
st.subheader("Charts (Close, SMA20, SMA50, VWAP, OR Levels)")
chart_ticker = st.selectbox("Select for chart", options=view["Ticker"].tolist(), index=0)
hist = full[full["Ticker"]==chart_ticker].copy()
if hist.empty:
    st.warning("No data for selected ticker.")
else:
    fig, ax = plt.subplots(figsize=(10,4), dpi=140)
    ax.plot(hist.index, hist["Close"], label="Close", color="#1f77b4", linewidth=1.4)
    ax.plot(hist.index, hist["SMA20"], label="SMA20", color="#2ca02c", linewidth=1.1)
    ax.plot(hist.index, hist["SMA50"], label="SMA50", color="#ff7f0e", linewidth=1.1)
    if "VWAP" in hist.columns:
        ax.plot(hist.index, hist["VWAP"], label="VWAP (session)", color="#9467bd", linewidth=1.0, alpha=0.8)
    # OR levels as horizontal lines for latest session
    latest_session = hist["SessionDate"].iloc[-1] if len(hist)>0 else TODAY_IST
    hh = hist[hist["SessionDate"]==latest_session]["ORH"].dropna()
    ll = hist[hist["SessionDate"]==latest_session]["ORL"].dropna()
    if not hh.empty:
        ax.axhline(hh.values[-1], color="#d62728", linestyle="--", alpha=0.7, label="ORH")
    if not ll.empty:
        ax.axhline(ll.values[-1], color="#17becf", linestyle="--", alpha=0.7, label="ORL")
    ax.set_title(f"{chart_ticker} — {interval} (IST)")
    ax.grid(True, alpha=0.2)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# ==================== Footer ====================
st.markdown("""
**Disclaimers:** Educational use only; market data may be delayed/inaccurate. Backtest thoroughly, include realistic fees/slippage, and use strict risk management. Trading involves risk of loss.
""")
