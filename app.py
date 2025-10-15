# app.py (Optimized FastScan)
# Purpose: A faster, lighter intraday screener for NSE with sensible defaults and
# optional Pro features you can turn on per need. Designed to scale better and
# reduce compute by (1) aggressive universe prefiltering, (2) short history,
# (3) minimal indicator windows, and (4) optional heavy features.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
from datetime import datetime
from zoneinfo import ZoneInfo
# NOTE: lazy import matplotlib inside the chart block to improve startup stability on cloud
# import matplotlib.pyplot as plt

# ==================== Page Config ====================
st.set_page_config(page_title="India Intraday Screener — FastScan", page_icon="⚡", layout="wide")

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

# ==================== Constants ======================
NSE_EQUITY_LIST_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"  # Reliable archive host
MARKET_OPEN_MIN = 9*60 + 15
MARKET_CLOSE_MIN = 15*60 + 30

# ==================== Utilities ======================
def _to_yahoo_symbol(symbol: str, exchange: str = "NS") -> str:
    s = symbol.strip().upper()
    return f"{s}.NS" if exchange == "NS" else (f"{s}.BO" if exchange == "BO" else s)

@st.cache_data(ttl=60*30)
def fetch_nse_universe() -> pd.DataFrame:
    """Fetch NSE Equity symbols from archive CSV. Returns SYMBOL, YF_TICKER."""
    try:
        headers = {"User-Agent":"Mozilla/5.0"}
        resp = requests.get(NSE_EQUITY_LIST_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if "SYMBOL" not in df.columns:
            raise ValueError("SYMBOL column not found")
        df["SYMBOL"] = df["SYMBOL"].astype(str)
        df["YF_TICKER"] = df["SYMBOL"].apply(lambda s: _to_yahoo_symbol(s, "NS"))
        df = df.drop_duplicates(subset=["YF_TICKER"]).reset_index(drop=True)
        return df[["SYMBOL","YF_TICKER"]]
    except Exception:
        # Fallback universe
        return pd.DataFrame({
            "SYMBOL":["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","BHARTIARTL","KOTAKBANK","LT","ITC"],
            "YF_TICKER":["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS","LT.NS","ITC.NS"]
        })

# Yahoo intraday safe periods (conservative)
def max_allowed_period(interval: str) -> str:
    i = interval.lower()
    if i == "1m":
        return "7d"     # 1m often limited to ~7 days per request
    if i in {"2m","5m","15m","30m"}:
        return "30d"    # practical cap around ~30 days
    if i in {"60m","90m","1h"}:
        return "180d"   # 1h can go longer; keep capped
    return "365d"      # daily

def clamp_period_days(request_days: int, interval: str) -> int:
    mx_days = int(max_allowed_period(interval).replace("d",""))
    return min(request_days, mx_days)

@st.cache_data(ttl=60*15)
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

# ==================== Light Indicator Engine ======================
def _ensure_long(df_multi: pd.DataFrame) -> pd.DataFrame:
    df_multi = df_multi.swaplevel(axis=1).sort_index(axis=1)
    frames = []
    for t in df_multi.columns.get_level_values(0).unique():
        sub = df_multi[t].copy()
        sub["Ticker"] = t
        frames.append(sub)
    long = pd.concat(frames, axis=0)
    return long.dropna(subset=["Close"], how="any")

def add_ist_cols(g: pd.DataFrame) -> pd.DataFrame:
    idx = g.index
    if idx.tz is None:
        idx = idx.tz_localize(UTC)
    ist_idx = idx.tz_convert(IST)
    g = g.copy(); g.index = ist_idx
    g["SessionDate"] = g.index.date
    g["MinutesSinceOpen"] = ((g.index.hour*60 + g.index.minute) - MARKET_OPEN_MIN).astype(int)
    g["MinutesToClose"] = MARKET_CLOSE_MIN - (g.index.hour*60 + g.index.minute)
    return g

# Compute minimal indicators only on the last N bars per ticker
BAR_NEED = 120  # enough for SMA50 + MACD stabilization

def compute_min_indicators(long_df: pd.DataFrame) -> pd.DataFrame:
    def calc_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_index()
        g = add_ist_cols(g)
        g = g.tail(BAR_NEED)  # reduce compute
        g["SMA20"] = g["Close"].rolling(20, min_periods=20).mean()
        g["SMA50"] = g["Close"].rolling(50, min_periods=50).mean()
        # RSI14 (Wilder approx)
        d = g["Close"].diff(); up = np.where(d>0, d, 0.0); dn = np.where(d<0, -d, 0.0)
        ru = pd.Series(up, index=g.index).rolling(14).mean(); rd = pd.Series(dn, index=g.index).rolling(14).mean()
        rs = ru / (rd + 1e-9); g["RSI14"] = 100 - (100 / (1 + rs))
        # MACD
        ema12 = g["Close"].ewm(span=12, adjust=False).mean()
        ema26 = g["Close"].ewm(span=26, adjust=False).mean()
        g["MACD"] = ema12 - ema26
        g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
        g["MACD_Hist"] = g["MACD"] - g["MACD_Signal"]
        # Session VWAP (fast)
        tp = (g["High"] + g["Low"] + g["Close"]) / 3.0
        g["VWAP"] = g.groupby("SessionDate").apply(
            lambda df: ( (tp.loc[df.index]*df["Volume"]).cumsum() / (df["Volume"].replace(0,np.nan).cumsum()) ).values
        ).reset_index(level=0, drop=True)
        return g
    return long_df.groupby("Ticker", group_keys=False).apply(calc_group)

# ==================== Signal & Score ======================
def base_signal_rule(row: pd.Series) -> str:
    s20, s50, rsi = row.get("SMA20"), row.get("SMA50"), row.get("RSI14")
    macd, sig = row.get("MACD"), row.get("MACD_Signal")
    if pd.notna(s20) and pd.notna(s50) and pd.notna(rsi) and pd.notna(macd) and pd.notna(sig):
        buy = (s20 > s50) and (40 <= rsi <= 70) and (macd > sig)
        sell = (s20 < s50) and ((rsi > 70) or (macd < sig))
        if buy: return "BUY"
        if sell: return "SELL"
    return "HOLD"

def fast_score(row: pd.Series) -> float:
    close = row.get("Close", np.nan)
    s20, s50 = row.get("SMA20", np.nan), row.get("SMA50", np.nan)
    rsi = row.get("RSI14", np.nan)
    macd_hist = row.get("MACD_Hist", np.nan)
    vwap = row.get("VWAP", np.nan)
    sma_comp = 0.0 if not np.isfinite(close*s20*s50) else (s20 - s50) / max(close, 1e-9)
    rsi_comp = 0.0 if not np.isfinite(rsi) else np.clip((rsi - 50.0)/20.0, -1.2, 1.2)
    macd_comp = 0.0 if not np.isfinite(macd_hist) else np.clip(macd_hist, -3.0, 3.0)
    vwap_comp = 0.0 if not (np.isfinite(vwap) and np.isfinite(close)) else np.clip((close - vwap)/max(close,1e-9), -0.02, 0.02)
    w_sma, w_rsi, w_macd, w_vwap = 0.35, 0.25, 0.30, 0.10
    return float(w_sma*sma_comp + w_rsi*rsi_comp + w_macd*macd_comp + w_vwap*vwap_comp)

# ==================== Sidebar ======================
st.sidebar.title("⚙️ FastScan Settings")

universe_source = st.sidebar.selectbox("Universe", ["NSE (auto)", "Upload CSV (SYMBOL or YF_TICKER)", "Manual tickers"])
interval = st.sidebar.selectbox("Interval", ["5m","15m","30m","1h"], index=0)
history_days_req = st.sidebar.slider("History (days)", 5, 60, 10, step=5)
history_days = clamp_period_days(history_days_req, interval)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Daily Liquidity Prefilter")
min_price = st.sidebar.number_input("Min Price (₹)", min_value=1.0, value=50.0, step=1.0)
max_price = st.sidebar.number_input("Max Price (₹)", min_value=10.0, value=5000.0, step=10.0)
min_avg_turn_cr = st.sidebar.number_input("Min 20‑D Avg Turnover (₹ Cr)", min_value=0.0, value=10.0, step=0.5)
max_universe = st.sidebar.slider("Max universe size (top by turnover)", 20, 1000, 200, step=20)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Modes & Gates")
quick_scan = st.sidebar.checkbox("QuickScan (fast) — recommended", value=True)
require_vwap_align = st.sidebar.checkbox("Gate: VWAP alignment (BUY Close>VWAP; SELL Close<VWAP)", value=True)
# Pro options (off by default)
pro_orb = st.sidebar.checkbox("Gate: Opening Range Breakout (ORB)", value=False)
orb_minutes = st.sidebar.slider("ORB minutes", 5, 60, 30, step=5)
htf_confirm = st.sidebar.checkbox("Gate: 1h SMA trend confirm", value=False)

top_n = st.sidebar.slider("Top N results", 5, 50, 10, step=5)
run = st.sidebar.button("🔁 Run Scan")

# ==================== Universe ======================
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
st.title("⚡ India Intraday Screener — FastScan")
st.caption("Lightweight intraday screener: daily prefilter → short history → minimal indicators. Data via Yahoo Finance.")

# Early exit ensures health check passes quickly on cloud
if not run:
    st.info("Configure and click **Run Scan**.")
    st.stop()

# Guard for empty universe to avoid unnecessary fetches
if not yf_tickers:
    st.warning("Universe is empty.")
    st.stop()

# ==================== Step 1: Daily Prefilter ====================
with st.spinner("Prefiltering daily liquidity…"):
    daily = yf_download_batch(yf_tickers, period="60d", interval="1d")
if daily.empty:
    st.error("Daily data fetch failed.")
    st.stop()

daily_long = _ensure_long(daily)

def calc_daily_liq(g):
    g = g.sort_index()
    g["Turnover"] = g["Close"] * g["Volume"]
    avg_turn = g["Turnover"].tail(20).mean() if len(g) >= 1 else np.nan
    last_close = g["Close"].iloc[-1]
    return pd.Series({"AvgTurnCr": (avg_turn or 0.0)/1e7, "LastClose": last_close})

liq = daily_long.groupby("Ticker").apply(calc_daily_liq).reset_index()
liq = liq[(liq["LastClose"]>=min_price) & (liq["LastClose"]<=max_price) & (liq["AvgTurnCr"]>=min_avg_turn_cr)]
liq = liq.sort_values("AvgTurnCr", ascending=False).head(max_universe)
keep_tickers = liq["Ticker"].tolist()
st.caption(f"Kept {len(keep_tickers)} tickers after liquidity bounds and top-by-turnover cut.")
if len(keep_tickers) == 0:
    st.error("No tickers passed liquidity filters.")
    st.stop()

# ==================== Step 2: Intraday Download (short) ====================
period = f"{history_days}d"
with st.spinner(f"Downloading intraday ({interval}, {period}) for {len(keep_tickers)} tickers…"):
    intra = yf_download_batch(keep_tickers, period=period, interval=interval)
if intra.empty:
    st.error("Intraday fetch failed. Try shorter history / fewer tickers.")
    st.stop()

# ==================== Step 3: Indicators (minimal) ====================
with st.spinner("Computing minimal indicators…"):
    long_df = _ensure_long(intra)
    full = compute_min_indicators(long_df)

# ==================== Step 4: Latest row, Signals & Score ====================
latest = full.groupby("Ticker").tail(1).reset_index()
latest["BaseSignal"] = latest.apply(base_signal_rule, axis=1)
latest["Score"] = latest.apply(fast_score, axis=1)

# VWAP gate (simple, no slope)
if require_vwap_align:
    def vwap_gate(row):
        if row["BaseSignal"] == "BUY":
            return pd.notna(row["VWAP"]) and (row["Close"] > row["VWAP"])
        if row["BaseSignal"] == "SELL":
            return pd.notna(row["VWAP"]) and (row["Close"] < row["VWAP"])
        return False
    latest["VWAP_OK"] = latest.apply(vwap_gate, axis=1)
else:
    latest["VWAP_OK"] = latest["BaseSignal"].isin(["BUY","SELL"])

# ORB gate (optional; computed quickly from minimal bars)
if pro_orb:
    def or_levels_for_latest(g):
        g = g.sort_index()
        sess = g["SessionDate"].iloc[-1] if len(g)>0 else None
        w = g[g["SessionDate"]==sess]
        open_window = w[(w["MinutesSinceOpen"]>=0) & (w["MinutesSinceOpen"]<orb_minutes)]
        if open_window.empty:
            return pd.Series({"ORH": np.nan, "ORL": np.nan})
        return pd.Series({"ORH": open_window["High"].max(), "ORL": open_window["Low"].min()})
    or_df = full.groupby("Ticker").apply(or_levels_for_latest).reset_index()
    latest = latest.merge(or_df, on="Ticker", how="left")
    def orb_gate(row):
        if row["BaseSignal"] == "BUY" and pd.notna(row.get("ORH", np.nan)):
            return row["Close"] > row["ORH"]
        if row["BaseSignal"] == "SELL" and pd.notna(row.get("ORL", np.nan)):
            return row["Close"] < row["ORL"]
        return False
    latest["ORB_OK"] = latest.apply(orb_gate, axis=1)
else:
    latest["ORH"] = np.nan; latest["ORL"] = np.nan; latest["ORB_OK"] = True

# 1h confirmation (optional)
if htf_confirm:
    cands = latest[latest["VWAP_OK"] & latest["ORB_OK"]].copy()
    cands = cands.reindex(cands["Score"].abs().sort_values(ascending=False).head(200).index)
    with st.spinner("1h confirmation for top candidates…"):
        htf = yf_download_batch(cands["Ticker"].tolist(), period="60d", interval="1h")
    if not htf.empty:
        htf_long = _ensure_long(htf)
        def htf_bias(g):
            g = g.sort_index()
            g["SMA20"] = g["Close"].rolling(20, min_periods=20).mean()
            g["SMA50"] = g["Close"].rolling(50, min_periods=50).mean()
            s20 = g["SMA20"].iloc[-1] if len(g)>=20 else np.nan
            s50 = g["SMA50"].iloc[-1] if len(g)>=50 else np.nan
            if pd.notna(s20) and pd.notna(s50):
                return pd.Series({"HTF_Trend": "UP" if s20>s50 else ("DOWN" if s20<s50 else "FLAT")})
            return pd.Series({"HTF_Trend": "NA"})
        conf = htf_long.groupby("Ticker").apply(htf_bias).reset_index()
        latest = latest.merge(conf, on="Ticker", how="left")
        def htf_ok(row):
            if row["BaseSignal"] == "BUY":  return row.get("HTF_Trend","NA") == "UP"
            if row["BaseSignal"] == "SELL": return row.get("HTF_Trend","NA") == "DOWN"
            return False
        latest["HTF_OK"] = latest.apply(htf_ok, axis=1)
    else:
        latest["HTF_Trend"] = "NA"; latest["HTF_OK"] = False
else:
    latest["HTF_Trend"] = "NA"; latest["HTF_OK"] = True

# Final decision
latest["Signal"] = "HOLD"
mask_buy = (latest["BaseSignal"]=="BUY") & latest["VWAP_OK"] & latest["ORB_OK"] & latest["HTF_OK"]
mask_sell = (latest["BaseSignal"]=="SELL") & latest["VWAP_OK"] & latest["ORB_OK"] & latest["HTF_OK"]
latest.loc[mask_buy, "Signal"] = "BUY"
latest.loc[mask_sell, "Signal"] = "SELL"

# ==================== Display ======================
disp_cols = ["Ticker","Close","SMA20","SMA50","RSI14","MACD","MACD_Signal","MACD_Hist","VWAP","ORH","ORL","Score","BaseSignal","Signal","HTF_Trend"]
view = latest[disp_cols].copy()

buys = view[view["Signal"]=="BUY"].sort_values("Score", ascending=False).head(top_n)
sells = view[view["Signal"]=="SELL"].sort_values("Score", ascending=True).head(top_n)

c1,c2 = st.columns(2)
with c1:
    st.subheader(f"Top {len(buys)} BUY")
    st.dataframe(buys, use_container_width=True)
with c2:
    st.subheader(f"Top {len(sells)} SELL")
    st.dataframe(sells, use_container_width=True)

# Download CSV
csv_data = view.sort_values("Ticker").to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Latest Signals (CSV)",
    data=csv_data,
    file_name=f"fastscan_signals_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)

# Chart (on-demand only; prevents heavy render)
st.markdown("---")
st.subheader("On-demand chart (Close + SMA20/50 + VWAP + OR Levels)")
chart_ticker = st.selectbox("Select ticker", options=view["Ticker"].tolist())
if chart_ticker:
    # Lazy import here to avoid startup failures if matplotlib wheels are unavailable
    import matplotlib.pyplot as plt
    hist = full[full["Ticker"]==chart_ticker].copy()
    if hist.empty:
        st.warning("No data for selected ticker.")
    else:
        fig, ax = plt.subplots(figsize=(10,4), dpi=140)
        ax.plot(hist.index, hist["Close"], label="Close", color="#1f77b4", linewidth=1.3)
        ax.plot(hist.index, hist["SMA20"], label="SMA20", color="#2ca02c", linewidth=1.0)
        ax.plot(hist.index, hist["SMA50"], label="SMA50", color="#ff7f0e", linewidth=1.0)
        ax.plot(hist.index, hist["VWAP"], label="VWAP", color="#9467bd", linewidth=0.9, alpha=0.8)
        if "ORH" in view.columns and not pd.isna(view[view["Ticker"]==chart_ticker]["ORH"]).all():
            orh = view[view["Ticker"]==chart_ticker]["ORH"].iloc[0]
            if pd.notna(orh): ax.axhline(orh, color="#d62728", linestyle="--", alpha=0.6, label="ORH")
        if "ORL" in view.columns and not pd.isna(view[view["Ticker"]==chart_ticker]["ORL"]).all():
            orl = view[view["Ticker"]==chart_ticker]["ORL"].iloc[0]
            if pd.notna(orl): ax.axhline(orl, color="#17becf", linestyle="--", alpha=0.6, label="ORL")
        ax.set_title(f"{chart_ticker} — {interval} (IST)")
        ax.grid(True, alpha=0.2)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

st.markdown("""
**Notes:**
- Intraday history is clamped conservatively to avoid empty Yahoo responses (e.g., ~7d for 1m; ~30d for 5m; ~180d for 1h).
- Use higher liquidity thresholds and smaller universe size for fastest scans.
- This tool is for educational purposes only; not investment advice.
""")
