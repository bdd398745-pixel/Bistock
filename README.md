# ⚡ India Intraday Screener — Pro (SMA/RSI/MACD + VWAP + ORB)

A production-ready Streamlit app to screen **all NSE equities** intraday with **SMA/RSI/MACD**, **VWAP alignment**, **Opening Range Breakout (ORB)** filters, **1h trend confirmation**, and a **risk-based position size helper**.

> **Disclaimer:** For **educational and informational** use only. Not investment advice. Data from Yahoo Finance may be delayed or inaccurate.

---

## ✨ Highlights

- **Universe:** NSE auto symbols (via NSE archive) or custom CSV/manual.
- **Indicators:** SMA20/SMA50, RSI(14), MACD(12,26,9), VWAP (session), ATR(14).
- **Signals:**
  - **BUY:** SMA20 > SMA50, 40 ≤ RSI ≤ 70, MACD > Signal (+ optional VWAP & ORB gates)
  - **SELL:** SMA20 < SMA50, (RSI > 70 or MACD < Signal) (+ optional VWAP & ORB gates)
  - **HOLD:** otherwise
- **Ranking:** Intraday score emphasizing **momentum (Ret_3)**, **MACD hist strength**, **SMA distance**, and **VWAP distance/slope**.
- **Filters:**
  - **Daily liquidity prefilter** by price bounds and 20‑day avg **turnover (₹ Cr)**.
  - **Intraday turnover filter** (last 60 bars).
  - **Session hygiene:** skip first/last N minutes.
  - **ORB requirement:** configurable opening range breakout gate.
  - **HTF confirm:** optional **1h SMA20/50** trend alignment.
- **Charts:** Close + SMA20/50 + VWAP, with ORH/ORL overlays.
- **Tools:** CSV export of latest signals, ATR‑based position size helper.
- **Performance:** Chunked yfinance downloads, multi-threading, caching (TTL), prefiltering.

---

## 🚀 Quickstart (Local)

```bash
# 1) Clone your repo & create a virtual env
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

### Sidebar Tips
- Use `Interval=5m` with `History ≤ 30d` (Yahoo intraday limits). 1m is limited to ~7 days per request. citeturn3search14turn3search13
- NSE symbol list is fetched from the official **EQUITY_L.csv** archive. citeturn3search11turn3search7
- Auto-refresh uses the **streamlit-autorefresh** component; enable during market hours. citeturn3search1turn3search2

---

## 📦 Deploy on Streamlit Community Cloud

1. Create a GitHub repo (e.g., `india-intraday-screener-pro`).
2. Add these files:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml` *(optional aesthetics)*
   - `data/watchlist_example.csv` *(optional)*
3. Go to **https://share.streamlit.io** → **New app** → select repo/branch/file.
4. **Settings:** Python 3.11; enable persistent caching if available.
5. Deploy. The app is responsive for mobile.

---

## 🧠 How the app keeps it fast & pragmatic

- **Prefilter first:** A quick **daily** pass eliminates illiquid names before heavy intraday fetch.
- **Chunking:** Tick downloads in batches to avoid throttling and failures.
- **Caching:** `st.cache_data(ttl=…)` reduces repeat cost within TTL windows.
- **Gating:** Optional **VWAP/ORB** gates reduce false positives; **1h confirm** nudges trades in direction of the broader intraday trend.

---

## 🔌 File List

- `app.py`: Main Streamlit app with all features.
- `requirements.txt`: Dependencies (incl. `streamlit-autorefresh`).
- `.streamlit/config.toml`: Theme/layout.
- `data/watchlist_example.csv`: Simple example watchlist.

---

## 📄 License
