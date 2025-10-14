# ⚡ India Intraday Screener — FastScan

A streamlined Streamlit app optimized for **speed**: prefilter by daily liquidity, download **short intraday history**, compute **minimal indicators**, and rank BUY/SELL quickly. Use Pro gates (VWAP, ORB, 1h confirm) only when needed.

> **Disclaimer:** Educational/informational only. Not investment advice. Yahoo data may be delayed/inaccurate.

---

## 🚀 Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Why it’s faster
- **Daily prefilter** → keep top **N by turnover** within price bounds.
- **Short intraday period** (default 10 days @ 5m), **clamped** to safe Yahoo limits to prevent empty fetches.
- **Minimal indicators** on just the **last ~120 bars** per ticker (SMA20/50, RSI14, MACD, VWAP).
- **Optional heavy gates** (VWAP, ORB, HTF) are **off** by default.
- **On-demand charts** only for selected ticker.

### Data & Limits
- Yahoo intraday limits: **1m ≈ 7 days per request**, **5m/15m/30m ≈ ~30 days**, **1h longer** (conservative caps used). citeturn6search34turn6search35
- NSE universe list fetched from the **EQUITY_L.csv** archive endpoint. citeturn6search28
- Auto-refresh options exist via third-party components if needed (not enabled here to keep compute low). citeturn6search22turn6search23

---

## Sidebar Controls
- **Universe**: NSE (auto) / CSV / Manual
- **Interval**: 5m/15m/30m/1h
- **History**: days (auto-clamped)
- **Liquidity Prefilter**: min/max price, min 20‑D avg turnover (₹ Cr), **Max Universe Size** (top by turnover)
- **Modes & Gates**:
  - **QuickScan (fast)** — default
  - **VWAP gate** — BUY Close > VWAP; SELL Close < VWAP (on by default)
  - **ORB gate** — optional; OR window configurable
  - **1h confirm** — optional; check SMA20 vs SMA50
- **Top N** results & **CSV export**

---

## Deployment (Streamlit Cloud)
1. Push `app.py`, `requirements.txt`, optional `.streamlit/config.toml`, and `data/watchlist_example.csv`.
2. Create app at **https://share.streamlit.io** → select repo & `app.py`.
3. Set Python 3.11; consider persistent caching.

---

## Files
- `app.py` — FastScan app (optimized).
- `requirements.txt` — minimal deps.
- `.streamlit/config.toml` — theme (optional).
- `data/watchlist_example.csv` — sample tickers.

---

## License
MIT
