# ui/streamlit_app.py — v4.8.3-Final (Stable Cached + Paginated + Gradient Pro + Portfolio Fix)
"""
Crypto Futures Scanner — v4.8.3-Final
- Cached scan (TTL=300s)
- Pagination + Gradient UI
- Decode compressed OHLCV (ohlcv_z)
- Fix: df_page undefined
- Fix: confidence safe fill
- Fix: portfolio display + working path for systemd
"""

import sys
import os
import json
import io
import time
import traceback
import zlib
import base64
import math
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- ensure working directory & import path ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- backend imports ----
from scanner.scanner_core import scan
from scanner.datafetch import test_ping_binance_fapi
try:
    from scanner.history import list_signals
except Exception:
    list_signals = None

# ---- storage paths ----
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
TMP_RESULTS_FILE = "/tmp/crypto_scanner_results.json"

# ---- helper functions ----
def save_to_portfolio(entry: dict):
    """Append signal info to portfolio.json"""
    try:
        if PORTFOLIO_FILE.exists():
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                arr = json.load(f)
        else:
            arr = []
        arr.append(entry)
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(arr, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save portfolio: {e}")
        return False

def decode_ohlcv_z(b64z: str):
    """Decode base64+zlib compressed OHLCV JSON."""
    if not b64z:
        return None
    try:
        zb = base64.b64decode(b64z.encode("ascii"))
        j = zlib.decompress(zb).decode("utf-8")
        df = pd.read_json(j, orient="split")
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        return df
    except Exception:
        return None

def save_last_results(pkt):
    try:
        with open(TMP_RESULTS_FILE, "w") as f:
            json.dump(pkt, f)
    except Exception:
        pass

def load_last_results():
    try:
        if os.path.exists(TMP_RESULTS_FILE):
            with open(TMP_RESULTS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        return None
    return None

# ---- page config & CSS ----
st.set_page_config(page_title="Crypto Futures Scanner v4.8.3", layout="wide")
st.markdown("""
<style>
.signal-card { border-radius: 14px; padding: 14px 18px; margin-bottom:10px; color: white; box-shadow: 0 5px 15px rgba(0,0,0,0.25); }
.metric { font-size:22px; font-weight:700; margin-top:4px; }
.small-muted { color:#cfcfcf; font-size:13px; }
.icon-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Crypto Futures Scanner — v4.8.3 Development (DYOR)")

# ---- controls ----
with st.expander("⚙️ Scanner Configuration", expanded=True):
    colA, colB, colC = st.columns(3)
    with colA:
        exchange = st.selectbox("Exchange", ["binance", "bybit"])
    with colB:
        top_n = st.selectbox("Top N coins to scan", [10, 25, 50, 100, 150, 200], index=1)
    with colC:
        timeframe = st.selectbox("Primary timeframe", ["15m", "1h", "4h", "1d"], index=1)
    col1, col2 = st.columns(2)
    with col1:
        candles = st.slider("Candles to fetch", 200, 1200, 500, step=50)
    with col2:
        delay = st.number_input("Delay between requests (s)", min_value=0.0, max_value=2.0, value=0.15, step=0.01)
    mtf = st.multiselect("Multi-timeframe confirmation (primary first)", ["15m", "1h", "4h", "12h", "1d"], default=[timeframe, "4h", "1d"])

# ---- ping test ----
if st.button("🔁 Test Binance FAPI"):
    st.info("Testing fapi.binance.com ...")
    try:
        ok = test_ping_binance_fapi()
        st.success("✅ Binance FAPI reachable" if ok else "⚠️ Binance FAPI not reachable")
    except Exception as e:
        st.warning(f"Ping error: {e}")

st.write("---")

# ---- cached scan wrapper ----
@st.cache_data(ttl=300, show_spinner=False)
def cached_scan(exchange, top_n, timeframe, candles, delay, mtf):
    pkt = scan(exchange, top_n=top_n, timeframe=timeframe, limit_ohlcv=candles, delay_between_requests=delay, mtf=mtf)
    try:
        if isinstance(pkt, dict):
            save_last_results(pkt)
    except Exception:
        pass
    return pkt

# ---- restore session ----
if "scan_packet" not in st.session_state:
    st.session_state["scan_packet"] = None
if not st.session_state["scan_packet"]:
    cached = load_last_results()
    if cached:
        st.session_state["scan_packet"] = cached
        st.info("🔄 Restored last scan results from cache.")

# ---- manual scan ----
if st.button("🚀 Start Manual Scan", use_container_width=True):
    with st.spinner(f"Scanning top {top_n} symbols on {exchange} ..."):
        try:
            pkt = cached_scan(exchange, top_n, timeframe, candles, delay, mtf)
            if isinstance(pkt, list):
                pkt = {"results": pkt, "metrics": {}}
            st.session_state["scan_packet"] = pkt
            save_last_results(pkt)
            st.success("✅ Scan complete (cached)")
        except Exception as e:
            st.error(f"Scan failed: {e}")
            st.code(traceback.format_exc())

pkt = st.session_state.get("scan_packet")
if not pkt:
    st.info("No scan results available. Run a manual scan.")
    st.stop()

# ---- normalize results ----
results = pkt.get("results", []) if isinstance(pkt, dict) else (pkt if isinstance(pkt, list) else [])
metrics = pkt.get("metrics", {}) if isinstance(pkt, dict) else {}

try:
    df = pd.json_normalize(results)
except Exception as e:
    st.error(f"Failed to normalize results: {e}")
    df = pd.DataFrame()

for c in ["entry","tp1","tp2","sl","support","resistance","confidence"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
if "confidence" not in df.columns:
    df["confidence"] = 0
else:
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)

# ---- pagination ----
page_size = st.sidebar.selectbox("Signals per page", [5,10,15,20], index=1)
total_items = len(results)
total_pages = max(1, math.ceil(total_items / page_size))
page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
start = (page - 1) * page_size
end = start + page_size

# ---- metrics panel ----
if metrics:
    st.sidebar.markdown("### Scan metrics")
    for k,v in metrics.items():
        st.sidebar.write(f"- {k}: {v}")
    st.sidebar.write("---")

# ---- compact summary ----
st.markdown("### 📋 Compact Summary (paged)")
df_page = pd.DataFrame()
if not df.empty:
    cols = [c for c in ["symbol","signal","confidence","entry","tp1","tp2","sl","support","resistance"] if c in df.columns]
    df_page = df.iloc[start:end].reset_index(drop=True)
    st.dataframe(df_page[cols], use_container_width=True, height=240)
else:
    st.info("No results to show in summary table.")
    df_page = pd.DataFrame(columns=["symbol","signal","confidence"])

st.write("---")
st.markdown(f"## 🎯 Detailed Signals — Page {page}/{total_pages} — Showing {len(df_page)} items")

# ---- render details (loop) ----
for r in results[start:end]:
    if not isinstance(r, dict):
        continue
    symbol = r.get("symbol") or "UNKNOWN"
    signal = r.get("signal") or "NEUTRAL"
    confidence = r.get("confidence") or 0
    entry = r.get("entry"); tp1 = r.get("tp1"); tp2 = r.get("tp2"); sl = r.get("sl")
    support = r.get("support"); resistance = r.get("resistance")
    reasons = r.get("reasons") or []; meta = r.get("meta") or {}
    mtf_breakdown = meta.get("signals_per_tf", {})

    # gradient card
    grad = "linear-gradient(90deg,#00b09b,#96c93d)" if signal=="LONG" else \
           "linear-gradient(90deg,#ff416c,#ff4b2b)" if signal=="SHORT" else \
           "linear-gradient(90deg,#616161,#9e9e9e)"
    btn_color = "#00a56b" if signal=="LONG" else "#ff416c" if signal=="SHORT" else "#7b7b7b"

    st.markdown(f"""
    <div class="signal-card" style="background:{grad}">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div><div style="font-size:14px;"><strong>{symbol}</strong></div>
        <div class="metric">{signal} — {round(float(confidence),1)}%</div></div>
        <div style="text-align:right;">
          <button style="background:{btn_color};color:white;border:none;padding:8px 12px;border-radius:8px;font-weight:700;">{signal}</button>
          <div class="small-muted">Entry: {entry or '-'} • TP1: {tp1 or '-'} • SL: {sl or '-'}</div>
        </div>
      </div></div>
    """, unsafe_allow_html=True)

    with st.expander(f"{symbol} — Trading Plan & Indicators", expanded=False):
        left, right = st.columns([0.6,0.4])
        with left:
            st.markdown("#### 🧾 Trading Plan")
            st.write(f"**Entry:** {entry}")
            st.write(f"**TP1:** {tp1} | **TP2:** {tp2}")
            st.write(f"**SL:** {sl}")
            st.write(f"**Support:** {support} | **Resistance:** {resistance}")
            st.write("---")
            if reasons:
                st.markdown("#### 🧠 Reasons")
                for rr in reasons[:10]:
                    st.write(f"- {rr}")
            if st.button("💾 Save to Portfolio", key=f"save_{symbol}_{page}"):
                payload = {"ts": int(time.time()), "symbol": symbol, "signal": signal,
                           "confidence": confidence, "entry": entry, "tp1": tp1, "tp2": tp2,
                           "sl": sl, "support": support, "resistance": resistance}
                st.success("Saved to portfolio" if save_to_portfolio(payload) else "Failed")

        with right:
            st.markdown("#### 📈 Chart (lazy)")
            show_chart = st.checkbox("Show multi-panel chart", key=f"chart_{symbol}_{page}")
            if not show_chart:
                st.caption("Chart hidden — enable to render.")
            else:
                dfc = decode_ohlcv_z(r.get("ohlcv_z")) if r.get("ohlcv_z") else None
                if not isinstance(dfc, pd.DataFrame) or dfc.empty:
                    st.warning("No cached OHLCV snapshot available.")
                else:
                    for c in ["open","high","low","close","volume"]:
                        dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
                    X = dfc.index
                    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                                        row_heights=[0.42,0.12,0.12,0.15,0.18],
                                        vertical_spacing=0.04,
                                        subplot_titles=("Price","RSI","Stoch RSI","MACD","Volume"))
                    if all(c in dfc.columns for c in ["open","high","low","close"]):
                        fig.add_trace(go.Candlestick(x=X, open=dfc["open"], high=dfc["high"], low=dfc["low"], close=dfc["close"], name="Price"), row=1, col=1)
                    for ema in ["ema_9","ema_21","ema_50","ema_200"]:
                        if ema in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc[ema], name=ema.upper(), line=dict(width=1.2)), row=1, col=1)
                    theme = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
                    fig.update_layout(height=700, template=theme, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

st.write("---")

# ---- portfolio viewer & history ----
p1, p2 = st.columns(2)
with p1:
    st.markdown("### 📁 Portfolio (saved)")
    if PORTFOLIO_FILE.exists():
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                arr = json.load(f)
            dfp = pd.DataFrame(arr)
            if not dfp.empty:
                dfp["ts"] = pd.to_datetime(dfp["ts"], unit="s")
            st.dataframe(dfp.sort_values("ts", ascending=False).reset_index(drop=True),
                         use_container_width=True, height=300)
        except Exception as e:
            st.error(f"Failed to load portfolio: {e}")
    else:
        st.info("No portfolio entries yet. Save from a trading plan.")

with p2:
    st.markdown("### 🕒 Scan History")
    if list_signals and st.button("Load recent 50 signals"):
        try:
            rows = list_signals(limit=50)
            if rows:
                dfh = pd.DataFrame(rows)
                dfh["ts"] = pd.to_datetime(dfh["ts"], unit="s")
                st.dataframe(dfh[["ts","symbol","signal","confidence","entry","tp1","tp2","sl"]],
                             use_container_width=True, height=300)
            else:
                st.info("No history found.")
        except Exception as e:
            st.error(f"Failed to load history: {e}")
    else:
        st.caption("History module not available.")

st.caption("v4.8.3-Final — Cached + Paginated + Gradient UI + Portfolio restored")
