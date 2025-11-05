# ui/streamlit_app.py — v4.8.2 Cached + Paginated + Gradient Pro
"""
Crypto Futures Scanner — v4.8.2
- Cached scan (st.cache_data TTL=300s)
- Pagination for summary table and paged detail rendering
- Gradient card UI (Long/Short/Neutral)
- Decode compressed OHLCV (ohlcv_z) if present
- Backwards compatible with scanner_core outputs (dict{'results', 'metrics'} or list)
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

# ---- project path fix ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- backend imports ----
from scanner.scanner_core import scan
from scanner.datafetch import test_ping_binance_fapi
try:
    from scanner.history import list_signals
except Exception:
    list_signals = None

# ---- storage ----
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
TMP_RESULTS_FILE = "/tmp/crypto_scanner_results.json"

# ---- helper functions ----
def save_to_portfolio(entry: dict):
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
    """Decode base64+zlib compressed DataFrame (orient=split) back to pandas DataFrame."""
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
st.set_page_config(page_title="Crypto Futures Scanner v4.8.2", layout="wide")
st.markdown("""
<style>
.signal-card { border-radius: 14px; padding: 14px 18px; margin-bottom:10px; color: white; box-shadow: 0 5px 15px rgba(0,0,0,0.25); }
.metric { font-size:22px; font-weight:700; margin-top:4px; }
.small-muted { color:#cfcfcf; font-size:13px; }
.icon-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Crypto Futures Scanner — v4.8.2 (Cached & Paginated)")

# ---- controls ----
with st.expander("⚙️ Scanner Configuration", expanded=True):
    colA, colB, colC = st.columns(3)
    with colA:
        exchange = st.selectbox("Exchange", ["binance", "bybit"])
    with colB:
        top_n = st.selectbox("Top N coins to scan", [50, 100, 150, 200], index=1)
    with colC:
        timeframe = st.selectbox("Primary timeframe", ["15m", "1h", "4h", "1d"], index=1)
    col1, col2 = st.columns(2)
    with col1:
        candles = st.slider("Candles to fetch", 200, 1200, 500, step=50)
    with col2:
        delay = st.number_input("Delay between requests (s)", min_value=0.0, max_value=2.0, value=0.15, step=0.01)
    mtf = st.multiselect("Multi-timeframe confirmation (primary first)", ["15m", "1h", "4h", "12h", "1d"], default=[timeframe, "4h", "1d"])

# ---- ping button ----
col_ping, _ = st.columns([1,4])
with col_ping:
    if st.button("🔁 Test Binance FAPI"):
        st.info("Testing fapi.binance.com ...")
        ok = False
        try:
            ok = test_ping_binance_fapi()
        except Exception as e:
            st.warning(f"Ping error: {e}")
        if ok:
            st.success("✅ Binance FAPI reachable")
        else:
            st.warning("⚠️ Binance FAPI not reachable — fallback endpoints may be used")

st.write("---")

# ---- cached scan wrapper ----
@st.cache_data(ttl=300, show_spinner=False)
def cached_scan(exchange, top_n, timeframe, candles, delay, mtf):
    # returns whatever scan() returns; we also persist to tmp file for recovery
    pkt = scan(exchange, top_n=top_n, timeframe=timeframe, limit_ohlcv=candles, delay_between_requests=delay, mtf=mtf)
    try:
        # ensure JSON-serializable for cache persistence
        if isinstance(pkt, dict):
            save_last_results(pkt)
    except Exception:
        pass
    return pkt

# ---- restore previous if session lost ----
if "scan_packet" not in st.session_state:
    st.session_state["scan_packet"] = None
if not st.session_state["scan_packet"]:
    cached = load_last_results()
    if cached:
        st.session_state["scan_packet"] = cached
        st.info("🔄 Restored last scan results from local cache (session recovery).")

# ---- scan trigger ----
col_scan_left, col_scan_right = st.columns([2,1])
with col_scan_left:
    if st.button("🚀 Start Manual Scan", use_container_width=True):
        with st.spinner(f"Scanning top {top_n} symbols on {exchange} ..."):
            try:
                pkt = cached_scan(exchange, top_n, timeframe, candles, delay, mtf)
                # compatibility: some older scan returned list, normalize to dict
                if isinstance(pkt, list):
                    pkt = {"results": pkt, "metrics": {}}
                if isinstance(pkt, dict) and "results" in pkt:
                    st.session_state["scan_packet"] = pkt
                    save_last_results(pkt)
                    st.success("✅ Scan complete (cached)")
                else:
                    st.error("Scan returned unexpected payload")
            except Exception as e:
                st.error(f"Scan failed: {e}")
                st.code(traceback.format_exc())
with col_scan_right:
    st.write(" ")

pkt = st.session_state.get("scan_packet")
if not pkt:
    st.info("No scan results available. Run a manual scan.")
    st.stop()

# ---- normalize results list ----
if isinstance(pkt, dict):
    results = pkt.get("results", []) or []
    metrics = pkt.get("metrics", {}) or {}
else:
    results = pkt if isinstance(pkt, list) else []
    metrics = {}

# ---- build dataframe safely ----
try:
    df = pd.json_normalize(results)
except Exception as e:
    st.error(f"Failed to normalize results: {e}")
    df = pd.DataFrame()

# cast common numeric cols
for c in ["entry","tp1","tp2","sl","support","resistance","confidence"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["confidence"] = df.get("confidence", 0).fillna(0)

# ---- pagination controls ----
page_size = st.sidebar.selectbox("Signals per page", [5,10,15,20], index=1)
total_items = len(results)
total_pages = max(1, math.ceil(total_items / page_size))
page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
start = (page - 1) * page_size
end = start + page_size

# ---- metrics panel ----
if metrics:
    st.sidebar.markdown("### Scan metrics")
    st.sidebar.write(f"- Duration: {metrics.get('total_sec','?')}s")
    st.sidebar.write(f"- Avg latency: {metrics.get('avg_latency','?')}s")
    st.sidebar.write(f"- Concurrency: {metrics.get('concurrency','?')}")
    st.sidebar.write(f"- Success: {metrics.get('success','?')}/{metrics.get('total','?')}")
    st.sidebar.write("---")

# ---- compact summary (paged) ----
st.markdown("### 📋 Compact Summary (paged)")
if not df.empty:
    display_cols = [c for c in ["symbol","signal","confidence","entry","tp1","tp2","sl","support","resistance"] if c in df.columns]
    df_page = df.iloc[start:end].reset_index(drop=True)
    st.dataframe(df_page[display_cols], use_container_width=True, height=240)
else:
    st.info("No results to show in summary table.")

st.write("---")

# ---- detailed signals (only page items) ----
st.markdown(f"## 🎯 Detailed Signals — Page {page}/{total_pages} — Showing {len(df_page)} items")

for idx, r in enumerate(results[start:end]):
    if not r or not isinstance(r, dict):
        continue
    symbol = r.get("symbol") or r.get("market") or "UNKNOWN"
    signal = r.get("signal") or "NEUTRAL"
    confidence = r.get("confidence") or 0
    entry = r.get("entry")
    tp1 = r.get("tp1")
    tp2 = r.get("tp2")
    sl = r.get("sl")
    support = r.get("support")
    resistance = r.get("resistance")
    reasons = r.get("reasons") or []
    meta = r.get("meta") or {}
    mtf_breakdown = meta.get("signals_per_tf", {})

    # gradient color
    if signal == "LONG":
        grad = "linear-gradient(90deg,#00b09b,#96c93d)"
        btn_color = "#00a56b"
    elif signal == "SHORT":
        grad = "linear-gradient(90deg,#ff416c,#ff4b2b)"
        btn_color = "#ff416c"
    else:
        grad = "linear-gradient(90deg,#616161,#9e9e9e)"
        btn_color = "#7b7b7b"

    # header card
    st.markdown(f"""
    <div class="signal-card" style="background:{grad}">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <div style="font-size:14px; opacity:0.95;"><strong>{symbol}</strong></div>
          <div class="metric">{signal} — {round(float(confidence),1)}%</div>
        </div>
        <div style="text-align:right;">
          <div style="margin-bottom:6px;"><button style="background:{btn_color}; color:white; border:none; padding:8px 12px; border-radius:8px; font-weight:700;">{signal}</button></div>
          <div class="small-muted">Entry: {entry or '-'}  •  TP1: {tp1 or '-'}  •  SL: {sl or '-'}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"{symbol} — Trading Plan & Indicators", expanded=False):
        left, right = st.columns([0.6, 0.4])
        with left:
            st.markdown("#### 🧾 Trading Plan")
            st.write(f"**Entry:** {entry}")
            rr_text = ""
            try:
                if entry is not None and tp1 is not None and sl is not None:
                    denom = abs(entry - sl)
                    if denom > 0:
                        rr_text = f" (R:R ~ {round((tp1-entry)/denom,2)})"
            except Exception:
                rr_text = ""
            st.write(f"**TP1:** {tp1}{rr_text}")
            st.write(f"**TP2:** {tp2}")
            st.write(f"**SL:** {sl}")
            st.write(f"**Support:** {support}  |  **Resistance:** {resistance}")
            st.write("---")

            st.markdown("#### 🧭 Multi-Timeframe Summary")
            if mtf_breakdown:
                cols_tf = st.columns(len(mtf) if mtf else max(1, len(mtf_breakdown)))
                for i, tfname in enumerate(mtf if mtf else list(mtf_breakdown.keys())):
                    with cols_tf[min(i, len(cols_tf)-1)]:
                        val = mtf_breakdown.get(tfname, "NEUTRAL")
                        dot_color = "#00ff88" if val=="LONG" else "#ff6f61" if val=="SHORT" else "#9e9e9e"
                        st.markdown(f"<span class='icon-dot' style='background:{dot_color}'></span> {tfname}: {val}", unsafe_allow_html=True)
                        # show small score if present
                        if meta.get("scores_per_tf") and tfname in meta.get("scores_per_tf"):
                            try:
                                sc = meta["scores_per_tf"][tfname]
                                st.caption(f"Score: {round(sc*100,1)}%")
                            except Exception:
                                pass
            else:
                st.write("No multi-timeframe breakdown.")

            st.write("---")
            if reasons:
                st.markdown("#### 🧠 Reasons")
                for rr in reasons[:10]:
                    st.write(f"- {rr}")

            if st.button("💾 Save to Portfolio", key=f"save_{symbol}_{page}"):
                payload = {
                    "ts": int(time.time()),
                    "symbol": symbol,
                    "signal": signal,
                    "confidence": confidence,
                    "entry": entry,
                    "tp1": tp1,
                    "tp2": tp2,
                    "sl": sl,
                    "support": support,
                    "resistance": resistance,
                }
                ok = save_to_portfolio(payload)
                if ok:
                    st.success("Saved to portfolio")
                else:
                    st.error("Failed to save")

        with right:
            st.markdown("#### 📈 Chart (lazy)")
            show_chart = st.checkbox("Show multi-panel chart", key=f"chart_{symbol}_{page}")
            if not show_chart:
                st.caption("Chart is hidden — check to render (reduces UI load).")
            else:
                # prefer compressed snapshot ohlcv_z else raw ohlcv_data
                dfc = None
                if r.get("ohlcv_z"):
                    dfc = decode_ohlcv_z(r.get("ohlcv_z"))
                elif r.get("ohlcv_data"):
                    try:
                        dfc = pd.DataFrame(r.get("ohlcv_data"))
                        if "datetime" in dfc.columns:
                            dfc["datetime"] = pd.to_datetime(dfc["datetime"])
                            dfc = dfc.set_index("datetime")
                    except Exception:
                        dfc = None

                if dfc is None or dfc.empty:
                    st.warning("No cached OHLCV snapshot available for this symbol.")
                else:
                    try:
                        # ensure types
                        for c in ["open","high","low","close","volume"]:
                            if c in dfc.columns:
                                dfc[c] = pd.to_numeric(dfc[c], errors="coerce")

                        X = dfc.index
                        fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                                            row_heights=[0.42,0.12,0.12,0.15,0.18],
                                            vertical_spacing=0.04,
                                            subplot_titles=("Price","RSI","Stoch RSI","MACD","Volume"))

                        # candles
                        if all(c in dfc.columns for c in ["open","high","low","close"]):
                            fig.add_trace(go.Candlestick(x=X, open=dfc["open"], high=dfc["high"], low=dfc["low"], close=dfc["close"], name="Price"), row=1, col=1)
                        # EMAs
                        for ema in ["ema_9","ema_21","ema_50","ema_200"]:
                            if ema in dfc.columns:
                                fig.add_trace(go.Scatter(x=X, y=dfc[ema], name=ema.upper(), line=dict(width=1.2)), row=1, col=1)
                        # supertrend
                        if "supertrend" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["supertrend"], name="Supertrend", line=dict(color="#00e676", width=1.2, dash="dot")), row=1, col=1)
                        # plan lines
                        for key, style, colr in [("entry","solid","#00e676"),("tp1","dash","#64b5f6"),("tp2","dot","#1976d2"),("sl","dash","#e53935")]:
                            v = r.get(key)
                            if v:
                                fig.add_hline(y=v, line=dict(dash=style, width=1.2, color=colr), row=1, col=1)
                        # RSI
                        if "rsi" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["rsi"], name="RSI"), row=2, col=1)
                            fig.add_hline(y=70, line=dict(dash="dash"), row=2, col=1)
                            fig.add_hline(y=30, line=dict(dash="dash"), row=2, col=1)
                        # Stoch
                        if "stoch_k" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["stoch_k"]*100, name="StochK"), row=3, col=1)
                            fig.add_trace(go.Scatter(x=X, y=dfc["stoch_d"]*100, name="StochD"), row=3, col=1)
                        # MACD
                        if "macd" in dfc.columns and "macd_signal" in dfc.columns:
                            hist = (dfc["macd"] - dfc["macd_signal"]).fillna(0)
                            fig.add_trace(go.Bar(x=X, y=hist, name="Hist", marker_color=[("#26a69a" if v>=0 else "#ef5350") for v in hist]), row=4, col=1)
                            fig.add_trace(go.Scatter(x=X, y=dfc["macd"], name="MACD"), row=4, col=1)
                            fig.add_trace(go.Scatter(x=X, y=dfc["macd_signal"], name="Signal"), row=4, col=1)
                        # volume
                        if "volume" in dfc.columns:
                            fig.add_trace(go.Bar(x=X, y=dfc["volume"], name="Volume"), row=5, col=1)

                        theme = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
                        fig.update_layout(height=760, template=theme, showlegend=True, margin=dict(t=40,b=20))
                        st.plotly_chart(fig, use_container_width=True)

                        # csv download
                        buf = io.StringIO()
                        dfc.to_csv(buf)
                        st.download_button("📥 Download candle CSV", buf.getvalue(), file_name=f"{symbol}_{timeframe}.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Chart rendering failed: {e}")
                        st.code(traceback.format_exc())

st.write("---")

# ---- portfolio viewer & history ----
p1, p2 = st.columns([1,1])
with p1:
    st.markdown("### 📁 Portfolio (saved)")
    if PORTFOLIO_FILE.exists():
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                arr = json.load(f)
            dfp = pd.DataFrame(arr)
            if not dfp.empty:
                dfp["ts"] = pd.to_datetime(dfp["ts"], unit="s")
            st.dataframe(dfp.sort_values("ts", ascending=False).reset_index(drop=True), use_container_width=True, height=300)
        except Exception as e:
            st.error(f"Failed to load portfolio: {e}")
    else:
        st.info("No portfolio entries yet. Save from a trading plan.")

with p2:
    st.markdown("### 🕒 Scan History")
    if list_signals:
        if st.button("Load recent 50 signals"):
            try:
                rows = list_signals(limit=50)
                if rows:
                    dfh = pd.DataFrame(rows)
                    dfh["ts"] = pd.to_datetime(dfh["ts"], unit="s")
                    st.dataframe(dfh[["ts","symbol","signal","confidence","entry","tp1","tp2","sl"]], use_container_width=True, height=300)
                else:
                    st.info("No history records found.")
            except Exception as e:
                st.error(f"Failed to load history: {e}")
    else:
        st.caption("History module not available in scanner.history")

st.caption("v4.8.2 — Cached (5min) + Paginated + Gradient Pro. Use page controls in sidebar and 'Show multi-panel chart' to render charts lazily.")
