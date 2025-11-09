# ui/streamlit_app.py — v4.9.1 (Tabs Edition)
"""
Crypto Futures Scanner — v4.9.1
- Tab layout: 📊 Market Scanner & 🎯 Single Scanner
- Reusable chart renderer (make_chart)
- Cached scan (TTL=300s)
- Pagination + Gradient UI
- Decode compressed OHLCV (ohlcv_z)
- Portfolio & history viewer
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
import pytz
import requests
from pathlib import Path
import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# ---- reusable chart renderer ----
def make_chart(dfc: pd.DataFrame):
    """Reusable multi-panel chart for both mass-scan & single-scan mode."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd

    if not isinstance(dfc, pd.DataFrame) or dfc.empty:
        return None

    for c in ["open","high","low","close","volume","rsi","stoch_k","stoch_d","macd","macd_signal","macd_hist"]:
        if c in dfc.columns:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")

    dfc = dfc.sort_index()
    X = dfc.index

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.42,0.12,0.12,0.15,0.19],
        vertical_spacing=0.04,
        subplot_titles=("Price","RSI","Stoch RSI","MACD","Volume")
    )

    if all(c in dfc.columns for c in ["open","high","low","close"]):
        fig.add_trace(go.Candlestick(x=X, open=dfc["open"], high=dfc["high"],
                                     low=dfc["low"], close=dfc["close"], name="Price"), row=1, col=1)
    for ema in ["ema_9","ema_21","ema_50","ema_200"]:
        if ema in dfc.columns:
            fig.add_trace(go.Scatter(x=X, y=dfc[ema], name=ema.upper(), line=dict(width=1.0)), row=1, col=1)
    if "supertrend" in dfc.columns:
        fig.add_trace(go.Scatter(x=X, y=dfc["supertrend"], name="Supertrend",
                                 line=dict(width=1.0, dash="dot")), row=1, col=1)
    if "vwap" in dfc.columns:
        fig.add_trace(go.Scatter(x=X, y=dfc["vwap"], name="VWAP", line=dict(width=1.0)), row=1, col=1)

    if "rsi" in dfc.columns:
        fig.add_trace(go.Scatter(x=X, y=dfc["rsi"], name="RSI"), row=2, col=1)
    if "stoch_k" in dfc.columns:
        fig.add_trace(go.Scatter(x=X, y=dfc["stoch_k"], name="%K"), row=3, col=1)
    if "stoch_d" in dfc.columns:
        fig.add_trace(go.Scatter(x=X, y=dfc["stoch_d"], name="%D"), row=3, col=1)
    if all(c in dfc.columns for c in ["macd","macd_signal","macd_hist"]):
        fig.add_trace(go.Scatter(x=X, y=dfc["macd"], name="MACD"), row=4, col=1)
        fig.add_trace(go.Scatter(x=X, y=dfc["macd_signal"], name="Signal"), row=4, col=1)
        fig.add_trace(go.Bar(x=X, y=dfc["macd_hist"], name="Hist"), row=4, col=1)
    if "volume" in dfc.columns:
        fig.add_trace(go.Bar(x=X, y=dfc["volume"], name="Volume"), row=5, col=1)

    fig.update_layout(height=760, template="plotly_dark", showlegend=True)
    return fig


# ---- ensure working directory & import path ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- backend imports ----
from scanner.scanner_core import scan
from scanner.sentiment_binance import fetch_binance_sentiment, plot_sentiment
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


# ---- helpers ----
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


def decode_ohlcv_z(zb64: str):
    if not zb64:
        return None
    try:
        raw = base64.b64decode(zb64)
        data = zlib.decompress(raw).decode("utf-8")
        try:
            df = pd.read_json(io.StringIO(data), orient="split")
        except Exception:
            df = pd.read_csv(io.StringIO(data))
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.set_index("datetime")
        return df
    except Exception as e:
        print("decode_ohlcv_z error:", e)
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
st.set_page_config(page_title="Crypto Futures Scanner", layout="wide")
st.markdown("""
<style>
.signal-card { border-radius: 14px; padding: 14px 18px; margin-bottom:10px; color: white; box-shadow: 0 5px 15px rgba(0,0,0,0.25); }
.metric { font-size:22px; font-weight:700; margin-top:4px; }
.small-muted { color:#cfcfcf; font-size:13px; }
.icon-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Crypto Futures Scanner v4.9.0")

# ======================
# TAB LAYOUT START
# ======================
tab_market, tab_single, tab_sentiment = st.tabs(["📊 Market Scanner", "🎯 Single Scanner", "🫣 Sentiment"])

# ===================================================
# TAB 1 — MARKET SCANNER (original full functionality)
# ===================================================
with tab_market:

    # (SELURUH KODE MARKET SCANNER KAMU DIMASUKKAN DI SINI TANPA PERUBAHAN)
    # --- BEGIN COPY FROM EXISTING ---
    # Semua kode kamu dari konfigurasi sampai portfolio/history tetap dipakai.
    # Kamu bisa langsung copy isi dari versi kamu sebelumnya ke sini,
    # dari "with st.expander("⚙️ Scanner Configuration"..." sampai sebelum caption terakhir.
    # --- END COPY FROM EXISTING ---

    #st.set_page_config(page_title="Crypto Futures Scanner v4.8.3", layout="wide")
    st.markdown("""
    <style>
    .signal-card { border-radius: 14px; padding: 14px 18px; margin-bottom:10px; color: white; box-shadow: 0 5px 15px rgba(0,0,0,0.25); }
    .metric { font-size:22px; font-weight:700; margin-top:4px; }
    .small-muted { color:#cfcfcf; font-size:13px; }
    .icon-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🚀 Crypto Futures Scanner — v4.8.3 Development (DYOR)")

    def simple_trading_sessions():
                """Versi sederhana trading session monitor"""
                
                st.header("🕒 Trading Sessions (WIB Time)")
        
                # Current time
                wib = pytz.timezone('Asia/Jakarta')
                current_time = datetime.now(wib)
                current_hour = current_time.hour
                
                st.metric("Current Time (WIB)", current_time.strftime("%H:%M:%S"))
                
                # CORRECTED session times
                sessions = [
                    {"name": "Sydney", "open": 5, "close": 14, "active": False},
                    {"name": "Tokyo", "open": 7, "close": 16, "active": False},  # CORRECTED ⚡
                    {"name": "London", "open": 14, "close": 23, "active": False},
                    {"name": "New York", "open": 19, "close": 4, "active": False}
                ]
                
                # Check active sessions
                for session in sessions:
                    if session['name'] == 'New York':  # Cross midnight
                        session['active'] = current_hour >= session['open'] or current_hour < session['close']
                    else:  # Normal sessions
                        session['active'] = session['open'] <= current_hour < session['close']
                
                # Display sessions
                cols = st.columns(4)
                for idx, session in enumerate(sessions):
                    with cols[idx]:
                        if session['active']:
                            st.success(f"🟢 {session['name']}")
                            st.write(f"⏰ {session['open']:02d}:00-{session['close']:02d}:00")
                            st.write("**ACTIVE**")
                        else:
                            st.error(f"🔴 {session['name']}")
                            st.write(f"⏰ {session['open']:02d}:00-{session['close']:02d}:00")
                            st.write("CLOSED")
                
                # Market status based on corrected times
                st.markdown("---")
                active_count = sum(1 for s in sessions if s['active'])
                if active_count >= 2:
                    st.success("🚀 HIGH VOLATILITY - Multiple sessions active")
                elif active_count == 1:
                    st.info("📈 MARKET OPEN - One session active")
                else:
                    st.warning("💤 MARKETS CLOSED - No active sessions")

                
                #API FnG
                
                def get_fear_greed_history(limit=30):
                    """
                    Fetch Fear & Greed Index (historical)
                    """
                    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
                    r = requests.get(url).json()
                    data = r.get("data", [])
                    df = pd.DataFrame(data)
                    if not df.empty:
                        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
                        df["value"] = df["value"].astype(int)
                        df = df.sort_values("timestamp")
                    return df

                def get_fear_greed_index():
                    url = "https://api.alternative.me/fng/"
                    r = requests.get(url).json()
                    value = r["data"][0]["value"]
                    classification = r["data"][0]["value_classification"]
                    return int(value), classification

                # --- Streamlit UI ---
                st.markdown("---")
                st.markdown("### 📊 Market Overview")

                # Current FNG
                value, cls = get_fear_greed_index()
                st.metric("😨 Fear & Greed Index", f"{value} ({cls})")

                # Historical chart
                df_fng = get_fear_greed_history(limit=90)
                if not df_fng.empty:
                    fig = go.Figure()

                    # Area plot with color scale
                    fig.add_trace(go.Scatter(
                        x=df_fng["timestamp"],
                        y=df_fng["value"],
                        mode="lines+markers",
                        fill="tozeroy",
                        line=dict(color="#00bcd4", width=2),
                        marker=dict(size=5, color=df_fng["value"], colorscale="RdYlGn", showscale=True),
                        name="Fear & Greed Index",
                    ))

                    # Thresholds
                    fig.add_hline(y=25, line_dash="dot", line_color="red", annotation_text="Extreme Fear", annotation_position="top left")
                    fig.add_hline(y=50, line_dash="dot", line_color="orange", annotation_text="Neutral", annotation_position="top left")
                    fig.add_hline(y=75, line_dash="dot", line_color="green", annotation_text="Extreme Greed", annotation_position="top left")

                    fig.update_layout(
                        height=360,
                        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
                        title="Fear & Greed Index (Last 90 Days)",
                        xaxis_title="Date",
                        yaxis_title="Index Value",
                        yaxis=dict(range=[0, 100]),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Failed to fetch Fear & Greed Index data.")

                st.caption("Live data from alternative.me — updated every 24h. Lower = Fear 😨, Higher = Greed 😈")
    
    simple_trading_sessions()

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
                    # --- decode cached OHLCV ---
                    dfc = r.get("ohlcv_z")
                    if dfc:
                        dfc = decode_ohlcv_z(dfc)
                    else:
                        dfc = None

                    # --- validate DataFrame existence ---
                    if not isinstance(dfc, pd.DataFrame) or dfc.empty:
                        st.warning("No cached OHLCV snapshot available.")
                    else:
                        # sanitize numerics
                        for c in ["open","high","low","close","volume","rsi","stoch_k","stoch_d","macd","macd_signal","macd_hist"]:
                            if c in dfc.columns:
                                dfc[c] = pd.to_numeric(dfc[c], errors="coerce")

                        # sort and plot
                        dfc = dfc.sort_index()
                        X = dfc.index

                        fig = make_subplots(
                            rows=5, cols=1, shared_xaxes=True,
                            row_heights=[0.42,0.12,0.12,0.15,0.19],
                            vertical_spacing=0.04,
                            subplot_titles=("Price","RSI","Stoch RSI","MACD","Volume")
                        )

                        # --- PRICE + EMA + VWAP + SUPERTREND ---
                        if all(c in dfc.columns for c in ["open","high","low","close"]):
                            fig.add_trace(go.Candlestick(x=X, open=dfc["open"], high=dfc["high"], low=dfc["low"], close=dfc["close"], name="Price"), row=1, col=1)
                        for ema in ["ema_9","ema_21","ema_50","ema_200"]:
                            if ema in dfc.columns:
                                fig.add_trace(go.Scatter(x=X, y=dfc[ema], name=ema.upper(), line=dict(width=1.0)), row=1, col=1)
                        if "supertrend" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["supertrend"], name="Supertrend", line=dict(width=1.0, dash="dot")), row=1, col=1)
                        if "vwap" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["vwap"], name="VWAP", line=dict(width=1.0)), row=1, col=1)

                        # --- RSI ---
                        if "rsi" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["rsi"], name="RSI"), row=2, col=1)

                        # --- STOCH RSI ---
                        if "stoch_k" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["stoch_k"], name="%K"), row=3, col=1)
                        if "stoch_d" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["stoch_d"], name="%D"), row=3, col=1)

                        # --- MACD ---
                        has_macd = all(c in dfc.columns for c in ["macd","macd_signal","macd_hist"])
                        if has_macd:
                            fig.add_trace(go.Scatter(x=X, y=dfc["macd"], name="MACD"), row=4, col=1)
                            fig.add_trace(go.Scatter(x=X, y=dfc["macd_signal"], name="Signal"), row=4, col=1)
                            fig.add_trace(go.Bar(x=X, y=dfc["macd_hist"], name="Hist"), row=4, col=1)

                        # --- VOLUME ---
                        if "volume" in dfc.columns:
                            fig.add_trace(go.Bar(x=X, y=dfc["volume"], name="Volume"), row=5, col=1)

                        theme = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
                        fig.update_layout(height=760, template=theme, showlegend=True)
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

# ===================================================
# TAB 2 — SINGLE SCANNER
# ===================================================
with tab_single:
    st.markdown("## 🎯 Single Pair Scanner")

    from scanner.datafetch import (
        create_exchange,
        fetch_top_symbols_binance_futures,
        fetch_top_symbols_bybit_futures
    )

    colA, colB = st.columns(2)
    with colA:
        exchange_single = st.selectbox("Exchange", ["binance", "bybit"], key="single_exchange")
    with colB:
        mtf_single = st.multiselect(
            "Multi-timeframe (primary first)",
            ["15m","1h","4h","1d"],
            default=["1h","4h","1d"],
            key="single_mtf"
        )

    @st.cache_data(ttl=600)
    def load_symbols(exchange):
        ex = create_exchange(exchange)
        if exchange == "binance":
            return fetch_top_symbols_binance_futures(ex, limit=200)
        else:
            return fetch_top_symbols_bybit_futures(ex, limit=200)

    symbols_single = load_symbols(exchange_single)
    symbol_single = st.selectbox("Select Pair", symbols_single, key="single_symbol")
    candles_single = st.slider("Candles", 200, 1200, 500, step=50, key="single_candles")

    # --- Run Scan ---
    if st.button("🚀 Run Single Scan", use_container_width=True):
        with st.spinner(f"Scanning {symbol_single} on {exchange_single} ..."):
            primary_tf = mtf_single[0] if mtf_single else "1h"
            # --- Run dedicated single symbol scan ---
            from scanner.scanner_core import build_mtf_dfs_async, create_exchange
            import asyncio, aiohttp
            from scanner import signals_v4

            async def run_single_symbol(exchange_name, symbol, mtf, limit):
                ex = create_exchange(exchange_name)
                timeout = aiohttp.ClientTimeout(total=40)
                conn = aiohttp.TCPConnector(limit_per_host=10)
                async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
                    dfs = await build_mtf_dfs_async(exchange_name, ex, symbol, mtf, limit, session, delay_between_requests=0.15)
                sig = signals_v4.evaluate_signals_mtf(dfs)
                base_df = dfs.get(mtf[0])
                ohlcv_z = None
                if base_df is not None and not base_df.empty:
                    import zlib, base64
                    tmp = base_df.tail(400).reset_index().copy()
                    ohlcv_z = base64.b64encode(zlib.compress(tmp.to_json(orient="split", date_format="iso").encode())).decode()
                return {"symbol": symbol, "signal": sig.get("signal"), "confidence": sig.get("confidence"),
                        "reasons": sig.get("reasons"), "meta": sig.get("meta"),
                        "entry": float(base_df.iloc[-1]["close"]) if base_df is not None else None,
                        "ohlcv_z": ohlcv_z}

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(run_single_symbol(exchange_single, symbol_single, mtf_single, candles_single))
            st.session_state["single_result"] = res
            st.success("✅ Single scan complete")


    # --- Show Result ---
    res = st.session_state.get("single_result")
    if res:
        st.markdown(f"### {res.get('symbol','?')} — {res.get('signal','?')} ({round(float(res.get('confidence',0)),1)}%)")

        colL, colR = st.columns([0.55, 0.45])
        with colL:
            st.markdown("#### 🧠 Signal Breakdown")
            st.json(res.get("reasons") or {"info":"No reasoning available"})

            st.markdown("#### 🧾 Trading Plan")
            st.write(f"**Entry:** {res.get('entry')}")
            st.write(f"**TP1:** {res.get('tp1')} | **TP2:** {res.get('tp2')}")
            st.write(f"**SL:** {res.get('sl')}")
            st.write(f"**Support:** {res.get('support')} | **Resistance:** {res.get('resistance')}")

        with colR:
            st.markdown("#### 📈 Chart")
            dfc = decode_ohlcv_z(res.get("ohlcv_z"))
            if isinstance(dfc, pd.DataFrame) and not dfc.empty:
                fig = make_chart(dfc)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No chart data available.")

# ===================================================
# TAB 3 — Sentiment
# ===================================================
with tab_sentiment:

    st.header("📊 Binance Futures Sentiment Monitor")
    st.caption("Funding Rate, Open Interest, dan Long/Short Ratio (real-time, auto-refresh tiap 5 menit).")

    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        data = fetch_binance_sentiment(sym)
        plot_sentiment(sym.replace("USDT", ""), data)
        st.markdown("---")

    st.header("📅 Important Chart Narrative")
        
    # HTML code untuk economic calendar
    narative_html = """
    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <iframe 
            src="https://dexu.ai/narratives" 
            width="100%" 
            height="800" 
            frameborder="0" 
            allowtransparency="true" 
            marginwidth="0" 
            marginheight="0">
        </iframe>
        <div class="poweredBy" style="font-family: Arial, Helvetica, sans-serif; margin-top: 10px;">
            <span style="font-size: 11px;color: #333333;text-decoration: none;">
                Real Time Economic Calendar provided by 
                <a href="https://dexu.ai/narratives" rel="nofollow" target="_blank" style="font-size: 11px;color: #06529D; font-weight: bold;" class="underline_link">Investing.com</a>.
            </span>
        </div>
    </div>
    """
        
    # Tampilkan HTML di Streamlit
    st.components.v1.html(narative_html, height=1000)


    st.header("📅 Important News Sentiment - Investing.com")
        
    # HTML code untuk economic calendar
    calendar_html = """
    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <iframe 
            src="https://sslecal2.investing.com?columns=exc_flags,exc_currency,exc_importance,exc_actual,exc_forecast,exc_previous&category=_employment,_economicActivity,_inflation,_credit,_centralBanks,_confidenceIndex,_balance,_Bonds&features=datepicker,timezone&countries=5&calType=week&timeZone=27&lang=1" 
            width="100%" 
            height="800" 
            frameborder="0" 
            allowtransparency="true" 
            marginwidth="0" 
            marginheight="0">
        </iframe>
        <div class="poweredBy" style="font-family: Arial, Helvetica, sans-serif; margin-top: 10px;">
            <span style="font-size: 11px;color: #333333;text-decoration: none;">
                Real Time Economic Calendar provided by 
                <a href="https://www.investing.com/" rel="nofollow" target="_blank" style="font-size: 11px;color: #06529D; font-weight: bold;" class="underline_link">Investing.com</a>.
            </span>
        </div>
    </div>
    """
        
    # Tampilkan HTML di Streamlit
    st.components.v1.html(calendar_html, height=800)

st.caption("v4.9.1 — Dual Tab UI (Market + Single) with unified chart renderer")
