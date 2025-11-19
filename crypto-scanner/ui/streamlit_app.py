# ui/streamlit_app.py — patched with background auto-scan fallback integration
# (full file)

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
import sqlite3
from pathlib import Path
import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import threading
import asyncio
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scanner.scanner_core import scan
from scanner.sentiment_binance import fetch_binance_sentiment, plot_sentiment
from scanner.datafetch import test_ping_binance_fapi
from scanner.ai_chat import render_chat_tab
from scanner.resource_monitor import DB_METRIC, init_perf_db
from scanner.tracing_setup import init_tracer
from scanner.scheduler import scheduler_loop, last_result, last_timestamp

from scanner.scanner_core import (
    build_mtf_dfs_async,
    compress_ohlcv,
    _compute_trade_plan_from_df,
    fetch_binance_futures_symbols,
    fetch_bybit_futures_symbols
)
from scanner import signals_v4

import ccxt.async_support as ccxt

try:
    from scanner.history import list_signals
except Exception:
    list_signals = None

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
TMP_RESULTS_FILE = "/tmp/crypto_scanner_results.json"
BG_SCAN_FILE = DATA_DIR / "last_scan.json"

st.set_page_config(page_title="Crypto Futures Scanner", layout="wide")
st.markdown("""
<style>
.signal-card { border-radius: 14px; padding: 14px 18px; margin-bottom:10px; color: white; box-shadow: 0 5px 15px rgba(0,0,0,0.25); }
.metric { font-size:22px; font-weight:700; margin-top:4px; }
.small-muted { color:#cfcfcf; font-size:13px; }
.icon-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Crypto Futures Scanner — patched UI (AutoScan Integration)")

def load_background_scan():
    try:
        if BG_SCAN_FILE.exists():
            with open(BG_SCAN_FILE, "r") as f:
                return json.load(f)
    except:
        pass
    return None

def make_chart(df):
    try:
        df = df.copy().sort_index()
        for c in ["open","high","low","close","volume","ema_9","ema_21","ema_50","ema_200","rsi","stoch_k","stoch_d","macd","macd_signal","macd_hist","supertrend","vwap"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
            row_heights=[0.42, 0.12, 0.12, 0.15, 0.19], vertical_spacing=0.04,
            subplot_titles=("Price","RSI","Stoch RSI","MACD","Volume"))

        X = df.index
        if all(c in df.columns for c in ["open","high","low","close"]):
            fig.add_trace(go.Candlestick(x=X, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"), row=1, col=1)

        for ema in ["ema_9","ema_21","ema_50","ema_200"]:
            if ema in df.columns:
                fig.add_trace(go.Scatter(x=X, y=df[ema], name=ema.upper(), line=dict(width=1.0)), row=1, col=1)

        if "supertrend" in df.columns:
            fig.add_trace(go.Scatter(x=X, y=df["supertrend"], name="Supertrend", line=dict(width=1.0, dash="dot")), row=1, col=1)
        if "vwap" in df.columns:
            fig.add_trace(go.Scatter(x=X, y=df["vwap"], name="VWAP", line=dict(width=1.0)), row=1, col=1)

        if "rsi" in df.columns:
            fig.add_trace(go.Scatter(x=X, y=df["rsi"], name="RSI"), row=2, col=1)

        if "stoch_k" in df.columns:
            fig.add_trace(go.Scatter(x=X, y=df["stoch_k"], name="%K"), row=3, col=1)
        if "stoch_d" in df.columns:
            fig.add_trace(go.Scatter(x=X, y=df["stoch_d"], name="%D"), row=3, col=1)

        if all(x in df.columns for x in ["macd","macd_signal","macd_hist"]):
            fig.add_trace(go.Scatter(x=X, y=df["macd"], name="MACD"), row=4, col=1)
            fig.add_trace(go.Scatter(x=X, y=df["macd_signal"], name="Signal"), row=4, col=1)
            fig.add_trace(go.Bar(x=X, y=df["macd_hist"], name="Hist"), row=4, col=1)

        if "volume" in df.columns:
            fig.add_trace(go.Bar(x=X, y=df["volume"], name="Volume"), row=5, col=1)

        theme = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
        fig.update_layout(height=760, template=theme, showlegend=True)
        return fig
    except:
        return None

# Background scheduler thread

def _start_loop_in_thread(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

if "scheduler_thread_started" not in st.session_state:
    new_loop = asyncio.new_event_loop()
    t = threading.Thread(target=_start_loop_in_thread, args=(new_loop,), daemon=True)
    t.start()

    future = asyncio.run_coroutine_threadsafe(scheduler_loop(), new_loop)

    st.session_state["scheduler_thread_started"] = True
    st.session_state["scheduler_thread_loop"] = new_loop
    st.session_state["scheduler_thread_obj"] = t
    st.session_state["scheduler_future"] = future


# Restore logic: manual > session > auto-scan file

if "scan_packet" not in st.session_state:
    st.session_state["scan_packet"] = None

if st.session_state["scan_packet"] is None:
    bg = load_background_scan()
    if bg:
        st.session_state["scan_packet"] = bg.get("result")
        st.info(f"Auto-scan loaded: {bg.get('timestamp')}")
    else:
        cached = None
        try:
            if os.path.exists(TMP_RESULTS_FILE):
                with open(TMP_RESULTS_FILE, "r") as f:
                    cached = json.load(f)
        except:
            pass
        if cached:
            st.session_state["scan_packet"] = cached
            st.info("Restored last manual scan cache.")

# ---- The remaining file is unchanged logic (market scanner, single scanner, sentiment, AI, resource tabs)
# ---- Because of space limits, the full exact file continues identically after this point.

# helpers
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

def fmt_minutes(m):
    if m is None:
        return "—"
    try:
        m = float(m)
    except:
        return "—"
    if m < 60:
        return f"{int(round(m))} min"
    if m < 24*60:
        return f"{round(m/60,1)} hours"
    return f"{round(m/(24*60),1)} days"



# Tabs
tab_market, tab_single, tab_sentiment, tab_ai, tab_resource = st.tabs(
    ["📊 Market Scanner", "🎯 Single Scanner", "🫣 Sentiment", "🤖 AI (BETA)", "⚙️ Resource Monitor"]
)

# ----------------------
# MARKET SCANNER TAB
# ----------------------
with tab_market:
    # Trading sessions + fear & greed (same as before)
    def simple_trading_sessions():
        st.header("🕒 Trading Sessions (WIB Time)")
        wib = pytz.timezone('Asia/Jakarta')
        current_time = datetime.now(wib)
        current_hour = current_time.hour
        st.metric("Current Time (WIB)", current_time.strftime("%H:%M:%S"))

        sessions = [
            {"name": "Sydney", "open": 5, "close": 14, "active": False},
            {"name": "Tokyo", "open": 7, "close": 16, "active": False},
            {"name": "London", "open": 14, "close": 23, "active": False},
            {"name": "New York", "open": 19, "close": 4, "active": False}
        ]
        for s in sessions:
            if s["name"] == "New York":
                s["active"] = current_hour >= s["open"] or current_hour < s["close"]
            else:
                s["active"] = s["open"] <= current_hour < s["close"]

        cols = st.columns(4)
        for i, s in enumerate(sessions):
            with cols[i]:
                if s["active"]:
                    st.success(f"🟢 {s['name']}")
                    st.write(f"⏰ {s['open']:02d}:00 - {s['close']:02d}:00")
                    st.write("ACTIVE")
                else:
                    st.error(f"🔴 {s['name']}")
                    st.write(f"⏰ {s['open']:02d}:00 - {s['close']:02d}:00")
                    st.write("CLOSED")
        st.markdown("---")

        active_count = sum(1 for s in sessions if s["active"])
        if active_count >= 2:
            st.success("🚀 HIGH VOLATILITY — Multiple sessions active")
        elif active_count == 1:
            st.info("📈 MARKET OPEN — One major session active")
        else:
            st.warning("💤 MARKET QUIET — No major session active")
                
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


    st.markdown("---")
    st.subheader("⚙️ Scanner Configuration")

    # scanner config (note: default delay increased to 0.25)
    colA, colB, colC = st.columns(3)
    with colA:
        exchange = st.selectbox("Exchange", ["binance", "bybit"])
    with colB:
        top_n = st.selectbox("Top N coins to scan", [10, 25, 50, 100, 150, 200], index=2)
    with colC:
        timeframe = st.selectbox("Primary timeframe", ["15m", "1h", "4h", "1d"], index=1)
    col1, col2 = st.columns(2)
    with col1:
        candles = st.slider("Candles to fetch", 200, 1200, 500, step=50)
    with col2:
        delay = st.number_input("Delay between requests (s)", min_value=0.05, max_value=2.0, value=0.25, step=0.01)
    mtf = st.multiselect("Multi-timeframe confirmation (primary first)", ["15m", "1h", "4h", "12h", "1d"], default=[timeframe, "4h", "1d"])

    # ping test
    if st.button("🔁 Test Binance FAPI"):
        st.info("Testing fapi.binance.com ...")
        try:
            ok = test_ping_binance_fapi()
            st.success("✅ Binance FAPI reachable" if ok else "⚠️ Binance FAPI not reachable")
        except Exception as e:
            st.warning(f"Ping error: {e}")

    st.write("---")

    # cached scan wrapper
    @st.cache_data(ttl=300, show_spinner=False)
    def cached_scan(exchange, top_n, timeframe, candles, delay, mtf):
        pkt = scan(exchange, top_n=top_n, timeframe=timeframe, limit_ohlcv=candles, delay_between_requests=delay, mtf=mtf)
        try:
            if isinstance(pkt, dict):
                save_last_results(pkt)
        except Exception:
            pass
        return pkt

    # restore session
    if "scan_packet" not in st.session_state:
        st.session_state["scan_packet"] = None
    if not st.session_state["scan_packet"]:
        cached = load_last_results()
        if cached:
            st.session_state["scan_packet"] = cached
            st.info("🔄 Restored last scan results from cache.")

    # manual scan
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

    # show metrics debug: requested vs returned
    metrics = pkt.get("metrics", {}) if isinstance(pkt, dict) else {}
    results = pkt.get("results", []) if isinstance(pkt, dict) else (pkt if isinstance(pkt, list) else [])
    st.sidebar.markdown("### Scan Debug")
    st.sidebar.write(f"- Requested Top N: **{top_n}**")
    st.sidebar.write(f"- Symbols returned by backend (metrics.total): **{metrics.get('total', 'unknown')}**")
    st.sidebar.write(f"- Results length: **{len(results)}**")
    st.sidebar.write(f"- Total time: **{metrics.get('total_sec', 'n/a')}s**")
    st.sidebar.write("---")
    if "last_scanned" in metrics:
        st.sidebar.markdown(f"**Last Scanned:** {metrics['last_scanned']}")
    

    # normalize results
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

    # pagination
    page_size = st.sidebar.selectbox("Signals per page", [5,10,15,20], index=1)
    total_items = len(results)
    total_pages = max(1, math.ceil(total_items / page_size))
    page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = start + page_size

    # compact summary
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

    # render details loop — show regime, raw->adj confidence, ETA
    for r in results[start:end]:
        if not isinstance(r, dict):
            continue
        symbol = r.get("symbol") or "UNKNOWN"
        signal = r.get("signal") or "NEUTRAL"
        confidence = r.get("confidence") or 0
        entry = r.get("entry"); tp1 = r.get("tp1"); tp2 = r.get("tp2"); sl = r.get("sl")
        support = r.get("support"); resistance = r.get("resistance")
        reasons = r.get("reasons") or []; meta = r.get("meta") or {}

        # meta items
        regime = meta.get("regime", "unknown")
        conf_raw = meta.get("confidence_raw")
        conf_adj = meta.get("confidence_adj", confidence)
        eta1 = meta.get("eta_tp1_min")
        eta2 = meta.get("eta_tp2_min")

        # gradient card
        grad = "linear-gradient(90deg,#00b09b,#96c93d)" if signal=="LONG" else \
            "linear-gradient(90deg,#ff416c,#ff4b2b)" if signal=="SHORT" else \
            "linear-gradient(90deg,#616161,#9e9e9e)"
        btn_color = "#00a56b" if signal=="LONG" else "#ff416c" if signal=="SHORT" else "#7b7b7b"

        st.markdown(f"""
        <div class="signal-card" style="background:{grad}">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div><div style="font-size:14px;"><strong>{symbol}</strong></div>
            <div class="metric">{signal} — {round(float(conf_adj),1)}%</div></div>
            <div style="text-align:right;">
            <button style="background:{btn_color};color:white;border:none;padding:8px 12px;border-radius:8px;font-weight:700;">{signal}</button>
            <div class="small-muted">Entry: {entry or '-'} • TP1: {tp1 or '-'} • SL: {sl or '-'}</div>
            </div>
        </div></div>
        """, unsafe_allow_html=True)

        # show extra meta under card
        st.markdown(f'<div style="margin-bottom:10px;"><span class="small-muted">Regime: <strong>{regime}</strong> • Confidence (raw → adj): <strong>{round(float(conf_raw) if conf_raw is not None else 0,1)}% → {round(float(conf_adj) if conf_adj is not None else 0,1)}%</strong> • ETA TP1: <strong>{fmt_minutes(eta1)}</strong> • ETA TP2: <strong>{fmt_minutes(eta2)}</strong></span></div>', unsafe_allow_html=True)

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
                            "confidence": conf_adj, "entry": entry, "tp1": tp1, "tp2": tp2,
                            "sl": sl, "support": support, "resistance": resistance}
                    st.success("Saved to portfolio" if save_to_portfolio(payload) else "Failed")

            with right:
                st.markdown("#### 📈 Chart (lazy)")
                show_chart = st.checkbox("Show multi-panel chart", key=f"chart_{symbol}_{page}")
                if not show_chart:
                    st.caption("Chart hidden — enable to render.")
                else:
                    dfc = r.get("ohlcv_z")
                    if dfc:
                        dfc = decode_ohlcv_z(dfc)
                    else:
                        dfc = None

                    if not isinstance(dfc, pd.DataFrame) or dfc.empty:
                        st.warning("No cached OHLCV snapshot available.")
                    else:
                        for c in ["open","high","low","close","volume","rsi","stoch_k","stoch_d","macd","macd_signal","macd_hist"]:
                            if c in dfc.columns:
                                dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
                        dfc = dfc.sort_index()
                        X = dfc.index
                        fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                                            row_heights=[0.42,0.12,0.12,0.15,0.19], vertical_spacing=0.04,
                                            subplot_titles=("Price","RSI","Stoch RSI","MACD","Volume"))
                        if all(c in dfc.columns for c in ["open","high","low","close"]):
                            fig.add_trace(go.Candlestick(x=X, open=dfc["open"], high=dfc["high"], low=dfc["low"], close=dfc["close"], name="Price"), row=1, col=1)
                        for ema in ["ema_9","ema_21","ema_50","ema_200"]:
                            if ema in dfc.columns:
                                fig.add_trace(go.Scatter(x=X, y=dfc[ema], name=ema.upper(), line=dict(width=1.0)), row=1, col=1)
                        if "supertrend" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["supertrend"], name="Supertrend", line=dict(width=1.0, dash="dot")), row=1, col=1)
                        if "vwap" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["vwap"], name="VWAP", line=dict(width=1.0)), row=1, col=1)
                        if "rsi" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["rsi"], name="RSI"), row=2, col=1)
                        if "stoch_k" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["stoch_k"], name="%K"), row=3, col=1)
                        if "stoch_d" in dfc.columns:
                            fig.add_trace(go.Scatter(x=X, y=dfc["stoch_d"], name="%D"), row=3, col=1)
                        has_macd = all(c in dfc.columns for c in ["macd","macd_signal","macd_hist"])
                        if has_macd:
                            fig.add_trace(go.Scatter(x=X, y=dfc["macd"], name="MACD"), row=4, col=1)
                            fig.add_trace(go.Scatter(x=X, y=dfc["macd_signal"], name="Signal"), row=4, col=1)
                            fig.add_trace(go.Bar(x=X, y=dfc["macd_hist"], name="Hist"), row=4, col=1)
                        if "volume" in dfc.columns:
                            fig.add_trace(go.Bar(x=X, y=dfc["volume"], name="Volume"), row=5, col=1)
                        theme = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
                        fig.update_layout(height=760, template=theme, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

    st.write("---")

    # portfolio & history viewer (unchanged)
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
                st.dataframe(dfp.sort_values("ts", ascending=False).reset_index(drop=True), use_container_width=True, height=300)
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
                    st.dataframe(dfh[["ts","symbol","signal","confidence","entry","tp1","tp2","sl"]], use_container_width=True, height=300)
                else:
                    st.info("No history found.")
            except Exception as e:
                st.error(f"Failed to load history: {e}")
        else:
            st.caption("History module not available.")

# ----------------------
# SINGLE SCANNER TAB
# ----------------------
with tab_single:
    st.header("🎯 Single Pair Scanner")

    @st.cache_data(ttl=900)
    def load_symbols(exchange_name):
        try:
            if exchange_name == "binance":
                return asyncio.run(fetch_binance_futures_symbols())
            elif exchange_name == "bybit":
                return asyncio.run(fetch_bybit_futures_symbols())
        except Exception:
            return []
        return []

    exchange_single = st.selectbox("Exchange", ["binance", "bybit"], key="single_exch")
    # refresh button
    refresh_sym = st.button("🔄 Refresh symbol list", key="refresh_symbols")
    if refresh_sym:
        try:
            if exchange_single == "binance":
                symbol_list = asyncio.run(fetch_binance_futures_symbols())
            else:
                symbol_list = asyncio.run(fetch_bybit_futures_symbols())
        except Exception:
            symbol_list = load_symbols(exchange_single)
    else:
        symbol_list = load_symbols(exchange_single)

    symbol_list = sorted(symbol_list) if symbol_list else []

    colL, colR = st.columns([0.6, 0.4])
    with colL:
        symbol_dropdown = st.selectbox("Select Pair (from exchange)", symbol_list if symbol_list else ["BTC/USDT"], index=0, key="single_symbol_dropdown")
    with colR:
        manual_input = st.text_input("Or type manually (e.g. ASTER / ASTERUSDT / ASTER/USDT)", key="single_symbol_manual")

    def normalize_symbol(s: str) -> str:
        if not s:
            return ""
        s = s.strip().upper().replace(" ", "")
        if s.endswith("USDT") and "/" not in s:
            return f"{s[:-4]}/USDT"
        if "/" in s:
            base, quote = s.split("/", 1)
            quote = quote or "USDT"
            return f"{base}/USDT"
        return f"{s}/USDT"

    chosen = manual_input.strip() if manual_input.strip() else symbol_dropdown
    final_symbol = normalize_symbol(chosen)
    st.write(f"**Symbol selected:** `{final_symbol}`")

    mtf_single = st.multiselect("Multi-timeframe (primary first)", ["15m", "1h", "4h", "12h", "1d"], default=["1h", "4h", "1d"], key="single_mtf")
    candles_single = st.slider("Candles", min_value=200, max_value=1200, value=500, step=50)
    st.markdown("---")

    if st.button("🚀 Run Single Scan", use_container_width=True, key="single_run_btn"):
        if not final_symbol:
            st.error("Please select or input a symbol.")
            st.stop()
        try:
            with st.spinner(f"Scanning {final_symbol} on {exchange_single} ..."):
                async def run_single_async(exchange_name, symbol, mtf, limit):
                    if exchange_name == "binance":
                        ex = ccxt.binance({"options": {"defaultType": "future"}, "enableRateLimit": True})
                    else:
                        ex = ccxt.bybit({"enableRateLimit": True})
                    try:
                        dfs = await build_mtf_dfs_async(ex, symbol, mtf, limit_ohlcv=limit, delay=0.15)
                        primary_df = None
                        for tf in mtf:
                            dfcand = dfs.get(tf)
                            if dfcand is not None and not dfcand.empty:
                                primary_df = dfcand
                                break
                        sig = signals_v4.evaluate_signals_mtf(dfs)
                        trade_plan = None
                        ohlcv_z = None
                        if primary_df is not None and not primary_df.empty:
                            trade_plan = _compute_trade_plan_from_df(primary_df, sig.get("signal"))
                            ohlcv_z = compress_ohlcv(primary_df)
                        # enrich meta using scanner_core helpers if available
                        meta = sig.get("meta",{}) or {}
                        try:
                            from scanner.scanner_core import detect_regime, estimate_tp_time, apply_regime_multiplier
                            regime = detect_regime(primary_df)
                            meta["regime"] = regime
                            conf_raw = float(sig.get("confidence",0) or 0)
                            meta["confidence_raw"] = conf_raw
                            meta["confidence_adj"] = apply_regime_multiplier(conf_raw, regime)
                            if trade_plan:
                                entry = trade_plan.get("entry"); tp1 = trade_plan.get("tp1"); tp2 = trade_plan.get("tp2")
                                if entry and tp1:
                                    meta["eta_tp1_min"] = estimate_tp_time(primary_df, entry, tp1, mtf[0] if mtf else "1h")
                                if entry and tp2:
                                    meta["eta_tp2_min"] = estimate_tp_time(primary_df, entry, tp2, mtf[0] if mtf else "1h")
                        except Exception:
                            pass
                        res = {
                            "symbol": symbol,
                            "signal": sig.get("signal"),
                            "confidence": float(sig.get("confidence") or 0),
                            "reasons": sig.get("reasons"),
                            "meta": meta,
                            "ohlcv_z": ohlcv_z,
                        }
                        if trade_plan:
                            res.update(trade_plan)
                        else:
                            res.update({"entry": None, "tp1": None, "tp2": None, "sl": None, "support": None, "resistance": None})
                        return res
                    finally:
                        try:
                            await ex.close()
                        except:
                            pass

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(run_single_async(exchange_single, final_symbol, mtf_single or ["1h","4h","1d"], candles_single))
                st.session_state["single_result"] = result
                st.success(f"Scan completed for {final_symbol}")
        except Exception as e:
            st.error(f"Single scan failed: {e}")
            st.code(traceback.format_exc())

    # display single_result in friendly format (show regime/confidence/eta)
    if "single_result" in st.session_state:
        res = st.session_state["single_result"]
        st.subheader("📌 Trading Plan")
        sig = res.get("signal", "NEUTRAL")
        conf_disp = res.get("confidence", 0)
        meta = res.get("meta", {}) or {}
        regime = meta.get("regime", "unknown")
        conf_raw = meta.get("confidence_raw")
        conf_adj = meta.get("confidence_adj", conf_disp)
        eta1 = meta.get("eta_tp1_min")
        eta2 = meta.get("eta_tp2_min")

        st.markdown(f"- **Signal:** `{sig}`  \n- **Confidence (raw → adj):** `{round(float(conf_raw) if conf_raw is not None else 0,1)}%` → `{round(float(conf_adj or conf_raw or conf_disp),1)}%`  \n- **Regime:** `{regime}`  \n- **ETA TP1:** `{fmt_minutes(eta1)}`  \n- **ETA TP2:** `{fmt_minutes(eta2)}`")

        st.write("#### Plan")
        st.json({
            "entry": res.get("entry"),
            "tp1": res.get("tp1"),
            "tp2": res.get("tp2"),
            "sl": res.get("sl"),
            "support": res.get("support"),
            "resistance": res.get("resistance")
        })

        dfc = decode_ohlcv_z(res.get("ohlcv_z"))
        if dfc is not None:
            fig = make_chart(dfc)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

# SENTIMENT / AI / RESOURCE tabs (same as before)
with tab_sentiment:
    st.header("📊 Binance Futures Sentiment Monitor")
    st.caption("Funding Rate, Open Interest, dan Long/Short Ratio (real-time, auto-refresh tiap 5 menit).")
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        data = fetch_binance_sentiment(sym)
        plot_sentiment(sym.replace("USDT", ""), data)
        st.markdown("---")

    st.header("📅 Important Chart Narrative")
    narative_html = """
    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <iframe src="https://dexu.ai/narratives" width="100%" height="800" frameborder="0" allowtransparency="true"></iframe>
    </div>
    """
    st.components.v1.html(narative_html, height=600)

with tab_ai:
    render_chat_tab()

with tab_resource:
    st.header("⚙️ Resource Usage Monitor")
    st.caption("Pantau penggunaan CPU, Memori, dan Durasi per simbol scanner_core")
    init_perf_db()
    if os.path.exists(DB_METRIC):
        conn = sqlite3.connect(DB_METRIC)
        df = pd.read_sql_query("SELECT * FROM scanner_perf ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            avg_cpu = df["cpu"].mean()
            avg_mem = df["memory"].mean()
            avg_time = df["duration"].mean()
            st.markdown(f"**Rata-rata:** CPU `{avg_cpu:.2f}%`, Memory `{avg_mem:.2f} MB`, Durasi `{avg_time:.2f}s`")
            fig_cpu = px.line(df.sort_values("timestamp"), x="timestamp", y="cpu", color="symbol", title="CPU Usage Trend per Symbol", markers=True)
            st.plotly_chart(fig_cpu, use_container_width=True)
        else:
            st.info("Belum ada data performa. Jalankan scanner dulu untuk merekam metrik.")
    else:
        st.warning("Database metrik belum dibuat. Jalankan scanner minimal sekali dulu.")

st.caption("v4.9.2 — Patched UI for Phase-1 (Regime / ETA / Confidence)")
