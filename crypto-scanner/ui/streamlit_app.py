"""
ui/streamlit_app.py — v4.7.4 Lightweight UX
- Pagination (page_size=10)
- Lazy chart rendering (only when user clicks Show Chart)
- Decodes compressed 'ohlcv_z' field
- Session restore + gradient cards preserved
"""

import warnings, sys, os, json, io, time, traceback, math, zlib, base64
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# suppress details
warnings.filterwarnings("ignore")
st.set_option("client.showErrorDetails", False)

# ensure project root import
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scanner.scanner_core import scan
from scanner.datafetch import test_ping_binance_fapi
try:
    from scanner.history import list_signals
except Exception:
    list_signals = None

# storage
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_FILE = DATA_DIR / "portfolio.json"
TMP_RESULTS_FILE = "/tmp/crypto_scanner_results.json"

def load_last_results():
    if os.path.exists(TMP_RESULTS_FILE):
        try:
            with open(TMP_RESULTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_last_results(pkt):
    try:
        with open(TMP_RESULTS_FILE, "w") as f:
            json.dump(pkt, f)
    except Exception:
        pass

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

def decode_ohlcv_z(b64z: str) -> pd.DataFrame:
    """Decode base64 zlib compressed DataFrame JSON -> pandas DataFrame"""
    if not b64z:
        return None
    try:
        zb = base64.b64decode(b64z.encode("ascii"))
        j = zlib.decompress(zb).decode("utf-8")
        df = pd.read_json(j, orient="split")
        # ensure datetime index
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        return df
    except Exception:
        return None

# page
st.set_page_config(page_title="Crypto Scanner v4.7.4", layout="wide")
st.title("🚀 Crypto Futures Scanner — v4.7.4 Lightweight UX")

CARD_CSS = """
<style>
.signal-card { border-radius: 12px; padding: 12px; color: white; margin-bottom: 8px; box-shadow: 0 6px 18px rgba(0,0,0,0.22); }
.small-muted { color: #bdbdbd; font-size:13px }
.icon-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:8px; }
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

# controls
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
        delay = st.number_input("Delay between requests (s)", 0.0, 2.0, 0.12, 0.01)
    mtf = st.multiselect("Multi-timeframe", ["15m", "1h", "4h", "12h", "1d"], default=[timeframe, "4h", "1d"])

col_t1, _ = st.columns([1,4])
with col_t1:
    if st.button("🔁 Ping Binance FAPI"):
        st.info("Pinging fapi.binance.com ...")
        ok = test_ping_binance_fapi()
        if ok:
            st.success("✅ Binance FAPI reachable")
        else:
            st.warning("⚠️ Binance FAPI not reachable — fallback used")

st.write("---")

# session restore
if "scan_packet" not in st.session_state or not st.session_state["scan_packet"]:
    cached = load_last_results()
    if cached:
        st.session_state["scan_packet"] = cached
        st.info("🔄 Restored last scan results (session recovery)")

# scan button
if st.button("🚀 Start Manual Scan", use_container_width=True):
    with st.spinner("Scanning — please wait..."):
        try:
            pkt = scan(exchange, top_n=top_n, timeframe=timeframe, limit_ohlcv=candles, delay_between_requests=delay, mtf=mtf)
            if isinstance(pkt, dict) and "results" in pkt:
                st.session_state["scan_packet"] = pkt
                save_last_results(pkt)
                st.success("✅ Scan complete")
            elif isinstance(pkt, list):
                st.session_state["scan_packet"] = {"results": pkt, "metrics": {}}
                save_last_results(st.session_state["scan_packet"])
                st.success("✅ Scan complete (legacy)")
            else:
                st.error("Unexpected scan return type")
        except Exception as e:
            st.error(f"Scan failed: {e}")
            st.code(traceback.format_exc())

# display
pkt = st.session_state.get("scan_packet")
if not pkt:
    st.info("No results yet. Click Start Manual Scan.")
else:
    results = pkt.get("results", []) if isinstance(pkt, dict) else pkt
    metrics = pkt.get("metrics", {}) if isinstance(pkt, dict) else {}

    # pagination
    page_size = st.sidebar.selectbox("Signals per page", [5,10,15,20], index=1)
    total_pages = max(1, math.ceil(len(results) / page_size))
    page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = start + page_size
    page_results = results[start:end]

    # metrics
    if metrics:
        st.info(f"⏱️ Duration: {metrics.get('total_sec','?')}s | Avg latency: {metrics.get('avg_latency','?')}s | Concurrency: {metrics.get('concurrency','?')} | Success: {metrics.get('success','?')}/{metrics.get('total','?')}")

    # compact summary (full table but paged rendering)
    df = pd.DataFrame(results)
    for c in ["entry","tp1","tp2","sl","support","resistance","confidence"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["confidence"] = df.get("confidence", 0).fillna(0)
    df = df.fillna("")
    st.markdown("### 📋 Compact Summary (full)")
    # show only current page rows for table too (lighter)
    st.dataframe(df.iloc[start:end].reset_index(drop=True), use_container_width=True, height=240)

    st.write("---")
    st.markdown(f"## 🎯 Detailed Signals — page {page}/{total_pages}")

    # render only page_results (light)
    for r in page_results:
        if not r or "symbol" not in r:
            continue
        symbol = r.get("symbol")
        signal = r.get("signal", "NEUTRAL")
        confidence = r.get("confidence", 0) or 0
        entry = r.get("entry"); tp1 = r.get("tp1"); tp2 = r.get("tp2"); sl = r.get("sl")
        support = r.get("support"); resistance = r.get("resistance")
        reasons = r.get("reasons") or []

        if signal == "LONG":
            grad = "linear-gradient(90deg,#006f57,#00c67a)"
        elif signal == "SHORT":
            grad = "linear-gradient(90deg,#b22222,#ff7043)"
        else:
            grad = "linear-gradient(90deg,#546e7a,#37474f)"

        st.markdown(f"""
        <div class="signal-card" style="background:{grad}">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div><b>{symbol}</b> — {signal}</div>
            <div><b>{round(float(confidence),1)}%</b></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"{symbol} — Trading Plan & Chart", expanded=False):
            left, right = st.columns([0.55, 0.45])
            with left:
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
                st.markdown("**Reasons:**")
                for rs in reasons[:6]:
                    st.write(f"- {rs}")
                if st.button("💾 Save to Portfolio", key=f"save_{symbol}"):
                    ok = save_to_portfolio({
                        "ts": int(time.time()), "symbol": symbol, "signal": signal,
                        "confidence": confidence, "entry": entry, "tp1": tp1, "tp2": tp2,
                        "sl": sl, "support": support, "resistance": resistance
                    })
                    st.success("✅ Saved" if ok else "❌ Failed")

            with right:
                # checkbox lazy load for chart
                show_chart = st.checkbox("📈 Show multi-panel chart (lazy)", key=f"chart_chk_{symbol}")
                if not show_chart:
                    st.caption("Chart hidden (check to render).")
                else:
                    b64z = r.get("ohlcv_z")
                    dfc = None
                    if b64z:
                        dfc = decode_ohlcv_z(b64z)
                    if dfc is None:
                        st.warning("No cached OHLCV snapshot available for this symbol.")
                    else:
                        try:
                            # ensure numeric
                            for c in ["open","high","low","close","volume"]:
                                if c in dfc.columns:
                                    dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
                            X = dfc.index
                            fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                                                row_heights=[0.42,0.12,0.12,0.16,0.18],
                                                vertical_spacing=0.04,
                                                subplot_titles=("Price","RSI","Stoch RSI","MACD","Volume"))
                            fig.add_trace(go.Candlestick(x=X, open=dfc["open"], high=dfc["high"],
                                                         low=dfc["low"], close=dfc["close"], name="Price"), row=1, col=1)
                            for ema in ["ema_9","ema_21","ema_50","ema_200"]:
                                if ema in dfc.columns:
                                    fig.add_trace(go.Scatter(x=X, y=dfc[ema], name=ema.upper(), line=dict(width=1.2)), row=1, col=1)
                            for key, style in [("entry","solid"),("tp1","dash"),("tp2","dot"),("sl","dash")]:
                                v = r.get(key)
                                if v: fig.add_hline(y=v, line=dict(dash=style,width=1.2), row=1, col=1)
                            if "rsi" in dfc.columns:
                                fig.add_trace(go.Scatter(x=X, y=dfc["rsi"], name="RSI"), row=2, col=1)
                                fig.add_hline(y=70, line=dict(dash="dash"), row=2, col=1)
                                fig.add_hline(y=30, line=dict(dash="dash"), row=2, col=1)
                            if "stoch_k" in dfc.columns and "stoch_d" in dfc.columns:
                                fig.add_trace(go.Scatter(x=X, y=dfc["stoch_k"]*100, name="Stoch K"), row=3, col=1)
                                fig.add_trace(go.Scatter(x=X, y=dfc["stoch_d"]*100, name="Stoch D"), row=3, col=1)
                            if "macd" in dfc.columns and "macd_signal" in dfc.columns:
                                hist = (dfc["macd"] - dfc["macd_signal"]).fillna(0)
                                fig.add_trace(go.Bar(x=X, y=hist, name="Hist"), row=4, col=1)
                                fig.add_trace(go.Scatter(x=X, y=dfc["macd"], name="MACD"), row=4, col=1)
                                fig.add_trace(go.Scatter(x=X, y=dfc["macd_signal"], name="Signal"), row=4, col=1)
                            if "volume" in dfc.columns:
                                fig.add_trace(go.Bar(x=X, y=dfc["volume"], name="Volume"), row=5, col=1)
                            theme = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
                            fig.update_layout(height=760, template=theme, showlegend=True, margin=dict(t=30,b=10))
                            st.plotly_chart(fig, use_container_width=True)
                            # Download CSV
                            buf = io.StringIO()
                            dfc.to_csv(buf)
                            st.download_button("📥 Download CSV", buf.getvalue(), file_name=f"{symbol}_{timeframe}.csv", mime="text/csv")
                        except Exception as e:
                            st.error(f"Chart render failed: {e}")
                            st.code(traceback.format_exc())

st.write("---")
st.caption("v4.7.4 — Lightweight UX: pagination + lazy chart + compressed OHLCV")
