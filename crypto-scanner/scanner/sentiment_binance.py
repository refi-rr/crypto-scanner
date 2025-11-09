# scanner/sentiment_binance.py
"""
Sentiment Data Fetcher — Binance Futures Public API
----------------------------------------------------
Menyediakan fungsi untuk mengambil data:
- Funding Rate
- Open Interest
- Global Long/Short Ratio
dari Binance Futures API (tanpa API key).

Dipakai di UI (Streamlit) sebagai tab "Sentiment".
"""

import requests
import plotly.graph_objects as go
import streamlit as st


@st.cache_data(ttl=300)
def fetch_binance_sentiment(symbol="BTCUSDT"):
    """
    Ambil data sentimen futures dari Binance (Funding Rate, Open Interest, Long/Short Ratio)
    ttl=300 artinya cache 5 menit
    """
    base = "https://fapi.binance.com"
    res = {"symbol": symbol}
    try:
        # Funding Rate
        fr = requests.get(f"{base}/fapi/v1/fundingRate?symbol={symbol}&limit=1", timeout=5).json()
        res["funding_rate"] = float(fr[0]["fundingRate"]) * 100  # ubah ke persen

        # Open Interest (USDT)
        oi = requests.get(f"{base}/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=1", timeout=5).json()
        res["open_interest"] = float(oi[0]["sumOpenInterestValue"])

        # Long/Short Ratio
        ls = requests.get(f"{base}/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m&limit=1", timeout=5).json()
        res["long_short_ratio"] = float(ls[0]["longShortRatio"])
        res["long_pct"] = float(ls[0]["longAccount"]) * 100
        res["short_pct"] = float(ls[0]["shortAccount"]) * 100

    except Exception as e:
        res["error"] = str(e)
    return res


def plot_sentiment(symbol, data):
    """
    Tampilkan metrik dan grafik sentimen futures.
    """
    st.subheader(f"📈 {symbol} Futures Sentiment")

    if "error" in data:
        st.error(f"Gagal memuat data {symbol}: {data['error']}")
        return

    # === Funding Rate Gauge ===
    fr = data["funding_rate"]
    st.markdown(f"**Funding Rate:** `{fr:.4f}%`")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fr,
        delta={
            'reference': 0,
            'increasing': {'color': "#16c784"},
            'decreasing': {'color': "#ea3943"}
        },
        gauge={
            'axis': {'range': [-0.1, 0.1], 'tickformat': '.2%', 'tickwidth': 1, 'tickcolor': "#444"},
            'bar': {'color': "#ffffff"},
            'bgcolor': "#111",
            'borderwidth': 1,
            'bordercolor': "#333",
            'steps': [
                {'range': [-0.1, -0.05], 'color': '#511'},
                {'range': [-0.05, 0], 'color': '#ea3943'},
                {'range': [0, 0.05], 'color': '#16c784'},
                {'range': [0.05, 0.1], 'color': '#173'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.8,
                'value': fr
            }
        },
        title={'text': "Funding Rate Gauge", 'font': {'size': 16}}
    ))

    fig_gauge.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=0),
        template="plotly_dark",
        transition={'duration': 500}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # === Open Interest ===
    st.markdown(f"**Open Interest:** `{data['open_interest'] / 1e6:.2f} M USDT`")

    # === Long vs Short Ratio ===
    ratio = data["long_short_ratio"]
    long_pct = data["long_pct"]
    short_pct = data["short_pct"]

    st.markdown(f"**Long/Short Ratio:** `{ratio:.2f}` — Long: {long_pct:.1f}% | Short: {short_pct:.1f}%`")

    # Grafik bar Long vs Short
    fig_bar = go.Figure(go.Bar(
        x=[long_pct, short_pct],
        y=["Long", "Short"],
        orientation='h',
        marker_color=["#16c784", "#ea3943"],
        text=[f"{long_pct:.1f}%", f"{short_pct:.1f}%"],
        textposition="auto"
    ))
    fig_bar.update_layout(
        title=f"Long vs Short Distribution — {symbol}",
        xaxis_title="Percentage",
        yaxis_title="",
        template="plotly_dark",
        height=250
    )
    st.plotly_chart(fig_bar, use_container_width=True)
