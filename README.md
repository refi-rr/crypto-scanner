Crypto Scanner with features:


| Kategori               | Indikator Utama                                     | Fungsi                                      | Status           |
| ---------------------- | --------------------------------------------------- | ------------------------------------------- | ---------------- |
| **Trend**              | EMA9, EMA21, EMA50, EMA200                          | Deteksi arah jangka pendek-menengah-panjang | ✅ Sudah          |
| **Momentum**           | MACD (MACD line, Signal line, Histogram)            | Konfirmasi momentum dan cross signal        | ✅ Sudah          |
| **Strength**           | ADX                                                 | Mengukur kekuatan trend (filter sideways)   | ✅ Sudah          |
| **Volatilitas**        | ATR                                                 | Untuk hitung TP/SL dinamis                  | ✅ Sudah          |
| **Support/Resistance** | Rolling min/max 20 candle                           | Untuk deteksi level psikologis              | ✅ Sudah          |
| **Oscillator**         | RSI                                                 | Deteksi overbought/oversold                 | ✅ Sudah sebagian |
| **Multi-Timeframe**    | Perbandingan sinyal antar TF (mis. 15m, 1h, 4h, 1d) | Konfirmasi makro–mikro trend                | ✅ Sudah          |

**Kemampuan trading/analisis**

Aplikasi bisa melakukan:

🔍 Real-time market scanning top 50–200 pair futures.

📊 Multi-timeframe analysis (misalnya 1h + 4h + 1d confluence).

💡 Sinyal otomatis:

LONG jika banyak indikator bullish selaras,

SHORT jika bearish dominan,

Confidence (%) = tingkat keselarasan antar indikator/timeframe.

🎯 Level manajemen risiko otomatis:

Entry = harga terakhir.

TP1/TP2 & SL dihitung berbasis ATR.

Support/Resistance dari rolling 20 candle terakhir.

💾 Logging sinyal historis ke database + backtest sederhana.

📑 Portfolio untuk menyimpan sinyal yang kamu anggap valid.

📈 Visualisasi interaktif: candlestick, EMA, MACD, RSI, dll, di Streamlit.

⚙️ 4. Keunggulan teknis

Async & resilient fetching (aiohttp + retry + fallback → ccxt).

Caching hasil scan 5 menit (@st.cache_data).

Auto database init tanpa perlu migrasi manual.

UI modular dan responsive dengan pagination dan gradient card.

Fail-safe: tiap tahap (fetch, compute, save) punya try/except lengkap.
