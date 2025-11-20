# 🚀 AI-Augmented Crypto Futures Scanner (v2.3)

A high-performance, asynchronous quantitative trading engine designed to detect high-probability setups in Crypto Futures markets. It features multi-timeframe analysis, weighted signal fusion, real-time observability, and a local AI assistant for market context.

---

## 📋 Table of Contents
- [System Overview](#-system-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Module Breakdown](#-module-breakdown)
- [Installation & Usage](#-installation--usage)
- [Observability & Metrics](#-observability--metrics)

---

## 🔍 System Overview

This is not a simple indicator scanner. It is a **Decision Support System (DSS)** that aggregates data from Binance/Bybit Futures, processes it through a weighted consensus engine, and presents actionable intelligence via a Streamlit Dashboard. It integrates **Local LLMs (DeepSeek/Qwen)** via Ollama to provide qualitative analysis alongside quantitative metrics.

---

## ✨ Key Features

### 1. Multi-Timeframe Signal Fusion
The engine analyzes `1h`, `4h`, and `1d` timeframes simultaneously to find confluence.
- **Weighted Consensus:** Combines EMA Trend, MACD, RSI, StochRSI, OBV, and Supertrend with dynamic weights.
- **Regime Detection:** Automatically classifies market state (Trending, Ranging, High Volatility) using ADX and Bollinger Band Width to adjust confidence scores.

### 2. "Anti-Rungkad" Risk Management (Smart Filters)
Built-in defensive logic to filter out dangerous setups before they reach the dashboard:
- **Overextension Block:** Rejects signals if price deviation from EMA is > 2 ATR (preventing FOMO entries).
- **Trap Detection:** Identifies long wicks (liquidation traps) and oversized candles.
- **Smart Trade Plan:** Auto-calculates Entry, Stop Loss (ATR-based), and dual Take Profit targets with estimated ETA.

### 3. AI-Powered Analyst (Local RAG)
- **Context-Aware Chat:** Integrated chat interface using **Ollama**.
- **Market Injection:** Feeds real-time Funding Rates and Open Interest data into the LLM prompt for grounded analysis.
- **Persistent Memory:** Stores conversation history in SQLite (`ai_memory.db`).

### 4. Enterprise-Grade Observability
- **Tracing:** Integrated with **Jaeger (OpenTelemetry)** to trace execution bottlenecks per symbol.
- **Resource Monitoring:** Tracks CPU (core equivalent) and RAM usage per scan cycle, logging performance to SQLite.

---

## 🏗 Architecture

```mermaid
graph TD
    Ex[Exchange (Binance/Bybit)] -->|Async Fetch| DF[DataFetch Module]
    DF -->|OHLCV Data| Ind[Indicators Engine]
    Ind -->|Processed DF| Sig[Signal Fusion (v4)]
    
    subgraph Logic Core
    Sig --> Trend[Trend Scoring]
    Sig --> Risk[Risk/Overextension Filter]
    end
    
    Logic Core -->|JSON Result| DB[(SQLite / JSON)]
    
    subgraph UI Layer
    DB --> ST[Streamlit Dashboard]
    User -->|Chat Prompt| AI[AI Module (Ollama)]
    AI -->|Analysis| ST
    end
    
    subgraph Monitoring
    Res[Resource Monitor] -->|Metrics| DB
    Trace[OpenTelemetry] -->|Traces| Jaeger
    end



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
