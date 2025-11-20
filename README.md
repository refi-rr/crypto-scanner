**#🚀 AI-Augmented Crypto Futures Scanner (v2.3)**

A high-performance, asynchronous quantitative trading engine designed to detect high-probability setups in Crypto Futures markets. It features multi-timeframe analysis, weighted signal fusion, real-time observability, and a local AI assistant for market context.

---

**##📋 Table of Contents**
- [System Overview](#-system-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Module Breakdown](#-module-breakdown)
- [Installation & Usage](#-installation--usage)
- [Observability & Metrics](#-observability--metrics)

---

**## 🔍 System Overview**

This is not a simple indicator scanner. It is a **Decision Support System (DSS)** that aggregates data from Binance/Bybit Futures, processes it through a weighted consensus engine, and presents actionable intelligence via a Streamlit Dashboard. It integrates **Local LLMs (DeepSeek/Qwen)** via Ollama to provide qualitative analysis alongside quantitative metrics.

---

**##✨ Key Features**

### 1. Multi-Timeframe Signal Fusion
The engine analyzes `1h`, `4h`, and `1d` timeframes simultaneously to find confluence.
- **Weighted Consensus:** Combines EMA Trend, MACD, RSI, StochRSI, OBV, and Supertrend with dynamic weights.
- **Regime Detection:** Automatically classifies market state (Trending, Ranging, High Volatility) using ADX and Bollinger Band Width to adjust confidence scores.

###2. "Anti-Rungkad" Risk Management (Smart Filters)
Built-in defensive logic to filter out dangerous setups before they reach the dashboard:
- **Overextension Block:** Rejects signals if price deviation from EMA is > 2 ATR (preventing FOMO entries).
- **Trap Detection:** Identifies long wicks (liquidation traps) and oversized candles.
- **Smart Trade Plan:** Auto-calculates Entry, Stop Loss (ATR-based), and dual Take Profit targets with estimated ETA.

###3. AI-Powered Analyst (Local RAG)
- **Context-Aware Chat:** Integrated chat interface using **Ollama**.
- **Market Injection:** Feeds real-time Funding Rates and Open Interest data into the LLM prompt for grounded analysis.
- **Persistent Memory:** Stores conversation history in SQLite (`ai_memory.db`).

### 4. Enterprise-Grade Observability
- **Tracing:** Integrated with **Jaeger (OpenTelemetry)** to trace execution bottlenecks per symbol.
- **Resource Monitoring:** Tracks CPU (core equivalent) and RAM usage per scan cycle, logging performance to SQLite.

---

**##📂 Module Breakdown**

Core Engine (scanner/)
scanner_core.py: The orchestrator. Manages the async event loop, triggers data fetching, runs the signal pipeline, and calculates Trade Plans (Entry/SL/TP).

signals_v4.py: The "Brain". Contains the weighted scoring logic (evaluate_signals_mtf) and the aggressive overextension filters.

indicators.py: The "Math". Calculates technical indicators (Heikin Ashi, Supertrend, ADX) and detects 12+ Candlestick Patterns (Hammer, Engulfing, etc.).

trend_scoring.py: Specialized logic for grading trend strength (-100 to +100).

datafetch.py: Robust async data retriever with HTTP fallback to Binance Vision if CCXT fails.

Infrastructure
run_scheduler.py: Cron-like scheduler that triggers scans at specific 4-hour candle closes (07:00, 11:00, etc.).

tracing_setup.py: Configures OpenTelemetry for Jaeger integration.

resource_monitor.py: Tracks system performance (CPU/RAM) to prevent server overload.

User Interface (ui/)
streamlit_app.py: The frontend. Features interactive Plotly charts, signal tables with filtering, and system status indicators.

ai_chat.py: Handles the interaction with the local LLM and manages chat history database.

**##🚀 Installation & Usage**
**Prerequisites**
Python 3.9+
Ollama (running locally on port 11434)
Jaeger Agent (optional, for tracing)

**Setup**
Install Dependencies:
pip install -r requirements.txt
Key deps: ccxt, pandas, streamlit, plotly, opentelemetry-api, psutil, aiohttp

Run the Scanner (Background):
python -m scanner.run_scheduler

Launch the Dashboard:
streamlit run ui/streamlit_app.py

**##📊 Observability & Metrics**
The system logs performance data to data/scanner_metrics.db.
CPU Score: CPU usage is converted to "Core Equivalent" to measure load accurately on multi-core VPS.
Tracing: If configured, traces are sent to 172.24.0.2:6831 (Jaeger Host).
