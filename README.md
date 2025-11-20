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
    Ex["Exchange (Binance/Bybit)"] -->|Async Fetch| DF["DataFetch Module"]
    DF -->|"OHLCV Data"| Ind["Indicators Engine"]
    Ind -->|"Processed DF"| Sig["Signal Fusion (v4)"]
    
    subgraph Logic Core
    Sig --> Trend["Trend Scoring"]
    Sig --> Risk["Risk/Overextension Filter"]
    end
    
    Logic Core -->|"JSON Result"| DB[("SQLite / JSON")]
    
    subgraph UI Layer
    DB --> ST["Streamlit Dashboard"]
    User -->|"Chat Prompt"| AI["AI Module (Ollama)"]
    AI -->|Analysis| ST
    end
    
    subgraph Monitoring
    Res["Resource Monitor"] -->|Metrics| DB
    Trace["OpenTelemetry"] -->|Traces| Jaeger
    end

