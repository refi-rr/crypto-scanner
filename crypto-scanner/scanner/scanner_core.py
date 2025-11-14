import asyncio
import logging
import time
import zlib
import base64
import traceback
from typing import Optional

import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
import aiohttp

from .datafetch import fetch_ohlcv_async
from .indicators import compute_indicators
from . import signals_v4

# ===== TRACING FIX =====
from scanner.tracing_setup import init_tracer
tracer = init_tracer(
    service_name="crypto-scanner",
    agent_host="172.24.0.2",
    agent_port=6831
)

from opentelemetry.trace import Status, StatusCode
from opentelemetry.context import attach, detach, get_current

logger = logging.getLogger(__name__)


# ============================================================
# FETCH SYMBOLS (RELIABLE TOP-N FROM BINANCE/BYBIT REST API)
# ============================================================

async def fetch_binance_futures_symbols():
    """
    Fetch list USDT futures perpetual from Binance Futures API.
    More reliable than ccxt.load_markets().
    """
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=10) as r:
                js = await r.json()

        symbols = []
        for sym in js["symbols"]:
            if sym.get("contractType") == "PERPETUAL" and sym.get("quoteAsset") == "USDT":
                symbols.append(sym["symbol"].replace("USDT", "/USDT"))

        return symbols

    except Exception as e:
        logger.error(f"[fetch_binance_futures_symbols] ERROR: {e}")
        logger.error(traceback.format_exc())
        return []


async def fetch_bybit_futures_symbols():
    """
    Fetch list USDT linear perpetual from Bybit API.
    """
    url = "https://api.bybit.com/v5/market/instruments-info?category=linear"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=10) as r:
                js = await r.json()

        arr = js.get("result", {}).get("list", [])
        symbols = []
        for d in arr:
            if d.get("quoteCoin") == "USDT":
                symbols.append(d["symbol"].replace("USDT", "/USDT"))

        return symbols

    except Exception as e:
        logger.error(f"[fetch_bybit_futures_symbols] ERROR: {e}")
        logger.error(traceback.format_exc())
        return []


# ============================================================
# COMPRESS OHLCV
# ============================================================

def compress_ohlcv(df: Optional[pd.DataFrame]) -> str:
    if df is None or df.empty:
        return ""
    try:
        df2 = df.copy()
        if df2.index.name is not None:
            df2 = df2.reset_index()
        js = df2.to_json(orient="split", date_format="iso")
        comp = zlib.compress(js.encode("utf-8"))
        return base64.b64encode(comp).decode("ascii")
    except Exception:
        return ""


# ============================================================
# MULTI TIMEFRAME BUILDER
# ============================================================

async def build_mtf_dfs_async(exchange, symbol, mtf, limit_ohlcv=500, delay=0.15):
    dfs = {}
    for tf in mtf:
        try:
            df = await fetch_ohlcv_async(exchange, symbol, tf, limit_ohlcv)
            if df is None or df.empty:
                dfs[tf] = None
                await asyncio.sleep(delay)
                continue

            df = compute_indicators(df)
            dfs[tf] = df
            await asyncio.sleep(delay)

        except Exception as e:
            logger.warning(f"[MTF] {symbol} {tf} failed: {e}")
            dfs[tf] = None
    return dfs


# ============================================================
# TRADE PLAN GENERATOR
# ============================================================

def _compute_trade_plan_from_df(df: pd.DataFrame, signal: str):
    try:
        if df is None or df.empty:
            return None

        last = df.iloc[-1]
        last_close = float(last.get("close", np.nan))
        if np.isnan(last_close):
            return None

        win = min(50, max(5, int(len(df) / 4)))
        support = float(df["low"].rolling(win, min_periods=1).min().iloc[-1])
        resistance = float(df["high"].rolling(win, min_periods=1).max().iloc[-1])

        atr = float(df["close"].diff().abs().rolling(14, min_periods=1).mean().iloc[-1])
        if np.isnan(atr) or atr <= 0:
            atr = max(
                (resistance - support) * 0.02 if (resistance > support) else last_close * 0.01,
                1e-6
            )

        entry = last_close

        if signal == "LONG":
            sl = entry - 1.5 * atr
            tp1 = entry + 1.0 * atr
            tp2 = entry + 3.0 * atr
        elif signal == "SHORT":
            sl = entry + 1.5 * atr
            tp1 = entry - 1.0 * atr
            tp2 = entry - 3.0 * atr
        else:
            return {
                "entry": None, "tp1": None, "tp2": None, "sl": None,
                "support": support, "resistance": resistance
            }

        if signal == "LONG":
            tp1 = min(tp1, resistance - atr * 0.1)
            tp2 = min(tp2, resistance)
            sl = max(sl, support - atr * 0.2)
        elif signal == "SHORT":
            tp1 = max(tp1, support + atr * 0.1)
            tp2 = max(tp2, support)
            sl = min(sl, resistance + atr * 0.2)

        def _r(v):
            try: return float(round(v, 8))
            except: return None

        return {
            "entry": _r(entry),
            "tp1": _r(tp1),
            "tp2": _r(tp2),
            "sl": _r(sl),
            "support": _r(support),
            "resistance": _r(resistance)
        }

    except Exception as e:
        logger.debug(f"Trade plan compute error: {e}")
        return None


# ============================================================
# MAIN SCANNER
# ============================================================

async def scan_async(exchange_name="binance", top_n=50, timeframe="1h",
                     limit_ohlcv=500, delay_between_requests=0.15,
                     mtf=None, concurrency=10):

    if mtf is None:
        mtf = [timeframe, "4h", "1d"]

    logger.info(f"🔍 Starting scan on {exchange_name.upper()} ({top_n} symbols) ...")
    t0 = time.time()

    # -------------------------------
    # 1. Get list of top symbols
    # -------------------------------
    try:
        if exchange_name.lower() == "binance":
            symbols_all = await fetch_binance_futures_symbols()

        elif exchange_name.lower() == "bybit":
            symbols_all = await fetch_bybit_futures_symbols()

        else:
            raise ValueError("Unsupported exchange")

        if not symbols_all:
            raise RuntimeError("Symbol list is empty")

        symbols = symbols_all[:top_n]

    except Exception as e:
        logger.error(f"[symbol_fetch] FAILED: {e}")
        fallback = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","ADA/USDT",
                    "XRP/USDT","DOGE/USDT","DOT/USDT","LINK/USDT","LTC/USDT"]
        symbols = fallback[:top_n]

    # -------------------------------
    # 2. Create exchange for OHLCV
    # -------------------------------
    try:
        if exchange_name.lower() == "binance":
            exchange = ccxt.binance({
                "options": {"defaultType": "future"},
                "enableRateLimit": True
            })
        elif exchange_name.lower() == "bybit":
            exchange = ccxt.bybit({"enableRateLimit": True})
    except Exception as e:
        logger.error(f"Failed to create exchange {exchange_name}: {e}")
        return {"results": [], "metrics": {"error": str(e)}}

    sem = asyncio.Semaphore(concurrency)
    results = []
    current_ctx = get_current()

    # -------------------------------
    # 3. Process per symbol
    # -------------------------------
    async def process_symbol(sym):
        async with sem:
            with tracer.start_as_current_span(f"scan_symbol_{sym}") as span:
                span.set_attribute("symbol", sym)

                try:
                    dfs = await build_mtf_dfs_async(
                        exchange, sym, mtf, limit_ohlcv, delay_between_requests
                    )

                    span.set_attribute("mtf_loaded", str(list(dfs.keys())))

                    primary_df = None
                    for tf in mtf:
                        dfcand = dfs.get(tf)
                        if dfcand is not None and not dfcand.empty:
                            primary_df = dfcand
                            break

                    if primary_df is None:
                        span.set_status(Status(StatusCode.ERROR, "no_valid_data"))
                        results.append({
                            "symbol": sym, "signal": "NEUTRAL", "confidence": 0,
                            "entry": None, "tp1": None, "tp2": None, "sl": None,
                            "support": None, "resistance": None,
                            "reasons": ["no_valid_data"],
                            "meta": {}, "ohlcv_z": ""
                        })
                        return

                    sig = signals_v4.evaluate_signals_mtf(dfs)
                    span.set_attribute("signal", sig.get("signal"))
                    span.set_attribute("confidence", sig.get("confidence"))

                    final_signal = sig.get("signal", "NEUTRAL")
                    confidence = sig.get("confidence", 0)

                    if confidence == 0 and final_signal != "NEUTRAL":
                        confidence = 40.0

                    trade_plan = _compute_trade_plan_from_df(primary_df, final_signal)
                    ohlcv_z = compress_ohlcv(primary_df)

                    res = {
                        "symbol": sym,
                        "signal": final_signal,
                        "confidence": float(confidence),
                        "reasons": sig.get("reasons"),
                        "meta": sig.get("meta"),
                        "ohlcv_z": ohlcv_z,
                    }

                    if trade_plan:
                        res.update(trade_plan)
                    else:
                        res.update({
                            "entry": None, "tp1": None, "tp2": None, "sl": None,
                            "support": None, "resistance": None,
                        })

                    results.append(res)
                    span.set_status(Status(StatusCode.OK))

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    results.append({
                        "symbol": sym,
                        "signal": "NEUTRAL",
                        "confidence": 0,
                        "entry": None, "tp1": None, "tp2": None, "sl": None,
                        "support": None, "resistance": None,
                        "reasons": [str(e)],
                        "meta": {}, "ohlcv_z": ""
                    })

    await asyncio.gather(*[process_symbol(s) for s in symbols])

    await exchange.close()
    t1 = time.time()

    return {
        "results": results,
        "metrics": {
            "exchange": exchange_name,
            "total": len(symbols),
            "success": len(results),
            "total_sec": round(t1 - t0, 2),
            "avg_latency": round((t1 - t0) / max(len(symbols), 1), 3),
            "concurrency": concurrency,
        }
    }


# ============================================================
# SYNC WRAPPER
# ============================================================

def scan(exchange, top_n=50, timeframe="1h",
         limit_ohlcv=500, delay_between_requests=0.15, mtf=None):

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(
            scan_async(
                exchange_name=exchange,
                top_n=top_n,
                timeframe=timeframe,
                limit_ohlcv=limit_ohlcv,
                delay_between_requests=delay_between_requests,
                mtf=mtf,
            )
        )
    except Exception as e:
        logger.error(f"scan() failed: {e}")
        return {"results": [], "metrics": {"error": str(e)}}
