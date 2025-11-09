# scanner_core.py — v4.7.7a Fix: restore return results + proper result propagation
import asyncio
import aiohttp
import time
import random
import os
import json
import zlib
import base64
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Optional
from opentelemetry.trace import Status, StatusCode

from scanner.tracing_setup import init_tracer
tracer = init_tracer("crypto-scanner")

from scanner.datafetch import (
    create_exchange,
    fetch_top_symbols_binance_futures,
    fetch_top_symbols_bybit_futures,
)
from scanner.indicators import compute_indicators
from scanner import signals_v4

try:
    from scanner.history import init_db, log_signal
    init_db()
except Exception:
    def log_signal(x): pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_CONCURRENCY = 15
MAX_CONCURRENCY = 30
MIN_CONCURRENCY = 2
_THREAD_POOL = ThreadPoolExecutor(max_workers=14)

BINANCE_KLINES_URLS = [
    "https://fapi.binance.com/fapi/v1/klines",
    "https://data-api.binance.vision/api/v3/klines",
    "https://futures.binance.com/fapi/v1/klines",
]

RETRY_COUNT = 3
RETRY_BACKOFF_BASE = 0.5

def normalize_symbol_for_binance(sym: str) -> str:
    s = str(sym).split(":")[0].replace("/", "").upper()
    if not s.endswith("USDT"):
        s = f"{s}USDT"
    return s

def ohlcv_to_df(ohlcv: List[List[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv)
    if df.shape[1] >= 6:
        df = df.iloc[:, :6]
        df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("datetime", inplace=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# async helpers
async def fetch_http_with_retries(session: aiohttp.ClientSession, url: str, params: dict):
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                return await resp.json()
        except Exception:
            if attempt < RETRY_COUNT:
                await asyncio.sleep(RETRY_BACKOFF_BASE * 2 ** (attempt - 1))
            else:
                raise

async def fetch_symbol_klines(symbol: str, tf: str, limit: int, session: aiohttp.ClientSession):
    params = {"symbol": normalize_symbol_for_binance(symbol), "interval": tf, "limit": limit}
    for url in BINANCE_KLINES_URLS:
        try:
            return await fetch_http_with_retries(session, url, params)
        except Exception:
            await asyncio.sleep(random.uniform(0.05, 0.2))
    return None

async def fetch_ohlcv_for_symbol(exchange_name, exchange_obj, symbol, timeframe, limit, aiohttp_session):
    with tracer.start_as_current_span("fetch_ohlcv_single") as span:
        span.set_attribute("symbol", symbol)
        span.set_attribute("exchange", exchange_name)
        span.set_attribute("timeframe", timeframe)
        try:
            if exchange_name == "binance" and aiohttp_session:
                res = await fetch_symbol_klines(symbol, timeframe, limit, aiohttp_session)
                span.set_status(Status(StatusCode.OK))
                return res
            func = exchange_obj.fetch_ohlcv
            result = await asyncio.get_event_loop().run_in_executor(_THREAD_POOL, lambda: func(symbol, timeframe, limit))
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            return None

async def build_mtf_dfs_async(exchange_name, exchange_obj, symbol, timeframes, limit, aiohttp_session, delay_between_requests):
    with tracer.start_as_current_span("build_mtf_dfs") as span:
        dfs = {}
        span.set_attribute("symbol", symbol)
        for tf in timeframes:
            res = await fetch_ohlcv_for_symbol(exchange_name, exchange_obj, symbol, tf, limit, aiohttp_session)
            if not res:
                dfs[tf] = None
                continue
            try:
                df = ohlcv_to_df(res)
                df = compute_indicators(df)
                df["support"] = df["low"].rolling(20).min()
                df["resistance"] = df["high"].rolling(20).max()
                dfs[tf] = df
            except Exception as e:
                span.record_exception(e)
                dfs[tf] = None
            await asyncio.sleep(delay_between_requests)
        span.set_status(Status(StatusCode.OK))
        return dfs

def scan(exchange_name="binance", top_n=25, timeframe="1h", limit_ohlcv=500, delay_between_requests=0.15, mtf=None):
    with tracer.start_as_current_span("scan_total") as span:
        span.set_attribute("exchange", exchange_name)
        span.set_attribute("timeframe", timeframe)
        span.set_attribute("coins_limit", top_n)
        try:
            start_time = time.time()
            ex = create_exchange(exchange_name)
            mtf = mtf or [timeframe]

            # symbols
            with tracer.start_as_current_span("fetch_symbols") as sspan:
                if exchange_name == "binance":
                    symbols = fetch_top_symbols_binance_futures(ex, limit=top_n)
                else:
                    symbols = fetch_top_symbols_bybit_futures(ex, limit=top_n)
                sspan.set_attribute("symbols_count", len(symbols))

            if not symbols:
                span.set_status(Status(StatusCode.ERROR))
                return {"results": [], "metrics": {}}

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _run():
                results = []
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                    sem = asyncio.Semaphore(10)
                    async def handle_symbol(sym):
                        with tracer.start_as_current_span("process_symbol") as sspan:
                            sspan.set_attribute("symbol", sym)
                            try:
                                async with sem:
                                    dfs = await build_mtf_dfs_async(exchange_name, ex, sym, mtf, limit_ohlcv, session, delay_between_requests)
                                    sig = signals_v4.evaluate_signals_mtf(dfs)
                                    res = {
                                        "symbol": sym,
                                        "signal": sig.get("signal"),
                                        "confidence": sig.get("confidence", 0),
                                        "reasons": sig.get("reasons", []),
                                    }
                                    sspan.set_status(Status(StatusCode.OK))
                                    return res
                            except Exception as e:
                                sspan.record_exception(e)
                                sspan.set_status(Status(StatusCode.ERROR))
                                return {"symbol": sym, "error": str(e)}
                    tasks = [asyncio.create_task(handle_symbol(s)) for s in symbols]
                    done = await asyncio.gather(*tasks, return_exceptions=True)
                    for d in done:
                        if isinstance(d, dict):
                            results.append(d)
                return results

            results = loop.run_until_complete(_run())
            duration = round(time.time() - start_time, 2)
            span.set_status(Status(StatusCode.OK))
            span.set_attribute("duration_sec", duration)
            return {"results": results, "metrics": {"duration_sec": duration, "count": len(results)}}

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            return {"results": [{"error": str(e)}], "metrics": {}}
