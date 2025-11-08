# scanner/scanner_core.py — v4.7.6 Traced OHLCV + Jaeger
"""
scanner_core v4.7.6 — Resilient OHLCV + Jaeger Tracing
- Menambahkan distributed tracing ke setiap tahapan utama scan()
- Span-level timing: create_exchange, fetch_symbols, build_mtf_dfs_async, evaluate_signals
- Masih kompatibel penuh dengan versi sebelumnya
"""

import asyncio
import aiohttp
import time
import random
import os
import json
import zlib
import base64
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import logging
from typing import List, Any, Optional

# === Tracing setup ===
from scanner.tracing_setup import init_tracer
tracer = init_tracer("crypto-futures-scanner")

from scanner.datafetch import (
    create_exchange,
    test_ping_binance_fapi,
    test_ping_bybit,
    fetch_top_symbols_binance_futures,
    fetch_top_symbols_bybit_futures,
)
from scanner.indicators import compute_indicators
from scanner import signals_v4

# optional history
try:
    from scanner.history import init_db, log_signal
    init_db()
except Exception:
    def log_signal(x): pass

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# CONFIG
DEFAULT_CONCURRENCY = 15
MAX_CONCURRENCY = 30
MIN_CONCURRENCY = 2
CACHE_FILE = "/tmp/binance_exchangeinfo_cache.json"
CACHE_TTL = 600  # seconds
_THREAD_POOL = ThreadPoolExecutor(max_workers=14)
BINANCE_KLINES_URLS = [
    "https://fapi.binance.com/fapi/v1/klines",
    "https://data-api.binance.vision/api/v3/klines",
    "https://futures.binance.com/fapi/v1/klines",
]
RETRY_COUNT = 3
RETRY_BACKOFF_BASE = 0.5  # seconds

# -----------------------
# Helpers
# -----------------------
def tf_to_binance_interval(tf: str) -> str:
    return tf

def normalize_symbol_for_binance(sym: str) -> str:
    s = str(sym).split(":")[0]
    s = s.replace("/", "").upper()
    if not s.endswith("USDT"):
        s = f"{s}USDT"
    return s

def ohlcv_to_df(ohlcv: List[List[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv)
    if df.shape[1] >= 6:
        df = df.iloc[:, :6]
        df.columns = ["ts", "open", "high", "low", "close", "volume"]
    else:
        df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("datetime", inplace=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def get_cached_symbols():
    try:
        if os.path.exists(CACHE_FILE):
            age = time.time() - os.path.getmtime(CACHE_FILE)
            if age < CACHE_TTL:
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
    except Exception:
        pass
    return None

def save_cached_symbols(symbols):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(symbols, f)
    except Exception:
        pass

def compress_df_to_b64z(df: pd.DataFrame) -> Optional[str]:
    try:
        j = df.reset_index().to_json(orient="split", date_format="iso")
        zb = zlib.compress(j.encode("utf-8"), level=6)
        b64 = base64.b64encode(zb).decode("ascii")
        return b64
    except Exception as e:
        logger.debug(f"compress_df_to_b64z failed: {e}")
        return None

# -----------------------
# HTTP fetch with retry/backoff
# -----------------------
async def fetch_http_with_retries(session: aiohttp.ClientSession, url: str, params: dict, max_retries: int = RETRY_COUNT, initial_timeout: int = 10):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=max(initial_timeout, 8))
            async with session.get(url, params=params, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                return await resp.json()
        except Exception as e:
            last_exc = e
            backoff = RETRY_BACKOFF_BASE * (2 ** (attempt - 1)) * random.uniform(0.8, 1.3)
            logger.debug(f"HTTP fetch attempt {attempt}/{max_retries} failed for {params.get('symbol')} @ {url}: {e} — sleeping {backoff:.2f}s")
            if attempt < max_retries:
                await asyncio.sleep(backoff)
            continue
    raise last_exc or RuntimeError("HTTP fetch failed after retries")

async def fetch_symbol_klines(symbol: str, tf: str, limit: int, session: aiohttp.ClientSession):
    params = {"symbol": normalize_symbol_for_binance(symbol), "interval": tf_to_binance_interval(tf), "limit": limit}
    last_exc = None
    for url in BINANCE_KLINES_URLS:
        try:
            data = await fetch_http_with_retries(session, url, params)
            return data
        except Exception as e:
            last_exc = e
            await asyncio.sleep(random.uniform(0.05, 0.18))
            continue
    raise last_exc or RuntimeError("No response from Binance endpoints")

# -----------------------
# Hybrid fetch (HTTP -> ccxt fallback)
# -----------------------
async def fetch_ohlcv_for_symbol(exchange_name: str, exchange_obj, symbol: str, timeframe: str, limit: int, aiohttp_session: Optional[aiohttp.ClientSession], use_http_primary: bool = True):
    with tracer.start_as_current_span("fetch_ohlcv_single") as span:
        span.set_attribute("symbol", symbol)
        span.set_attribute("timeframe", timeframe)
        if exchange_name == "binance" and aiohttp_session and use_http_primary:
            try:
                payload = await fetch_symbol_klines(symbol, timeframe, limit, aiohttp_session)
                return payload
            except Exception as e:
                logger.debug(f"Binance HTTP klines failed for {symbol} [{timeframe}]: {e}; will fallback to ccxt.")
        loop = asyncio.get_event_loop()
        try:
            func = exchange_obj.fetch_ohlcv
            result = await loop.run_in_executor(_THREAD_POOL, lambda: func(symbol, timeframe, limit))
            return result
        except Exception as e:
            logger.warning(f"ccxt.fetch_ohlcv fallback failed for {symbol} [{timeframe}]: {e}")
            span.record_exception(e)
            return None

# -----------------------
# Build multi timeframe dfs
# -----------------------
async def build_mtf_dfs_async(exchange_name: str, exchange_obj, symbol: str, timeframes: List[str], limit: int, aiohttp_session: Optional[aiohttp.ClientSession], delay_between_requests: float):
    with tracer.start_as_current_span("build_mtf_dfs") as span:
        span.set_attribute("symbol", symbol)
        dfs = {}
        tasks = [fetch_ohlcv_for_symbol(exchange_name, exchange_obj, symbol, tf, limit, aiohttp_session) for tf in timeframes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tf, res in zip(timeframes, results):
            if isinstance(res, Exception) or res is None:
                logger.warning(f"No data returned for {symbol}[{tf}]")
                dfs[tf] = None
                continue
            try:
                df = ohlcv_to_df(res)
                df = compute_indicators(df)
                if "low" in df.columns and "high" in df.columns:
                    df["support"] = df["low"].rolling(20).min()
                    df["resistance"] = df["high"].rolling(20).max()
                dfs[tf] = df
            except Exception as e:
                logger.warning(f"Failed to build df for {symbol}[{tf}]: {e}")
                dfs[tf] = None
            await asyncio.sleep(delay_between_requests * random.uniform(0.6, 1.4))
        return dfs

# -----------------------
# Main scan()
# -----------------------
def scan(exchange_name: str = "binance",
         top_n: int = 50,
         timeframe: str = "1h",
         limit_ohlcv: int = 500,
         delay_between_requests: float = 0.15,
         mtf: Optional[List[str]] = None):

    with tracer.start_as_current_span("scan_total") as span:
        mtf = mtf or [timeframe]
        exchange_name = exchange_name.lower().strip()
        start_time = time.time()
        logger.info(f"🔍 Starting scan on {exchange_name.upper()} ({top_n} symbols) ...")

        try:
            with tracer.start_as_current_span("create_exchange"):
                ex = create_exchange(exchange_name)
        except Exception as e:
            msg = f"Failed to create exchange instance: {e}"
            logger.error(msg)
            span.record_exception(e)
            return {"results":[{"error": msg}], "metrics": {}}

        with tracer.start_as_current_span("fetch_symbols"):
            symbols = None
            if exchange_name == "binance":
                symbols = get_cached_symbols()
            if not symbols:
                try:
                    if exchange_name == "binance":
                        symbols = fetch_top_symbols_binance_futures(ex, limit=top_n)
                        save_cached_symbols(symbols)
                    else:
                        symbols = fetch_top_symbols_bybit_futures(ex, limit=top_n)
                except Exception as e:
                    logger.error(f"Failed to fetch symbols: {e}")
                    span.record_exception(e)
                    return {"results":[{"error": str(e)}], "metrics": {}}

        if not symbols:
            msg = "No symbols found."
            logger.warning(msg)
            return {"results":[{"error": msg}], "metrics": {}}

        symbols = symbols[:top_n]
        concurrency = min(MAX_CONCURRENCY, max(MIN_CONCURRENCY, int(DEFAULT_CONCURRENCY * (top_n / 50))))
        logger.info(f"Using concurrency={concurrency} for async fetch tasks.")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run_all():
            results = []
            timeout = aiohttp.ClientTimeout(total=40)
            conn = aiohttp.TCPConnector(limit_per_host=concurrency, ttl_dns_cache=300)
            async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
                sem = asyncio.Semaphore(concurrency)

                async def process_symbol(sym: str):
                    with tracer.start_as_current_span("process_symbol") as span_sym:
                        span_sym.set_attribute("symbol", sym)
                        async with sem:
                            res = {"symbol": sym}
                            t0 = time.time()
                            try:
                                dfs = await build_mtf_dfs_async(exchange_name, ex, sym, mtf, limit_ohlcv, session, delay_between_requests)
                                sig = signals_v4.evaluate_signals_mtf(dfs)
                                base_df = dfs.get(mtf[0])
                                entry = tp1 = tp2 = sl = support = resistance = None
                                if base_df is not None and not base_df.empty:
                                    latest = base_df.iloc[-1]
                                    entry = float(latest.get("close") or 0)
                                    atr = float(latest.get("atr") or 0)
                                    support = float(latest.get("support") or 0)
                                    resistance = float(latest.get("resistance") or 0)
                                    if atr > 0:
                                        if sig.get("signal") == "LONG":
                                            sl = entry - 1.5 * atr
                                            tp1 = entry + 2 * atr
                                            tp2 = entry + 3 * atr
                                        elif sig.get("signal") == "SHORT":
                                            sl = entry + 1.5 * atr
                                            tp1 = entry - 2 * atr
                                            tp2 = entry - 3 * atr

                                ohlcv_z = None
                                try:
                                    candidate_df = base_df if (base_df is not None and not base_df.empty) else None
                                    if candidate_df is not None:
                                        tmp = candidate_df.tail(400).reset_index().copy()
                                        if "datetime" not in tmp.columns:
                                            tmp.rename(columns={tmp.columns[0]: "datetime"}, inplace=True)
                                        for c in ["open","high","low","close","volume"]:
                                            if c not in tmp.columns:
                                                tmp[c] = None
                                        ohlcv_z = compress_df_to_b64z(tmp)
                                except Exception as e:
                                    logger.warning(f"⚠️ Failed to build compressed OHLCV for {sym}: {e}")
                                    span_sym.record_exception(e)
                                    ohlcv_z = None

                                res.update({
                                    "signal": sig.get("signal"),
                                    "confidence": sig.get("confidence"),
                                    "reasons": sig.get("reasons"),
                                    "meta": sig.get("meta"),
                                    "entry": entry,
                                    "tp1": tp1,
                                    "tp2": tp2,
                                    "sl": sl,
                                    "support": support,
                                    "resistance": resistance,
                                    "ohlcv_z": ohlcv_z,
                                })

                                log_signal({
                                    "ts": int(time.time()),
                                    "symbol": sym,
                                    "timeframes": mtf,
                                    "signal": sig.get("signal"),
                                    "confidence": sig.get("confidence"),
                                    "entry": entry,
                                    "tp1": tp1, "tp2": tp2, "sl": sl,
                                    "reasons": sig.get("reasons"),
                                    "meta": sig.get("meta"),
                                })

                                res["_latency"] = round(time.time() - t0, 3)
                                return res
                            except Exception as e:
                                logger.warning(f"Error scanning {sym}: {e}")
                                span_sym.record_exception(e)
                                res["error"] = str(e)
                                res["_latency"] = round(time.time() - t0, 3)
                                return res

                tasks = [asyncio.create_task(process_symbol(s)) for s in symbols]
                done = await asyncio.gather(*tasks, return_exceptions=True)
                for d in done:
                    if isinstance(d, Exception):
                        logger.warning(f"Task exception: {d}")
                        continue
                    results.append(d)
            return results

        all_results = loop.run_until_complete(_run_all())
        total_sec = round(time.time() - start_time, 2)
        success = len([r for r in all_results if not r.get("error")])
        avg_latency = round(sum([r.get("_latency", 0) or 0 for r in all_results]) / max(1, len(all_results)), 3)
        metrics = {"total_sec": total_sec, "avg_latency": avg_latency, "concurrency": concurrency, "success": success, "total": len(all_results)}
        for r in all_results:
            r.pop("_latency", None)

        logger.info(f"✅ Scan finished for {len(all_results)} symbols on {exchange_name.upper()}. success={success}/{len(all_results)}")
        return {"results": all_results, "metrics": metrics}
