import asyncio
import aiohttp
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_exchange(exchange_name: str = "binance"):
    """
    Factory function to create a ccxt exchange instance with proper configuration.
    Supports 'binance' and 'bybit' (default: binance futures).
    """
    try:
        exchange_name = exchange_name.lower().strip()

        if exchange_name == "binance":
            ex = ccxt.binance({
                "enableRateLimit": True,
                "timeout": 10000,
                "options": {"defaultType": "future"}
            })
        elif exchange_name == "bybit":
            ex = ccxt.bybit({
                "enableRateLimit": True,
                "timeout": 10000,
                "options": {"defaultType": "swap"}
            })
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        logger.info(f"[create_exchange] Created ccxt.{exchange_name} instance")
        return ex

    except Exception as e:
        logger.error(f"Failed to create exchange {exchange_name}: {e}")
        return None

async def fetch_ohlcv_async(exchange, symbol, timeframe="1h", limit=500):
    """
    Async-safe OHLCV fetch with multi-endpoint fallback (Binance Futures + Vision).
    Returns DataFrame or empty if all fail.
    """
    async def _via_ccxt():
        try:
            data = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if data:
                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.drop(columns=["timestamp"])
                return df
        except Exception as e:
            logger.debug(f"[ccxt_fetch_fail] {symbol} {timeframe}: {e}")
        return pd.DataFrame()

    async def _via_http():
        try:
            sym = symbol.replace("/", "")
            url = f"https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval={timeframe}&limit={limit}"
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url, timeout=8) as r:
                    if r.status == 200:
                        js = await r.json()
                        if not js:
                            return pd.DataFrame()
                        df = pd.DataFrame(js, columns=[
                            "timestamp","open","high","low","close","volume",
                            "_1","_2","_3","_4","_5","_6"
                        ])
                        df = df[["timestamp","open","high","low","close","volume"]]
                        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                        df = df.drop(columns=["timestamp"])
                        return df
        except Exception as e:
            logger.debug(f"[http_fallback_fail] {symbol} {timeframe}: {e}")
        return pd.DataFrame()

    # ---- execution ----
    df = await _via_ccxt()
    if df.empty:
        df = await _via_http()

    if df.empty:
        logger.warning(f"[fetch_ohlcv_async] No data for {symbol} ({timeframe})")
    return df

def fetch_top_symbols_binance_futures(exchange=None, limit: int = 100):
    """
    Fetch top perpetual symbols from Binance Futures (USDT-margined contracts).
    'exchange' arg is ignored, kept only for backward compatibility.
    """
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        js = resp.json()
        symbols = []

        for s in js.get("symbols", []):
            if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING":
                sym = s.get("symbol")
                if sym.endswith("USDT"):
                    symbols.append(sym.replace("USDT", "/USDT"))

        symbols = sorted(symbols, key=lambda x: (not x.startswith("BTC/"), not x.startswith("ETH/"), x))
        return symbols[:limit]

    except Exception as e:
        logger.warning(f"Failed to fetch top symbols from Binance Futures: {e}")
        return []

def fetch_top_symbols_bybit_futures(limit: int = 100):
    """
    Fetch top USDT perpetual trading pairs from Bybit.
    Returns a list of symbols like ['BTC/USDT', 'ETH/USDT', ...].
    """
    url = "https://api.bybit.com/v5/market/instruments-info?category=linear"
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        js = resp.json()
        symbols = []

        for item in js.get("result", {}).get("list", []):
            if (
                item.get("quoteCoin") == "USDT"
                and item.get("status") == "Trading"
                and "symbol" in item
            ):
                s = item["symbol"]
                if s.endswith("USDT"):
                    # Normalize to match ccxt style, e.g., BTCUSDT -> BTC/USDT
                    symbols.append(s.replace("USDT", "/USDT"))

        # Sort alphabetically with BTC and ETH first
        symbols = sorted(symbols, key=lambda x: (not x.startswith("BTC/"), not x.startswith("ETH/"), x))
        return symbols[:limit]

    except Exception as e:
        logger.warning(f"Failed to fetch top symbols from Bybit Futures: {e}")
        return []


def test_ping_binance_fapi(timeout=5):
    """
    Quick connectivity test to Binance Futures (FAPI) endpoint.
    Returns True if reachable, False otherwise.
    """
    url = "https://fapi.binance.com/fapi/v1/ping"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            logger.info("✅ Binance FAPI reachable")
            return True
        else:
            logger.warning(f"⚠️ Binance FAPI ping failed: {resp.status_code}")
            return False
    except Exception as e:
        logger.warning(f"Binance FAPI ping failed: {e}")
        return False
