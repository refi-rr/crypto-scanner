# scanner/datafetch.py
"""
datafetch v4.6 — Complete export surface
- Exports: create_exchange, test_ping_binance_fapi, test_ping_bybit,
  fetch_top_symbols_binance_futures, fetch_top_symbols_bybit_futures
- Uses direct HTTP fallback to Binance Vision if ccxt.load_markets fails
- Does NOT call load_markets in create_exchange to avoid forcing exchangeInfo on init
"""

import logging
import time
import requests
import ccxt
from ccxt.base.errors import RequestTimeout, NetworkError, ExchangeNotAvailable

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ------------------------
# Exchange factory
# ------------------------
def create_exchange(name: str, timeout_ms: int = 20000):
    """
    Create CCXT exchange instance without forcing load_markets().
    For Binance: force futures base URL so ccxt won't hit api.binance.com.
    """
    name = name.lower()
    try:
        if name == "binance":
            ex = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "future", "adjustForTimeDifference": True},
                "urls": {
                    "api": {
                        "public": "https://fapi.binance.com/fapi/v1",
                        "private": "https://fapi.binance.com/fapi/v1",
                    }
                },
                "timeout": timeout_ms,
            })
            logger.info("Created ccxt.binance instance (fapi forced).")
            return ex

        elif name == "bybit":
            ex = ccxt.bybit({
                "enableRateLimit": True,
                "options": {"defaultType": "future", "adjustForTimeDifference": True},
                "timeout": timeout_ms,
            })
            logger.info("Created ccxt.bybit instance.")
            return ex

        else:
            raise ValueError(f"Exchange '{name}' not supported")

    except Exception as e:
        logger.error(f"Failed to create exchange {name}: {e}")
        raise


# ------------------------
# Ping helpers
# ------------------------
def test_ping_binance_fapi(timeout: float = 5.0) -> bool:
    """Lightweight ping to Binance Futures (fapi.binance.com)."""
    url = "https://fapi.binance.com/fapi/v1/ping"
    try:
        r = requests.get(url, timeout=timeout)
        ok = (r.status_code == 200)
        if not ok:
            logger.warning(f"Binance FAPI ping returned {r.status_code}")
        return ok
    except Exception as e:
        logger.warning(f"Binance FAPI ping failed: {e}")
        return False


def test_ping_bybit(timeout: float = 5.0):
    """Ping Bybit public endpoint; returns (ok: bool, latency_ms: float|None)."""
    url = "https://api.bybit.com/v5/market/time"
    try:
        start = time.time()
        r = requests.get(url, timeout=timeout)
        latency = round((time.time() - start) * 1000, 1)
        ok = (r.status_code == 200)
        if not ok:
            logger.warning(f"Bybit ping returned {r.status_code}")
        return ok, latency
    except Exception as e:
        logger.warning(f"Bybit ping failed: {e}")
        return False, None


# ------------------------
# Internal helper: direct HTTP fetch of exchangeInfo (fallback)
# ------------------------
def _fetch_markets_fapi_requests(timeout=8):
    """
    Try direct HTTP requests to Binance FAPI / Vision to collect symbols.
    Returns list of normalized symbols like 'BTC/USDT'.
    """
    urls = [
        "https://fapi.binance.com/fapi/v1/exchangeInfo",
        "https://data-api.binance.vision/api/v3/exchangeInfo",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                logger.warning(f"Fallback {url} returned {r.status_code}")
                continue
            payload = r.json()
            syms = []
            for s in payload.get("symbols", []):
                if s.get("quoteAsset") != "USDT":
                    continue
                if s.get("status") != "TRADING":
                    continue
                symbol = s.get("symbol")
                # Normalize to CCXT-like 'BASE/QUOTE'
                if "/" not in symbol:
                    if symbol.endswith("USDT"):
                        base = symbol[:-4]
                        norm = f"{base}/USDT"
                    else:
                        norm = symbol
                else:
                    norm = symbol
                # Filter perpetuals preferentially if contractType present
                contract_type = s.get("contractType") or s.get("contract_type") or None
                if contract_type:
                    if str(contract_type).upper() == "PERPETUAL":
                        syms.append(norm)
                else:
                    # if no contract type info (vision might not include), accept it
                    syms.append(norm)
            # de-duplicate preserving order
            uniq = []
            for x in syms:
                if x not in uniq:
                    uniq.append(x)
            logger.info(f"Fetched {len(uniq)} symbols from fallback {url}")
            return uniq
        except Exception as e:
            logger.warning(f"Fallback fetch {url} failed: {e}")
            continue
    return []


# ------------------------
# Symbol fetchers (Binance + Bybit)
# ------------------------
def fetch_top_symbols_binance_futures(exchange, limit=100, try_ccxt=True):
    """
    Fetch top Binance USDT PERPETUAL symbols safely.
    - try_ccxt: attempt exchange.load_markets() once; if fails, fallback to HTTP
    """
    symbols = []
    if try_ccxt:
        try:
            logger.info("Attempting exchange.load_markets() via ccxt (Binance)...")
            # Avoid forced full reload; let ccxt decide caching
            markets = exchange.load_markets(reload=False)
            for s, m in markets.items():
                try:
                    # filter for USDT perpetual futures
                    info = m.get("info", {}) or {}
                    mtype = m.get("type")
                    if "USDT" in s and "BUSD" not in s:
                        if (mtype == "future") or (str(info.get("contractType", "")).upper() == "PERPETUAL"):
                            symbols.append(s)
                except Exception:
                    continue
            if symbols:
                logger.info(f"ccxt.load_markets found {len(symbols)} perpetual symbols.")
                return symbols[:limit]
            else:
                logger.warning("ccxt.load_markets returned no perpetuals; will fallback.")
        except (RequestTimeout, NetworkError, ExchangeNotAvailable) as e:
            logger.warning(f"ccxt.load_markets failed for Binance: {e}")
        except Exception as e:
            logger.warning(f"Unexpected ccxt.load_markets error (Binance): {e}")

    # Fallback to direct HTTP
    logger.info("Falling back to direct HTTP to fetch Binance symbols (FAPI/Vision).")
    try_syms = _fetch_markets_fapi_requests()
    if try_syms:
        return try_syms[:limit]
    logger.error("Failed to obtain Binance symbols via both ccxt and HTTP fallback.")
    return []


def fetch_top_symbols_bybit_futures(exchange, limit=100):
    """
    Fetch top Bybit USDT perpetual symbols.
    Use ccxt.load_markets ideally. No HTTP fallback provided (Bybit has different API).
    """
    symbols = []
    try:
        logger.info("Loading Bybit markets via ccxt...")
        markets = exchange.load_markets(reload=False)
        for s, m in markets.items():
            try:
                if "USDT" in s:
                    mtype = m.get("type")
                    # Some bybit markets have id like 'BTCUSDT' and info may include 'symbol'
                    info = m.get("info", {}) or {}
                    # try to detect perpetual via id or info fields
                    mid = str(m.get("id") or info.get("symbol") or "").upper()
                    if (mtype in ("linear", "future")) or ("PERP" in mid or "PERPETUAL" in mid):
                        symbols.append(s)
            except Exception:
                continue
        if symbols:
            logger.info(f"Found {len(symbols)} Bybit perpetual symbols.")
            return symbols[:limit]
        else:
            logger.warning("No Bybit perpetual symbols found via ccxt.load_markets.")
            return []
    except (RequestTimeout, NetworkError, ExchangeNotAvailable) as e:
        logger.warning(f"Bybit load_markets failed: {e}")
        return []
    except Exception as e:
        logger.warning(f"Unexpected Bybit fetch error: {e}")
        return []


# ------------------------
# End of datafetch.py
# ------------------------
