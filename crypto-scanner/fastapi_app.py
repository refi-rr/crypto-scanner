# fastapi_app.py — FULL API (NO ohlcv_z + scheduler result + FNG)
# Version 1.3

from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import requests
import json
import os
from pathlib import Path

from scanner.scanner_core import scan_async

# -------------------------------------------------
# PATHS
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = PROJECT_ROOT / "data"
SCHEDULER_FILE = DATA_DIR / "last_scan.json"

app = FastAPI(
    title="Crypto Scanner API",
    version="1.3",
    description="Massive Scan, Single Scan, Scheduler Result, and Fear-Greed"
)

# -------------------------------------------------
# RESPONSE MODELS
# -------------------------------------------------
class ScanResult(BaseModel):
    symbol: str
    signal: str
    confidence: float
    entry: float | None
    tp1: float | None
    tp2: float | None
    sl: float | None
    support: float | None
    resistance: float | None
    reasons: list[str]
    meta: dict


class ScanPacket(BaseModel):
    results: list[ScanResult]
    metrics: dict


# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------------------------
# FEAR & GREED — CURRENT
# -------------------------------------------------
@app.get("/feargreed/current")
def fear_greed_current():
    r = requests.get("https://api.alternative.me/fng/", timeout=10).json()
    d = r["data"][0]
    return {
        "value": int(d["value"]),
        "classification": d["value_classification"],
        "timestamp": d["timestamp"]
    }


# -------------------------------------------------
# FEAR & GREED — HISTORY
# -------------------------------------------------
@app.get("/feargreed/history")
def fear_greed_history(limit: int = 90):
    r = requests.get(
        f"https://api.alternative.me/fng/?limit={limit}&format=json",
        timeout=10
    ).json()
    return r.get("data", [])


# -------------------------------------------------
# MASSIVE SCAN (NO ohlcv_z)
# -------------------------------------------------
@app.get("/scan", response_model=ScanPacket)
async def run_massive_scan(
    exchange: str = "binance",
    top_n: int = 50,
    timeframe: str = "1h",
    candles: int = 500
):
    pkt = await scan_async(
        exchange_name=exchange,
        top_n=top_n,
        timeframe=timeframe,
        limit_ohlcv=candles,
        delay_between_requests=0.25,
        mtf=[timeframe, "4h", "1d"]
    )

    clean_results = []
    for r in pkt["results"]:
        clean_results.append({
            k: v for k, v in r.items() if k != "ohlcv_z"
        })

    return {
        "results": clean_results,
        "metrics": pkt.get("metrics", {})
    }


# -------------------------------------------------
# SINGLE SCAN (NO ohlcv_z)
# -------------------------------------------------
@app.get("/scan/single", response_model=ScanResult)
async def run_single_scan(
    symbol: str,
    exchange: str = "binance",
    timeframe: str = "1h",
    candles: int = 500
):
    pkt = await scan_async(
        exchange_name=exchange,
        top_n=1,
        timeframe=timeframe,
        limit_ohlcv=candles,
        delay_between_requests=0.20,
        mtf=[timeframe, "4h", "1d"]
    )

    if not pkt["results"]:
        return {}

    r = pkt["results"][0]
    clean_r = {k: v for k, v in r.items() if k != "ohlcv_z"}
    return clean_r


# -------------------------------------------------
#  NEW: GET LAST SCHEDULER RESULT
# -------------------------------------------------
@app.get("/scheduler/latest")
def scheduler_latest():
    if not SCHEDULER_FILE.exists():
        return {
            "success": False,
            "message": "scheduler file not found",
            "path_checked": str(SCHEDULER_FILE)
        }

    try:
        with open(SCHEDULER_FILE, "r") as f:
            data = json.load(f)

        # bersihkan ohlcv_z
        for r in data.get("result", []):
            if "ohlcv_z" in r:
                r.pop("ohlcv_z")

        return {
            "success": True,
            "timestamp": data.get("timestamp"),
            "results": data.get("result", []),
            "metrics": data.get("metrics", {})
        }

    except Exception as e:
        return {
            "success": False,
            "message": "failed to parse scheduler file",
            "error": str(e),
            "path_checked": str(SCHEDULER_FILE)
        }
