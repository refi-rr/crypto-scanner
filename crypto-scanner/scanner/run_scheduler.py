# run_scheduler.py
import asyncio
import datetime
import json
import os
import time

from scanner.scanner_core import scan_async

RESULT_PATH = "data/last_scan.json"
TARGET_HOURS = [7, 11, 15, 19, 23, 3]

os.makedirs("data", exist_ok=True)

async def scheduler_loop():
    print("[scheduler] started (background mode)")

    while True:
        now = datetime.datetime.now()
        hour = now.hour
        minute = now.minute

        # trigger only exactly at target hour
        if hour in TARGET_HOURS and minute == 0:
            print(f"[scheduler] Running auto scan at {now}")

            try:
                result = await scan_async(
                    exchange_name="binance",
                    top_n=50,
                    timeframe="1h",
                    limit_ohlcv=500,
                    delay_between_requests=0.5,
                    mtf=["1h", "4h", "1d"]
                )

                payload = {
                    "timestamp": now.isoformat(),
                    "result": result
                }

                with open(RESULT_PATH, "w") as f:
                    json.dump(payload, f)

                print("[scheduler] scan saved to last_scan.json")

            except Exception as e:
                print(f"[scheduler] FAILED: {e}")

            # avoid double-trigger
            await asyncio.sleep(70)

        await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(scheduler_loop())
