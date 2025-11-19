import asyncio
import datetime
import logging

from scanner.scanner_core import scan_async

logger = logging.getLogger(__name__)

TARGET_HOURS = [7, 11, 15, 19, 23, 3]

last_result = None
last_timestamp = None
scheduler_started = False

async def scheduler_loop():
    global last_result, last_timestamp, scheduler_started

    if scheduler_started:
        return
    scheduler_started = True

    logger.info("[scheduler] started")

    while True:
        now = datetime.datetime.now()
        hour = now.hour
        minute = now.minute

        # Check if we hit target hour (minute == 0 to avoid multiple trigger)
        if hour in TARGET_HOURS and minute == 0:
            try:
                logger.info(f"[scheduler] Running auto scan at {now}")
                result = await scan_async(
                    exchange_name="binance",
                    top_n=50,
                    timeframe="1h",
                    limit_ohlcv=500,
                    delay_between_requests=0.50,
                    mtf=["1h","4h","1d"],
                )
                last_result = result
                last_timestamp = now
                logger.info("[scheduler] Auto scan complete.")

            except Exception as e:
                logger.error(f"[scheduler] Auto scan failed: {e}")

            await asyncio.sleep(70)  # avoid retrigger same hour

        await asyncio.sleep(10)  # check every 10 sec
