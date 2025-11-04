# scanner/signals_v4.py
"""
signals_v4.4 — Quantified Signal 2.0
-----------------------------------
✓ Enhanced technical logic (no UI changes)
✓ Adds Stoch RSI momentum, MACD golden/death cross, volume breakout
✓ Weighted confidence scoring
✓ Backward-compatible output
"""

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Helper: compute cross direction
# -------------------------------------------------------------------------
def cross_signal(series_a, series_b):
    """Detect simple crossovers: return 1 for golden cross, -1 for death cross, 0 otherwise."""
    if len(series_a) < 2 or len(series_b) < 2:
        return 0
    prev = series_a.iloc[-2] - series_b.iloc[-2]
    curr = series_a.iloc[-1] - series_b.iloc[-1]
    if prev < 0 and curr > 0:
        return 1  # golden cross
    elif prev > 0 and curr < 0:
        return -1  # death cross
    return 0


# -------------------------------------------------------------------------
# Core signal evaluation
# -------------------------------------------------------------------------
def evaluate_signals_mtf(dfs: dict):
    """
    Evaluate multi-timeframe signals.
    Returns dict: {"signal": "LONG"/"SHORT"/"NEUTRAL", "confidence": float, "reasons": list, "meta": dict}
    """
    reasons = []
    total_score = 0
    max_score = 0
    meta = {}

    for tf, df in dfs.items():
        if df is None or len(df) < 5:
            continue

        latest = df.iloc[-1]
        score = 0
        tf_reasons = []

        # --- EMA CROSS ---
        ema9 = latest.get("ema_9")
        ema21 = latest.get("ema_21")
        if ema9 and ema21:
            if ema9 > ema21:
                tf_reasons.append(f"[{tf}] EMA9 > EMA21 (bullish)")
                score += 1
            elif ema9 < ema21:
                tf_reasons.append(f"[{tf}] EMA9 < EMA21 (bearish)")
                score -= 1
        max_score += 1

        # --- MACD DIRECTION + CROSS ---
        macd = latest.get("macd")
        signal = latest.get("signal")
        if macd is not None and signal is not None:
            if macd > signal:
                tf_reasons.append(f"[{tf}] MACD > Signal (bullish)")
                score += 1
            elif macd < signal:
                tf_reasons.append(f"[{tf}] MACD < Signal (bearish)")
                score -= 1

            # golden/death cross detection
            cross = cross_signal(df["macd"], df["signal"])
            if cross == 1:
                tf_reasons.append(f"[{tf}] MACD Golden Cross 🔰")
                score += 1
            elif cross == -1:
                tf_reasons.append(f"[{tf}] MACD Death Cross ☠️")
                score -= 1
        max_score += 1

        # --- ADX STRENGTH ---
        adx = latest.get("adx")
        if adx:
            if adx < 20:
                tf_reasons.append(f"[{tf}] Weak trend (ADX < 20)")
                score -= 0.5
            elif adx > 25:
                tf_reasons.append(f"[{tf}] Strong trend (ADX > 25)")
                score += 0.5
        max_score += 0.5

        # --- STOCH RSI MOMENTUM ---
        k = latest.get("stoch_k")
        d = latest.get("stoch_d")
        if k is not None and d is not None:
            if k > d and k < 20:
                tf_reasons.append(f"[{tf}] Stoch RSI bullish reversal (K={k:.1f}, D={d:.1f})")
                score += 1
            elif k < d and k > 80:
                tf_reasons.append(f"[{tf}] Stoch RSI bearish reversal (K={k:.1f}, D={d:.1f})")
                score -= 1
        max_score += 1

        # --- VOLUME BREAKOUT ---
        vol = latest.get("volume")
        if "vol_sma" in df.columns:
            vol_sma = df["vol_sma"].iloc[-1]
            if vol_sma and vol > 2 * vol_sma:
                tf_reasons.append(f"[{tf}] Volume breakout detected (x{vol / vol_sma:.2f})")
                score += 0.5
        max_score += 0.5

        # accumulate
        total_score += score
        reasons.extend(tf_reasons)
        meta[tf] = {"score": score, "reasons": tf_reasons}

    # ---------------------------------------------------------------------
    # Final aggregation
    # ---------------------------------------------------------------------
    confidence = 0 if max_score == 0 else round(abs(total_score) / max_score * 100, 1)
    direction = "NEUTRAL"
    if total_score > 0.5:
        direction = "LONG"
    elif total_score < -0.5:
        direction = "SHORT"

    return {
        "signal": direction,
        "confidence": confidence,
        "reasons": reasons,
        "meta": meta,
    }
