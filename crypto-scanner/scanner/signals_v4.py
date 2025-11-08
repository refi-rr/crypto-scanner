# scanner/signals_v4.py — v4.8
"""
evaluate_signals_mtf(dfs)
- Input: dfs = { '15m': df, '1h': df, '4h': df, ... } (some dfs may be None)
- Output: dict {
    "signal": "LONG"|"SHORT"|"NEUTRAL",
    "confidence": float(0-100),
    "reasons": [str],
    "meta": {"signals_per_tf": {tf: "LONG"/"SHORT"/"NEUTRAL"}}
  }
- Uses weighted fusion of EMA trend, MACD cross, StochRSI momentum, RSI zone, OBV trend, Supertrend dir, ADX filter
"""

import numpy as np

# weights for per-TF scoring (primary first will be given externally by scanner_core order)
BASE_WEIGHTS = {
    "ema": 0.25,
    "macd": 0.20,
    "stoch": 0.15,
    "rsi": 0.10,
    "obv": 0.10,
    "supertrend": 0.20,
    "vwap": 0.10,
    "ha_slope": 0.05
}

# normalize score to [-1,1]
def _sign_func(v):
    if v > 0: return 1
    if v < 0: return -1
    return 0

def _score_from_bool(flag, pos=1.0, neg=-1.0):
    return pos if flag else neg

def _per_tf_signal(df):
    """
    Given a single timeframe df (with indicators) compute:
    - signal: 1 long, -1 short, 0 neutral
    - reasons: list of text reasons
    - score_norm: float between -1 to 1 (negative bearish, positive bullish)
    """
    reasons = []
    # hard guard
    if df is None or len(df) < 3:
        return {"signal": 0, "score": 0.0, "reasons": ["no_data"]}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    # ---------- initialize all scores up front (no scope surprises) ----------
    ema_score = 0.0
    macd_score = 0.0
    stoch_score = 0.0
    rsi_score = 0.0
    obv_score = 0.0
    super_score = 0.0
    vwap_score = 0.0
    adx_factor = 1.0
    chop_factor = 1.0

    # ---------- EMA trend ----------
    try:
        if last.get("ema_9") is not None and last.get("ema_21") is not None:
            if last["ema_9"] > last["ema_21"]:
                ema_score = 1.0
                reasons.append("EMA9>EMA21 (bull)")
            elif last["ema_9"] < last["ema_21"]:
                ema_score = -1.0
                reasons.append("EMA9<EMA21 (bear)")
    except Exception:
        pass

    # ---------- MACD ----------
    try:
        macd = last.get("macd"); macd_sig = last.get("macd_signal")
        pm = prev.get("macd");   ps = prev.get("macd_signal")
        if macd is not None and macd_sig is not None and pm is not None and ps is not None:
            if pm < ps and macd > macd_sig:
                macd_score = 1.0; reasons.append("MACD golden_cross")
            elif pm > ps and macd < macd_sig:
                macd_score = -1.0; reasons.append("MACD death_cross")
            else:
                macd_score = 0.5 if macd > macd_sig else -0.5 if macd < macd_sig else 0.0
    except Exception:
        pass

    # ---------- StochRSI ----------
    try:
        k = last.get("stoch_k"); d = last.get("stoch_d")
        pk = prev.get("stoch_k"); pd_ = prev.get("stoch_d")
        if k is not None and d is not None and pk is not None and pd_ is not None:
            if pk < pd_ and k > d and k < 0.2:
                stoch_score = 1.0; reasons.append("StochRSI cross (oversold) bullish")
            elif pk > pd_ and k < d and k > 0.8:
                stoch_score = -1.0; reasons.append("StochRSI cross (overbought) bearish")
            else:
                stoch_score = 0.4 if k > d else -0.4 if k < d else 0.0
    except Exception:
        pass

    # ---------- RSI ----------
    try:
        r = last.get("rsi")
        if r is not None:
            if r < 30:
                rsi_score = 0.6; reasons.append("RSI oversold")
            elif r > 70:
                rsi_score = -0.6; reasons.append("RSI overbought")
            else:
                rsi_score = 0.1 if r > 50 else -0.1
    except Exception:
        pass

    # MFI (Money Flow Index)
    mfi_score = 0.0
    try:
        mfi = last.get("mfi")
        if mfi is not None and not np.isnan(mfi):
            if mfi < 20:
                mfi_score = 0.6
                reasons.append(f"MFI {round(mfi,1)} oversold (buy pressure)")
            elif mfi > 80:
                mfi_score = -0.6
                reasons.append(f"MFI {round(mfi,1)} overbought (sell pressure)")
            else:
                # subtle bias
                mfi_score = 0.1 if mfi > 50 else -0.1
    except Exception:
        mfi_score = 0.0


    # ---------- OBV ----------
    try:
        obv = last.get("obv")
        if obv is not None and "obv" in df.columns:
            m = df["obv"].rolling(30, min_periods=1).mean().iloc[-1]
            obv_score = 0.25 if obv > m else -0.25
            reasons.append("OBV > MA30 (buy pressure)" if obv > m else "OBV < MA30 (sell pressure)")
    except Exception:
        pass

    # ---------- Supertrend ----------
    try:
        if last.get("supertrend_dir") is not None:
            if bool(last["supertrend_dir"]):
                super_score = 1.0; reasons.append("Supertrend up")
            else:
                super_score = -1.0; reasons.append("Supertrend down")
    except Exception:
        pass

    # ---------- VWAP (robust) ----------
    try:
        if "vwap" in df.columns:
            v = last.get("vwap", None)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                if last["close"] > v:
                    vwap_score = 0.5; reasons.append("Price above VWAP (bullish bias)")
                elif last["close"] < v:
                    vwap_score = -0.5; reasons.append("Price below VWAP (bearish bias)")
    except Exception as e:
        reasons.append(f"VWAP calc error: {type(e).__name__}")
    
    # Heikin Ashi slope (trend consistency)
    ha_score = 0.0
    try:
        slope = last.get("ha_slope")
        if slope is not None and not np.isnan(slope):
            if slope > 0:
                ha_score = 0.5
                reasons.append("Heikin Ashi slope rising (trend intact)")
            elif slope < 0:
                ha_score = -0.5
                reasons.append("Heikin Ashi slope falling (trend weakening)")
    except Exception:
        ha_score = 0.0


    # ---------- ADX strength filter ----------
    try:
        adx = last.get("adx") or 0.0
        if adx < 20:
            adx_factor = 0.6
            reasons.append("Weak trend (ADX<20)")
    except Exception:
        adx_factor = 1.0

    # ---------- CHOP regime (robust, always defined) ----------
    try:
        cval = last.get("chop", 50.0)
        if cval is None or (isinstance(cval, float) and np.isnan(cval)):
            cval = 50.0
        if cval > 60:
            chop_factor = 0.7; reasons.append(f"High CHOP {round(float(cval),1)} (ranging)")
        elif cval < 35:
            chop_factor = 1.1; reasons.append(f"Low CHOP {round(float(cval),1)} (trending)")
        else:
            chop_factor = 1.0
    except Exception as e:
        chop_factor = 1.0
        reasons.append(f"CHOP calc error: {type(e).__name__}")
    
        # ---------- Candle Pattern Reasoning ----------
    try:
        if "pattern" in df.columns:
            last_pattern = df["pattern"].dropna().iloc[-1] if not df["pattern"].dropna().empty else None
            if last_pattern:
                if "Hammer" in last_pattern:
                    reasons.append(f"{last_pattern} (bullish reversal)")
                elif "Shooting Star" in last_pattern:
                    reasons.append(f"{last_pattern} (bearish reversal)")
                elif "Engulfing" in last_pattern:
                    if "Bullish" in last_pattern:
                        reasons.append(f"{last_pattern} pattern (bullish momentum shift)")
                    else:
                        reasons.append(f"{last_pattern} pattern (bearish momentum shift)")
                elif last_pattern == "Doji":
                    reasons.append("Doji (market indecision)")
                elif last_pattern == "Marubozu":
                    reasons.append("Marubozu (strong trend continuation)")
                elif last_pattern == "Morning Star":
                    reasons.append("Morning Star (bullish reversal pattern)")
                elif last_pattern == "Evening Star":
                    reasons.append("Evening Star (bearish reversal pattern)")
                elif last_pattern == "Three White Soldiers":
                    reasons.append("Three White Soldiers (strong bullish continuation)")
                elif last_pattern == "Three Black Crows":
                    reasons.append("Three Black Crows (strong bearish continuation)")
                elif last_pattern == "Bullish Harami":
                    reasons.append("Bullish Harami (potential bullish reversal)")
                elif last_pattern == "Bearish Harami":
                    reasons.append("Bearish Harami (potential bearish reversal)")
    except Exception:
        pass


    # ---------- Weighted aggregation ----------
    total_weight = 0.0
    weighted = 0.0
    mapping = [
        ("ema",        ema_score,   BASE_WEIGHTS["ema"]),
        ("macd",       macd_score,  BASE_WEIGHTS["macd"]),
        ("stoch",      stoch_score, BASE_WEIGHTS["stoch"]),
        ("rsi",        rsi_score,   BASE_WEIGHTS["rsi"]),
        ("obv",        obv_score,   BASE_WEIGHTS["obv"]),
        ("supertrend", super_score, BASE_WEIGHTS["supertrend"]),
        ("vwap",       vwap_score,  BASE_WEIGHTS.get("vwap", 0.10)),
        ("ha_slope", ha_score, BASE_WEIGHTS["ha_slope"]),
    ]
    for _, sc, w in mapping:
        if sc is None or w is None:
            continue
        weighted += sc * w
        total_weight += w

    score = 0.0 if total_weight == 0 else (weighted / total_weight) * adx_factor * chop_factor
    score = max(-1.0, min(1.0, score))

    sig = 1 if score >= 0.35 else -1 if score <= -0.35 else 0
    return {"signal": sig, "score": float(score), "reasons": reasons}



def evaluate_signals_mtf(dfs):
    """
    dfs: dict tf->df (order: primary first)
    returns: {
      "signal": "LONG"/"SHORT"/"NEUTRAL",
      "confidence": 0-100 float,
      "reasons": [...],
      "meta": {"signals_per_tf": {tf: "LONG"/"SHORT"/"NEUTRAL", "score": float}}
    }
    """
    if not dfs or not isinstance(dfs, dict):
        return {"signal": "NEUTRAL", "confidence": 0.0, "reasons": ["no_data"], "meta": {"signals_per_tf": {}}}

    # Determine TF order: preserve insertion order from scanner_core (primary first)
    tf_keys = list(dfs.keys())
    per_tf = {}
    reasons = []
    agg_score = 0.0
    weight_sum = 0.0

    # weights by TF: primary heavier
    base_tf_weight = 0.5
    if len(tf_keys) == 1:
        tf_weights = {tf_keys[0]: 1.0}
    else:
        # primary gets base_tf_weight, remaining share rest equally
        remaining = 1.0 - base_tf_weight
        per_rem = remaining / max(1, (len(tf_keys)-1))
        tf_weights = {}
        for i, tf in enumerate(tf_keys):
            tf_weights[tf] = base_tf_weight if i == 0 else per_rem

    for tf in tf_keys:
        res = _per_tf_signal(dfs.get(tf))
        signum = res["signal"]
        score = res["score"]
        per_tf[tf] = {
            "signal": "LONG" if signum == 1 else "SHORT" if signum == -1 else "NEUTRAL",
            "score": float(score),
            "reasons": res.get("reasons", [])
        }
        reasons.extend([f"[{tf}] {r}" for r in res.get("reasons", [])])
        w = tf_weights.get(tf, 0.0)
        agg_score += score * w
        weight_sum += w

    if weight_sum == 0:
        final_score = 0.0
    else:
        final_score = agg_score / weight_sum

    # scale to 0..100
    confidence = abs(final_score) * 100.0

    final_signal = "NEUTRAL"
    if final_score >= 0.35:
        final_signal = "LONG"
    elif final_score <= -0.35:
        final_signal = "SHORT"

    meta = {"signals_per_tf": {tf: per_tf[tf]["signal"] for tf in tf_keys},
            "scores_per_tf": {tf: per_tf[tf]["score"] for tf in tf_keys},
            "tf_weights": tf_weights}

    # create human readable reasons (top unique)
    unique_reasons = []
    for r in reasons:
        if r not in unique_reasons:
            unique_reasons.append(r)
    if not unique_reasons:
        unique_reasons = ["no_significant_reasons"]

    return {"signal": final_signal, "confidence": round(confidence, 1), "reasons": unique_reasons, "meta": meta}
