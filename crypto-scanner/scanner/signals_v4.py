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
    "supertrend": 0.20
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
    if df is None or len(df) < 3:
        return {"signal": 0, "score": 0.0, "reasons": ["no_data"]}

    # use last row
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    # EMA trend: compare ema_9 vs ema_21 and relative to ema_50
    ema_score = 0.0
    try:
        if last.get("ema_9") is not None and last.get("ema_21") is not None:
            if last["ema_9"] > last["ema_21"]:
                ema_score = 1.0
                reasons.append(f"EMA9>{'EMA21'} (bull)")
            elif last["ema_9"] < last["ema_21"]:
                ema_score = -1.0
                reasons.append(f"EMA9<EMA21 (bear)")
    except Exception:
        ema_score = 0.0

    # MACD cross
    macd_score = 0.0
    try:
        macd = last.get("macd")
        macd_sig = last.get("macd_signal")
        prev_macd = prev.get("macd")
        prev_sig = prev.get("macd_signal")
        if macd is not None and macd_sig is not None and prev_macd is not None and prev_sig is not None:
            # golden cross
            if prev_macd < prev_sig and macd > macd_sig:
                macd_score = 1.0
                reasons.append("MACD golden_cross")
            elif prev_macd > prev_sig and macd < macd_sig:
                macd_score = -1.0
                reasons.append("MACD death_cross")
            else:
                # momentum direction
                macd_score = 0.5 if macd > macd_sig else -0.5 if macd < macd_sig else 0.0
    except Exception:
        macd_score = 0.0

    # Stoch RSI
    stoch_score = 0.0
    try:
        k = last.get("stoch_k")
        d = last.get("stoch_d")
        if k is not None and d is not None:
            # oversold -> bullish reversal if cross up under 20
            if prev.get("stoch_k") is not None and prev.get("stoch_d") is not None:
                if prev["stoch_k"] < prev["stoch_d"] and k > d and k < 0.2:
                    stoch_score = 1.0
                    reasons.append("StochRSI cross (oversold) bullish")
                elif prev["stoch_k"] > prev["stoch_d"] and k < d and k > 0.8:
                    stoch_score = -1.0
                    reasons.append("StochRSI cross (overbought) bearish")
                else:
                    # momentum
                    stoch_score = 0.4 if k > d else -0.4 if k < d else 0.0
    except Exception:
        stoch_score = 0.0

    # RSI zones
    rsi_score = 0.0
    try:
        r = last.get("rsi")
        if r is not None:
            if r < 30:
                rsi_score = 0.6
                reasons.append("RSI oversold")
            elif r > 70:
                rsi_score = -0.6
                reasons.append("RSI overbought")
            else:
                # directional tweak
                rsi_score = 0.1 if r > 50 else -0.1
    except Exception:
        rsi_score = 0.0

    # OBV trend (compare last vs mean)
    obv_score = 0.0
    try:
        obv = last.get("obv")
        if obv is not None:
            m = df['obv'].rolling(30, min_periods=1).mean().iloc[-1]
            obv_score = 0.25 if obv > m else -0.25
            if obv > m:
                reasons.append("OBV > MA30 (buy pressure)")
            else:
                reasons.append("OBV < MA30 (sell pressure)")
    except Exception:
        obv_score = 0.0

    # Supertrend direction (True = uptrend)
    super_score = 0.0
    try:
        if last.get("supertrend_dir") is not None:
            if last["supertrend_dir"]:
                super_score = 1.0
                reasons.append("Supertrend up")
            else:
                super_score = -1.0
                reasons.append("Supertrend down")
    except Exception:
        super_score = 0.0

    # ADX strength filter (if ADX low, damp signals)
    adx = last.get("adx") or 0.0
    adx_factor = 1.0
    if adx < 20:
        adx_factor = 0.6
        reasons.append("Weak trend (ADX<20)")

    # Weighted aggregation
    total_weight = 0.0
    weighted = 0.0

    # mapping
    mapping = [
        ("ema", ema_score, BASE_WEIGHTS["ema"]),
        ("macd", macd_score, BASE_WEIGHTS["macd"]),
        ("stoch", stoch_score, BASE_WEIGHTS["stoch"]),
        ("rsi", rsi_score, BASE_WEIGHTS["rsi"]),
        ("obv", obv_score, BASE_WEIGHTS["obv"]),
        ("supertrend", super_score, BASE_WEIGHTS["supertrend"]),
    ]
    for name, sc, w in mapping:
        if sc is None:
            continue
        weighted += sc * w
        total_weight += w

    if total_weight == 0:
        score = 0.0
    else:
        score = (weighted / total_weight) * adx_factor

    # normalize between -1 and 1 (it should already be in that range)
    score = max(-1.0, min(1.0, score))

    # decide signal
    sig = 0
    if score >= 0.35:
        sig = 1
    elif score <= -0.35:
        sig = -1
    else:
        sig = 0

    return {"signal": sig, "score": score, "reasons": reasons}

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
