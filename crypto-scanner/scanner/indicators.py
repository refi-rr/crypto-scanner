# scanner/indicators.py - v4.8
"""
compute_indicators(df)
- Input: pandas DataFrame indexed by datetime with columns: open, high, low, close, volume
- Adds columns: ema_9, ema_21, ema_50, ema_200, macd, macd_signal, macd_hist,
                rsi, stoch_rsi_k, stoch_rsi_d, atr, supertrend, obv, adx,
                bb_upper, bb_lower, bb_width
- Defensive: returns df unchanged if input invalid
"""

import pandas as pd
import numpy as np

def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def _rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def _stoch_rsi(rsi_series, period=14, k_period=3, d_period=3):
    # stoch of RSI
    min_rsi = rsi_series.rolling(period, min_periods=1).min()
    max_rsi = rsi_series.rolling(period, min_periods=1).max()
    denom = (max_rsi - min_rsi).replace(0, np.nan)
    stoch = (rsi_series - min_rsi) / denom
    stoch_k = stoch.rolling(k_period, min_periods=1).mean() * 1.0
    stoch_d = stoch_k.rolling(d_period, min_periods=1).mean()
    return stoch_k.fillna(0), stoch_d.fillna(0)

def _atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr.fillna(0)

def _supertrend(df, period=10, multiplier=3.0):
    """
    Supertrend implementation (returns series of supertrend value and final trend boolean)
    supertrend value stored as 'supertrend', and 'supertrend_dir' True for uptrend else False
    """
    atr = _atr(df, period=period)
    hl2 = (df['high'] + df['low']) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    sup = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="bool")

    # initialize
    prev_final_upper = prev_final_lower = np.nan
    prev_trend = True  # True = uptrend

    for i in range(len(df)):
        if i == 0:
            prev_final_upper = upperband.iat[0]
            prev_final_lower = lowerband.iat[0]
            sup.iat[0] = np.nan
            direction.iat[0] = True
            continue
        cur_up = upperband.iat[i]
        cur_low = lowerband.iat[i]
        prev_close = df['close'].iat[i-1]

        final_upper = cur_up if (cur_up < prev_final_upper or prev_close > prev_final_upper) else prev_final_upper
        final_lower = cur_low if (cur_low > prev_final_lower or prev_close < prev_final_lower) else prev_final_lower

        # determine trend
        if df['close'].iat[i] > final_upper:
            cur_trend = True
        elif df['close'].iat[i] < final_lower:
            cur_trend = False
        else:
            cur_trend = prev_trend

        sup.iat[i] = final_lower if cur_trend else final_upper
        direction.iat[i] = cur_trend

        prev_final_upper = final_upper
        prev_final_lower = final_lower
        prev_trend = cur_trend

    return sup.fillna(method='ffill'), direction.fillna(True)

def _macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def _obv(df):
    # On Balance Volume
    obv = pd.Series(index=df.index, dtype="float64")
    obv.iat[0] = 0.0
    for i in range(1, len(df)):
        if df['close'].iat[i] > df['close'].iat[i-1]:
            obv.iat[i] = obv.iat[i-1] + (df['volume'].iat[i] or 0)
        elif df['close'].iat[i] < df['close'].iat[i-1]:
            obv.iat[i] = obv.iat[i-1] - (df['volume'].iat[i] or 0)
        else:
            obv.iat[i] = obv.iat[i-1]
    return obv.fillna(method='ffill').fillna(0)

def _adx(df, period=14):
    # Wilder's ADX
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().fillna(method='ffill').fillna(0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean().fillna(0)
    return adx

def compute_indicators(df):
    """
    Compute and attach indicators to df. Safely returns df unchanged if invalid input.
    """
    if df is None or len(df) < 3:
        return df

    df = df.copy()

    # ensure numeric
    for c in ['open','high','low','close','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan

    # EMAs
    try:
        df['ema_9'] = _ema(df['close'], 9)
        df['ema_21'] = _ema(df['close'], 21)
        df['ema_50'] = _ema(df['close'], 50)
        df['ema_200'] = _ema(df['close'], 200)
    except Exception:
        df['ema_9'] = df['ema_21'] = df['ema_50'] = df['ema_200'] = np.nan

    # MACD
    try:
        df['macd'], df['macd_signal'], df['macd_hist'] = _macd(df['close'])
    except Exception:
        df['macd'] = df['macd_signal'] = df['macd_hist'] = np.nan

    # RSI
    try:
        df['rsi'] = _rsi(df['close'], period=14)
    except Exception:
        df['rsi'] = np.nan

    # Stochastic RSI
    try:
        k, d = _stoch_rsi(df['rsi'], period=14, k_period=3, d_period=3)
        df['stoch_k'] = k
        df['stoch_d'] = d
    except Exception:
        df['stoch_k'] = df['stoch_d'] = np.nan

    # ATR
    try:
        df['atr'] = _atr(df, period=14)
    except Exception:
        df['atr'] = np.nan

    # Supertrend
    try:
        sup, sup_dir = _supertrend(df, period=10, multiplier=3.0)
        df['supertrend'] = sup
        df['supertrend_dir'] = sup_dir.astype(bool)
    except Exception:
        df['supertrend'] = np.nan
        df['supertrend_dir'] = True

    # OBV
    try:
        df['obv'] = _obv(df)
    except Exception:
        df['obv'] = np.nan

    # ADX
    try:
        df['adx'] = _adx(df, period=14)
    except Exception:
        df['adx'] = np.nan

    # Bollinger Bands
    try:
        ma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = ma20 + 2 * std20
        df['bb_lower'] = ma20 - 2 * std20
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma20.replace(0, np.nan)
    except Exception:
        df['bb_upper'] = df['bb_lower'] = df['bb_width'] = np.nan

    return df
