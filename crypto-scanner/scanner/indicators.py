# scanner/indicators.py
"""
Advanced indicators: EMA (9,21,50,200), ATR, ADX, RSI, StochRSI, MACD, Bollinger Bands, Volume SMA.
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty or len(df) < 10:
        return df

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # EMA series
    try:
        df['ema_9'] = EMAIndicator(close, window=9).ema_indicator()
        df['ema_21'] = EMAIndicator(close, window=21).ema_indicator()
        df['ema_50'] = EMAIndicator(close, window=50).ema_indicator()
        df['ema_200'] = EMAIndicator(close, window=200).ema_indicator()
    except Exception:
        # fallback using pandas ewm
        df['ema_9'] = close.ewm(span=9, adjust=False).mean()
        df['ema_21'] = close.ewm(span=21, adjust=False).mean()
        df['ema_50'] = close.ewm(span=50, adjust=False).mean()
        df['ema_200'] = close.ewm(span=200, adjust=False).mean()

    # RSI
    try:
        df['rsi'] = RSIIndicator(close, window=14).rsi()
    except Exception:
        df['rsi'] = np.nan

    # Stoch RSI
    try:
        stoch = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
        df['stoch_k'] = stoch.stochrsi_k()
        df['stoch_d'] = stoch.stochrsi_d()
    except Exception:
        df['stoch_k'] = np.nan
        df['stoch_d'] = np.nan

    # MACD
    try:
        macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
    except Exception:
        df['macd'] = df['macd_signal'] = df['macd_diff'] = np.nan

    # Bollinger Bands
    try:
        bb = BollingerBands(close, window=20, window_dev=2)
        df['bb_h'] = bb.bollinger_hband()
        df['bb_m'] = bb.bollinger_mavg()
        df['bb_l'] = bb.bollinger_lband()
    except Exception:
        df['bb_h'] = df['bb_m'] = df['bb_l'] = np.nan

    # ATR
    try:
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        df['atr'] = atr.average_true_range()
    except Exception:
        df['atr'] = np.nan

    # ADX (trend strength)
    try:
        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        df['adx'] = adx.adx()
        df['pdi'] = adx.adx_pos()
        df['mdi'] = adx.adx_neg()
    except Exception:
        df['adx'] = df['pdi'] = df['mdi'] = np.nan

    # Support/Resistance using rolling high/low
    df['sr_support'] = df['low'].rolling(window=20, min_periods=1).min()
    df['sr_resistance'] = df['high'].rolling(window=20, min_periods=1).max()

    # Volume SMA
    df['vol_sma_20'] = volume.rolling(window=20, min_periods=1).mean()
    df['vol_spike'] = df['volume'] / (df['vol_sma_20'] + 1e-9)

    return df
