# scanner/history.py
"""
History storage & simple backtester using SQLite.
- Stores signals (timestamp, symbol, timeframe_set, signal, confidence, entry, tp1, tp2, sl, reasons, meta)
- Simple backtest: check next N candles for TP/SL hit (naive simulation)
"""

import sqlite3
import json
import time
from typing import Dict, Any, Optional, List
import pandas as pd

DB_PATH = "scanner_signals_history.db"

def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER,
        symbol TEXT,
        timeframes TEXT,
        signal TEXT,
        confidence INTEGER,
        entry REAL,
        tp1 REAL,
        tp2 REAL,
        sl REAL,
        reasons TEXT,
        meta TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_signal(record: Dict[str, Any]):
    """
    record fields: ts (int), symbol, timeframes (list), signal, confidence, entry, tp1, tp2, sl, reasons (list), meta (dict)
    """
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO signals (ts, symbol, timeframes, signal, confidence, entry, tp1, tp2, sl, reasons, meta)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        int(record.get('ts', time.time())),
        record['symbol'],
        json.dumps(record.get('timeframes', [])),
        record['signal'],
        int(record.get('confidence', 0)),
        record.get('entry'),
        record.get('tp1'),
        record.get('tp2'),
        record.get('sl'),
        json.dumps(record.get('reasons', [])),
        json.dumps(record.get('meta', {}))
    ))
    conn.commit()
    conn.close()

def list_signals(limit: int = 100) -> List[Dict[str, Any]]:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT ts, symbol, timeframes, signal, confidence, entry, tp1, tp2, sl, reasons, meta FROM signals ORDER BY ts DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            'ts': r[0],
            'symbol': r[1],
            'timeframes': json.loads(r[2]),
            'signal': r[3],
            'confidence': r[4],
            'entry': r[5],
            'tp1': r[6],
            'tp2': r[7],
            'sl': r[8],
            'reasons': json.loads(r[9] or '[]'),
            'meta': json.loads(r[10] or '{}'),
        })
    return out

def simple_backtest_signal(df_price: pd.DataFrame, rec: Dict[str, Any], lookahead: int = 24) -> Dict[str, Any]:
    """
    Naive backtest: given historical df_price (with index datetime, columns open/high/low/close),
    and a signal record rec with entry/tp1/sl, simulate next 'lookahead' candles.
    Returns outcome dict: {'result': 'win_tp1'|'win_tp2'|'hit_sl'|'no_hit', 'hit_index': i or None}
    Logic: within next N candles, check if high >= tp1 (for LONG) before low <= sl; prefer TP2 if hit.
    NOTE: this is naive intrabar-check using candle high/low sequence (not tick data).
    """
    outcome = {'result': 'no_hit', 'hit_index': None, 'notes': ''}
    if df_price is None or df_price.empty:
        outcome['notes'] = 'no price data'
        return outcome
    sig = rec.get('signal')
    entry = rec.get('entry')
    tp1 = rec.get('tp1')
    tp2 = rec.get('tp2')
    sl = rec.get('sl')
    # iterate next candles
    df2 = df_price.copy().reset_index(drop=True)  # assume df_price is ordered oldest->newest, and includes next candles
    n = min(lookahead, len(df2))
    for i in range(n):
        hi = float(df2.loc[i, 'high'])
        lo = float(df2.loc[i, 'low'])
        # LONG: check tp1/tp2 first or sl first? We'll prefer TP if reached earlier in same candle
        if sig == 'LONG':
            # if candle both high >= tp1 and low <= sl in same candle, ambiguous: approximate by proximity to open
            if hi >= tp1:
                # check if hi >= tp2 as well
                if tp2 and hi >= tp2:
                    return {'result': 'win_tp2', 'hit_index': i}
                return {'result': 'win_tp1', 'hit_index': i}
            if lo <= sl:
                return {'result': 'hit_sl', 'hit_index': i}
        elif sig == 'SHORT':
            if lo <= tp1:
                if tp2 and lo <= tp2:
                    return {'result': 'win_tp2', 'hit_index': i}
                return {'result': 'win_tp1', 'hit_index': i}
            if hi >= sl:
                return {'result': 'hit_sl', 'hit_index': i}
    return outcome
