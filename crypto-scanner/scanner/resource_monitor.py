# ================================================================
# resource_monitor.py — v2.3 (Async-Safe + Jaeger + CPU Core Equiv)
# ================================================================

import psutil
import sqlite3
import os
import time
import asyncio
from datetime import datetime
from opentelemetry import trace

# ================================================================
# KONFIGURASI
# ================================================================
DB_METRIC = "data/scanner_metrics.db"


# ================================================================
# DATABASE SETUP
# ================================================================
def init_perf_db():
    """Inisialisasi database SQLite untuk menyimpan metrik performa scanner."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_METRIC)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scanner_perf (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT,
            cpu REAL,
            cpu_core_equiv REAL,
            memory REAL,
            duration REAL
        )
    """)
    conn.commit()
    conn.close()


def log_performance(symbol: str, cpu: float, mem: float, dur: float):
    """Simpan hasil metrik performa scanner ke DB SQLite."""
    cpu_core_equiv = round(cpu / 100, 3)
    conn = sqlite3.connect(DB_METRIC)
    conn.execute(
        "INSERT INTO scanner_perf (symbol, cpu, cpu_core_equiv, memory, duration) VALUES (?, ?, ?, ?, ?)",
        (symbol, cpu, cpu_core_equiv, mem, dur),
    )
    conn.commit()
    conn.close()


# ================================================================
# SYNC MODE
# ================================================================
def measure_block(symbol: str, func, *args, **kwargs):
    """Jalankan fungsi sinkron sambil mengukur CPU, mem, dan durasi."""
    process = psutil.Process(os.getpid())
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)
    t0 = time.time()

    result = func(*args, **kwargs)

    duration = time.time() - t0
    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)

    cpu_delta = cpu_after - cpu_before
    mem_delta = mem_after - mem_before
    cpu_core_equiv = round(cpu_delta / 100, 3)

    log_performance(symbol, cpu_delta, mem_delta, duration)
    print(
        f"[PERF] {symbol}: CPU {cpu_delta:.1f}% ({cpu_core_equiv:.2f} cores) | "
        f"Mem +{mem_delta:.2f}MB | {duration:.2f}s"
    )

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("resource_monitor") as span:
        span.set_attribute("symbol", symbol)
        span.set_attribute("cpu_usage_percent", cpu_delta)
        span.set_attribute("cpu_core_equiv", cpu_core_equiv)
        span.set_attribute("memory_used_mb", mem_delta)
        span.set_attribute("execution_time_sec", duration)

    return result


# ================================================================
# ASYNC MODE
# ================================================================
async def measure_block_async(symbol: str, coro_func, *args, **kwargs):
    """Jalankan coroutine (fungsi async) sambil mengukur CPU, mem, dan durasi."""
    process = psutil.Process(os.getpid())
    cpu_before = process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 * 1024)
    t0 = time.time()

    result = await coro_func(*args, **kwargs)

    duration = time.time() - t0
    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 * 1024)

    cpu_delta = cpu_after - cpu_before
    mem_delta = mem_after - mem_before
    cpu_core_equiv = round(cpu_delta / 100, 3)

    log_performance(symbol, cpu_delta, mem_delta, duration)
    print(
        f"[PERF] {symbol}: CPU {cpu_delta:.1f}% ({cpu_core_equiv:.2f} cores) | "
        f"Mem +{mem_delta:.2f}MB | {duration:.2f}s (async)"
    )

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("resource_monitor_async") as span:
        span.set_attribute("symbol", symbol)
        span.set_attribute("cpu_usage_percent", cpu_delta)
        span.set_attribute("cpu_core_equiv", cpu_core_equiv)
        span.set_attribute("memory_used_mb", mem_delta)
        span.set_attribute("execution_time_sec", duration)

    return result


# ================================================================
# UTILITAS (OPSIONAL)
# ================================================================
def get_system_resource_snapshot():
    """Ambil snapshot resource server (CPU, RAM, Disk)."""
    cpu_usage = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "cpu_usage_percent": cpu_usage,
        "memory_used_mb": memory.used / (1024 * 1024),
        "memory_total_mb": memory.total / (1024 * 1024),
        "disk_used_gb": disk.used / (1024 * 1024 * 1024),
        "disk_total_gb": disk.total / (1024 * 1024 * 1024),
    }
