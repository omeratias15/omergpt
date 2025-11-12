"""
src/ingestion/historical_loader.py

Binance REST Historical Ingestion Full Script

Compatible with omerGPT + DuckDB, including is_closed field required by db_manager

Production-ready, fault-tolerant, async, and configurable.
"""

import sys, os, time, logging, json, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import asyncio
import pandas as pd
from typing import List

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

import aiohttp

from storage.db_manager import DatabaseManager

logger = logging.getLogger("omerGPT.ingestion.historical_loader")

def load_config():
    # Supports .env or configs/config.yaml
    if HAS_DOTENV:
        load_dotenv()
    cfg = {
        "symbols": os.getenv("BINANCE_HIST_SYMBOLS", "").split(",") if os.getenv("BINANCE_HIST_SYMBOLS", "") else ["BTCUSDT", "ETHUSDT"],
        "interval": os.getenv("BINANCE_HIST_INTERVAL", "1h"),
        "days_back": int(os.getenv("BINANCE_HIST_DAYS_BACK", "365")),
        "db_path": os.getenv("MARKET_DB_PATH", "data/market_data.duckdb"),
        "batch_size": int(os.getenv("BINANCE_HIST_BATCH_SIZE", "1000")),
        "rate_limit": int(os.getenv("BINANCE_HIST_RATE_LIMIT", "10")),
        "logfile": os.getenv("BINANCE_HIST_LOG", None)
    }
    cfg_path = "configs/config.yaml"
    if HAS_YAML and os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                ycfg = yaml.safe_load(f)
                for k, v in ycfg.get("historical_loader", {}).items():
                    cfg[k] = v
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}")
    return cfg

def setup_structured_logging(logfile=None):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_dict = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage()
            }
            return json.dumps(log_dict)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fileh = logging.FileHandler(logfile)
        fileh.setLevel(logging.INFO)
        fileh.setFormatter(JsonFormatter())
        logger.addHandler(fileh)

async def async_backoff(func, *args, retries=5, initial=1.0, maxdelay=30.0, **kwargs):
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            back = min(initial * (2**attempt), maxdelay) + random.uniform(0, min(0.5, 0.3*initial))
            logger.warning(f"Retry {attempt+1}/{retries} after error: {e}. Backoff {back:.2f}s")
            await asyncio.sleep(back)
    logger.error(f"Max retries exceeded for {func.__name__}")
    return None

async def fetch_binance_ohlcv_async(symbol: str, interval: str, start_ts: int, end_ts: int, batch_size: int = 1000, rate_limit: int = 10) -> pd.DataFrame:
    # Async, robust, auto-retry
    all_rows: List = []
    ts = start_ts
    api = "https://api.binance.com/api/v3/klines"
    sem = asyncio.Semaphore(rate_limit)

    async with aiohttp.ClientSession() as session:
        while ts < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": ts,
                "endTime": end_ts,
                "limit": batch_size
            }
            await sem.acquire()
            try:
                async with session.get(api, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    response = await resp.json()
            except Exception as e:
                logger.warning(json.dumps({
                    "operation":"fetch_ohlcv",
                    "symbol":symbol,
                    "interval":interval,
                    "start":ts,
                    "err":str(e)
                }))
                await asyncio.sleep(2)
                continue
            finally:
                sem.release()
            if not response or type(response) != list:
                logger.error(json.dumps({
                    "operation":"fetch_ohlcv",
                    "symbol":symbol,
                    "interval":interval,
                    "params":params,
                    "result":"empty"
                }))
                break
            all_rows.extend(response)
            ts_new = response[-1][0] + 1
            logger.info(json.dumps({
                "operation":"batch_fetch",
                "symbol":symbol,
                "interval":interval,
                "rows":len(response),
                "from":response[0][0],
                "to":response[-1][0]
            }))
            if ts_new == ts or len(response) < batch_size:
                break
            ts = ts_new
            await asyncio.sleep(0.2)
    if not all_rows:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return df

def df_to_records(df: pd.DataFrame, symbol: str, interval: str):
    df['symbol'] = symbol
    df['interval'] = interval
    df['ts_ms'] = df['timestamp'].astype('int64') // 10**6
    df['is_closed'] = True
    records = df[['symbol', 'interval', 'ts_ms', 'open', 'high', 'low', 'close', 'volume', 'is_closed']].to_dict('records')
    return records

async def ingest_historical(symbols: List[str], interval: str, days_back: int, db_path: str = "data/market_data.duckdb", batch_size: int = 1000, rate_limit: int = 10):
    start = int((time.time() - days_back * 24 * 3600) * 1000)
    now = int(time.time() * 1000)
    db = DatabaseManager(db_path)
    for symbol in symbols:
        logger.info(json.dumps({
            "operation": "begin_download",
            "symbol": symbol,
            "interval":interval
        }))
        df = await async_backoff(
            fetch_binance_ohlcv_async,
            symbol, interval, start, now,
            batch_size=batch_size,
            rate_limit=rate_limit
        )
        if df is None:
            logger.error(json.dumps({
                "operation": "abort_symbol",
                "symbol": symbol
            }))
            continue
        logger.info(json.dumps({
            "operation": "download_done",
            "symbol": symbol,
            "interval": interval,
            "candles": len(df)
        }))
        records = df_to_records(df, symbol, interval)
        try:
            await db.insert_candles_batch(records)
            logger.info(json.dumps({
                "operation": "db_insert",
                "symbol": symbol,
                "records": len(records)
            }))
        except Exception as e:
            logger.error(json.dumps({
                "operation": "db_error",
                "symbol": symbol,
                "error": str(e)
            }))
    db.close()
    logger.info(json.dumps({"operation":"shutdown"}))

async def shutdown():
    logger.info(json.dumps({"operation": "shutdown"}))

if __name__ == "__main__":
    cfg = load_config()
    setup_structured_logging(cfg.get("logfile", None))
    asyncio.run(ingest_historical(
        symbols=cfg["symbols"],
        interval=cfg["interval"],
        days_back=cfg["days_back"],
        db_path=cfg["db_path"],
        batch_size=cfg["batch_size"],
        rate_limit=cfg["rate_limit"]
    ))
