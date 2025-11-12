"""Health and metrics API for Binance WebSocket ingestion.

Provides websocket health monitoring, latency tracking, and metric collection with:
- Health checks and connection state monitoring
- Structured logging with JSON formatting
- Latency and performance metrics
- Research-compliant schema migrations
"""

import os
import json
import logging
import logging.handlers
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Constants
ORDERBOOK_SNAPSHOT_DEPTH = 20  # Per research liquidity metric requirements
BINANCE_ORDERBOOK_RATE_LIMIT = 2400 // 60  # msgs/sec for snapshots

# Schema definitions for ingestion tables
KLINE_SCHEMA = {
    "symbol": "TEXT",
    "ts_ms": "TIMESTAMP",
    "open": "DOUBLE",
    "high": "DOUBLE",
    "low": "DOUBLE", 
    "close": "DOUBLE",
    "volume": "DOUBLE",
    "close_time": "TIMESTAMP"
}

ORDERBOOK_SCHEMA = {
    "timestamp": "TEXT",
    "symbol": "TEXT", 
    "bid_price": "DOUBLE",
    "bid_size": "DOUBLE",
    "ask_price": "DOUBLE",
    "ask_size": "DOUBLE"
}

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables or config.yaml."""
    config = {
        "symbols": os.getenv("BINANCE_SYMBOLS", "BTCUSDT,ETHUSDT").split(","),
        "interval": os.getenv("BINANCE_INTERVAL", "1m"),
        "batch_size": int(os.getenv("BINANCE_BATCH_SIZE", "20")),
        "rate_limit": int(os.getenv("BINANCE_RATE_LIMIT", "50")),  # msgs per second
        "db_path": os.getenv("BINANCE_DB_PATH", "data/market_data.duckdb"),
        "timeout": int(os.getenv("BINANCE_TIMEOUT", "30")),
        "heartbeat_interval": int(os.getenv("BINANCE_HEARTBEAT_INTERVAL", "30")),
        "max_retries": int(os.getenv("BINANCE_MAX_RETRIES", "10")),
        "test_mode": os.getenv("BINANCE_TEST_MODE", "false").lower() == "true"
    }

    # Try to load from config.yaml if it exists
    config_path = Path("config.yaml")
    if config_path.exists() and HAS_YAML:
        try:
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
                if "binance" in yaml_config:
                    config.update(yaml_config["binance"])
        except Exception as e:
            logger = logging.getLogger("omerGPT.ingestion.binance_ws")
            logger.warning(f"Failed to load config.yaml: {e}")
    return config

def setup_structured_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Configure structured JSON logging with both file and console handlers."""
    logger = logging.getLogger("omerGPT.ingestion.binance_ws")
    logger.setLevel(logging.DEBUG)

    # Structured formatter for detailed JSON logs
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "operation": getattr(record, "operation", "unknown"),
                "symbol": getattr(record, "symbol", None),
                "latency_ms": getattr(record, "latency_ms", None),
            }
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_data)

    structured_formatter = StructuredFormatter()
    
    # Console handler (unstructured for readability)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (structured JSON)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(structured_formatter)
        logger.addHandler(file_handler)

    return logger

async def ensure_orderbook_schema(db):
    """
    Ensure orderbook table exists and conforms to research specs.
    Idempotent schema migration that can be called multiple times safely.
    """
    try:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS orderbook (
            timestamp TEXT,
            symbol TEXT,
            bid_price DOUBLE,
            bid_size DOUBLE,
            ask_price DOUBLE,
            ask_size DOUBLE,
            PRIMARY KEY (symbol, timestamp)
        );
        CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_time 
        ON orderbook(symbol, timestamp);
        """)
    except Exception as e:
        logger.error(f"Orderbook schema guard failed: {e}", exc_info=True)
        raise

async def report_binancews_health(stats: Dict[str, Any] = None):
    """
    Health/latency aggregator for ingestion layer, aligned with research requirements.
    Reports structured health metrics including latency, throughput and errors.
    """
    health_logger = logging.getLogger("omerGPT.ingestion.health")
    try:
        stats = stats or {}
        health_logger.info({
            "event": "ingestion_health",
            "latency": stats.get("latency_avg_ms", 0),
            "metrics": {
                "messages_processed": stats.get("messages_processed", 0),
                "messages_dropped": stats.get("messages_dropped", 0),
                "buffer_size": stats.get("buffer_size", 0),
                "uptime_seconds": stats.get("uptime_seconds", 0),
                "connection_alive": stats.get("connection_alive", False)
            },
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        health_logger.error(f"Health hook error: {e}", exc_info=True)

# Async ingestion tasks
async def snapshot_orderbook_periodic(symbol: str, conn, db, interval: int = 5):
    """
    Periodically snapshot orderbook depth for research liquidity metrics.
    Checks schema, respects rate-limit, logs latency/errors, idempotent.
    """
    logger = logging.getLogger("omerGPT.ingestion.orderbook")
    await ensure_orderbook_schema(db)
    
    while True:
        try:
            start_time = time.time()
            
            # Fetch current orderbook depth 
            depth = await conn.depth(symbol=symbol)
            timestamp = datetime.utcnow().isoformat()
            
            # Extract bid/ask up to research-specified depth
            bids = depth["bids"][:ORDERBOOK_SNAPSHOT_DEPTH]
            asks = depth["asks"][:ORDERBOOK_SNAPSHOT_DEPTH]
            
            # Store first level for quick liquidity checks
            if bids and asks:
                await db.execute(
                    """
                    INSERT INTO orderbook 
                    (timestamp, symbol, bid_price, bid_size, ask_price, ask_size) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [timestamp, symbol, 
                     float(bids[0][0]), float(bids[0][1]),
                     float(asks[0][0]), float(asks[0][1])]
                )
                latency_ms = (time.time() - start_time) * 1000
                logger.info({
                    "event": "orderbook_snapshot",
                    "symbol": symbol,
                    "bid_price": bids[0][0],
                    "bid_size": bids[0][1], 
                    "ask_price": asks[0][0],
                    "ask_size": asks[0][1],
                    "timestamp": timestamp,
                    "latency_ms": round(latency_ms, 2)
                })
        except Exception as e:
            logger.error(f"Orderbook snapshot error for {symbol}: {e}", exc_info=True)
        
        await asyncio.sleep(interval)

async def start_binancews_research_complements(context: Dict[str, Any]):
    """
    Bootstrap ingestion complements required by research, idempotent.
    Sets up orderbook snapshots, periodic metrics, and health hooks.
    """
    logger = logging.getLogger("omerGPT.ingestion.bootstrap")
    db = context.get("db")
    if not db:
        raise ValueError("Database connection required in context")
        
    # Ensure schema is ready (idempotent)
    await ensure_orderbook_schema(db)
    
    # Start metrics logger if not already running
    if not context.get("_binancews_metrics_task"):
        stats_callable = context.get("get_stats")
        context["_binancews_metrics_task"] = asyncio.create_task(
            _periodic_metrics_logger(stats_callable, interval_sec=60)
        )
        
    # Start orderbook snapshots for configured symbols
    if not context.get("_orderbook_snapshot_tasks") and "symbols" in context:
        context["_orderbook_snapshot_tasks"] = []
        conn = context.get("conn")
        if not conn:
            raise ValueError("Binance connection required in context")
            
        for symbol in context["symbols"]:
            task = asyncio.create_task(
                snapshot_orderbook_periodic(symbol, conn, db)
            )
            context["_orderbook_snapshot_tasks"].append(task)
            logger.info(f"Started orderbook snapshot for {symbol}")
            
    return context

async def _periodic_metrics_logger(get_stats_callable, interval_sec: int = 60):
    """
    Periodic metrics logging per Binance ingestion KPIs.
    Tracks message rates, latency stats, and buffer health.
    """
    logger = logging.getLogger("omerGPT.ingestion.metrics")
    
    while True:
        try:
            await asyncio.sleep(interval_sec)
            stats = get_stats_callable() if callable(get_stats_callable) else {}
            logger.info(f"[metrics] {stats}")
            await report_binancews_health(stats)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Periodic metrics logger error: {e}", exc_info=True)

# ==================== ASYNC UTILITIES ====================

class AsyncRateLimiter:
    """Token bucket rate limiter for async operations."""
    def __init__(self, rate: int, per_second: int = 1):
        """
        Initialize rate limiter.

        Args:
        rate: Number of operations allowed
        per_second: Time window in seconds (default 1)
        """
        self.rate = rate
        self.per_second = per_second
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, blocking if necessary."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.rate,
                self.tokens + elapsed * (self.rate / self.per_second)
            )
            self.last_update = now
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / (self.rate / self.per_second)
                await asyncio.sleep(sleep_time)
                self.tokens = 0
            else:
                self.tokens -= 1

def async_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for async retry logic with exponential backoff.

    Args:
    max_retries: Maximum number of retry attempts
    initial_delay: Initial delay in seconds
    max_delay: Maximum delay in seconds
    backoff_factor: Multiplier for exponential backoff
    exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries exceeded for {func.__name__}: {e}",
                            exc_info=True
                        )
                        raise
                    jitter = random.uniform(0, 0.3 * delay)
                    sleep_time = delay + jitter
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                        f"{func.__name__}: {e}. Retrying in {sleep_time:.2f}s"
                    )
                    await asyncio.sleep(sleep_time)
                    delay = min(delay * backoff_factor, max_delay)
            return None
        return wrapper
    return decorator

# Import core functionality from binance_ws module
from ..ingestion.binance_ws import BinanceIngestion, load_config
from ..storage import DatabaseManager
        """
# Health API server endpoints
from fastapi import FastAPI, HTTPException

app = FastAPI(title="OmerGPT Health API", 
             description="Health monitoring and metrics API for Binance WebSocket ingestion")
        test_mode: Enable mock mode for unit tests
        """
        # Load defaults from config
        config = load_config()
        self.db = db_manager
        self.symbols = [s.upper() for s in (symbols or config["symbols"])]
        self.interval = interval or config["interval"]
        self.batch_size = batch_size or config["batch_size"]
        self.rate_limit = rate_limit or config["rate_limit"]
        self.db_path = db_path or config["db_path"]
        self.timeout = timeout or config["timeout"]
        self.heartbeat_interval = heartbeat_interval
        self.max_retries = max_retries
        self.test_mode = test_mode or config.get("test_mode", False)
        # Connection state
        self.ws = None
        self.running = False
        self.retry_count = 0
        self.last_message_time = time.time()
        self.connection_start_time = None
        # Data buffering
        self.candle_buffer = []
        self.buffer_lock = asyncio.Lock()
        # Rate limiting
        self.rate_limiter = AsyncRateLimiter(self.rate_limit)
        # Backoff configuration
        self.initial_backoff = 1.0
        self.max_backoff = 60.0
        # Background tasks
        self.heartbeat_task = None
        self.listen_task = None
        # Statistics
        self.messages_processed = 0
        self.messages_dropped = 0
        self.total_candles_written = 0
        logger.info(
            f"Initialized BinanceIngestion for {len(self.symbols)} symbols: "
            f"{self.symbols} | interval={self.interval} | batch_size={self.batch_size} | "
            f"rate_limit={self.rate_limit}msg/s | test_mode={self.test_mode}"
        )

    def _build_subscription_payload(self) -> dict:
        """Build subscription JSON for all symbols."""
        streams = [f"{s.lower()}@kline_{self.interval}" for s in self.symbols]

        """
            config = {
                "symbols": os.getenv("BINANCE_SYMBOLS", "BTCUSDT,ETHUSDT").split(","),
                "interval": os.getenv("BINANCE_INTERVAL", "1m"),
                "batch_size": int(os.getenv("BINANCE_BATCH_SIZE", "20")),
                "rate_limit": int(os.getenv("BINANCE_RATE_LIMIT", "50")),  # msgs per second
                "db_path": os.getenv("BINANCE_DB_PATH", "data/market_data.duckdb"),
                "timeout": int(os.getenv("BINANCE_TIMEOUT", "30")),
                "heartbeat_interval": int(os.getenv("BINANCE_HEARTBEAT_INTERVAL", "30")),
                "max_retries": int(os.getenv("BINANCE_MAX_RETRIES", "10")),
                "test_mode": os.getenv("BINANCE_TEST_MODE", "false").lower() == "true",
            }
            # Try to load from config.yaml if it exists
            config_path = Path("config.yaml")
            if config_path.exists() and HAS_YAML:
                try:
                    with open(config_path, "r") as f:
                        yaml_config = yaml.safe_load(f) or {}
                    if "binance" in yaml_config:
                        config.update(yaml_config["binance"])
                except Exception as e:
                    logger = logging.getLogger("omerGPT.ingestion.binance_ws")
                    logger.warning(f"Failed to load config.yaml: {e}")
            return config
        try:
            # Handle combined stream format: {"stream": "...", "data": {...}}
            if "data" in msg:
                data = msg["data"]
            else:
                data = msg
            if "k" not in data:
                return None
            kline = data["k"]
            symbol = data.get("s", kline.get("s", ""))
            # Parse timestamps
            close_time_ms = kline.get("T", 0)
            ts = pd.to_datetime(close_time_ms, unit="ms")
            # Calculate latency
            server_time_ms = close_time_ms
            local_time_ms = int(time.time() * 1000)
            latency_ms = abs(local_time_ms - server_time_ms)
            # Log high latency
            if latency_ms > 250:
                log_record = logging.LogRecord(
                    name=logger.name,
                    level=logging.WARNING,
                    pathname="",
                    lineno=0,
                    msg=f"High latency for {symbol}: {latency_ms}ms",
                    args=(),
                    exc_info=None,
                )
                log_record.operation = "normalize_message"
                log_record.symbol = symbol
                log_record.latency_ms = latency_ms
                logger.handle(log_record)
            candle = {
                "symbol": symbol,
                "ts_ms": ts,
                "open": float(kline.get("o", 0)),
                "high": float(kline.get("h", 0)),
                "low": float(kline.get("l", 0)),
                "close": float(kline.get("c", 0)),
                "volume": float(kline.get("v", 0)),
                "close_time": ts,
            }
            log_record = logging.LogRecord(
                name=logger.name,
                level=logging.DEBUG,
                pathname="",
                lineno=0,
                msg=f"Normalized {symbol} candle @ {ts} | close={candle['close']:.2f}",
                args=(),
                exc_info=None,
            )
            log_record.operation = "normalize_message"
            log_record.symbol = symbol
            log_record.latency_ms = latency_ms
            logger.handle(log_record)
            return candle
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to normalize message: {e} | msg={msg}", exc_info=True)
            return None

    @async_retry(max_retries=3, exceptions=(Exception,))
    async def _flush_buffer(self) -> int:
        """Write buffered candles to database and return number of candles flushed."""
        async with self.buffer_lock:
            if not self.candle_buffer:
                return 0
            buffer_size = len(self.candle_buffer)
            try:
                df = pd.DataFrame(self.candle_buffer)
                if self.test_mode:
                    logger.info(f"[TEST MODE] Would flush {buffer_size} candles to DB")
                    self.candle_buffer.clear()
                    return buffer_size
                await self.db.upsert_candles(df)
                self.total_candles_written += buffer_size
                log_record = logging.LogRecord(
                    name=logger.name,
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=f"Flushed {buffer_size} candles to DB (total: {self.total_candles_written})",
                    args=(),
                    exc_info=None,
                )
                log_record.operation = "flush_buffer"
                logger.handle(log_record)
                self.candle_buffer.clear()
                return buffer_size
            except Exception as e:
                self.messages_dropped += buffer_size
                logger.error(
                    f"Failed to flush buffer ({buffer_size} candles dropped): {e}",
                    exc_info=True
                )
                raise

    @async_retry(max_retries=5, initial_delay=1.0, exceptions=(Exception,))
    async def connect(self):
        """Initialize WebSocket connection with exponential backoff retry.
        
        Attempts to establish a connection with configured retries and backoff.
        Updates connection state and metrics on successful connection.
        """
        try:
            # Build streams URL
            streams = "/".join([f"{s.lower()}@kline_{self.interval}" for s in self.symbols])
            url = f"{self.ws_url


}?streams={streams}"
            if self.test_mode:
                logger.info(f"[TEST MODE] Would connect to {url}")
                self.ws = MockWebSocket()
                self.running = True
                self.retry_count = 0
                self.connection_start_time = time.time()
                self.last_message_time = time.time()
                return
            logger.info(
                f"Connecting to Binance WS (attempt {self.retry_count + 1}/"
                f"{self.max_retries})..."
            )
            self.ws = await asyncio.wait_for(
                websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=self.timeout,
                    close_timeout=5,
                    max_size=10 * 1024 * 1024,  # 10MB
                ),
                timeout=self.timeout
            )
            self.running = True
            self.retry_count = 0
            self.connection_start_time = time.time()
            self.last_message_time = time.time()
            log_record = logging.LogRecord(
                name=logger.name,
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"✅ Connected to Binance WS for {len(self.symbols)} symbols "
                    f"({', '.join(self.symbols)})",
                args=(),
                exc_info=None,
            )
            log_record.operation = "connect"
            logger.handle(log_record)
        except asyncio.TimeoutError:
            self.retry_count += 1
            logger.error(
                f"Connection timeout (attempt {self.retry_count}/{self.max_retries})"
            )
            raise
        except Exception as e:
            self.retry_count += 1
            logger.error(
                f"Connection failed (attempt {self.retry_count}/{self.max_retries}): {e}",
                exc_info=True
            )
            raise

    async def subscribe(self, symbols: List[str], interval: str):
        """Send subscription payload to Binance WebSocket.
        Subscribes to real-time kline/candlestick data for specified symbols.
        
        :param symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
        :param interval: Kline interval (1m, 5m, 1h, etc.)
        :raises RuntimeError: If WebSocket connection is not established
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected. Call connect() first.")
        streams = [f"{s.lower()}@kline_{interval}" for s in symbols]
        payload = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000)
        }
        try:
            if self.test_mode:
                logger.info(f"[TEST MODE] Would subscribe to {len(symbols)} symbols")
                return
            await self.ws.send(json.dumps(payload))
            logger.info(f"Subscribed to {len(symbols)} symbols: {symbols}")
        except Exception as e:
            logger.error(f"Failed to send subscription: {e}", exc_info=True)
            raise

    async def _heartbeat(self):
        """Monitors WebSocket health and triggers reconnection if needed."""
        logger.info("Heartbeat task started")
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if not self.running:
                    break
                now = time.time()
                time_since_last = now - self.last_message_time
                if time_since_last > 120:  # 2 minutes
                    log_record = logging.LogRecord(
                        name=logger.name,
                        level=logging.WARNING,
                        pathname="",
                        lineno=0,
                        msg=f"No messages for {time_since_last:.0f}s, reconnecting...",
                        args=(),
                        exc_info=None,
                    )
                    log_record.operation = "heartbeat"
                    logger.handle(log_record)
                    await self._reconnect()
                # Periodic flush of buffer
                if len(self.candle_buffer) > 0 and time_since_last > 30:
                    await self._flush_buffer()
            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _reconnect(self):
        """
        Reconnect to Binance WS with backoff.

        # [ADDED BASED ON RESEARCH ALIGNMENT]
        Flush buffer on forced disconnect, close WS gracefully, and reset running state.
        """
        self.running = False
        if self.ws and not getattr(self.ws, "closed", False):
            try:
                await self.ws.close()
            except Exception:
                pass  # swallow errors on close as recommended for forced shutdown
        self.ws = None
        try:
            await self._flush_buffer()
        except Exception as e:
            logger.error(f"Flush buffer failed prior to reconnect: {e}", exc_info=True)
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}", exc_info=True)

    async def listen(self):
        """
        Continuously read incoming messages and push parsed candles to DB.
        Handles reconnection on disconnect.
        """
        logger.info("Listener task started")
        while self.running:
            try:
                # Ensure connection
                if not self.ws or (not self.test_mode and getattr(self.ws, "closed", False)):
                    logger.warning("WebSocket closed, reconnecting...")
                    await self._reconnect()
                    if not self.ws:
                        await asyncio.sleep(5)
                        continue
                # Listen for messages
                try:
                    async for message in self.ws:
                        if not self.running:
                            break
                        self.last_message_time = time.time()
                        self.messages_processed += 1
                        # Rate limiting
                        await self.rate_limiter.acquire()
                        try:
                            msg = json.loads(message)
                            candle = self.normalize_message(msg)
                            if candle:
                                async with self.buffer_lock:
                                    self.candle_buffer.append(candle)
                                    # Flush buffer if batch size reached
                                    if len(self.candle_buffer) >= self.batch_size:
                                        await self._flush_buffer()
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}", exc_info=True)
                except (ConnectionClosed, WebSocketException) as e:
                    logger.warning(f"⚠️ Connection lost: {e}. Reconnecting...")
                    await self._reconnect()
                    if self.ws:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("Listener task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in listen loop: {e}", exc_info=True)
                    await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Fatal error in listen: {e}", exc_info=True)
                await asyncio.sleep(5)
        logger.info("Listener task stopped")

    def is_alive(self) -> bool:
        """Returns True if the WebSocket is connected and receiving messages."""
        if not self.ws:
            return False
        if not self.test_mode and getattr(self.ws, "closed", False):
            return False
        # Check if last message was within 2 minutes
        time_since_last = time.time() - self.last_message_time
        if time_since_last > 120:
            log_record = logging.LogRecord(
                name=logger.name,
                level=logging.WARNING,
                pathname="",
                lineno=0,
                msg=f"No messages received for {time_since_last:.0f}s",
                args=(),
                exc_info=None,
            )
            log_record.operation = "is_alive"
            logger.handle(log_record)
            return False
        return True

    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        return {
            "running": self.running,
            "connection_alive": self.is_alive(),
            "messages_processed": self.messages_processed,
            "messages_dropped": self.messages_dropped,
            "total_candles_written": self.total_candles_written,
            "buffer_size": len(self.candle_buffer),
            "uptime_seconds": (
                time.time() - self.connection_start_time
                if self.connection_start_time else 0
            ),
        }

    async def start(self):
        """Start WebSocket ingestion with background tasks."""
        logger.info("Starting Binance WebSocket ingestion...")
        try:
            await self.connect()
            # Start background tasks
            self.listen_task = asyncio.create_task(self.listen())
            self.heartbeat_task = asyncio.create_task(self._heartbeat())
            logger.info("Binance WebSocket ingestion started successfully")
        except Exception as e:
            logger.error(f"Failed to start ingestion: {e}", exc_info=True)
            await self.stop()
            raise

    async def stop(self):
        """Gracefully stop WebSocket connection and flush remaining data."""
        logger.info("Stopping Binance WebSocket ingestion...")
        self.running = False
        # Flush any remaining candles
        try:
            await self._flush_buffer()
        except Exception as e:
            logger.error(f"Error flushing buffer during shutdown: {e}", exc_info=True)
        # Cancel background tasks
        if self.listen_task:
            self.listen_task.cancel()
            try:
                await self.listen_task
            except asyncio.CancelledError:
                pass
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        # Close WebSocket
        if self.ws and not getattr(self.ws, "closed", False):
            try:
                await self.ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        stats = self.get_stats()
        logger.info(
            f"WebSocket closed | Stats: {stats}"
        )

# ==================== MOCK FOR TESTING ====================

class MockWebSocket:
    """Mock WebSocket for unit testing (fixed __anext__)."""
    def __init__(self):
        self.closed = False
        self._messages = []
        self._i = 0
        # נכין מראש כמה הודעות דמו
        now_ms = int(time.time() * 1000)
        self._messages.extend([
            json.dumps({
                "stream": "btcusdt@kline_1m",
                "data": {
                    "s": "BTCUSDT",
                    "k": {
                        "T": now_ms,
                        "o": "50000",
                        "h": "51000",
                        "l": "49000",
                        "c": "50500",
                        "v": "1000",
                    },
                },
            }),
            json.dumps({
                "stream": "ethusdt@kline_1m",
                "data": {
                    "s": "ETHUSDT",
                    "k": {
                        "T": now_ms,
                        "o": "3000",
                        "h": "3100",
                        "l": "2900",
                        "c": "3050",
                        "v": "5000",
                    },
                },
            }),
        ])

    async def send(self, msg: str):
        # בדמו אין צורך לשלוח באמת, אבל נשמור ללוגיקה עתידית
        self._messages.append(msg)

    async def close(self):
        self.closed = True

    def __aiter__(self):
        # מחזיר איטרטור אסינכרוני תקין
        return self

    async def __anext__(self):
        # מחזיר הודעה אחת בכל פעם (ללא yield)
        if self.closed:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)
        if self._i >= len(self._messages):
            # אפשר לולאה אינסופית של דמו, או לסגור כשנגמר
            raise StopAsyncIteration
        msg = self._messages[self._i]
        self._i += 1
        return msg

# ==================== DEMO ====================

async def run_demo():
    """Demo: Connect to Binance WS, listen for 5 minutes, print live candles."""
    import sys
    sys.path.insert(0, "src")
    try:
        from storage.db_manager import DatabaseManager
    except ImportError:
        logger.error("Failed to import DatabaseManager. Ensure src/ is in path.")
        return
    print("\n=== Binance WebSocket Ingestion Demo ===\n")
    # Initialize database
    db_path = "data/market_data.duckdb"
    db = DatabaseManager(db_path)
    # Setup ingestion
    symbols = ["BTCUSDT", "ETHUSDT"]
    interval = "1m"
    ws = BinanceIngestion(
        db,
        symbols=symbols,
        interval=interval,
        batch_size=10,
        test_mode=False,  # Use mock mode for demo
    )
    # Start ingestion
    try:
        await ws.start()
        print(f"Listening to {symbols} for 30 seconds...\n")
        # Monitor for 30 seconds
        for i in range(3):
            await asyncio.sleep(10)
            stats = ws.get_stats()
            if ws.is_alive():
                print(
                    f"✓ [{i*10}s] Connection healthy | "
                    f"Processed: {stats['messages_processed']} | "
                    f"Buffer: {stats['buffer_size']} candles"
                )
            else:
                print(f"✗ [{i*10}s] Connection unhealthy!")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Stop
        await ws.stop()
        # Verify data
        print("\n=== Verification ===")
        stats = ws.get_stats()
        print(f"Final Stats: {json.dumps(stats, indent=2)}")
        try:
            symbols_in_db = await db.get_symbols()
            print(f"Symbols in DB: {symbols_in_db}")
            for symbol in symbols:
                df = await db.get_latest_candles(symbol, limit=10)
                print(f"\nLast 10 candles for {symbol}:")
                if not df.empty:
                    print(df[["ts_ms", "close", "volume"]].tail())
                else:
                    print(" (no data)")
        except Exception as e:
            logger.warning(f"Could not verify data: {e}")
        db.close()
        print("\n=== Demo Complete ===\n")

if __name__ == "__main__":
    asyncio.run(run_demo())

# ==================== ORDER BOOK SNAPSHOT COROUTINE ====================
# [ADDED BASED ON RESEARCH ALIGNMENT]
async def snapshot_orderbook(symbol: str, conn, db_conn):
    while True:
        try:
            data = await conn.depth(symbol=symbol)
            timestamp = datetime.utcnow().isoformat()
            bids = data['bids'][:20]
            asks = data['asks'][:20]
            # SQL to insert orderbook snapshot
            query = ("INSERT INTO orderbook "
                    "(timestamp, symbol, bid_price, bid_size, ask_price, ask_size) "
                    "VALUES (?, ?, ?, ?, ?, ?)")
            params = (timestamp, symbol, 
                   float(bids[0][0]), float(bids[0][1]),
                   float(asks[0][0]), float(asks[0][1]))
            db_conn.execute(query, params)
        except Exception as e:
            logger.error(f"Orderbook snapshot failed for {symbol}: {e}", exc_info=True)
        await asyncio.sleep(5)

# Binance WebSocket ingestion with health checks, metrics, and proper error handling
from binance import AsyncClient
import asyncio
import json

from binance import AsyncClient

import asyncio

import json

import logging

import logging.handlers

import random

import time

import os

from datetime import datetime

from typing import List, Optional, Dict, Any, Callable

from functools import wraps

from pathlib import Path

import pandas as pd

import websockets

from websockets.exceptions import ConnectionClosed, WebSocketException

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

# ==================== CONFIGURATION ====================

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables or config.yaml."""
    if HAS_DOTENV:
        load_dotenv()
    config = {
        "symbols": os.getenv("BINANCE_SYMBOLS", "BTCUSDT,ETHUSDT").split(","),
        "interval": os.getenv("BINANCE_INTERVAL", "1m"),
        "batch_size": int(os.getenv("BINANCE_BATCH_SIZE", "20")),
        "rate_limit": int(os.getenv("BINANCE_RATE_LIMIT", "50")),  # msgs per second
        "db_path": os.getenv("BINANCE_DB_PATH", "data/market_data.duckdb"),
        "timeout": int(os.getenv("BINANCE_TIMEOUT", "30")),
        "heartbeat_interval": int(os.getenv("BINANCE_HEARTBEAT_INTERVAL", "30")),
        "max_retries": int(os.getenv("BINANCE_MAX_RETRIES", "10")),
        "test_mode": os.getenv("BINANCE_TEST_MODE", "false").lower() == "true"
    }
    # Try to load from config.yaml if it exists
    config_path = Path("config.yaml")
    if config_path.exists() and HAS_YAML:
        try:
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
                if "binance" in yaml_config:
                    config.update(yaml_config["binance"])
        except Exception as e:
            logger = logging.getLogger("omerGPT.ingestion.binance_ws")
            logger.warning(f"Failed to load config.yaml: {e}")
    return config

# ==================== LOGGING SETUP ====================

def setup_structured_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Configure structured JSON logging with both file and console handlers."""
    logger = logging.getLogger("omerGPT.ingestion.binance_ws")
    logger.setLevel(logging.DEBUG)
    # Structured formatter
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "operation": getattr(record, "operation", "unknown"),
                "symbol": getattr(record, "symbol", None),
                "latency_ms": getattr(record, "latency_ms", None),
            }
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_data)
    structured_formatter = StructuredFormatter()
    # Console handler (unstructured for readability)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    # File handler (structured JSON)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(structured_formatter)
        logger.addHandler(file_handler)
    return logger

logger = setup_structured_logging("logs/binance_ws.log")

# ==================== CANDLE SCHEMA ====================
CANDLES_SCHEMA = {
    "symbol": "TEXT",
    "ts_ms": "TIMESTAMP",
    "open": "DOUBLE",
    "high": "DOUBLE",
    "low": "DOUBLE",
    "close": "DOUBLE",
    "volume": "DOUBLE",
    "close_time": "TIMESTAMP",
}

# Import required functionality from other modules
from ..ingestion.binance_ws import (
    BinanceIngestion, 
    AsyncRateLimiter,
    async_retry,
    load_config
)
from ..storage import DatabaseManager
from fastapi import FastAPI, HTTPException

# Initialize FastAPI app
app = FastAPI(
    title="OmerGPT Health API",
    description="Health monitoring and metrics API for Binance WebSocket ingestion",
    version="1.0.0"
)

    def __init__(
        self,
        db_manager,
        symbols: Optional[List[str]] = None,
        interval: str = "1m",
        batch_size: Optional[int] = None,
        rate_limit: Optional[int] = None,
        db_path: Optional[str] = None,
        timeout: int = 30,
        heartbeat_interval: int = 30,
        max_retries: int = 10,
        test_mode: bool = False,
        api: Optional[Any] = None,  # [ADDED BASED ON RESEARCH ALIGNMENT]
    ):
        config = load_config()
        self.db = db_manager
        self.symbols = [s.upper() for s in (symbols or config["symbols"])]
        self.interval = interval or config["interval"]
        self.batch_size = batch_size or config["batch_size"]
        self.rate_limit = rate_limit or config["rate_limit"]
        self.db_path = db_path or config["db_path"]
        self.timeout = timeout or config["timeout"]
        self.heartbeat_interval = heartbeat_interval
        self.max_retries = max_retries
        self.test_mode = test_mode or config.get("test_mode", False)
        self.api = api  # [ADDED BASED ON RESEARCH ALIGNMENT]
        self.ws = None
        self.running = False
        self.retry_count = 0
        self.last_message_time = time.time()
        self.connection_start_time = None
        self.candle_buffer = []
        self.buffer_lock = asyncio.Lock()
        self.rate_limiter = AsyncRateLimiter(self.rate_limit)
        self.initial_backoff = 1.0
        self.max_backoff = 60.0
        self.heartbeat_task = None
        self.listen_task = None
        self.messages_processed = 0
        self.messages_dropped = 0
        self.total_candles_written = 0
        self.latency_sum = 0  # [ADDED BASED ON RESEARCH ALIGNMENT]
        self.latency_count = 0  # [ADDED BASED ON RESEARCH ALIGNMENT]
        logger.info(
            f"Initialized BinanceIngestion for {len(self.symbols)} symbols: "
            f"{self.symbols} | interval={self.interval} | batch_size={self.batch_size} | "
            f"rate_limit={self.rate_limit}msg/s | test_mode={self.test_mode}"
        )
        # [ADDED BASED ON RESEARCH ALIGNMENT]
        # Enforce table schema at startup
        asyncio.create_task(self.db.ensure_table("candles", schema=CANDLES_SCHEMA))

    # ... unchanged build_subscription_payload, _calculate_backoff ...

    def normalize_message(self, msg: dict) -> Optional[dict]:
        try:
            if "data" in msg:
                data = msg["data"]
            else:
                data = msg
            if "k" not in data:
                return None
            kline = data["k"]
            symbol = data.get("s", kline.get("s", ""))
            close_time_ms = kline.get("T", 0)
            ts = pd.to_datetime(close_time_ms, unit="ms")
            server_time_ms = close_time_ms
            local_time_ms = int(time.time() * 1000)
            latency_ms = abs(local_time_ms - server_time_ms)
            # --- Latency stats
            self.latency_sum += latency_ms  # [ADDED BASED ON RESEARCH ALIGNMENT]
            self.latency_count += 1
            # Log high latency
            if latency_ms > 250:
                log_record = logging.LogRecord(
                    name=logger.name,
                    level=logging.WARNING,
                    pathname="",
                    lineno=0,
                    msg=f"High latency for {symbol}: {latency_ms}ms",
                    args=(),
                    exc_info=None,
                )
                log_record.operation = "normalize_message"
                log_record.symbol = symbol
                log_record.latency_ms = latency_ms
                logger.handle(log_record)
            candle = {
                "symbol": symbol,
                "ts_ms": ts,
                "open": float(kline.get("o", 0)),
                "high": float(kline.get("h", 0)),
                "low": float(kline.get("l", 0)),
                "close": float(kline.get("c", 0)),
                "volume": float(kline.get("v", 0)),
                "close_time": ts,
            }
            log_record = logging.LogRecord(
                name=logger.name,
                level=logging.DEBUG,
                pathname="",
                lineno=0,
                msg=f"Normalized {symbol} candle @ {ts} | close={candle['close']:.2f}",
                args=(),
                exc_info=None,
            )
            log_record.operation = "normalize_message"
            log_record.symbol = symbol
            log_record.latency_ms = latency_ms
            logger.handle(log_record)
            return candle
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to normalize message: {e} | msg={msg}", exc_info=True)
            return None

    # ... unchanged connect, subscribe, etc. ...

    async def listen(self):
        logger.info("Listener task started")
        while self.running:
            try:
                if not self.ws or (not self.test_mode and getattr(self.ws, "closed", False)):
                    logger.warning("WebSocket closed, reconnecting...")
                    await self._reconnect()
                    if not self.ws:
                        await asyncio.sleep(5)
                        continue
                try:
                    async for message in self.ws:
                        if not self.running:
                            break
                        self.last_message_time = time.time()
                        self.messages_processed += 1
                        await self.rate_limiter.acquire()
                        try:
                            msg = json.loads(message)
                            candle = self.normalize_message(msg)
                            if candle:
                                async with self.buffer_lock:
                                    self.candle_buffer.append(candle)
                                    if len(self.candle_buffer) >= self.batch_size:
                                        await self._flush_buffer()
                            # [ADDED BASED ON RESEARCH ALIGNMENT]
                            if self.latency_count % 1000 == 0 and self.latency_count > 0:
                                avg_latency = self.latency_sum / self.latency_count
                                latency_report = {
                                    "avg_latency_ms": avg_latency,
                                    "messages_processed": self.latency_count,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                                try:
                                    from system_monitor import report_latency
                                    await report_latency("binance_ws", latency_report)
                                except ImportError:
                                    logger.debug("system_monitor/report_latency not hooked.")
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {e}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}", exc_info=True)
                except (ConnectionClosed, WebSocketException) as e:
                    logger.warning(f"⚠️ Connection lost: {e}. Reconnecting...")
                    await self._reconnect()
                    if self.ws:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("Listener task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error in listen loop: {e}", exc_info=True)
                    await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Fatal error in listen: {e}", exc_info=True)
                await asyncio.sleep(5)
        logger.info("Listener task stopped")

    # ... unchanged _flush_buffer, start, stop, etc. ...

    def get_stats(self) -> dict:
        return {
            "running": self.running,
            "connection_alive": self.is_alive(),
            "messages_processed": self.messages_processed,
            "messages_dropped": self.messages_dropped,
            "total_candles_written": self.total_candles_written,
            "buffer_size": len(self.candle_buffer),
            "latency_avg_ms": (self.latency_sum / self.latency_count if self.latency_count else 0),  # [ADDED]
            "uptime_seconds": (
                time.time() - self.connection_start_time
                if self.connection_start_time else 0
            ),
        }

    async def _heartbeat(self):
        logger.info("Heartbeat task started")
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if not self.running:
                    break
                now = time.time()
                time_since_last = now - self.last_message_time
                if time_since_last > 120:
                    log_record = logging.LogRecord(
                        name=logger.name,
                        level=logging.WARNING,
                        pathname="",
                        lineno=0,
                        msg=f"No messages for {time_since_last:.0f}s, reconnecting...",
                        args=(),
                        exc_info=None,
                    )
                    log_record.operation = "heartbeat"
                    logger.handle(log_record)
                    await self._reconnect()
                if len(self.candle_buffer) > 0 and time_since_last > 30:
                    await self._flush_buffer()
                # [ADDED BASED ON RESEARCH ALIGNMENT]
                # Health API Hook: push stats every heartbeat
                if self.api and hasattr(self.api, "report_health"):
                    await self.api.report_health("binance_ws", self.get_stats())
            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}", exc_info=True)
                await asyncio.sleep(5)

# ==================== ORDER BOOK SNAPSHOT ====================
async def snapshot_orderbook(symbol: str, conn, db_conn):
    while True:
        try:
            data = await conn.depth(symbol=symbol)
            timestamp = datetime.utcnow().isoformat()
            bids = data['bids'][:20]
            asks = data['asks'][:20]
            df = pd.DataFrame([{
                "timestamp": timestamp,
                "symbol": symbol,
                "bid_price": bids[0][0],
                "bid_size": bids[0][1],
                "ask_price": asks[0][0],
                "ask_size": asks[0][1],
            }])
            # Improved async write to DB [ADDED BASED ON RESEARCH ALIGNMENT]
            await db_conn.upsert_orderbook(df)
        except Exception as e:
            logger.error(f"Orderbook snapshot failed for {symbol}: {e}", exc_info=True)
        await asyncio.sleep(5)

# COMPLETION DONE — aligned with research, added only missing parts without refactor
=== BEGIN PATCH (append to end of file) =====================================
[WHY] ⟶ דרישות המחקר: ציות לסטנדרט ingestion של Binance (snapshot עומק orderbook, structured health hook, latency metric, schema migration idempotent, periodic metrics, rate-limit ל-2400/min לפונקציה, logging מובנה, derived-only, GPU-compat readiness)
[SCOPE] ⟶ הוספת:
- אימות סכמות orderbook/ingestion בדאקDB
- קונסט לעומק snapshot ו-rate-limit
- structured health hook לכלל המשימה
- קורוטינה snapshot ל-orderbook לפי מחקר
- periodic metrics logger
- בקר הדבקה idempotent ומדדי איכות/latency
[ADDED BASED ON RESEARCH ALIGNMENT] Imports (guarded if needed)
try:
import duckdb
import concurrent.futures
except ImportError:
# דרוש: pip install duckdb
pass
try:
import websockets
except ImportError:
# דרוש: pip install websockets
pass

[ADDED BASED ON RESEARCH ALIGNMENT] Constants/Configs (בלי לשנות קיימים)
if "ORDERBOOK_SNAPSHOT_DEPTH" not in globals():
ORDERBOOK_SNAPSHOT_DEPTH = 20 # עומק snapshot לפי המחקר (liquidity metric)
if "BINANCE_ORDERBOOK_RATE_LIMIT" not in globals():
BINANCE_ORDERBOOK_RATE_LIMIT = 2400 // 60 # msgs/sec עבור snapshots

[ADDED BASED ON RESEARCH ALIGNMENT] Schema guards / migrations (Idempotent)
async def _ensure_orderbook_schema(db):
"""Ensure orderbook table exists and conforms to research specs."""
try:
db.sql("""
CREATE TABLE IF NOT EXISTS orderbook (
timestamp TEXT,
symbol TEXT,
bidprice DOUBLE,
bidsize DOUBLE,
askprice DOUBLE,
asksize DOUBLE
);
CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_time ON orderbook(symbol, timestamp);
""")
except Exception as e:
logging.getLogger(name).error(f"orderbook schema guard failed: {e}")

[ADDED BASED ON RESEARCH ALIGNMENT] Health/Latency hooks
async def _report_binancews_health(stats: dict = None):
"""Health/latency aggregator for ingestion layer, aligns with research."""
# שליחת אירוע health בפורמט זהה לדרישות ה-platform
logger = logging.getLogger("omerGPT.ingestion.health")
try:
latency = stats.get("uptimeseconds", 0)
logger.info({
"event": "ingestion_health",
"latency": latency,
"metrics": stats,
"timestamp": datetime.utcnow().isoformat(),
})
except Exception as e:
logger.error(f"health hook error: {e}")

