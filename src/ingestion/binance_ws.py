"""
src/ingestion/binance_ws.py

Production-grade asynchronous Binance WebSocket ingestion module.

Streams live candle data into DuckDB via DatabaseManager with fault tolerance,
backoff, rate limiting, structured logging, health checks, and graceful shutdown.
"""
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


# ==================== BINANCE INGESTION ====================

class BinanceIngestion:
    """
    Async Binance WebSocket ingestion for multi-symbol real-time kline data.
    
    Features:
    - Exponential backoff with jitter on reconnect
    - Async rate limiting for message processing
    - Structured JSON logging with operation tracking
    - Latency monitoring (server_time vs local_time)
    - Batch DB writes via DatabaseManager
    - Health check & auto-reconnect with heartbeat task
    - Graceful shutdown with buffer flush
    - Error recovery and retry logic
    - Runtime configuration via .env or config.yaml
    - Mock mode for unit testing
    """
    
    ws_url = "wss://stream.binance.com:9443/stream"
    
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
        
    ):
        """
        Initialize Binance WebSocket ingestion.
        
        Args:
            db_manager: DatabaseManager instance for persistence
            symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
                    If None, loaded from config
            interval: Kline interval (1m, 5m, 1h, etc.)
            batch_size: Number of candles to buffer before DB write
            rate_limit: Messages per second limit
            db_path: Path to DuckDB database
            timeout: WebSocket timeout in seconds
            heartbeat_interval: Health check interval in seconds
            max_retries: Maximum reconnection attempts
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
        # Use a bounded asyncio.Queue for backpressure and memory safety
        queue_size = int(os.getenv("BINANCE_QUEUE_MAXSIZE", "5000"))
        self.candle_queue = asyncio.Queue(maxsize=queue_size)
        self.flush_interval = int(os.getenv("BINANCE_FLUSH_INTERVAL", "1"))  # seconds
        
        # Rate limiting
        self.rate_limiter = AsyncRateLimiter(self.rate_limit)
        
        # Backoff configuration
        self.initial_backoff = 1.0
        self.max_backoff = 60.0
        
        # Background tasks
        self.heartbeat_task = None
        self.listen_task = None
        self.writer_task = None
        
        # Statistics
        self.messages_processed = 0
        self.messages_dropped = 0
        self.total_candles_written = 0
        # DB/metrics
        self.db_write_failures = 0
        self.db_write_time_total = 0.0
        
        logger.info(
            f"Initialized BinanceIngestion for {len(self.symbols)} symbols: "
            f"{self.symbols} | interval={self.interval} | batch_size={self.batch_size} | "
            f"rate_limit={self.rate_limit}msg/s | test_mode={self.test_mode}"
        )
    
    def _build_subscription_payload(self) -> dict:
        """Build subscription JSON for all symbols."""
        streams = [f"{s.lower()}@kline_{self.interval}" for s in self.symbols]
        return {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000)
        }
    
    def _calculate_backoff(self) -> float:
        """Calculate exponential backoff with jitter."""
        backoff = min(
            self.initial_backoff * (2 ** self.retry_count),
            self.max_backoff
        )
        jitter = random.uniform(0, 0.3 * backoff)
        return backoff + jitter
    
    def normalize_message(self, msg: dict) -> Optional[dict]:
        """
        Convert raw Binance WS message to normalized candle dict.
        
        Args:
            msg: Raw WebSocket message JSON
        
        Returns:
            Normalized candle dict or None if invalid
        """
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
            
            # Use integer millisecond timestamps to match DB schema and avoid
            # timezone/precision ambiguity. DuckDB/DatabaseManager expects
            # ts_ms/close_time as integers (ms) and will convert as needed.
            candle = {
                "symbol": symbol,
                "ts_ms": int(close_time_ms),
                "open": float(kline.get("o", 0)),
                "high": float(kline.get("h", 0)),
                "low": float(kline.get("l", 0)),
                "close": float(kline.get("c", 0)),
                "volume": float(kline.get("v", 0)),
                "close_time": int(close_time_ms),
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
        """
        Flush candle buffer to database.
        
        Returns:
            Number of candles flushed
        """
        """
        Drain the candle queue and flush all items to the DB. This method is
        used for shutdown/heartbeat flushes where we want to persist whatever
        is in the queue immediately.
        """
        drained = []
        try:
            while not self.candle_queue.empty():
                drained.append(self.candle_queue.get_nowait())
        except Exception:
            # get_nowait can raise if racey; ignore and continue with what we have
            pass

        buffer_size = len(drained)
        if buffer_size == 0:
            return 0

        try:
            df = pd.DataFrame(drained)

            if self.test_mode:
                logger.info(f"[TEST MODE] Would flush {buffer_size} candles to DB")
                return buffer_size

            start = time.time()
            await self.db.upsert_candles(df)
            elapsed = time.time() - start
            self.db_write_time_total += elapsed
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

            return buffer_size

        except Exception as e:
            self.messages_dropped += buffer_size
            self.db_write_failures += 1
            logger.error(
                f"Failed to flush buffer ({buffer_size} candles dropped): {e}",
                exc_info=True
            )
            raise

    async def _writer_loop(self):
        """
        Background writer that batches candles from the queue and flushes to DB.
        Collects up to self.batch_size items or flushes after self.flush_interval.
        """
        logger.info("Writer task started")
        batch = []
        while self.running:
            try:
                try:
                    item = await asyncio.wait_for(self.candle_queue.get(), timeout=self.flush_interval)
                    batch.append(item)
                except asyncio.TimeoutError:
                    # timeout: flush what we have
                    pass

                # drain up to batch_size
                while len(batch) < self.batch_size and not self.candle_queue.empty():
                    try:
                        batch.append(self.candle_queue.get_nowait())
                    except Exception:
                        break

                if batch:
                    try:
                        df = pd.DataFrame(batch)
                        start = time.time()
                        if not self.test_mode:
                            await self.db.upsert_candles(df)
                        elapsed = time.time() - start
                        self.db_write_time_total += elapsed
                        self.total_candles_written += len(batch)
                        log_record = logging.LogRecord(
                            name=logger.name,
                            level=logging.INFO,
                            pathname="",
                            lineno=0,
                            msg=f"Writer flushed {len(batch)} candles to DB (total: {self.total_candles_written})",
                            args=(),
                            exc_info=None,
                        )
                        log_record.operation = "writer_flush"
                        logger.handle(log_record)
                    except Exception as e:
                        self.db_write_failures += 1
                        logger.error(f"Writer failed to flush batch: {e}", exc_info=True)
                    finally:
                        batch.clear()

            except asyncio.CancelledError:
                logger.info("Writer task cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in writer loop: {e}", exc_info=True)
                await asyncio.sleep(1)

        # Flush any remaining
        if batch:
            try:
                df = pd.DataFrame(batch)
                if not self.test_mode:
                    await self.db.upsert_candles(df)
                self.total_candles_written += len(batch)
            except Exception:
                logger.error("Failed flushing remaining batch on writer shutdown", exc_info=True)
        logger.info("Writer task stopped")
    
    @async_retry(max_retries=5, initial_delay=1.0, exceptions=(Exception,))
    async def connect(self):
        """Initialize WebSocket connection with exponential backoff retry."""
        try:
            # Build streams URL
            streams = "/".join([f"{s.lower()}@kline_{self.interval}" for s in self.symbols])
            url = f"{self.ws_url}?streams={streams}"
            
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
        """
        Send subscription payload to Binance WS.
        
        Args:
            symbols: List of symbols to subscribe
            interval: Kline interval
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
        """
        Background heartbeat task that checks WebSocket health and reconnects
        if idle for too long.
        """
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
                
                # Periodic flush of queue
                if self.candle_queue.qsize() > 0 and time_since_last > 30:
                    await self._flush_buffer()
            
            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled")
                break
            
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _reconnect(self):
        """Reconnect to Binance WS with backoff."""
        self.running = False
        
        if self.ws and not getattr(self.ws, "closed", False):

            try:
                await self.ws.close()
            except Exception:
                pass
        
        self.ws = None
        
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
                                try:
                                    # Try to push to queue; if full, drop and count
                                    self.candle_queue.put_nowait(candle)
                                except asyncio.QueueFull:
                                    self.messages_dropped += 1
                                    logger.warning(
                                        f"Candle queue full. Dropping candle for {candle.get('symbol')}")

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
        """
        Health check: returns True if connection is active.
        
        Returns:
            True if WebSocket is open and received message recently
        """
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
            "queue_size": self.candle_queue.qsize(),
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
            # Writer task batches queue -> DB
            self.writer_task = asyncio.create_task(self._writer_loop())
            
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

        if self.writer_task:
            self.writer_task.cancel()
            try:
                await self.writer_task
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
                    }
                }
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
                    }
                }
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

# The interactive/demo runner is defined later in this file (single unified demo).
# The older demo block was removed to avoid duplication.

# ==================== ORDER BOOK SNAPSHOT COROUTINE ====================

async def snapshot_orderbook(symbol: str, client, db_manager):
    """
    Periodically snapshot the top of book for `symbol` using an async REST
    client (e.g., python-binance AsyncClient) and persist into DB via
    DatabaseManager.

    Args:
        symbol: trading pair symbol
        client: async Binance REST client (may implement depth/get_order_book/get_depth)
        db_manager: DatabaseManager instance
    """
    while True:
        try:
            # Support multiple client method names for robustness
            data = None
            if client is None:
                logger.debug("No REST client provided for orderbook snapshot; skipping")
                await asyncio.sleep(5)
                continue

            if hasattr(client, "depth"):
                data = await client.depth(symbol=symbol)
            elif hasattr(client, "get_order_book"):
                data = await client.get_order_book(symbol=symbol)
            elif hasattr(client, "get_depth"):
                data = await client.get_depth(symbol=symbol)
            else:
                logger.error("REST client does not expose depth/get_order_book/get_depth")
                await asyncio.sleep(5)
                continue

            # Normalize and persist the top-of-book (first bid/ask)
            bids = data.get("bids") if isinstance(data, dict) else None
            asks = data.get("asks") if isinstance(data, dict) else None
            if not bids or not asks:
                logger.warning(f"Orderbook snapshot empty for {symbol}: {data}")
                await asyncio.sleep(5)
                continue

            ts_ms = int(time.time() * 1000)
            bid_price, bid_size = float(bids[0][0]), float(bids[0][1])
            ask_price, ask_size = float(asks[0][0]), float(asks[0][1])

            # Prepare DataFrame and use DatabaseManager bulk insert for efficiency
            df = pd.DataFrame([{
                "ts_ms": ts_ms,
                "symbol": symbol,
                "bid_price": bid_price,
                "bid_size": bid_size,
                "ask_price": ask_price,
                "ask_size": ask_size,
            }])

            try:
                # Upsert into orderbook table; insert_bulk will run sync work in executor
                await db_manager.insert_bulk("orderbook", df, mode="insert")
            except Exception as e:
                logger.error(f"Failed to persist orderbook snapshot for {symbol}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Orderbook snapshot failed for {symbol}: {e}", exc_info=True)
        await asyncio.sleep(5)

# ... כל המחלקות, פונקציות, ובקרות קיימות ...

# ==================== DEMO (עדכון להרצת coroutine) ====================

async def run_demo():
    """Demo: Connect to Binance WS, listen for 5 minutes, print live candles and save orderbook snapshots."""
    import sys
    sys.path.insert(0, "src")
    try:
        from storage.db_manager import DatabaseManager
    except ImportError:
        logger.error("Failed to import DatabaseManager. Ensure src/ is in path.")
        return

    print("\n=== Binance WebSocket Ingestion Demo ===\n")
    db_path = "data/market_data.duckdb"
    db = DatabaseManager(db_path)
    symbols = ["BTCUSDT", "ETHUSDT"]
    interval = "1m"
    ws = BinanceIngestion(
        db,
        symbols=symbols,
        interval=interval,
        batch_size=10,
        test_mode=False, # Use mock mode for demo
    )

    # Start ingestion
    try:
        await ws.start()
        print(f"Listening to {symbols} for 30 seconds...\n")


        # Start orderbook snapshot tasks for each symbol. Use an async REST
        # client for snapshots (python-binance AsyncClient) when not in test_mode.
        client = None
        if not ws.test_mode:
            try:
                client = await AsyncClient.create()
                logger.info("Async REST client created for orderbook snapshots")
            except Exception as e:
                logger.warning(f"Could not create AsyncClient: {e}; continuing without REST snapshots")

        tasks = []
        for symbol in symbols:
            tasks.append(asyncio.create_task(snapshot_orderbook(symbol, client, db)))

        # Monitor for 30 seconds
        for i in range(3):
            await asyncio.sleep(10)
            stats = ws.get_stats()
            if ws.is_alive():
                print(
                    f"✓ [{i*10}s] Connection healthy | "
                    f"Processed: {stats['messages_processed']} | "
                    f"Queue: {stats['queue_size']} candles"
                )
            else:
                print(f"✗ [{i*10}s] Connection unhealthy!")

        # Clean shutdown
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        await ws.stop()
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
        # Close REST client if created
        try:
            if client:
                if hasattr(client, 'close_connection'):
                    await client.close_connection()
                elif hasattr(client, 'close'):
                    await client.close()
        except Exception:
            logger.debug("Failed to close AsyncClient cleanly")

        db.close()
        print("\n=== Demo Complete ===\n")

if __name__ == "__main__":
    asyncio.run(run_demo())

# ... כל קוד המקור שנמצא בקובץ המקורי נשאר בדיוק כפי שהיה ...
# === AUTO PATCH: Backward compatibility for main.py ===
BinanceWebSocketClient = BinanceIngestion
