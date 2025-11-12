"""
src/ingestion/kraken_ws.py

Production-grade asynchronous Kraken WebSocket ingestion module.

Streams live OHLC, trade, and ticker data into DuckDB via DatabaseManager
with fault tolerance, rate limiting, structured logging, health checks,
automatic resubscription, and graceful shutdown.
"""

import asyncio
import json
import logging
import logging.handlers
import random
import time
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable, Set
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
        "symbols": os.getenv("KRAKEN_SYMBOLS", "BTC/USDT,ETH/USDT").split(","),
        "channels": os.getenv("KRAKEN_CHANNELS", "ohlc,trade,ticker").split(","),
        "ohlc_interval": int(os.getenv("KRAKEN_OHLC_INTERVAL", "1")),
        "batch_size": int(os.getenv("KRAKEN_BATCH_SIZE", "20")),
        "rate_limit": int(os.getenv("KRAKEN_RATE_LIMIT", "50")),  # msgs per second
        "db_path": os.getenv("KRAKEN_DB_PATH", "data/market_data.duckdb"),
        "timeout": int(os.getenv("KRAKEN_TIMEOUT", "30")),
        "heartbeat_interval": int(os.getenv("KRAKEN_HEARTBEAT_INTERVAL", "30")),
        "max_retries": int(os.getenv("KRAKEN_MAX_RETRIES", "10")),
        "test_mode": os.getenv("KRAKEN_TEST_MODE", "false").lower() == "true",
    }
    
    # Try to load from config.yaml if it exists
    config_path = Path("config.yaml")
    if config_path.exists() and HAS_YAML:
        try:
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
                if "kraken" in yaml_config:
                    config.update(yaml_config["kraken"])
        except Exception as e:
            logger = logging.getLogger("omerGPT.ingestion.kraken_ws")
            logger.warning(f"Failed to load config.yaml: {e}")
    
    return config


# ==================== LOGGING SETUP ====================

def setup_structured_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Configure structured JSON logging with both file and console handlers."""
    logger = logging.getLogger("omerGPT.ingestion.kraken_ws")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Structured formatter
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "operation": getattr(record, "operation", "unknown"),
                "pair": getattr(record, "pair", None),
                "channel": getattr(record, "channel", None),
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
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(structured_formatter)
        logger.addHandler(file_handler)
    
    return logger


logger = setup_structured_logging("logs/kraken_ws.log")


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


# ==================== KRAKEN INGESTION ====================

class KrakenIngestion:
    """
    Async Kraken WebSocket ingestion for multi-pair real-time OHLC, trade, and ticker data.
    
    Features:
    - Multiple channel support (ohlc, trade, ticker)
    - Exponential backoff with jitter on reconnect
    - Async rate limiting for message processing
    - Structured JSON logging with operation tracking
    - Latency monitoring (server_time vs local_time)
    - Batch DB writes via DatabaseManager
    - Health check & auto-reconnect with heartbeat task
    - Automatic resubscription after reconnect
    - Graceful shutdown with buffer flush
    - Error recovery and retry logic
    - Runtime configuration via .env or config.yaml
    - Mock mode for unit testing
    """
    
    WS_URL = "wss://ws.kraken.com"
    
class KrakenIngestion:
    def __init__(
        self,
        db_manager,
        symbols=None,
        channels=None,
        ohlc_interval=None,
        batch_size=20,
        rate_limit=50,
        db_path=None,
        timeout=None,
        heartbeat_interval=None,
        max_retries=5,
        test_mode=False,
        interval="1m",
    ):
        self.symbols = symbols or ["BTC/USD"]
        self.interval = interval
        self.batch_size = batch_size
        self.rate_limit = rate_limit
        self.test_mode = test_mode

        self.ws_url= "wss://ws.kraken.com"
        self.running = False
        self.connection_alive = False
        self.stats = {
            "running": False,
            "connection_alive": False,
            "messages_processed": 0,
            "messages_dropped": 0,
            "total_candles_written": 0,
            "queue_size": 0,
            "uptime_seconds": 0,
        }

        import logging
        self.logger = logging.getLogger("omerGPT.ingestion.kraken_ws")
        self.logger.info(
            f"Initialized KrakenIngestion for {len(self.symbols)} symbols "
            f"| interval={self.interval} | batch_size={self.batch_size} "
            f"| rate_limit={self.rate_limit}msg/s | test_mode={self.test_mode}"
        )

        """
        Initialize Kraken WebSocket ingestion.
        
        Args:
            db_manager: DatabaseManager instance for persistence
            symbols: List of trading pairs (e.g., ["BTC/USDT", "ETH/USDT"])
                    If None, loaded from config
            channels: List of channels to subscribe (ohlc, trade, ticker)
            ohlc_interval: OHLC interval in minutes (1, 5, 15, 30, 60, etc.)
            batch_size: Number of records to buffer before DB write
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
        self.symbols = [self._format_pair(s) for s in (symbols or config["symbols"])]
        self.channels = channels or config.get("channels", ["ohlc", "trade", "ticker"])
        self.ohlc_interval = ohlc_interval or config.get("ohlc_interval", 1)
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
        
        # Channel management
        self.subscribed_channels: Set[str] = set()
        
        # Data buffering
        self.ohlc_buffer = []
        self.trade_buffer = []
        self.ticker_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        # Rate limiting
        self.rate_limiter = AsyncRateLimiter(self.rate_limit)
        
        # Backoff configuration
        self.initial_backoff = 1.0
        self.max_backoff = 60.0
        
        # Background tasks
        self.heartbeat_task = None
        self.listen_task = None
        self.flush_task = None
        
        # Statistics
        self.messages_processed = 0
        self.messages_dropped = 0
        self.total_records_written = 0
        
        logger.info(
            f"Initialized KrakenIngestion for {len(self.symbols)} pairs: "
            f"{self.symbols} | channels={self.channels} | "
            f"ohlc_interval={self.ohlc_interval}m | batch_size={self.batch_size} | "
            f"rate_limit={self.rate_limit}msg/s | test_mode={self.test_mode}"
        )
    
    def _format_pair(self, symbol: str) -> str:
        """Convert BTCUSDT to BTC/USDT format for Kraken."""
        if "/" in symbol:
            return symbol
        
        # Try to split common patterns
        if "USDT" in symbol:
            base = symbol.replace("USDT", "")
            return f"{base}/USDT"
        elif "USD" in symbol:
            base = symbol.replace("USD", "")
            return f"{base}/USD"
        elif "EUR" in symbol:
            base = symbol.replace("EUR", "")
            return f"{base}/EUR"
        else:
            # Default: assume last 3-4 chars are quote currency
            return f"{symbol[:-4]}/{symbol[-4:]}"
    
    def _normalize_pair(self, pair: str) -> str:
        """Convert Kraken pair format to unified format: BTC/USDT -> BTCUSDT."""
        return pair.replace("/", "")
    
    def _calculate_backoff(self) -> float:
        """Calculate exponential backoff with jitter."""
        backoff = min(
            self.initial_backoff * (2 ** self.retry_count),
            self.max_backoff
        )
        jitter = random.uniform(0, 0.3 * backoff)
        return backoff + jitter
    
    def normalize_ohlc_message(self, msg: list) -> Optional[dict]:
        """
        Convert raw Kraken OHLC message to normalized candle dict.
        
        Kraken message format:
        [channelID, [time, etime, open, high, low, close, vwap, volume, count], "ohlc-X", "PAIR"]
        
        Args:
            msg: Raw WebSocket message as list
        
        Returns:
            Normalized candle dict or None if invalid
        """
        try:
            # Validate message structure
            if not isinstance(msg, list) or len(msg) < 4:
                return None
            
            # Check if it's an OHLC message
            channel_name = msg[2]
            if not channel_name.startswith("ohlc"):
                return None
            
            ohlc_data = msg[1]
            pair = msg[3]
            
            # Parse fields: [time, etime, open, high, low, close, vwap, volume, count]
            timestamp_unix = float(ohlc_data[1])  # etime = end time
            ts = pd.to_datetime(timestamp_unix, unit="s")
            
            # Calculate latency
            local_time_unix = time.time()
            latency_ms = abs((local_time_unix - timestamp_unix) * 1000)
            
            # Log high latency
            if latency_ms > 300:
                log_record = logging.LogRecord(
                    name=logger.name,
                    level=logging.WARNING,
                    pathname="",
                    lineno=0,
                    msg=f"High latency for {pair}: {latency_ms:.0f}ms",
                    args=(),
                    exc_info=None,
                )
                log_record.operation = "normalize_ohlc"
                log_record.pair = pair
                log_record.channel = "ohlc"
                log_record.latency_ms = latency_ms
                logger.handle(log_record)
            
            # Normalize symbol
            symbol = self._normalize_pair(pair)
            
            candle = {
                "symbol": symbol,
                "ts_ms": ts,
                "open": float(ohlc_data[2]),
                "high": float(ohlc_data[3]),
                "low": float(ohlc_data[4]),
                "close": float(ohlc_data[5]),
                "volume": float(ohlc_data[7]),
            }
            
            log_record = logging.LogRecord(
                name=logger.name,
                level=logging.DEBUG,
                pathname="",
                lineno=0,
                msg=f"Normalized OHLC {pair} -> {symbol} @ {ts} | close={candle['close']:.2f}",
                args=(),
                exc_info=None,
            )
            log_record.operation = "normalize_ohlc"
            log_record.pair = pair
            log_record.channel = "ohlc"
            log_record.latency_ms = latency_ms
            logger.handle(log_record)
            
            return candle
        
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.error(f"Failed to normalize OHLC message: {e} | msg={msg}", exc_info=True)
            return None
    
    def normalize_trade_message(self, msg: list) -> Optional[dict]:
        """
        Convert raw Kraken trade message to normalized trade dict.
        
        Kraken message format:
        [channelID, [[price, lot_volume, time, side, ordertype, misc], ...], "trade", "PAIR"]
        
        Args:
            msg: Raw WebSocket message as list
        
        Returns:
            Normalized trade dict or None if invalid
        """
        try:
            if not isinstance(msg, list) or len(msg) < 4:
                return None
            
            channel_name = msg[2]
            if channel_name != "trade":
                return None
            
            trades = msg[1]
            pair = msg[3]
            symbol = self._normalize_pair(pair)
            
            trade_list = []
            for trade in trades:
                if len(trade) < 6:
                    continue
                
                price = float(trade[0])
                volume = float(trade[1])
                time_unix = float(trade[2])
                ts = pd.to_datetime(time_unix, unit="s")
                side = trade[3]  # "b" or "s"
                
                trade_dict = {
                    "symbol": symbol,
                    "ts_ms": ts,
                    "price": price,
                    "volume": volume,
                    "side": side,
                }
                trade_list.append(trade_dict)
            
            latency_ms = abs((time.time() - time_unix) * 1000)
            
            log_record = logging.LogRecord(
                name=logger.name,
                level=logging.DEBUG,
                pathname="",
                lineno=0,
                msg=f"Normalized {len(trade_list)} trades for {pair} -> {symbol}",
                args=(),
                exc_info=None,
            )
            log_record.operation = "normalize_trades"
            log_record.pair = pair
            log_record.channel = "trade"
            log_record.latency_ms = latency_ms
            logger.handle(log_record)
            
            return trade_list if trade_list else None
        
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.error(f"Failed to normalize trade message: {e} | msg={msg}", exc_info=True)
            return None
    
    def normalize_ticker_message(self, msg: list) -> Optional[dict]:
        """
        Convert raw Kraken ticker message to normalized ticker dict.
        
        Kraken message format:
        [channelID, {fields}, "ticker", "PAIR"]
        
        Args:
            msg: Raw WebSocket message as list
        
        Returns:
            Normalized ticker dict or None if invalid
        """
        try:
            if not isinstance(msg, list) or len(msg) < 4:
                return None
            
            channel_name = msg[2]
            if channel_name != "ticker":
                return None
            
            ticker_data = msg[1]
            pair = msg[3]
            symbol = self._normalize_pair(pair)
            
            ts = pd.to_datetime(datetime.utcnow())
            
            ticker = {
                "symbol": symbol,
                "ts_ms": ts,
                "bid": float(ticker_data.get("b", [0])[0]),
                "ask": float(ticker_data.get("a", [0])[0]),
                "last_trade": float(ticker_data.get("c", [0])[0]),
                "volume_24h": float(ticker_data.get("v", [0])[0]),
                "vwap_24h": float(ticker_data.get("p", [0])[0]),
            }
            
            latency_ms = 0
            
            log_record = logging.LogRecord(
                name=logger.name,
                level=logging.DEBUG,
                pathname="",
                lineno=0,
                msg=f"Normalized ticker for {pair} -> {symbol} | last={ticker['last_trade']:.2f}",
                args=(),
                exc_info=None,
            )
            log_record.operation = "normalize_ticker"
            log_record.pair = pair
            log_record.channel = "ticker"
            logger.handle(log_record)
            
            return ticker
        
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.error(f"Failed to normalize ticker message: {e} | msg={msg}", exc_info=True)
            return None
    
    @async_retry(max_retries=3, exceptions=(Exception,))
    async def _flush_buffer(self, buffer_type: str = "all") -> int:
        """
        Flush data buffer to database.
        
        Args:
            buffer_type: "ohlc", "trade", "ticker", or "all"
        
        Returns:
            Total number of records flushed
        """
        async with self.buffer_lock:
            total_flushed = 0
            
            # Flush OHLC
            if buffer_type in ["ohlc", "all"] and self.ohlc_buffer:
                try:
                    buffer_size = len(self.ohlc_buffer)
                    df = pd.DataFrame(self.ohlc_buffer)
                    
                    if self.test_mode:
                        logger.info(f"[TEST MODE] Would flush {buffer_size} OHLC candles to DB")
                    else:
                        await self.db.upsert_candles(df)
                    
                    self.total_records_written += buffer_size
                    total_flushed += buffer_size
                    
                    log_record = logging.LogRecord(
                        name=logger.name,
                        level=logging.INFO,
                        pathname="",
                        lineno=0,
                        msg=f"Flushed {buffer_size} OHLC candles to DB (total: {self.total_records_written})",
                        args=(),
                        exc_info=None,
                    )
                    log_record.operation = "flush_buffer"
                    log_record.channel = "ohlc"
                    logger.handle(log_record)
                    
                    self.ohlc_buffer.clear()
                
                except Exception as e:
                    self.messages_dropped += len(self.ohlc_buffer)
                    logger.error(f"Failed to flush OHLC buffer: {e}", exc_info=True)
                    raise
            
            # Flush trades
            if buffer_type in ["trade", "all"] and self.trade_buffer:
                try:
                    buffer_size = len(self.trade_buffer)
                    df = pd.DataFrame(self.trade_buffer)
                    
                    if self.test_mode:
                        logger.info(f"[TEST MODE] Would flush {buffer_size} trades to DB")
                    else:
                        await self.db.upsert_trades(df)
                    
                    self.total_records_written += buffer_size
                    total_flushed += buffer_size
                    
                    log_record = logging.LogRecord(
                        name=logger.name,
                        level=logging.INFO,
                        pathname="",
                        lineno=0,
                        msg=f"Flushed {buffer_size} trades to DB (total: {self.total_records_written})",
                        args=(),
                        exc_info=None,
                    )
                    log_record.operation = "flush_buffer"
                    log_record.channel = "trade"
                    logger.handle(log_record)
                    
                    self.trade_buffer.clear()
                
                except Exception as e:
                    self.messages_dropped += len(self.trade_buffer)
                    logger.error(f"Failed to flush trade buffer: {e}", exc_info=True)
                    raise
            
            # Flush tickers
            if buffer_type in ["ticker", "all"] and self.ticker_buffer:
                try:
                    buffer_size = len(self.ticker_buffer)
                    df = pd.DataFrame(self.ticker_buffer)
                    
                    if self.test_mode:
                        logger.info(f"[TEST MODE] Would flush {buffer_size} tickers to DB")
                    else:
                        await self.db.upsert_tickers(df)
                    
                    self.total_records_written += buffer_size
                    total_flushed += buffer_size
                    
                    log_record = logging.LogRecord(
                        name=logger.name,
                        level=logging.INFO,
                        pathname="",
                        lineno=0,
                        msg=f"Flushed {buffer_size} tickers to DB (total: {self.total_records_written})",
                        args=(),
                        exc_info=None,
                    )
                    log_record.operation = "flush_buffer"
                    log_record.channel = "ticker"
                    logger.handle(log_record)
                    
                    self.ticker_buffer.clear()
                
                except Exception as e:
                    self.messages_dropped += len(self.ticker_buffer)
                    logger.error(f"Failed to flush ticker buffer: {e}", exc_info=True)
                    raise
            
            return total_flushed
    
    async def _periodic_flush(self):
        """Background task that periodically flushes buffers."""
        logger.info("Periodic flush task started")
        
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if not self.running:
                    break
                
                # Flush if any buffer has data
                if any([self.ohlc_buffer, self.trade_buffer, self.ticker_buffer]):
                    await self._flush_buffer("all")
            
            except asyncio.CancelledError:
                logger.info("Periodic flush task cancelled")
                break
            
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}", exc_info=True)
    
    async def _heartbeat(self):
        """
        Background heartbeat task that checks WebSocket health and reconnects
        if idle for too long. Also handles periodic pings.
        """
        logger.info("Heartbeat task started")
        
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if not self.running:
                    break
                
                now = time.time()
                time_since_last = now - self.last_message_time
                
                # Check for idle connection
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
                
                # Send ping to keep connection alive
                elif self.ws and not self.ws.closed:
                    try:
                        ping_msg = {"event": "ping"}
                        await self.ws.send(json.dumps(ping_msg))
                        
                        log_record = logging.LogRecord(
                            name=logger.name,
                            level=logging.DEBUG,
                            pathname="",
                            lineno=0,
                            msg="Sent ping to Kraken WS",
                            args=(),
                            exc_info=None,
                        )
                        log_record.operation = "heartbeat"
                        logger.handle(log_record)
                    
                    except Exception as e:
                        logger.error(f"Error sending ping: {e}")
            
            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled")
                break
            
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}", exc_info=True)
    
    async def _reconnect(self):
        """Reconnect to Kraken WS with backoff."""
        self.running = False
        
        if self.ws and not self.ws.closed:
            try:
                await self.ws.close()
            except Exception:
                pass
        
        self.ws = None
        
        try:
            await self.connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}", exc_info=True)
    
    @async_retry(max_retries=5, initial_delay=1.0, exceptions=(Exception,))
    async def connect(self):
        """Initialize WebSocket connection with exponential backoff retry."""
        try:
            if self.test_mode:
                logger.info(f"[TEST MODE] Would connect to {self.WS_URL}")
                self.ws = MockWebSocket()
                self.running = True
                self.retry_count = 0
                self.connection_start_time = time.time()
                self.last_message_time = time.time()
                return
            
            logger.info(
                f"Connecting to Kraken WS (attempt {self.retry_count + 1}/"
                f"{self.max_retries})..."
            )
            
            self.ws = await asyncio.wait_for(
                websockets.connect(
                    self.ws_url

,
                    ping_interval=None,  # Manual ping via heartbeat
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
                msg=f"✅ Connected to Kraken WS for {len(self.symbols)} pairs "
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
    
    async def subscribe(self, symbols: List[str], channels: List[str]):
        """
        Send subscription payload to Kraken WS for multiple channels.
        
        Args:
            symbols: List of pairs to subscribe
            channels: List of channels (ohlc, trade, ticker)
        """
        if not self.ws:
            raise RuntimeError("WebSocket not connected. Call connect() first.")
        
        try:
            if self.test_mode:
                logger.info(f"[TEST MODE] Would subscribe to {channels} for {len(symbols)} symbols")
                return
            
            for channel in channels:
                if channel == "ohlc":
                    payload = {
                        "event": "subscribe",
                        "pair": symbols,
                        "subscription": {
                            "name": "ohlc",
                            "interval": self.ohlc_interval
                        }
                    }
                elif channel == "trade":
                    payload = {
                        "event": "subscribe",
                        "pair": symbols,
                        "subscription": {"name": "trade"}
                    }
                elif channel == "ticker":
                    payload = {
                        "event": "subscribe",
                        "pair": symbols,
                        "subscription": {"name": "ticker"}
                    }
                else:
                    logger.warning(f"Unknown channel: {channel}")
                    continue
                
                await self.ws.send(json.dumps(payload))
                
                log_record = logging.LogRecord(
                    name=logger.name,
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=f"Subscribed to {channel.upper()} for {len(symbols)} pairs",
                    args=(),
                    exc_info=None,
                )
                log_record.operation = "subscribe"
                log_record.channel = channel
                logger.handle(log_record)
        
        except Exception as e:
            logger.error(f"Failed to send subscription: {e}", exc_info=True)
            raise
    
    async def listen(self):
        """
        Continuously read incoming messages and push parsed data to buffers.
        Handles reconnection on disconnect and automatic resubscription.
        """
        logger.info("Listener task started")
        
        while self.running:
            try:
                # Ensure connection
                if not self.ws or (not self.test_mode and self.ws.closed):
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
                            
                            # Handle dict messages (system events)
                            if isinstance(msg, dict):
                                event = msg.get("event")
                                
                                if event == "heartbeat":
                                    logger.debug("💓 Heartbeat received")
                                
                                elif event == "pong":
                                    logger.debug("🏓 Pong received")
                                
                                elif event == "subscriptionStatus":
                                    status = msg.get("status")
                                    channel = msg.get("subscription", {}).get("name", "unknown")
                                    pair = msg.get("pair", "unknown")
                                    
                                    if status == "subscribed":
                                        self.subscribed_channels.add(channel)
                                        log_record = logging.LogRecord(
                                            name=logger.name,
                                            level=logging.INFO,
                                            pathname="",
                                            lineno=0,
                                            msg=f"✅ Subscription confirmed: {channel} for {pair}",
                                            args=(),
                                            exc_info=None,
                                        )
                                        log_record.operation = "subscribe_status"
                                        log_record.pair = pair
                                        log_record.channel = channel
                                        logger.handle(log_record)
                                    
                                    elif status == "error":
                                        log_record = logging.LogRecord(
                                            name=logger.name,
                                            level=logging.ERROR,
                                            pathname="",
                                            lineno=0,
                                            msg=f"❌ Subscription error for {channel}: {msg.get('errorMessage', msg)}",
                                            args=(),
                                            exc_info=None,
                                        )
                                        log_record.operation = "subscribe_error"
                                        log_record.pair = pair
                                        log_record.channel = channel
                                        logger.handle(log_record)
                                
                                elif event == "systemStatus":
                                    log_record = logging.LogRecord(
                                        name=logger.name,
                                        level=logging.INFO,
                                        pathname="",
                                        lineno=0,
                                        msg=f"System status: {msg.get('status', 'unknown')}",
                                        args=(),
                                        exc_info=None,
                                    )
                                    log_record.operation = "system_status"
                                    logger.handle(log_record)
                                
                                elif event == "error":
                                    logger.error(f"❌ Kraken error: {msg.get('errorMessage', msg)}")
                            
                            # Handle list messages (data)
                            elif isinstance(msg, list):
                                if len(msg) < 2:
                                    continue
                                
                                channel_name = msg[2] if len(msg) > 2 else None
                                
                                # Route to appropriate handler
                                if channel_name and channel_name.startswith("ohlc"):
                                    candle = self.normalize_ohlc_message(msg)
                                    if candle:
                                        async with self.buffer_lock:
                                            self.ohlc_buffer.append(candle)
                                        
                                        # Flush if batch size reached
                                        if len(self.ohlc_buffer) >= self.batch_size:
                                            await self._flush_buffer("ohlc")
                                
                                elif channel_name == "trade":
                                    trades = self.normalize_trade_message(msg)
                                    if trades:
                                        async with self.buffer_lock:
                                            self.trade_buffer.extend(trades)
                                        
                                        # Flush if batch size reached
                                        if len(self.trade_buffer) >= self.batch_size:
                                            await self._flush_buffer("trade")
                                
                                elif channel_name == "ticker":
                                    ticker = self.normalize_ticker_message(msg)
                                    if ticker:
                                        async with self.buffer_lock:
                                            self.ticker_buffer.append(ticker)
                                        
                                        # Flush if batch size reached
                                        if len(self.ticker_buffer) >= self.batch_size:
                                            await self._flush_buffer("ticker")
                        
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
        
        if not self.test_mode and self.ws.closed:
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
            "total_records_written": self.total_records_written,
            "ohlc_buffer_size": len(self.ohlc_buffer),
            "trade_buffer_size": len(self.trade_buffer),
            "ticker_buffer_size": len(self.ticker_buffer),
            "subscribed_channels": list(self.subscribed_channels),
            "uptime_seconds": (
                time.time() - self.connection_start_time
                if self.connection_start_time else 0
            ),
        }
    
    async def start(self):
        """Start WebSocket ingestion with background tasks."""
        logger.info("Starting Kraken WebSocket ingestion...")
        
        try:
            await self.connect()
            await self.subscribe(self.symbols, self.channels)
            
            # Start background tasks
            self.listen_task = asyncio.create_task(self.listen())
            self.heartbeat_task = asyncio.create_task(self._heartbeat())
            self.flush_task = asyncio.create_task(self._periodic_flush())
            
            logger.info("Kraken WebSocket ingestion started successfully")
        
        except Exception as e:
            logger.error(f"Failed to start ingestion: {e}", exc_info=True)
            await self.stop()
            raise
    
    async def stop(self):
        """Gracefully stop WebSocket connection and flush all remaining data."""
        logger.info("Stopping Kraken WebSocket ingestion...")
        
        self.running = False
        
        # Flush all remaining data
        try:
            await self._flush_buffer("all")
        except Exception as e:
            logger.error(f"Error flushing buffers during shutdown: {e}", exc_info=True)
        
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
        
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket
        if self.ws and not self.ws.closed:
            try:
                await self.ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        stats = self.get_stats()
        logger.info(f"WebSocket closed | Stats: {json.dumps(stats, indent=2)}")


# ==================== MOCK FOR TESTING ====================

class MockWebSocket:
    """Mock WebSocket for unit testing."""
    
    def __init__(self):
        self.closed = False
        self.messages = []
    
    async def send(self, msg):
        self.messages.append(msg)
    
    async def close(self):
        self.closed = True
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        # Yield mock messages
        mock_messages = [
            json.dumps({
                "event": "subscriptionStatus",
                "status": "subscribed",
                "subscription": {"name": "ohlc", "interval": 1},
                "pair": "BTC/USDT",
            }),
            json.dumps([
                0,
                [
                    int(time.time()),
                    int(time.time()),
                    "50000",
                    "51000",
                    "49000",
                    "50500",
                    "50200",
                    "1000",
                    100
                ],
                "ohlc-1m",
                "BTC/USDT"
            ]),
            json.dumps([
                1,
                [["50100", "0.5", int(time.time()), "b", "market", ""]],
                "trade",
                "BTC/USDT"
            ]),
            json.dumps([
                2,
                {
                    "b": ["50200", 1.0],
                    "a": ["50300", 1.0],
                    "c": ["50250", 0.5],
                    "v": ["5000", "10000"],
                    "p": ["50150", "50180"],
                },
                "ticker",
                "BTC/USDT"
            ]),
            json.dumps({"event": "heartbeat"}),
        ]
        
        for msg in mock_messages:
            await asyncio.sleep(0.1)
            yield msg


# ==================== DEMO ====================

async def run_demo():
    """Demo: Connect to Kraken WS, listen for 5 minutes, print live data."""
    import sys
    
    sys.path.insert(0, "src")
    
    try:
        from storage.db_manager import DatabaseManager
    except ImportError:
        logger.error("Failed to import DatabaseManager. Ensure src/ is in path.")
        return
    
    print("\n=== Kraken WebSocket Ingestion Demo ===\n")
    
    # Initialize database
    db_path = "data/market_data.duckdb"
    db = DatabaseManager(db_path)
    
    # Setup ingestion
    symbols = ["BTC/USDT", "ETH/USDT"]
    channels = ["ohlc", "trade", "ticker"]
    ws = KrakenIngestion(
        db,
        symbols=symbols,
        channels=channels,
        batch_size=10,
        test_mode=True,  # Use mock mode for demo
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
                    f"OHLC: {stats['ohlc_buffer_size']} | "
                    f"Trades: {stats['trade_buffer_size']} | "
                    f"Tickers: {stats['ticker_buffer_size']}"
                )
            else:
                print(f"✗ [{i*10}s] Connection unhealthy!")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Stop
        await ws.stop()
        
        # Print final stats
        print("\n=== Final Stats ===")
        stats = ws.get_stats()
        print(json.dumps(stats, indent=2))
        
        # Verify data
        print("\n=== Verification ===")
        
        try:
            symbols_in_db = await db.get_symbols()
            print(f"Symbols in DB: {symbols_in_db}")
            
            for symbol in ["BTCUSDT", "ETHUSDT"]:
                df = await db.get_latest_candles(symbol, limit=5)
                print(f"\nLast 5 candles for {symbol}:")
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
