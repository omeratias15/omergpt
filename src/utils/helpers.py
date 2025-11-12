"""
src/utils/helpers.py

Utility module for OmerGPT.

Provides async helpers, formatting tools, error handling,
logging configuration, and performance timing utilities.

Used across all modules to ensure consistency and reliability.
"""

import asyncio
import functools
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional


# ==================== LOGGING ====================

def setup_logger(
    name: str = "omerGPT",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure structured logging for OmerGPT.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Default format
    if not format_string:
        format_string = (
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
        )
    
    formatter = logging.Formatter(
        format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ==================== DECORATORS ====================

def timeit(func: Callable) -> Callable:
    """
    Measure execution time of sync or async functions.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with timing
    """
    logger = logging.getLogger("omerGPT.timeit")
    
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                logger.debug(f"⏱️ {func.__name__}: {duration:.3f}s")
        
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                logger.debug(f"⏱️ {func.__name__}: {duration:.3f}s")
        
        return sync_wrapper


def retry_async(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0
) -> Callable:
    """
    Retry async function on failure with exponential backoff.
    
    Args:
        retries: Maximum number of attempts
        delay: Initial delay in seconds
        backoff: Backoff multiplier
        max_delay: Maximum delay between retries
        
    Returns:
        Decorator
    """
    logger = logging.getLogger("omerGPT.retry")
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < retries:
                try:
                    return await func(*args, **kwargs)
                
                except asyncio.CancelledError:
                    raise
                
                except Exception as e:
                    attempt += 1
                    
                    if attempt >= retries:
                        logger.error(
                            f"❌ {func.__name__} failed after {retries} attempts: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"⚠️ {func.__name__} failed (attempt {attempt}/{retries}): {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    await asyncio.sleep(current_delay)
                    current_delay = min(current_delay * backoff, max_delay)
        
        return wrapper
    
    return decorator


def rate_limit(
    calls_per_second: float = 1.0
) -> Callable:
    """
    Rate-limit async function calls.
    
    Args:
        calls_per_second: Maximum calls per second
        
    Returns:
        Decorator
    """
    logger = logging.getLogger("omerGPT.rate_limit")
    
    def decorator(func: Callable) -> Callable:
        limiter = AsyncRateLimiter(int(calls_per_second))
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with limiter:
                return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# ==================== ASYNC UTILITIES ====================

class AsyncRateLimiter:
    """
    Simple async rate limiter using asyncio.Semaphore.
    
    Ensures maximum N requests per second.
    """
    
    def __init__(self, rate_per_second: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            rate_per_second: Maximum requests per second
        """
        self.rate_per_second = max(1, rate_per_second)
        self.semaphore = asyncio.Semaphore(self.rate_per_second)
        self.last_reset = time.time()

    async def __aenter__(self):
        """Acquire rate limit slot."""
        await self.semaphore.acquire()

    async def __aexit__(self, exc_type, exc, tb):
        """Release rate limit slot after delay."""
        delay = 1.0 / self.rate_per_second
        await asyncio.sleep(delay)
        self.semaphore.release()


@asynccontextmanager
async def measure_async(label: str, logger_name: str = "omerGPT"):
    """
    Context manager for timing async operations.
    
    Args:
        label: Operation label
        logger_name: Logger name
        
    Yields:
        None
    """
    logger = logging.getLogger(logger_name)
    start = time.perf_counter()
    
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.debug(f"⏱️ [{label}] completed in {duration:.3f}s")


async def safe_gather(
    *tasks: asyncio.Task,
    return_exceptions: bool = True,
    logger_name: str = "omerGPT"
) -> List[Any]:
    """
    Gather async tasks safely, catching exceptions.
    
    Args:
        *tasks: Asyncio tasks
        return_exceptions: Include exceptions in results
        logger_name: Logger name
        
    Returns:
        List of results
    """
    logger = logging.getLogger(logger_name)
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        return results
    
    except Exception as e:
        logger.error(f"❌ Gather error: {e}", exc_info=True)
        return []


async def async_timeout(
    coro: asyncio.coroutine,
    timeout_seconds: float = 30.0,
    logger_name: str = "omerGPT"
) -> Optional[Any]:
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout_seconds: Timeout in seconds
        logger_name: Logger name
        
    Returns:
        Result or None on timeout
    """
    logger = logging.getLogger(logger_name)
    
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    
    except asyncio.TimeoutError:
        logger.warning(f"⏱️ Operation timed out after {timeout_seconds}s")
        return None
    
    except Exception as e:
        logger.error(f"❌ Timeout operation error: {e}")
        return None


# ==================== FORMATTING UTILITIES ====================

def fmt_money(value: float, symbol: str = "$", precision: int = 2) -> str:
    """Format number as currency."""
    return f"{symbol}{value:,.{precision}f}"


def fmt_pct(value: float, precision: int = 2) -> str:
    """Format number as percentage."""
    return f"{value*100:.{precision}f}%"


def fmt_pct_change(old: float, new: float, precision: int = 2) -> str:
    """Format percentage change."""
    if old == 0:
        return "N/A"
    change = ((new - old) / abs(old)) * 100
    symbol = "+" if change > 0 else ""
    return f"{symbol}{change:.{precision}f}%"


def fmt_ts(ts: Any, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format timestamp."""
    if hasattr(ts, "strftime"):
        return ts.strftime(fmt)
    elif isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts).strftime(fmt)
    else:
        return str(ts)


def fmt_bytes(num_bytes: int, precision: int = 2) -> str:
    """Format bytes as human-readable."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.{precision}f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.{precision}f} PB"


def fmt_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ==================== ERROR HANDLING ====================

class OMERGPTError(Exception):
    """Base exception for OmerGPT."""
    pass


class ConfigError(OMERGPTError):
    """Configuration error."""
    pass


class DataError(OMERGPTError):
    """Data error."""
    pass


class APIError(OMERGPTError):
    """API error."""
    pass


def handle_exception(
    exc: Exception,
    context: str = "",
    logger_name: str = "omerGPT"
) -> None:
    """
    Handle exception with logging.
    
    Args:
        exc: Exception to handle
        context: Context information
        logger_name: Logger name
    """
    logger = logging.getLogger(logger_name)
    
    if isinstance(exc, OMERGPTError):
        logger.error(f"❌ OmerGPT Error{' (' + context + ')' if context else ''}: {exc}")
    else:
        logger.error(
            f"❌ Unexpected Error{' (' + context + ')' if context else ''}: {exc}",
            exc_info=True
        )


# ==================== PERFORMANCE UTILITIES ====================

class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self, logger_name: str = "omerGPT"):
        self.logger = logging.getLogger(logger_name)
        self.metrics = {}
        self.start_time = time.time()
    
    def record(self, key: str, value: float, unit: str = ""):
        """Record a metric."""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
        self.logger.debug(f"📊 {key}: {value} {unit}")
    
    def average(self, key: str) -> float:
        """Get average of metric."""
        if key in self.metrics and self.metrics[key]:
            return sum(self.metrics[key]) / len(self.metrics[key])
        return 0.0
    
    def max(self, key: str) -> float:
        """Get max of metric."""
        if key in self.metrics and self.metrics[key]:
            return max(self.metrics[key])
        return 0.0
    
    def uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time
    
    def summary(self) -> str:
        """Get performance summary."""
        lines = [
            f"Uptime: {fmt_duration(self.uptime())}",
            "Metrics:"
        ]
        
        for key, values in self.metrics.items():
            if values:
                avg = sum(values) / len(values)
                lines.append(f"  {key}: avg={avg:.3f}, max={max(values):.3f}, count={len(values)}")
        
        return "\n".join(lines)


# ==================== TESTING UTILITIES ====================

async def wait_for_condition(
    condition_func: Callable,
    timeout: float = 10.0,
    check_interval: float = 0.1,
    logger_name: str = "omerGPT"
) -> bool:
    """
    Wait for async condition to become true.
    
    Args:
        condition_func: Async function returning bool
        timeout: Max wait time in seconds
        check_interval: Check interval in seconds
        logger_name: Logger name
        
    Returns:
        True if condition met, False if timeout
    """
    logger = logging.getLogger(logger_name)
    start = time.time()
    
    while (time.time() - start) < timeout:
        try:
            if await condition_func():
                return True
        except Exception as e:
            logger.debug(f"Condition check error: {e}")
        
        await asyncio.sleep(check_interval)
    
    logger.warning(f"⏱️ Condition timeout after {timeout}s")
    return False


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import asyncio
    
    # Setup logger
    logger = setup_logger(level=logging.DEBUG)
    
    # Test timeit decorator
    @timeit
    async def example_async():
        await asyncio.sleep(0.1)
        return "async result"
    
    @timeit
    def example_sync():
        time.sleep(0.1)
        return "sync result"
    
    # Test retry decorator
    @retry_async(retries=3, delay=0.1)
    async def example_retry():
        return "retry result"
    
    # Test rate limiter
    @rate_limit(calls_per_second=2)
    async def example_rate_limited():
        logger.info("Rate-limited call")
        return "rate limited result"
    
    async def main():
        print("\n=== OmerGPT Helpers Test ===\n")
        
        # Test formatting
        print(f"Money: {fmt_money(1234.5678)}")
        print(f"Percentage: {fmt_pct(0.1234)}")
        print(f"Change: {fmt_pct_change(100, 110)}")
        print(f"Bytes: {fmt_bytes(1024*1024*5)}")
        print(f"Duration: {fmt_duration(3661.5)}")
        print()
        
        # Test async utilities
        print("Running async example...")
        result = await example_async()
        print(f"Result: {result}\n")
        
        print("Running sync example...")
        result = example_sync()
        print(f"Result: {result}\n")
        
        print("Running retry example...")
        result = await example_retry()
        print(f"Result: {result}\n")
        
        print("Running rate-limited examples...")
        for i in range(4):
            result = await example_rate_limited()
        print()
        
        # Test performance monitor
        monitor = PerformanceMonitor()
        monitor.record("query_time", 0.123, "s")
        monitor.record("query_time", 0.145, "s")
        monitor.record("query_time", 0.098, "s")
        print("Performance Summary:")
        print(monitor.summary())
    
    asyncio.run(main())
