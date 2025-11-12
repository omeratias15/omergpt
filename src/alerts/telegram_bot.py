"""
src/alerts/telegram_bot.py

Asynchronous Telegram alerting module for OmerGPT.

Listens for new signals from DuckDB and sends structured alerts via Telegram Bot API.
Supports rich formatting, throttling, and custom notification channels.
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import pandas as pd

logger = logging.getLogger("omerGPT.alerts.telegram")


class TelegramBot:
    """
    Asynchronous Telegram bot for trading signal alerts.
    
    Features:
    - Monitors DuckDB for new signals
    - Queue-based message processing
    - Rate limiting and throttling
    - Exponential backoff retry on errors
    - Multi-channel support
    """
    
    TELEGRAM_API = "https://api.telegram.org/bot"
    
    def __init__(
        self,
        db_manager,
        token: str,
        chat_ids: List[str],
        max_per_minute: int = 5,
        retry_attempts: int = 3,
        polling_interval: int = 30
    ):
        """
        Initialize Telegram bot.
        
        Args:
            db_manager: DatabaseManager instance
            token: Telegram Bot API token
            chat_ids: List of chat IDs to send alerts to
            max_per_minute: Maximum messages per minute per chat
            retry_attempts: Number of retry attempts for failed sends
            polling_interval: Polling interval in seconds
        """
        self.db = db_manager
        self.token = token
        self.chat_ids = [str(cid) for cid in chat_ids]
        self.max_per_minute = max_per_minute
        self.retry_attempts = retry_attempts
        self.polling_interval = polling_interval
        
        # Message queue and rate limiting
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.sent_times: Dict[str, deque] = {cid: deque() for cid in self.chat_ids}
        
        # Worker state
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "signals_fetched": 0,
            "alerts_sent": 0,
            "alerts_failed": 0,
            "signals_marked_sent": 0
        }
        
        logger.info(
            f"TelegramBot initialized: {len(chat_ids)} channels, "
            f"max_per_minute={max_per_minute}"
        )

    async def fetch_new_signals(self) -> pd.DataFrame:
        """
        Fetch unsent signals from DuckDB (status='new').
        
        Returns:
            DataFrame with new signals
        """
        try:
            query = """
                SELECT symbol, ts_ms, signal_type, confidence, reason,
                       anomaly_score, rsi, momentum, feature_snapshot, status
                FROM signals
                WHERE status = 'new'
                ORDER BY ts_ms DESC
                LIMIT 100
            """
            
            result = self.db.conn.execute(query)
            df = result.df()
            
            if len(df) > 0:
                self.stats["signals_fetched"] += len(df)
                logger.info(f"Fetched {len(df)} new signals from DB")
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch signals: {e}", exc_info=True)
            return pd.DataFrame()

    def _format_signal_message(self, signal: Dict) -> str:
        """
        Format signal into a human-readable Telegram message with MarkdownV2.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Formatted message string
        """
        symbol = signal.get("symbol", "UNKNOWN")
        signal_type = signal.get("signal_type", "HOLD")
        confidence = signal.get("confidence", 0.0)
        reason = signal.get("reason", "No reason provided")
        anomaly_score = signal.get("anomaly_score", 0.0)
        rsi = signal.get("rsi", 0.0)
        momentum = signal.get("momentum", 0.0)
        ts_ms = signal.get("ts_ms", datetime.now())
        
        # Emoji mapping
        type_emoji = {
            "BUY": "🚀",
            "SELL": "🔴",
            "HOLD": "⏸️"
        }.get(signal_type, "📊")
        
        # Confidence indicators
        conf_bars = "▮" * int(confidence * 10) + "▯" * (10 - int(confidence * 10))
        
        # Format timestamp
        if isinstance(ts_ms, pd.Timestamp):
            ts_str = ts_ms.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(ts_ms)
        
        message = f"""📊 *New Trading Signal Detected\!*

{type_emoji} *{signal_type} Signal*

*Pair:* `{symbol}`
*Time:* `{ts_str} UTC`

*Confidence:* {confidence:.1%} {conf_bars}

*Anomaly Score:* `{anomaly_score:.3f}`

*Technical Context:*
• RSI: `{rsi:.1f}`
• Momentum: `{momentum:+.4f}`

*Reason:* _{reason}_

 *Always manage risk properly"""
        
        return message

    async def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str = "MarkdownV2"
    ) -> bool:
        """
        Send message to Telegram using Bot API.
        
        Args:
            chat_id: Target chat ID
            text: Message text
            parse_mode: Parse mode (MarkdownV2, HTML, None)
            
        Returns:
            True if successful, False otherwise
        """
        await self.message_queue.put({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode
        })
        return True

    async def _send_message_http(self, payload: Dict) -> bool:
        """
        Send message via Telegram Bot API with retry logic.
        
        Args:
            payload: Message payload
            
        Returns:
            True if sent successfully
        """
        chat_id = payload["chat_id"]
        
        for attempt in range(self.retry_attempts):
            try:
                url = f"{self.TELEGRAM_API}{self.token}/sendMessage"
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status == 200:
                            self.stats["alerts_sent"] += 1
                            logger.info(
                                f"✓ Alert sent to {chat_id} "
                                f"(attempt {attempt + 1}/{self.retry_attempts})"
                            )
                            return True
                        
                        elif resp.status == 429:
                            # Rate limit
                            retry_after = int(resp.headers.get("Retry-After", 5))
                            logger.warning(
                                f"⏱️ Rate limit hit. Waiting {retry_after}s..."
                            )
                            await asyncio.sleep(retry_after)
                        
                        else:
                            error_text = await resp.text()
                            logger.error(
                                f"❌ Telegram API error {resp.status}: {error_text}"
                            )
                            if attempt < self.retry_attempts - 1:
                                wait = 2 ** attempt
                                logger.debug(f"Retrying in {wait}s...")
                                await asyncio.sleep(wait)
            
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Request timeout (attempt {attempt + 1})")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
            
            except Exception as e:
                logger.error(f"❌ Send error: {e}", exc_info=True)
                self.stats["alerts_failed"] += 1
                return False
        
        self.stats["alerts_failed"] += 1
        return False

    async def _rate_limit_check(self, chat_id: str):
        """
        Enforce rate limiting for a specific chat.
        
        Args:
            chat_id: Target chat ID
        """
        now = time.time()
        sent_deque = self.sent_times[chat_id]
        
        # Remove timestamps older than 60 seconds
        while sent_deque and (now - sent_deque[0]) > 60:
            sent_deque.popleft()
        
        # Wait if rate limit exceeded
        if len(sent_deque) >= self.max_per_minute:
            oldest = sent_deque[0]
            wait_time = 60 - (now - oldest) + 0.1
            if wait_time > 0:
                logger.debug(
                    f"Rate limit reached for {chat_id} "
                    f"({len(sent_deque)}/{self.max_per_minute}), "
                    f"waiting {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)

    async def update_signal_status(self, symbol: str, ts_ms, signal_type: str, status: str = "sent"):
        """
        Mark signal as processed in database.
        
        Args:
            symbol: Trading pair symbol
            ts_ms: Signal timestamp
            signal_type: Signal type (BUY/SELL/HOLD)
            status: New status (sent/retry/failed)
        """
        try:
            self.db.conn.execute("""
                UPDATE signals
                SET status = ?
                WHERE symbol = ? AND ts_ms = ? AND signal_type = ?
            """, (status, symbol, ts_ms, signal_type))
            
            self.stats["signals_marked_sent"] += 1
            logger.debug(f"Updated signal status: {symbol} → {status}")
        
        except Exception as e:
            logger.error(f"Failed to update signal status: {e}", exc_info=True)

    async def _worker(self):
        """
        Background worker that monitors signals and sends alerts.
        """
        logger.info("✅ Telegram worker started")
        
        while self.running:
            try:
                # Fetch new signals
                signals_df = await self.fetch_new_signals()
                
                if signals_df.empty:
                    await asyncio.sleep(self.polling_interval)
                    continue
                
                # Send alerts for each signal
                for _, signal in signals_df.iterrows():
                    # Format message
                    signal_dict = signal.to_dict()
                    message = self._format_signal_message(signal_dict)
                    
                    # Send to all channels
                    for chat_id in self.chat_ids:
                        # Rate limit check
                        await self._rate_limit_check(chat_id)
                        
                        # Send
                        success = await self._send_message_http({
                            "chat_id": chat_id,
                            "text": message,
                            "parse_mode": "MarkdownV2"
                        })
                        
                        # Track time
                        if success:
                            self.sent_times[chat_id].append(time.time())
                            
                            # Update status
                            await self.update_signal_status(
                                signal_dict["symbol"],
                                signal_dict["ts_ms"],
                                signal_dict["signal_type"],
                                "sent"
                            )
                        else:
                            # Retry later
                            await self.update_signal_status(
                                signal_dict["symbol"],
                                signal_dict["ts_ms"],
                                signal_dict["signal_type"],
                                "retry"
                            )
                    
                    # Small delay between signals
                    await asyncio.sleep(1)
                
                await asyncio.sleep(self.polling_interval)
            
            except asyncio.CancelledError:
                logger.info("Worker cancelled")
                break
            
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.info("⏹️ Telegram worker stopped")

    async def start(self):
        """Start the background alert worker."""
        if self.running:
            logger.warning("Worker already running")
            return
        
        self.running = True
        self.worker_task = asyncio.create_task(self._worker())
        logger.info("✅ Telegram bot started")

    async def stop(self):
        """Stop the background worker gracefully."""
        logger.info("Stopping Telegram bot...")
        self.running = False
        
        if self.worker_task:
            try:
                await asyncio.wait_for(self.worker_task, timeout=5)
            except asyncio.TimeoutError:
                self.worker_task.cancel()
        
        logger.info("⏹️ Telegram bot stopped")

    def get_stats(self) -> Dict:
        """Get bot statistics."""
        return self.stats.copy()


# ==================== DEMO ====================

async def run_demo():
    """
    Demo: Run Telegram bot for 5 minutes with manual signal injection.
    """
    import sys
    sys.path.insert(0, "src")
    
    from storage.db_manager import DatabaseManager
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    print("\n=== Telegram Bot Alert Demo ===\n")
    
    # Get credentials
    token = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "123456789")
    
    if token == "YOUR_BOT_TOKEN":
        print("⚠️  Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        return
    
    # Initialize database
    db = DatabaseManager("data/market_data.duckdb")
    
    # Create bot
    bot = TelegramBot(
        db_manager=db,
        token=token,
        chat_ids=[chat_id],
        max_per_minute=5,
        polling_interval=30
    )
    
    # Start bot
    await bot.start()
    
    try:
        print("Running Telegram bot for 5 minutes...\n")
        
        for i in range(10):  # 10 * 30s = 5 min
            await asyncio.sleep(30)
            stats = bot.get_stats()
            print(f"[{(i+1)*30}s] Sent: {stats['alerts_sent']} | "
                  f"Failed: {stats['alerts_failed']} | "
                  f"Queue: {bot.message_queue.qsize()}\n")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        await bot.stop()
        
        final_stats = bot.get_stats()
        print(f"\n=== Final Stats ===")
        print(f"Signals fetched: {final_stats['signals_fetched']}")
        print(f"Alerts sent: {final_stats['alerts_sent']}")
        print(f"Alerts failed: {final_stats['alerts_failed']}")
        print(f"Signals marked sent: {final_stats['signals_marked_sent']}")
        
        db.close()
        print("\n=== Demo Complete ===\n")


if __name__ == "__main__":
    asyncio.run(run_demo())
import asyncio
from datetime import datetime, timedelta

last_sent = {}

async def send_alert_limited(symbol: str, message: str, bot, chat_id):
    now = datetime.utcnow()
    if symbol in last_sent and now - last_sent[symbol] < timedelta(minutes=5):
        return  # דלג — נשלחה התראה לאחרונה
    last_sent[symbol] = now

    try:
        await bot.send_message(chat_id=chat_id, text=message)
    except Exception as e:
        print(f"[WARN] Failed to send Telegram alert for {symbol}: {e}")


# [PATCH] Add per-symbol 5-minute limiter
    # [PATCH] Per-symbol rate limit: 1 alert per symbol / 5 min
    def _symbol_rate_ok(self, symbol):
        now = datetime.utcnow().timestamp()
        self.sent_times.setdefault(symbol, [])
        self.sent_times[symbol] = [t for t in self.sent_times[symbol] if now - t < 300]
        if self.sent_times[symbol]:
            return False
        self.sent_times[symbol].append(now)
        return True

