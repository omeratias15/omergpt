"""
Error Notifier - Telegram alerts for omerGPT failures
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import os
from typing import Optional

logger = logging.getLogger(__name__)

class ErrorNotifier:
    """Send error notifications via Telegram"""

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            logger.warning("Telegram credentials not found. Notifications disabled.")

    async def send_message(self, message: str, priority: str = "INFO"):
        """Send a message via Telegram"""
        if not self.enabled:
            logger.info(f"[{priority}] {message}")
            return False

        try:
            import aiohttp

            emoji_map = {
                "INFO": "ℹ️",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "CRITICAL": "🚨",
                "SUCCESS": "✅"
            }

            emoji = emoji_map.get(priority, "📢")
            formatted_message = f"{emoji} **omerGPT Alert**\n\n{message}\n\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": formatted_message,
                "parse_mode": "Markdown"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        logger.info(f"Notification sent: {priority}")
                        return True
                    else:
                        logger.error(f"Telegram API error: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    async def send_error_notification(self, error_message: str):
        """Send error notification"""
        return await self.send_message(error_message, priority="ERROR")

    async def send_critical_notification(self, error_message: str):
        """Send critical error notification"""
        return await self.send_message(error_message, priority="CRITICAL")

    async def send_restart_notification(self, process_name: str):
        """Send process restart notification"""
        message = f"Process **{process_name}** has been restarted successfully."
        return await self.send_message(message, priority="WARNING")

    async def send_recovery_notification(self, process_name: str):
        """Send recovery success notification"""
        message = f"Process **{process_name}** recovered and running normally."
        return await self.send_message(message, priority="SUCCESS")

# Global notifier instance
_notifier = None

def get_notifier() -> ErrorNotifier:
    """Get or create notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = ErrorNotifier()
    return _notifier

async def send_error_notification(message: str):
    """Convenience function to send error notification"""
    notifier = get_notifier()
    return await notifier.send_error_notification(message)

async def send_critical_notification(message: str):
    """Convenience function to send critical notification"""
    notifier = get_notifier()
    return await notifier.send_critical_notification(message)

async def send_restart_notification(process_name: str):
    """Convenience function to send restart notification"""
    notifier = get_notifier()
    return await notifier.send_restart_notification(process_name)

async def send_recovery_notification(process_name: str):
    """Convenience function to send recovery notification"""
    notifier = get_notifier()
    return await notifier.send_recovery_notification(process_name)

# Test function
async def test_notification():
    """Test notification system"""
    notifier = ErrorNotifier()
    await notifier.send_message("omerGPT notification system test", priority="INFO")
    print("Test notification sent")

if __name__ == "__main__":
    asyncio.run(test_notification())
