"""
CoinGecko Market/Sentiment Poller
Periodically scans CoinGecko for crypto market trends, social metrics, sentiment blending
Integrates with sentiment_index.py for composite index calculation
Async batching, rate limits, error handling, writes results to DuckDB
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional
from datetime import datetime
import yaml
import os

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
POLL_INTERVAL = 60  # seconds, adjust per config ("coingecko_scan_interval")
COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]  # Extend as needed

class CoinGeckoScan:
    def __init__(
        self,
        coins: List[str] = COINS,
        db_manager=None,
        on_sentiment_update: Optional[callable] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            coins: List of coin IDs as per CoinGecko
            db_manager: Storage/DB manager instance
            on_sentiment_update: Async callback(sentiment_dict)
            logger: Logger instance
        """
        self.coins = coins
        self.db_manager = db_manager
        self.on_sentiment_update = on_sentiment_update
        self.logger = logger or logging.getLogger("CoinGeckoScan")
        self._stop = asyncio.Event()
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = {"polls": 0, "errors": 0, "updated": 0}

    async def start(self):
        """Main polling loop"""
        self.logger.info("🚀 Starting CoinGecko sentiment scan...")
        self.session = aiohttp.ClientSession()
        while not self._stop.is_set():
            try:
                await self._poll_coins()
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                self.logger.info("⛔ CoinGecko scan cancelled")
                break
            except Exception as e:
                self.logger.error(f"❌ Scan error: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(10)

        await self.session.close()

    async def stop(self):
        """Graceful shutdown"""
        self.logger.info("🛑 Stopping CoinGecko scan...")
        self._stop.set()
        if self.session:
            await self.session.close()
        self.logger.info(f"📊 Final stats: {self.stats}")

    async def _poll_coins(self):
        """Scan CoinGecko for each coin and aggregate sentiment"""
        for coin in self.coins:
            data = await self._fetch_coin_data(coin)
            if data:
                sentiment = self._analyze_sentiment(coin, data)
                self.stats["updated"] += 1
                # Store to DB
                if self.db_manager:
                    await self.db_manager.insert_sentiment(coin, sentiment)

                # Trigger callback
                if self.on_sentiment_update:
                    await self.on_sentiment_update(sentiment)

        self.stats["polls"] += 1

    async def _fetch_coin_data(self, coin: str) -> Optional[Dict]:
        """Fetch market/social metrics from CoinGecko"""
        url = f"{COINGECKO_API_URL}/coins/{coin}"
        try:
            async with self.session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data
                else:
                    self.logger.warning(f"❌ Failed to fetch {coin}: {resp.status}")
        except Exception as e:
            self.logger.error(f"❌ Error fetching {coin}: {e}")
        return None

    def _analyze_sentiment(self, coin: str, data: Dict) -> Dict:
        """
        Blend market/social stats into a sentiment index (0.0-1.0)
        Combine price_change_percentage_24h, market_cap, community_score, developer_score
        """
        market_data = data.get("market_data", {})
        sentiment = {
            "timestamp": datetime.now().timestamp(),
            "coin": coin,
            "price_usd": market_data.get("current_price", {}).get("usd", 0),
            "volume_usd": market_data.get("total_volume", {}).get("usd", 0),
            "cap_rank": data.get("market_cap_rank", None),
            "trend_score": float(data.get("community_score", 0)),
            "dev_score": float(data.get("developer_score", 0)),
            "social_score": float(data.get("public_interest_score", 0)),
            "change_24h": float(market_data.get("price_change_percentage_24h", 0)),
            "sentiment_index": self._blend_index(data)
        }
        self.logger.info(f"💡 Sentiment[{coin}]: idx={sentiment['sentiment_index']:.2f} 24h={sentiment['change_24h']:.2f}%")
        return sentiment

    def _blend_index(self, data: Dict) -> float:
        """
        Weighted average of social/community/developer 0-1, price change normalized
        """
        market_data = data.get("market_data", {})
        scores = [
            float(data.get("community_score", 0)),
            float(data.get("developer_score", 0)),
            float(data.get("public_interest_score", 0)),
            float(market_data.get("price_change_percentage_24h", 0)) / 100  # Normalize %
        ]
        # Simple normalization and blend
        return min(max(sum(scores) / len(scores), 0.0), 1.0)

    def get_stats(self) -> Dict:
        return self.stats.copy()


# Example callback
async def print_sentiment(sentiment):
    print(f"⏫ {sentiment['coin']} sentiment: {sentiment['sentiment_index']:.2f}")

async def main():
    logging.basicConfig(level=logging.INFO)
    cgScan = CoinGeckoScan(on_sentiment_update=print_sentiment)
    task = asyncio.create_task(cgScan.start())
    await asyncio.sleep(120)  # Run for 2 minutes
    await cgScan.stop()
    await task

if __name__ == "__main__":
    asyncio.run(main())
