"""
src/sentiment_analysis/reddit_fetcher.py

Production-grade asynchronous Reddit JSON scraper for cryptocurrency sentiment analysis.
Fetches recent posts from r/CryptoMarkets, r/Bitcoin, and r/Ethereum without API key.
Includes exponential backoff retry logic and DuckDB storage for persistent data.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

import aiohttp
import duckdb

logger = logging.getLogger("omerGPT.sentiment.reddit")


class RedditFetcher:
    """
    Async Reddit scraper for cryptocurrency-related subreddits.
    
    Features:
    - No authentication required (public JSON endpoints)
    - Exponential backoff on failures
    - Rate-limit aware with configurable delays
    - Persistent storage in DuckDB
    - Multi-subreddit parallel fetching
    """
    
    def __init__(self, db_path: str = "data/sentiment_data.duckdb"):
        """
        Initialize Reddit fetcher with DuckDB connection.
        
        Args:
            db_path: Path to DuckDB database for storing posts
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
        # Default subreddits to monitor
        self.subreddits = ["CryptoMarkets", "Bitcoin", "Ethereum"]
        
        # Reddit public JSON endpoint
        self.reddit_url = "https://www.reddit.com/r/{}/new.json?limit=50"
        
        # User agent for Reddit API
        self.headers = {
            "User-Agent": "omerGPT-bot/1.0"
        }
        
        # Create database table
        self._create_table()
        
        logger.info(f"RedditFetcher initialized with DB: {db_path}")
    
    def _create_table(self):
        """Create reddit_posts table if it doesn't exist."""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS reddit_posts (
                    subreddit VARCHAR NOT NULL,
                    title VARCHAR NOT NULL,
                    score INTEGER,
                    num_comments INTEGER,
                    created_utc BIGINT NOT NULL,
                    permalink VARCHAR NOT NULL,
                    retrieved_ts BIGINT DEFAULT CAST(epoch(current_timestamp) AS BIGINT),
                    PRIMARY KEY (subreddit, permalink)
                )
            """)
            
            logger.info("reddit_posts table ready")
        
        except Exception as e:
            logger.error(f"Failed to create table: {e}", exc_info=True)
            raise
    
    async def fetch_subreddit(
        self,
        session: aiohttp.ClientSession,
        subreddit: str,
        max_retries: int = 5
    ) -> List[Dict]:
        """
        Fetch recent posts from a specific subreddit with retry logic.
        
        Args:
            session: aiohttp client session
            subreddit: Subreddit name (without r/ prefix)
            max_retries: Maximum number of retry attempts
        
        Returns:
            List of post dictionaries
        """
        url = self.reddit_url.format(subreddit)
        posts = []
        retries = 0
        
        while retries < max_retries:
            try:
                async with session.get(
                    url,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    
                    if resp.status == 200:
                        data = await resp.json()
                        
                        for item in data.get("data", {}).get("children", []):
                            p = item.get("data", {})
                            posts.append({
                                "subreddit": p.get("subreddit", subreddit),
                                "title": p.get("title", ""),
                                "score": int(p.get("score", 0)),
                                "num_comments": int(p.get("num_comments", 0)),
                                "created_utc": int(p.get("created_utc", 0)),
                                "permalink": p.get("permalink", ""),
                            })
                        
                        logger.info(
                            f"Fetched {len(posts)} posts from r/{subreddit}"
                        )
                        return posts
                    
                    elif resp.status == 429:
                        # Rate limit - wait longer
                        retry_after = int(resp.headers.get("Retry-After", 60))
                        logger.warning(
                            f"Rate limited on r/{subreddit}, waiting {retry_after}s"
                        )
                        await asyncio.sleep(retry_after)
                        retries += 1
                    
                    else:
                        logger.warning(
                            f"HTTP {resp.status} fetching r/{subreddit}"
                        )
                        retries += 1
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching r/{subreddit}")
                retries += 1
            
            except Exception as e:
                logger.error(f"Error fetching r/{subreddit}: {e}")
                retries += 1
            
            # Exponential backoff
            if retries < max_retries:
                backoff = 2 ** retries
                logger.info(f"Retrying r/{subreddit} in {backoff}s...")
                await asyncio.sleep(backoff)
        
        logger.error(
            f"Failed to fetch r/{subreddit} after {max_retries} attempts"
        )
        return posts
    
    async def fetch_all(
        self,
        subreddits: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Fetch posts from multiple subreddits in parallel.
        
        Args:
            subreddits: List of subreddit names (default: self.subreddits)
        
        Returns:
            Combined list of all posts
        """
        if subreddits is None:
            subreddits = self.subreddits
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_subreddit(session, sub) for sub in subreddits
            ]
            results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_posts = [post for sublist in results for post in sublist]
        
        logger.info(f"Total fetched: {len(all_posts)} posts")
        return all_posts
    
    async def save_posts(self, posts: List[Dict]):
        """
        Save posts to DuckDB database.
        
        Args:
            posts: List of post dictionaries
        """
        if not posts:
            logger.warning("No posts to save")
            return
        
        try:
            # Prepare data for batch insert
            params = [
                (
                    p["subreddit"],
                    p["title"],
                    p["score"],
                    p["num_comments"],
                    p["created_utc"],
                    p["permalink"],
                    int(time.time())
                )
                for p in posts
            ]
            
            # Use INSERT OR REPLACE for idempotency
            self.conn.executemany("""
                INSERT OR REPLACE INTO reddit_posts 
                (subreddit, title, score, num_comments, created_utc, permalink, retrieved_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, params)
            
            logger.info(f"Saved {len(params)} posts to database")
        
        except Exception as e:
            logger.error(f"Failed to save posts: {e}", exc_info=True)
    
    async def fetch_and_store(
        self,
        subreddits: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Complete workflow: fetch and store posts.
        
        Args:
            subreddits: List of subreddit names
        
        Returns:
            List of fetched posts
        """
        posts = await self.fetch_all(subreddits)
        await self.save_posts(posts)
        return posts
    
    def get_recent_posts(self, hours: int = 24, limit: int = 100) -> List[Dict]:
        """
        Retrieve recently stored posts from database.
        
        Args:
            hours: Time window in hours
            limit: Maximum number of posts to return
        
        Returns:
            List of post dictionaries
        """
        cutoff_ts = int(time.time() - (hours * 3600))
        
        result = self.conn.execute("""
            SELECT subreddit, title, score, num_comments, created_utc, permalink
            FROM reddit_posts
            WHERE created_utc > ?
            ORDER BY created_utc DESC
            LIMIT ?
        """, (cutoff_ts, limit))
        
        posts = [
            {
                "subreddit": row[0],
                "title": row[1],
                "score": row[2],
                "num_comments": row[3],
                "created_utc": row[4],
                "permalink": row[5],
            }
            for row in result.fetchall()
        ]
        
        return posts
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Test block
if __name__ == "__main__":
    import pandas as pd
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    async def test_reddit():
        """Test Reddit fetcher with real data."""
        print("Testing RedditFetcher...")
        
        fetcher = RedditFetcher("data/test_sentiment.duckdb")
        
        # Fetch from all default subreddits
        print("\n1. Fetching posts from Reddit...")
        posts = await fetcher.fetch_and_store()
        
        print(f"\n2. Fetched {len(posts)} total posts")
        print("\nSample titles:")
        for i, post in enumerate(posts[:5], 1):
            print(f"  {i}. [{post['subreddit']}] {post['title'][:60]}...")
        
        # Read from DB
        print("\n3. Reading from database...")
        df = fetcher.conn.execute("""
            SELECT subreddit, title, score, num_comments 
            FROM reddit_posts 
            ORDER BY created_utc DESC 
            LIMIT 10
        """).df()
        
        print("\nDatabase sample:")
        print(df)
        
        fetcher.close()
        print("\nTest completed successfully!")
    
    asyncio.run(test_reddit())
