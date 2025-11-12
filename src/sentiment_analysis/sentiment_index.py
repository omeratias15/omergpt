"""
src/sentiment_analysis/sentiment_index.py

Daily sentiment index aggregator for cryptocurrency markets.
Combines Reddit posts with FinBERT sentiment scores to generate normalized
sentiment indices for BTC, ETH, and other tracked symbols.
Stores results in DuckDB and exports to JSON/CSV for downstream analysis.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import duckdb
import pandas as pd

logger = logging.getLogger("omerGPT.sentiment.index")


class SentimentIndexGenerator:
    """
    Aggregate and normalize sentiment scores into daily indices per symbol.
    
    Features:
    - Combine Reddit + FinBERT analysis results
    - Calculate weighted sentiment scores
    - Normalize to [-1, 1] range
    - Per-symbol and market-wide indices
    - Export to multiple formats (DuckDB, JSON, CSV)
    """
    
    def __init__(self, db_path: str = "data/sentiment_data.duckdb"):
        """
        Initialize sentiment index generator.
        
        Args:
            db_path: Path to DuckDB database containing sentiment data
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        
        # Symbol to subreddit mapping
        self.symbol_subreddit_map = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum",
            "SOL": "CryptoMarkets",
        }
        
        # Create sentiment index table
        self._create_table()
        
        logger.info(f"SentimentIndexGenerator initialized: {db_path}")
    
    def _create_table(self):
        """Create sentiment_index table if it doesn't exist."""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_index (
                    date DATE NOT NULL,
                    symbol VARCHAR NOT NULL,
                    avg_sentiment DOUBLE,
                    normalized_sentiment DOUBLE,
                    post_count INTEGER,
                    positive_count INTEGER,
                    negative_count INTEGER,
                    neutral_count INTEGER,
                    created_at BIGINT DEFAULT CAST(epoch(CURRENT_TIMESTAMP) AS BIGINT),
                    PRIMARY KEY (date, symbol)
                )
            """)
            
            logger.info("sentiment_index table ready")
        
        except Exception as e:
            logger.error(f"Failed to create sentiment_index table: {e}", exc_info=True)
            raise
    
    async def compute_daily_index(
        self,
        symbol: str,
        date: Optional[str] = None
    ) -> Dict:
        """
        Compute daily sentiment index for a specific symbol.
        
        Args:
            symbol: Crypto symbol (BTC, ETH, SOL)
            date: Date string (YYYY-MM-DD), defaults to today
        
        Returns:
            Dictionary with sentiment metrics
        """
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")
        
        subreddit = self.symbol_subreddit_map.get(symbol)
        if not subreddit:
            logger.warning(f"No subreddit mapping for symbol: {symbol}")
            return self._empty_result(symbol, date)
        
        try:
            # Query posts with sentiment scores for the date
            query = """
                SELECT 
                    title,
                    score,
                    num_comments,
                    created_utc
                FROM reddit_posts
                WHERE subreddit = ?
                AND DATE(DATETIME(created_utc, 'unixepoch')) = ?
            """
            
            df = self.conn.execute(query, (subreddit, date)).df()
            
            if df.empty:
                logger.warning(f"No posts found for {symbol} on {date}")
                return self._empty_result(symbol, date)
            
            # Simulate FinBERT analysis (in production, this would query actual results)
            # For now, we'll use a simple heuristic based on post engagement
            df['sentiment_score'] = df.apply(
                lambda row: self._estimate_sentiment(row['title'], row['score']),
                axis=1
            )
            
            # Classify sentiments
            df['sentiment_label'] = df['sentiment_score'].apply(
                lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral')
            )
            
            # Calculate aggregated metrics
            avg_sentiment = df['sentiment_score'].mean()
            normalized_sentiment = max(-1.0, min(1.0, avg_sentiment))
            
            positive_count = int((df['sentiment_label'] == 'Positive').sum())
            negative_count = int((df['sentiment_label'] == 'Negative').sum())
            neutral_count = int((df['sentiment_label'] == 'Neutral').sum())
            
            result = {
                'date': date,
                'symbol': symbol,
                'avg_sentiment': float(avg_sentiment),
                'normalized_sentiment': float(normalized_sentiment),
                'post_count': len(df),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
            }
            
            logger.info(
                f"{symbol} {date}: sentiment={normalized_sentiment:.2f}, "
                f"posts={len(df)} (P:{positive_count} N:{negative_count})"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to compute sentiment for {symbol}: {e}", exc_info=True)
            return self._empty_result(symbol, date)
    
    def _estimate_sentiment(self, title: str, score: int) -> float:
        """
        Estimate sentiment from title and engagement score.
        This is a placeholder - in production use actual FinBERT results.
        
        Args:
            title: Post title text
            score: Reddit post score (upvotes - downvotes)
        
        Returns:
            Sentiment score between -1 and 1
        """
        # Simple keyword-based heuristic
        positive_words = ['bullish', 'surge', 'moon', 'pump', 'rally', 'green', 'up']
        negative_words = ['bearish', 'crash', 'dump', 'down', 'red', 'fear', 'sell']
        
        title_lower = title.lower()
        
        # Count keyword occurrences
        pos_count = sum(1 for word in positive_words if word in title_lower)
        neg_count = sum(1 for word in negative_words if word in title_lower)
        
        # Combine keyword analysis with engagement
        keyword_score = (pos_count - neg_count) * 0.3
        engagement_score = min(1.0, max(-1.0, score / 100.0)) * 0.7
        
        return keyword_score + engagement_score
    
    def _empty_result(self, symbol: str, date: str) -> Dict:
        """Return empty result structure."""
        return {
            'date': date,
            'symbol': symbol,
            'avg_sentiment': 0.0,
            'normalized_sentiment': 0.0,
            'post_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
        }
    
    async def compute_all_symbols(
        self,
        date: Optional[str] = None
    ) -> List[Dict]:
        """
        Compute sentiment index for all tracked symbols.
        
        Args:
            date: Date string (YYYY-MM-DD), defaults to today
        
        Returns:
            List of sentiment index dictionaries
        """
        results = []
        
        for symbol in self.symbol_subreddit_map.keys():
            result = await self.compute_daily_index(symbol, date)
            results.append(result)
        
        return results
    
    async def save_to_db(self, records: List[Dict]):
        """
        Save sentiment index records to DuckDB.
        
        Args:
            records: List of sentiment index dictionaries
        """
        if not records:
            logger.warning("No records to save")
            return
        
        try:
            params = [
                (
                    r['date'],
                    r['symbol'],
                    r['avg_sentiment'],
                    r['normalized_sentiment'],
                    r['post_count'],
                    r['positive_count'],
                    r['negative_count'],
                    r['neutral_count'],
                )
                for r in records
            ]
            
            self.conn.executemany("""
                INSERT OR REPLACE INTO sentiment_index 
                (date, symbol, avg_sentiment, normalized_sentiment, 
                 post_count, positive_count, negative_count, neutral_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, params)
            
            logger.info(f"Saved {len(params)} sentiment records to database")
        
        except Exception as e:
            logger.error(f"Failed to save sentiment records: {e}", exc_info=True)
    
    def export_to_json_csv(
        self,
        records: List[Dict],
        output_dir: str = "data/exports"
    ):
        """
        Export sentiment index to JSON and CSV files.
        
        Args:
            records: List of sentiment index dictionaries
            output_dir: Output directory for exports
        """
        if not records:
            logger.warning("No records to export")
            return
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            date = records[0]['date']
            
            # Export JSON
            json_path = os.path.join(output_dir, f"sentiment_index_{date}.json")
            with open(json_path, 'w') as f:
                json.dump(records, f, indent=2)
            
            # Export CSV
            csv_path = os.path.join(output_dir, f"sentiment_index_{date}.csv")
            pd.DataFrame(records).to_csv(csv_path, index=False)
            
            logger.info(f"Exported sentiment index to {json_path} and {csv_path}")
        
        except Exception as e:
            logger.error(f"Failed to export sentiment index: {e}", exc_info=True)
    
    async def generate_daily_index(
        self,
        date: Optional[str] = None
    ) -> List[Dict]:
        """
        Complete workflow: compute, save, and export daily sentiment index.
        
        Args:
            date: Date string (YYYY-MM-DD), defaults to today
        
        Returns:
            List of sentiment index records
        """
        records = await self.compute_all_symbols(date)
        await self.save_to_db(records)
        self.export_to_json_csv(records)
        return records
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Test block
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    async def test_sentiment_index():
        """Test sentiment index generation."""
        print("Testing SentimentIndexGenerator...")
        
        generator = SentimentIndexGenerator("data/test_sentiment.duckdb")
        
        # Generate today's index
        print("\n1. Computing daily sentiment index...")
        today = datetime.utcnow().strftime("%Y-%m-%d")
        records = await generator.generate_daily_index(today)
        
        print(f"\n2. Generated {len(records)} sentiment indices:")
        print(f"{'Symbol':<8} {'Sentiment':<10} {'Posts':<8} {'P/N/Neu'}")
        print("-" * 50)
        
        for rec in records:
            print(
                f"{rec['symbol']:<8} "
                f"{rec['normalized_sentiment']:>9.2f} "
                f"{rec['post_count']:>7} "
                f"{rec['positive_count']}/{rec['negative_count']}/{rec['neutral_count']}"
            )
        
        generator.close()
        print("\nTest completed successfully!")
    
    asyncio.run(test_sentiment_index())
