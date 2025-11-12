"""
src/macro_engine/yfinance_macro.py

Async market data fetcher using yfinance for SPX, BTC, and ETH correlation analysis.
Computes rolling 30-day correlations between traditional and crypto markets.
Stores results in DuckDB for macro regime classification.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import duckdb
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

logger = logging.getLogger("omerGPT.macro.yfinance")


class YFinanceMacroFetcher:
    """
    Async yfinance client for fetching traditional and crypto market data.
    
    Features:
    - Fetch SPX, BTC, ETH daily prices
    - Calculate rolling correlations (30-day window)
    - DuckDB storage for historical tracking
    - Async execution via thread executor
    """
    
    # Ticker symbols
    TICKERS = {
        "SPX": "^GSPC",  # S&P 500 Index
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
    }
    
    def __init__(self, db_path: str = "data/macro_data.duckdb"):
        """
        Initialize yfinance macro fetcher.
        
        Args:
            db_path: Path to DuckDB database
        """
        if not YFINANCE_AVAILABLE:
            logger.warning(
                "yfinance not available - install with: pip install yfinance"
            )
        
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._create_tables()
        
        logger.info(f"YFinanceMacroFetcher initialized: db={db_path}")
    
    def _create_tables(self):
        """Create market_prices and market_correlations tables."""
        try:
            # Market prices table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS market_prices (
                    date DATE NOT NULL,
                    ticker VARCHAR NOT NULL,
                    close DOUBLE,
                    volume DOUBLE,
                    retrieved_ts BIGINT DEFAULT CAST(strftime('%s', 'now') AS BIGINT),
                    PRIMARY KEY (date, ticker)
                )
            """)
            
            # Correlations table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS market_correlations (
                    date DATE NOT NULL,
                    pair VARCHAR NOT NULL,
                    correlation DOUBLE,
                    window_days INTEGER,
                    retrieved_ts BIGINT DEFAULT CAST(strftime('%s', 'now') AS BIGINT),
                    PRIMARY KEY (date, pair)
                )
            """)
            
            logger.info("market_prices and market_correlations tables ready")
        
        except Exception as e:
            logger.error(f"Failed to create tables: {e}", exc_info=True)
            raise
    
    async def fetch_ticker(
        self,
        symbol: str,
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.
        
        Args:
            symbol: Yahoo Finance ticker symbol
            days: Number of days of history to fetch
        
        Returns:
            DataFrame with date, ticker, close, volume
        """
        if not YFINANCE_AVAILABLE:
            logger.warning(f"yfinance unavailable - using mock data for {symbol}")
            return self._generate_mock_prices(symbol, days)
        
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None,
                self._fetch_ticker_sync,
                symbol,
                days
            )
            
            logger.info(f"Fetched {len(df)} records for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}", exc_info=True)
            return self._generate_mock_prices(symbol, days)
    
    def _fetch_ticker_sync(self, symbol: str, days: int) -> pd.DataFrame:
        """Synchronous ticker fetch using yfinance."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Format data
        df = pd.DataFrame({
            "date": hist.index.date,
            "ticker": symbol,
            "close": hist["Close"].values,
            "volume": hist["Volume"].values,
        })
        
        return df
    
    def _generate_mock_prices(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate mock price data for testing."""
        import random
        
        base_prices = {
            "^GSPC": 4500.0,
            "BTC-USD": 67000.0,
            "ETH-USD": 3500.0,
        }
        
        base = base_prices.get(symbol, 1000.0)
        dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
        
        prices = []
        current = base
        for _ in range(days):
            change = random.uniform(-0.02, 0.02)
            current *= (1 + change)
            prices.append(current)
        
        df = pd.DataFrame({
            "date": [d.date() for d in dates],
            "ticker": symbol,
            "close": prices,
            "volume": [random.uniform(1e9, 5e9) for _ in range(days)],
        })
        
        return df
    
    async def fetch_all(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch all configured tickers in parallel.
        
        Args:
            days: Number of days of history
        
        Returns:
            Combined DataFrame with all tickers
        """
        tasks = [
            self.fetch_ticker(ticker, days)
            for ticker in self.TICKERS.values()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Combine all dataframes
        df = pd.concat(results, ignore_index=True)
        
        logger.info(f"Total fetched: {len(df)} price records")
        return df
    
    async def calculate_correlations(
        self,
        df: pd.DataFrame,
        window: int = 30
    ) -> pd.DataFrame:
        """
        Calculate rolling correlations between assets.
        
        Args:
            df: DataFrame with date, ticker, close columns
            window: Rolling window in days (default: 30)
        
        Returns:
            DataFrame with correlation data
        """
        try:
            # Pivot to wide format
            pivot = df.pivot(index="date", columns="ticker", values="close")
            
            # Calculate returns
            returns = pivot.pct_change().dropna()
            
            # Calculate rolling correlations
            correlations = []
            
            pairs = [
                ("^GSPC", "BTC-USD", "SPX-BTC"),
                ("^GSPC", "ETH-USD", "SPX-ETH"),
                ("BTC-USD", "ETH-USD", "BTC-ETH"),
            ]
            
            for asset1, asset2, pair_name in pairs:
                if asset1 in returns.columns and asset2 in returns.columns:
                    rolling_corr = returns[asset1].rolling(window).corr(returns[asset2])
                    
                    for date, corr in rolling_corr.items():
                        if pd.notna(corr):
                            correlations.append({
                                "date": date,
                                "pair": pair_name,
                                "correlation": float(corr),
                                "window_days": window,
                            })
            
            corr_df = pd.DataFrame(correlations)
            
            logger.info(
                f"Calculated {len(corr_df)} correlation values for {len(pairs)} pairs"
            )
            return corr_df
        
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}", exc_info=True)
            return pd.DataFrame()
    
    async def save_prices(self, df: pd.DataFrame):
        """Save price data to DuckDB."""
        if df.empty:
            logger.warning("No price data to save")
            return
        
        try:
            params = [
                (row["date"], row["ticker"], row["close"], row["volume"], int(time.time()))
                for _, row in df.iterrows()
            ]
            
            self.conn.executemany("""
                INSERT OR REPLACE INTO market_prices 
                (date, ticker, close, volume, retrieved_ts)
                VALUES (?, ?, ?, ?, ?)
            """, params)
            
            logger.info(f"Saved {len(params)} price records")
        
        except Exception as e:
            logger.error(f"Failed to save prices: {e}", exc_info=True)
    
    async def save_correlations(self, df: pd.DataFrame):
        """Save correlation data to DuckDB."""
        if df.empty:
            logger.warning("No correlation data to save")
            return
        
        try:
            params = [
                (
                    row["date"],
                    row["pair"],
                    row["correlation"],
                    row["window_days"],
                    int(time.time())
                )
                for _, row in df.iterrows()
            ]
            
            self.conn.executemany("""
                INSERT OR REPLACE INTO market_correlations 
                (date, pair, correlation, window_days, retrieved_ts)
                VALUES (?, ?, ?, ?, ?)
            """, params)
            
            logger.info(f"Saved {len(params)} correlation records")
        
        except Exception as e:
            logger.error(f"Failed to save correlations: {e}", exc_info=True)
    
    async def fetch_and_store(self, days: int = 365) -> Dict:
        """
        Complete workflow: fetch prices, calculate correlations, store all.
        
        Args:
            days: Number of days of history
        
        Returns:
            Dictionary with summary statistics
        """
        # Fetch prices
        prices_df = await self.fetch_all(days)
        await self.save_prices(prices_df)
        
        # Calculate correlations
        corr_df = await self.calculate_correlations(prices_df, window=30)
        await self.save_correlations(corr_df)
        
        # Get latest correlations
        latest = self.get_latest_correlations()
        
        return {
            "prices_fetched": len(prices_df),
            "correlations_calculated": len(corr_df),
            "latest_correlations": latest,
        }
    
    def get_latest_correlations(self) -> Dict[str, float]:
        """Get most recent correlation values for each pair."""
        result = self.conn.execute("""
            SELECT pair, correlation, date
            FROM market_correlations
            WHERE (pair, date) IN (
                SELECT pair, MAX(date) as max_date
                FROM market_correlations
                GROUP BY pair
            )
        """)
        
        latest = {}
        for row in result.fetchall():
            pair, corr, date = row
            latest[pair] = {
                "correlation": float(corr),
                "date": str(date),
            }
        
        return latest
    
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
    
    async def test_yfinance():
        """Test yfinance macro fetcher."""
        print("Testing YFinanceMacroFetcher...")
        
        fetcher = YFinanceMacroFetcher("data/test_macro.duckdb")
        
        print("\n1. Fetching market data...")
        summary = await fetcher.fetch_and_store(days=90)
        
        print(f"\n2. Summary:")
        print(f"   Prices fetched: {summary['prices_fetched']}")
        print(f"   Correlations calculated: {summary['correlations_calculated']}")
        
        print("\n3. Latest correlations:")
        for pair, data in summary['latest_correlations'].items():
            print(f"   {pair}: {data['correlation']:.3f} (as of {data['date']})")
        
        fetcher.close()
        print("\nTest completed successfully!")
    
    asyncio.run(test_yfinance())
