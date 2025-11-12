"""
src/macro_engine/fred_macro.py

Asynchronous FRED (Federal Reserve Economic Data) API fetcher for macroeconomic indicators.
Retrieves key macro metrics: CPI, DXY (Dollar Index proxy), VIX, and Federal Funds Rate.
Includes mock data fallback for offline/failed API calls and stores results in DuckDB.
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional

import aiohttp
import duckdb

logger = logging.getLogger("omerGPT.macro.fred")


class FREDMacroFetcher:
    """
    Async FRED API client for fetching macroeconomic indicators.
    
    Features:
    - Fetch CPI, DXY proxy, VIX, Federal Funds Rate
    - Exponential backoff retry logic
    - Mock data fallback when API unavailable
    - DuckDB storage for historical tracking
    """
    
    # FRED API configuration
    FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"
    
    # Series IDs for key indicators
    SERIES_IDS = {
        "CPIAUCSL": "Consumer Price Index (CPI)",
        "DEXUSEU": "US Dollar Index (DXY Proxy)",
        "VIXCLS": "VIX Volatility Index",
        "FEDFUNDS": "Federal Funds Rate",
    }
    
    def __init__(self, db_path: str = "data/macro_data.duckdb", api_key: Optional[str] = None):
        """
        Initialize FRED macro fetcher.
        
        Args:
            db_path: Path to DuckDB database
            api_key: FRED API key (from env if not provided)
        """
        self.db_path = db_path
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        
        if not self.api_key:
            logger.warning(
                "FRED_API_KEY not set - will use mock data fallback. "
                "Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        
        # Initialize database connection
        self.conn = duckdb.connect(db_path)
        self._create_table()
        
        logger.info(f"FREDMacroFetcher initialized: db={db_path}")
    
    def _create_table(self):
        """Create macro_indicators table if it doesn't exist."""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_indicators (
                    date DATE NOT NULL,
                    indicator VARCHAR NOT NULL,
                    value DOUBLE,
                    description VARCHAR,
                    retrieved_ts BIGINT DEFAULT CAST(strftime('%s', 'now') AS BIGINT),
                    PRIMARY KEY (date, indicator)
                )
            """)
            
            logger.info("macro_indicators table ready")
        
        except Exception as e:
            logger.error(f"Failed to create table: {e}", exc_info=True)
            raise
    
    async def fetch_series(
        self,
        series_id: str,
        max_retries: int = 3
    ) -> List[Dict]:
        """
        Fetch a single FRED series with retry logic.
        
        Args:
            series_id: FRED series ID (e.g., 'CPIAUCSL')
            max_retries: Maximum retry attempts
        
        Returns:
            List of dictionaries with date, indicator, value, description
        """
        if not self.api_key:
            logger.warning(f"No API key - using mock data for {series_id}")
            return self._generate_mock_data(series_id)
        
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": 100,  # Last 100 observations
        }
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.FRED_API_BASE,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as resp:
                        
                        if resp.status == 200:
                            data = await resp.json()
                            observations = data.get("observations", [])
                            
                            results = []
                            for obs in observations:
                                value_str = obs.get("value", ".")
                                
                                # Skip missing values
                                if value_str == ".":
                                    continue
                                
                                try:
                                    value = float(value_str)
                                except ValueError:
                                    continue
                                
                                results.append({
                                    "date": obs.get("date"),
                                    "indicator": series_id,
                                    "value": value,
                                    "description": self.SERIES_IDS.get(series_id, ""),
                                })
                            
                            logger.info(
                                f"Fetched {len(results)} observations for {series_id}"
                            )
                            return results
                        
                        elif resp.status == 400:
                            error_text = await resp.text()
                            logger.error(
                                f"FRED API error 400 for {series_id}: {error_text}"
                            )
                            break  # Don't retry on bad request
                        
                        else:
                            logger.warning(
                                f"FRED API HTTP {resp.status} for {series_id}"
                            )
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {series_id} (attempt {attempt+1})")
            
            except Exception as e:
                logger.error(f"Error fetching {series_id}: {e}")
            
            # Exponential backoff
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.info(f"Retrying {series_id} in {backoff}s...")
                await asyncio.sleep(backoff)
        
        # Fallback to mock data
        logger.warning(f"API failed for {series_id} - using mock data")
        return self._generate_mock_data(series_id)
    
    def _generate_mock_data(self, series_id: str) -> List[Dict]:
        """
        Generate realistic mock data for a series.
        
        Args:
            series_id: FRED series ID
        
        Returns:
            List of mock data dictionaries
        """
        import random
        from datetime import datetime, timedelta
        
        # Base values for each indicator
        base_values = {
            "CPIAUCSL": 310.0,  # CPI ~310
            "DEXUSEU": 1.08,    # EUR/USD ~1.08
            "VIXCLS": 15.0,     # VIX ~15
            "FEDFUNDS": 5.25,   # Fed Funds ~5.25%
        }
        
        base = base_values.get(series_id, 100.0)
        results = []
        
        # Generate last 30 days of mock data
        for i in range(30):
            date = (datetime.utcnow() - timedelta(days=30-i)).strftime("%Y-%m-%d")
            
            # Add small random variation
            variation = random.uniform(-0.02, 0.02)
            value = base * (1 + variation)
            
            results.append({
                "date": date,
                "indicator": series_id,
                "value": round(value, 2),
                "description": f"{self.SERIES_IDS.get(series_id, '')} (MOCK)",
            })
        
        return results
    
    async def fetch_all(self) -> List[Dict]:
        """
        Fetch all configured macro indicators in parallel.
        
        Returns:
            Combined list of all indicator observations
        """
        tasks = [
            self.fetch_series(series_id) 
            for series_id in self.SERIES_IDS.keys()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_data = [item for sublist in results for item in sublist]
        
        logger.info(f"Total fetched: {len(all_data)} macro observations")
        return all_data
    
    async def save_to_db(self, records: List[Dict]):
        """
        Save macro indicator records to DuckDB.
        
        Args:
            records: List of indicator dictionaries
        """
        if not records:
            logger.warning("No records to save")
            return
        
        try:
            params = [
                (
                    r["date"],
                    r["indicator"],
                    r["value"],
                    r["description"],
                    int(time.time()),
                )
                for r in records
            ]
            
            self.conn.executemany("""
                INSERT OR REPLACE INTO macro_indicators 
                (date, indicator, value, description, retrieved_ts)
                VALUES (?, ?, ?, ?, ?)
            """, params)
            
            logger.info(f"Saved {len(params)} macro records to database")
        
        except Exception as e:
            logger.error(f"Failed to save macro records: {e}", exc_info=True)
    
    async def fetch_and_store(self) -> List[Dict]:
        """
        Complete workflow: fetch all indicators and store in database.
        
        Returns:
            List of fetched records
        """
        records = await self.fetch_all()
        await self.save_to_db(records)
        return records
    
    def get_latest_values(self) -> Dict[str, float]:
        """
        Get the most recent value for each indicator.
        
        Returns:
            Dictionary mapping indicator name to latest value
        """
        result = self.conn.execute("""
            SELECT indicator, value, date
            FROM macro_indicators
            WHERE (indicator, date) IN (
                SELECT indicator, MAX(date) as max_date
                FROM macro_indicators
                GROUP BY indicator
            )
        """)
        
        latest = {}
        for row in result.fetchall():
            indicator, value, date = row
            latest[indicator] = {
                "value": float(value),
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
    import pandas as pd
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    async def test_fred():
        """Test FRED macro fetcher."""
        print("Testing FREDMacroFetcher...")
        
        fetcher = FREDMacroFetcher("data/test_macro.duckdb")
        
        print("\n1. Fetching macro indicators...")
        records = await fetcher.fetch_and_store()
        
        print(f"\n2. Fetched {len(records)} total records")
        
        # Show sample data
        print("\n3. Sample records:")
        df = pd.DataFrame(records[:10])
        print(df[["date", "indicator", "value", "description"]])
        
        # Show latest values
        print("\n4. Latest indicator values:")
        latest = fetcher.get_latest_values()
        for indicator, data in latest.items():
            print(f"   {indicator}: {data['value']:.2f} (as of {data['date']})")
        
        fetcher.close()
        print("\nTest completed successfully!")
    
    asyncio.run(test_fred())
