"""
src/storage/db_manager.py

Enhanced Asynchronous DuckDB Manager for OmerGPT.

Complete implementation with:
- Transaction context management
- Bulk insert optimization
- Health/status queries
- Meta information tracking
- Async checkpoint loop
- Smart logging
- AI Agent integration hooks
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
import numpy as np

logger = logging.getLogger("omerGPT.storage.db_manager")


class DatabaseManager:
    """
    Production-grade async DuckDB manager for OmerGPT.

    Enhanced features:
    - Transactional operations with context manager
    - Bulk insert with DataFrame registration
    - Meta information tracking
    - Automatic checkpoint scheduling
    - Health/status queries for orchestrator
    - Integration hooks for AI Agent
    """

    def __init__(self, db_path: str = "data/market_data.duckdb", auto_checkpoint: bool = True):
        """Initialize database manager."""
        self.db_path = db_path
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.lock = asyncio.Lock()
        self.auto_checkpoint_enabled = auto_checkpoint

        # Create data directory
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs("data/archive", exist_ok=True)

        # Initialize
        self._connect()
        self._init_schema()
        self._init_meta_table()
        self._optimize_db()

        logger.info(f"✓ DatabaseManager initialized: {db_path}")

    def _connect(self):
        """Establish DuckDB connection."""
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"✓ Connected to DuckDB: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            raise

    def _optimize_db(self):
        """Optimize DuckDB settings."""
        try:
            self.conn.execute("PRAGMA wal_autocheckpoint=1000")
            self.conn.execute("PRAGMA memory_limit='4GB'")
            self.conn.execute("PRAGMA threads=4")
            logger.info("✓ Database optimizations applied")
        except Exception as e:
            logger.warning(f"⚠️ Optimization warning: {e}")

    def _init_meta_table(self):
        """Create meta information table."""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS meta_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Initialize standard keys
            meta_keys = {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "last_checkpoint": datetime.now().isoformat()
            }

            for key, value in meta_keys.items():
                self.conn.execute("""
                    INSERT OR REPLACE INTO meta_info (key, value)
                    VALUES (?, ?)
                """, (key, value))

            logger.info("✓ Meta table initialized")
        except Exception as e:
            logger.warning(f"⚠️ Meta table warning: {e}")

    def _init_schema(self):
        """Create all required tables."""
        try:
            # Candles
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT,
                    ts_ms BIGINT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    close_time BIGINT
                )
            """)

            # Features - FIXED: ts_ms and computed_at are now BIGINT
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    symbol TEXT NOT NULL,
                    ts_ms BIGINT NOT NULL,
                    return_1m DOUBLE,
                    volatility_5m DOUBLE,
                    volatility_15m DOUBLE,
                    volatility_60m DOUBLE,
                    momentum_5m DOUBLE,
                    momentum_15m DOUBLE,
                    momentum_60m DOUBLE,
                    rsi_14 DOUBLE,
                    atr DOUBLE,
                    atr_pct DOUBLE,
                    macd DOUBLE,
                    bb_up DOUBLE,
                    bb_dn DOUBLE,
                    vol_ma DOUBLE,
                    spread DOUBLE,
                    ob_imbalance DOUBLE,
                    corr_btc_eth DOUBLE,
                    computed_at BIGINT,
                    PRIMARY KEY (symbol, ts_ms)
                )
            """)

            # Anomaly Events
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_events (
                    ts_ms TIMESTAMP NOT NULL,
                    symbol TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity INT NOT NULL,
                    confidence DOUBLE NOT NULL,
                    meta JSON,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ts_ms, symbol, event_type)
                )
            """)

            # Trading Signals
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    symbol TEXT NOT NULL,
                    ts_ms TIMESTAMP NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence DOUBLE NOT NULL,
                    reason TEXT,
                    anomaly_score DOUBLE,
                    rsi DOUBLE,
                    momentum DOUBLE,
                    feature_snapshot JSON,
                    status TEXT DEFAULT 'new',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sent_at TIMESTAMP,
                    PRIMARY KEY (symbol, ts_ms, signal_type)
                )
            """)

            # On-chain Metrics
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS onchain_metrics (
                    address TEXT NOT NULL,
                    tx_hash TEXT NOT NULL,
                    block_number INTEGER,
                    timestamp TIMESTAMP NOT NULL,
                    value_eth DOUBLE,
                    gas_price_gwei DOUBLE,
                    is_token BOOLEAN,
                    token_symbol TEXT,
                    token_name TEXT,
                    contract_address TEXT,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tx_hash)
                )
            """)

            # Trades (for backtesting/live)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    symbol TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    entry_price DOUBLE NOT NULL,
                    exit_price DOUBLE,
                    size DOUBLE NOT NULL,
                    pnl_dollars DOUBLE,
                    pnl_pct DOUBLE,
                    duration_minutes DOUBLE,
                    status TEXT DEFAULT 'open',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, entry_time)
                )
            """)

            # Orderbook snapshots (top-of-book)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS orderbook (
                    ts_ms BIGINT,
                    symbol TEXT,
                    bid_price REAL,
                    bid_size REAL,
                    ask_price REAL,
                    ask_size REAL
                )
            """)

            # Create Indexes
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_candles_symbol_time
                ON candles (symbol, ts_ms DESC)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_symbol_time
                ON features (symbol, ts_ms DESC)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_anomaly_symbol_time
                ON anomaly_events (symbol, ts_ms DESC)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol_time
                ON signals (symbol, ts_ms DESC)
            """)

            logger.info("✓ All tables and indexes initialized")
        except Exception as e:
            logger.error(f"❌ Schema initialization failed: {e}", exc_info=True)
            raise

    # ==================== TRANSACTION CONTEXT ====================

    @asynccontextmanager
    async def transaction(self):
        """
        Async transaction context manager.

        Usage:
            async with db.transaction() as conn:
                conn.execute("INSERT INTO table VALUES (...)")
        """
        async with self.lock:
            try:
                self.conn.execute("BEGIN TRANSACTION")
                yield self.conn
                self.conn.execute("COMMIT")
                logger.debug("✓ Transaction committed")
            except Exception as e:
                self.conn.execute("ROLLBACK")
                logger.error(f"❌ Transaction rolled back: {e}")
                raise

    # ==================== BULK INSERT ====================

    async def insert_bulk(
        self,
        table: str,
        df: pd.DataFrame,
        mode: str = "insert"
    ) -> int:
        """
        Bulk insert DataFrame with high performance.

        Args:
            table: Target table name
            df: DataFrame to insert
            mode: 'insert' or 'replace'

        Returns:
            Number of rows inserted
        """
        if df.empty:
            return 0
        
        # --- BRUTAL FIX: Force convert ALL datetime columns to int64 ---
        for col in df.columns:
            # Check if column dtype contains 'datetime' string
            if 'datetime' in str(df[col].dtype).lower():
                try:
                    # Remove timezone if present
                    if hasattr(df[col].dt, 'tz') and df[col].dt.tz is not None:
                        df[col] = df[col].dt.tz_localize(None)
                    # Convert to int64 milliseconds
                    df[col] = (df[col].astype('int64') // 1_000_000).astype('int64')
                    logger.debug(f"Converted {col} from datetime to int64")
                except Exception as e:
                    logger.warning(f"Failed to convert {col}: {e}")

        async with self.lock:
            try:
                loop = asyncio.get_running_loop()
                rows = await loop.run_in_executor(
                    None, self._bulk_insert_sync, table, df, mode
                )
                logger.debug(f"STORAGE/INSERT/{table.upper()}: {rows} rows")
                return rows
            except Exception as e:
                logger.error(f"❌ Bulk insert failed: {e}", exc_info=True)
                raise

    def _bulk_insert_sync(self, table: str, df: pd.DataFrame, mode: str) -> int:
        """Synchronous bulk insert."""
        try:
            self.conn.register("_tmp_bulk", df)

            if mode == "replace":
                # Get primary key columns
                pk_cols = self._get_primary_key_cols(table)
                if pk_cols:
                    on_conflict = ", ".join([f"{col} = EXCLUDED.{col}" for col in df.columns if col not in pk_cols])
                    query = f"""
                        INSERT INTO {table} SELECT * FROM _tmp_bulk
                        ON CONFLICT ({", ".join(pk_cols)}) DO UPDATE SET {on_conflict}
                    """
                else:
                    query = f"INSERT INTO {table} SELECT * FROM _tmp_bulk"
            else:
                query = f"INSERT INTO {table} SELECT * FROM _tmp_bulk"

            self.conn.execute(query)
            self.conn.unregister("_tmp_bulk")

            return len(df)
        except Exception as e:
            logger.error(f"Bulk insert sync error: {e}")
            raise

    def _get_primary_key_cols(self, table: str) -> List[str]:
        """Get primary key columns for a table."""
        try:
            result = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            pk_cols = [row[1] for row in result if row[5]]  # pk field is at index 5
            return pk_cols
        except Exception as e:
            logger.error(f"Failed to get primary keys: {e}")
            return []

    # ==================== HEALTH & STATUS ====================

    async def get_table_counts(self) -> Dict[str, int]:
        """
        Get row counts for all key tables.

        Returns:
            Dict with table name -> count
        """
        try:
            loop = asyncio.get_running_loop()
            counts = await loop.run_in_executor(None, self._get_table_counts_sync)
            return counts
        except Exception as e:
            logger.error(f"Failed to get table counts: {e}")
            return {}

    def _get_table_counts_sync(self) -> Dict[str, int]:
        """Synchronous table counts query."""
        tables = ["candles", "features", "anomaly_events", "signals", "onchain_metrics", "trades"]
        counts = {}

        for table in tables:
            try:
                result = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = result[0] if result else 0
            except Exception as e:
                logger.warning(f"Failed to count {table}: {e}")
                counts[table] = 0

        return counts

    async def get_latest_timestamp(self, table: str) -> Optional[datetime]:
        """
        Get most recent timestamp for a table.

        Args:
            table: Table name

        Returns:
            Latest timestamp or None
        """
        try:
            loop = asyncio.get_running_loop()
            ts = await loop.run_in_executor(None, self._get_latest_timestamp_sync, table)
            return ts
        except Exception as e:
            logger.error(f"Failed to get latest timestamp: {e}")
            return None

    def _get_latest_timestamp_sync(self, table: str) -> Optional[datetime]:
        """Synchronous latest timestamp query."""
        try:
            columns = [r[1] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()]
            ts_col = "ts_ms" if "ts_ms" in columns else "ts"
            result = self.conn.execute(f"SELECT MAX({ts_col}) FROM {table}").fetchone()
            return result[0] if result and result[0] else None
        except Exception as e:
            logger.error(f"Timestamp query error for {table}: {e}")
            return None

    async def get_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive health report.

        Returns:
            Health status dict
        """
        try:
            counts = await self.get_table_counts()

            report = {
                "timestamp": datetime.now().isoformat(),
                "status": "healthy",
                "tables": counts,
                "latest_timestamps": {}
            }

            for table in ["candles", "features", "anomaly_events", "signals"]:
                ts = await self.get_latest_timestamp(table)
                if ts:
                    report["latest_timestamps"][table] = ts.isoformat()

            return report
        except Exception as e:
            logger.error(f"Health report error: {e}")
            return {"status": "error", "error": str(e)}

    # ==================== META INFORMATION ====================

    async def set_meta(self, key: str, value: Any):
        """Store metadata."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._set_meta_sync, key, value)
            logger.debug(f"Meta: {key} = {value}")
        except Exception as e:
            logger.error(f"Failed to set meta: {e}")

    def _set_meta_sync(self, key: str, value: Any):
        """Synchronous meta set."""
        value_str = json.dumps(value) if not isinstance(value, str) else value
        self.conn.execute("""
            INSERT OR REPLACE INTO meta_info (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, value_str))

    async def get_meta(self, key: str) -> Optional[str]:
        """Retrieve metadata."""
        try:
            loop = asyncio.get_running_loop()
            value = await loop.run_in_executor(None, self._get_meta_sync, key)
            return value
        except Exception as e:
            logger.error(f"Failed to get meta: {e}")
            return None

    def _get_meta_sync(self, key: str) -> Optional[str]:
        """Synchronous meta get."""
        result = self.conn.execute(
            "SELECT value FROM meta_info WHERE key = ?",
            (key,)
        ).fetchone()
        return result[0] if result else None

    # ==================== AI AGENT INTEGRATION ====================

    async def fetch_recent_data(
        self,
        table: str,
        limit: int = 100,
        lookback_minutes: int = 60
    ) -> pd.DataFrame:
        """
        Fetch recent data for AI Agent analysis.

        Args:
            table: Table to query
            limit: Row limit
            lookback_minutes: Time window

        Returns:
            DataFrame with recent data
        """
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None,
                self._fetch_recent_data_sync,
                table,
                limit,
                lookback_minutes
            )
            logger.debug(f"Agent query: {table} -> {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Agent data fetch failed: {e}")
            return pd.DataFrame()

    def _fetch_recent_data_sync(
        self,
        table: str,
        limit: int,
        lookback_minutes: int
    ) -> pd.DataFrame:
        """Synchronous recent data fetch."""
        try:
            cutoff = datetime.now() - timedelta(minutes=lookback_minutes)

            query = f"""
                SELECT * FROM {table}
                WHERE ts_ms >= ?
                ORDER BY ts_ms DESC
                LIMIT ?
            """

            result = self.conn.execute(query, (cutoff, limit))
            return result.df()
        except Exception as e:
            logger.error(f"Recent data fetch error: {e}")
            return pd.DataFrame()

    # ==================== CHECKPOINT & MAINTENANCE ====================

    async def checkpoint(self):
        """Force database checkpoint."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._checkpoint_sync)
            await self.set_meta("last_checkpoint", datetime.now().isoformat())
            logger.debug("✓ Checkpoint completed")
        except Exception as e:
            logger.warning(f"⚠️ Checkpoint warning: {e}")

    def _checkpoint_sync(self):
        """Synchronous checkpoint."""
        self.conn.execute("CHECKPOINT")

    async def checkpoint_loop(self, interval: int = 300):
        """
        Automatic checkpoint loop.

        Args:
            interval: Checkpoint interval in seconds
        """
        logger.info(f"✓ Checkpoint loop started (interval={interval}s)")

        while True:
            try:
                await asyncio.sleep(interval)
                await self.checkpoint()
            except asyncio.CancelledError:
                logger.info("Checkpoint loop cancelled")
                break
            except Exception as e:
                logger.error(f"Checkpoint loop error: {e}")

    # ==================== UPSERT OPERATIONS ====================

    async def upsert_candles(self, df: pd.DataFrame) -> int:
        """Upsert candles with deduplication."""
        return await self.insert_bulk("candles", df, mode="replace")

    async def upsert_features(self, df: pd.DataFrame) -> int:
        """Upsert features with deduplication."""
        return await self.insert_bulk("features", df, mode="replace")

    async def upsert_signals(self, df: pd.DataFrame) -> int:
        """Upsert signals with deduplication."""
        return await self.insert_bulk("signals", df, mode="replace")

    async def insert_event(self, event: Dict) -> bool:
        """Insert anomaly event."""
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self._insert_event_sync, event)
            return result
        except Exception as e:
            logger.error(f"Event insert failed: {e}")
            return False

    def _insert_event_sync(self, event: Dict) -> bool:
        """Synchronous event insert."""
        meta_json = json.dumps(event.get("meta", {}))
        self.conn.execute("""
            INSERT INTO anomaly_events (ts_ms, symbol, event_type, severity, confidence, meta)
            VALUES (?, ?, ?, ?, ?, ?::JSON)
            ON CONFLICT (ts_ms, symbol, event_type) DO UPDATE SET
                severity = EXCLUDED.severity,
                confidence = EXCLUDED.confidence,
                meta = EXCLUDED.meta
        """, (
            event["ts_ms"],
            event["symbol"],
            event["event_type"],
            event["severity"],
            event["confidence"],
            meta_json
        ))
        return True

    # ==================== QUERY OPERATIONS ====================

    async def get_symbols(self) -> List[str]:
        """Get all unique symbols."""
        try:
            loop = asyncio.get_running_loop()
            symbols = await loop.run_in_executor(None, self._get_symbols_sync)
            return symbols
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []

    def _get_symbols_sync(self) -> List[str]:
        """Synchronous symbols retrieval."""
        result = self.conn.execute("SELECT DISTINCT symbol FROM candles ORDER BY symbol")
        return [row[0] for row in result.fetchall()]

    async def get_latest_candles(
        self,
        symbol: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """Retrieve latest candles."""
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None, self._get_latest_candles_sync, symbol, limit
            )
            return df
        except Exception as e:
            logger.error(f"Failed to retrieve candles: {e}")
            return pd.DataFrame()

    def _get_latest_candles_sync(self, symbol: str, limit: int) -> pd.DataFrame:
        """Synchronous candle retrieval."""
        query = """
            SELECT * FROM candles
            WHERE symbol = ?
            ORDER BY ts_ms DESC
            LIMIT ?
        """
        result = self.conn.execute(query, (symbol, limit))
        df = result.df()
        return df.iloc[::-1].reset_index(drop=True) if not df.empty else df

    async def get_candles_range(
        self,
        symbol: str,
        start_ts: datetime,
        end_ts: datetime
    ) -> pd.DataFrame:
        """Retrieve candles within timestamp range."""
        try:
            loop = asyncio.get_running_loop()
            df = await loop.run_in_executor(
                None, self._get_candles_range_sync, symbol, start_ts, end_ts
            )
            return df
        except Exception as e:
            logger.error(f"Failed to retrieve candles range: {e}")
            return pd.DataFrame()

    def _get_candles_range_sync(
        self, symbol: str, start_ts: datetime, end_ts: datetime
    ) -> pd.DataFrame:
        """Synchronous range query."""
        query = """
            SELECT * FROM candles
            WHERE symbol = ? AND ts_ms BETWEEN ? AND ?
            ORDER BY ts_ms ASC
        """
        result = self.conn.execute(query, (symbol, start_ts, end_ts))
        return result.df()

    # ==================== CLEANUP & ARCHIVAL ====================

    async def prune_older_than(self, days: int = 90) -> Dict[str, int]:
        """Archive and delete old data."""
        try:
            loop = asyncio.get_running_loop()
            stats = await loop.run_in_executor(None, self._prune_older_than_sync, days)
            logger.info(f"Pruning complete: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return {}

    def _prune_older_than_sync(self, days: int) -> Dict[str, int]:
        """Synchronous pruning."""
        cutoff_ts = datetime.now() - timedelta(days=days)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats = {}

        try:
            for table in ["candles", "features", "anomaly_events", "signals"]:
                query_select = f"SELECT * FROM {table} WHERE ts_ms < ?"
                df = self.conn.execute(query_select, (cutoff_ts,)).df()

                if len(df) > 0:
                    archive_path = f"data/archive/{table}_{timestamp_str}.parquet"
                    df.to_parquet(archive_path, compression="snappy")
                    self.conn.execute(f"DELETE FROM {table} WHERE ts_ms < ?", (cutoff_ts,))
                    stats[table] = len(df)

            return stats
        except Exception as e:
            logger.error(f"Pruning error: {e}")
            return stats

    def close(self):
        """Close database connection."""
        try:
            if self.conn:
                self.conn.close()
                logger.info("✓ Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")


if __name__ == "__main__":
    import asyncio
    import logging

    # Setup logging for visible console output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    async def test_db():
        db = DatabaseManager("data/test.duckdb")

        # Test meta
        await db.set_meta("test_key", "test_value")
        value = await db.get_meta("test_key")
        print(f"Meta: {value}")

        # Test health
        health = await db.get_health_report()
        print(f"Health: {health}")

        # Test counts
        counts = await db.get_table_counts()
        print(f"Counts: {counts}")

        db.close()

    asyncio.run(test_db())
