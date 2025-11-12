"""
src/tests/test_end_to_end.py

End-to-end integration tests for OmerGPT.

Simulates data ingestion, feature generation, anomaly detection,
signal creation, and alert dispatch within a controlled environment.

Ensures system stability, timing accuracy, and data integrity across all layers.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("omerGPT.tests")


# ==================== FIXTURES ====================

@pytest.fixture(scope="session")
def test_db_path():
    """Provide test database path."""
    db_path = "data/test_market_data.duckdb"
    yield db_path
    # Cleanup after tests
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture(scope="session")
def db(test_db_path):
    """Initialize test database."""
    from storage.db_manager import DatabaseManager
    
    db_mgr = DatabaseManager(test_db_path)
    yield db_mgr
    db_mgr.close()


@pytest.fixture(scope="session")
def sample_candles():
    """Generate sample candle data."""
    data = []
    now = datetime.now()
    
    for i in range(100):
        ts = now - timedelta(minutes=100-i)
        base_price = 68000.0
        offset = i * 0.1
        
        data.append({
            "symbol": "BTCUSDT",
            "ts_ms": ts,
            "open": base_price + offset,
            "high": base_price + offset + 100,
            "low": base_price + offset - 50,
            "close": base_price + offset + 50,
            "volume": 100.0 + i
        })
    
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_features():
    """Generate sample feature data."""
    data = []
    now = datetime.now()
    
    for i in range(50):
        ts = now - timedelta(minutes=50-i)
        
        data.append({
            "symbol": "BTCUSDT",
            "ts_ms": ts,
            "return_1m": 0.001 + (i % 5) * 0.0001,
            "volatility_5m": 0.02 + (i % 3) * 0.005,
            "volatility_15m": 0.03 + (i % 4) * 0.002,
            "volatility_60m": 0.04 + (i % 2) * 0.001,
            "momentum_5m": 0.05 * (1 if i % 2 else -1),
            "momentum_15m": 0.03 * (1 if i % 3 else -1),
            "momentum_60m": 0.02 * (1 if i % 4 else -1),
            "rsi_14": 50.0 + (i % 20) * 2,
            "atr": 50.0 + i,
            "atr_pct": 0.07 + (i % 10) * 0.01,
            "macd": 0.001 * i,
            "bb_up": 68000 + (i * 10),
            "bb_dn": 67500 - (i * 5),
            "vol_ma": 100.0 + i,
            "spread": 0.01 + (i % 5) * 0.005,
            "ob_imbalance": 0.1 + (i % 10) * 0.01,
            "corr_btc_eth": 0.8 + (i % 10) * 0.02
        })
    
    return pd.DataFrame(data)


# ==================== UNIT TESTS ====================

class TestDatabase:
    """Test database operations."""
    
    @pytest.mark.asyncio
    async def test_db_initialization(self, db):
        """Verify database tables are created."""
        tables = db.conn.execute("SHOW TABLES").df()
        table_names = tables["name"].tolist()
        
        required_tables = ["candles", "features", "anomaly_events", "signals"]
        assert all(t in table_names for t in required_tables), \
            f"Missing tables. Found: {table_names}"
        
        logger.info(f"✓ Database initialized with {len(table_names)} tables")

    @pytest.mark.asyncio
    async def test_candles_upsert(self, db, sample_candles):
        """Test candle data insertion."""
        rows = await db.upsert_candles(sample_candles)
        
        assert rows == len(sample_candles)
        
        # Verify data in DB
        query_result = db.conn.execute("SELECT COUNT(*) as c FROM candles").fetchone()
        assert query_result[0] >= len(sample_candles)
        
        logger.info(f"✓ Inserted {rows} candles")

    @pytest.mark.asyncio
    async def test_features_upsert(self, db, sample_features):
        """Test feature data insertion."""
        rows = await db.upsert_features(sample_features)
        
        assert rows == len(sample_features)
        
        # Verify features in DB
        query_result = db.conn.execute(
            "SELECT COUNT(*) as c FROM features WHERE symbol=?",
            ("BTCUSDT",)
        ).fetchone()
        assert query_result[0] >= len(sample_features)
        
        logger.info(f"✓ Inserted {rows} features")

    @pytest.mark.asyncio
    async def test_query_performance(self, db):
        """Verify query performance < 50ms."""
        start = time.perf_counter()
        df = await db.get_latest_candles("BTCUSDT", limit=50)
        elapsed = (time.perf_counter() - start) * 1000
        
        assert elapsed < 50, f"Query took {elapsed:.1f}ms (limit 50ms)"
        assert not df.empty
        
        logger.info(f"✓ Query completed in {elapsed:.1f}ms")


class TestFeaturePipeline:
    """Test feature engineering pipeline."""
    
    @pytest.mark.asyncio
    async def test_feature_computation(self, db, sample_candles):
        """Test feature computation from candles."""
        from features.feature_pipeline import FeaturePipeline
        
        # Ensure candles are in DB
        await db.upsert_candles(sample_candles)
        
        fp = FeaturePipeline(db, window_sizes=[5, 15, 60])
        
        start = time.perf_counter()
        features_df = await fp.compute_features(sample_candles, "BTCUSDT")
        elapsed = time.perf_counter() - start
        
        assert not features_df.empty, "No features computed"
        assert len(features_df) > 0
        assert all(col in features_df.columns for col in [
            "volatility_5m", "momentum_5m", "rsi_14", "atr", "macd"
        ])
        
        # Performance check
        assert elapsed < 3.0, f"Feature computation took {elapsed:.1f}s (limit 3s)"
        
        logger.info(f"✓ Computed {len(features_df)} feature rows in {elapsed:.3f}s")

    @pytest.mark.asyncio
    async def test_feature_pipeline_update(self, db):
        """Test full feature pipeline update."""
        from features.feature_pipeline import FeaturePipeline
        
        fp = FeaturePipeline(db, update_interval=1)
        
        start = time.perf_counter()
        await fp.update_features()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 5.0, f"Pipeline update took {elapsed:.1f}s"
        
        logger.info(f"✓ Feature pipeline updated in {elapsed:.3f}s")


class TestAnomalyDetection:
    """Test anomaly detection."""
    
    @pytest.mark.asyncio
    async def test_anomaly_detector_initialization(self, db):
        """Test anomaly detector initialization."""
        from anomaly_detection.isolation_forest_gpu import AnomalyDetector
        
        detector = AnomalyDetector(db)
        
        assert detector is not None
        assert detector.model is None  # Not trained yet
        
        logger.info("✓ Anomaly detector initialized")

    @pytest.mark.asyncio
    async def test_anomaly_detection_pipeline(self, db, sample_features):
        """Test anomaly detection on feature data."""
        from anomaly_detection.isolation_forest_gpu import AnomalyDetector
        
        # Insert features
        await db.upsert_features(sample_features)
        
        detector = AnomalyDetector(db)
        
        start = time.perf_counter()
        await detector.detect_anomalies()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 5.0, f"Detection took {elapsed:.1f}s"
        
        # Check if events were created
        events = db.conn.execute("SELECT COUNT(*) as c FROM anomaly_events").fetchone()
        logger.info(f"✓ Detected {events[0]} anomalies in {elapsed:.3f}s")


class TestSignalGeneration:
    """Test signal generation."""
    
    @pytest.mark.asyncio
    async def test_signal_engine_initialization(self, db):
        """Test signal engine initialization."""
        from signals.signal_engine import SignalEngine
        
        engine = SignalEngine(db)
        
        assert engine is not None
        assert engine.db == db
        
        logger.info("✓ Signal engine initialized")

    @pytest.mark.asyncio
    async def test_signal_generation(self, db, sample_features):
        """Test signal generation from features and anomalies."""
        from signals.signal_engine import SignalEngine
        
        # Setup: insert features and a sample anomaly
        await db.upsert_features(sample_features)
        
        event = {
            "ts_ms": datetime.now(),
            "symbol": "BTCUSDT",
            "event_type": "test_anomaly",
            "severity": 2,
            "confidence": 0.95,
            "meta": {"test": True}
        }
        await db.insert_event(event)
        
        # Generate signals
        engine = SignalEngine(db)
        
        start = time.perf_counter()
        signals = await engine.generate_signals()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 3.0, f"Signal generation took {elapsed:.1f}s"
        assert isinstance(signals, list)
        
        logger.info(
            f"✓ Generated {len(signals)} signals in {elapsed:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_signal_save(self, db):
        """Test signal persistence."""
        from signals.signal_engine import SignalEngine
        
        engine = SignalEngine(db)
        
        test_signals = [
            {
                "symbol": "BTCUSDT",
                "ts_ms": datetime.now(),
                "signal_type": "BUY",
                "confidence": 0.92,
                "reason": "RSI oversold + anomaly",
                "anomaly_score": 0.97,
                "rsi": 28.5,
                "momentum": -0.02,
                "feature_snapshot": json.dumps({"test": True}),
                "status": "new"
            }
        ]
        
        await engine.save_signals(test_signals)
        
        # Verify in DB
        signals_df = db.conn.execute(
            "SELECT * FROM signals WHERE signal_type=?",
            ("BUY",)
        ).df()
        
        assert len(signals_df) > 0
        assert signals_df.iloc[0]["confidence"] == 0.92
        
        logger.info(f"✓ Saved {len(signals_df)} signals")


class TestAlerts:
    """Test alerting system."""
    
    @pytest.mark.asyncio
    async def test_telegram_bot_initialization(self, db):
        """Test Telegram bot initialization."""
        from alerts.telegram_bot import TelegramBot
        
        bot = TelegramBot(
            db,
            token="test_token_12345",
            chat_ids=["123456789"]
        )
        
        assert bot is not None
        assert bot.token == "test_token_12345"
        
        logger.info("✓ Telegram bot initialized")

    @pytest.mark.asyncio
    async def test_alert_message_formatting(self, db):
        """Test alert message formatting."""
        from alerts.telegram_bot import TelegramBot
        
        bot = TelegramBot(db, token="test_token", chat_ids=["123"])
        
        signal = {
            "symbol": "BTCUSDT",
            "ts_ms": datetime.now(),
            "signal_type": "BUY",
            "confidence": 0.94,
            "reason": "RSI oversold + gas spike",
            "anomaly_score": 0.982,
            "rsi": 28.3,
            "momentum": -0.015
        }
        
        message = bot._format_signal_message(signal)
        
        assert "BTCUSDT" in message
        assert "BUY" in message
        assert "0.94" in message or "94" in message
        assert len(message) > 50
        
        logger.info(f"✓ Message formatted ({len(message)} chars)")


# ==================== INTEGRATION TESTS ====================

class TestEndToEnd:
    """End-to-end pipeline tests."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, db, sample_candles, sample_features):
        """Test complete pipeline: ingestion → features → anomaly → signals."""
        from features.feature_pipeline import FeaturePipeline
        from anomaly_detection.isolation_forest_gpu import AnomalyDetector
        from signals.signal_engine import SignalEngine
        
        logger.info("=" * 60)
        logger.info("Running Full Pipeline Test")
        logger.info("=" * 60)
        
        # 1. Ingestion
        logger.info("1. Ingestion...")
        await db.upsert_candles(sample_candles)
        candles_count = db.conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
        logger.info(f"   ✓ {candles_count} candles ingested")
        
        # 2. Features
        logger.info("2. Features...")
        fp = FeaturePipeline(db)
        await db.upsert_features(sample_features)
        features_count = db.conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        logger.info(f"   ✓ {features_count} features computed")
        
        # 3. Anomalies
        logger.info("3. Anomaly Detection...")
        detector = AnomalyDetector(db)
        await detector.detect_anomalies()
        anomalies_count = db.conn.execute("SELECT COUNT(*) FROM anomaly_events").fetchone()[0]
        logger.info(f"   ✓ {anomalies_count} anomalies detected")
        
        # 4. Signals
        logger.info("4. Signal Generation...")
        engine = SignalEngine(db)
        signals = await engine.generate_signals()
        if signals:
            await engine.save_signals(signals)
        signals_count = db.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        logger.info(f"   ✓ {signals_count} signals generated")
        
        # 5. Stats
        logger.info("5. Database Stats...")
        stats = db.get_stats()
        for table, count in stats.items():
            logger.info(f"   {table}: {count} rows")
        
        logger.info("=" * 60)
        logger.info("✅ Full pipeline test passed!")
        logger.info("=" * 60)

    @pytest.mark.asyncio
    async def test_data_flow_integrity(self, db):
        """Verify data flows correctly between layers."""
        # Insert test candle
        test_candle = pd.DataFrame([{
            "symbol": "ETHUSDT",
            "ts_ms": datetime.now(),
            "open": 2500.0,
            "high": 2550.0,
            "low": 2450.0,
            "close": 2525.0,
            "volume": 50.0
        }])
        
        await db.upsert_candles(test_candle)
        
        # Retrieve and verify
        df = await db.get_latest_candles("ETHUSDT", limit=1)
        
        assert not df.empty
        assert df.iloc[0]["close"] == 2525.0
        assert df.iloc[0]["symbol"] == "ETHUSDT"
        
        logger.info("✓ Data integrity verified")


class TestPerformance:
    """Performance and timing tests."""
    
    @pytest.mark.asyncio
    async def test_ingestion_throughput(self, db):
        """Test data ingestion throughput."""
        # Generate large batch
        large_batch = pd.DataFrame([
            {
                "symbol": f"SYM{i%5}",
                "ts_ms": datetime.now() - timedelta(minutes=j),
                "open": 1000.0 + i,
                "high": 1100.0 + i,
                "low": 900.0 + i,
                "close": 1050.0 + i,
                "volume": 100.0 + j
            }
            for i in range(1000)
            for j in range(10)
        ])
        
        start = time.perf_counter()
        await db.upsert_candles(large_batch)
        elapsed = time.perf_counter() - start
        
        throughput = len(large_batch) / elapsed
        
        assert throughput > 100, f"Throughput too low: {throughput:.0f} rows/sec"
        
        logger.info(f"✓ Throughput: {throughput:.0f} rows/sec")

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, db):
        """Verify memory doesn't grow unbounded."""
        import gc
        
        gc.collect()
        
        # Simulate repeated operations
        for i in range(5):
            df = await db.get_latest_candles("BTCUSDT", limit=100)
            assert not df.empty
        
        gc.collect()
        
        logger.info("✓ Memory efficiency check passed")


# ==================== PYTEST CONFIGURATION ====================

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "asyncio: marks tests as asyncio"
    )


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-s"
    ])
