# =====================================================
# ✅ RESEARCH COMPLIANCE WRAPPER §2.3–§3.1 (GPU Isolation Forest)
# =====================================================
import time
import yaml
import statistics
import logging

# Load config for batch/contamination control
try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {}

GPU_ENABLED = CONFIG.get("gpu", {}).get("enabled", True)
BATCH_INTERVAL = CONFIG.get("features", {}).get("batch_interval", 5)
CONTAMINATION = CONFIG.get("anomaly_detection", {}).get("model_params", {}).get("contamination", 0.01)
LATENCY_TARGET_MS = 100

logger = logging.getLogger("IForestResearchLayer")
logger.info(
    f"[Init] GPU={GPU_ENABLED} | Batch={BATCH_INTERVAL}s | Contamination={CONTAMINATION} | p95 target={LATENCY_TARGET_MS}ms"
)
# =====================================================

"""
src/anomaly_detection/isolation_forest_gpu.py

GPU-accelerated anomaly detection module using Isolation Forest.

Loads recent feature data, fits an isolation model (via cuML or scikit-learn),
computes anomaly scores for each symbol, and writes results to DuckDB.

Designed for streaming mode: incremental updates, low latency, and adaptive retraining.
"""

import asyncio
import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest as SKIsolationForest

try:
    import cupy as cp
    import cudf
    from cuml.ensemble import IsolationForest as CUMLIsolationForest
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger("omerGPT.anomaly_detection.isolation_forest")


class AnomalyDetector:
    """
    GPU-accelerated anomaly detection using Isolation Forest.
    
    Features:
    - Supports GPU (cuML) and CPU (scikit-learn) backends
    - Streaming mode with incremental retraining
    - Model checkpointing
    - Critical anomaly alerts
    - Feature normalization and validation
    """
    
    CHECKPOINT_DIR = "checkpoints"
    MODEL_FILE = "isolation_forest.pkl"
    SCALER_FILE = "feature_scaler.pkl"
    
    def __init__(
        self,
        db_manager,
        model_params: Optional[Dict] = None,
        gpu_enabled: bool = False,
        retrain_interval: int = 3600
    ):
        """
        Initialize anomaly detector.
        
        Args:
            db_manager: DatabaseManager instance
            model_params: Model hyperparameters
            gpu_enabled: Force GPU usage (if available)
            retrain_interval: Time in seconds between retraining
        """
        self.db = db_manager
        self.gpu_enabled = gpu_enabled and GPU_AVAILABLE
        self.retrain_interval = retrain_interval
        
        # Default model parameters
        self.model_params = model_params or {
            "n_estimators": 200,
            "max_samples": "auto",
            "contamination": 0.01,
            "random_state": 42
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.last_retrain = None
        self.running = False
        
        # Statistics
        self.stats = {
            "batches_processed": 0,
            "anomalies_detected": 0,
            "critical_anomalies": 0,
            "retrains": 0,
            "errors": 0
        }
        
        # Create checkpoint directory
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        
        # Load existing model if available
        self._load_checkpoint()
        
        logger.info(
            f"AnomalyDetector initialized: "
            f"gpu={self.gpu_enabled}, "
            f"contamination={self.model_params.get('contamination', 0.02)}"
        )

    def _get_model_class(self):
        """Return appropriate model class based on GPU availability."""
        if self.gpu_enabled:
            return CUMLIsolationForest
        else:
            return SKIsolationForest

    def _check_gpu(self) -> bool:
        """Verify GPU availability."""
        try:
            if not GPU_AVAILABLE:
                return False
            arr = cp.array([1.0, 2.0])
            cp.sum(arr)
            logger.info("✅ GPU (cuML) available")
            return True
        except Exception as e:
            logger.warning(f"⚠️ GPU check failed: {e}")
            return False

    def _load_checkpoint(self):
        """Load model and scaler from checkpoint if available."""
        try:
            model_path = os.path.join(self.CHECKPOINT_DIR, self.MODEL_FILE)
            scaler_path = os.path.join(self.CHECKPOINT_DIR, self.SCALER_FILE)
            
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"✅ Model checkpoint loaded: {model_path}")
            
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info(f"✅ Scaler checkpoint loaded: {scaler_path}")
        
        except Exception as e:
            logger.warning(f"⚠️ Checkpoint loading failed: {e}")

    def _save_checkpoint(self):
        """Save model and scaler to checkpoint."""
        try:
            if self.model is None:
                return
            
            model_path = os.path.join(self.CHECKPOINT_DIR, self.MODEL_FILE)
            scaler_path = os.path.join(self.CHECKPOINT_DIR, self.SCALER_FILE)
            
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"💾 Checkpoint saved: {model_path}")
        
        except Exception as e:
            logger.error(f"❌ Checkpoint save failed: {e}")

    async def load_recent_features(self, window_minutes: int = 120) -> pd.DataFrame:
        """
        Load recent feature data from DuckDB.
        
        Args:
            window_minutes: Historical window in minutes
            
        Returns:
            DataFrame with features and metadata
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=window_minutes)
            
            # Query features from DB
            query = """
                SELECT symbol, ts_ms, 
                       return_1m, volatility_5m, volatility_15m, volatility_60m,
                       momentum_5m, momentum_15m, momentum_60m,
                       rsi_14, atr, atr_pct, macd, bb_up, bb_dn,
                       vol_ma, spread, ob_imbalance, corr_btc_eth
                FROM features
                WHERE ts_ms BETWEEN ? AND ?
                ORDER BY ts_ms ASC
            """
            
            result = self.db.conn.execute(
                query,
                (start_time, end_time)
            )
            df = result.df()
            
            if len(df) == 0:
                logger.warning(f"No features found in last {window_minutes}m")
                return pd.DataFrame()
            
            logger.info(f"Loaded {len(df)} feature rows")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load features: {e}", exc_info=True)
            return pd.DataFrame()

    async def fit_model(self, df: pd.DataFrame):
        """
        Fit or update the Isolation Forest model.
        
        Args:
            df: Feature DataFrame (numeric columns)
        """
        try:
            if len(df) < 32:
                logger.warning(f"Insufficient data for training: {len(df)} rows")
                return
            
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove potentially constant or problematic features
            numeric_cols = [c for c in numeric_cols if c not in ["ts_ms"]]
            
            if not numeric_cols:
                logger.error("No numeric features found")
                return
            
            self.feature_columns = numeric_cols
            
            # Prepare data
            X = df[numeric_cols].fillna(0)
            
            # Normalize
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit model
            model_class = self._get_model_class()
            self.model = model_class(**self.model_params)
            self.model.fit(X_scaled)
            
            self.last_retrain = datetime.now()
            self.stats["retrains"] += 1
            
            logger.info(
                f"✅ Fitted IsolationForest on {len(df)} samples, "
                f"{len(numeric_cols)} features"
            )
            
            # Save checkpoint
            self._save_checkpoint()
        
        except Exception as e:
            logger.error(f"Model fitting failed: {e}", exc_info=True)
            self.stats["errors"] += 1

    async def compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute anomaly scores for each row.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            DataFrame with anomaly scores and flags
        """
        try:
            if self.model is None:
                logger.warning("No fitted model available")
                return pd.DataFrame()
            
            if self.feature_columns is None:
                logger.error("Feature columns not defined")
                return pd.DataFrame()
            
            # Prepare data
            X = df[self.feature_columns].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Compute scores
            if self.gpu_enabled:
                # cuML returns scores as GPU arrays
                raw_scores = self.model.decision_function(X_scaled)
                scores = cp.asnumpy(raw_scores)
            else:
                scores = self.model.decision_function(X_scaled)
            
            # Normalize scores to [0, 1]
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            
            # Compute threshold (98th percentile)
            threshold = np.percentile(scores_normalized, 98)
            
            # Detect anomalies
            is_anomaly = scores_normalized > threshold
            
            # Create result DataFrame
            result = pd.DataFrame({
                "symbol": df["symbol"],
                "ts_ms": df["ts_ms"],
                "anomaly_score": scores_normalized,
                "threshold": threshold,
                "is_anomaly": is_anomaly
            })
            
            return result
        
        except Exception as e:
            logger.error(f"Score computation failed: {e}", exc_info=True)
            self.stats["errors"] += 1
            return pd.DataFrame()

    async def _process_anomalies(self, scores_df: pd.DataFrame):
        """
        Process detected anomalies and store in database.
        
        Args:
            scores_df: DataFrame with anomaly scores
        """
        try:
            anomalies = scores_df[scores_df["is_anomaly"]].copy()
            
            if len(anomalies) == 0:
                return
            
            self.stats["anomalies_detected"] += len(anomalies)
            
            # Detect critical anomalies (score > 0.95)
            critical = anomalies[anomalies["anomaly_score"] > 0.95]
            if len(critical) > 0:
                self.stats["critical_anomalies"] += len(critical)
                for _, row in critical.iterrows():
                    logger.warning(
                        f"🚨 CRITICAL anomaly for {row['symbol']} @ {row['ts_ms']} | "
                        f"score={row['anomaly_score']:.4f}"
                    )
            
            # Store in database
            events_list = []
            for _, row in anomalies.iterrows():
                event = {
                    "ts_ms": row["ts_ms"],
                    "symbol": row["symbol"],
                    "event_type": "isolation_forest",
                    "severity": 3 if row["anomaly_score"] > 0.95 else 2,
                    "confidence": float(row["anomaly_score"]),
                    "meta": json.dumps({
                        "score": float(row["anomaly_score"]),
                        "threshold": float(row["threshold"])
                    })
                }
                events_list.append(event)
            
            # Batch insert
            for event in events_list:
                await self.db.insert_event(event)
            
            logger.info(f"Stored {len(anomalies)} anomaly events")
        
        except Exception as e:
            logger.error(f"Anomaly processing failed: {e}", exc_info=True)
            self.stats["errors"] += 1

    async def detect_anomalies(self):
        """
        Main pipeline: load → fit → score → store results.
        """
        try:
            # Load features
            features_df = await self.load_recent_features(window_minutes: 10080)
            
            if features_df.empty:
                logger.warning("No features to process")
                return
            
            # Fit model if needed
            should_retrain = (
                self.model is None or 
                (self.last_retrain and 
                 (datetime.now() - self.last_retrain).total_seconds() > self.retrain_interval)
            )
            
            if should_retrain:
                await self.fit_model(features_df)
            
            # Compute scores
            scores_df = await self.compute_scores(features_df)
            
            if scores_df.empty:
                return
            
            # Process anomalies
            await self._process_anomalies(scores_df)
            
            self.stats["batches_processed"] += 1
            logger.info(f"Anomaly detection batch complete ({len(scores_df)} rows)")
        
        except Exception as e:
            logger.error(f"Detection pipeline failed: {e}", exc_info=True)
            self.stats["errors"] += 1

    async def run(self):
        """Main detection loop."""
        self.running = True
        logger.info("✅ Anomaly detector started")
        
        while self.running:
            try:
                await self.detect_anomalies()
                await asyncio.sleep(60)  # Run every 60 seconds
            
            except asyncio.CancelledError:
                logger.info("Detector cancelled")
                break
            
            except Exception as e:
                logger.error(f"Detector error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def stop(self):
        """Stop detector."""
        logger.info("Stopping anomaly detector...")
        self.running = False

    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return self.stats.copy()


# ==================== DEMO ====================

async def run_demo():
    """
    Demo: Run anomaly detection for 5 minutes.
    """
    import sys
    sys.path.insert(0, "src")
    
    from storage.db_manager import DatabaseManager
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    print("\n=== Anomaly Detection Demo ===\n")
    
    # Initialize database
    db = DatabaseManager("data/market_data.duckdb")
    
    # Create detector
    detector = AnomalyDetector(
        db_manager=db,
        gpu_enabled=GPU_AVAILABLE,
        model_params={
            "n_estimators": 200,
            "contamination": 0.01
        }
    )
    
    # Start detector
    detector_task = asyncio.create_task(detector.run())
    
    try:
        print("Running anomaly detection for 5 minutes...\n")
        
        # Monitor for 5 minutes
        for i in range(5):
            await asyncio.sleep(60)
            stats = detector.get_stats()
            print(f"[{(i+1)*60}s] Stats: {stats}\n")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        await detector.stop()
        detector_task.cancel()
        
        final_stats = detector.get_stats()
        print(f"\n=== Final Stats ===")
        print(f"Batches: {final_stats['batches_processed']}")
        print(f"Anomalies: {final_stats['anomalies_detected']}")
        print(f"Critical: {final_stats['critical_anomalies']}")
        print(f"Retrains: {final_stats['retrains']}")
        print(f"Errors: {final_stats['errors']}")
        
        db.close()
        print("\n=== Demo Complete ===\n")


if __name__ == "__main__":
    asyncio.run(run_demo())


    features = gdf[["spread", "mid_price", "obi", "depth_ratio"]]
    model.fit(features)

    gdf["score"] = model.decision_function(features)
    gdf["is_anomaly"] = model.predict(features)

    anomalies = gdf[gdf["is_anomaly"] == -1]

    for row in anomalies.to_pandas().itertuples(index=False):
        db_conn.execute("""
            INSERT INTO anomaly_events (timestamp, symbol, score, is_anomaly)
            VALUES (?, ?, ?, ?)
        """, (row.timestamp, row.symbol, float(row.score), int(row.is_anomaly)))
# =====================================================
# ✅ GPU CONTINUOUS DETECTION LOOP (Research §2.3–§3.1)
# =====================================================
if __name__ == "__main__":
    import asyncio
    import sys
    sys.path.insert(0, "src")
    from storage.db_manager import DatabaseManager

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("🚀 Starting Continuous GPU Isolation Forest Loop")

    db = DatabaseManager("data/market_data.duckdb")
    detector = AnomalyDetector(
        db_manager=db,
        gpu_enabled=GPU_ENABLED,
        model_params={
            "n_estimators": 100,
            "max_samples": 1.0,
            "contamination": CONTAMINATION,
            "random_state": 42,
        },
    )

    latency_log = []
    iteration = 0

    async def continuous_loop():
        global iteration
        while True:
            t0 = time.perf_counter()
            try:
                await detector.detect_anomalies()
                stats = detector.get_stats()
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                stats = {"batches_processed": 0, "anomalies_detected": 0, "critical_anomalies": 0}

            elapsed_ms = (time.perf_counter() - t0) * 1000
            latency_log.append(elapsed_ms)
            latency_log = latency_log[-200:]
            p95 = statistics.quantiles(latency_log, n=100)[94] if len(latency_log) >= 10 else elapsed_ms

            logger.info(
                f"[{iteration:05d}] Batches={stats['batches_processed']} | Anomalies={stats['anomalies_detected']} | "
                f"Critical={stats['critical_anomalies']} | p95={p95:.1f}ms | GPU={GPU_ENABLED}"
            )

            if p95 > LATENCY_TARGET_MS:
                logger.warning(f"⚠️ p95 latency {p95:.1f}ms exceeds {LATENCY_TARGET_MS}ms target!")

            iteration += 1
            await asyncio.sleep(BATCH_INTERVAL)

    try:
        asyncio.run(continuous_loop())
    except KeyboardInterrupt:
        logger.info("🛑 Graceful shutdown triggered by user")

# === AUTO PATCH START ===
# Auto-added: enforce contamination=0.01 & GPU usage
if hasattr(self, 'contamination'):
    self.contamination = 0.01
if hasattr(self, 'gpu_enabled'):
    self.gpu_enabled = True
# === AUTO PATCH END ===

