import logging
from typing import Tuple, List
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import os
import time
import yaml
import asyncio
from pathlib import Path

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

# Import dependent feature modules
from features.correlation import GPUCorrelationEngine
from sentiment_analysis.sentiment_index import SentimentIndexGenerator


logger = logging.getLogger(__name__)

# === Load configuration and runtime params ===
try:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {}

BATCH_INTERVAL = CONFIG.get("features", {}).get("batch_interval", 5)
LATENCY_TARGET_MS = 100


# ==============================================================
# HMM REGIME DETECTOR
# ==============================================================

class HMMRegimeDetector:
    """3-state Gaussian HMM for market regime detection (calm / volatile / extreme)."""

    def __init__(self, config: dict):
        self.config = config
        self.gpu_enabled = config.get("gpu", {}).get("enabled", False) and GPU_AVAILABLE
        self.n_states = config.get("anomaly_detection", {}).get("hmm_states", 3)
        self.training_window = config.get("anomaly_detection", {}).get("hmm_training_window", 30)
        self.model_path = config.get("anomaly_detection", {}).get("hmm_model_path", "checkpoints/hmm_regime.pkl")

        self.model = None
        self.scaler = StandardScaler()
        self.state_labels = {0: "calm", 1: "volatile", 2: "extreme"}
        self.last_train_time = None

        os.makedirs(os.path.dirname(self.model_path) or "checkpoints", exist_ok=True)
        logger.info(f"HMMRegimeDetector initialized | states={self.n_states} | GPU={self.gpu_enabled}")

    def requires_retraining(self) -> bool:
        """Check whether model retraining is needed based on time or missing model."""
        if self.model is None or self.last_train_time is None:
            return True
        elapsed = (time.time() - self.last_train_time) / 60.0
        return elapsed > self.training_window

    def train(self, returns: np.ndarray, volatility: np.ndarray):
        """Train Gaussian HMM on returns and volatility."""
        try:
            X = np.column_stack((returns, volatility))
            X_scaled = self.scaler.fit_transform(X)

            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=200,
                random_state=42
            )
            self.model.fit(X_scaled)
            self.last_train_time = time.time()

            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)

            logger.info(f"✅ HMM model trained and saved to {self.model_path}")
        except Exception as e:
            logger.error(f"HMM training failed: {e}", exc_info=True)

    def predict(self, returns: np.ndarray, volatility: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Predict regime and return (state_index, confidence, probabilities)."""
        if self.model is None:
            logger.warning("No trained HMM model — returning default calm state.")
            return 0, 0.0, np.array([1.0, 0.0, 0.0])

        X = np.column_stack((returns, volatility))
        X_scaled = self.scaler.transform(X)

        states = self.model.predict(X_scaled)
        last_state = states[-1]
        try:
            probs = self.model.predict_proba(X_scaled)[-1]
        except Exception:
            counts = np.bincount(states, minlength=self.n_states)
            probs = counts / counts.sum()

        confidence = float(probs[last_state])
        return int(last_state), confidence, probs

    def get_regime_label(self, state: int) -> str:
        return self.state_labels.get(state, "unknown")


# ==============================================================
# FEATURE PIPELINE WRAPPER
# ==============================================================

class FeaturePipeline:
    """Wrapper class for feature generation and HMM regime detection."""

    def __init__(self, db_manager, window_sizes: List[int] = None, update_interval=60, gpu_enabled=False):
        self.db = db_manager
        self.window_sizes = window_sizes or [5, 15, 60]
        self.update_interval = update_interval
        self.gpu_enabled = gpu_enabled
        self.xp = cp if (gpu_enabled and GPU_AVAILABLE) else np

        self.correlation_engine = GPUCorrelationEngine(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        self.sentiment_index = SentimentIndexGenerator("data/sentiment_data.duckdb")

        self.detector = HMMRegimeDetector(CONFIG)

        logger.info(f"✅ FeaturePipeline initialized | GPU={gpu_enabled}")

    async def update_features(self):
        """Compute features and update market regime label."""
        start_time = time.perf_counter()

        symbols = await self.db.get_symbols()
        if not symbols:
            logger.warning("No symbols found in database.")
            return

        all_features = []

        for symbol in symbols:
            candles = await self.db.get_latest_candles(symbol, limit=500)
            if candles.empty:
                continue

            candles["return"] = candles["close"].pct_change()
            candles["volatility"] = candles["return"].rolling(30).std() * np.sqrt(30)

            returns = candles["return"].dropna().values[-200:]
            vol = candles["volatility"].dropna().values[-200:]

            if len(returns) < 20:
                continue

            if self.detector.requires_retraining():
                logger.info(f"🔁 Retraining HMM model for {symbol}")
                self.detector.train(returns, vol)

            state, confidence, probs = self.detector.predict(returns, vol)
            label = self.detector.get_regime_label(state)

            # Compute indicators
            rsi_14 = self.momentum.compute_rsi(candles["close"], 14)
            momentum_15m = self.momentum.compute_momentum(candles["close"], 15)
            garch_vol = self.garch.forecast_volatility(candles["close"])

            feature_row = {
                "symbol": symbol,
                "ts_ms": datetime.utcnow(),
                "return_1m": returns[-1],
                "volatility_5m": vol[-1],
                "rsi_14": rsi_14,
                "momentum_15m": momentum_15m,
                "garch_volatility": garch_vol,
                "regime_state": label,
                "regime_confidence": confidence
            }
            all_features.append(feature_row)

        if all_features:
            df = pd.DataFrame(all_features)
            await self.db.upsert_features(df)

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"📊 Feature update complete | {len(all_features)} symbols | {elapsed:.1f} ms")

    async def run_pipeline(self):
        """Continuous feature update loop."""
        logger.info(f"🚀 FeaturePipeline started (interval={self.update_interval}s)")
        try:
            while True:
                await self.update_features()
                await asyncio.sleep(self.update_interval)
        except asyncio.CancelledError:
            logger.info("🧹 FeaturePipeline cancelled gracefully.")
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)

    async def stop(self):
        logger.info("🧹 FeaturePipeline stopped.")


# ==============================================================
# Optional Standalone Debug Loop
# ==============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from storage.db_manager import DatabaseManager

    async def main():
        db = DatabaseManager("data/market_data.duckdb")
        pipeline = FeaturePipeline(db, gpu_enabled=False, update_interval=10)
        await pipeline.update_features()
        db.close()

    asyncio.run(main())
