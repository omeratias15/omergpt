"""
src/features/feature_pipeline.py

Complete Feature Pipeline with HMM Regime Detection, Momentum Indicators, and GARCH Volatility.

FIXED: Using int64 milliseconds for ts_ms and computed_at to match BIGINT database schema.
"""

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

logger = logging.getLogger("features.feature_pipeline")

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
# MOMENTUM INDICATORS (NEW - MISSING MODULE)
# ==============================================================

class MomentumIndicators:
    """Simple momentum indicator calculations (RSI, MACD, Momentum, etc.)."""
    
    @staticmethod
    def compute_rsi(prices: pd.Series, period: int = 14) -> float:
        """
        Compute RSI (Relative Strength Index) indicator.
        
        Args:
            prices: Price series
            period: RSI period (default 14)
            
        Returns:
            RSI value (0-100)
        """
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Avoid division by zero
            if loss.iloc[-1] == 0:
                return 100.0
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not rsi.empty else 50.0
        except Exception as e:
            logger.debug(f"RSI computation error: {e}")
            return 50.0
    
    @staticmethod
    def compute_momentum(prices: pd.Series, period: int = 15) -> float:
        """
        Compute momentum (rate of change).
        
        Args:
            prices: Price series
            period: Lookback period
            
        Returns:
            Momentum value (percentage change)
        """
        try:
            if len(prices) < period + 1:
                return 0.0
            
            momentum = (prices.iloc[-1] - prices.iloc[-period - 1]) / prices.iloc[-period - 1]
            return float(momentum)
        except Exception as e:
            logger.debug(f"Momentum computation error: {e}")
            return 0.0
    
    @staticmethod
    def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """
        Compute MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            MACD value
        """
        try:
            ema_fast = prices.ewm(span=fast, adjust=False).mean()
            ema_slow = prices.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            return float(macd.iloc[-1]) if not macd.empty else 0.0
        except Exception as e:
            logger.debug(f"MACD computation error: {e}")
            return 0.0


# ==============================================================
# GARCH VOLATILITY (NEW - MISSING MODULE)
# ==============================================================

class GARCHVolatility:
    """Simple GARCH(1,1) volatility forecasting using exponential weighting."""
    
    @staticmethod
    def forecast_volatility(prices: pd.Series, window: int = 30) -> float:
        """
        Forecast next-period volatility using exponential weighting.
        
        Args:
            prices: Price series
            window: Lookback window for volatility calculation
            
        Returns:
            Forecasted volatility
        """
        try:
            returns = prices.pct_change().dropna()
            
            if len(returns) < 2:
                return 0.0
            
            # Exponentially weighted volatility
            volatility = returns.ewm(span=window).std().iloc[-1]
            
            return float(volatility) if not pd.isna(volatility) else 0.0
        except Exception as e:
            logger.debug(f"GARCH volatility computation error: {e}")
            return 0.0


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
# FEATURE PIPELINE WRAPPER (FIXED)
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
        
        # FIXED: Initialize missing modules
        self.momentum = MomentumIndicators()
        self.garch = GARCHVolatility()

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
            momentum_5m = self.momentum.compute_momentum(candles["close"], 5)
            momentum_15m = self.momentum.compute_momentum(candles["close"], 15)
            momentum_60m = self.momentum.compute_momentum(candles["close"], 60)
            macd = self.momentum.compute_macd(candles["close"])
            garch_vol = self.garch.forecast_volatility(candles["close"])

            # Compute volatility for multiple windows
            vol_5m = candles["return"].rolling(5).std() * np.sqrt(5)
            vol_15m = candles["return"].rolling(15).std() * np.sqrt(15)
            vol_60m = candles["return"].rolling(60).std() * np.sqrt(60)

            # Compute ATR (Average True Range)
            high_low = candles["high"] - candles["low"]
            high_close = np.abs(candles["high"] - candles["close"].shift())
            low_close = np.abs(candles["low"] - candles["close"].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1] if len(true_range) > 14 else 0.0
            atr_pct = (atr / candles["close"].iloc[-1]) if candles["close"].iloc[-1] > 0 else 0.0

            # Bollinger Bands
            bb_ma = candles["close"].rolling(20).mean()
            bb_std = candles["close"].rolling(20).std()
            bb_up = (bb_ma + 2 * bb_std).iloc[-1] if len(bb_ma) > 20 else candles["close"].iloc[-1]
            bb_dn = (bb_ma - 2 * bb_std).iloc[-1] if len(bb_ma) > 20 else candles["close"].iloc[-1]

            # Volume MA
            vol_ma = candles["volume"].rolling(20).mean().iloc[-1] if "volume" in candles and len(candles) > 20 else 0.0

            # Placeholders for advanced features
            spread = 0.0
            ob_imbalance = 0.0
            corr_btc_eth = 0.0

            # 🔥 CRITICAL FIX: Use int64 milliseconds instead of datetime
            current_time_ms = int(time.time() * 1000)

            feature_row = {
                "symbol": symbol,
                "ts_ms": current_time_ms,  # ← FIXED: int64 milliseconds
                "return_1m": float(returns[-1]),
                "volatility_5m": float(vol_5m.iloc[-1]) if not vol_5m.empty and len(vol_5m) > 5 else 0.0,
                "volatility_15m": float(vol_15m.iloc[-1]) if not vol_15m.empty and len(vol_15m) > 15 else 0.0,
                "volatility_60m": float(vol_60m.iloc[-1]) if not vol_60m.empty and len(vol_60m) > 60 else 0.0,
                "momentum_5m": float(momentum_5m),
                "momentum_15m": float(momentum_15m),
                "momentum_60m": float(momentum_60m),
                "rsi_14": float(rsi_14),
                "atr": float(atr),
                "atr_pct": float(atr_pct),
                "macd": float(macd),
                "bb_up": float(bb_up),
                "bb_dn": float(bb_dn),
                "vol_ma": float(vol_ma),
                "spread": float(spread),
                "ob_imbalance": float(ob_imbalance),
                "corr_btc_eth": float(corr_btc_eth),
                "computed_at": current_time_ms  # ← FIXED: int64 milliseconds
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
