# =====================================================
# ✅ RESEARCH-COMPLIANT EXECUTION LAYER (v2.3–v3.1)
# =====================================================
import time
import yaml
import statistics
from datetime import datetime

# --- Load config for dynamic batch + contamination ---
try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {}

BATCH_INTERVAL = CONFIG.get("features", {}).get("batch_interval", 5)
CONTAMINATION = CONFIG.get("anomaly_detection", {}).get("model_params", {}).get("contamination", 0.01)
LATENCY_TARGET_MS = 100

import logging
logger = logging.getLogger(__name__)
logger.info(f"[Init] GPU Batch={BATCH_INTERVAL}s | Contamination={CONTAMINATION} | Target p95={LATENCY_TARGET_MS}ms")
# =====================================================
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

try:
    from cuml.ensemble import IsolationForest as cuIsolationForest
    from cuml.cluster import DBSCAN as cuDBSCAN
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    GPU_AVAILABLE = False

from .hmm_regime import HMMRegimeDetector

logger = logging.getLogger(__name__)


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining:
    - Isolation Forest (density-based outliers)
    - DBSCAN (cluster-based outliers)
    - HMM Regime Detection (state transitions)
    
    Produces unified anomaly score 0-1.
    """
    
    def __init__(self, config: dict):
        """
        Initialize ensemble detector.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.gpu_enabled = config.get('gpu', {}).get('enabled', False) and GPU_AVAILABLE
        
        # Isolation Forest params
        self.contamination = config.get('anomaly_detection', {}).get('contamination', 0.01)
        self.n_estimators = config.get('anomaly_detection', {}).get('n_estimators', 100)
        
        # DBSCAN params
        self.dbscan_eps = config.get('anomaly_detection', {}).get('dbscan_eps', 0.5)
        self.dbscan_min_samples = config.get('anomaly_detection', {}).get('dbscan_min_samples', 10)
        
        # Ensemble weights
        self.weights = config.get('anomaly_detection', {}).get('ensemble_weights', {
            'isolation_forest': 0.4,
            'dbscan': 0.3,
            'hmm': 0.3
        })
        
        # Initialize models
        self.isolation_forest = None
        self.hmm_detector = HMMRegimeDetector(config)
        
        self.anomaly_history = []
        
        logger.info(f"EnsembleAnomalyDetector initialized. GPU: {self.gpu_enabled}")
    
    def train(self, features_df: pd.DataFrame) -> None:
        """
        Train anomaly detection models.
        
        Args:
            features_df: Feature DataFrame
        """
        logger.info(f"Training ensemble detector on {len(features_df)} samples")
        
        feature_cols = [c for c in features_df.columns if c not in ['timestamp', 'symbol']]
        X = features_df[feature_cols].fillna(0).values
        
        if len(X) < 100:
            logger.warning(f"Insufficient data for training: {len(X)} samples")
            return
        
        # Train Isolation Forest
        if self.gpu_enabled:
            X_gpu = cudf.DataFrame(X)
            self.isolation_forest = cuIsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42
            )
        else:
            self.isolation_forest = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
        
        self.isolation_forest.fit(X_gpu if self.gpu_enabled else X)
        logger.info("Isolation Forest trained")
        
        # Train HMM on returns and volatility
        if 'return_1' in feature_cols and 'volatility_std_20' in feature_cols:
            returns = features_df['return_1'].fillna(0).values
            volatility = features_df['volatility_std_20'].fillna(0).values
            self.hmm_detector.train(returns, volatility)
        else:
            logger.warning("Missing return_1 or volatility_std_20 for HMM training")
    
    def predict(self, features_df: pd.DataFrame) -> Dict:
        """
        Predict anomalies using ensemble of detectors.
        
        Args:
            features_df: Feature DataFrame
            
        Returns:
            Dictionary with anomaly scores and predictions
        """
        if self.isolation_forest is None:
            logger.warning("Models not trained")
            return {'anomaly_detected': False, 'score': 0.0}
        
        feature_cols = [c for c in features_df.columns if c not in ['timestamp', 'symbol']]
        X = features_df[feature_cols].fillna(0).values
        
        if len(X) == 0:
            return {'anomaly_detected': False, 'score': 0.0}
        
        # Component scores
        if_score = self._isolation_forest_score(X)
        dbscan_score = self._dbscan_score(X)
        hmm_score = self._hmm_score(features_df)
        
        # Weighted ensemble score
        ensemble_score = (
            self.weights['isolation_forest'] * if_score +
            self.weights['dbscan'] * dbscan_score +
            self.weights['hmm'] * hmm_score
        )
        
        # Anomaly threshold
        anomaly_threshold = self.config.get('anomaly_detection', {}).get('threshold', 0.7)
        anomaly_detected = ensemble_score > anomaly_threshold
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'anomaly_detected': bool(anomaly_detected),
            'ensemble_score': float(ensemble_score),
            'component_scores': {
                'isolation_forest': float(if_score),
                'dbscan': float(dbscan_score),
                'hmm': float(hmm_score)
            },
            'threshold': anomaly_threshold
        }
        
        if anomaly_detected:
            logger.info(f"Anomaly detected: score={ensemble_score:.3f}")
        
        self.anomaly_history.append(result)
        
        return result
    
    def _isolation_forest_score(self, X: np.ndarray) -> float:
        """
        Compute normalized anomaly score from Isolation Forest.
        
        Returns:
            Score in [0, 1] where 1 = strong anomaly
        """
        try:
            if self.gpu_enabled:
                X_gpu = cudf.DataFrame(X)
                scores = self.isolation_forest.decision_function(X_gpu)
                if hasattr(scores, 'values'):
                    scores = scores.values.get()
            else:
                scores = self.isolation_forest.decision_function(X)
            
            # decision_function returns negative values for anomalies
            # Normalize to [0, 1]
            normalized = 1 / (1 + np.exp(scores[-1]))  # Last sample
            
            return float(normalized)
            
        except Exception as e:
            logger.error(f"Isolation Forest scoring failed: {e}")
            return 0.0
    
    def _dbscan_score(self, X: np.ndarray) -> float:
        """
        Compute anomaly score from DBSCAN clustering.
        Noise points (label=-1) are anomalies.
        
        Returns:
            Score in [0, 1]
        """
        try:
            if len(X) < self.dbscan_min_samples:
                return 0.0
            
            if self.gpu_enabled:
                X_gpu = cudf.DataFrame(X)
                dbscan = cuDBSCAN(
                    eps=self.dbscan_eps,
                    min_samples=self.dbscan_min_samples
                )
                labels = dbscan.fit_predict(X_gpu)
                if hasattr(labels, 'values'):
                    labels = labels.values.get()
            else:
                dbscan = DBSCAN(
                    eps=self.dbscan_eps,
                    min_samples=self.dbscan_min_samples,
                    n_jobs=-1
                )
                labels = dbscan.fit_predict(X)
            
            # Last sample is anomaly if labeled as noise (-1)
            last_label = labels[-1]
            is_noise = last_label == -1
            
            # Distance to nearest cluster center (if not noise)
            if not is_noise:
                # Calculate distance to cluster center
                cluster_mask = labels == last_label
                cluster_samples = X[cluster_mask]
                cluster_center = cluster_samples.mean(axis=0)
                
                distance = np.linalg.norm(X[-1] - cluster_center)
                cluster_std = cluster_samples.std(axis=0).mean()
                
                normalized_distance = distance / (cluster_std + 1e-8)
                score = min(1.0, normalized_distance / 3.0)  # 3-sigma threshold
            else:
                score = 1.0  # Noise point
            
            return float(score)
            
        except Exception as e:
            logger.error(f"DBSCAN scoring failed: {e}")
            return 0.0
    
    def _hmm_score(self, features_df: pd.DataFrame) -> float:
        """
        Compute anomaly score from HMM regime detection.
        Extreme regime = high score.
        
        Returns:
            Score in [0, 1]
        """
        try:
            if 'return_1' not in features_df.columns or 'volatility_std_20' not in features_df.columns:
                return 0.0
            
            returns = features_df['return_1'].fillna(0).values[-20:]
            volatility = features_df['volatility_std_20'].fillna(0).values[-20:]
            
            if len(returns) < 10:
                return 0.0
            
            current_state, confidence, state_probs = self.hmm_detector.predict(returns, volatility)
            
            # State 0=calm, 1=volatile, 2=extreme
            # Map state to anomaly score
            state_scores = {0: 0.0, 1: 0.5, 2: 1.0}
            base_score = state_scores.get(current_state, 0.0)
            
            # Weight by confidence
            score = base_score * confidence
            
            return float(score)
            
        except Exception as e:
            logger.error(f"HMM scoring failed: {e}")
            return 0.0
    
    def get_anomaly_summary(self, lookback_hours: int = 24) -> pd.DataFrame:
        """
        Get summary of recent anomalies.
        
        Args:
            lookback_hours: Hours to look back
            
        Returns:
            DataFrame with anomaly events
        """
        if len(self.anomaly_history) == 0:
            return pd.DataFrame()
        
        cutoff_time = datetime.now().timestamp() - (lookback_hours * 3600)
        
        recent_anomalies = [
            a for a in self.anomaly_history 
            if datetime.fromisoformat(a['timestamp']).timestamp() > cutoff_time
        ]
        
        if len(recent_anomalies) == 0:
            return pd.DataFrame()
        
        summary_data = []
        for anomaly in recent_anomalies:
            summary_data.append({
                'timestamp': anomaly['timestamp'],
                'detected': anomaly['anomaly_detected'],
                'score': anomaly['ensemble_score'],
                'if_score': anomaly['component_scores']['isolation_forest'],
                'dbscan_score': anomaly['component_scores']['dbscan'],
                'hmm_score': anomaly['component_scores']['hmm']
            })
        
        return pd.DataFrame(summary_data)
    
    def calibrate_threshold(self, features_df: pd.DataFrame, target_fpr: float = 0.01) -> float:
        """
        Calibrate anomaly threshold to achieve target false positive rate.
        
        Args:
            features_df: Validation feature DataFrame
            target_fpr: Target false positive rate (e.g., 0.01 = 1%)
            
        Returns:
            Calibrated threshold
        """
        logger.info(f"Calibrating threshold for FPR={target_fpr}")
        
        feature_cols = [c for c in features_df.columns if c not in ['timestamp', 'symbol']]
        X = features_df[feature_cols].fillna(0).values
        
        scores = []
        for i in range(len(X)):
            if_score = self._isolation_forest_score(X[i:i+1])
            dbscan_score = self._dbscan_score(X[max(0, i-100):i+1])
            
            ensemble_score = (
                self.weights['isolation_forest'] * if_score +
                self.weights['dbscan'] * dbscan_score
            )
            scores.append(ensemble_score)
        
        scores = np.array(scores)
        threshold = np.percentile(scores, (1 - target_fpr) * 100)
        
        logger.info(f"Calibrated threshold: {threshold:.3f}")
        return float(threshold)
# =====================================================
# ✅ CONTINUOUS GPU ENSEMBLE EXECUTION LOOP
# =====================================================
if __name__ == "__main__":
    logger.info("🚀 Starting Ensemble GPU Batch Runner")

    detector = EnsembleAnomalyDetector(CONFIG)
    latency_log: list[float] = []
    loop_count = 0

    # === Example mock-up data generator ===
    def generate_mock_features(n: int = 300) -> pd.DataFrame:
        np.random.seed(42)
        df = pd.DataFrame({
            "timestamp": [datetime.utcnow().isoformat()] * n,
            "symbol": ["BTCUSDT"] * n,
            "return_1": np.random.normal(0, 0.01, n),
            "volatility_std_20": np.abs(np.random.normal(0.01, 0.005, n)),
            "momentum_5": np.random.normal(0, 1, n),
            "rsi_14": np.random.uniform(30, 70, n),
        })
        return df

    while True:
        t0 = time.perf_counter()
        features_df = generate_mock_features()

        # Retrain if needed
        if detector.isolation_forest is None:
            detector.train(features_df)

        # Predict anomalies
        result = detector.predict(features_df)

        # Measure latency
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latency_log.append(elapsed_ms)
        latency_log = latency_log[-200:]
        p95 = statistics.quantiles(latency_log, n=100)[94] if len(latency_log) >= 10 else elapsed_ms

        logger.info(
            f"[{loop_count:05d}] Anomaly={result['anomaly_detected']} | "
            f"Score={result['ensemble_score']:.3f} | p95={p95:.1f}ms | GPU={detector.gpu_enabled}"
        )

        if p95 > LATENCY_TARGET_MS:
            logger.warning(f"⚠️ Latency p95={p95:.1f}ms exceeds {LATENCY_TARGET_MS}ms target!")

        loop_count += 1
        time.sleep(BATCH_INTERVAL)
