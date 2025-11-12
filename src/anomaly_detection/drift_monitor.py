import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Model drift detection using PSI, KS test, and Wasserstein distance.
    Monitors feature distribution shifts to trigger model retraining.
    """
    
    def __init__(self, config: dict):
        """
        Initialize drift monitor.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.psi_threshold = config.get('validation', {}).get('psi_threshold', 0.2)
        self.ks_threshold = config.get('validation', {}).get('ks_threshold', 0.3)
        self.ks_pvalue_threshold = config.get('validation', {}).get('ks_pvalue_threshold', 0.01)
        self.wasserstein_threshold = config.get('validation', {}).get('wasserstein_threshold', 0.1)
        
        self.reference_distributions = {}
        self.drift_history = []
        
        logger.info("DriftMonitor initialized")
    
    def set_reference(self, features_df: pd.DataFrame, label: str = 'baseline') -> None:
        """
        Set reference distribution for drift detection.
        
        Args:
            features_df: DataFrame with features
            label: Reference label identifier
        """
        feature_cols = [c for c in features_df.columns if c not in ['timestamp', 'symbol']]
        
        self.reference_distributions[label] = {}
        
        for col in feature_cols:
            values = features_df[col].dropna().values
            if len(values) > 0:
                self.reference_distributions[label][col] = {
                    'values': values,
                    'mean': values.mean(),
                    'std': values.std(),
                    'quantiles': np.percentile(values, [10, 25, 50, 75, 90])
                }
        
        logger.info(f"Reference distribution '{label}' set with {len(feature_cols)} features")
    
    def detect_drift(self, features_df: pd.DataFrame, reference_label: str = 'baseline') -> Dict:
        """
        Detect distribution drift using multiple statistical tests.
        
        Args:
            features_df: Current feature DataFrame
            reference_label: Reference distribution to compare against
            
        Returns:
            Dictionary with drift metrics and alerts
        """
        if reference_label not in self.reference_distributions:
            logger.warning(f"Reference distribution '{reference_label}' not found")
            return {'drift_detected': False, 'reason': 'No reference'}
        
        feature_cols = [c for c in features_df.columns if c not in ['timestamp', 'symbol']]
        reference = self.reference_distributions[reference_label]
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'reference_label': reference_label,
            'features_tested': 0,
            'psi_violations': 0,
            'ks_violations': 0,
            'wasserstein_violations': 0,
            'drift_detected': False,
            'feature_drift': {}
        }
        
        for col in feature_cols:
            if col not in reference:
                continue
            
            current_values = features_df[col].dropna().values
            reference_values = reference[col]['values']
            
            if len(current_values) < 30 or len(reference_values) < 30:
                continue
            
            drift_results['features_tested'] += 1
            
            # PSI (Population Stability Index)
            psi_score = self._calculate_psi(reference_values, current_values)
            
            # KS (Kolmogorov-Smirnov) test
            ks_statistic, ks_pvalue = stats.ks_2samp(reference_values, current_values)
            
            # Wasserstein distance
            wasserstein_dist = stats.wasserstein_distance(reference_values, current_values)
            
            # Normalize Wasserstein by reference std
            wasserstein_normalized = wasserstein_dist / (reference[col]['std'] + 1e-8)
            
            # Drift flags
            psi_drift = psi_score > self.psi_threshold
            ks_drift = ks_statistic > self.ks_threshold or ks_pvalue < self.ks_pvalue_threshold
            wasserstein_drift = wasserstein_normalized > self.wasserstein_threshold
            
            if psi_drift:
                drift_results['psi_violations'] += 1
            if ks_drift:
                drift_results['ks_violations'] += 1
            if wasserstein_drift:
                drift_results['wasserstein_violations'] += 1
            
            # Store detailed metrics for high-drift features
            if psi_drift or ks_drift or wasserstein_drift:
                drift_results['feature_drift'][col] = {
                    'psi': float(psi_score),
                    'ks_statistic': float(ks_statistic),
                    'ks_pvalue': float(ks_pvalue),
                    'wasserstein': float(wasserstein_normalized),
                    'drift_flags': {
                        'psi': psi_drift,
                        'ks': ks_drift,
                        'wasserstein': wasserstein_drift
                    }
                }
        
        # Overall drift detection: >30% features show drift
        total_violations = (drift_results['psi_violations'] + 
                          drift_results['ks_violations'] + 
                          drift_results['wasserstein_violations'])
        
        max_possible = drift_results['features_tested'] * 3
        drift_percentage = total_violations / max_possible if max_possible > 0 else 0
        
        drift_results['drift_percentage'] = drift_percentage
        drift_results['drift_detected'] = drift_percentage > 0.3
        
        # Log results
        if drift_results['drift_detected']:
            logger.warning(f"Drift detected: {drift_percentage:.1%} violations "
                         f"(PSI: {drift_results['psi_violations']}, "
                         f"KS: {drift_results['ks_violations']}, "
                         f"Wasserstein: {drift_results['wasserstein_violations']})")
        
        self.drift_history.append(drift_results)
        
        return drift_results
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))
        
        Args:
            reference: Reference distribution
            current: Current distribution
            buckets: Number of bins for discretization
            
        Returns:
            PSI score
        """
        # Define bin edges from reference distribution
        bin_edges = np.percentile(reference, np.linspace(0, 100, buckets + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        # Bin both distributions
        reference_counts, _ = np.histogram(reference, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages
        reference_pct = reference_counts / len(reference)
        current_pct = current_counts / len(current)
        
        # Avoid division by zero
        reference_pct = np.where(reference_pct == 0, 0.0001, reference_pct)
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)
        
        # Calculate PSI
        psi = np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))
        
        return psi
    
    def get_drift_summary(self, lookback_hours: int = 24) -> pd.DataFrame:
        """
        Get summary of recent drift detections.
        
        Args:
            lookback_hours: Hours to look back
            
        Returns:
            DataFrame with drift summary
        """
        if len(self.drift_history) == 0:
            return pd.DataFrame()
        
        cutoff_time = datetime.now().timestamp() - (lookback_hours * 3600)
        
        recent_drifts = [
            d for d in self.drift_history 
            if datetime.fromisoformat(d['timestamp']).timestamp() > cutoff_time
        ]
        
        if len(recent_drifts) == 0:
            return pd.DataFrame()
        
        summary_df = pd.DataFrame(recent_drifts)[
            ['timestamp', 'features_tested', 'psi_violations', 
             'ks_violations', 'wasserstein_violations', 'drift_percentage', 'drift_detected']
        ]
        
        return summary_df
    
    def get_feature_drift_ranking(self, reference_label: str = 'baseline') -> pd.DataFrame:
        """
        Rank features by cumulative drift score.
        
        Args:
            reference_label: Reference distribution label
            
        Returns:
            DataFrame with features ranked by drift severity
        """
        if len(self.drift_history) == 0:
            return pd.DataFrame()
        
        feature_scores = {}
        
        for drift_result in self.drift_history[-10:]:  # Last 10 checks
            for feature, metrics in drift_result.get('feature_drift', {}).items():
                if feature not in feature_scores:
                    feature_scores[feature] = {
                        'psi_sum': 0,
                        'ks_sum': 0,
                        'wasserstein_sum': 0,
                        'count': 0
                    }
                
                feature_scores[feature]['psi_sum'] += metrics['psi']
                feature_scores[feature]['ks_sum'] += metrics['ks_statistic']
                feature_scores[feature]['wasserstein_sum'] += metrics['wasserstein']
                feature_scores[feature]['count'] += 1
        
        # Calculate averages
        ranking = []
        for feature, scores in feature_scores.items():
            count = scores['count']
            ranking.append({
                'feature': feature,
                'avg_psi': scores['psi_sum'] / count,
                'avg_ks': scores['ks_sum'] / count,
                'avg_wasserstein': scores['wasserstein_sum'] / count,
                'drift_frequency': count / min(10, len(self.drift_history))
            })
        
        ranking_df = pd.DataFrame(ranking)
        if len(ranking_df) > 0:
            ranking_df['composite_score'] = (
                ranking_df['avg_psi'] + 
                ranking_df['avg_ks'] + 
                ranking_df['avg_wasserstein']
            )
            ranking_df = ranking_df.sort_values('composite_score', ascending=False)
        
        return ranking_df
    
    def should_retrain(self, min_drift_checks: int = 3) -> Tuple[bool, str]:
        """
        Determine if model retraining is required.
        
        Args:
            min_drift_checks: Minimum number of consecutive drift detections
            
        Returns:
            Tuple of (should_retrain, reason)
        """
        if len(self.drift_history) < min_drift_checks:
            return False, "Insufficient drift history"
        
        recent_drifts = self.drift_history[-min_drift_checks:]
        consecutive_drift = all(d['drift_detected'] for d in recent_drifts)
        
        if consecutive_drift:
            avg_drift_pct = np.mean([d['drift_percentage'] for d in recent_drifts])
            return True, f"Consecutive drift detected: {avg_drift_pct:.1%} avg violations"
        
        # Check severe drift in last check
        last_drift = self.drift_history[-1]
        if last_drift['drift_percentage'] > 0.5:
            return True, f"Severe drift: {last_drift['drift_percentage']:.1%} violations"
        
        return False, "No retraining required"
"""
GPU Data Drift Monitor (PSI/KS/Wasserstein/Jensen-Shannon)
- Periodically assesses distribution drift in features vs baseline (reference)
- Triggers adaptive model retraining via adaptive_learning/trainer.py on drift events
- GPU batch calc via cuPy, dashboard/alert output, logs drift metrics to DuckDB for audit
"""

import cupy as cp
import cudf
import duckdb
import logging
import os
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from typing import List, Optional, Dict

FEATURE_TABLE = "features_combined"
DRIFT_TABLE = "feature_drift"

class DriftMonitorEngine:
    def __init__(self, db_path: str = "data/market_data.duckdb", logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.logger = logger or logging.getLogger("DriftMonitorEngine")
        self.stats = {"drift_events": 0, "errors": 0}
        self.retrain_triggered = False

    def load_features(self, limit: int = 512) -> cudf.DataFrame:
        query = f"""
            SELECT *
            FROM {FEATURE_TABLE}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        df_cpu = self.conn.execute(query).fetchdf()
        df_gpu = cudf.DataFrame.from_pandas(df_cpu)
        return df_gpu

    def _calc_psi(self, ref: cp.ndarray, cur: cp.ndarray, bins: int = 10) -> float:
        """Population Stability Index - CPU calculation for now"""
        # Calculate bin edges
        ref_hist, ref_bins = np.histogram(cp.asnumpy(ref), bins=bins)
        cur_hist, _ = np.histogram(cp.asnumpy(cur), bins=ref_bins)
        ref_percents = ref_hist / (np.sum(ref_hist) + 1e-8)
        cur_percents = cur_hist / (np.sum(cur_hist) + 1e-8)
        psi = np.sum((ref_percents - cur_percents) * np.log((ref_percents + 1e-8) / (cur_percents + 1e-8)))
        return float(psi)

    def _calc_ks(self, ref: cp.ndarray, cur: cp.ndarray) -> float:
        ks = ks_2samp(cp.asnumpy(ref), cp.asnumpy(cur)).statistic
        return float(ks)

    def _calc_wasserstein(self, ref: cp.ndarray, cur: cp.ndarray) -> float:
        return float(wasserstein_distance(cp.asnumpy(ref), cp.asnumpy(cur)))

    def _calc_jsd(self, ref: cp.ndarray, cur: cp.ndarray, bins: int = 10) -> float:
        # Jensen-Shannon Divergence
        ref_hist, ref_bins = np.histogram(cp.asnumpy(ref), bins=bins, density=True)
        cur_hist, _ = np.histogram(cp.asnumpy(cur), bins=ref_bins, density=True)
        m = (ref_hist + cur_hist) / 2
        kl_ref = np.sum(ref_hist * np.log((ref_hist + 1e-8) / (m + 1e-8)))
        kl_cur = np.sum(cur_hist * np.log((cur_hist + 1e-8) / (m + 1e-8)))
        jsd = 0.5 * (kl_ref + kl_cur)
        return float(jsd)

    def run_drift(self, feature_col: str = "close", ref_window: int = 256, cur_window: int = 256):
        df = self.load_features(limit=ref_window + cur_window)
        ref = df[feature_col][:ref_window].to_gpu_array()
        cur = df[feature_col][ref_window:ref_window+cur_window].to_gpu_array()

        psi = self._calc_psi(ref, cur)
        ks = self._calc_ks(ref, cur)
        ws = self._calc_wasserstein(ref, cur)
        jsd = self._calc_jsd(ref, cur)

        drift_event = {
            "timestamp": float(df["timestamp"].iloc[-1]),
            "feature": feature_col,
            "psi": psi,
            "ks": ks,
            "wasserstein": ws,
            "jsd": jsd,
        }
        self.logger.info(f"📉 Drift metrics[{feature_col}]: PSI={psi:.3f}, KS={ks:.3f}, WS={ws:.3f}, JSD={jsd:.3f}")

        self.write_to_db(drift_event)
        drift_triggered = (psi > 0.1 or ks > 0.15 or jsd > 0.04)
        if drift_triggered and not self.retrain_triggered:
            self.stats["drift_events"] += 1
            self.trigger_retrain(feature_col)
            self.retrain_triggered = True

    def write_to_db(self, event: Dict):
        try:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {DRIFT_TABLE} (
                    timestamp FLOAT, feature VARCHAR, psi FLOAT, ks FLOAT, wasserstein FLOAT, jsd FLOAT
                );
            """)
            self.conn.execute(f"""
                INSERT INTO {DRIFT_TABLE} VALUES (?, ?, ?, ?, ?, ?);
            """, (
                event["timestamp"], event["feature"], event["psi"], event["ks"], event["wasserstein"], event["jsd"]
            ))
            self.logger.info(f"✅ Drift event written to DB")
        except Exception as e:
            self.logger.error(f"❌ Drift DB error: {e}")
            self.stats["errors"] += 1

    def trigger_retrain(self, feature_col: str):
        self.logger.warning(f"🚨 Trigger adaptive_learning/trainer.py retrain for feature: {feature_col}")

    def get_stats(self) -> Dict:
        return self.stats.copy()

# Test main:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = DriftMonitorEngine()
    engine.run_drift(feature_col="close")
    print("✅ Drift monitoring complete")

# === AUTO PATCH START ===
# Auto-added: Retrain trigger when drift exceeds thresholds
try:
    if should_retrain():
        print("⚠️ Drift threshold exceeded — triggering retrain job.")
        from src.adaptive_learning import trainer
        trainer.run_retrain_job()
except Exception as e:
    print("[AUTO PATCH] Drift monitor retrain trigger failed:", e)
# === AUTO PATCH END ===

