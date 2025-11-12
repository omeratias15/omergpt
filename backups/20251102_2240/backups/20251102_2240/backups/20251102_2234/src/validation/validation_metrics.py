"""
GPU-Accelerated Validation Metrics Calculator
- Computes Sharpe, Sortino, Information Ratio, hit-rate, max drawdown, precision/recall
- Fully vectorized using cuPy, writes results to DuckDB and exports CSV
- Used in dashboard validation and model evaluation/backtesting
"""

import cupy as cp
import duckdb
import logging
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

TABLE_SIGNALS = "signals"
METRICS_TABLE = "validation_metrics"

class ValidationMetricsGPU:
    def __init__(self, db_path: str = "data/market_data.duckdb", logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.logger = logger or logging.getLogger("ValidationMetricsGPU")
        self.stats = {"metrics": 0, "errors": 0}

    def load_signal_data(self, limit: int = 1024) -> pd.DataFrame:
        query = f"""
            SELECT *
            FROM {TABLE_SIGNALS}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        df = self.conn.execute(query).fetchdf()
        return df

    def compute_metrics(self, returns: cp.ndarray, signals: cp.ndarray, benchmark: Optional[cp.ndarray] = None) -> Dict:
        # Sharpe Ratio (annualized)
        sharpe = cp.mean(returns) / (cp.std(returns) + 1e-8) * cp.sqrt(252)
        # Sortino (annualized)
        downside = cp.std(returns[returns < 0]) + 1e-8
        sortino = cp.mean(returns) / downside * cp.sqrt(252)
        # Max Drawdown
        cum_returns = cp.cumsum(returns)
        roll_max = cp.maximum.accumulate(cum_returns)
        drawdowns = cum_returns - roll_max
        max_drawdown = cp.min(drawdowns)
        # Information Ratio (vs benchmark)
        if benchmark is not None:
            active_ret = returns - benchmark
            ir = cp.mean(active_ret) / (cp.std(active_ret) + 1e-8) * cp.sqrt(252)
        else:
            ir = float("nan")
        # Hit-rate
        hit_rate = cp.mean(signals == (returns > 0))
        # Precision/Recall (binary signal vs realized positive returns)
        tp = cp.sum((signals == 1) & (returns > 0))
        fp = cp.sum((signals == 1) & (returns <= 0))
        fn = cp.sum((signals == 0) & (returns > 0))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        # Correlation Decay (lag 1)
        correlation_decay = cp.corrcoef(returns[:-1], returns[1:])[0, 1]
        metrics = dict(
            sharpe=float(sharpe),
            sortino=float(sortino),
            information_ratio=float(ir),
            max_drawdown=float(max_drawdown),
            hit_rate=float(hit_rate),
            precision=float(precision),
            recall=float(recall),
            correlation_decay=float(correlation_decay)
        )
        return metrics

    def run_pipeline(self, limit: int = 1024, export_csv: bool = True):
        df = self.load_signal_data(limit)
        returns = cp.array(df["return"].values, dtype=cp.float32)
        signals = cp.array(df["signal"].values, dtype=cp.int32)
        benchmark = cp.array(df["benchmark_return"].values, dtype=cp.float32) if "benchmark_return" in df else None
        metrics = self.compute_metrics(returns, signals, benchmark)
        self.write_to_db(metrics)
        if export_csv:
            self.export_to_csv(metrics)
        self.stats["metrics"] += 1
        self.logger.info(f"✅ Validation metrics computed: {metrics}")

    def write_to_db(self, metrics: Dict):
        try:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {METRICS_TABLE} (
                    sharpe FLOAT, sortino FLOAT, information_ratio FLOAT, max_drawdown FLOAT,
                    hit_rate FLOAT, precision FLOAT, recall FLOAT, correlation_decay FLOAT
                );
            """)
            self.conn.execute(f"""
                INSERT INTO {METRICS_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                metrics["sharpe"], metrics["sortino"], metrics["information_ratio"], metrics["max_drawdown"],
                metrics["hit_rate"], metrics["precision"], metrics["recall"], metrics["correlation_decay"]
            ))
            self.logger.info("✅ Validation metrics written to DB")
        except Exception as e:
            self.logger.error(f"❌ DB write error: {e}")
            self.stats["errors"] += 1

    def export_to_csv(self, metrics: Dict, filename: str = "reports/metrics.csv"):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            pd.DataFrame([metrics]).to_csv(filename, index=False)
            self.logger.info(f"✅ Validation metrics exported as CSV: {filename}")
        except Exception as e:
            self.logger.error(f"❌ CSV export error: {e}")
            self.stats["errors"] += 1

    def get_stats(self) -> Dict:
        return self.stats.copy()

# Test main:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = ValidationMetricsGPU()
    engine.run_pipeline(limit=500)
    print("✅ Validation metrics computation complete")
