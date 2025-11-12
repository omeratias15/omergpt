"""
GPU-Accelerated Walk-Forward Backtest Runner with Embargo
- Runs walk-forward backtests over historical DuckDB signal data
- Handles time-embargo, realistic slippage, and rolling validation windows
- Generates full JSON/HTML reports for dashboard and validation_metrics logging
"""

import cupy as cp
import duckdb
import logging
import os
import pandas as pd
import json
from typing import Optional, Dict

TABLE_SIGNALS = "signals"
BACKTEST_REPORTS_DIR = "reports"

class BacktestRunner:
    def __init__(
        self,
        db_path: str = "data/market_data.duckdb",
        embargo: int = 5,  # Embargo length (bars excluded after signal event)
        window_size: int = 100,  # Rolling window for validation
        logger: Optional[logging.Logger] = None,
    ):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.embargo = embargo
        self.window_size = window_size
        self.logger = logger or logging.getLogger("BacktestRunner")
        self.stats = {"runs": 0, "errors": 0}
        os.makedirs(BACKTEST_REPORTS_DIR, exist_ok=True)

    def load_data(self, limit: int = 2000) -> pd.DataFrame:
        query = f"""
            SELECT *
            FROM {TABLE_SIGNALS}
            ORDER BY timestamp ASC
            LIMIT {limit}
        """
        df = self.conn.execute(query).fetchdf()
        return df

    def walk_forward(self, df: pd.DataFrame) -> Dict:
        n = df.shape[0]
        # Convert to GPU
        returns = cp.array(df["return"].values, dtype=cp.float32)
        signals = cp.array(df["signal"].values, dtype=cp.int32)
        metrics = []
        embargoed = cp.zeros(n, dtype=cp.bool_)
        for start in range(0, n - self.window_size, self.window_size):
            end = start + self.window_size
            r_win = returns[start:end]
            s_win = signals[start:end]
            # Apply embargo: mark embargoed after any positive signal
            embargo_marks = cp.where(s_win == 1)[0]
            for idx in embargo_marks:
                embargoed[start+idx : start+idx+self.embargo] = True
            valid_idx = ~embargoed[start:end]
            # Metrics for each window
            if cp.sum(valid_idx) < 10:
                continue
            m = self._compute_metrics(r_win[valid_idx], s_win[valid_idx])
            m["start_bar"] = int(start)
            m["end_bar"] = int(end)
            metrics.append(m)
        self.stats["runs"] += 1
        return {"windows": metrics, "total_windows": len(metrics)}

    def _compute_metrics(self, returns, signals) -> Dict:
        sharpe = cp.mean(returns) / (cp.std(returns) + 1e-8) * cp.sqrt(252)
        max_dd = self._max_drawdown(returns)
        hit_rate = cp.mean(signals == (returns > 0))
        return {"sharpe": float(sharpe), "max_drawdown": float(max_dd), "hit_rate": float(hit_rate)}

    def _max_drawdown(self, rets: cp.ndarray) -> float:
        cum_ret = cp.cumsum(rets)
        roll_max = cp.maximum.accumulate(cum_ret)
        drawdowns = cum_ret - roll_max
        return float(cp.min(drawdowns))

    def generate_report(self, results: Dict, filename: str = "backtest_report"):
        json_path = os.path.join(BACKTEST_REPORTS_DIR, f"{filename}.json")
        html_path = os.path.join(BACKTEST_REPORTS_DIR, f"{filename}.html")
        try:
            with open(json_path, "w") as jf:
                json.dump(results, jf, indent=2)
            # Basic HTML table
            windows = results["windows"]
            html = "<html><body><h2>Backtest Report</h2><table border='1'><tr><th>Start</th><th>End</th><th>Sharpe</th><th>Drawdown</th><th>Hit Rate</th></tr>"
            for w in windows:
                html += f"<tr><td>{w['start_bar']}</td><td>{w['end_bar']}</td><td>{w['sharpe']:.2f}</td><td>{w['max_drawdown']:.2f}</td><td>{w['hit_rate']:.2f}</td></tr>"
            html += "</table></body></html>"
            with open(html_path, "w") as hf:
                hf.write(html)
            self.logger.info(f"✅ Backtest reports generated: {json_path}, {html_path}")
        except Exception as e:
            self.logger.error(f"❌ Report generation error: {e}")
            self.stats["errors"] += 1

    def run_pipeline(self, limit: int = 2000, filename: str = "backtest_report"):
        df = self.load_data(limit)
        results = self.walk_forward(df)
        self.generate_report(results, filename)
        self.logger.info(f"✅ Walk-forward backtest completed.")

    def get_stats(self) -> Dict:
        return self.stats.copy()

# Test main:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runner = BacktestRunner()
    runner.run_pipeline(limit=1000, filename="bt_run")
    print("✅ Backtest runner complete")
