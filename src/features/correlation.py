"""
GPU Rolling Correlation Module
- Computes rolling Pearson and Spearman correlations between features/assets
- Uses cuDF/cuPy for all major operations, falls back to CPU if GPU unavailable
- Writes correlation features to DuckDB (feature_correlation table)
- Yields dicts for downstream models

Assumes candle/price data are loaded from DuckDB via db_manager
"""
try:
    import cudf
    import cupy as cp
    GPU_ENABLED = True
except ImportError:
    import pandas as cudf
    import numpy as cp
    GPU_ENABLED = False

import numpy as np
import duckdb
import logging
from typing import List, Dict, Optional

WINDOW_SIZE = 50  # Rolling window (configurable)
GPU_ENABLED = True  # Auto-detect in __init__
DB_PATH = "data/market_data.duckdb"
TABLE_CANDLES = "candles"
TABLE_OUTPUT = "feature_correlation"

class GPUCorrelationEngine:
    def __init__(
        self, 
        window_size: int = WINDOW_SIZE, 
        symbols: Optional[List[str]] = None,
        db_path: str = DB_PATH,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            window_size: Size of rolling window
            symbols: Asset symbols to correlate (e.g. ["BTCUSDT","ETHUSDT"])
            db_path: DuckDB database file
            logger: Logger instance
        """
        self.window_size = window_size
        self.symbols = symbols or []
        self.logger = logger or logging.getLogger("GPUCorrelationEngine")
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.gpu_enabled = self._check_gpu()
        self.stats = {"pairs": 0, "computed": 0, "errors": 0}

    def _check_gpu(self) -> bool:
        try:
            arr = cp.array([1.0, 2.0])
            cp.sum(arr)
            self.logger.info("✅ GPU detected for cuDF/cuPy correlation")
            return True
        except Exception:
            self.logger.warning("⚠️ GPU not available, using CPU fallback")
            return False

    def load_data(self, symbols: List[str]) -> cudf.DataFrame:
        """Loads OHLCV data from DuckDB into cuDF DataFrame"""
        query = f"""
            SELECT timestamp, symbol, close
            FROM {TABLE_CANDLES}
            WHERE symbol IN ({','.join([f"'{s}'" for s in symbols])})
            ORDER BY timestamp ASC
        """
        df_cpu = self.conn.execute(query).fetchdf()
        df_gpu = cudf.DataFrame.from_pandas(df_cpu)
        return df_gpu

    def rolling_corr(self, df: cudf.DataFrame, method: str = "pearson") -> cudf.DataFrame:
        """
        Compute rolling Pearson/Spearman correlation matrix for each asset pair
        Returns cuDF DataFrame with columns: timestamp, asset_a, asset_b, corr_pearson, corr_spearman
        """
        results = []
        assets = df["symbol"].unique().to_pandas()
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset_a = assets[i]
                asset_b = assets[j]
                df_a = df[df["symbol"] == asset_a].sort_values("timestamp")
                df_b = df[df["symbol"] == asset_b].sort_values("timestamp")

                merged = df_a[["timestamp", "close"]].merge(
                    df_b[["timestamp", "close"]], left_on="timestamp", right_on="timestamp", 
                    suffixes=(f"_{asset_a}", f"_{asset_b}")
                )

                closes_a = merged[f"close_{asset_a}"].to_gpu_array()
                closes_b = merged[f"close_{asset_b}"].to_gpu_array()

                # Pearson rolling
                corr_pearson = self._rolling_corr_gpu(closes_a, closes_b, method="pearson")
                # Spearman rolling
                corr_spearman = self._rolling_corr_gpu(closes_a, closes_b, method="spearman")

                result = cudf.DataFrame({
                    "timestamp": merged["timestamp"],
                    "asset_a": asset_a,
                    "asset_b": asset_b,
                    "corr_pearson": corr_pearson,
                    "corr_spearman": corr_spearman
                })
                results.append(result)
                self.stats["pairs"] += 1

        out = cudf.concat(results) if results else cudf.DataFrame()
        self.stats["computed"] += len(out)
        return out

    def _rolling_corr_gpu(self, x: cp.ndarray, y: cp.ndarray, method: str = "pearson") -> cp.ndarray:
        """
        Compute rolling correlation on GPU (Pearson/Spearman)
        """
        n = len(x)
        corrs = cp.zeros(n, dtype=cp.float32)
        for i in range(self.window_size - 1, n):
            a = x[i + 1 - self.window_size:i + 1]
            b = y[i + 1 - self.window_size:i + 1]
            if method == "pearson":
                corrs[i] = cp.corrcoef(a, b)[0, 1]
            elif method == "spearman":
                ra = cp.argsort(cp.argsort(a))
                rb = cp.argsort(cp.argsort(b))
                corrs[i] = cp.corrcoef(ra, rb)[0, 1]
            else:
                corrs[i] = cp.nan
        return corrs

    def write_to_db(self, df: cudf.DataFrame):
        """
        Write correlation results to DuckDB table (feature_correlation)
        """
        try:
            pdf = df.to_pandas()
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_OUTPUT} AS SELECT * FROM pdf LIMIT 0;")
            self.conn.execute(f"INSERT INTO {TABLE_OUTPUT} SELECT * FROM pdf;")
            self.logger.info(f"✅ Wrote {len(pdf)} rows to {TABLE_OUTPUT}")
        except Exception as e:
            self.logger.error(f"❌ DB write error: {e}")
            self.stats["errors"] += 1

    def compute_and_store(self):
        """Orchestrate full pipeline"""
        if not self.symbols:
            self.logger.error("❌ No symbols set for correlation")
            return
        df = self.load_data(self.symbols)
        corr_df = self.rolling_corr(df)
        self.write_to_db(corr_df)

    def get_stats(self) -> Dict:
        return self.stats.copy()


# Test main
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = GPUCorrelationEngine(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    engine.compute_and_store()
    print("✅ Correlation computation complete.")
