"""
Whale Transaction Clustering with cuML DBSCAN
- Runs DBSCAN on large on-chain transfer data (from Etherscan or similar)
- Finds clusters of whale activities, triggers Telegram/API alerts for significant groups
- GPU accelerated, async-batch-ready, writes cluster meta to DuckDB
"""

import cudf
import cupy as cp
import duckdb
import logging
import os
from cuml.cluster import DBSCAN
from typing import Optional, List, Dict

WHALE_TABLE = "whale_transactions"
CLUSTER_TABLE = "whale_clusters"

class WhaleDBSCANClusterer:
    def __init__(self, db_path: str = "data/omergpt.db", eps: float = 0.5, min_samples: int = 3, logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.eps = eps
        self.min_samples = min_samples
        self.logger = logger or logging.getLogger("WhaleDBSCANClusterer")
        self.stats = {"clusters": 0, "alerts": 0, "errors": 0}

    def load_whale_data(self, limit: int = 128) -> cudf.DataFrame:
        query = f"""
            SELECT timestamp, value_eth, value_usd, gas_price, block
            FROM {WHALE_TABLE}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        df_cpu = self.conn.execute(query).fetchdf()
        df_gpu = cudf.DataFrame.from_pandas(df_cpu)
        return df_gpu

    def run_dbscan(self, df: cudf.DataFrame) -> cudf.DataFrame:
        X = df[["value_eth", "value_usd", "gas_price"]].astype("float32")
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clusterer.fit_predict(X)
        df["cluster_id"] = labels
        self.logger.info(f"✅ DBSCAN clusters: {labels.max().get()+1} formed")

        # alert if large cluster detected
        cluster_counts = df["cluster_id"].value_counts()
        for cluster_id, count in zip(cluster_counts.index.to_pandas(), cluster_counts.values.to_pandas()):
            if count >= 5 and cluster_id >= 0:
                self.stats["alerts"] += 1
                self._send_alert(cluster_id, count)
        self.stats["clusters"] += int(cluster_counts.shape[0])
        return df

    def write_to_db(self, df: cudf.DataFrame):
        try:
            pdf = df.to_pandas()
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {CLUSTER_TABLE} AS SELECT * FROM pdf LIMIT 0;")
            self.conn.execute(f"INSERT INTO {CLUSTER_TABLE} SELECT * FROM pdf;")
            self.logger.info(f"✅ {len(pdf)} clustered whale tx written")
        except Exception as e:
            self.logger.error(f"❌ DB write error: {e}")
            self.stats["errors"] += 1

    def _send_alert(self, cluster_id, count):
        self.logger.warning(f"🐋 Whale DBSCAN ALERT: Cluster {cluster_id} has {count} tx (integrate Telegram)")

    def run_pipeline(self, limit: int = 128):
        df = self.load_whale_data(limit)
        cluster_df = self.run_dbscan(df)
        self.write_to_db(cluster_df)

    def get_stats(self) -> Dict:
        return self.stats.copy()

# Test main:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = WhaleDBSCANClusterer()
    engine.run_pipeline(limit=100)
    print("✅ Whale DBSCAN clustering complete")


# [PATCH] Add 10-minute window filter
        # [PATCH] Ensure data limited to last 10 minutes
        df = df[df['timestamp'] >= (datetime.utcnow().timestamp()*1000 - 10*60*1000)]
