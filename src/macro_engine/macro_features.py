"""
src/macro_engine/macro_features.py

Composite macro risk feature generator combining FRED and Yahoo Finance data.
Computes risk-on/risk-off regime index using standardized z-scores of macro indicators.
Exports results to DuckDB and JSON for downstream consumption by signal engine.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger("omerGPT.macro.features")


class MacroFeatureEngine:
    """
    Macro feature generator for risk regime classification.
    
    Features:
    - Load FRED indicators (DXY, VIX, FEDFUNDS, CPI)
    - Load Yahoo Finance correlations (SPX-BTC, SPX-ETH)
    - Compute standardized z-scores for all indicators
    - Generate composite risk-on/risk-off index
    - Export to DuckDB and JSON
    """
    
    def __init__(self, db_path: str = "data/macro_data.duckdb"):
        """
        Initialize macro feature engine.
        
        Args:
            db_path: Path to DuckDB database with macro data
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._create_table()
        
        logger.info(f"MacroFeatureEngine initialized: db={db_path}")
    
    def _create_table(self):
        """Create macro_risk_index table if it doesn't exist."""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_risk_index (
                    date DATE NOT NULL PRIMARY KEY,
                    risk_index DOUBLE,
                    risk_regime VARCHAR,
                    dxy_zscore DOUBLE,
                    vix_zscore DOUBLE,
                    fed_zscore DOUBLE,
                    spx_btc_corr DOUBLE,
                    spx_eth_corr DOUBLE,
                    created_at BIGINT DEFAULT CAST(strftime('%s', 'now') AS BIGINT)
                )
            """)
            
            logger.info("macro_risk_index table ready")
        
        except Exception as e:
            logger.error(f"Failed to create table: {e}", exc_info=True)
            raise
    
    async def load_fred_data(self) -> pd.DataFrame:
        """
        Load FRED macro indicators from database.
        
        Returns:
            DataFrame with columns: date, indicator, value
        """
        try:
            df = self.conn.execute("""
                SELECT date, indicator, value
                FROM macro_indicators
                ORDER BY date DESC
            """).df()
            
            logger.info(f"Loaded {len(df)} FRED indicator records")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load FRED data: {e}", exc_info=True)
            return pd.DataFrame()
    
    async def load_correlation_data(self) -> pd.DataFrame:
        """
        Load market correlation data from database.
        
        Returns:
            DataFrame with columns: date, pair, correlation
        """
        try:
            df = self.conn.execute("""
                SELECT date, pair, correlation
                FROM market_correlations
                WHERE pair IN ('SPX-BTC', 'SPX-ETH')
                ORDER BY date DESC
            """).df()
            
            logger.info(f"Loaded {len(df)} correlation records")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load correlation data: {e}", exc_info=True)
            return pd.DataFrame()
    
    def compute_zscore(self, series: pd.Series, window: int = 90) -> pd.Series:
        """
        Compute rolling z-score for a time series.
        
        Args:
            series: Pandas Series of values
            window: Rolling window for mean/std calculation (default: 90 days)
        
        Returns:
            Series of z-scores
        """
        rolling_mean = series.rolling(window=window, min_periods=30).mean()
        rolling_std = series.rolling(window=window, min_periods=30).std()
        
        zscore = (series - rolling_mean) / (rolling_std + 1e-8)
        return zscore
    
    async def compute_risk_index(self) -> pd.DataFrame:
        """
        Compute composite risk-on/risk-off index.
        
        Formula:
            risk_index = (SPX_corr - DXY_z - VIX_z - FED_z) / 4
        
        Interpretation:
            Positive = Risk-On (favorable for crypto)
            Negative = Risk-Off (unfavorable for crypto)
        
        Returns:
            DataFrame with risk index and regime classification
        """
        try:
            # Load data
            fred_df = await self.load_fred_data()
            corr_df = await self.load_correlation_data()
            
            if fred_df.empty:
                logger.warning("No FRED data available")
                return pd.DataFrame()
            
            # Pivot FRED data
            fred_pivot = fred_df.pivot(
                index="date", columns="indicator", values="value"
            )
            
            # Compute z-scores for key indicators
            fred_pivot["dxy_zscore"] = self.compute_zscore(
                fred_pivot.get("DEXUSAL", pd.Series([]))
            )
            fred_pivot["vix_zscore"] = self.compute_zscore(
                fred_pivot.get("VIXCLS", pd.Series([]))
            )
            fred_pivot["fed_zscore"] = self.compute_zscore(
                fred_pivot.get("FEDFUNDS", pd.Series([]))
            )
            
            # Add correlation data
            if not corr_df.empty:
                corr_pivot = corr_df.pivot(
                    index="date", columns="pair", values="correlation"
                )
                
                fred_pivot = fred_pivot.join(corr_pivot, how="left")
                fred_pivot["spx_btc_corr"] = fred_pivot.get("SPX-BTC", 0.0).fillna(0.0)
                fred_pivot["spx_eth_corr"] = fred_pivot.get("SPX-ETH", 0.0).fillna(0.0)
            else:
                fred_pivot["spx_btc_corr"] = 0.0
                fred_pivot["spx_eth_corr"] = 0.0
            
            # Compute composite risk index
            # Risk-On: High SPX correlation, Low DXY, Low VIX, Low Fed Rate
            fred_pivot["risk_index"] = (
                fred_pivot["spx_btc_corr"]
                - fred_pivot["dxy_zscore"].fillna(0)
                - fred_pivot["vix_zscore"].fillna(0)
                - fred_pivot["fed_zscore"].fillna(0)
            ) / 4.0
            
            # Classify regime
            fred_pivot["risk_regime"] = fred_pivot["risk_index"].apply(
                lambda x: "RISK_ON" if x > 0.1 else ("RISK_OFF" if x < -0.1 else "NEUTRAL")
            )
            
            # Prepare output
            result = fred_pivot[[
                "risk_index",
                "risk_regime",
                "dxy_zscore",
                "vix_zscore",
                "fed_zscore",
                "spx_btc_corr",
                "spx_eth_corr",
            ]].reset_index()
            
            logger.info(f"Computed risk index for {len(result)} dates")
            return result
        
        except Exception as e:
            logger.error(f"Risk index computation failed: {e}", exc_info=True)
            return pd.DataFrame()
    
    async def save_to_db(self, df: pd.DataFrame):
        """
        Save risk index data to DuckDB.
        
        Args:
            df: DataFrame with risk index data
        """
        if df.empty:
            logger.warning("No risk index data to save")
            return
        
        try:
            params = [
                (
                    row["date"],
                    float(row["risk_index"]),
                    row["risk_regime"],
                    float(row["dxy_zscore"]),
                    float(row["vix_zscore"]),
                    float(row["fed_zscore"]),
                    float(row["spx_btc_corr"]),
                    float(row["spx_eth_corr"]),
                )
                for _, row in df.iterrows()
            ]
            
            self.conn.executemany("""
                INSERT OR REPLACE INTO macro_risk_index 
                (date, risk_index, risk_regime, dxy_zscore, vix_zscore, 
                 fed_zscore, spx_btc_corr, spx_eth_corr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, params)
            
            logger.info(f"Saved {len(params)} risk index records")
        
        except Exception as e:
            logger.error(f"Failed to save risk index: {e}", exc_info=True)
    
    def export_latest_regime(self, output_dir: str = "data/exports") -> Dict:
        """
        Export latest risk regime to JSON file.
        
        Args:
            output_dir: Output directory for JSON export
        
        Returns:
            Dictionary with latest risk regime data
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            result = self.conn.execute("""
                SELECT date, risk_index, risk_regime
                FROM macro_risk_index
                ORDER BY date DESC
                LIMIT 1
            """).fetchone()
            
            if not result:
                logger.warning("No risk regime data available")
                return {}
            
            regime_data = {
                "date": str(result[0]),
                "risk_index": float(result[1]),
                "risk_regime": result[2],
                "updated_at": datetime.utcnow().isoformat(),
            }
            
            json_path = os.path.join(output_dir, "latest_risk_regime.json")
            with open(json_path, "w") as f:
                json.dump(regime_data, f, indent=2)
            
            logger.info(f"Exported latest regime: {regime_data['risk_regime']}")
            return regime_data
        
        except Exception as e:
            logger.error(f"Failed to export regime: {e}", exc_info=True)
            return {}
    
    async def generate_risk_features(self) -> Dict:
        """
        Complete workflow: compute risk index, save to DB, export to JSON.
        
        Returns:
            Dictionary with latest risk regime
        """
        df = await self.compute_risk_index()
        await self.save_to_db(df)
        regime = self.export_latest_regime()
        return regime
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Test block
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    async def test_macro_features():
        """Test macro feature engine."""
        print("Testing MacroFeatureEngine...")
        
        engine = MacroFeatureEngine("data/test_macro.duckdb")
        
        print("\n1. Computing risk index...")
        regime = await engine.generate_risk_features()
        
        print(f"\n2. Latest risk regime:")
        print(f"   Date: {regime.get('date', 'N/A')}")
        print(f"   Risk Index: {regime.get('risk_index', 0):.3f}")
        print(f"   Regime: {regime.get('risk_regime', 'UNKNOWN')}")
        
        # Show recent history
        print("\n3. Recent risk regime history:")
        df = engine.conn.execute("""
            SELECT date, risk_index, risk_regime
            FROM macro_risk_index
            ORDER BY date DESC
            LIMIT 10
        """).df()
        
        print(df)
        
        engine.close()
        print("\nTest completed successfully!")
    
    asyncio.run(test_macro_features())
