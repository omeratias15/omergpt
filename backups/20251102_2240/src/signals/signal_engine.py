"""
src/signals/signal_engine.py

Signal generation engine for OmerGPT.

Consumes anomaly events and recent features from DuckDB,
applies rule-based and ML-driven logic to generate actionable trading signals.

Outputs buy/sell/hold signals with confidence scores, stored in `signals` table.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("omerGPT.signals.engine")


class SignalEngine:
    def __init__(
        self,
        db_manager,
        thresholds: Optional[Dict] = None,
        atr_spike_threshold: float = 1.5   # 👈 הוספה כאן
    ):
        self.db = db_manager
        self.atr_spike_threshold = atr_spike_threshold  # 👈 שמירה של הערך

        self.thresholds = thresholds or {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "volatility_spike": 1.5,
            "momentum_threshold": 0.02,
            "anomaly_buy": 0.95,
            "anomaly_sell": 0.95,
            "min_confidence": 0.3
        }

        self.running = False
        self.stats = {
            "signals_generated": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "errors": 0
        }

        logger.info(f"SignalEngine initialized with thresholds: {self.thresholds} | ATR spike threshold={self.atr_spike_threshold}")


    async def load_latest_features(
        self,
        symbol: str,
        lookback_minutes: int = 30
    ) -> pd.DataFrame:
        """
        Load recent features for a given symbol.
        
        Args:
            symbol: Trading pair symbol
            lookback_minutes: Historical window in minutes
            
        Returns:
            DataFrame with recent features
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=lookback_minutes)
            
            # Query features from DB
            query = """
                SELECT symbol, ts_ms, 
                       return_1m, volatility_5m, volatility_15m, volatility_60m,
                       momentum_5m, momentum_15m, momentum_60m,
                       rsi_14, atr, atr_pct, macd, bb_up, bb_dn,
                       vol_ma, spread, ob_imbalance
                FROM features
                WHERE symbol = ? AND ts_ms BETWEEN ? AND ?
                ORDER BY ts_ms ASC
            """
            
            result = self.db.conn.execute(
                query,
                (symbol, start_time, end_time)
            )
            df = result.df()
            
            if len(df) == 0:
                logger.debug(f"No features for {symbol} in last {lookback_minutes}m")
                return pd.DataFrame()
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to load features for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    async def load_recent_anomalies(
        self,
        symbol: str,
        lookback_minutes: int = 30
    ) -> pd.DataFrame:
        """
        Load recent anomaly events for a given symbol.
        
        Args:
            symbol: Trading pair symbol
            lookback_minutes: Historical window in minutes
            
        Returns:
            DataFrame with recent anomalies
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=lookback_minutes)
            
            # Query anomaly events from DB
            query = """
                SELECT ts_ms, symbol, event_type, severity, confidence, meta
                FROM anomaly_events
                WHERE symbol = ? AND ts_ms BETWEEN ? AND ?
                ORDER BY ts_ms DESC
            """
            
            result = self.db.conn.execute(
                query,
                (symbol, start_time, end_time)
            )
            df = result.df()
            
            if len(df) == 0:
                logger.debug(f"No anomalies for {symbol} in last {lookback_minutes}m")
                return pd.DataFrame()
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to load anomalies for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def _compute_confidence(
        self,
        anomaly_score: float,
        rsi: float,
        volatility: float,
        momentum: float,
        has_volume: bool
    ) -> float:
        """
        Compute signal confidence score based on multiple factors.
        
        Args:
            anomaly_score: Anomaly detector score (0-1)
            rsi: RSI indicator (0-100)
            volatility: Current volatility
            momentum: Momentum indicator
            has_volume: Volume confirmation flag
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.3  # Base
        
        # Anomaly strength
        if anomaly_score > 0.95:
            confidence += 0.3
        elif anomaly_score > 0.80:
            confidence += 0.15
        
        # RSI extreme conditions
        if (rsi < self.thresholds["rsi_oversold"] or 
            rsi > self.thresholds["rsi_overbought"]):
            confidence += 0.2
        
        # Volatility spike
        if volatility > self.thresholds["volatility_spike"]:
            confidence += 0.1
        
        # Momentum strength
        if abs(momentum) > self.thresholds["momentum_threshold"]:
            confidence += 0.1
        
        # Volume confirmation
        if has_volume:
            confidence += 0.1
        
        return min(1.0, confidence)

    async def generate_signals(self) -> List[Dict]:
        """
        Main pipeline: combine anomalies and features to create actionable signals.
        
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        try:
            # Get all symbols
            symbols = await self.db.get_symbols()
            
            if not symbols:
                logger.warning("No symbols found in database")
                return signals
            
            # Process each symbol
            for symbol in symbols:
                # Load data
                features_df = await self.load_latest_features(symbol, lookback_minutes=30)
                anomalies_df = await self.load_recent_anomalies(symbol, lookback_minutes=30)
                
                if features_df.empty or anomalies_df.empty:
                    logger.debug(f"Skipping {symbol} - insufficient data")
                    continue
                
                # Get latest values
                latest_feature = features_df.iloc[-1]
                latest_anomaly = anomalies_df.iloc[0]  # Most recent
                
                # Extract values
                rsi = latest_feature.get("rsi_14", 50)
                volatility_15m = latest_feature.get("volatility_15m", 0)
                momentum_5m = latest_feature.get("momentum_5m", 0)
                anomaly_score = latest_anomaly.get("confidence", 0)
                
                # Check for volume (proxy: compare to moving average)
                vol_ma = latest_feature.get("vol_ma", 0)
                current_vol = latest_feature.get("volume", 0) if "volume" in latest_feature else 0
                has_volume = current_vol > vol_ma if vol_ma > 0 else False
                
                # Determine signal type
                signal_type = "HOLD"
                reason = ""
                
                # BUY signal logic
                if (anomaly_score > self.thresholds["anomaly_buy"] and 
                    rsi < self.thresholds["rsi_oversold"]):
                    signal_type = "BUY"
                    reason = f"RSI oversold ({rsi:.1f}) + anomaly spike ({anomaly_score:.3f})"
                
                elif (anomaly_score > self.thresholds["anomaly_buy"] and 
                      momentum_5m > self.thresholds["momentum_threshold"]):
                    signal_type = "BUY"
                    reason = f"Anomaly breakout + positive momentum ({momentum_5m:.4f})"
                
                # SELL signal logic
                elif (anomaly_score > self.thresholds["anomaly_sell"] and 
                      rsi > self.thresholds["rsi_overbought"]):
                    signal_type = "SELL"
                    reason = f"RSI overbought ({rsi:.1f}) + anomaly spike ({anomaly_score:.3f})"
                
                elif (anomaly_score > self.thresholds["anomaly_sell"] and 
                      momentum_5m < -self.thresholds["momentum_threshold"]):
                    signal_type = "SELL"
                    reason = f"Anomaly breakdown + negative momentum ({momentum_5m:.4f})"
                
                # Compute confidence
                confidence = self._compute_confidence(
                    anomaly_score=anomaly_score,
                    rsi=rsi,
                    volatility=volatility_15m,
                    momentum=momentum_5m,
                    has_volume=has_volume
                )
                
                # Skip low-confidence signals
                if confidence < self.thresholds["min_confidence"]:
                    continue
                
                # Create signal
                signal = {
                    "symbol": symbol,
                    "ts_ms": latest_feature["ts_ms"],
                    "signal_type": signal_type,
                    "confidence": confidence,
                    "reason": reason,
                    "anomaly_score": anomaly_score,
                    "rsi": rsi,
                    "momentum": momentum_5m,
                    "feature_snapshot": json.dumps({
                        "rsi_14": float(rsi),
                        "volatility_15m": float(volatility_15m),
                        "momentum_5m": float(momentum_5m),
                        "macd": float(latest_feature.get("macd", 0)),
                        "anomaly_score": float(anomaly_score)
                    }),
                    "status": "new"
                }
                
                signals.append(signal)
                
                # Log strong signals
                if confidence > 0.95:
                    logger.warning(
                        f"🚀 Strong {signal_type} signal for {symbol} | "
                        f"confidence={confidence:.3f} | reason={reason}"
                    )
                
                # Update stats
                self.stats["signals_generated"] += 1
                if signal_type == "BUY":
                    self.stats["buy_signals"] += 1
                elif signal_type == "SELL":
                    self.stats["sell_signals"] += 1
                else:
                    self.stats["hold_signals"] += 1
            
            logger.info(f"Generated {len(signals)} signals across {len(symbols)} symbols")
            return signals
        
        except Exception as e:
            logger.error(f"Signal generation failed: {e}", exc_info=True)
            self.stats["errors"] += 1
            return signals

    async def save_signals(self, signals: List[Dict]):
        """
        Persist generated signals to DuckDB.
        
        Args:
            signals: List of signal dictionaries
        """
        if not signals:
            return
        
        try:
            # Add signals table if not exists (extension to db_manager)
            self.db.conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    symbol TEXT NOT NULL,
                    ts_ms TIMESTAMP NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence DOUBLE NOT NULL,
                    reason TEXT,
                    anomaly_score DOUBLE,
                    rsi DOUBLE,
                    momentum DOUBLE,
                    feature_snapshot JSON,
                    status TEXT,
                    PRIMARY KEY (symbol, ts_ms, signal_type)
                )
            """)
            
            # Insert signals (upsert on duplicate)
            for signal in signals:
                self.db.conn.execute("""
                    INSERT INTO signals 
                    (symbol, ts_ms, signal_type, confidence, reason, 
                     anomaly_score, rsi, momentum, feature_snapshot, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (symbol, ts_ms, signal_type) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        reason = EXCLUDED.reason,
                        status = EXCLUDED.status
                """, (
                    signal["symbol"],
                    signal["ts_ms"],
                    signal["signal_type"],
                    signal["confidence"],
                    signal["reason"],
                    signal["anomaly_score"],
                    signal["rsi"],
                    signal["momentum"],
                    signal["feature_snapshot"],
                    signal["status"]
                ))
            
            logger.info(f"Saved {len(signals)} signals to database")
        
        except Exception as e:
            logger.error(f"Failed to save signals: {e}", exc_info=True)

    async def run(self, interval: int = 60):
        """
        Main signal generation loop.
        
        Args:
            interval: Loop interval in seconds
        """
        self.running = True
        logger.info("✅ Signal engine started")
        
        while self.running:
            try:
                signals = await self.generate_signals()
                await self.save_signals(signals)
                await asyncio.sleep(interval)
            
            except asyncio.CancelledError:
                logger.info("Signal engine cancelled")
                break
            
            except Exception as e:
                logger.error(f"Engine error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def stop(self):
        """Stop signal engine."""
        logger.info("Stopping signal engine...")
        self.running = False

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return self.stats.copy()


# ==================== DEMO ====================

async def run_demo():
    """
    Demo: Run signal engine for 5 minutes.
    """
    import sys
    sys.path.insert(0, "src")
    
    from storage.db_manager import DatabaseManager
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    print("\n=== Signal Engine Demo ===\n")
    
    # Initialize database
    db = DatabaseManager("data/market_data.duckdb")
    
    # Create engine
    engine = SignalEngine(
        db_manager=db,
        thresholds={
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "volatility_spike": 1.5,
            "momentum_threshold": 0.02,
            "anomaly_buy": 0.90,
            "anomaly_sell": 0.90,
            "min_confidence": 0.5
        }
    )
    
    # Start engine
    engine_task = asyncio.create_task(engine.run(interval=60))
    
    try:
        print("Running signal engine for 5 minutes...\n")
        
        # Monitor for 5 minutes
        for i in range(5):
            await asyncio.sleep(60)
            stats = engine.get_stats()
            print(f"[{(i+1)*60}s] BUY: {stats['buy_signals']} | "
                  f"SELL: {stats['sell_signals']} | "
                  f"HOLD: {stats['hold_signals']} | "
                  f"Total: {stats['signals_generated']}\n")
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    finally:
        await engine.stop()
        engine_task.cancel()
        
        final_stats = engine.get_stats()
        print(f"\n=== Final Stats ===")
        print(f"Total signals: {final_stats['signals_generated']}")
        print(f"BUY: {final_stats['buy_signals']}")
        print(f"SELL: {final_stats['sell_signals']}")
        print(f"HOLD: {final_stats['hold_signals']}")
        print(f"Errors: {final_stats['errors']}")
        
        db.close()
        print("\n=== Demo Complete ===\n")


if __name__ == "__main__":
    asyncio.run(run_demo())
