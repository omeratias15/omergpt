"""
src/backtesting/backtest_runner.py

Backtesting runner for OmerGPT.

Simulates end-to-end trading pipeline over historical data:
features, anomalies, signals, and PnL tracking.

Supports configurable time ranges, replay speeds, and performance metrics.
"""

import asyncio
import csv
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("omerGPT.backtesting")


class BacktestRunner:
    """
    Backtesting engine for OmerGPT trading signals.
    
    Features:
    - Replays historical data through signal pipeline
    - Tracks position management and PnL
    - Computes performance metrics (Sharpe, Drawdown, Win Rate)
    - Generates detailed trade logs
    """
    
    def __init__(
        self,
        db_manager,
        start_date: str,
        end_date: str,
        symbols: List[str],
        initial_balance: float = 10000.0,
        position_size_pct: float = 0.10,
        slippage: float = 0.001,
        commission: float = 0.001
    ):
        """
        Initialize backtest runner.
        
        Args:
            db_manager: DatabaseManager instance
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to backtest
            initial_balance: Starting capital in USD
            position_size_pct: % of balance to risk per trade
            slippage: Slippage factor (default 0.1%)
            commission: Commission per trade (default 0.1%)
        """
        self.db = db_manager
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.symbols = symbols
        
        # Capital management
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.timestamps = [self.start_date]
        
        # Trading
        self.positions: Dict[str, Dict] = {}  # {symbol: {entry_price, size, entry_time}}
        self.trades: List[Dict] = []  # [{symbol, entry_price, exit_price, size, pnl, duration}]
        
        # Parameters
        self.position_size_pct = position_size_pct
        self.slippage = slippage
        self.commission = commission
        
        # Statistics
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_ratio": 0.0,
            "total_pnl": 0.0,
            "final_balance": 0.0,
            "return_pct": 0.0
        }
        
        logger.info(
            f"BacktestRunner initialized: {start_date} to {end_date}, "
            f"{len(symbols)} symbols, ${initial_balance:.2f} initial"
        )

    async def load_historical_data(self, symbol: str) -> pd.DataFrame:
        """
        Load historical candle data for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert pandas Timestamp → integer milliseconds
            start_ts = int(self.start_date.timestamp() * 1000)
            end_ts = int(self.end_date.timestamp() * 1000)

            df = await self.db.get_candles_range(
    symbol,
    start_ts,
    end_ts
)

            
            if df.empty:
                logger.warning(f"No historical data for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Loaded {len(df)} candles for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load historical data for {symbol}: {e}")
            return pd.DataFrame()

    async def load_historical_features(self, symbol: str) -> pd.DataFrame:
        """
        Load historical features for a symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            DataFrame with computed features
        """
        try:
            query = """
                SELECT symbol, ts_ms, 
                       return_1m, volatility_15m, momentum_5m, rsi_14, macd, atr
                FROM features
                WHERE symbol = ? AND ts_ms BETWEEN ? AND ?
                ORDER BY ts_ms ASC
            """
            
            result = self.db.conn.execute(
                query,
                (symbol, self.start_date, self.end_date)
            )
            df = result.df()
            
            if df.empty:
                logger.warning(f"No historical features for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Loaded {len(df)} feature rows for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load features: {e}")
            return pd.DataFrame()

    def _get_price_at_time(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        price_type: str = "close"
    ) -> Optional[float]:
        """
        Get price for a symbol at specific timestamp.
        
        Args:
            symbol: Trading pair
            timestamp: Target timestamp
            price_type: open/high/low/close
            
        Returns:
            Price or None if not found
        """
        try:
            query = f"""
                SELECT {price_type} FROM candles
                WHERE symbol = ? AND ts_ms <= ?
                ORDER BY ts_ms DESC LIMIT 1
            """
            
            result = self.db.conn.execute(query, (symbol, timestamp)).fetchone()
            
            if result:
                price = float(result[0])
                # Apply slippage
                price *= (1 + self.slippage)
                return price
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None

    def _apply_slippage_and_commission(self, price: float, is_buy: bool) -> float:
        """
        Apply slippage and commission to price.
        
        Args:
            price: Base price
            is_buy: True for buy, False for sell
            
        Returns:
            Adjusted price
        """
        # Slippage
        if is_buy:
            price *= (1 + self.slippage)
        else:
            price *= (1 - self.slippage)
        
        # Commission
        price *= (1 + self.commission)
        
        return price

    def _execute_buy_signal(self, symbol: str, timestamp: pd.Timestamp, confidence: float):
        """Execute BUY signal."""
        # Skip if already in position
        if symbol in self.positions:
            logger.debug(f"Already in position for {symbol}, skipping BUY")
            return
        
        # Get entry price
        entry_price = self._get_price_at_time(symbol, timestamp, "close")
        if entry_price is None:
            logger.warning(f"No price data for {symbol} at {timestamp}")
            return
        
        # Apply slippage/commission
        entry_price = self._apply_slippage_and_commission(entry_price, is_buy=True)
        
        # Calculate position size
        risk_amount = self.balance * self.position_size_pct * confidence
        size = risk_amount / entry_price
        
        if size > 0 and risk_amount <= self.balance:
            self.positions[symbol] = {
                "entry_price": entry_price,
                "size": size,
                "entry_time": timestamp,
                "entry_balance": risk_amount
            }
            self.balance -= risk_amount
            
            logger.info(
                f"BUY {symbol} @ ${entry_price:.2f}, "
                f"size={size:.4f}, balance=${self.balance:.2f}"
            )
        else:
            logger.warning(f"Insufficient balance for BUY {symbol}")

    def _execute_sell_signal(self, symbol: str, timestamp: pd.Timestamp):
        """Execute SELL signal."""
        # Skip if no position
        if symbol not in self.positions:
            logger.debug(f"No position for {symbol}, skipping SELL")
            return
        
        position = self.positions[symbol]
        
        # Get exit price
        exit_price = self._get_price_at_time(symbol, timestamp, "close")
        if exit_price is None:
            logger.warning(f"No price data for {symbol} at {timestamp}")
            return
        
        # Apply slippage/commission
        exit_price = self._apply_slippage_and_commission(exit_price, is_buy=False)
        
        # Calculate PnL
        entry_price = position["entry_price"]
        size = position["size"]
        pnl_dollars = (exit_price - entry_price) * size
        pnl_pct = (exit_price - entry_price) / entry_price
        duration = timestamp - position["entry_time"]
        
        # Update balance
        self.balance += position["entry_balance"] + pnl_dollars
        
        # Record trade
        trade = {
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl_dollars": pnl_dollars,
            "pnl_pct": pnl_pct,
            "entry_time": position["entry_time"],
            "exit_time": timestamp,
            "duration_minutes": duration.total_seconds() / 60
        }
        self.trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(
            f"SELL {symbol} @ ${exit_price:.2f}, "
            f"PnL=${pnl_dollars:.2f} ({pnl_pct*100:.2f}%), "
            f"balance=${self.balance:.2f}"
        )

    async def generate_signals_for_period(
        self,
        symbol: str,
        features_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Generate trading signals for historical data.
        
        Args:
            symbol: Trading pair
            features_df: Historical features
            
        Returns:
            List of signals
        """
        signals = []
        
        for idx, row in features_df.iterrows():
            # Simple rule-based signal generation
            rsi = row.get("rsi_14", 50)
            momentum = row.get("momentum_5m", 0)
            volatility = row.get("volatility_15m", 0)
            
            signal_type = "HOLD"
            confidence = 0.0
            
            # BUY signal: RSI oversold + positive momentum
            if rsi < 30 and momentum > 0.01:
                signal_type = "BUY"
                confidence = min(0.95, 1.0 - (rsi / 30))
            
            # SELL signal: RSI overbought + negative momentum
            elif rsi > 70 and momentum < -0.01:
                signal_type = "SELL"
                confidence = min(0.95, (rsi - 70) / 30)
            
            if signal_type != "HOLD":
                signals.append({
                    "symbol": symbol,
                    "ts_ms": row["ts_ms"],
                    "signal_type": signal_type,
                    "confidence": confidence,
                    "rsi": rsi,
                    "momentum": momentum
                })
        
        logger.info(f"Generated {len(signals)} signals for {symbol}")
        return signals

    async def run_backtest(self, symbol: Optional[str] = None) -> Dict:
        """
        Run backtest for specified symbol(s).
        
        Args:
            symbol: Single symbol or None for all
            
        Returns:
            Backtest results dictionary
        """
        symbols_to_test = [symbol] if symbol else self.symbols
        
        logger.info(f"Starting backtest for {len(symbols_to_test)} symbol(s)")
        
        for sym in symbols_to_test:
            logger.info(f"\nProcessing {sym}...")
            
            # Load historical data
            candles_df = await self.load_historical_data(sym)
            features_df = await self.load_historical_features(sym)
            
            if candles_df.empty or features_df.empty:
                logger.warning(f"Skipping {sym} - insufficient data")
                continue
            
            # Generate signals
            signals = await self.generate_signals_for_period(sym, features_df)
            
            # Execute signals
            for signal in signals:
                timestamp = signal["ts_ms"]
                
                if signal["signal_type"] == "BUY":
                    self._execute_buy_signal(sym, timestamp, signal["confidence"])
                
                elif signal["signal_type"] == "SELL":
                    self._execute_sell_signal(sym, timestamp)
                
                # Track equity
                self._update_equity_curve(timestamp)
        
        # Close remaining positions
        self._close_all_positions()
        
        # Calculate statistics
        self._calculate_statistics()
        
        return self.stats

    def _update_equity_curve(self, timestamp: pd.Timestamp):
        """Update equity curve at given timestamp."""
        # Calculate current equity (balance + open positions value)
        current_equity = self.balance
        
        for sym, pos in self.positions.items():
            price = self._get_price_at_time(sym, timestamp, "close")
            if price:
                current_equity += pos["size"] * price
        
        self.equity_curve.append(current_equity)
        self.timestamps.append(timestamp)

    def _close_all_positions(self):
        """Close all remaining open positions."""
        logger.info("Closing all remaining positions...")
        
        for symbol in list(self.positions.keys()):
            self._execute_sell_signal(symbol, self.end_date)

    def _calculate_statistics(self):
        """Calculate backtest statistics."""
        if not self.trades:
            logger.warning("No trades executed")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic stats
        self.stats["total_trades"] = len(trades_df)
        self.stats["winning_trades"] = len(trades_df[trades_df["pnl_dollars"] > 0])
        self.stats["losing_trades"] = len(trades_df[trades_df["pnl_dollars"] < 0])
        
        # PnL stats
        winning_trades = trades_df[trades_df["pnl_dollars"] > 0]
        losing_trades = trades_df[trades_df["pnl_dollars"] < 0]
        
        self.stats["avg_win"] = winning_trades["pnl_dollars"].mean() if len(winning_trades) > 0 else 0.0
        self.stats["avg_loss"] = losing_trades["pnl_dollars"].mean() if len(losing_trades) > 0 else 0.0
        self.stats["largest_win"] = trades_df["pnl_dollars"].max()
        self.stats["largest_loss"] = trades_df["pnl_dollars"].min()
        
        # Ratios
        self.stats["win_ratio"] = self.stats["winning_trades"] / self.stats["total_trades"]
        self.stats["total_pnl"] = trades_df["pnl_dollars"].sum()
        self.stats["final_balance"] = self.balance + self.stats["total_pnl"]
        self.stats["return_pct"] = (
            (self.stats["final_balance"] - self.initial_balance) / self.initial_balance
        )
        
        # Sharpe Ratio
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        if len(returns) > 1:
            daily_returns = pd.Series(returns)
            self.stats["sharpe_ratio"] = (
                daily_returns.mean() / (daily_returns.std() + 1e-10) * np.sqrt(252)
            )
        
        # Max Drawdown
        equity_array = np.array(self.equity_curve)
        cummax = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - cummax) / cummax
        self.stats["max_drawdown"] = np.min(drawdown)

    def print_summary(self):
        """Print backtest summary."""
        logger.info("\n" + "="*70)
        logger.info("BACKTEST SUMMARY")
        logger.info("="*70)
        logger.info(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Final Balance: ${self.stats['final_balance']:,.2f}")
        logger.info(f"Total Return: {self.stats['return_pct']*100:.2f}%")
        logger.info(f"Total PnL: ${self.stats['total_pnl']:,.2f}")
        logger.info("="*70)
        logger.info(f"Total Trades: {self.stats['total_trades']}")
        logger.info(f"Winning Trades: {self.stats['winning_trades']}")
        logger.info(f"Losing Trades: {self.stats['losing_trades']}")
        logger.info(f"Win Ratio: {self.stats['win_ratio']*100:.2f}%")
        logger.info(f"Avg Win: ${self.stats['avg_win']:,.2f}")
        logger.info(f"Avg Loss: ${self.stats['avg_loss']:,.2f}")
        logger.info(f"Largest Win: ${self.stats['largest_win']:,.2f}")
        logger.info(f"Largest Loss: ${self.stats['largest_loss']:,.2f}")
        logger.info("="*70)
        logger.info(f"Sharpe Ratio: {self.stats['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {self.stats['max_drawdown']*100:.2f}%")
        logger.info("="*70 + "\n")

    def save_results(self, output_path: str = "backtest_results.csv"):
        """
        Save detailed backtest results to CSV.
        
        Args:
            output_path: Output file path
        """
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
            # Save trades
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(output_path, index=False)
            logger.info(f"✓ Saved {len(trades_df)} trades to {output_path}")
            
            # Save equity curve
            equity_path = output_path.replace(".csv", "_equity.csv")
            equity_df = pd.DataFrame({
                "timestamp": self.timestamps,
                "equity": self.equity_curve
            })
            equity_df.to_csv(equity_path, index=False)
            logger.info(f"✓ Saved equity curve to {equity_path}")
            
            # Save summary stats
            stats_path = output_path.replace(".csv", "_stats.json")
            with open(stats_path, "w") as f:
                json.dump(self.stats, f, indent=2, default=str)
            logger.info(f"✓ Saved stats to {stats_path}")
        
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


# ==================== MAIN EXECUTION ====================

async def run_backtest_demo():
    """Demo: Run backtest on sample data."""
    import sys
    sys.path.insert(0, "src")
    
    from storage.db_manager import DatabaseManager
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Initialize database
    db = DatabaseManager("data/market_data.duckdb")
    
    # Create backtest runner
    runner = BacktestRunner(
        db,
        start_date="2024-01-01",
        end_date="2024-12-31",
        symbols=["BTCUSDT", "ETHUSDT"],
        initial_balance=10000.0,
        position_size_pct=0.10
    )
    
    # Run backtest
    results = await runner.run_backtest()
    
    # Print summary
    runner.print_summary()
    
    # Save results
    runner.save_results("backtest_results.csv")
    
    db.close()


if __name__ == "__main__":
    asyncio.run(run_backtest_demo())
