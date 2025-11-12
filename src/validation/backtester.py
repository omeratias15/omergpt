import logging
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    Rolling walk-forward backtester with GPU acceleration.
    Implements expanding/sliding window validation with equity tracking.
    """
    
    def __init__(self, config: dict):
        """
        Initialize walk-forward backtester.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.gpu_enabled = config.get('gpu', {}).get('enabled', False) and GPU_AVAILABLE
        
        self.train_window = config.get('validation', {}).get('train_window_days', 60)
        self.test_window = config.get('validation', {}).get('test_window_days', 30)
        self.step_size = config.get('validation', {}).get('step_size_days', 7)
        
        self.initial_capital = config.get('validation', {}).get('initial_capital', 10000.0)
        self.position_size = config.get('validation', {}).get('position_size', 0.01)
        self.transaction_cost = config.get('validation', {}).get('transaction_cost', 0.001)
        
        self.metrics_calculator = PerformanceMetrics(config)
        
        self.results_dir = config.get('validation', {}).get('results_dir', 'reports/backtest')
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.random_seed = config.get('validation', {}).get('random_seed', 42)
        np.random.seed(self.random_seed)
        
        logger.info(f"WalkForwardBacktester initialized. Train: {self.train_window}d, "
                   f"Test: {self.test_window}d, Step: {self.step_size}d")
    
    def backtest(
        self,
        data: pd.DataFrame,
        signal_generator: Callable,
        retrain_func: Optional[Callable] = None,
        symbol: str = 'BTCUSDT'
    ) -> Dict:
        """
        Execute walk-forward backtest.
        
        Args:
            data: DataFrame with columns [timestamp, close, features...]
            signal_generator: Function that takes features and returns signals
            retrain_func: Optional function to retrain model on training data
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting walk-forward backtest for {symbol}, {len(data)} samples")
        
        if len(data) < self.train_window + self.test_window:
            logger.error("Insufficient data for walk-forward backtest")
            return {}
        
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        trades = []
        equity_curve = [self.initial_capital]
        signals_log = []
        
        start_idx = 0
        fold_num = 0
        
        while start_idx + self.train_window + self.test_window <= len(data):
            fold_num += 1
            
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window
            
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]
            
            logger.info(f"Fold {fold_num}: Train [{start_idx}:{train_end}], Test [{train_end}:{test_end}]")
            
            if retrain_func is not None:
                try:
                    retrain_func(train_data)
                    logger.info(f"Model retrained on fold {fold_num}")
                except Exception as e:
                    logger.error(f"Retraining failed on fold {fold_num}: {e}")
            
            fold_trades, fold_equity, fold_signals = self._backtest_fold(
                test_data, 
                signal_generator,
                equity_curve[-1]
            )
            
            trades.extend(fold_trades)
            equity_curve.extend(fold_equity)
            signals_log.extend(fold_signals)
            
            start_idx += self.step_size
        
        if len(trades) == 0:
            logger.warning("No trades executed during backtest")
            return {}
        
        results = self._compile_results(
            trades, 
            equity_curve, 
            signals_log,
            data,
            symbol
        )
        
        self._export_results(results, symbol)
        
        logger.info(f"Backtest completed: {len(trades)} trades, "
                   f"Final equity: ${equity_curve[-1]:.2f}, "
                   f"Total return: {results['total_return']:.2f}%")
        
        return results
    
    def _backtest_fold(
        self,
        test_data: pd.DataFrame,
        signal_generator: Callable,
        starting_equity: float
    ) -> Tuple[List[Dict], List[float], List[Dict]]:
        """
        Backtest a single fold.
        
        Args:
            test_data: Test period DataFrame
            signal_generator: Signal generation function
            starting_equity: Equity at start of fold
            
        Returns:
            Tuple of (trades, equity_curve, signals_log)
        """
        trades = []
        equity = starting_equity
        equity_curve = []
        signals_log = []
        
        position = 0.0
        entry_price = 0.0
        
        for idx, row in test_data.iterrows():
            try:
                signal = signal_generator(row)
                
                signals_log.append({
                    'timestamp': row['timestamp'],
                    'signal': signal,
                    'close': row['close']
                })
                
                if signal > 0 and position == 0:
                    position_value = equity * self.position_size
                    shares = position_value / row['close']
                    entry_price = row['close']
                    position = shares
                    
                    cost = position_value * self.transaction_cost
                    equity -= cost
                    
                    trades.append({
                        'timestamp': row['timestamp'],
                        'type': 'buy',
                        'price': entry_price,
                        'shares': shares,
                        'cost': cost
                    })
                
                elif signal < 0 and position > 0:
                    exit_price = row['close']
                    position_value = position * exit_price
                    
                    pnl = (exit_price - entry_price) * position
                    cost = position_value * self.transaction_cost
                    
                    equity += pnl - cost
                    
                    trades.append({
                        'timestamp': row['timestamp'],
                        'type': 'sell',
                        'price': exit_price,
                        'shares': position,
                        'pnl': pnl,
                        'cost': cost,
                        'return': (exit_price - entry_price) / entry_price
                    })
                    
                    position = 0.0
                    entry_price = 0.0
                
                current_value = equity + (position * row['close'] if position > 0 else 0)
                equity_curve.append(current_value)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        if position > 0:
            exit_price = test_data.iloc[-1]['close']
            pnl = (exit_price - entry_price) * position
            equity += pnl
            
            trades.append({
                'timestamp': test_data.iloc[-1]['timestamp'],
                'type': 'close',
                'price': exit_price,
                'shares': position,
                'pnl': pnl,
                'return': (exit_price - entry_price) / entry_price
            })
        
        return trades, equity_curve, signals_log
    
    def _compile_results(
        self,
        trades: List[Dict],
        equity_curve: List[float],
        signals_log: List[Dict],
        data: pd.DataFrame,
        symbol: str
    ) -> Dict:
        """
        Compile comprehensive backtest results.
        
        Args:
            trades: List of trade dictionaries
            equity_curve: Equity progression
            signals_log: Signal history
            data: Full data DataFrame
            symbol: Trading symbol
            
        Returns:
            Results dictionary
        """
        trade_returns = [t['return'] for t in trades if 'return' in t]
        
        if len(trade_returns) == 0:
            trade_returns = [0.0]
        
        equity_array = np.array(equity_curve)
        returns_array = np.array(trade_returns)
        
        benchmark_returns = data['close'].pct_change().dropna().values
        if len(benchmark_returns) > len(returns_array):
            benchmark_returns = benchmark_returns[:len(returns_array)]
        
        metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            returns_array,
            equity_array,
            benchmark_returns
        )
        
        results = {
            'symbol': symbol,
            'backtest_start': data.iloc[0]['timestamp'],
            'backtest_end': data.iloc[-1]['timestamp'],
            'initial_capital': self.initial_capital,
            'final_equity': equity_curve[-1],
            'total_return': ((equity_curve[-1] / self.initial_capital) - 1) * 100,
            'num_trades': len(trades),
            'num_signals': len(signals_log),
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve,
            'signals': signals_log
        }
        
        return results
    
    def _export_results(self, results: Dict, symbol: str) -> None:
        """
        Export backtest results to CSV and DuckDB.
        
        Args:
            results: Results dictionary
            symbol: Trading symbol
        """
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        trades_df = pd.DataFrame(results['trades'])
        equity_df = pd.DataFrame({
            'index': range(len(results['equity_curve'])),
            'equity': results['equity_curve']
        })
        signals_df = pd.DataFrame(results['signals'])
        
        trades_path = os.path.join(self.results_dir, f'{symbol}_trades_{timestamp_str}.csv')
        equity_path = os.path.join(self.results_dir, f'{symbol}_equity_{timestamp_str}.csv')
        signals_path = os.path.join(self.results_dir, f'{symbol}_signals_{timestamp_str}.csv')
        
        trades_df.to_csv(trades_path, index=False)
        equity_df.to_csv(equity_path, index=False)
        signals_df.to_csv(signals_path, index=False)
        
        logger.info(f"Results exported to {self.results_dir}")
        
        try:
            self._export_to_duckdb(results, symbol)
        except Exception as e:
            logger.warning(f"DuckDB export failed: {e}")
    
    def _export_to_duckdb(self, results: Dict, symbol: str) -> None:
        """
        Export results to DuckDB.
        
        Args:
            results: Results dictionary
            symbol: Trading symbol
        """
        import duckdb
        
        db_path = self.config.get('storage', {}).get('duckdb_path', 'data/market_data.duckdb')
        
        conn = duckdb.connect(db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                initial_capital DOUBLE,
                final_equity DOUBLE,
                total_return DOUBLE,
                num_trades INTEGER,
                sharpe_ratio DOUBLE,
                sortino_ratio DOUBLE,
                max_drawdown DOUBLE,
                win_rate DOUBLE
            )
        """)
        
        conn.execute("""
            INSERT INTO backtest_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            symbol,
            results['initial_capital'],
            results['final_equity'],
            results['total_return'],
            results['num_trades'],
            results['metrics']['sharpe_ratio'],
            results['metrics']['sortino_ratio'],
            results['metrics']['max_drawdown'],
            results['metrics']['win_loss']['win_rate']
        ))
        
        trades_df = pd.DataFrame(results['trades'])
        conn.register('trades_temp', trades_df)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_trades (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                type VARCHAR,
                price DOUBLE,
                shares DOUBLE,
                pnl DOUBLE,
                return DOUBLE
            )
        """)
        
        conn.execute("""
            INSERT INTO backtest_trades 
            SELECT timestamp, ? as symbol, type, price, shares, 
                   COALESCE(pnl, 0) as pnl, COALESCE(return, 0) as return
            FROM trades_temp
        """, [symbol])
        
        conn.close()
        
        logger.info(f"Results saved to DuckDB: {db_path}")
    
    def monte_carlo_simulation(
        self,
        returns: np.ndarray,
        num_simulations: int = 1000,
        num_periods: int = 252
    ) -> Dict:
        """
        Run Monte Carlo simulation on trade returns.
        
        Args:
            returns: Historical trade returns
            num_simulations: Number of simulations to run
            num_periods: Number of periods to simulate
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running Monte Carlo: {num_simulations} sims, {num_periods} periods")
        
        if len(returns) < 10:
            logger.warning("Insufficient returns for Monte Carlo")
            return {}
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if self.gpu_enabled:
            cp.random.seed(self.random_seed)
            simulated_returns = cp.random.normal(mean_return, std_return, (num_simulations, num_periods))
            simulated_equity = cp.cumprod(1 + simulated_returns, axis=1) * self.initial_capital
            
            final_values = simulated_equity[:, -1]
            percentiles = cp.percentile(final_values, [5, 25, 50, 75, 95])
            
            results = {
                'mean_final_value': float(cp.mean(final_values)),
                'std_final_value': float(cp.std(final_values)),
                'percentile_5': float(percentiles[0]),
                'percentile_25': float(percentiles[1]),
                'median': float(percentiles[2]),
                'percentile_75': float(percentiles[3]),
                'percentile_95': float(percentiles[4]),
                'probability_profit': float(cp.mean(final_values > self.initial_capital))
            }
        else:
            np.random.seed(self.random_seed)
            simulated_returns = np.random.normal(mean_return, std_return, (num_simulations, num_periods))
            simulated_equity = np.cumprod(1 + simulated_returns, axis=1) * self.initial_capital
            
            final_values = simulated_equity[:, -1]
            percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
            
            results = {
                'mean_final_value': float(np.mean(final_values)),
                'std_final_value': float(np.std(final_values)),
                'percentile_5': float(percentiles[0]),
                'percentile_25': float(percentiles[1]),
                'median': float(percentiles[2]),
                'percentile_75': float(percentiles[3]),
                'percentile_95': float(percentiles[4]),
                'probability_profit': float(np.mean(final_values > self.initial_capital))
            }
        
        logger.info(f"Monte Carlo complete: P(profit)={results['probability_profit']:.2%}")
        
        return results
