import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Comprehensive performance metrics for trading signals and anomaly detection.
    Implements Sharpe, Sortino, Information Ratio, Hit Rate, and Correlation Decay.
    """
    
    def __init__(self, config: dict):
        """
        Initialize performance metrics calculator.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.gpu_enabled = config.get('gpu', {}).get('enabled', False) and GPU_AVAILABLE
        self.risk_free_rate = config.get('validation', {}).get('risk_free_rate', 0.0)
        self.periods_per_year = config.get('validation', {}).get('periods_per_year', 8760)
        self.min_samples = config.get('validation', {}).get('min_samples', 30)
        
        logger.info(f"PerformanceMetrics initialized. GPU: {self.gpu_enabled}")
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, annualize: bool = True) -> float:
        """
        Calculate Sharpe ratio.
        
        Sharpe = (mean_return - risk_free_rate) / std_return
        
        Args:
            returns: Array of returns
            annualize: If True, annualize the ratio
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < self.min_samples:
            logger.warning(f"Insufficient samples for Sharpe: {len(returns)}")
            return 0.0
        
        if self.gpu_enabled:
            returns_gpu = cp.asarray(returns)
            mean_return = float(cp.mean(returns_gpu))
            std_return = float(cp.std(returns_gpu))
        else:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        sharpe = (mean_return - self.risk_free_rate) / std_return
        
        if annualize:
            sharpe = sharpe * np.sqrt(self.periods_per_year)
        
        return float(sharpe)
    
    def calculate_sortino_ratio(self, returns: np.ndarray, annualize: bool = True) -> float:
        """
        Calculate Sortino ratio (uses downside deviation instead of total std).
        
        Sortino = (mean_return - risk_free_rate) / downside_std
        
        Args:
            returns: Array of returns
            annualize: If True, annualize the ratio
            
        Returns:
            Sortino ratio
        """
        if len(returns) < self.min_samples:
            logger.warning(f"Insufficient samples for Sortino: {len(returns)}")
            return 0.0
        
        if self.gpu_enabled:
            returns_gpu = cp.asarray(returns)
            mean_return = float(cp.mean(returns_gpu))
            downside_returns = returns_gpu[returns_gpu < 0]
            
            if len(downside_returns) > 0:
                downside_std = float(cp.std(downside_returns))
            else:
                downside_std = 0.0
        else:
            mean_return = np.mean(returns)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return - self.risk_free_rate) / downside_std
        
        if annualize:
            sortino = sortino * np.sqrt(self.periods_per_year)
        
        return float(sortino)
    
    def calculate_information_ratio(self, signal_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """
        Calculate Information Ratio.
        
        IR = mean(active_return) / std(active_return)
        where active_return = signal_return - benchmark_return
        
        Args:
            signal_returns: Returns from signal-based strategy
            benchmark_returns: Benchmark returns (e.g., buy-and-hold)
            
        Returns:
            Information ratio
        """
        if len(signal_returns) != len(benchmark_returns):
            logger.error("Signal and benchmark returns must have same length")
            return 0.0
        
        if len(signal_returns) < self.min_samples:
            logger.warning(f"Insufficient samples for IR: {len(signal_returns)}")
            return 0.0
        
        if self.gpu_enabled:
            signal_gpu = cp.asarray(signal_returns)
            benchmark_gpu = cp.asarray(benchmark_returns)
            active_returns = signal_gpu - benchmark_gpu
            
            mean_active = float(cp.mean(active_returns))
            std_active = float(cp.std(active_returns))
        else:
            active_returns = signal_returns - benchmark_returns
            mean_active = np.mean(active_returns)
            std_active = np.std(active_returns)
        
        if std_active == 0:
            return 0.0
        
        ir = mean_active / std_active
        
        return float(ir)
    
    def calculate_hit_rate(self, predictions: np.ndarray, actuals: np.ndarray, threshold: float = 0.0) -> float:
        """
        Calculate hit rate (percentage of correct directional predictions).
        
        Args:
            predictions: Predicted price movements or signals
            actuals: Actual price movements
            threshold: Minimum movement to consider (filters noise)
            
        Returns:
            Hit rate as percentage (0-100)
        """
        if len(predictions) != len(actuals):
            logger.error("Predictions and actuals must have same length")
            return 0.0
        
        if len(predictions) < self.min_samples:
            logger.warning(f"Insufficient samples for hit rate: {len(predictions)}")
            return 0.0
        
        if self.gpu_enabled:
            predictions_gpu = cp.asarray(predictions)
            actuals_gpu = cp.asarray(actuals)
            
            pred_sign = cp.sign(predictions_gpu)
            actual_sign = cp.sign(actuals_gpu)
            
            mask = cp.abs(actuals_gpu) > threshold
            
            if cp.sum(mask) == 0:
                return 0.0
            
            correct = cp.sum((pred_sign == actual_sign) & mask)
            total = cp.sum(mask)
            
            hit_rate = float(correct / total) * 100
        else:
            pred_sign = np.sign(predictions)
            actual_sign = np.sign(actuals)
            
            mask = np.abs(actuals) > threshold
            
            if np.sum(mask) == 0:
                return 0.0
            
            correct = np.sum((pred_sign == actual_sign) & mask)
            total = np.sum(mask)
            
            hit_rate = (correct / total) * 100
        
        return float(hit_rate)
    
    def calculate_correlation_decay(
        self, 
        signal_severity: np.ndarray, 
        price_moves: np.ndarray,
        horizons: List[int] = [1, 4, 12, 24]
    ) -> Dict[str, float]:
        """
        Calculate correlation between signal severity and price moves at multiple horizons.
        Effective signals should show correlation decay over time.
        
        Args:
            signal_severity: Array of signal strengths (0-1)
            price_moves: Array of corresponding price movements
            horizons: Time horizons in hours to measure correlation
            
        Returns:
            Dictionary mapping horizon to correlation coefficient
        """
        if len(signal_severity) != len(price_moves):
            logger.error("Signal and price arrays must have same length")
            return {}
        
        correlations = {}
        
        for horizon in horizons:
            if horizon >= len(signal_severity):
                continue
            
            signal_subset = signal_severity[:-horizon]
            price_subset = price_moves[horizon:]
            
            if len(signal_subset) < self.min_samples:
                continue
            
            if self.gpu_enabled:
                signal_gpu = cp.asarray(signal_subset)
                price_gpu = cp.asarray(price_subset)
                
                corr_matrix = cp.corrcoef(signal_gpu, price_gpu)
                correlation = float(corr_matrix[0, 1])
            else:
                correlation = np.corrcoef(signal_subset, price_subset)[0, 1]
            
            if not np.isnan(correlation):
                correlations[f'{horizon}h'] = float(correlation)
        
        return correlations
    
    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve: Array of cumulative equity values
            
        Returns:
            Tuple of (max_drawdown_pct, peak_index, trough_index)
        """
        if len(equity_curve) < 2:
            return 0.0, 0, 0
        
        if self.gpu_enabled:
            equity_gpu = cp.asarray(equity_curve)
            cummax = cp.maximum.accumulate(equity_gpu)
            drawdowns = (equity_gpu - cummax) / cummax
            
            max_dd_idx = int(cp.argmin(drawdowns))
            max_dd = float(drawdowns[max_dd_idx]) * 100
            
            peak_idx = int(cp.argmax(cummax[:max_dd_idx + 1]))
        else:
            cummax = np.maximum.accumulate(equity_curve)
            drawdowns = (equity_curve - cummax) / cummax
            
            max_dd_idx = np.argmin(drawdowns)
            max_dd = drawdowns[max_dd_idx] * 100
            
            peak_idx = np.argmax(cummax[:max_dd_idx + 1])
        
        return float(max_dd), int(peak_idx), int(max_dd_idx)
    
    def calculate_calmar_ratio(self, returns: np.ndarray, equity_curve: np.ndarray) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Array of returns
            equity_curve: Equity curve for drawdown calculation
            
        Returns:
            Calmar ratio
        """
        if len(returns) < self.min_samples:
            return 0.0
        
        annual_return = np.mean(returns) * self.periods_per_year
        max_dd, _, _ = self.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0.0
        
        calmar = annual_return / abs(max_dd)
        
        return float(calmar)
    
    def calculate_win_loss_ratio(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate win/loss statistics.
        
        Args:
            returns: Array of trade returns
            
        Returns:
            Dictionary with win rate, avg win, avg loss, win/loss ratio
        """
        if len(returns) == 0:
            return {'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0, 'win_loss_ratio': 0.0}
        
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / len(returns) * 100
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        
        return {
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'win_loss_ratio': float(win_loss_ratio)
        }
    
    def calculate_comprehensive_metrics(
        self, 
        returns: np.ndarray,
        equity_curve: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Calculate all performance metrics.
        
        Args:
            returns: Array of strategy returns
            equity_curve: Cumulative equity curve
            benchmark_returns: Optional benchmark returns for IR calculation
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_returns': len(returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'calmar_ratio': self.calculate_calmar_ratio(returns, equity_curve),
            'max_drawdown': self.calculate_max_drawdown(equity_curve)[0],
            'total_return': float((equity_curve[-1] / equity_curve[0] - 1) * 100) if len(equity_curve) > 0 else 0.0,
            'win_loss': self.calculate_win_loss_ratio(returns)
        }
        
        if benchmark_returns is not None:
            metrics['information_ratio'] = self.calculate_information_ratio(returns, benchmark_returns)
        
        logger.info(f"Calculated comprehensive metrics: Sharpe={metrics['sharpe_ratio']:.2f}, "
                   f"Sortino={metrics['sortino_ratio']:.2f}, MaxDD={metrics['max_drawdown']:.2f}%")
        
        return metrics
    
    def calculate_rolling_metrics(
        self, 
        returns: np.ndarray,
        window: int = 100
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Array of returns
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        if len(returns) < window:
            logger.warning(f"Insufficient data for rolling metrics: {len(returns)} < {window}")
            return pd.DataFrame()
        
        rolling_data = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns[i - window:i]
            window_equity = np.cumsum(window_returns) + 1.0
            
            metrics = {
                'index': i,
                'sharpe': self.calculate_sharpe_ratio(window_returns, annualize=False),
                'sortino': self.calculate_sortino_ratio(window_returns, annualize=False),
                'max_dd': self.calculate_max_drawdown(window_equity)[0],
                'win_rate': self.calculate_win_loss_ratio(window_returns)['win_rate']
            }
            
            rolling_data.append(metrics)
        
        return pd.DataFrame(rolling_data)
    
    def export_metrics(self, metrics: Dict, output_path: str) -> None:
        """
        Export metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            output_path: Output file path
        """
        import json
        
        try:
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Metrics exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
