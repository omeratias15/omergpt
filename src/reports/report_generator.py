"""
src/reports/report_generator.py
Comprehensive performance report generator for omerGPT.
Creates HTML and PDF reports with PnL analysis, Sharpe ratio, drawdown metrics,
sentiment vs macro correlation, signal performance, and adaptive learning statistics.
Uses Jinja2 templates and ReportLab for professional-quality output.
"""
import asyncio
import base64
import io
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Template = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image as RLImage
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from storage.db_manager import DatabaseManager

logger = logging.getLogger("omerGPT.reports.generator")


class PerformanceMetrics:
    """
    Calculate trading performance metrics from historical data.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        logger.info("PerformanceMetrics initialized")
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            prices: Price series
        
        Returns:
            Returns series
        """
        return prices.pct_change().fillna(0.0)
    
    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns.
        
        Args:
            returns: Returns series
        
        Returns:
            Cumulative returns series
        """
        return (1 + returns).cumprod() - 1
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 365
    ) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Returns series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
        
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = (
            excess_returns.mean() / excess_returns.std()
        ) * np.sqrt(periods_per_year)
        
        return float(sharpe)
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 365
    ) -> float:
        """
        Calculate annualized Sortino ratio.
        
        Args:
            returns: Returns series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
        
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = (
            excess_returns.mean() / downside_returns.std()
        ) * np.sqrt(periods_per_year)
        
        return float(sortino)
    
    def calculate_max_drawdown(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            returns: Returns series
        
        Returns:
            Dictionary with drawdown metrics
        """
        if len(returns) < 2:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_duration": 0,
                "current_drawdown": 0.0,
            }
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = float(drawdown.min())
        
        # Calculate drawdown duration
        is_in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for in_dd in is_in_drawdown:
            if in_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        current_dd = float(drawdown.iloc[-1])
        
        return {
            "max_drawdown": max_dd,
            "max_drawdown_duration": max_dd_duration,
            "current_drawdown": current_dd,
        }
    
    def calculate_win_rate(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate win rate and related metrics.
        
        Args:
            returns: Returns series
        
        Returns:
            Dictionary with win rate metrics
        """
        if len(returns) == 0:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
            }
        
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0.0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0.0
        
        total_wins = winning_trades.sum() if len(winning_trades) > 0 else 0.0
        total_losses = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0.0
        
        profit_factor = (
            total_wins / total_losses if total_losses > 0 else 0.0
        )
        
        return {
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
        }
    
    def calculate_correlation(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> float:
        """
        Calculate correlation between two series.
        
        Args:
            series1: First series
            series2: Second series
        
        Returns:
            Correlation coefficient
        """
        if len(series1) < 2 or len(series2) < 2:
            return 0.0
        
        # Align series
        combined = pd.DataFrame({
            "s1": series1,
            "s2": series2
        }).dropna()
        
        if len(combined) < 2:
            return 0.0
        
        corr = combined["s1"].corr(combined["s2"])
        
        return float(corr) if not np.isnan(corr) else 0.0


class DataAggregator:
    """
    Aggregate data from DuckDB for report generation.
    """
    
    def __init__(self, db_manager: DBManager):
        """
        Initialize data aggregator.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
        logger.info("DataAggregator initialized")
    
    async def get_performance_data(
        self,
        symbol: str = "BTCUSDT",
        lookback_days: int = 30,
    ) -> Dict:
        """
        Get performance data for report.
        
        Args:
            symbol: Trading symbol
            lookback_days: Days of historical data
        
        Returns:
            Dictionary with performance data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        logger.info(
            f"Fetching performance data: {symbol}, "
            f"{start_time.date()} to {end_time.date()}"
        )
        
        # Get candles
        candles = await self.db.query(
            """
            SELECT timestamp, close, volume
            FROM candles
            WHERE symbol = ?
            AND timestamp >= ?
            AND timestamp <= ?
            ORDER BY timestamp
            """,
            (symbol, int(start_time.timestamp() * 1000), int(end_time.timestamp() * 1000))
        )
        
        if not candles:
            logger.warning("No candles data found")
            return {}
        
        df = pd.DataFrame(candles)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate metrics
        metrics_calc = PerformanceMetrics()
        
        returns = metrics_calc.calculate_returns(df['close'])
        cumulative_returns = metrics_calc.calculate_cumulative_returns(returns)
        
        sharpe = metrics_calc.calculate_sharpe_ratio(returns)
        sortino = metrics_calc.calculate_sortino_ratio(returns)
        drawdown_metrics = metrics_calc.calculate_max_drawdown(returns)
        win_metrics = metrics_calc.calculate_win_rate(returns)
        
        # Get sentiment data
        sentiment = await self.db.query(
            """
            SELECT date, sentiment_index
            FROM sentiment_daily
            WHERE date >= ?
            AND date <= ?
            ORDER BY date
            """,
            (start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d'))
        )
        
        sentiment_df = pd.DataFrame(sentiment) if sentiment else pd.DataFrame()
        
        # Get macro data
        macro = await self.db.query(
            """
            SELECT date, risk_index
            FROM macro_features
            WHERE date >= ?
            AND date <= ?
            ORDER BY date
            """,
            (start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d'))
        )
        
        macro_df = pd.DataFrame(macro) if macro else pd.DataFrame()
        
        # Calculate correlations
        sentiment_corr = 0.0
        macro_corr = 0.0
        
        if not sentiment_df.empty:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            df['date'] = df['datetime'].dt.date
            df['date'] = pd.to_datetime(df['date'])
            
            daily_returns = df.groupby('date')['close'].last().pct_change()
            sentiment_series = sentiment_df.set_index('date')['sentiment_index']
            
            sentiment_corr = metrics_calc.calculate_correlation(
                daily_returns,
                sentiment_series
            )
        
        if not macro_df.empty:
            macro_df['date'] = pd.to_datetime(macro_df['date'])
            
            if 'date' in df.columns:
                daily_returns = df.groupby('date')['close'].last().pct_change()
                macro_series = macro_df.set_index('date')['risk_index']
                
                macro_corr = metrics_calc.calculate_correlation(
                    daily_returns,
                    macro_series
                )
        
        # Compile results
        performance = {
            "symbol": symbol,
            "period": {
                "start": start_time.strftime('%Y-%m-%d'),
                "end": end_time.strftime('%Y-%m-%d'),
                "days": lookback_days,
            },
            "returns": {
                "total": float(cumulative_returns.iloc[-1]),
                "mean_daily": float(returns.mean()),
                "std_daily": float(returns.std()),
                "annualized": float(returns.mean() * 365),
            },
            "risk_metrics": {
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                **drawdown_metrics,
            },
            "trade_metrics": win_metrics,
            "correlations": {
                "sentiment": sentiment_corr,
                "macro_risk": macro_corr,
            },
            "price_data": {
                "initial": float(df['close'].iloc[0]),
                "final": float(df['close'].iloc[-1]),
                "high": float(df['close'].max()),
                "low": float(df['close'].min()),
            },
        }
        
        logger.info("Performance data aggregation complete")
        
        return performance
    
    async def get_signal_stats(
        self,
        lookback_days: int = 30
    ) -> Dict:
        """
        Get signal generation statistics.
        
        Args:
            lookback_days: Days of historical data
        
        Returns:
            Dictionary with signal stats
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        # Try to get signal data (table may not exist yet)
        try:
            signals = await self.db.query(
                """
                SELECT timestamp, signal_type, confidence
                FROM signals
                WHERE timestamp >= ?
                AND timestamp <= ?
                ORDER BY timestamp
                """,
                (int(start_time.timestamp() * 1000), int(end_time.timestamp() * 1000))
            )
            
            if not signals:
                return {
                    "total_signals": 0,
                    "by_type": {},
                    "avg_confidence": 0.0,
                }
            
            signals_df = pd.DataFrame(signals)
            
            by_type = signals_df['signal_type'].value_counts().to_dict()
            avg_confidence = float(signals_df['confidence'].mean())
            
            return {
                "total_signals": len(signals_df),
                "by_type": by_type,
                "avg_confidence": avg_confidence,
            }
        
        except Exception as e:
            logger.warning(f"Could not fetch signal stats: {e}")
            return {
                "total_signals": 0,
                "by_type": {},
                "avg_confidence": 0.0,
            }
    
    async def get_adaptive_stats(self) -> Dict:
        """
        Get adaptive learning statistics.
        
        Returns:
            Dictionary with adaptive learning stats
        """
        try:
            # Try to load adaptive log
            log_path = "adaptive_log.json"
            
            if not os.path.exists(log_path):
                return {
                    "parameter_changes": 0,
                    "latest_params": {},
                }
            
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            changes = log_data.get("changes", [])
            
            if not changes:
                return {
                    "parameter_changes": 0,
                    "latest_params": {},
                }
            
            latest = changes[-1]
            
            return {
                "parameter_changes": len(changes),
                "latest_params": latest.get("parameters", {}),
                "last_update": latest.get("timestamp"),
            }
        
        except Exception as e:
            logger.warning(f"Could not fetch adaptive stats: {e}")
            return {
                "parameter_changes": 0,
                "latest_params": {},
            }


class HTMLReportGenerator:
    """
    Generate HTML reports using Jinja2 templates.
    """
    
    def __init__(self):
        """Initialize HTML report generator."""
        self.template = self._create_template()
        logger.info("HTMLReportGenerator initialized")
    
    def _create_template(self) -> str:
        """
        Create HTML template.
        
        Returns:
            HTML template string
        """
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>omerGPT Performance Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card.positive {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        .metric-card.negative {
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        }
        .metric-title {
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-good {
            background-color: #2ecc71;
            color: white;
        }
        .status-warning {
            background-color: #f39c12;
            color: white;
        }
        .status-danger {
            background-color: #e74c3c;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>omerGPT Performance Report</h1>
            <p>Generated: {{ timestamp }}</p>
            <p>Period: {{ period.start }} to {{ period.end }} ({{ period.days }} days)</p>
        </div>

        <h2>Performance Overview</h2>
        <div class="metrics-grid">
            <div class="metric-card {% if returns.total > 0 %}positive{% else %}negative{% endif %}">
                <div class="metric-title">Total Return</div>
                <div class="metric-value">{{ "%.2f"|format(returns.total * 100) }}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Sharpe Ratio</div>
                <div class="metric-value">{{ "%.2f"|format(risk_metrics.sharpe_ratio) }}</div>
            </div>
            <div class="metric-card {% if risk_metrics.max_drawdown > -0.1 %}positive{% else %}negative{% endif %}">
                <div class="metric-title">Max Drawdown</div>
                <div class="metric-value">{{ "%.2f"|format(risk_metrics.max_drawdown * 100) }}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Win Rate</div>
                <div class="metric-value">{{ "%.1f"|format(trade_metrics.win_rate * 100) }}%</div>
            </div>
        </div>

        <h2>Risk Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{{ "%.3f"|format(risk_metrics.sharpe_ratio) }}</td>
                <td>
                    {% if risk_metrics.sharpe_ratio > 1.5 %}
                    <span class="status-badge status-good">Excellent</span>
                    {% elif risk_metrics.sharpe_ratio > 1.0 %}
                    <span class="status-badge status-warning">Good</span>
                    {% else %}
                    <span class="status-badge status-danger">Poor</span>
                    {% endif %}
                </td>
            </tr>
            <tr>
                <td>Sortino Ratio</td>
                <td>{{ "%.3f"|format(risk_metrics.sortino_ratio) }}</td>
                <td>
                    {% if risk_metrics.sortino_ratio > 2.0 %}
                    <span class="status-badge status-good">Excellent</span>
                    {% elif risk_metrics.sortino_ratio > 1.0 %}
                    <span class="status-badge status-warning">Good</span>
                    {% else %}
                    <span class="status-badge status-danger">Poor</span>
                    {% endif %}
                </td>
            </tr>
            <tr>
                <td>Maximum Drawdown</td>
                <td>{{ "%.2f"|format(risk_metrics.max_drawdown * 100) }}%</td>
                <td>
                    {% if risk_metrics.max_drawdown > -0.1 %}
                    <span class="status-badge status-good">Low</span>
                    {% elif risk_metrics.max_drawdown > -0.2 %}
                    <span class="status-badge status-warning">Moderate</span>
                    {% else %}
                    <span class="status-badge status-danger">High</span>
                    {% endif %}
                </td>
            </tr>
            <tr>
                <td>Current Drawdown</td>
                <td>{{ "%.2f"|format(risk_metrics.current_drawdown * 100) }}%</td>
                <td>-</td>
            </tr>
        </table>

        <h2>Trade Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Win Rate</td>
                <td>{{ "%.2f"|format(trade_metrics.win_rate * 100) }}%</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td>{{ "%.4f"|format(trade_metrics.avg_win * 100) }}%</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td>{{ "%.4f"|format(trade_metrics.avg_loss * 100) }}%</td>
            </tr>
            <tr>
                <td>Profit Factor</td>
                <td>{{ "%.2f"|format(trade_metrics.profit_factor) }}</td>
            </tr>
        </table>

        <h2>Sentiment & Macro Correlation</h2>
        <table>
            <tr>
                <th>Factor</th>
                <th>Correlation</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>Sentiment Index</td>
                <td>{{ "%.3f"|format(correlations.sentiment) }}</td>
                <td>
                    {% if correlations.sentiment|abs > 0.5 %}Strong
                    {% elif correlations.sentiment|abs > 0.3 %}Moderate
                    {% else %}Weak{% endif %}
                    {% if correlations.sentiment > 0 %}Positive{% else %}Negative{% endif %}
                </td>
            </tr>
            <tr>
                <td>Macro Risk Index</td>
                <td>{{ "%.3f"|format(correlations.macro_risk) }}</td>
                <td>
                    {% if correlations.macro_risk|abs > 0.5 %}Strong
                    {% elif correlations.macro_risk|abs > 0.3 %}Moderate
                    {% else %}Weak{% endif %}
                    {% if correlations.macro_risk > 0 %}Positive{% else %}Negative{% endif %}
                </td>
            </tr>
        </table>

        {% if signal_stats.total_signals > 0 %}
        <h2>Signal Performance</h2>
        <p>Total Signals Generated: <strong>{{ signal_stats.total_signals }}</strong></p>
        <p>Average Confidence: <strong>{{ "%.2f"|format(signal_stats.avg_confidence * 100) }}%</strong></p>
        
        {% if signal_stats.by_type %}
        <table>
            <tr>
                <th>Signal Type</th>
                <th>Count</th>
            </tr>
            {% for signal_type, count in signal_stats.by_type.items() %}
            <tr>
                <td>{{ signal_type }}</td>
                <td>{{ count }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        {% endif %}

        {% if adaptive_stats.parameter_changes > 0 %}
        <h2>Adaptive Learning</h2>
        <p>Parameter Changes: <strong>{{ adaptive_stats.parameter_changes }}</strong></p>
        {% if adaptive_stats.last_update %}
        <p>Last Update: <strong>{{ adaptive_stats.last_update }}</strong></p>
        {% endif %}
        
        {% if adaptive_stats.latest_params %}
        <table>
            <tr>
                <th>Parameter</th>
                <th>Current Value</th>
            </tr>
            {% for param, value in adaptive_stats.latest_params.items() %}
            <tr>
                <td>{{ param }}</td>
                <td>{{ "%.3f"|format(value) }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        {% endif %}

        <div class="footer">
            <p>omerGPT - GPU-Accelerated Autonomous Crypto Intelligence Platform</p>
            <p>© 2025 - Generated automatically</p>
        </div>
    </div>
</body>
</html>
"""
    
    def generate(self, data: Dict) -> str:
        """
        Generate HTML report from data.
        
        Args:
            data: Report data dictionary
        
        Returns:
            HTML string
        """
        if not JINJA2_AVAILABLE:
            logger.warning("Jinja2 not available, using simple HTML")
            return self._generate_simple_html(data)
        
        try:
            template = Template(self.template)
            html = template.render(**data)
            return html
        except Exception as e:
            logger.error(f"Error generating HTML: {e}")
            return self._generate_simple_html(data)
    
    def _generate_simple_html(self, data: Dict) -> str:
        """Generate simple HTML without Jinja2."""
        perf = data.get("performance", {})
        
        html = f"""
        <html>
        <head><title>omerGPT Report</title></head>
        <body>
        <h1>omerGPT Performance Report</h1>
        <p>Generated: {data.get('timestamp')}</p>
        <h2>Performance Metrics</h2>
        <p>Total Return: {perf.get('returns', {}).get('total', 0) * 100:.2f}%</p>
        <p>Sharpe Ratio: {perf.get('risk_metrics', {}).get('sharpe_ratio', 0):.2f}</p>
        <p>Max Drawdown: {perf.get('risk_metrics', {}).get('max_drawdown', 0) * 100:.2f}%</p>
        </body>
        </html>
        """
        return html


class PDFReportGenerator:
    """
    Generate PDF reports using ReportLab.
    """
    
    def __init__(self):
        """Initialize PDF report generator."""
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available, PDF generation disabled")
        
        self.styles = getSampleStyleSheet() if REPORTLAB_AVAILABLE else None
        logger.info("PDFReportGenerator initialized")
    
    def generate(self, data: Dict, output_path: str):
        """
        Generate PDF report.
        
        Args:
            data: Report data dictionary
            output_path: Output PDF file path
        """
        if not REPORTLAB_AVAILABLE:
            logger.error("ReportLab required for PDF generation")
            return
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        
        story.append(Paragraph("omerGPT Performance Report", title_style))
        story.append(Spacer(1, 12))
        
        # Metadata
        normal_style = self.styles['Normal']
        
        timestamp = data.get('timestamp', 'N/A')
        story.append(Paragraph(f"<b>Generated:</b> {timestamp}", normal_style))
        
        perf = data.get('performance', {})
        period = perf.get('period', {})
        
        story.append(Paragraph(
            f"<b>Period:</b> {period.get('start')} to {period.get('end')} "
            f"({period.get('days')} days)",
            normal_style
        ))
        
        story.append(Spacer(1, 20))
        
        # Performance Overview
        heading_style = self.styles['Heading2']
        story.append(Paragraph("Performance Overview", heading_style))
        story.append(Spacer(1, 12))
        
        returns = perf.get('returns', {})
        risk_metrics = perf.get('risk_metrics', {})
        trade_metrics = perf.get('trade_metrics', {})
        
        overview_data = [
            ['Metric', 'Value'],
            ['Total Return', f"{returns.get('total', 0) * 100:.2f}%"],
            ['Annualized Return', f"{returns.get('annualized', 0) * 100:.2f}%"],
            ['Sharpe Ratio', f"{risk_metrics.get('sharpe_ratio', 0):.3f}"],
            ['Sortino Ratio', f"{risk_metrics.get('sortino_ratio', 0):.3f}"],
            ['Max Drawdown', f"{risk_metrics.get('max_drawdown', 0) * 100:.2f}%"],
            ['Win Rate', f"{trade_metrics.get('win_rate', 0) * 100:.2f}%"],
        ]
        
        overview_table = Table(overview_data, colWidths=[3*inch, 2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        # Correlations
        story.append(Paragraph("Sentiment & Macro Correlation", heading_style))
        story.append(Spacer(1, 12))
        
        correlations = perf.get('correlations', {})
        
        corr_data = [
            ['Factor', 'Correlation'],
            ['Sentiment Index', f"{correlations.get('sentiment', 0):.3f}"],
            ['Macro Risk Index', f"{correlations.get('macro_risk', 0):.3f}"],
        ]
        
        corr_table = Table(corr_data, colWidths=[3*inch, 2*inch])
        corr_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(corr_table)
        story.append(Spacer(1, 20))
        
        # Signal stats if available
        signal_stats = data.get('signal_stats', {})
        
        if signal_stats.get('total_signals', 0) > 0:
            story.append(Paragraph("Signal Performance", heading_style))
            story.append(Spacer(1, 12))
            
            story.append(Paragraph(
                f"Total Signals: {signal_stats['total_signals']}",
                normal_style
            ))
            story.append(Paragraph(
                f"Avg Confidence: {signal_stats.get('avg_confidence', 0) * 100:.2f}%",
                normal_style
            ))
            story.append(Spacer(1, 20))
        
        # Adaptive learning stats
        adaptive_stats = data.get('adaptive_stats', {})
        
        if adaptive_stats.get('parameter_changes', 0) > 0:
            story.append(Paragraph("Adaptive Learning", heading_style))
            story.append(Spacer(1, 12))
            
            story.append(Paragraph(
                f"Parameter Changes: {adaptive_stats['parameter_changes']}",
                normal_style
            ))
            
            if adaptive_stats.get('last_update'):
                story.append(Paragraph(
                    f"Last Update: {adaptive_stats['last_update']}",
                    normal_style
                ))
            
            latest_params = adaptive_stats.get('latest_params', {})
            
            if latest_params:
                param_data = [['Parameter', 'Value']]
                for param, value in latest_params.items():
                    param_data.append([param, f"{value:.3f}"])
                
                param_table = Table(param_data, colWidths=[3*inch, 2*inch])
                param_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                
                story.append(Spacer(1, 12))
                story.append(param_table)
        
        # Footer
        story.append(Spacer(1, 40))
        footer_style = ParagraphStyle(
            'Footer',
            parent=normal_style,
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER,
        )
        
        story.append(Paragraph(
            "omerGPT - GPU-Accelerated Autonomous Crypto Intelligence Platform",
            footer_style
        ))
        story.append(Paragraph("© 2025 - Generated automatically", footer_style))
        
        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated: {output_path}")


class ReportGenerator:
    """
    Main report generator orchestrator.
    Coordinates data aggregation and report generation in multiple formats.
    """
    
    def __init__(
        self,
        db_path: str = "data/omergpt.db",
        output_dir: str = "reports",
    ):
        """
        Initialize report generator.
        
        Args:
            db_path: Path to DuckDB database
            output_dir: Directory for output reports
        """
        self.db_path = db_path
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.db_manager = None
        self.aggregator = None
        self.html_generator = HTMLReportGenerator()
        self.pdf_generator = PDFReportGenerator()
        
        logger.info(f"ReportGenerator initialized: output_dir={output_dir}")
    
    async def initialize(self):
        """Initialize database connection and aggregator."""
        self.db_manager = DatabaseManager(self.db_path)
        
        
        self.aggregator = DataAggregator(self.db_manager)
        
        logger.info("ReportGenerator initialization complete")
    
    async def generate(
        self,
        symbol: str = "BTCUSDT",
        lookback_days: int = 30,
        formats: List[str] = ["html", "pdf"],
    ) -> Dict[str, str]:
        """
        Generate performance report.
        
        Args:
            symbol: Trading symbol
            lookback_days: Days of historical data
            formats: List of output formats ('html', 'pdf')
        
        Returns:
            Dictionary mapping format to output filepath
        """
        logger.info(f"Generating report: {symbol}, {lookback_days} days, {formats}")
        
        # Aggregate data
        performance = await self.aggregator.get_performance_data(
            symbol=symbol,
            lookback_days=lookback_days,
        )
        
        signal_stats = await self.aggregator.get_signal_stats(
            lookback_days=lookback_days
        )
        
        adaptive_stats = await self.aggregator.get_adaptive_stats()
        
        # Prepare report data
        report_data = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "symbol": symbol,
            "performance": performance,
            "signal_stats": signal_stats,
            "adaptive_stats": adaptive_stats,
            **performance,  # Flatten for template access
        }
        
        # Generate reports
        output_files = {}
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if "html" in formats:
            html_path = os.path.join(
                self.output_dir,
                f"summary_{timestamp_str}.html"
            )
            
            html_content = self.html_generator.generate(report_data)
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            output_files["html"] = html_path
            logger.info(f"HTML report saved: {html_path}")
        
        if "pdf" in formats:
            pdf_path = os.path.join(
                self.output_dir,
                f"summary_{timestamp_str}.pdf"
            )
            
            self.pdf_generator.generate(report_data, pdf_path)
            
            output_files["pdf"] = pdf_path
            logger.info(f"PDF report saved: {pdf_path}")
        
        logger.info("Report generation complete")
        
        return output_files
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.db_manager:
            await self.db_manager.close()
        logger.info("ReportGenerator cleanup complete")


# Test block
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    async def test_report_generator():
        """Test report generator with mock data."""
        print("Testing ReportGenerator...")
        print(f"Jinja2 available: {JINJA2_AVAILABLE}")
        print(f"ReportLab available: {REPORTLAB_AVAILABLE}\n")
        
        # Create mock performance data
        mock_data = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "symbol": "BTCUSDT",
            "period": {
                "start": "2025-10-01",
                "end": "2025-10-29",
                "days": 28,
            },
            "returns": {
                "total": 0.156,
                "mean_daily": 0.0055,
                "std_daily": 0.025,
                "annualized": 2.0075,
            },
            "risk_metrics": {
                "sharpe_ratio": 1.85,
                "sortino_ratio": 2.42,
                "max_drawdown": -0.082,
                "max_drawdown_duration": 5,
                "current_drawdown": -0.015,
            },
            "trade_metrics": {
                "win_rate": 0.64,
                "avg_win": 0.0085,
                "avg_loss": -0.0052,
                "profit_factor": 2.35,
            },
            "correlations": {
                "sentiment": 0.42,
                "macro_risk": -0.28,
            },
            "signal_stats": {
                "total_signals": 47,
                "by_type": {
                    "volatility_spike": 28,
                    "structure_break": 19,
                },
                "avg_confidence": 0.72,
            },
            "adaptive_stats": {
                "parameter_changes": 12,
                "latest_params": {
                    "atr_threshold": 1.85,
                    "volume_threshold": 1.42,
                    "min_confidence": 0.65,
                },
                "last_update": "2025-10-29 14:30:00",
            },
        }
        
        print("1. Testing HTML report generation...")
        html_gen = HTMLReportGenerator()
        html_output = html_gen.generate({**mock_data, "performance": mock_data})
        
        os.makedirs("reports", exist_ok=True)
        html_path = "reports/test_report.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
        
        print(f"   HTML report saved: {html_path}")
        print(f"   Size: {len(html_output)} characters\n")
        
        if REPORTLAB_AVAILABLE:
            print("2. Testing PDF report generation...")
            pdf_gen = PDFReportGenerator()
            pdf_path = "reports/test_report.pdf"
            
            pdf_gen.generate({**mock_data, "performance": mock_data}, pdf_path)
            
            print(f"   PDF report saved: {pdf_path}\n")
        else:
            print("2. PDF generation skipped (ReportLab not available)\n")
        
        print("3. Testing performance metrics calculator...")
        metrics = PerformanceMetrics()
        
        # Generate mock returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.02)
        
        sharpe = metrics.calculate_sharpe_ratio(returns)
        sortino = metrics.calculate_sortino_ratio(returns)
        dd_metrics = metrics.calculate_max_drawdown(returns)
        win_metrics = metrics.calculate_win_rate(returns)
        
        print(f"   Sharpe Ratio: {sharpe:.3f}")
        print(f"   Sortino Ratio: {sortino:.3f}")
        print(f"   Max Drawdown: {dd_metrics['max_drawdown']:.3f}")
        print(f"   Win Rate: {win_metrics['win_rate']:.3f}\n")
        
        print("Test completed successfully!")
    
    asyncio.run(test_report_generator())
