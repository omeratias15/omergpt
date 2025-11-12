import streamlit as st
import logging
import sys
import os
from pathlib import Path
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import pynvml
    GPU_MONITORING = True
except ImportError:
    GPU_MONITORING = False

import psutil
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardApp:
    """
    Real-time monitoring dashboard for omerGPT intelligence platform.
    Displays regime detection, anomalies, performance metrics, and system stats.
    """
    
    def __init__(self, config_path: str = 'configs/config.yaml'):
        """
        Initialize dashboard application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.refresh_interval = self.config.get('dashboard', {}).get('refresh_interval', 10)
        
        self.gpu_available = GPU_MONITORING and self.config.get('gpu', {}).get('enabled', False)
        
        if self.gpu_available:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                logger.warning(f"GPU monitoring unavailable: {e}")
                self.gpu_available = False
        
        logger.info("Dashboard initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration."""
        import yaml
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def run(self):
        """Main dashboard execution loop."""
        st.set_page_config(
            page_title="omerGPT Intelligence Platform",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🧠 omerGPT Real-Time Intelligence Dashboard")
        st.markdown("*GPU-Accelerated Crypto & FX Anomaly Detection Platform*")
        
        with st.sidebar:
            st.header("⚙️ Controls")
            
            auto_refresh = st.checkbox("Auto-refresh", value=True)
            refresh_rate = st.slider("Refresh rate (seconds)", 5, 60, self.refresh_interval)
            
            st.markdown("---")
            st.subheader("📊 Data Sources")
            
            symbols = self.config.get('data_sources', {}).get('symbols', ['BTCUSDT', 'ETHUSDT'])
            selected_symbol = st.selectbox("Symbol", symbols)
            
            st.markdown("---")
            st.subheader("🔧 System Info")
            st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            regime_state = self._get_current_regime()
            regime_color = {'calm': '🟢', 'volatile': '🟡', 'extreme': '🔴'}.get(regime_state, '⚪')
            st.metric(
                "Current Regime",
                f"{regime_color} {regime_state.upper()}",
                delta=None
            )
        
        with col2:
            anomaly_count = self._get_recent_anomalies(hours=1)
            st.metric(
                "Anomalies (1h)",
                anomaly_count,
                delta=anomaly_count - self._get_recent_anomalies(hours=2, offset=1)
            )
        
        with col3:
            current_volatility = self._get_current_volatility(selected_symbol)
            st.metric(
                "Volatility (24h)",
                f"{current_volatility:.2%}",
                delta=None
            )
        
        with col4:
            sharpe = self._get_latest_sharpe()
            st.metric(
                "Sharpe Ratio (30d)",
                f"{sharpe:.2f}",
                delta=None
            )
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Live Feed", "🎯 Performance", "💻 System", "📊 Analytics"])
        
        with tab1:
            self._render_live_feed(selected_symbol)
        
        with tab2:
            self._render_performance_metrics()
        
        with tab3:
            self._render_system_monitor()
        
        with tab4:
            self._render_analytics(selected_symbol)
        
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()
    
    def _get_current_regime(self) -> str:
        """Fetch current market regime."""
        try:
            import duckdb
            
            db_path = self.config.get('storage', {}).get('duckdb_path', 'data/market_data.duckdb')
            conn = duckdb.connect(db_path, read_only=True)
            
            result = conn.execute("""
                SELECT regime FROM regime_states 
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            
            conn.close()
            
            return result[0] if result else 'unknown'
        except Exception as e:
            logger.debug(f"Regime fetch failed: {e}")
            return 'calm'
    
    def _get_recent_anomalies(self, hours: int = 1, offset: int = 0) -> int:
        """Count recent anomaly events."""
        try:
            import duckdb
            
            db_path = self.config.get('storage', {}).get('duckdb_path', 'data/market_data.duckdb')
            conn = duckdb.connect(db_path, read_only=True)
            
            cutoff = datetime.now() - timedelta(hours=hours + offset)
            end = datetime.now() - timedelta(hours=offset)
            
            result = conn.execute("""
                SELECT COUNT(*) FROM anomaly_events 
                WHERE timestamp >= ? AND timestamp < ?
            """, [cutoff, end]).fetchone()
            
            conn.close()
            
            return result[0] if result else 0
        except Exception as e:
            logger.debug(f"Anomaly count failed: {e}")
            return 0
    
    def _get_current_volatility(self, symbol: str) -> float:
        """Calculate 24h realized volatility."""
        try:
            import duckdb
            
            db_path = self.config.get('storage', {}).get('duckdb_path', 'data/market_data.duckdb')
            conn = duckdb.connect(db_path, read_only=True)
            
            cutoff = datetime.now() - timedelta(hours=24)
            
            df = conn.execute("""
                SELECT close FROM price_ticks 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp
            """, [symbol, cutoff]).df()
            
            conn.close()
            
            if len(df) > 1:
                returns = np.log(df['close'].values[1:] / df['close'].values[:-1])
                volatility = np.std(returns) * np.sqrt(365 * 24)
                return volatility
            
            return 0.0
        except Exception as e:
            logger.debug(f"Volatility calc failed: {e}")
            return 0.0
    
    def _get_latest_sharpe(self) -> float:
        """Get latest Sharpe ratio from validation."""
        try:
            import duckdb
            
            db_path = self.config.get('storage', {}).get('duckdb_path', 'data/market_data.duckdb')
            conn = duckdb.connect(db_path, read_only=True)
            
            result = conn.execute("""
                SELECT sharpe_ratio FROM backtest_results 
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            
            conn.close()
            
            return result[0] if result else 0.0
        except Exception as e:
            logger.debug(f"Sharpe fetch failed: {e}")
            return 0.0
    
    def _render_live_feed(self, symbol: str):
        """Render live anomaly and signal feed."""
        st.subheader("🔴 Live Anomaly Feed")
        
        try:
            import duckdb
            
            db_path = self.config.get('storage', {}).get('duckdb_path', 'data/market_data.duckdb')
            conn = duckdb.connect(db_path, read_only=True)
            
            cutoff = datetime.now() - timedelta(hours=24)
            
            df = conn.execute("""
                SELECT timestamp, event_type, symbol, severity, metadata
                FROM anomaly_events 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 50
            """, [cutoff]).df()
            
            conn.close()
            
            if len(df) > 0:
                for _, row in df.iterrows():
                    severity_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(row['severity'], '⚪')
                    
                    with st.expander(f"{severity_color} {row['event_type']} - {row['symbol']} at {row['timestamp'].strftime('%H:%M:%S')}"):
                        st.json(row['metadata'])
            else:
                st.info("No recent anomalies detected")
        
        except Exception as e:
            st.error(f"Failed to load anomaly feed: {e}")
    
    def _render_performance_metrics(self):
        """Render performance metrics and equity curves."""
        st.subheader("📊 Rolling Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                import duckdb
                
                db_path = self.config.get('storage', {}).get('duckdb_path', 'data/market_data.duckdb')
                conn = duckdb.connect(db_path, read_only=True)
                
                df = conn.execute("""
                    SELECT timestamp, sharpe_ratio, sortino_ratio, max_drawdown, win_rate
                    FROM backtest_results 
                    ORDER BY timestamp DESC LIMIT 10
                """).df()
                
                conn.close()
                
                if len(df) > 0:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No backtest results available")
            
            except Exception as e:
                st.warning(f"Metrics unavailable: {e}")
        
        with col2:
            try:
                equity_file = self._get_latest_equity_curve()
                
                if equity_file and os.path.exists(equity_file):
                    equity_df = pd.read_csv(equity_file)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_df['index'],
                        y=equity_df['equity'],
                        mode='lines',
                        name='Equity',
                        line=dict(color='#00ff00', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Equity Curve",
                        xaxis_title="Time Step",
                        yaxis_title="Equity ($)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No equity curve available")
            
            except Exception as e:
                st.warning(f"Equity curve unavailable: {e}")
    
    def _render_system_monitor(self):
        """Render system resource monitoring."""
        st.subheader("💻 System Resources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**CPU & Memory**")
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            st.progress(cpu_percent / 100, text=f"CPU: {cpu_percent:.1f}%")
            st.progress(memory.percent / 100, text=f"RAM: {memory.percent:.1f}% ({memory.used / 1e9:.1f} GB / {memory.total / 1e9:.1f} GB)")
            
            disk = psutil.disk_usage('/')
            st.progress(disk.percent / 100, text=f"Disk: {disk.percent:.1f}% ({disk.used / 1e9:.1f} GB / {disk.total / 1e9:.1f} GB)")
        
        with col2:
            if self.gpu_available:
                st.markdown("**GPU (RTX 5080)**")
                
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    gpu_mem_percent = (mem_info.used / mem_info.total) * 100
                    
                    st.progress(utilization.gpu / 100, text=f"GPU Utilization: {utilization.gpu}%")
                    st.progress(gpu_mem_percent / 100, text=f"VRAM: {gpu_mem_percent:.1f}% ({mem_info.used / 1e9:.1f} GB / {mem_info.total / 1e9:.1f} GB)")
                    
                    temp_color = "🟢" if temperature < 70 else "🟡" if temperature < 80 else "🔴"
                    st.metric("GPU Temperature", f"{temp_color} {temperature}°C")
                
                except Exception as e:
                    st.error(f"GPU monitoring error: {e}")
            else:
                st.info("GPU monitoring unavailable")
    
    def _render_analytics(self, symbol: str):
        """Render analytical charts and correlation heatmaps."""
        st.subheader("📊 Market Analytics")
        
        try:
            import duckdb
            
            db_path = self.config.get('storage', {}).get('duckdb_path', 'data/market_data.duckdb')
            conn = duckdb.connect(db_path, read_only=True)
            
            cutoff = datetime.now() - timedelta(days=7)
            
            df = conn.execute("""
                SELECT timestamp, close, volume 
                FROM price_ticks 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp
            """, [symbol, cutoff]).df()
            
            conn.close()
            
            if len(df) > 0:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3],
                    subplot_titles=("Price", "Volume")
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['close'],
                        mode='lines',
                        name='Close',
                        line=dict(color='#00ffff', width=1)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=df['timestamp'],
                        y=df['volume'],
                        name='Volume',
                        marker=dict(color='#ff00ff')
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    height=600,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No price data available")
        
        except Exception as e:
            st.error(f"Analytics rendering failed: {e}")
    
    def _get_latest_equity_curve(self) -> Optional[str]:
        """Find most recent equity curve CSV."""
        results_dir = self.config.get('validation', {}).get('results_dir', 'reports/backtest')
        
        if not os.path.exists(results_dir):
            return None
        
        equity_files = [f for f in os.listdir(results_dir) if 'equity' in f and f.endswith('.csv')]
        
        if len(equity_files) == 0:
            return None
        
        equity_files.sort(reverse=True)
        return os.path.join(results_dir, equity_files[0])


def main():
    """Entry point for dashboard application."""
    app = DashboardApp()
    app.run()


if __name__ == "__main__":
    main()
