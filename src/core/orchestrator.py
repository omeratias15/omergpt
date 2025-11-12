# =====================================================
# ✅ RESEARCH COMPLIANCE WRAPPER §4.1–§4.3 (Dashboard / Orchestrator)
# =====================================================
import time
import yaml
import statistics
import logging
from datetime import datetime

# Load central research configuration
try:
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception:
    CONFIG = {}

REFRESH_INTERVAL = CONFIG.get("dashboard", {}).get("refresh_interval", 5)
GPU_ENABLED = CONFIG.get("gpu", {}).get("enabled", True)
LATENCY_TARGET_MS = 150
SYNC_TOLERANCE_S = 2.0

logger = logging.getLogger("DashboardResearchLayer")
logger.info(
    f"[Init] Dashboard compliance layer active | GPU={GPU_ENABLED} | Refresh={REFRESH_INTERVAL}s | p95 target={LATENCY_TARGET_MS}ms"
)
# =====================================================

"""
src/dashboard/app.py

Interactive real-time dashboard for OmerGPT.

Displays live data from the database:
market prices, on-chain metrics, features, anomaly scores, and signals.

Built with Dash/Plotly + FastAPI backend.
Supports WebSocket updates and async refresh cycles.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from plotly.subplots import make_subplots

logger = logging.getLogger("omerGPT.dashboard")


class Dashboard:
    """
    Real-time trading dashboard for OmerGPT.
    
    Features:
    - Live price charts with candlestick data
    - Feature trend visualization
    - Anomaly score tracking
    - Trading signal display
    - Multi-symbol support
    - Auto-refresh capability
    """
    
    def __init__(
        self,
        db_manager,
        refresh_interval: int = 5,
        host: str = "0.0.0.0",
        port: int = 8050
    ):
        """
        Initialize dashboard.
        
        Args:
            db_manager: DatabaseManager instance
            refresh_interval: Seconds between data updates
            host: Server host
            port: Server port
        """
        self.db = db_manager
        self.refresh_interval = refresh_interval
        self.host = host
        self.port = port
        
        # Initialize Dash app
        self.app = Dash(
            __name__,
            suppress_callback_exceptions=True,
            external_stylesheets=[
                "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap"
            ]
        )
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info(f"Dashboard initialized: {host}:{port}, refresh={refresh_interval}s")

    def _init_layout(self):
        """Initialize dashboard layout."""
        self.app.layout = html.Div(
            style={
                "fontFamily": "Roboto, sans-serif",
                "backgroundColor": "#0a0e27",
                "color": "#ffffff",
                "minHeight": "100vh",
                "padding": "20px"
            },
            children=[
                # Header
                html.Div(
                    style={
                        "textAlign": "center",
                        "marginBottom": "30px",
                        "borderBottom": "2px solid #1e3a8a"
                    },
                    children=[
                        html.H1("⚡ OmerGPT — Live Trading Intelligence Dashboard", style={
                            "fontSize": "32px",
                            "fontWeight": "700",
                            "margin": "0 0 10px 0",
                            "color": "#00d9ff"
                        }),
                        html.P("Real-time market analysis powered by AI 🤖", style={
                            "fontSize": "14px",
                            "color": "#888",
                            "margin": "0"
                        })
                    ]
                ),

                # Control Panel
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(4, 1fr)",
                        "gap": "15px",
                        "marginBottom": "20px",
                        "padding": "15px",
                        "backgroundColor": "#111827",
                        "borderRadius": "8px"
                    },
                    children=[
                        # Symbol Dropdown
                        html.Div([
                            html.Label("Symbol:", style={"fontWeight": "500"}),
                            dcc.Dropdown(
                                id="symbol-dropdown",
                                options=[{"label": s, "value": s} for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]],
                                value="BTCUSDT",
                                style={
                                    "backgroundColor": "#1f2937",
                                    "color": "#ffffff",
                                    "borderRadius": "4px"
                                }
                            )
                        ]),

                        # Time Range
                        html.Div([
                            html.Label("Lookback:", style={"fontWeight": "500"}),
                            dcc.Dropdown(
                                id="lookback-dropdown",
                                options=[
                                    {"label": "1 day", "value": 1440},
                                    {"label": "7 days", "value": 10080},
                                    {"label": "30 days", "value": 43200}
                                ],
                                value=1440,
                                style={
                                    "backgroundColor": "#1f2937",
                                    "color": "#ffffff",
                                    "borderRadius": "4px"
                                }
                            )
                        ]),

                        # Auto Refresh Toggle
                        html.Div([
                            html.Label("Auto Refresh:", style={"fontWeight": "500"}),
                            dcc.Dropdown(
                                id="refresh-toggle",
                                options=[
                                    {"label": "On", "value": "on"},
                                    {"label": "Off", "value": "off"}
                                ],
                                value="on",
                                style={
                                    "backgroundColor": "#1f2937",
                                    "color": "#ffffff",
                                    "borderRadius": "4px"
                                }
                            )
                        ]),

                        # Status
                        html.Div([
                            html.Label("Status:", style={"fontWeight": "500"}),
                            html.Div(
                                id="status-display",
                                style={
                                    "padding": "8px 12px",
                                    "backgroundColor": "#1f2937",
                                    "borderRadius": "4px",
                                    "textAlign": "center",
                                    "fontSize": "12px",
                                    "color": "#00d9ff"
                                },
                                children="Loading..."
                            )
                        ])
                    ]
                ),

                # Statistics Panel
                html.Div(
                    id="stats-panel",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(4, 1fr)",
                        "gap": "15px",
                        "marginBottom": "20px"
                    }
                ),

                # Price Chart
                html.Div(
                    style={"marginBottom": "20px", "backgroundColor": "#111827", "padding": "15px", "borderRadius": "8px"},
                    children=[
                        dcc.Graph(id="price-chart", style={"height": "400px"})
                    ]
                ),

                # Features Chart
                html.Div(
                    style={"marginBottom": "20px", "backgroundColor": "#111827", "padding": "15px", "borderRadius": "8px"},
                    children=[
                        dcc.Graph(id="features-chart", style={"height": "350px"})
                    ]
                ),

                # Anomaly & Signals Charts
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"},
                    children=[
                        html.Div(
                            style={"backgroundColor": "#111827", "padding": "15px", "borderRadius": "8px"},
                            children=[dcc.Graph(id="anomaly-chart", style={"height": "300px"})]
                        ),
                        html.Div(
                            style={"backgroundColor": "#111827", "padding": "15px", "borderRadius": "8px"},
                            children=[dcc.Graph(id="signals-chart", style={"height": "300px"})]
                        )
                    ]
                ),

                # Interval for auto-refresh
                dcc.Interval(
                    id="refresh-interval",
                    interval=self.refresh_interval * 1000,
                    n_intervals=0,
                    disabled=False
                )
            ]
        )

    def _setup_callbacks(self):
        """Setup Dash callbacks."""
        
        @self.app.callback(
            [
                Output("price-chart", "figure"),
                Output("features-chart", "figure"),
                Output("anomaly-chart", "figure"),
                Output("signals-chart", "figure"),
                Output("stats-panel", "children"),
                Output("status-display", "children")
            ],
            [
                Input("symbol-dropdown", "value"),
                Input("lookback-dropdown", "value"),
                Input("refresh-interval", "n_intervals"),
                Input("refresh-toggle", "value")
            ]
        )
        def update_dashboard(symbol, lookback_min, n_intervals, refresh_status):
            """Update all dashboard charts."""
            if refresh_status == "off" and n_intervals > 0:
                # Return existing figures if refresh is off
                return (
                    go.Figure(), go.Figure(), go.Figure(), go.Figure(),
                    [], f"Last update: {datetime.now().strftime('%H:%M:%S')}"
                )
            
            try:
                # Load data
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=lookback_min)
                
                candles_df = self.db.conn.execute(
                    "SELECT * FROM candles WHERE symbol = ? AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms",
                    (symbol, start_time, end_time)
                ).df()
                
                features_df = self.db.conn.execute(
                    "SELECT * FROM features WHERE symbol = ? AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms",
                    (symbol, start_time, end_time)
                ).df()
                
                anomalies_df = self.db.conn.execute(
                    "SELECT * FROM anomaly_events WHERE symbol = ? AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms",
                    (symbol, start_time, end_time)
                ).df()
                
                signals_df = self.db.conn.execute(
                    "SELECT * FROM signals WHERE symbol = ? AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms",
                    (symbol, start_time, end_time)
                ).df()
                
                # Generate charts
                fig_price = self._create_price_chart(candles_df, symbol)
                fig_features = self._create_features_chart(features_df)
                fig_anomaly = self._create_anomaly_chart(anomalies_df)
                fig_signals = self._create_signals_chart(signals_df, candles_df)
                
                # Create stats panel
                stats_children = self._create_stats_panel(candles_df, signals_df, anomalies_df)
                
                # Status
                status = f"✓ {len(candles_df)} candles | {len(signals_df)} signals | Updated: {datetime.now().strftime('%H:%M:%S')}"
                
                return fig_price, fig_features, fig_anomaly, fig_signals, stats_children, status
            
            except Exception as e:
                logger.error(f"Dashboard update error: {e}", exc_info=True)
                empty_fig = go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)
                return (
                    empty_fig, empty_fig, empty_fig, empty_fig,
                    [html.Div("Error loading data", style={"color": "red"})],
                    f"❌ Error"
                )

    def _create_price_chart(self, df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create candlestick price chart."""
        if df.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False)
        
        fig = go.Figure(data=[
            go.Candlestick(
                x=df["ts_ms"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price"
            )
        ])
        
        fig.update_layout(
            title=f"{symbol} — Price Action",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            template="plotly_dark",
            margin={"l": 50, "r": 50, "t": 50, "b": 50},
            xaxis_rangeslider_visible=False
        )
        
        return fig

    def _create_features_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create multi-axis features chart."""
        if df.empty:
            return go.Figure().add_annotation(text="No features data", showarrow=False)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("RSI & Momentum", "Volatility"),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=df["ts_ms"], y=df.get("rsi_14", []),
                name="RSI", line={"color": "#00d9ff"}
            ),
            row=1, col=1, secondary_y=False
        )
        
        # Momentum
        fig.add_trace(
            go.Scatter(
                x=df["ts_ms"], y=df.get("momentum_5m", []),
                name="Momentum", line={"color": "#ff006e"}
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Volatility
        fig.add_trace(
            go.Scatter(
                x=df["ts_ms"],
                y=df.get("volatility_15m", []),
                name="Volatility",
                fill="tozeroy",
                line={"color": "#ffbe0b"}
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            hovermode="x unified",
            template="plotly_dark",
            height=350,
            margin={"l": 50, "r": 50, "t": 50, "b": 50}
        )
        
        return fig

    def _create_anomaly_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create anomaly score visualization."""
        if df.empty:
            return go.Figure().add_annotation(text="No anomalies detected", showarrow=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df["ts_ms"],
            y=df["confidence"],
            mode="lines+markers",
            name="Anomaly Score",
            line={"color": "#ff6b6b", "width": 2},
            marker={"size": 6}
        ))
        
        # Add threshold line
        fig.add_hline(
            y=0.95,
            line_dash="dash",
            line_color="orange",
            annotation_text="Critical (0.95)"
        )
        
        fig.update_layout(
            title="Anomaly Detection Scores",
            xaxis_title="Time",
            yaxis_title="Confidence Score",
            hovermode="x unified",
            template="plotly_dark",
            margin={"l": 50, "r": 50, "t": 50, "b": 50}
        )
        
        return fig

    def _create_signals_chart(self, signals_df: pd.DataFrame, candles_df: pd.DataFrame) -> go.Figure:
        """Create trading signals chart."""
        fig = go.Figure()
        
        # Add price line
        if not candles_df.empty:
            fig.add_trace(go.Scatter(
                x=candles_df["ts_ms"],
                y=candles_df["close"],
                name="Price",
                line={"color": "#888888", "width": 1},
                opacity=0.5
            ))
        
        # Add BUY signals
        if not signals_df.empty:
            buy_signals = signals_df[signals_df["signal_type"] == "BUY"]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals["ts_ms"],
                    y=buy_signals["confidence"],
                    mode="markers",
                    name="BUY",
                    marker={"color": "#00d084", "size": 12, "symbol": "triangle-up"}
                ))
            
            # Add SELL signals
            sell_signals = signals_df[signals_df["signal_type"] == "SELL"]
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals["ts_ms"],
                    y=sell_signals["confidence"],
                    mode="markers",
                    name="SELL",
                    marker={"color": "#ff006e", "size": 12, "symbol": "triangle-down"}
                ))
        
        fig.update_layout(
            title="Trading Signals",
            xaxis_title="Time",
            yaxis_title="Confidence",
            hovermode="x unified",
            template="plotly_dark",
            margin={"l": 50, "r": 50, "t": 50, "b": 50}
        )
        
        return fig

    def _create_stats_panel(
        self,
        candles_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        anomalies_df: pd.DataFrame
    ) -> List:
        """Create statistics panel."""
        stats = []
        
        # Current Price
        if not candles_df.empty:
            current_price = candles_df.iloc[-1]["close"]
            change = ((current_price - candles_df.iloc[0]["close"]) / candles_df.iloc[0]["close"]) * 100
            stats.append(self._stat_card(
                f"${current_price:.2f}",
                "Current Price",
                f"{change:+.2f}%",
                "green" if change > 0 else "red"
            ))
        
        # Signal Count
        stats.append(self._stat_card(
            str(len(signals_df)),
            "Total Signals",
            f"{len(signals_df[signals_df['signal_type']=='BUY'])} BUY",
            "#00d9ff"
        ))
        
        # Anomalies
        stats.append(self._stat_card(
            str(len(anomalies_df)),
            "Anomalies",
            f"{len(anomalies_df[anomalies_df['confidence']>0.95])} Critical",
            "#ff6b6b"
        ))
        
        # Candles
        stats.append(self._stat_card(
            str(len(candles_df)),
            "Candles",
            "Data points",
            "#888"
        ))
        
        return stats

    def _stat_card(self, value: str, label: str, detail: str, color: str) -> html.Div:
        """Create a stat card component."""
        return html.Div(
            style={
                "backgroundColor": "#1f2937",
                "padding": "15px",
                "borderRadius": "8px",
                "borderLeft": f"4px solid {color}",
                "textAlign": "center"
            },
            children=[
                html.Div(value, style={"fontSize": "20px", "fontWeight": "700", "color": color}),
                html.Div(label, style={"fontSize": "12px", "color": "#888", "marginTop": "5px"}),
                html.Div(detail, style={"fontSize": "11px", "color": "#666", "marginTop": "3px"})
            ]
        )

    def run(self):
        """Start the dashboard server."""
        logger.info(f"🚀 Starting dashboard at http://{self.host}:{self.port}")
        self._init_layout()
        self.app.run_server(host=self.host, port=self.port, debug=False)

# =====================================================
# ✅ RESEARCH MONITORING LAYER (Runtime Metrics Collector)
# =====================================================
class ResearchMonitor:
    """
    Dashboard Research Monitor — tracks end-to-end orchestration timing and GPU sync.
    Corresponds to §4.2.1–§4.3.3 of the research.
    """

    def __init__(self, db):
        self.db = db
        self.latency_log = []
        self.stats = {"updates": 0, "sync_ok": 0, "sync_fail": 0}

    async def measure_refresh(self, coro):
        """Wrap dashboard update coroutine to measure refresh latency."""
        t0 = time.perf_counter()
        result = await coro
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.latency_log.append(elapsed_ms)
        self.latency_log = self.latency_log[-200:]
        self.stats["updates"] += 1
        return result

    def report_latency(self):
        """Return rolling latency p95."""
        if len(self.latency_log) < 5:
            return 0.0
        return statistics.quantiles(self.latency_log, n=100)[94]

    def check_sync(self):
        """Check timestamp alignment across key tables."""
        try:
            timestamps = []
            for table in ["features", "anomaly_events", "signals"]:
                ts = self.db.conn.execute(f"SELECT MAX(ts_ms) FROM {table}").fetchone()[0]
                if ts:
                    timestamps.append(ts)
            if len(timestamps) < 2:
                return True
            max_diff = max(timestamps) - min(timestamps)
            if max_diff.total_seconds() <= SYNC_TOLERANCE_S:
                self.stats["sync_ok"] += 1
                return True
            else:
                self.stats["sync_fail"] += 1
                logger.warning(f"⚠️ Desync detected between modules: Δt={max_diff.total_seconds():.2f}s")
                return False
        except Exception as e:
            logger.error(f"Sync check failed: {e}")
            return False

    def log_metrics(self):
        """Emit periodic metrics to log."""
        p95 = self.report_latency()
        logger.info(
            f"[Research] Refreshes={self.stats['updates']} | p95={p95:.1f}ms | "
            f"SyncOK={self.stats['sync_ok']} | SyncFail={self.stats['sync_fail']} | GPU={GPU_ENABLED}"
        )
        if p95 > LATENCY_TARGET_MS:
            logger.warning(f"⚠️ Dashboard p95 latency {p95:.1f}ms exceeds {LATENCY_TARGET_MS}ms target")


# =====================================================
# ✅ CONTINUOUS ORCHESTRATION MONITOR LOOP
# =====================================================
if __name__ == "__main__":
    import sys
    import asyncio
    sys.path.insert(0, "src")
    from storage.db_manager import DatabaseManager

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    db = DatabaseManager("data/market_data.duckdb")

    monitor = ResearchMonitor(db)

    dashboard = Dashboard(
        db_manager=db,
        refresh_interval=REFRESH_INTERVAL,
        host="0.0.0.0",
        port=8050
    )

    async def orchestrator_loop():
        """Async orchestrator control loop (real-time monitoring)"""
        while True:
            await asyncio.sleep(REFRESH_INTERVAL)
            synced = monitor.check_sync()
            monitor.log_metrics()
            if not synced:
                logger.warning("🔁 Triggering resync or feature rebuild (pending integration hook)")

    try:
        loop = asyncio.get_event_loop()
        loop.create_task(orchestrator_loop())
        dashboard.run()
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped gracefully")
        db.close()

