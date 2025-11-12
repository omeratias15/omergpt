# ==================== FIXED omerGPT.py ====================
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Ensure root path is included
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

"""
src/omerGPT.py

Main orchestration module for OmerGPT.

Coordinates ingestion, feature generation, anomaly detection,
signal generation, and alert dispatch.

Implements a fully asynchronous event-driven pipeline with error recovery,
logging, and scheduling.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Import modules
from storage.db_manager import DatabaseManager
from ingestion.binance_ws import BinanceIngestion
from ingestion.kraken_ws import KrakenIngestion
from ingestion.etherscan_poll import EtherscanPoller
from features.feature_pipeline import FeaturePipeline
from anomaly_detection.isolation_forest_gpu import AnomalyDetector
from signals.signal_engine import SignalEngine
from alerts.telegram_bot import TelegramBot
from sentiment_analysis.coingecko_scan import CoinGeckoScan
from sentiment_analysis.reddit_fetcher import RedditFetcher


class OmerGPT:
    """
    Main orchestrator for OmerGPT trading system.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()

        # Core services
        self.db = None
        self.binance_ingestion = None
        self.kraken_ingestion = None
        self.etherscan_poller = None
        self.feature_pipeline = None
        self.anomaly_detector = None
        self.signal_engine = None
        self.telegram_bot = None
        self.coingecko_poller = None
        self.reddit_ingest = None

        # Task management
        self.tasks: Dict[str, asyncio.Task] = {}
        self.running = False

        # Monitoring
        self.stats = {
            "start_time": None,
            "module_runs": {},
            "errors": {}
        }

        # Signal handlers
        self.shutdown_event = asyncio.Event()

        self.logger.info("✅ OmerGPT orchestrator initialized")

    def _load_config(self, path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"❌ Config file not found: {path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"❌ Invalid YAML: {e}")
            sys.exit(1)

    def _setup_logger(self) -> logging.Logger:
        """Setup structured logging."""
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"omerGPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8"
        )

        file_handler.setFormatter(logging.Formatter(log_format))

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        return logging.getLogger("omerGPT.orchestrator")

    async def initialize(self):
        """Initialize all modules."""
        self.logger.info("🔧 Initializing modules...")

        try:
            # Database - FIXED: Create instance, not class definition!
            db_path = self.config.get("database", {}).get("path", "data/market_data.duckdb")
            self.db = DatabaseManager(db_path)
            self.logger.info(f"✓ Database initialized: {db_path}")

            # Ingestion modules
            binance_cfg = self.config.get("data_sources", {}).get("binance", {})
            if binance_cfg.get("enabled", True):
                self.binance_ingestion = BinanceIngestion(
                    self.db,
                    symbols=binance_cfg.get("symbols", []),
                    interval=binance_cfg.get("interval", "1m")
                )
                self.logger.info("✓ Binance ingestion initialized")

            kraken_cfg = self.config.get("data_sources", {}).get("kraken", {})
            if kraken_cfg.get("enabled", True):
                self.kraken_ingestion = KrakenIngestion(
                    self.db,
                    symbols=kraken_cfg.get("symbols", []),
                    interval=kraken_cfg.get("interval", 1)
                )
                self.logger.info("✓ Kraken ingestion initialized")

            # Etherscan polling
            etherscan_cfg = self.config.get("data_sources", {}).get("etherscan", {})
            if etherscan_cfg.get("enabled", True):
                self.etherscan_poller = EtherscanPoller(
                    self.db,
                    api_key=etherscan_cfg.get("api_key", ""),
                    addresses=etherscan_cfg.get("addresses", []),
                    poll_interval=etherscan_cfg.get("poll_interval", 10)
                )
                self.logger.info("✓ Etherscan poller initialized")

            # CoinGecko polling
            coingecko_cfg = self.config.get("data_sources", {}).get("coingecko", {})
            if coingecko_cfg.get("enabled", True):
                self.coingecko_poller = CoinGeckoScan(
                    coins=["bitcoin", "ethereum", "solana", "dogecoin"],
                    db_manager=self.db
                )
                self.logger.info("✓ CoinGecko poller initialized")

            # Reddit sentiment ingestion
            reddit_cfg = self.config.get("data_sources", {}).get("reddit", {})
            if reddit_cfg.get("enabled", True):
                self.reddit_ingest = RedditFetcher("data/sentiment_data.duckdb")
                self.logger.info("✓ Reddit ingestion initialized")

            # Feature pipeline
            features_cfg = self.config.get("features", {})
            self.feature_pipeline = FeaturePipeline(
                self.db,
                window_sizes=features_cfg.get("window_sizes", [5, 15, 60]),
                update_interval=features_cfg.get("update_interval", 60),
                gpu_enabled=features_cfg.get("gpu_enabled", False)
            )
            self.logger.info("✓ Feature pipeline initialized")

            # Anomaly detector
            anomaly_cfg = self.config.get("anomaly_detection", {})
            self.anomaly_detector = AnomalyDetector(
                self.db,
                gpu_enabled=anomaly_cfg.get("gpu_enabled", False),
                retrain_interval=anomaly_cfg.get("retrain_interval", 3600)
            )
            self.logger.info("✓ Anomaly detector initialized")

            # Signal engine — fixed thresholds wrapper
            signals_cfg = self.config.get("signals", {})
            thresholds = signals_cfg.get("thresholds", {}) or {}

            thresholds.update({
                "atr_spike_threshold": signals_cfg.get("atr_spike_threshold", 0.02),
                "min_confidence": signals_cfg.get("min_confidence", 0.7),
                "volume_thresh": signals_cfg.get("volume_thresh", 1000)
            })

            self.signal_engine = SignalEngine(
                self.db,
                thresholds=thresholds
            )
            self.logger.info("✓ Signal engine initialized")

            # Telegram alerts
            alerts_cfg = self.config.get("alerts", {}).get("telegram", {})
            if alerts_cfg.get("enabled", True):
                self.telegram_bot = TelegramBot(
                    self.db,
                    token=alerts_cfg.get("token", ""),
                    chat_ids=alerts_cfg.get("chat_ids", []),
                    max_per_minute=alerts_cfg.get("max_per_minute", 5),
                    polling_interval=alerts_cfg.get("polling_interval", 30)
                )
                self.logger.info("✓ Telegram bot initialized")

            self.logger.info("✅ All modules initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}", exc_info=True)
            raise

    async def _safe_run(self, coro, name: str, retries: int = 5):
        retry_count = 0

        while retry_count < retries and self.running:
            try:
                await coro
                return

            except asyncio.CancelledError:
                self.logger.info(f"[{name}] cancelled gracefully")
                return

            except Exception as e:
                retry_count += 1
                self.logger.error(
                    f"[{name}] error: {e} (retry {retry_count}/{retries})",
                    exc_info=True
                )
                self.stats["errors"][name] = str(e)

                if retry_count < retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    self.logger.warning(f"[{name}] restarting in {wait_time}s...")
                    await asyncio.sleep(wait_time)

    async def _ingestion_loop(self):
        """Run data ingestion sources."""
        self.logger.info("🔄 Ingestion loop started")
        
        tasks = []
        
        if self.binance_ingestion:
            tasks.append(
                self._safe_run(
                    self.binance_ingestion.listen(),
                    "Binance"
                )
            )
        
        if self.kraken_ingestion:
            tasks.append(
                self._safe_run(
                    self.kraken_ingestion.listen(),
                    "Kraken"
                )
            )
        
        if self.etherscan_poller:
            tasks.append(
                self._safe_run(
                    self.etherscan_poller.poll(),
                    "Etherscan"
                )
            )

        # Lightweight on-chain metrics sampler
        from ingestion.etherscan_poll import poll_onchain_metrics
        etherscan_cfg = self.config.get("data_sources", {}).get("etherscan", {})
        api_key = etherscan_cfg.get("api_key", "")

        if api_key:
            tasks.append(
                self._safe_run(
                    poll_onchain_metrics(api_key, self.db, self.shutdown_event),
                    "EtherscanMetrics"
                )
            )

        if getattr(self, "coingecko_poller", None):
            tasks.append(
                self._safe_run(
                    self.coingecko_poller.start(),
                    "CoinGecko"
                )
            )

        if getattr(self, "reddit_ingest", None):
            tasks.append(
                self._safe_run(
                    self.reddit_ingest.fetch_and_store(),
                    "Reddit"
                )
            )
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _feature_loop(self):
        """Feature pipeline processing loop."""
        self.logger.info("📊 Feature loop started")
        
        while self.running:
            try:
                await self.feature_pipeline.update_features()
                self.stats["module_runs"]["features"] = datetime.now().isoformat()
                await asyncio.sleep(
                    self.config.get("features", {}).get("update_interval", 60)
                )
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                self.logger.error(f"Feature loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _anomaly_loop(self):
        """Anomaly detection processing loop."""
        self.logger.info("🎯 Anomaly loop started")
        
        while self.running:
            try:
                await self.anomaly_detector.detect_anomalies()
                self.stats["module_runs"]["anomaly"] = datetime.now().isoformat()
                await asyncio.sleep(
                    self.config.get("anomaly_detection", {}).get("check_interval", 60)
                )
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                self.logger.error(f"Anomaly loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _signal_loop(self):
        """Signal generation processing loop."""
        self.logger.info("📈 Signal loop started")
        
        while self.running:
            try:
                signals = await self.signal_engine.generate_signals()
                if signals:
                    await self.signal_engine.save_signals(signals)
                self.stats["module_runs"]["signals"] = datetime.now().isoformat()
                await asyncio.sleep(
                    self.config.get("signals", {}).get("update_interval", 60)
                )
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                self.logger.error(f"Signal loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _alert_loop(self):
        """Alert dispatch loop."""
        self.logger.info("📨 Alert loop started")
        
        if not self.telegram_bot:
            return
        
        try:
            await self.telegram_bot.start()
            await self.shutdown_event.wait()
        finally:
            await self.telegram_bot.stop()

    async def _monitor_loop(self):
        """Monitor system health and statistics."""
        self.logger.info("💓 Monitor loop started")
        
        while self.running:
            try:
                uptime = time.time() - self.stats["start_time"]
                
                stats_summary = {
                    "uptime_seconds": int(uptime),
                    "modules": self.stats["module_runs"],
                    "tasks": {
                        name: task.done() for name, task in self.tasks.items()
                    }
                }
                
                self.logger.info(f"💓 System health: {json.dumps(stats_summary, indent=2)}")
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
            
            except asyncio.CancelledError:
                break
            
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)

    async def _save_state(self):
        """Save orchestrator state to file."""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - self.stats["start_time"],
                "module_runs": self.stats["module_runs"],
                "errors": self.stats["errors"],
                "running_tasks": list(self.tasks.keys())
            }
            
            state_file = Path("state.json")
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    async def run(self):
        """
        Main orchestrator loop.
        """
        self.running = True
        self.stats["start_time"] = time.time()
        
        self.logger.info("🚀 Starting OmerGPT orchestrator...")
        
        try:
            # Initialize modules
            await self.initialize()
            
            # Create tasks for all components
            if self.binance_ingestion:
                await self.binance_ingestion.connect()
            
            if self.kraken_ingestion:
                await self.kraken_ingestion.connect()
            
            # Launch all processing loops
            self.tasks = {
                "ingestion": asyncio.create_task(self._ingestion_loop()),
                "features": asyncio.create_task(self._feature_loop()),
                "anomaly": asyncio.create_task(self._anomaly_loop()),
                "signals": asyncio.create_task(self._signal_loop()),
                "alerts": asyncio.create_task(self._alert_loop()),
                "monitor": asyncio.create_task(self._monitor_loop())
            }
            
            self.logger.info(f"✅ All {len(self.tasks)} processing loops started")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except KeyboardInterrupt:
            self.logger.info("⏹️ Shutdown signal received")
        
        except Exception as e:
            self.logger.error(f"❌ Orchestrator error: {e}", exc_info=True)
        
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown of all components."""
        self.logger.info("🛑 Initiating graceful shutdown...")
        
        self.running = False
        
        try:
            # Stop ingestion sources
            if self.binance_ingestion:
                await self.binance_ingestion.stop()
            if self.kraken_ingestion:
                await self.kraken_ingestion.stop()
            if self.etherscan_poller:
                await self.etherscan_poller.stop()

            # Stop processing components
            if self.feature_pipeline:
                await self.feature_pipeline.stop()
            if self.anomaly_detector:
                await self.anomaly_detector.stop()
            if self.signal_engine:
                await self.signal_engine.stop()
            if self.telegram_bot:
                await self.telegram_bot.stop()
            
            # Cancel all tasks
            for name, task in self.tasks.items():
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete (with timeout)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks.values(), return_exceptions=True),
                    timeout=10
                )
            except asyncio.TimeoutError:
                self.logger.warning("⏱️ Task shutdown timeout")
            
            # Save final state
            await self._save_state()
            
            # Close database
            if self.db:
                self.db.close()
            
            self.logger.info("✅ Shutdown complete")
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)

    def handle_signal(self, sig, frame):
        """Handle OS signal for graceful shutdown."""
        self.logger.info(f"Received signal {sig}")
        self.shutdown_event.set()


# ==================== ENTRY POINT ====================

async def main():
    """Main entry point."""
    # Create orchestrator
    orchestrator = OmerGPT("configs/config.yaml")
    
    # Setup signal handlers
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, orchestrator.handle_signal)
    
    # Run
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    asyncio.run(main())
