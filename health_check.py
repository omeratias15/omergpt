"""
health_check.py — OmerGPT System Health Check Utility
Author: Omer
Updated: 2025-11-03
Description:
Performs end-to-end diagnostics for OmerGPT system components including:
- Config validation
- Database connectivity
- Orchestrator initialization
- Feature detector sanity test
"""

import sys
import os
import asyncio
import logging
import time
import duckdb
import yaml
import numpy as np

# === PATH SETUP ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
SRC_PATH = os.path.join(ROOT_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("healthcheck")

# === CONFIGURATION LOADING ===
CONFIG_PATHS = [
    os.path.join(CONFIG_DIR, "config.yaml"),
    os.path.join(ROOT_DIR, "config.yaml"),
]

CONFIG = None
for path in CONFIG_PATHS:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            CONFIG = yaml.safe_load(f)
        logger.info(f"✅ Loaded configuration from: {path}")
        break

if CONFIG is None:
    logger.error("❌ Config file not found in configs/ or root directory.")
    raise FileNotFoundError("Config file not found in configs/ or root directory.")

# === IMPORT ORCHESTRATOR ===
try:
    from omerGPT import OmerGPT
except ImportError as e:
    logger.error(f"❌ Failed to import OmerGPT: {e}")
    sys.exit(1)

# === CHECK ESSENTIAL FILES ===
def check_essential_files():
    logger.info("📂 Checking essential project files...")
    essentials = [
        os.path.join(CONFIG_DIR, "config.yaml"),
        os.path.join(DATA_DIR, "market_data.duckdb"),
        os.path.join(SRC_PATH, "omerGPT.py"),
    ]
    for file in essentials:
        if not os.path.exists(file):
            logger.warning(f"⚠️ Missing file: {file}")
        else:
            logger.info(f"✅ Found: {file}")

# === DATABASE CONNECTIVITY ===
def check_database():
    db_path = os.path.join(DATA_DIR, "market_data.duckdb")
    if not os.path.exists(db_path):
        logger.warning(f"⚠️ Database file not found at {db_path}")
        return False
    try:
        conn = duckdb.connect(database=db_path, read_only=True)
        conn.execute("SELECT 1;").fetchone()
        conn.close()
        logger.info("✅ DuckDB connection successful.")
        return True
    except Exception as e:
        logger.error(f"❌ DuckDB connection failed: {e}")
        return False

# === INTEGRATION TEST: OMERGPT ===
async def check_omerGPT_initialization():
    try:
        logger.info("🚦 Starting OmerGPT system integration health check...")
        start_time = time.perf_counter()

        # Allow OmerGPT to receive config_path directly if it supports it
        try:
            omer = OmerGPT(config_path=os.path.join(CONFIG_DIR, "config.yaml"))
        except TypeError:
            omer = OmerGPT()

        await omer.initialize()
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"✅ Orchestrator initialized successfully in {elapsed:.2f}ms")

        modules = {
            "db": hasattr(omer, "db"),
            "feature_pipeline": hasattr(omer, "feature_pipeline"),
            "anomaly_detector": hasattr(omer, "anomaly_detector"),
            "signal_engine": hasattr(omer, "signal_engine"),
            "telegram_bot": hasattr(omer, "telegram_bot"),
        }

        for name, exists in modules.items():
            if exists:
                logger.info(f"✅ {name} module loaded successfully")
            else:
                logger.error(f"❌ {name} module is missing!")

        # === Test Feature Detector ===
        if hasattr(omer, "feature_pipeline") and hasattr(omer.feature_pipeline, "detector"):
            det = omer.feature_pipeline.detector
            try:
                returns = np.random.normal(0, 0.01, 200)
                volatility = np.abs(np.random.normal(0.01, 0.005, 200))
                state, conf, probs = det.predict(returns, volatility)
                label = det.get_regime_label(state)
                logger.info(f"📊 Sample HMM prediction → State={label} | Confidence={conf:.3f}")
            except Exception as e:
                logger.error(f"⚠️ Feature detector test failed: {e}")

        logger.info("🟢 System integration check complete.\n")
        return True

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}", exc_info=True)
        return False

# === MAIN EXECUTION ===
if __name__ == "__main__":
    logger.info("🧠 OmerGPT Full System Health Check Initiated")

    check_essential_files()
    db_ok = check_database()
    asyncio.run(check_omerGPT_initialization())

    logger.info("🏁 Health check finished.")
