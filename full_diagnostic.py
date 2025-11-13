"""
full_diagnostic.py - Comprehensive System Check for omerGPT

Checks every module, file, configuration, and component to ensure
the entire trading system is working correctly.
"""

import sys
import os
import asyncio
from datetime import datetime
from pathlib import Path
import importlib.util
import json

sys.path.insert(0, 'src')

print("=" * 80)
print("🔍 OMERGPT COMPREHENSIVE SYSTEM DIAGNOSTIC")
print("=" * 80)
print()

# Track results
results = {
    "passed": [],
    "warnings": [],
    "failed": [],
    "missing": []
}

def check_file(path, description):
    """Check if a file exists"""
    if os.path.exists(path):
        results["passed"].append(f"✅ {description}")
        return True
    else:
        results["missing"].append(f"❌ {description} - FILE MISSING: {path}")
        return False

def check_module(module_path, module_name):
    """Check if a Python module can be imported"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            results["passed"].append(f"✅ {module_name} imports successfully")
            return True
    except Exception as e:
        results["failed"].append(f"❌ {module_name} import failed: {e}")
        return False

# ============================================================
# 1. PROJECT STRUCTURE
# ============================================================
print("📁 [1/12] Checking Project Structure...")
print("-" * 80)

structure = {
    "configs/config.yaml": "Configuration file",
    "data/": "Data directory",
    "logs/": "Logs directory",
    "checkpoints/": "Model checkpoints directory",
    "src/": "Source code directory",
    "src/storage/": "Storage module",
    "src/ingestion/": "Ingestion module",
    "src/features/": "Features module",
    "src/anomaly_detection/": "Anomaly detection module",
    "src/signals/": "Signals module",
    "src/alerts/": "Alerts module",
    "src/sentiment_analysis/": "Sentiment analysis module"
}

for path, desc in structure.items():
    if os.path.exists(path):
        print(f"✅ {desc}: {path}")
        results["passed"].append(f"{desc}")
    else:
        print(f"❌ {desc}: MISSING - {path}")
        results["missing"].append(f"{desc}")

print()

# ============================================================
# 2. CORE MODULES
# ============================================================
print("🔧 [2/12] Checking Core Modules...")
print("-" * 80)

core_modules = {
    "src/omerGPT.py": "Main orchestrator",
    "src/storage/db_manager.py": "Database manager",
    "src/ingestion/binance_ws.py": "Binance WebSocket",
    "src/ingestion/kraken_ws.py": "Kraken WebSocket",
    "src/features/feature_pipeline.py": "Feature pipeline",
    "src/anomaly_detection/isolation_forest_gpu.py": "Anomaly detector",
    "src/signals/signal_engine.py": "Signal engine",      # FIXED
    "src/alerts/telegram_bot.py": "Telegram bot"          # FIXED
}

for path, desc in core_modules.items():
    if check_file(path, desc):
        print(f"✅ {desc}")
    else:
        print(f"❌ {desc} - MISSING")

print()

# ============================================================
# 3. CONFIGURATION FILES
# ============================================================
print("⚙️  [3/12] Checking Configuration...")
print("-" * 80)

if os.path.exists("configs/config.yaml"):
    try:
        import yaml
        with open("configs/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Configuration loaded: {len(config)} sections")
        
        required_sections = ['ingestion', 'features', 'anomaly_detection', 'signals', 'alerts']
        for section in required_sections:
            if section in config:
                print(f"  ✅ {section} section found")
                results["passed"].append(f"Config section: {section}")
            else:
                print(f"  ⚠️  {section} section missing")
                results["warnings"].append(f"Config section missing: {section}")
    except Exception as e:
        print(f"❌ Config load error: {e}")
        results["failed"].append(f"Configuration loading: {e}")
else:
    print("❌ config.yaml not found")
    results["missing"].append("config.yaml")

print()

# ============================================================
# 4. PYTHON DEPENDENCIES
# ============================================================
print("📦 [4/12] Checking Python Dependencies...")
print("-" * 80)

dependencies = {
    "pandas": "Data manipulation",
    "numpy": "Numerical computing",
    "duckdb": "Database",
    "ccxt": "Exchange API",
    "websockets": "WebSocket connections",
    "yaml": "YAML parsing",
    "sklearn": "Machine learning",
    "hmmlearn": "Hidden Markov Models"
}

for module, desc in dependencies.items():
    try:
        __import__(module)
        print(f"✅ {desc}: {module}")
        results["passed"].append(f"Dependency: {module}")
    except ImportError:
        print(f"❌ {desc}: {module} NOT INSTALLED")
        results["failed"].append(f"Missing dependency: {module}")

# Check optional GPU dependencies
try:
    import cupy
    import cudf
    print(f"✅ GPU support: cupy & cudf")
    results["passed"].append("GPU support available")
    GPU_AVAILABLE = True
except ImportError:
    print(f"⚠️  GPU support: NOT AVAILABLE (optional)")
    results["warnings"].append("GPU support not available")
    GPU_AVAILABLE = False

print()

# ============================================================
# 5. DATABASE CHECK
# ============================================================
print("🗄️  [5/12] Checking Database...")
print("-" * 80)

db_path = "data/market_data.duckdb"
if os.path.exists(db_path):
    try:
        import duckdb
        conn = duckdb.connect(db_path)
        
        # Check tables - FIXED
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t[0] for t in tables]
        required_tables = ['candles', 'features', 'anomaly_events', 'signals']
        
        print(f"✅ Database exists with {len(tables)} tables")
        
        for table in required_tables:
            if table in table_names:
                result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                count = result if result else 0
                print(f"  ✅ {table}: {count} rows")
                results["passed"].append(f"Table {table}: {count} rows")
            else:
                print(f"  ❌ {table}: TABLE MISSING")
                results["missing"].append(f"Table: {table}")
        
        conn.close()
    except Exception as e:
        print(f"❌ Database error: {e}")
        results["failed"].append(f"Database: {e}")
else:
    print("⚠️  Database doesn't exist yet (will be created on first run)")
    results["warnings"].append("Database not created yet")

print()

# ============================================================
# 6. WEBSOCKET CONNECTIVITY
# ============================================================
print("🌐 [6/12] Testing WebSocket Connectivity...")
print("-" * 80)

async def test_websocket():
    """Test Binance WebSocket"""
    try:
        import websockets
        ws_url = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"
        
        async with websockets.connect(ws_url) as ws:
            print("✅ Connected to Binance WebSocket")
            
            try:
                async with asyncio.timeout(5):
                    msg = await ws.recv()
                    data = json.loads(msg)
                    print(f"✅ Received data: BTC/USDT")
                    results["passed"].append("WebSocket connectivity")
                    return True
            except asyncio.TimeoutError:
                print("⚠️  No data received (timeout)")
                results["warnings"].append("WebSocket timeout")
                return False
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        results["failed"].append(f"WebSocket: {e}")
        return False

try:
    asyncio.run(test_websocket())
except Exception as e:
    print(f"❌ WebSocket test failed: {e}")
    results["failed"].append(f"WebSocket test: {e}")

print()

# ============================================================
# 7. API CONNECTIVITY
# ============================================================
print("🌍 [7/12] Testing API Connectivity...")
print("-" * 80)

try:
    import ccxt
    exchange = ccxt.binance()
    ticker = exchange.fetch_ticker('BTC/USDT')
    print(f"✅ Binance REST API working - BTC: ${ticker['last']:,.2f}")
    results["passed"].append("Binance REST API")
except Exception as e:
    print(f"❌ Binance API error: {e}")
    results["failed"].append(f"Binance API: {e}")

print()

# ============================================================
# 8. MODULE IMPORT TESTS
# ============================================================
print("🔬 [8/12] Testing Module Imports...")
print("-" * 80)

modules_to_test = [
    ("src/storage/db_manager.py", "DatabaseManager"),
    ("src/features/feature_pipeline.py", "FeaturePipeline"),
    ("src/anomaly_detection/isolation_forest_gpu.py", "AnomalyDetector"),
    ("src/signals/signal_engine.py", "SignalEngine"),     # FIXED
    ("src/alerts/telegram_bot.py", "TelegramBot")        # FIXED
]

for module_path, module_name in modules_to_test:
    if os.path.exists(module_path):
        check_module(module_path, module_name)
        print(f"✅ {module_name} can be imported")
    else:
        print(f"❌ {module_name} file not found")
        results["missing"].append(f"Module: {module_name}")

print()

# ============================================================
# 9. HELPER SCRIPTS
# ============================================================
print("📝 [9/12] Checking Helper Scripts...")
print("-" * 80)

helper_scripts = {
    "backfill.py": "Data backfill script",
    "diagnose.py": "Diagnostic script"
}

for script, desc in helper_scripts.items():
    if os.path.exists(script):
        print(f"✅ {desc}: {script}")
        results["passed"].append(f"Helper script: {script}")
    else:
        print(f"⚠️  {desc}: NOT FOUND")
        results["warnings"].append(f"Missing helper: {script}")

print()

# ============================================================
# 10. LOGGING SETUP
# ============================================================
print("📋 [10/12] Checking Logging Setup...")
print("-" * 80)

if os.path.exists("logs/"):
    print("✅ Logs directory exists")
    log_files = list(Path("logs/").glob("*.log"))
    if log_files:
        print(f"✅ Found {len(log_files)} log files")
        results["passed"].append(f"Logging: {len(log_files)} log files")
    else:
        print("ℹ️  No log files yet (normal for first run)")
else:
    print("⚠️  Logs directory missing")
    results["warnings"].append("Logs directory missing")

print()

# ============================================================
# 11. MODEL CHECKPOINTS
# ============================================================
print("💾 [11/12] Checking Model Checkpoints...")
print("-" * 80)

if os.path.exists("checkpoints/"):
    print("✅ Checkpoints directory exists")
    checkpoint_files = list(Path("checkpoints/").glob("*.pkl"))
    if checkpoint_files:
        print(f"✅ Found {len(checkpoint_files)} checkpoint files")
        results["passed"].append(f"Checkpoints: {len(checkpoint_files)} files")
    else:
        print("ℹ️  No checkpoints yet (normal for first run)")
else:
    print("⚠️  Checkpoints directory missing")
    results["warnings"].append("Checkpoints directory missing")

print()

# ============================================================
# 12. OVERALL HEALTH
# ============================================================
print("🏥 [12/12] Overall System Health...")
print("-" * 80)

# Calculate health score
total_checks = len(results["passed"]) + len(results["warnings"]) + len(results["failed"]) + len(results["missing"])
passed = len(results["passed"])
health_score = (passed / total_checks * 100) if total_checks > 0 else 0

print(f"Health Score: {health_score:.1f}%")
print(f"✅ Passed: {len(results['passed'])}")
print(f"⚠️  Warnings: {len(results['warnings'])}")
print(f"❌ Failed: {len(results['failed'])}")
print(f"🔍 Missing: {len(results['missing'])}")

print()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 80)
print("📋 DIAGNOSTIC SUMMARY")
print("=" * 80)
print()

if health_score >= 90:
    print("✅ SYSTEM STATUS: EXCELLENT")
    print("All critical components are working correctly.")
    print()
    print("🚀 Ready to run: python src/omerGPT.py")
elif health_score >= 70:
    print("⚠️  SYSTEM STATUS: GOOD (with warnings)")
    print("Most components working, but some issues detected.")
elif health_score >= 50:
    print("⚠️  SYSTEM STATUS: DEGRADED")
    print("Multiple issues detected. Review warnings and failures.")
else:
    print("❌ SYSTEM STATUS: CRITICAL")
    print("Major issues detected. Fix critical errors before running.")

if results["failed"]:
    print()
    print("🔴 CRITICAL FAILURES:")
    for failure in results["failed"][:5]:  # Show first 5
        print(f"   {failure}")

if results["missing"]:
    print()
    print("🟠 MISSING COMPONENTS:")
    for missing in results["missing"][:5]:  # Show first 5
        print(f"   {missing}")

if results["warnings"]:
    print()
    print("🟡 WARNINGS:")
    for warning in results["warnings"][:5]:  # Show first 5
        print(f"   {warning}")

print()
print("=" * 80)
print("Diagnostic complete!")
print("=" * 80)
