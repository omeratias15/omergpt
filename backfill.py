"""
backfill.py - Historical Data Backfill Script for omerGPT

Fetches historical OHLCV data from Binance and populates the database.
This gives the system enough data to start computing features and signals immediately.
"""

import ccxt
import pandas as pd
import duckdb
from datetime import datetime, timedelta
import sys
import os

print("=" * 70)
print("📥 OMERGPT HISTORICAL DATA BACKFILL")
print("=" * 70)
print()

# Configuration
SYMBOLS = [
    'BTCUSDT',
    'ETHUSDT',
    'SOLUSDT',
    'BNBUSDT',
    'XRPUSDT',
    'DOGEUSDT',
    'ADAUSDT',
    'AVAXUSDT',
    'MATICUSDT',
    'DOTUSDT'
]

HOURS_TO_FETCH = 3  # How many hours of historical data
DB_PATH = 'data/market_data.duckdb'

print(f"Symbols: {len(SYMBOLS)}")
print(f"History: {HOURS_TO_FETCH} hours")
print(f"Database: {DB_PATH}")
print()

# Initialize exchange and database
try:
    print("Initializing Binance exchange...")
    exchange = ccxt.binance()
    print("✅ Binance exchange initialized")
except Exception as e:
    print(f"❌ Failed to initialize Binance: {e}")
    sys.exit(1)

try:
    os.makedirs('data', exist_ok=True)
    print(f"Connecting to database: {DB_PATH}")
    conn = duckdb.connect(DB_PATH)
    print("✅ Database connected")
except Exception as e:
    print(f"❌ Failed to connect to database: {e}")
    sys.exit(1)

print()
print("-" * 70)
print("FETCHING HISTORICAL DATA")
print("-" * 70)
print()

# Calculate time range
since = int((datetime.now() - timedelta(hours=HOURS_TO_FETCH)).timestamp() * 1000)
total_candles = 0
failed_symbols = []

# Fetch data for each symbol
for i, symbol in enumerate(SYMBOLS, 1):
    try:
        print(f"[{i}/{len(SYMBOLS)}] Fetching {symbol}...", end=" ")
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe='1m',
            since=since,
            limit=180  # 3 hours = 180 minutes
        )
        
        if not ohlcv:
            print("⚠️  No data returned")
            failed_symbols.append(symbol)
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Add required columns
        df['close_time'] = df['ts_ms'] + 60000
        df['symbol'] = symbol
        
        # Reorder columns to match database schema
        df = df[['symbol', 'ts_ms', 'open', 'high', 'low', 'close', 'volume', 'close_time']]
        
        # Insert into database
        conn.register('tmp', df)
        conn.execute('INSERT INTO candles SELECT * FROM tmp')
        conn.unregister('tmp')
        
        total_candles += len(df)
        print(f"✅ {len(df)} candles")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        failed_symbols.append(symbol)

print()
print("-" * 70)
print("SUMMARY")
print("-" * 70)
print()

# Verify data in database
try:
    result = conn.execute("SELECT COUNT(*) FROM candles").fetchone()
    db_count = result if result else 0
    
    result = conn.execute("SELECT DISTINCT symbol FROM candles").fetchall()
    db_symbols = [s[0] for s in result]
    
    print(f"✅ Total candles inserted: {total_candles}")
    print(f"✅ Candles in database: {db_count}")
    print(f"✅ Symbols in database: {len(db_symbols)}")
    print(f"   {db_symbols}")
    
    if failed_symbols:
        print()
        print(f"⚠️  Failed symbols ({len(failed_symbols)}):")
        for sym in failed_symbols:
            print(f"   - {sym}")
    
except Exception as e:
    print(f"❌ Failed to verify data: {e}")

# Close database
conn.close()

print()
print("=" * 70)
print("🎉 BACKFILL COMPLETE!")
print("=" * 70)
print()
print("Next step: Run python src/omerGPT.py")
print()

