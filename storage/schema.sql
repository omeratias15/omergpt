-- =====================================================
-- storage/schema.sql  (omerGPT unified database schema)
-- Compatible with DuckDB 1.x
-- =====================================================

-- ========== RAW MARKET DATA ==========

CREATE TABLE IF NOT EXISTS price_ticks (
    ts BIGINT,
    symbol TEXT,
    price DOUBLE,
    volume DOUBLE,
    source TEXT,
    PRIMARY KEY (ts, symbol)
);

CREATE INDEX IF NOT EXISTS idx_price_ticks_symbol_ts
    ON price_ticks(symbol, ts);

CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT,
    ts_ms BIGINT,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    close_time BIGINT
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts
    ON candles(symbol, ts_ms);

CREATE TABLE IF NOT EXISTS trades (
    trade_id BIGINT,
    ts BIGINT,
    symbol TEXT,
    side TEXT,
    price DOUBLE,
    qty DOUBLE
);

CREATE TABLE IF NOT EXISTS orderbook (
    ts BIGINT,
    symbol TEXT,
    bid_price DOUBLE,
    ask_price DOUBLE,
    bid_qty DOUBLE,
    ask_qty DOUBLE
);

-- ========== FEATURE TABLES ==========

CREATE TABLE IF NOT EXISTS features (
    ts BIGINT,
    symbol TEXT,
    return DOUBLE,
    volatility DOUBLE,
    rsi DOUBLE,
    price DOUBLE
);

CREATE TABLE IF NOT EXISTS momentum_features (
    ts BIGINT,
    symbol TEXT,
    rsi DOUBLE,
    macd_line DOUBLE,
    macd_signal DOUBLE,
    macd_hist DOUBLE,
    bb_width DOUBLE
);

CREATE TABLE IF NOT EXISTS garch_features (
    ts BIGINT,
    symbol TEXT,
    vol_forecast_h1 DOUBLE,
    vol_forecast_h24 DOUBLE
);

-- ========== SENTIMENT & MACRO DATA ==========

CREATE TABLE IF NOT EXISTS reddit_sentiment (
    ts_ms BIGINT,
    subreddit TEXT,
    sentiment DOUBLE
);

CREATE TABLE IF NOT EXISTS market_overview (
    ts_ms BIGINT,
    active_cryptocurrencies INTEGER,
    markets INTEGER,
    btc_dominance DOUBLE,
    eth_dominance DOUBLE,
    total_mcap_usd DOUBLE,
    total_volume_usd DOUBLE
);

-- ========== ANOMALIES, SIGNALS & VALIDATION ==========

CREATE TABLE IF NOT EXISTS anomaly_events (
    ts BIGINT,
    symbol TEXT,
    event_type TEXT,
    severity TEXT,
    score DOUBLE,
    metadata JSON
);

CREATE INDEX IF NOT EXISTS idx_anomaly_symbol_ts
    ON anomaly_events(symbol, ts);

CREATE TABLE IF NOT EXISTS signals (
    id BIGINT,
    ts BIGINT,
    symbol TEXT,
    signal_type TEXT,
    confidence DOUBLE,
    details TEXT
);

CREATE TABLE IF NOT EXISTS validation_results (
    ts BIGINT,
    symbol TEXT,
    model TEXT,
    sharpe DOUBLE,
    information_ratio DOUBLE,
    hit_rate DOUBLE,
    decay DOUBLE,
    latency_ms DOUBLE
);

-- ========== SYSTEM HEALTH & METRICS ==========

CREATE TABLE IF NOT EXISTS system_metrics (
    ts BIGINT,
    cpu_usage DOUBLE,
    gpu_usage DOUBLE,
    mem_usage DOUBLE,
    latency_ms DOUBLE,
    status TEXT
);

CREATE TABLE IF NOT EXISTS meta_info (
    key TEXT,
    value TEXT
);

CREATE TABLE IF NOT EXISTS onchain_metrics (
    timestamp TIMESTAMP,
    block_number BIGINT,
    gas_used BIGINT,
    gas_price BIGINT,
    tx_count INTEGER
);

-- =====================================================
-- End of schema
-- =====================================================
