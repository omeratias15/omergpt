-- =====================================================
-- storage/schema.sql
-- Defines DuckDB schema for OmerGPT
-- =====================================================

-- ==================== CORE MARKET DATA ====================
CREATE TABLE IF NOT EXISTS price_ticks (
    ts BIGINT,
    symbol VARCHAR,
    price DOUBLE,
    volume DOUBLE,
    exchange VARCHAR,
    PRIMARY KEY (ts, symbol)
);

CREATE TABLE IF NOT EXISTS candles (
    symbol VARCHAR,
    ts_ms BIGINT,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    close_time BIGINT
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id BIGINT,
    ts BIGINT,
    symbol VARCHAR,
    side VARCHAR,
    price DOUBLE,
    qty DOUBLE
);

CREATE TABLE IF NOT EXISTS orderbook (
    ts BIGINT,
    symbol VARCHAR,
    bid_price DOUBLE,
    ask_price DOUBLE,
    bid_qty DOUBLE,
    ask_qty DOUBLE
);

-- ==================== FEATURES ====================
CREATE TABLE IF NOT EXISTS features (
    ts BIGINT,
    symbol VARCHAR,
    volatility DOUBLE,
    liquidity DOUBLE,
    correlation DOUBLE,
    PRIMARY KEY (ts, symbol)
);

CREATE TABLE IF NOT EXISTS momentum_features (
    ts BIGINT,
    rsi DOUBLE,
    macd_line DOUBLE,
    macd_signal DOUBLE,
    macd_hist DOUBLE,
    bb_width DOUBLE
);

CREATE TABLE IF NOT EXISTS garch_features (
    ts BIGINT,
    symbol VARCHAR,
    vol_forecast_h1 DOUBLE,
    vol_forecast_h24 DOUBLE
);

CREATE TABLE IF NOT EXISTS reddit_sentiment (
    ts_ms BIGINT,
    subreddit VARCHAR,
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

-- ==================== ANOMALIES / SIGNALS ====================
CREATE TABLE IF NOT EXISTS anomaly_events (
    id INTEGER AUTO_INCREMENT,
    ts BIGINT,
    symbol VARCHAR,
    score DOUBLE,
    model VARCHAR,
    severity VARCHAR,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER AUTO_INCREMENT,
    ts BIGINT,
    symbol VARCHAR,
    signal_type VARCHAR,
    confidence DOUBLE,
    details VARCHAR,
    PRIMARY KEY (id)
);

-- ==================== VALIDATION & MONITORING ====================
CREATE TABLE IF NOT EXISTS validation_results (
    ts BIGINT,
    symbol VARCHAR,
    model VARCHAR,
    sharpe DOUBLE,
    information_ratio DOUBLE,
    hit_rate DOUBLE,
    decay DOUBLE,
    latency_ms DOUBLE
);

CREATE TABLE IF NOT EXISTS system_metrics (
    ts BIGINT,
    cpu_usage DOUBLE,
    gpu_usage DOUBLE,
    mem_usage DOUBLE,
    latency_ms DOUBLE,
    status VARCHAR
);

-- ==================== META INFO ====================
CREATE TABLE IF NOT EXISTS meta_info (
    key VARCHAR,
    value VARCHAR
);
