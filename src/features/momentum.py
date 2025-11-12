# src/features/momentum.py
"""
Momentum indicators: RSI, MACD, Bollinger Bands.
Implements the technical indicators referenced in §3.1 of the OmerGPT research spec.
"""

import pandas as pd
import numpy as np
import logging

class MomentumFeatures:
    def __init__(self, db, window_short=12, window_long=26, rsi_period=14, bb_period=20):
        self.db = db
        self.logger = logging.getLogger("omerGPT.features.momentum")
        self.window_short = window_short
        self.window_long = window_long
        self.rsi_period = rsi_period
        self.bb_period = bb_period

    def _rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1 / self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _macd(self, prices: pd.Series):
        ema_short = prices.ewm(span=self.window_short, adjust=False).mean()
        ema_long = prices.ewm(span=self.window_long, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    def _bollinger(self, prices: pd.Series):
        ma = prices.rolling(self.bb_period).mean()
        std = prices.rolling(self.bb_period).std()
        upper = ma + (2 * std)
        lower = ma - (2 * std)
        bb_width = (upper - lower) / ma
        return ma, upper, lower, bb_width

    async def compute(self, symbol: str):
        """Compute momentum features for a given symbol and save to DuckDB."""
        query = f"SELECT ts, price FROM price_ticks WHERE symbol = '{symbol}' ORDER BY ts DESC LIMIT 1000"
        df = self.db.query_df(query)
        if df.empty:
            return

        df = df.sort_values("ts")
        prices = df["price"].astype(float)

        df["rsi"] = self._rsi(prices)
        df["macd_line"], df["macd_signal"], df["macd_hist"] = self._macd(prices)
        df["bb_ma"], df["bb_upper"], df["bb_lower"], df["bb_width"] = self._bollinger(prices)

        features = df[["ts", "rsi", "macd_line", "macd_signal", "macd_hist", "bb_width"]].dropna()
        self.db.insert_dataframe("momentum_features", features)
        self.logger.info(f"📈 Momentum features computed for {symbol} ({len(features)} rows)")
