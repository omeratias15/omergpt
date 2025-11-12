# requirement: numpy, pandas, torch, arch, pyyaml
import os
import logging
import numpy as np
import pandas as pd
import torch
from arch import arch_model
import yaml
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VolatilityEngine:
    """
    Full volatility computation engine for Holy Grail Platform.

    Computes:
      - Realized volatility
      - Historical volatility
      - GARCH(1,1) conditional volatility
      - Volatility-of-Volatility / Decay
      - Normalized ATR + regime detection
    Includes:
      - GPU acceleration (optional)
      - Timestamp alignment across exchanges
      - Feature store persistence
      - Drift monitor integration
      - Statistical validation of research consistency
    """

    def __init__(self, *args, use_gpu=False, **kwargs):
        import os
        import yaml

        config_path = kwargs.get("config_path", os.path.join(os.path.dirname(__file__), "../../configs/config.yaml"))
        config_path = os.path.abspath(config_path)

        if not os.path.exists(config_path):
            print(f"[WARNING] Config file not found at {config_path}. Using default settings.")
            self.config = {}
        else:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

        self.window = self.config.get("features", {}).get("vol_window", 300)
        self.garch_enabled = self.config.get("features", {}).get("garch", True)
        self.decay_window = self.config.get("features", {}).get("decay_window", 50)
        self.store_path = self.config.get("features", {}).get("store_path", "data/features/volatility.parquet")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Volatility engine initialized on {self.device}")

    # ---------------------------
    # Timestamp alignment
    # ---------------------------
    @staticmethod
    def align_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        """
        Align timestamps from multiple exchanges to unified 1-second UTC clock.
        """
        df = df.copy()
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.floor("1s")
        df = (
            df.groupby("timestamp")
            .agg({"high": "max", "low": "min", "close": "last"})
            .dropna()
            .reset_index()
        )
        return df

    # ---------------------------
    # Cross-exchange merging
    # ---------------------------
    @staticmethod
    def merge_cross_exchange(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Merge synchronized volatility features from two exchanges (e.g. Binance & Kraken)
        into a unified frame based on timestamp alignment (1s tolerance).
        """
        df1["timestamp"] = pd.to_datetime(df1["timestamp"])
        df2["timestamp"] = pd.to_datetime(df2["timestamp"])
        merged = pd.merge_asof(
            df1.sort_values("timestamp"),
            df2.sort_values("timestamp"),
            on="timestamp",
            suffixes=("_binance", "_kraken"),
            tolerance=pd.Timedelta("1s"),
        )
        return merged.dropna(subset=["close_binance", "close_kraken"])

    # ---------------------------
    # Volatility computations
    # ---------------------------
    @staticmethod
    def realized_volatility(prices: np.ndarray, window: int) -> float:
        log_ret = np.diff(np.log(prices[-window:]))
        return np.sqrt(np.sum(log_ret**2))

    @staticmethod
    def historical_volatility(prices: np.ndarray, window: int) -> float:
        log_ret = np.diff(np.log(prices[-window:]))
        return np.std(log_ret) * np.sqrt(252)

    @staticmethod
    def garch_volatility(returns: np.ndarray) -> float:
        if len(returns) < 30:
            return np.nan
        am = arch_model(returns * 100, mean="Zero", vol="GARCH", p=1, q=1)
        res = am.fit(disp="off")
        return res.conditional_volatility[-1] / 100

    @staticmethod
    def volatility_decay(vol_series: np.ndarray, window: int = 50) -> np.ndarray:
        """
        Rolling autocorrelation (lag-1) of volatility change.
        """
        if len(vol_series) < window + 1:
            return np.full_like(vol_series, np.nan)
        vol_diff = np.diff(vol_series, prepend=np.nan)
        decay = pd.Series(vol_diff).rolling(window=window).apply(
            lambda x: x.autocorr(lag=1), raw=False
        )
        return decay.to_numpy()

    # ---------------------------
    # ATR & Regime Detection
    # ---------------------------
    @staticmethod
    def average_true_range(df: pd.DataFrame, window: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        tr = ranges.max(axis=1)
        atr = tr.rolling(window).mean()
        return atr

    @staticmethod
    def detect_vol_regime(series: pd.Series, lookback: int = 300) -> pd.Series:
        """
        Map ATR percentile to regime: Calm / Normal / Turbulent / Extreme
        """
        percentiles = series.rolling(lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        regime = pd.cut(
            percentiles,
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["Calm", "Normal", "Turbulent", "Extreme"],
        )
        return regime

    # ---------------------------
    # GPU computation
    # ---------------------------
    def gpu_volatility(self, prices: torch.Tensor, window: int) -> torch.Tensor:
        log_prices = torch.log(prices)
        returns = log_prices[1:] - log_prices[:-1]
        sq = returns[-window:] ** 2
        return torch.sqrt(torch.sum(sq))

    # ---------------------------
    # Drift monitor integration
    # ---------------------------
    def publish_to_drift_monitor(self, df: pd.DataFrame):
        """
        Simulated integration with drift monitor.
        In real system, this would push to an async message queue or shared feature bus.
        """
        drift_input = df[["timestamp", "vol_decay", "regime"]].dropna().tail(1)
        if not drift_input.empty:
            logger.info(
                f"Publishing drift update → timestamp={drift_input['timestamp'].iloc[0]}, "
                f"decay={drift_input['vol_decay'].iloc[0]:.4f}, regime={drift_input['regime'].iloc[0]}"
            )

    # ---------------------------
    # Main pipeline
    # ---------------------------
    def calculate_volatility_metrics(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Compute all volatility metrics, validate results, and persist to feature store.
        """
        df = self.align_timestamps(df)
        prices = df["close"].to_numpy()
        log_ret = np.diff(np.log(prices))

        df["realized_vol"] = self.realized_volatility(prices, self.window)
        df["historical_vol"] = self.historical_volatility(prices, self.window)

        if self.garch_enabled:
            df["garch_vol"] = self.garch_volatility(log_ret)
        else:
            df["garch_vol"] = np.nan

        vol_series = df["realized_vol"].ffill().to_numpy()
        df["vol_decay"] = self.volatility_decay(vol_series, self.decay_window)

        df["atr"] = self.average_true_range(df, window=self.window)
        df["regime"] = self.detect_vol_regime(df["atr"], lookback=self.window)

        self.save_features(df, symbol)
        self.publish_to_drift_monitor(df)
        self.validate_statistics(df)

        return df

    # ---------------------------
    # Persistence
    # ---------------------------
    def save_features(self, df: pd.DataFrame, symbol: str):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        df.assign(symbol=symbol).to_parquet(self.store_path, index=False)
        logger.info(f"Saved volatility features for {symbol} → {self.store_path}")
        return True

    # ---------------------------
    # Statistical validation
    # ---------------------------
    def validate_statistics(self, df: pd.DataFrame):
        """
        Ensure numerical consistency between realized and GARCH volatilities,
        as required by §6.1 of the research paper.
        """
        if "garch_vol" in df.columns and df["garch_vol"].notna().sum() > 0:
            diff = abs(df["realized_vol"].mean() - df["garch_vol"].mean())
            if diff > 0.05:
                logger.warning(
                    f"Volatility divergence detected (mean diff={diff:.4f}) — "
                    "may indicate model drift."
                )
            else:
                logger.info("Volatility metrics validated successfully (research-compliant).")

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    import yfinance as yf

    data = yf.download("BTC-USD", period="5d", interval="1h")
    data = data.reset_index().rename(columns={"Datetime": "timestamp"})

    engine = VolatilityEngine()
    metrics = engine.calculate_volatility_metrics(data, symbol="BTCUSDT")
    print(metrics.tail(3))
