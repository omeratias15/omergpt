"""
src/adaptive_learning/trainer.py
Training orchestrator for the Adaptive Agent.
Loads historical data from DuckDB, normalizes features, runs training loops,
and saves checkpoints with comprehensive logging and GPU monitoring.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.adaptive_learning.adaptive_agent import AdaptiveAgent
from storage.db_manager import DatabaseManager

logger = logging.getLogger("omerGPT.adaptive_learning.trainer")


class FeatureNormalizer:
    """
    Feature normalization and preprocessing for training data.
    Supports min-max scaling, z-score normalization, and robust scaling.
    """
    
    def __init__(self, method: str = "minmax"):
        """
        Initialize feature normalizer.
        
        Args:
            method: Normalization method ('minmax', 'zscore', 'robust')
        """
        self.method = method
        self.stats = {}
        logger.info(f"FeatureNormalizer initialized: method={method}")
    
    def fit(self, data: pd.DataFrame, columns: List[str]):
        """
        Compute normalization statistics from training data.
        
        Args:
            data: Training dataframe
            columns: List of columns to normalize
        """
        for col in columns:
            if col not in data.columns:
                logger.warning(f"Column {col} not found in data")
                continue
            
            values = data[col].dropna().values
            
            if self.method == "minmax":
                self.stats[col] = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            elif self.method == "zscore":
                self.stats[col] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            elif self.method == "robust":
                self.stats[col] = {
                    "median": float(np.median(values)),
                    "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                }
        
        logger.info(f"Fitted normalizer on {len(self.stats)} features")
    
    def transform(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Apply normalization to data.
        
        Args:
            data: Input dataframe
            columns: Columns to normalize
        
        Returns:
            Normalized dataframe
        """
        normalized = data.copy()
        
        for col in columns:
            if col not in self.stats:
                logger.warning(f"No stats for column {col}, skipping")
                continue
            
            if col not in normalized.columns:
                continue
            
            stats = self.stats[col]
            
            if self.method == "minmax":
                min_val = stats["min"]
                max_val = stats["max"]
                if max_val > min_val:
                    normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
                else:
                    normalized[col] = 0.0
            
            elif self.method == "zscore":
                mean_val = stats["mean"]
                std_val = stats["std"]
                if std_val > 0:
                    normalized[col] = (normalized[col] - mean_val) / std_val
                else:
                    normalized[col] = 0.0
            
            elif self.method == "robust":
                median_val = stats["median"]
                iqr_val = stats["iqr"]
                if iqr_val > 0:
                    normalized[col] = (normalized[col] - median_val) / iqr_val
                else:
                    normalized[col] = 0.0
        
        return normalized
    
    def save(self, filepath: str):
        """Save normalizer statistics to JSON."""
        with open(filepath, 'w') as f:
            json.dump({
                "method": self.method,
                "stats": self.stats,
            }, f, indent=2)
        logger.info(f"Normalizer saved: {filepath}")
    
    def load(self, filepath: str):
        """Load normalizer statistics from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.method = data["method"]
            self.stats = data["stats"]
        logger.info(f"Normalizer loaded: {filepath}")


class DataLoader:
    """
    Load and prepare training data from DuckDB.
    Combines candles, features, sentiment, and macro data.
    """
    
    def __init__(self, db_manager: DBManager):
        """
        Initialize data loader.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
        logger.info("DataLoader initialized")
    
    async def load_training_data(
        self,
        symbol: str = "BTCUSDT",
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        Load historical data for training.
        
        Args:
            symbol: Trading symbol
            lookback_days: Number of days to load
        
        Returns:
            Combined dataframe with all features
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        logger.info(
            f"Loading training data: {symbol}, "
            f"{start_time.date()} to {end_time.date()}"
        )
        
        # Load candles
        candles = await self.db.query(
            """
            SELECT * FROM candles
            WHERE symbol = ?
            AND timestamp >= ?
            AND timestamp <= ?
            ORDER BY timestamp
            """,
            (symbol, int(start_time.timestamp() * 1000), int(end_time.timestamp() * 1000))
        )
        
        if not candles:
            logger.warning("No candles found")
            return pd.DataFrame()
        
        df = pd.DataFrame(candles)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Load features
        features = await self.db.query(
            """
            SELECT * FROM features
            WHERE symbol = ?
            AND timestamp >= ?
            AND timestamp <= ?
            ORDER BY timestamp
            """,
            (symbol, int(start_time.timestamp() * 1000), int(end_time.timestamp() * 1000))
        )
        
        if features:
            features_df = pd.DataFrame(features)
            df = df.merge(
                features_df,
                on=['symbol', 'timestamp'],
                how='left',
                suffixes=('', '_feat')
            )
        
        # Load sentiment data
        sentiment = await self.db.query(
            """
            SELECT date, sentiment_index
            FROM sentiment_daily
            WHERE date >= ?
            AND date <= ?
            ORDER BY date
            """,
            (start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d'))
        )
        
        if sentiment:
            sentiment_df = pd.DataFrame(sentiment)
            sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
            df['date'] = df['datetime'].dt.date
            df['date'] = pd.to_datetime(df['date'])
            
            df = df.merge(
                sentiment_df,
                on='date',
                how='left'
            )
            df['sentiment_index'] = df['sentiment_index'].fillna(0.0)
        else:
            df['sentiment_index'] = 0.0
        
        # Load macro features
        macro = await self.db.query(
            """
            SELECT date, risk_index
            FROM macro_features
            WHERE date >= ?
            AND date <= ?
            ORDER BY date
            """,
            (start_time.strftime('%Y-%m-%d'), end_time.strftime('%Y-%m-%d'))
        )
        
        if macro:
            macro_df = pd.DataFrame(macro)
            macro_df['date'] = pd.to_datetime(macro_df['date'])
            
            df = df.merge(
                macro_df,
                on='date',
                how='left'
            )
            df['risk_index'] = df['risk_index'].fillna(0.0)
        else:
            df['risk_index'] = 0.0
        
        # Compute derived features
        df = self._compute_derived_features(df)
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        return df
    
    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute additional derived features.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dataframe with additional features
        """
        if len(df) < 2:
            return df
        
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(periods=5).fillna(0.0)
        
        # Volume ratio
        df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        
        # Time features
        df['hour_of_day'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Volatility percentile (if ATR available)
        if 'atr' in df.columns:
            df['atr_pct'] = (df['atr'] / df['close']) * 100
            df['atr_percentile'] = df['atr'].rolling(
                window=100,
                min_periods=20
            ).apply(
                lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) > 0 else 50
            )
            df['atr_percentile'] = df['atr_percentile'].fillna(50.0)
        else:
            df['atr_pct'] = 2.0
            df['atr_percentile'] = 50.0
        
        return df


class EpisodeGenerator:
    """
    Generate training episodes from historical data.
    Each episode simulates a trading period with state-action-reward sequences.
    """
    
    def __init__(
        self,
        episode_length: int = 20,
        reward_horizon: int = 5,
    ):
        """
        Initialize episode generator.
        
        Args:
            episode_length: Number of steps per episode
            reward_horizon: Lookahead for computing rewards
        """
        self.episode_length = episode_length
        self.reward_horizon = reward_horizon
        logger.info(
            f"EpisodeGenerator initialized: length={episode_length}, "
            f"horizon={reward_horizon}"
        )
    
    def generate_episodes(
        self,
        data: pd.DataFrame,
        agent: AdaptiveAgent,
        num_episodes: int = 100,
    ) -> List[Dict]:
        """
        Generate training episodes from data.
        
        Args:
            data: Historical dataframe
            agent: Agent for encoding states
            num_episodes: Number of episodes to generate
        
        Returns:
            List of episode dictionaries
        """
        episodes = []
        
        if len(data) < self.episode_length + self.reward_horizon:
            logger.warning("Insufficient data for episode generation")
            return episodes
        
        max_start_idx = len(data) - self.episode_length - self.reward_horizon
        
        for ep_idx in range(num_episodes):
            # Random starting point
            start_idx = np.random.randint(0, max_start_idx)
            end_idx = start_idx + self.episode_length
            
            episode_data = data.iloc[start_idx:end_idx].copy()
            
            states = []
            actions = []
            rewards = []
            
            for i, row in episode_data.iterrows():
                # Encode state
                features = {
                    "atr_pct": row.get("atr_pct", 2.0),
                    "atr_percentile": row.get("atr_percentile", 50.0),
                    "sentiment_score": row.get("sentiment_index", 0.0),
                    "risk_index": row.get("risk_index", 0.0),
                    "volume_ratio": row.get("volume_ratio", 1.0),
                    "price_momentum": row.get("price_momentum", 0.0),
                    "hour_of_day": row.get("hour_of_day", 12),
                    "day_of_week": row.get("day_of_week", 3),
                    "recent_signal_count": 0,
                    "win_rate": 0.5,
                }
                
                state = agent.encode_state(features)
                states.append(state)
                
                # Generate random action for now (will be replaced by agent policy)
                action = np.random.randn(agent.action_dim).astype(np.float32) * 0.5
                action = np.tanh(action)
                actions.append(action)
                
                # Compute reward based on future price movement
                current_idx = episode_data.index.get_loc(i)
                future_idx = min(
                    current_idx + self.reward_horizon,
                    len(data) - 1
                )
                
                current_price = row.get("close", 0.0)
                future_row = data.iloc[start_idx + future_idx]
                future_price = future_row.get("close", current_price)
                
                if current_price > 0:
                    price_return = (future_price - current_price) / current_price
                else:
                    price_return = 0.0
                
                # Decode action to parameters
                params = agent.decode_action(action)
                
                # Reward = Sharpe-like metric considering action appropriateness
                # Higher volatility threshold in high-vol regime = good
                # Lower confidence in uncertain conditions = good
                vol_appropriate = 1.0 if features["atr_percentile"] > 70 else -0.5
                confidence_penalty = -0.1 if params["min_confidence"] < 0.5 else 0.0
                
                reward = (
                    price_return * 10.0 +  # Price movement
                    vol_appropriate * 0.2 +
                    confidence_penalty
                )
                
                rewards.append(float(reward))
            
            episodes.append({
                "states": np.array(states, dtype=np.float32),
                "actions": np.array(actions, dtype=np.float32),
                "rewards": rewards,
            })
        
        logger.info(f"Generated {len(episodes)} episodes")
        
        return episodes


class GPUMonitor:
    """
    Monitor GPU usage during training.
    """
    
    def __init__(self):
        """Initialize GPU monitor."""
        self.enabled = CUDA_AVAILABLE
        
        if self.enabled:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.pynvml = pynvml
                logger.info("GPU monitoring enabled")
            except Exception as e:
                logger.warning(f"GPU monitoring unavailable: {e}")
                self.enabled = False
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get current GPU statistics.
        
        Returns:
            Dictionary with GPU metrics
        """
        if not self.enabled:
            return {}
        
        try:
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            temp = self.pynvml.nvmlDeviceGetTemperature(
                self.handle,
                self.pynvml.NVML_TEMPERATURE_GPU
            )
            
            return {
                "gpu_util_pct": util.gpu,
                "mem_used_mb": mem_info.used / (1024 ** 2),
                "mem_total_mb": mem_info.total / (1024 ** 2),
                "mem_util_pct": (mem_info.used / mem_info.total) * 100,
                "temp_c": temp,
            }
        except Exception as e:
            logger.warning(f"GPU stats error: {e}")
            return {}


class Trainer:
    """
    Main training orchestrator for Adaptive Agent.
    Manages data loading, episode generation, training loop, and checkpointing.
    """
    
    def __init__(
        self,
        db_path: str = "data/omergpt.db",
        model_dir: str = "models",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        """
        Initialize trainer.
        
        Args:
            db_path: Path to DuckDB database
            model_dir: Directory for final models
            checkpoint_dir: Directory for training checkpoints
            log_dir: Directory for training logs
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install: pip install torch")
        
        self.db_path = db_path
        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize components
        self.db_manager = None
        self.agent = None
        self.normalizer = None
        self.data_loader = None
        self.episode_generator = None
        self.gpu_monitor = GPUMonitor()
        
        # Training state
        self.training_log = {
            "start_time": None,
            "end_time": None,
            "epochs": [],
            "losses": [],
            "rewards": [],
            "gpu_stats": [],
        }
        
        logger.info(
            f"Trainer initialized: db={db_path}, "
            f"model_dir={model_dir}, checkpoint_dir={checkpoint_dir}"
        )
    
    async def setup(
        self,
        state_dim: int = 10,
        action_dim: int = 3,
        learning_rate: float = 3e-4,
        use_gpu: bool = True,
    ):
        """
        Setup trainer components.
        
        Args:
            state_dim: State vector dimension
            action_dim: Action vector dimension
            learning_rate: Learning rate for optimizer
            use_gpu: Whether to use GPU acceleration
        """
        # Initialize database manager
        self.db_manager = DatabaseManager(self.db_path)
        
        # Initialize agent
        self.agent = AdaptiveAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            use_gpu=use_gpu,
            model_dir=self.model_dir,
        )
        
        # Initialize normalizer
        self.normalizer = FeatureNormalizer(method="minmax")
        
        # Initialize data loader
        self.data_loader = DataLoader(self.db_manager)
        
        # Initialize episode generator
        self.episode_generator = EpisodeGenerator(
            episode_length=20,
            reward_horizon=5,
        )
        
        logger.info("Trainer setup complete")
    
    async def load_and_prepare_data(
        self,
        symbol: str = "BTCUSDT",
        lookback_days: int = 90,
        train_split: float = 0.8,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare training/validation data.
        
        Args:
            symbol: Trading symbol
            lookback_days: Days of historical data
            train_split: Fraction of data for training
        
        Returns:
            Tuple of (train_df, val_df)
        """
        # Load data
        data = await self.data_loader.load_training_data(
            symbol=symbol,
            lookback_days=lookback_days,
        )
        
        if len(data) == 0:
            raise ValueError("No data loaded")
        
        # Split train/validation
        split_idx = int(len(data) * train_split)
        train_data = data.iloc[:split_idx].copy()
        val_data = data.iloc[split_idx:].copy()
        
        # Fit normalizer on training data
        feature_cols = [
            'atr_pct', 'atr_percentile', 'sentiment_index', 'risk_index',
            'volume_ratio', 'price_momentum', 'hour_of_day', 'day_of_week',
        ]
        
        available_cols = [col for col in feature_cols if col in train_data.columns]
        self.normalizer.fit(train_data, available_cols)
        
        # Normalize both datasets
        train_data = self.normalizer.transform(train_data, available_cols)
        val_data = self.normalizer.transform(val_data, available_cols)
        
        # Save normalizer
        normalizer_path = os.path.join(self.model_dir, "normalizer.json")
        self.normalizer.save(normalizer_path)
        
        logger.info(
            f"Data prepared: train={len(train_data)}, val={len(val_data)}"
        )
        
        return train_data, val_data
    
    async def train(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        episodes_per_epoch: int = 100,
        val_frequency: int = 5,
        checkpoint_frequency: int = 10,
    ):
        """
        Run training loop.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            episodes_per_epoch: Episodes to generate per epoch
            val_frequency: Validation frequency (epochs)
            checkpoint_frequency: Checkpoint save frequency (epochs)
        """
        self.training_log["start_time"] = datetime.now().isoformat()
        
        logger.info(
            f"Starting training: epochs={epochs}, batch={batch_size}, "
            f"episodes_per_epoch={episodes_per_epoch}"
        )
        
        # Load data
        train_data, val_data = await self.load_and_prepare_data()
        
        best_reward = -float('inf')
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Generate episodes
            episodes = self.episode_generator.generate_episodes(
                data=train_data,
                agent=self.agent,
                num_episodes=episodes_per_epoch,
            )
            
            # Add to agent buffer
            for episode in episodes:
                self.agent.buffer.add_episode(episode)
            
            # Train
            metrics = await self.agent.train_step(batch_size=batch_size)
            
            # Get GPU stats
            gpu_stats = self.gpu_monitor.get_stats()
            
            epoch_time = time.time() - epoch_start
            
            # Log
            self.training_log["epochs"].append(epoch)
            self.training_log["losses"].append(metrics["loss"])
            self.training_log["rewards"].append(metrics["avg_reward"])
            self.training_log["gpu_stats"].append(gpu_stats)
            
            log_msg = (
                f"Epoch {epoch}/{epochs} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Reward: {metrics['avg_reward']:.4f} | "
                f"Buffer: {metrics['buffer_size']} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            if gpu_stats:
                log_msg += (
                    f" | GPU: {gpu_stats.get('gpu_util_pct', 0):.0f}% | "
                    f"VRAM: {gpu_stats.get('mem_util_pct', 0):.0f}%"
                )
            
            logger.info(log_msg)
            
            # Validation
            if epoch % val_frequency == 0:
                val_episodes = self.episode_generator.generate_episodes(
                    data=val_data,
                    agent=self.agent,
                    num_episodes=20,
                )
                
                val_states = [ep["states"][0] for ep in val_episodes]
                val_metrics = await self.agent.evaluate(val_states)
                
                logger.info(
                    f"Validation | Avg Reward: {val_metrics['avg_reward']:.4f}"
                )
                
                # Save best model
                if val_metrics['avg_reward'] > best_reward:
                    best_reward = val_metrics['avg_reward']
                    best_model_path = self.agent.save_model("best_agent.pt")
                    logger.info(f"New best model saved: {best_model_path}")
            
            # Checkpoint
            if epoch % checkpoint_frequency == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"agent_epoch_{epoch}.pt"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.agent.policy.state_dict(),
                    "optimizer_state_dict": self.agent.optimizer.state_dict(),
                    "loss": metrics["loss"],
                    "reward": metrics["avg_reward"],
                }, checkpoint_path)
                
                logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        self.training_log["end_time"] = datetime.now().isoformat()
        
        # Save final model
        final_model_path = self.agent.save_model("adaptive_agent.pt")
        logger.info(f"Final model saved: {final_model_path}")
        
        # Save training log
        log_path = os.path.join(
            self.log_dir,
            f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        logger.info(f"Training log saved: {log_path}")
        logger.info("Training complete!")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.db_manager:
            await self.db_manager.close()
        logger.info("Trainer cleanup complete")


async def main(args):
    """Main training entrypoint."""
    # Configure logging
    log_file = os.path.join(
        "logs",
        f"trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    
    logger.info("="*80)
    logger.info("omerGPT Adaptive Learning Trainer")
    logger.info("="*80)
    logger.info(f"PyTorch: {TORCH_AVAILABLE}")
    logger.info(f"CUDA: {CUDA_AVAILABLE}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info("="*80)
    
    # Initialize trainer
    trainer = Trainer(
        db_path=args.db_path,
        model_dir=args.model_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    
    # Setup
    await trainer.setup(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        learning_rate=args.lr,
        use_gpu=args.use_gpu,
    )
    
    try:
        # Train
        await trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            episodes_per_epoch=args.episodes_per_epoch,
            val_frequency=args.val_frequency,
            checkpoint_frequency=args.checkpoint_frequency,
        )
    finally:
        # Cleanup
        await trainer.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Adaptive Agent for omerGPT"
    )
    
    # Data arguments
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/omergpt.db",
        help="Path to DuckDB database"
    )
    
    # Model arguments
    parser.add_argument(
        "--state-dim",
        type=int,
        default=10,
        help="State vector dimension"
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=3,
        help="Action vector dimension"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--episodes-per-epoch",
        type=int,
        default=100,
        help="Episodes to generate per epoch"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    
    # Validation/checkpoint arguments
    parser.add_argument(
        "--val-frequency",
        type=int,
        default=5,
        help="Validation frequency (epochs)"
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=10,
        help="Checkpoint save frequency (epochs)"
    )
    
    # GPU arguments
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU acceleration"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="use_gpu",
        help="Disable GPU acceleration"
    )
    
    # Directory arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory for final models"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for training checkpoints"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for training logs"
    )
    
    args = parser.parse_args()
    
    # Run training
    asyncio.run(main(args))
