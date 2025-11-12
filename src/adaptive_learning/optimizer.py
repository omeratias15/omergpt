"""
src/adaptive_learning/optimizer.py
Adaptive parameter manager for live production runs.
Monitors for new trained models, loads them, and dynamically updates
signal engine thresholds based on current market conditions.
Runs as async background task with hourly model checks and real-time parameter adjustment.
"""
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.adaptive_learning.adaptive_agent import AdaptiveAgent

logger = logging.getLogger("omerGPT.adaptive_learning.optimizer")


class ModelRegistry:
    """
    Registry for tracking trained models and their metadata.
    Monitors model directory for new checkpoints.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize model registry.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.last_scan_time = None
        
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"ModelRegistry initialized: {model_dir}")
    
    def scan_models(self) -> List[Dict]:
        """
        Scan model directory for available models.
        
        Returns:
            List of model metadata dictionaries
        """
        models = []
        
        for filename in os.listdir(self.model_dir):
            if not filename.endswith('.pt'):
                continue
            
            filepath = os.path.join(self.model_dir, filename)
            
            try:
                stat = os.stat(filepath)
                
                model_info = {
                    "filename": filename,
                    "filepath": filepath,
                    "size_mb": stat.st_size / (1024 ** 2),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime),
                    "created_time": datetime.fromtimestamp(stat.st_ctime),
                }
                
                # Try to load model metadata
                if TORCH_AVAILABLE:
                    try:
                        checkpoint = torch.load(
                            filepath,
                            map_location='cpu',
                            weights_only=False
                        )
                        
                        model_info["epoch"] = checkpoint.get("epoch")
                        model_info["loss"] = checkpoint.get("loss")
                        model_info["reward"] = checkpoint.get("reward")
                        model_info["state_dim"] = checkpoint.get("state_dim")
                        model_info["action_dim"] = checkpoint.get("action_dim")
                    except Exception as e:
                        logger.warning(f"Could not load metadata from {filename}: {e}")
                
                models.append(model_info)
                self.models[filename] = model_info
            
            except Exception as e:
                logger.warning(f"Error scanning {filename}: {e}")
        
        self.last_scan_time = datetime.now()
        
        logger.info(f"Scanned {len(models)} models")
        
        return models
    
    def get_latest_model(self, pattern: str = "best_agent.pt") -> Optional[Dict]:
        """
        Get latest model matching pattern.
        
        Args:
            pattern: Filename pattern to match
        
        Returns:
            Model info dictionary or None
        """
        self.scan_models()
        
        matching = [
            m for m in self.models.values()
            if pattern in m["filename"]
        ]
        
        if not matching:
            return None
        
        # Sort by modified time, most recent first
        matching.sort(key=lambda x: x["modified_time"], reverse=True)
        
        return matching[0]
    
    def get_best_model(self) -> Optional[Dict]:
        """
        Get best performing model based on reward metric.
        
        Returns:
            Model info dictionary or None
        """
        self.scan_models()
        
        models_with_reward = [
            m for m in self.models.values()
            if m.get("reward") is not None
        ]
        
        if not models_with_reward:
            return self.get_latest_model()
        
        # Sort by reward, highest first
        models_with_reward.sort(key=lambda x: x["reward"], reverse=True)
        
        return models_with_reward[0]


class ParameterSmoother:
    """
    Smooth parameter transitions to avoid abrupt threshold changes.
    Uses exponential moving average for gradual adaptation.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize parameter smoother.
        
        Args:
            alpha: Smoothing factor (0 = no update, 1 = full update)
        """
        self.alpha = alpha
        self.current_params = {}
        logger.info(f"ParameterSmoother initialized: alpha={alpha}")
    
    def update(
        self,
        new_params: Dict[str, float],
        force: bool = False
    ) -> Dict[str, float]:
        """
        Smooth parameter update using EMA.
        
        Args:
            new_params: New parameter values
            force: If True, skip smoothing and set directly
        
        Returns:
            Smoothed parameters
        """
        if not self.current_params or force:
            self.current_params = new_params.copy()
            return self.current_params
        
        smoothed = {}
        
        for key, new_val in new_params.items():
            if key in self.current_params:
                old_val = self.current_params[key]
                smoothed[key] = old_val + self.alpha * (new_val - old_val)
            else:
                smoothed[key] = new_val
        
        self.current_params = smoothed
        
        return smoothed
    
    def reset(self):
        """Reset smoother state."""
        self.current_params = {}


class AdaptiveOptimizer:
    """
    Adaptive parameter manager for live trading system.
    
    Responsibilities:
    - Monitor for new trained models
    - Load latest/best model
    - Compute optimal parameters based on current market state
    - Smooth parameter transitions
    - Update signal engine thresholds
    - Log all parameter changes
    """
    
    def __init__(
        self,
        model_dir: str = "models",
        log_file: str = "adaptive_log.json",
        check_interval: int = 3600,  # 1 hour
        smoothing_alpha: float = 0.3,
    ):
        """
        Initialize adaptive optimizer.
        
        Args:
            model_dir: Directory containing trained models
            log_file: Path to parameter change log
            check_interval: Model check interval (seconds)
            smoothing_alpha: Parameter smoothing factor
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, optimizer running in fallback mode")
        
        self.model_dir = model_dir
        self.log_file = log_file
        self.check_interval = check_interval
        
        # Initialize components
        self.registry = ModelRegistry(model_dir)
        self.smoother = ParameterSmoother(alpha=smoothing_alpha)
        self.agent: Optional[AdaptiveAgent] = None
        
        # Current state
        self.current_model_path = None
        self.current_params = {}
        self.last_update_time = None
        self.is_running = False
        
        # Parameter history
        self.param_history = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            "updates_count": 0,
            "model_loads": 0,
            "errors": 0,
            "last_error": None,
        }
        
        # Default fallback parameters
        self.default_params = {
            "atr_threshold": 2.0,
            "volume_threshold": 1.5,
            "min_confidence": 0.6,
        }
        
        logger.info(
            f"AdaptiveOptimizer initialized: check_interval={check_interval}s, "
            f"smoothing={smoothing_alpha}"
        )
    
    async def initialize(
        self,
        state_dim: int = 10,
        action_dim: int = 3,
        use_gpu: bool = True,
    ):
        """
        Initialize optimizer and load initial model.
        
        Args:
            state_dim: State vector dimension
            action_dim: Action vector dimension
            use_gpu: Whether to use GPU
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using default parameters")
            self.current_params = self.default_params.copy()
            return
        
        # Initialize agent
        self.agent = AdaptiveAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            use_gpu=use_gpu,
            model_dir=self.model_dir,
        )
        
        # Try to load best available model
        await self._load_best_model()
        
        # Initialize with default if no model loaded
        if not self.current_model_path:
            logger.warning("No trained model found, using defaults")
            self.current_params = self.default_params.copy()
            self.smoother.update(self.current_params, force=True)
        
        logger.info("AdaptiveOptimizer initialization complete")
    
    async def _load_best_model(self) -> bool:
        """
        Load best available model from registry.
        
        Returns:
            True if model loaded successfully
        """
        best_model = self.registry.get_best_model()
        
        if not best_model:
            logger.warning("No models found in registry")
            return False
        
        model_path = best_model["filepath"]
        
        # Check if already loaded
        if model_path == self.current_model_path:
            logger.debug(f"Model already loaded: {model_path}")
            return True
        
        try:
            self.agent.load_model(model_path)
            self.current_model_path = model_path
            self.stats["model_loads"] += 1
            
            logger.info(
                f"Loaded model: {best_model['filename']} | "
                f"Reward: {best_model.get('reward', 'N/A')} | "
                f"Modified: {best_model['modified_time']}"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            return False
    
    async def get_optimal_params(
        self,
        current_state: Dict
    ) -> Dict[str, float]:
        """
        Compute optimal parameters for current market state.
        
        Args:
            current_state: Current market features
        
        Returns:
            Optimal parameter dictionary
        """
        if not TORCH_AVAILABLE or self.agent is None:
            return self.default_params.copy()
        
        try:
            # Encode state
            state_vector = self.agent.encode_state(current_state)
            
            # Get action from policy (deterministic)
            action, _ = await self.agent.select_action(
                state_vector,
                deterministic=True
            )
            
            # Decode to parameters
            params = self.agent.decode_action(action)
            
            # Apply smoothing
            smoothed_params = self.smoother.update(params)
            
            return smoothed_params
        
        except Exception as e:
            logger.error(f"Error computing optimal params: {e}")
            self.stats["errors"] += 1
            self.stats["last_error"] = str(e)
            return self.default_params.copy()
    
    async def update_parameters(
        self,
        market_state: Dict,
        force_update: bool = False
    ) -> Dict[str, float]:
        """
        Update trading parameters based on current market state.
        
        Args:
            market_state: Current market features
            force_update: Force update even if recently updated
        
        Returns:
            Updated parameters
        """
        # Check if update needed
        if not force_update and self.last_update_time:
            time_since_update = (
                datetime.now() - self.last_update_time
            ).total_seconds()
            
            if time_since_update < 60:  # Minimum 1 minute between updates
                return self.current_params
        
        # Get optimal parameters
        new_params = await self.get_optimal_params(market_state)
        
        # Check for significant change
        param_changed = self._check_param_change(new_params)
        
        if param_changed or force_update:
            self.current_params = new_params
            self.last_update_time = datetime.now()
            self.stats["updates_count"] += 1
            
            # Log change
            await self._log_param_change(market_state, new_params)
            
            logger.info(
                f"Parameters updated | "
                f"ATR: {new_params['atr_threshold']:.3f} | "
                f"Vol: {new_params['volume_threshold']:.3f} | "
                f"Conf: {new_params['min_confidence']:.3f}"
            )
        
        return self.current_params
    
    def _check_param_change(
        self,
        new_params: Dict[str, float],
        threshold: float = 0.05
    ) -> bool:
        """
        Check if parameters changed significantly.
        
        Args:
            new_params: New parameter values
            threshold: Relative change threshold
        
        Returns:
            True if significant change detected
        """
        if not self.current_params:
            return True
        
        for key, new_val in new_params.items():
            old_val = self.current_params.get(key, new_val)
            
            if old_val == 0:
                continue
            
            rel_change = abs(new_val - old_val) / abs(old_val)
            
            if rel_change > threshold:
                return True
        
        return False
    
    async def _log_param_change(
        self,
        market_state: Dict,
        new_params: Dict[str, float]
    ):
        """
        Log parameter change to file.
        
        Args:
            market_state: Market state at time of change
            new_params: New parameter values
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.current_model_path,
            "market_state": {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in market_state.items()
            },
            "parameters": new_params,
            "previous_parameters": self.current_params,
        }
        
        self.param_history.append(log_entry)
        
        # Append to log file
        try:
            # Read existing log
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {"changes": []}
            
            # Append new entry
            log_data["changes"].append(log_entry)
            
            # Keep last 10000 entries
            if len(log_data["changes"]) > 10000:
                log_data["changes"] = log_data["changes"][-10000:]
            
            # Write back
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error writing log file: {e}")
    
    async def check_for_new_models(self) -> bool:
        """
        Check for and load new models if available.
        
        Returns:
            True if new model loaded
        """
        best_model = self.registry.get_best_model()
        
        if not best_model:
            return False
        
        model_path = best_model["filepath"]
        
        # Check if this is a new model
        if model_path != self.current_model_path:
            logger.info(f"New model detected: {best_model['filename']}")
            
            loaded = await self._load_best_model()
            
            if loaded:
                # Reset smoother for new model
                self.smoother.reset()
                logger.info("Model loaded, smoother reset")
                return True
        
        return False
    
    def get_thresholds(self) -> Dict[str, float]:
        """
        Get current threshold values for signal engine.
        
        Returns:
            Dictionary of threshold parameters
        """
        return self.current_params.copy()
    
    async def watch_loop(self):
        """
        Background task that periodically checks for new models.
        Runs continuously until stopped.
        """
        self.is_running = True
        
        logger.info(
            f"Starting model watch loop (interval: {self.check_interval}s)"
        )
        
        while self.is_running:
            try:
                # Check for new models
                new_model_loaded = await self.check_for_new_models()
                
                if new_model_loaded:
                    logger.info("New model loaded in watch loop")
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
            
            except asyncio.CancelledError:
                logger.info("Watch loop cancelled")
                break
            
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
                
                # Wait before retrying
                await asyncio.sleep(5)
        
        self.is_running = False
        logger.info("Watch loop stopped")
    
    def stop_watch_loop(self):
        """Stop the background watch loop."""
        self.is_running = False
    
    def get_stats(self) -> Dict:
        """
        Get optimizer statistics.
        
        Returns:
            Dictionary with optimizer stats
        """
        return {
            **self.stats,
            "current_model": self.current_model_path,
            "current_params": self.current_params,
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
            "param_history_size": len(self.param_history),
            "is_running": self.is_running,
        }
    
    def get_param_history(
        self,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get recent parameter change history.
        
        Args:
            limit: Maximum number of entries to return
        
        Returns:
            List of parameter change entries
        """
        history = list(self.param_history)
        return history[-limit:]
    
    async def compute_param_stats(self) -> Dict:
        """
        Compute statistics on parameter changes.
        
        Returns:
            Dictionary with parameter statistics
        """
        if not self.param_history:
            return {}
        
        # Extract parameter values
        param_values = defaultdict(list)
        
        for entry in self.param_history:
            for key, val in entry.get("parameters", {}).items():
                param_values[key].append(val)
        
        # Compute stats
        stats = {}
        
        for key, values in param_values.items():
            stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "current": float(values[-1]),
            }
        
        return stats


# Test block
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    async def test_optimizer():
        """Test adaptive optimizer."""
        print("Testing AdaptiveOptimizer...")
        print(f"PyTorch available: {TORCH_AVAILABLE}\n")
        
        # Initialize optimizer
        optimizer = AdaptiveOptimizer(
            model_dir="models",
            log_file="test_adaptive_log.json",
            check_interval=10,  # 10 seconds for testing
            smoothing_alpha=0.3,
        )
        
        await optimizer.initialize(
            state_dim=10,
            action_dim=3,
            use_gpu=False,
        )
        
        print("1. Testing model registry...")
        models = optimizer.registry.scan_models()
        print(f"   Found {len(models)} models\n")
        
        if models:
            best = optimizer.registry.get_best_model()
            print(f"   Best model: {best['filename']}")
            print(f"   Modified: {best['modified_time']}")
            if best.get('reward'):
                print(f"   Reward: {best['reward']:.4f}")
            print()
        
        print("2. Testing parameter computation...")
        market_state = {
            "atr_pct": 2.5,
            "atr_percentile": 75,
            "sentiment_score": 0.3,
            "risk_index": 0.15,
            "volume_ratio": 1.8,
            "price_momentum": 0.05,
            "hour_of_day": 14,
            "day_of_week": 3,
            "recent_signal_count": 2,
            "win_rate": 0.62,
        }
        
        params = await optimizer.get_optimal_params(market_state)
        print("   Optimal parameters:")
        for key, val in params.items():
            print(f"     {key}: {val:.3f}")
        print()
        
        print("3. Testing parameter updates...")
        for i in range(5):
            # Vary market state
            market_state["atr_percentile"] = 50 + i * 10
            market_state["sentiment_score"] = -0.5 + i * 0.2
            
            updated = await optimizer.update_parameters(
                market_state,
                force_update=True
            )
            
            print(f"   Update {i+1}:")
            for key, val in updated.items():
                print(f"     {key}: {val:.3f}")
        print()
        
        print("4. Testing threshold getter...")
        thresholds = optimizer.get_thresholds()
        print("   Current thresholds:")
        for key, val in thresholds.items():
            print(f"     {key}: {val:.3f}")
        print()
        
        print("5. Testing statistics...")
        stats = optimizer.get_stats()
        print("   Optimizer stats:")
        for key, val in stats.items():
            if key not in ["current_params"]:
                print(f"     {key}: {val}")
        print()
        
        print("6. Testing parameter history...")
        history = optimizer.get_param_history(limit=3)
        print(f"   History entries: {len(history)}")
        if history:
            print(f"   Latest entry timestamp: {history[-1]['timestamp']}")
        print()
        
        print("7. Testing parameter statistics...")
        param_stats = await optimizer.compute_param_stats()
        if param_stats:
            print("   Parameter statistics:")
            for key, stats in param_stats.items():
                print(f"     {key}:")
                print(f"       mean: {stats['mean']:.3f}")
                print(f"       std: {stats['std']:.3f}")
                print(f"       range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print()
        
        print("8. Testing watch loop (10 seconds)...")
        watch_task = asyncio.create_task(optimizer.watch_loop())
        
        await asyncio.sleep(10)
        
        optimizer.stop_watch_loop()
        
        try:
            await asyncio.wait_for(watch_task, timeout=2)
        except asyncio.TimeoutError:
            watch_task.cancel()
        
        print("   Watch loop stopped\n")
        
        print("Test completed successfully!")
    
    asyncio.run(test_optimizer())
