"""
src/adaptive_learning/adaptive_agent.py
GPU-accelerated Reinforcement Learning agent for adaptive trading strategy optimization.
Uses Policy Gradient (REINFORCE) algorithm with experience replay and PyTorch CUDA support.
Learns from historical volatility, sentiment, and macro features to optimize signal thresholds.
"""
import asyncio
import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    torch = None

logger = logging.getLogger("omerGPT.adaptive_learning.agent")


class PolicyNetwork(nn.Module):
    """
    Deep neural network for policy approximation.
    
    Architecture:
    - Input: State vector (volatility, sentiment, macro features)
    - Hidden: 3 layers with ReLU activation + dropout
    - Output: Action probabilities (threshold adjustments)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.2,
    ):
        """
        Initialize policy network.
        
        Args:
            state_dim: Dimension of input state vector
            action_dim: Dimension of action space (number of threshold parameters)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build fully connected layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layers for action distribution parameters
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(
            f"PolicyNetwork initialized: state_dim={state_dim}, "
            f"action_dim={action_dim}, hidden={hidden_dims}"
        )
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor (batch_size, state_dim)
        
        Returns:
            Tuple of (action_mean, action_log_std)
        """
        features = self.feature_extractor(state)
        
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def sample_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy distribution.
        
        Args:
            state: Input state tensor
            deterministic: If True, return mean action without sampling
        
        Returns:
            Tuple of (action, log_probability)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros_like(mean)
        else:
            # Sample from Gaussian distribution
            noise = torch.randn_like(mean)
            action = mean + std * noise
            
            # Compute log probability
            log_prob = -0.5 * (
                ((action - mean) / (std + 1e-8)) ** 2 +
                2 * log_std +
                np.log(2 * np.pi)
            ).sum(dim=-1)
        
        # Apply tanh squashing for bounded actions
        action = torch.tanh(action)
        
        return action, log_prob


class ExperienceBuffer:
    """
    Experience replay buffer for storing and sampling episodes.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize experience buffer.
        
        Args:
            max_size: Maximum number of episodes to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        logger.info(f"ExperienceBuffer initialized: max_size={max_size}")
    
    def add_episode(self, episode: Dict):
        """
        Add an episode to the buffer.
        
        Args:
            episode: Dictionary with keys: states, actions, rewards, log_probs
        """
        self.buffer.append(episode)
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """
        Sample random episodes from buffer.
        
        Args:
            batch_size: Number of episodes to sample
        
        Returns:
            List of episode dictionaries
        """
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=False
        )
        return [self.buffer[i] for i in indices]
    
    def clear(self):
        """Clear all episodes from buffer."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)


class AdaptiveAgent:
    """
    Reinforcement Learning agent for adaptive trading parameter optimization.
    
    Uses Policy Gradient (REINFORCE) with baseline for variance reduction.
    Learns optimal threshold adjustments based on market regime and features.
    """
    
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 3,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        use_gpu: bool = True,
        model_dir: str = "models",
    ):
        """
        Initialize adaptive agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (number of parameters to optimize)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for rewards
            use_gpu: Whether to use GPU acceleration
            model_dir: Directory for saving/loading models
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install: pip install torch")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.model_dir = model_dir
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Device selection
        self.device = torch.device(
            "cuda" if use_gpu and CUDA_AVAILABLE else "cpu"
        )
        
        # Initialize policy network
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.2,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )
        
        # Experience buffer
        self.buffer = ExperienceBuffer(max_size=5000)
        
        # Training statistics
        self.training_history = {
            "episodes": [],
            "losses": [],
            "rewards": [],
            "avg_rewards": [],
        }
        
        logger.info(
            f"AdaptiveAgent initialized on device: {self.device} | "
            f"State dim: {state_dim}, Action dim: {action_dim}"
        )
    
    def encode_state(self, features: Dict) -> np.ndarray:
        """
        Encode market features into state vector.
        
        Args:
            features: Dictionary with keys:
                - atr_pct: ATR as percentage of price
                - atr_percentile: ATR percentile rank
                - sentiment_score: Sentiment index (-1 to 1)
                - risk_index: Macro risk index
                - volume_ratio: Volume vs average
                - price_momentum: Short-term price momentum
        
        Returns:
            Normalized state vector (numpy array)
        """
        state = np.array([
            features.get("atr_pct", 0.0) / 5.0,  # Normalize by typical range
            features.get("atr_percentile", 50) / 100.0,
            (features.get("sentiment_score", 0.0) + 1.0) / 2.0,  # [-1,1] -> [0,1]
            (features.get("risk_index", 0.0) + 1.0) / 2.0,
            min(features.get("volume_ratio", 1.0) / 3.0, 1.0),
            (features.get("price_momentum", 0.0) + 0.1) / 0.2,  # Clip to [-0.1, 0.1]
            features.get("hour_of_day", 12) / 24.0,
            features.get("day_of_week", 3) / 7.0,
            features.get("recent_signal_count", 0) / 10.0,
            features.get("win_rate", 0.5),
        ], dtype=np.float32)
        
        # Clip to [0, 1] range
        state = np.clip(state, 0.0, 1.0)
        
        return state
    
    def decode_action(self, action: np.ndarray) -> Dict[str, float]:
        """
        Decode action vector into trading parameters.
        
        Args:
            action: Action vector from policy (values in [-1, 1])
        
        Returns:
            Dictionary of trading parameters:
                - atr_threshold: ATR spike threshold multiplier
                - volume_threshold: Volume spike threshold multiplier
                - min_confidence: Minimum signal confidence
        """
        # Map [-1, 1] actions to parameter ranges
        params = {
            "atr_threshold": 1.5 + action[0] * 1.0,  # Range: [0.5, 2.5]
            "volume_threshold": 1.3 + action[1] * 0.7,  # Range: [0.6, 2.0]
            "min_confidence": 0.4 + (action[2] + 1) * 0.25,  # Range: [0.4, 0.9]
        }
        
        return params
    
    async def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Select action based on current state.
        
        Args:
            state: State vector
            deterministic: If True, return mean action without sampling
        
        Returns:
            Tuple of (action, log_probability)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy.sample_action(
                state_tensor,
                deterministic=deterministic
            )
        
        action_np = action.cpu().numpy().flatten()
        log_prob_np = log_prob.cpu().item()
        
        return action_np, log_prob_np
    
    def compute_returns(self, rewards: List[float]) -> np.ndarray:
        """
        Compute discounted returns for an episode.
        
        Args:
            rewards: List of rewards for each step
        
        Returns:
            Array of discounted returns
        """
        returns = []
        G = 0.0
        
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = np.array(returns, dtype=np.float32)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    async def train_step(
        self,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Perform one training step using batch of episodes.
        
        Args:
            batch_size: Number of episodes to sample for training
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.buffer) < batch_size:
            logger.warning(
                f"Insufficient episodes in buffer: {len(self.buffer)}/{batch_size}"
            )
            return {"loss": 0.0, "avg_reward": 0.0}
        
        # Sample batch of episodes
        episodes = self.buffer.sample_batch(batch_size)
        
        total_loss = 0.0
        total_reward = 0.0
        
        self.optimizer.zero_grad()
        
        for episode in episodes:
            states = torch.FloatTensor(episode["states"]).to(self.device)
            actions = torch.FloatTensor(episode["actions"]).to(self.device)
            rewards = episode["rewards"]
            
            # Compute returns
            returns = self.compute_returns(rewards)
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            
            # Forward pass
            mean, log_std = self.policy.forward(states)
            std = torch.exp(log_std)
            
            # Compute log probabilities
            action_diff = (actions - torch.tanh(mean)) / (std + 1e-8)
            log_probs = -0.5 * (
                action_diff ** 2 +
                2 * log_std +
                np.log(2 * np.pi)
            ).sum(dim=-1)
            
            # Policy gradient loss
            loss = -(log_probs * returns_tensor).mean()
            
            # Accumulate gradients
            loss.backward()
            
            total_loss += loss.item()
            total_reward += returns[0]  # First return is total episode reward
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        avg_loss = total_loss / len(episodes)
        avg_reward = total_reward / len(episodes)
        
        # Record statistics
        self.training_history["losses"].append(avg_loss)
        self.training_history["rewards"].append(avg_reward)
        self.training_history["avg_rewards"].append(avg_reward)
        
        logger.debug(f"Train step: loss={avg_loss:.4f}, avg_reward={avg_reward:.4f}")
        
        return {
            "loss": avg_loss,
            "avg_reward": avg_reward,
            "buffer_size": len(self.buffer),
        }
    
    async def evaluate(
        self,
        test_states: List[np.ndarray],
        num_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate agent performance on test states.
        
        Args:
            test_states: List of test state vectors
            num_episodes: Number of episodes to evaluate
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.policy.eval()
        
        total_reward = 0.0
        actions_taken = []
        
        with torch.no_grad():
            for state in test_states[:num_episodes]:
                action, _ = await self.select_action(state, deterministic=True)
                actions_taken.append(action)
                
                # Simulate reward (placeholder - would use real backtesting)
                params = self.decode_action(action)
                simulated_reward = self._simulate_reward(state, params)
                total_reward += simulated_reward
        
        self.policy.train()
        
        avg_reward = total_reward / min(len(test_states), num_episodes)
        action_std = np.std(actions_taken, axis=0).mean()
        
        logger.info(
            f"Evaluation: avg_reward={avg_reward:.4f}, action_std={action_std:.4f}"
        )
        
        return {
            "avg_reward": avg_reward,
            "action_std": action_std,
            "episodes_evaluated": min(len(test_states), num_episodes),
        }
    
    def _simulate_reward(self, state: np.ndarray, params: Dict) -> float:
        """
        Simulate reward for state-action pair (placeholder for backtesting).
        
        In production, this would run a backtest with the given parameters
        and return Sharpe ratio or similar performance metric.
        
        Args:
            state: State vector
            params: Trading parameters
        
        Returns:
            Simulated reward (higher is better)
        """
        # Simplified reward simulation based on state features
        # In practice, this would use historical data and signal engine
        
        atr_pct = state[0] * 5.0
        sentiment = (state[2] * 2.0) - 1.0
        risk_idx = (state[3] * 2.0) - 1.0
        
        # Reward components
        volatility_match = 1.0 - abs(params["atr_threshold"] - 2.0) / 2.0
        sentiment_alignment = sentiment * risk_idx
        confidence_penalty = -0.1 if params["min_confidence"] > 0.7 else 0.0
        
        reward = volatility_match + sentiment_alignment * 0.5 + confidence_penalty
        
        # Add noise
        reward += np.random.randn() * 0.1
        
        return float(reward)
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save model weights and training history.
        
        Args:
            filename: Optional custom filename (default: adaptive_agent.pt)
        
        Returns:
            Path to saved model file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"adaptive_agent_{timestamp}.pt"
        
        model_path = os.path.join(self.model_dir, filename)
        
        checkpoint = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_history": self.training_history,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"Model saved: {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str):
        """
        Load model weights and training history.
        
        Args:
            model_path: Path to model checkpoint file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history = checkpoint.get("training_history", {})
        
        logger.info(f"Model loaded: {model_path}")
    
    def get_stats(self) -> Dict:
        """
        Get agent statistics.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            "buffer_size": len(self.buffer),
            "total_episodes": len(self.training_history.get("episodes", [])),
            "avg_loss": np.mean(self.training_history.get("losses", [0])[-100:]),
            "avg_reward": np.mean(self.training_history.get("rewards", [0])[-100:]),
            "device": str(self.device),
        }


# Test block
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    async def test_agent():
        """Test adaptive agent with mock training episode."""
        print("Testing AdaptiveAgent...")
        print(f"PyTorch available: {TORCH_AVAILABLE}")
        print(f"CUDA available: {CUDA_AVAILABLE}\n")
        
        # Initialize agent
        agent = AdaptiveAgent(
            state_dim=10,
            action_dim=3,
            learning_rate=3e-4,
            use_gpu=True,
        )
        
        print("1. Testing state encoding...")
        features = {
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
        
        state = agent.encode_state(features)
        print(f"   Encoded state shape: {state.shape}")
        print(f"   State values (first 5): {state[:5]}\n")
        
        print("2. Testing action selection...")
        action, log_prob = await agent.select_action(state)
        params = agent.decode_action(action)
        
        print(f"   Action: {action}")
        print(f"   Log prob: {log_prob:.4f}")
        print(f"   Decoded parameters:")
        for key, val in params.items():
            print(f"     {key}: {val:.3f}")
        print()
        
        print("3. Simulating training episode...")
        episode = {
            "states": [],
            "actions": [],
            "rewards": [],
        }
        
        for step in range(10):
            # Random state
            state = np.random.rand(10).astype(np.float32)
            action, _ = await agent.select_action(state)
            
            # Simulate reward
            params = agent.decode_action(action)
            reward = agent._simulate_reward(state, params)
            
            episode["states"].append(state)
            episode["actions"].append(action)
            episode["rewards"].append(reward)
        
        episode["states"] = np.array(episode["states"])
        episode["actions"] = np.array(episode["actions"])
        
        print(f"   Episode length: {len(episode['rewards'])}")
        print(f"   Total reward: {sum(episode['rewards']):.4f}\n")
        
        print("4. Adding episode to buffer...")
        agent.buffer.add_episode(episode)
        print(f"   Buffer size: {len(agent.buffer)}\n")
        
        print("5. Generating multiple episodes...")
        for i in range(50):
            ep = {
                "states": np.random.rand(10, 10).astype(np.float32),
                "actions": np.random.randn(10, 3).astype(np.float32) * 0.5,
                "rewards": list(np.random.randn(10) * 0.5),
            }
            agent.buffer.add_episode(ep)
        
        print(f"   Buffer size: {len(agent.buffer)}\n")
        
        print("6. Training for 5 steps...")
        for epoch in range(5):
            metrics = await agent.train_step(batch_size=16)
            print(
                f"   Epoch {epoch+1}: loss={metrics['loss']:.4f}, "
                f"avg_reward={metrics['avg_reward']:.4f}"
            )
        print()
        
        print("7. Saving model...")
        model_path = agent.save_model("test_agent.pt")
        print(f"   Saved to: {model_path}\n")
        
        print("8. Loading model...")
        agent.load_model(model_path)
        print("   Model loaded successfully\n")
        
        print("9. Agent statistics:")
        stats = agent.get_stats()
        for key, val in stats.items():
            print(f"   {key}: {val}")
        
        print("\nTest completed successfully!")
    
    asyncio.run(test_agent())

