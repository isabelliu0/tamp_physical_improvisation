"""RL-based implementation of the improvisational policy."""

from typing import Optional, TypeVar, cast

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from tamp_improv.approaches.base_improvisational_tamp_approach import (
    ImprovisationalPolicy,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class TrainingProgressCallback(BaseCallback):
    """Tracks and reports training progress.

    This gets called after each training step to monitor:
    - Success rate (how often we successfully clear the target area)
    - Episode lengths (how many steps it takes to succeed)
    """

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_history: list[bool] = []
        self.episode_lengths: list[int] = []
        self.current_length = 0  # Counter for current episode

    def _on_step(self) -> bool:
        """Called after each training step."""
        self.current_length += 1
        # dones indicates if episode ended this step
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        if dones[0]:  # If episode ended
            # Check if it ended due to success (clearing area) vs timeout
            success = not infos[0].get("TimeLimit.truncated", False)
            self.success_history.append(success)
            self.episode_lengths.append(self.current_length)
            self.current_length = 0

            # Print stats periodically
            if len(self.success_history) % self.check_freq == 0:
                recent_successes = self.success_history[-self.check_freq :]
                recent_lengths = self.episode_lengths[-self.check_freq :]
                success_rate = sum(recent_successes) / len(recent_successes)
                avg_length = sum(recent_lengths) / len(recent_lengths)
                print(f"Episodes: {len(self.success_history)}")
                print(f"Success rate: {success_rate:.2%}")
                print(f"Average episode length: {avg_length:.1f}")

        return True  # Continue training


class RLImprovisationalPolicy(ImprovisationalPolicy[ObsType, ActType]):
    """RL-based improvisational policy using PPO."""

    def __init__(self, env: gym.Env):
        """Initialize policy with a PPO model.

        PPO hyperparameters:
        - learning_rate: How fast to update the policy
        - n_steps: How many steps to collect before updating
        - batch_size: How many samples to use in each update
        - n_epochs: How many times to reuse each sample
        - gamma: Discount factor for future rewards
        """
        self.env = env
        # Create PPO model with the environment
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
        )

    def train(
        self, total_timesteps: int = 1_000_000, seed: Optional[int] = None
    ) -> None:
        """Train the policy.

        The training process:
        1. Reset environment
        2. Get observation
        3. Let policy choose action
        4. Take action, get new observation and reward
        5. If episode ends (success/timeout), reset env
        6. After n_steps, update policy to maximize reward
        7. Repeat until total_timesteps reached
        """
        callback = TrainingProgressCallback()
        if seed is not None:
            self.model.set_random_seed(seed)
        # Start training loop
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
        )

    def save(self, path: str) -> None:
        """Save the trained policy to disk."""
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load a trained policy from disk."""
        self.model = PPO.load(path)

    def get_action(self, obs: ObsType) -> ActType:
        """Get action from policy for current observation."""
        # Convert observation to numpy array for predict
        if isinstance(obs, (int, float)):
            np_obs = np.array([obs])
        else:
            np_obs = np.array(obs)

        action, _ = self.model.predict(np_obs)

        # Convert back to appropriate type
        if isinstance(obs, (int, float)):
            return cast(ActType, int(action[0]))
        return cast(ActType, action)
