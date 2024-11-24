"""RL-based policy implementation."""

from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from tamp_improv.approaches.improvisational.policies.base import (
    ActType,
    ObsType,
    Policy,
    TrainingData,
)


@dataclass
class RLConfig:
    """Configuration for RL policy."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99


class TrainingProgressCallback(BaseCallback):
    """Callback to track training progress."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_history: list[bool] = []
        self.episode_lengths: list[int] = []
        self.current_length = 0

    def _on_step(self) -> bool:
        self.current_length += 1
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        if dones[0]:
            success = not infos[0].get("TimeLimit.truncated", False)
            self.success_history.append(success)
            self.episode_lengths.append(self.current_length)
            self.current_length = 0

            if len(self.success_history) % self.check_freq == 0:
                recent = self.success_history[-self.check_freq :]
                print(f"Success rate: {sum(recent)/len(recent):.2%}")

        return True


class RLPolicy(Policy[ObsType, ActType]):
    """RL policy using PPO."""

    def __init__(self, seed: int, config: RLConfig | None = None) -> None:
        """Initialize policy."""
        super().__init__(seed)
        self.config = config or RLConfig()
        self.model: PPO | None = None

    @property
    def requires_training(self) -> bool:
        """Whether this policy requires training data and training."""
        return True

    def initialize(self, env: gym.Env) -> None:
        """Initialize policy with environment."""
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            seed=self._seed,
            verbose=1,
        )

    def train(self, env: gym.Env, train_data: TrainingData) -> None:
        """Train policy."""
        # Base initialization and precondition setup
        super().train(env, train_data)

        callback = TrainingProgressCallback()
        total_timesteps = train_data.config.get("total_timesteps", 100_000)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def get_action(self, obs: ObsType) -> ActType:
        """Get action from policy."""
        if self.model is None:
            raise ValueError("Policy not trained or loaded")

        # Convert observation
        if isinstance(obs, (int, float)):
            np_obs = np.array([obs])
        else:
            np_obs = np.array(obs)

        action, _ = self.model.predict(np_obs)

        # Convert action back
        if isinstance(obs, (int, float)):
            return cast(ActType, int(action[0]))
        return cast(ActType, action)

    def save(self, path: str) -> None:
        """Save policy."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load policy."""
        self.model = PPO.load(path)
