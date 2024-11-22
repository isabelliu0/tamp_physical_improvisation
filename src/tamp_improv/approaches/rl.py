"""RL-based improvisational policy implementation."""

from dataclasses import dataclass
from typing import Generic, cast

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from tamp_improv.approaches.base import (
    ActType,
    ImprovisationalPolicy,
    ObsType,
    PolicyConfig,
)


@dataclass
class RLPolicyConfig(PolicyConfig):
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
        """Called after each training step."""
        self.current_length += 1
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        if dones[0]:
            success = not infos[0].get("TimeLimit.truncated", False)
            self.success_history.append(success)
            self.episode_lengths.append(self.current_length)
            self.current_length = 0

            if len(self.success_history) % self.check_freq == 0:
                recent_successes = self.success_history[-self.check_freq :]
                recent_lengths = self.episode_lengths[-self.check_freq :]
                success_rate = sum(recent_successes) / len(recent_successes)
                avg_length = sum(recent_lengths) / len(recent_lengths)
                print(f"Episodes: {len(self.success_history)}")
                print(f"Success rate: {success_rate:.2%}")
                print(f"Average episode length: {avg_length:.1f}")

        return True


class RLImprovisationalPolicy(
    Generic[ObsType, ActType], ImprovisationalPolicy[ObsType, ActType]
):
    """RL policy using PPO."""

    def __init__(self, config: RLPolicyConfig) -> None:
        self.config = config
        self.model: PPO | None = None

    def train(
        self, env: gym.Env, total_timesteps: int, seed: int | None = None
    ) -> None:
        """Train policy."""
        if seed is None:
            seed = self.config.seed

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            seed=seed,
            verbose=1,
        )

        callback = TrainingProgressCallback()
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
