"""RL-based policy implementation."""

from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from tamp_improv.approaches.improvisational.policies.base import (
    ActType,
    ObsType,
    Policy,
    TrainingData,
)
from tamp_improv.utils.gpu_utils import DeviceContext


@dataclass
class RLConfig:
    """Configuration for RL policy."""

    learning_rate: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 5
    gamma: float = 0.99
    device: str = "cuda"


class TrainingProgressCallback(BaseCallback):
    """Callback to track training progress."""

    def __init__(self, check_freq: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_history: list[bool] = []
        self.episode_lengths: list[int] = []
        self.episode_rewards: list[float] = []
        self.current_length = 0
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        self.current_length += 1
        self.current_reward += self.locals["rewards"][0]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        if dones[0]:
            # Episode finished - record metrics
            success = not infos[0].get("TimeLimit.truncated", False)
            self.success_history.append(success)
            self.episode_lengths.append(self.current_length)
            self.episode_rewards.append(self.current_reward)

            # Reset counters
            self.current_length = 0
            self.current_reward = 0.0

            # Print prorgess regularly
            n_episodes = len(self.success_history)
            if n_episodes % self.check_freq == 0:
                recent_successes = self.success_history[-self.check_freq :]
                recent_lengths = self.episode_lengths[-self.check_freq :]
                recent_rewards = self.episode_rewards[-self.check_freq :]

                print("\nTraining Progress:")
                print(f"Episodes: {n_episodes}")
                print(
                    f"Recent Success%: {sum(recent_successes)/len(recent_successes):.2%}"
                )
                print(f"Recent Avg Episode Length: {np.mean(recent_lengths):.2f}")
                print(f"Recent Avg Reward: {np.mean(recent_rewards):.2f}")

        return True

    def _on_training_end(self) -> None:
        """Print final training statistics."""
        print("\nFinal Training Results:")
        if self.success_history:
            print(f"Overall Success Rate: {self._get_success_rate:.2%}")
            print(f"Overall Avg Episode Length: {self._get_avg_episode_length:.2f}")
            print(f"Overall Avg Reward: {self._get_avg_reward:.2f}")
        else:
            print("No episodes completed during training.")

    @property
    def _get_success_rate(self) -> float:
        """Get the success rate over all training."""
        if not self.success_history:
            return 0.0
        return float(sum(self.success_history) / len(self.success_history))

    @property
    def _get_avg_episode_length(self) -> float:
        """Get the average episode length over all training."""
        if not self.episode_lengths:
            return 0.0
        return float(np.mean(self.episode_lengths))

    @property
    def _get_avg_reward(self) -> float:
        """Get the average reward over all training."""
        if not self.episode_rewards:
            return 0.0
        return float(np.mean(self.episode_rewards))


class RLPolicy(Policy[ObsType, ActType]):
    """RL policy using PPO."""

    def __init__(self, seed: int, config: RLConfig | None = None) -> None:
        """Initialize policy."""
        super().__init__(seed)
        self.config = config or RLConfig()
        self.device_ctx = DeviceContext(self.config.device)
        print(f"RLPolicy initialized with device: {self.device_ctx.device}")
        self.model: PPO | None = None

    @property
    def requires_training(self) -> bool:
        """Whether this policy requires training data and training."""
        return True

    def initialize(self, env: gym.Env) -> None:
        """Initialize policy with environment."""
        if self.model is None:
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.config.learning_rate,
                n_steps=100,  # Default value, will be updated in train()
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                device=self.device_ctx.device,
                seed=self._seed,
                verbose=1,
            )

    def train(
        self,
        env: gym.Env,
        train_data: TrainingData,
        callback: BaseCallback | None = None,
    ) -> None:
        """Train policy."""
        # Call base class train to initialize and configure env
        super().train(env, train_data)

        print(f"\nStarting RL training on {len(train_data.states)} scenarios")
        print(f"\nStarting RL training on device: {self.device_ctx.device}")
        if self.device_ctx.device.type == "cuda":
            print(
                f"  CUDA device: {torch.cuda.get_device_name(self.device_ctx.device)}"
            )
            print(
                f"  CUDA memory before training: {torch.cuda.memory_allocated(self.device_ctx.device) / 1e9:.2f} GB"  # pylint: disable=line-too-long
            )

        # Initialize and train PPO
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.config.learning_rate,
            n_steps=train_data.config.get("max_steps", 100),
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            device=self.device_ctx.device,
            seed=self._seed,
            verbose=1,
        )

        if callback is None:
            callback = TrainingProgressCallback(
                check_freq=train_data.config.get("training_record_interval", 100)
            )

        # Calculate total timesteps to ensure we see each scenario multiple times
        episodes_per_scenario = train_data.config.get("episodes_per_scenario", 2)
        max_steps = train_data.config.get("max_steps", 100)
        total_timesteps = len(train_data.states) * episodes_per_scenario * max_steps

        print("Training Settings:")
        print(f"Max steps per episode: {max_steps}")
        print(f"Episodes per scenario: {episodes_per_scenario}")
        print(f"Total scenarios: {len(train_data.states)}")
        print(f"Total training timesteps: {total_timesteps}")

        # Train the model
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        if self.device_ctx.device.type == "cuda":
            print(
                f"  CUDA memory after training: {torch.cuda.memory_allocated(self.device_ctx.device) / 1e9:.2f} GB"  # pylint: disable=line-too-long
            )

    def get_action(self, obs: ObsType) -> ActType:
        """Get action from policy."""
        if self.model is None:
            raise ValueError("Policy not trained or loaded")

        obs_tensor = self.device_ctx(obs)
        obs_cpu = (
            obs_tensor.cpu() if torch.is_tensor(obs_tensor) else obs_tensor
        )  # move to CPU for stable_baselines3
        obs_numpy = self.device_ctx.numpy(obs_cpu)

        with torch.no_grad():
            action, _ = self.model.predict(obs_numpy, deterministic=True)

        # Convert back to numpy/original type
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
