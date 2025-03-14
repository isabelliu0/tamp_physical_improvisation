"""RL-based policy implementation."""

import json
import os
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from relational_structs import GroundAtom
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from tamp_improv.approaches.improvisational.policies.base import (
    ActType,
    ObsType,
    Policy,
    PolicyContext,
    TrainingData,
)
from tamp_improv.utils.gpu_utils import DeviceContext


@dataclass
class RLConfig:
    """Configuration for RL policy."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    n_epochs: int = 5
    gamma: float = 0.99
    ent_coef: float = 0.01
    device: str = "cuda"
    success_threshold: float = (
        0.7  # # Success rate threshold for a shortcut to be considered "verified"
    )
    eval_episodes: int = 5  # Number of episodes to run when evaluating a shortcut


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

        self._torch_generator = torch.Generator(device=self.device_ctx.device)
        self._torch_generator.manual_seed(seed)

        self.model: PPO | None = None
        self._current_context: PolicyContext | None = None

        # Verification results where we store by context hash
        self.verified_shortcuts = {}  # hash(context) -> bool
        self.shortcut_success_rates = {}  # hash(context) -> float

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
                ent_coef=self.config.ent_coef,
                device=self.device_ctx.device,
                seed=self._seed,
                verbose=1,
            )

    def can_initiate(self):
        """Check if the policy can be executed given the current context."""
        if not self._current_context or self.model is None:
            return False

        # Check if we have verification for this specific context
        context_hash = self._hash_context(self._current_context)
        if context_hash in self.verified_shortcuts:
            return self.verified_shortcuts[context_hash]

        # If not verified yet, be conservative
        return False

    def _hash_context(self, context: PolicyContext) -> int:
        """Create a hash of a context based on its atoms."""
        source_atoms = sorted(str(atom) for atom in context.current_atoms)
        target_atoms = sorted(str(atom) for atom in context.preimage)
        return f"{';'.join(source_atoms)}||{';'.join(target_atoms)}"

    def configure_context(self, context: PolicyContext[ObsType, ActType]) -> None:
        """Configure policy with context information."""
        self._current_context = context

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
            ent_coef=self.config.ent_coef,
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

        # Convert back to original type
        if isinstance(obs, (int, float)):
            return cast(ActType, int(action[0]))
        return cast(ActType, action)

    def verify_shortcuts(
        self, env: gym.Env, contexts: list[tuple[set[GroundAtom], set[GroundAtom], Any]]
    ) -> dict[str, bool]:
        """Verify which shortcuts this policy can successfully execute."""
        if self.model is None:
            raise ValueError("Policy must be trained before verifying shortcuts")

        print("\nVerifying shortcut capabilities...")

        results = {}

        for i, (source_atoms, target_preimage, source_state) in enumerate(contexts):
            print(f"\nVerifying shortcut {i+1}/{len(contexts)}")

            context = PolicyContext(
                current_atoms=source_atoms,
                preimage=target_preimage,
            )
            context_hash = self._hash_context(context)

            success_count = 0
            episode_lengths = []

            # Run evaluation episodes
            for episode in range(self.config.eval_episodes):
                print(f"  Episode {episode+1}/{self.config.eval_episodes}")

                # Reset environment to source state
                obs = env.reset_from_state(source_state)[0]  # type: ignore

                self.configure_context(context)

                success = False
                steps = 0
                max_steps = 100

                for step in range(max_steps):
                    action = self.get_action(obs)
                    obs, _, terminated, truncated, _ = env.step(action)
                    steps += 1

                    # Check if target preimage is achieved
                    atoms = env.perceiver.step(obs)
                    if target_preimage.issubset(atoms):
                        success = True
                        episode_lengths.append(steps)
                        break

                    if terminated or truncated:
                        break

                if success:
                    success_count += 1
                    print(f"  Success! Completed in {steps} steps")
                else:
                    print("  Failed!")

            # Calculate success rate
            success_rate = success_count / self.config.eval_episodes
            avg_length = np.mean(episode_lengths) if episode_lengths else float("inf")

            print(f"Shortcut {i+1} success rate: {success_rate:.2%}")
            if episode_lengths:
                print(f"Average completion steps: {avg_length:.1f}")

            # Store verification results
            is_verified = success_rate >= self.config.success_threshold
            self.verified_contexts[context_hash] = is_verified
            self.context_success_rates[context_hash] = success_rate
            results[context_hash] = is_verified

            if is_verified:
                print(f"Shortcut {i+1} VERIFIED (success rate meets threshold)")
            else:
                print(f"Shortcut {i+1} NOT VERIFIED (success rate below threshold)")

        print("\nVerification Results:")
        verified_count = sum(1 for v in results.values() if v)
        print(f"Verified {verified_count} out of {len(results)} shortcuts")

        return results

    def save(self, path: str) -> None:
        """Save policy."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)

        verification_path = f"{path}_verification.json"
        with open(verification_path, "w") as f:
            # Convert sets to lists for JSON serialization
            serializable_verified_contexts = {}
            for context_hash, is_verified in self.verified_contexts.items():
                serializable_verified_contexts[context_hash] = is_verified

            verification_data = {
                "verified_contexts": serializable_verified_contexts,
                "context_success_rates": self.context_success_rates,
            }
            json.dump(verification_data, f)

    def load(self, path: str) -> None:
        """Load policy."""
        self.model = PPO.load(path)

        verification_path = f"{path}_verification.json"
        if os.path.exists(verification_path):
            with open(verification_path, "r") as f:
                verification_data = json.load(f)
                self.verified_contexts = verification_data.get("verified_contexts", {})
                self.context_success_rates = verification_data.get(
                    "context_success_rates", {}
                )
