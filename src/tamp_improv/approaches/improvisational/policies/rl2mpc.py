"""RL2MPC policy implementation combining RL and MPC."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import cast

import gymnasium as gym
import numpy as np
import torch

from tamp_improv.approaches.improvisational.policies.base import (
    ActType,
    ObsType,
    Policy,
    PolicyContext,
    TrainingData,
)
from tamp_improv.approaches.improvisational.policies.mpc import MPCConfig, MPCPolicy
from tamp_improv.approaches.improvisational.policies.rl import (
    RLConfig,
    RLPolicy,
    TrainingProgressCallback,
)
from tamp_improv.utils.gpu_utils import DeviceContext


@dataclass
class RL2MPCConfig:
    """Configuration for RL2MPC policy."""

    rl_config: RLConfig = field(default_factory=RLConfig)
    mpc_config: MPCConfig = field(default_factory=MPCConfig)
    reward_threshold: float = -30.0  # Threshold to switch from RL to MPC
    window_size: int = 50  # Window size for computing average reward
    device: str = "cuda"


class RL2MPCCallback(TrainingProgressCallback):
    """Callback that stops training when reward threshold is reached."""

    def __init__(
        self,
        reward_threshold: float,
        window_size: int = 10,
        check_freq: int = 100,
        verbose: int = 1,
    ):
        super().__init__(check_freq=check_freq, verbose=verbose)
        self.reward_threshold = reward_threshold
        self.window_size = window_size
        self.threshold_reached = False

    def _on_step(self) -> bool:
        should_continue = super()._on_step()

        # Check threshold on episode completion
        if len(self.episode_rewards) >= self.window_size:
            recent_mean = np.mean(self.episode_rewards[-self.window_size :])
            if recent_mean >= self.reward_threshold:
                print(f"\nReward threshold {self.reward_threshold} reached!")
                print(f"Recent mean reward: {recent_mean:.2f}")
                self.threshold_reached = True
                return False
        return should_continue


class RL2MPCPolicy(Policy[ObsType, ActType]):
    """Policy that combines RL and MPC."""

    def __init__(self, seed: int, config: RL2MPCConfig | None = None) -> None:
        """Initialize policy."""
        super().__init__(seed)
        self.config = config or RL2MPCConfig()
        self.device_ctx = DeviceContext(self.config.device)
        self._torch_generator = torch.Generator(device=self.device_ctx.device)
        self._torch_generator.manual_seed(seed)

        # Ensure component policies use same device
        rl_config = self.config.rl_config
        rl_config.device = self.config.device
        mpc_config = self.config.mpc_config
        mpc_config.device = self.config.device

        # Create component policies
        self.rl_policy: RLPolicy[ObsType, ActType] = RLPolicy(
            seed, self.config.rl_config
        )
        self.mpc_policy: MPCPolicy[ObsType, ActType] = MPCPolicy(
            seed, self.config.mpc_config
        )

        # State tracking
        self._threshold_reached = False
        self._start_state: ObsType | None = None
        self._using_mpc = False
        self._env: gym.Env | None = None
        self._traj_env: gym.Env | None = None
        self._cached_trajectory: list[ActType] | None = None
        self._current_context: PolicyContext[ObsType, ActType] | None = None
        self._valid_shortcuts: set[tuple[int, int]] = set()

    @property
    def requires_training(self) -> bool:
        """Whether this policy requires training data and training."""
        return True

    def initialize(self, env: gym.Env) -> None:
        """Initialize both RL and MPC components."""
        self._env = env
        self.rl_policy.initialize(env)
        self.mpc_policy.initialize(env)
        self._traj_env = deepcopy(env)

    def can_initiate(self):
        """Check if the policy can be executed given the current context."""
        if self._current_context is None:
            return False
        source_id = self._current_context.info.get("source_node_id")
        target_id = self._current_context.info.get("target_node_id")
        return (source_id, target_id) in self._valid_shortcuts

    def configure_context(self, context: PolicyContext) -> None:
        """Configure policy context."""
        # Store previous context info
        prev_source_id = None
        prev_target_id = None
        if self._current_context is not None:
            prev_source_id = self._current_context.info.get("source_node_id")
            prev_target_id = self._current_context.info.get("target_node_id")

        # Update current context
        self._current_context = context
        self.mpc_policy.configure_context(context)
        new_source_id = context.info.get("source_node_id")
        new_target_id = context.info.get("target_node_id")
        if new_source_id != prev_source_id or new_target_id != prev_target_id:
            self._start_state = None  # Will be set in next get_action call
            self._cached_trajectory = None  # Clear cached trajectory

        # Restore cached trajectory if available
        if self._cached_trajectory is not None:
            self.mpc_policy.set_nominal_trajectory(self._cached_trajectory)

        # Configure trajectory environment
        if self._traj_env is not None and hasattr(self._traj_env, "configure_training"):
            self._traj_env.configure_training(
                TrainingData(
                    states=[],
                    current_atoms=[self._current_context.current_atoms],
                    goal_atoms=[self._current_context.goal_atoms],
                    config={"max_steps": self.mpc_policy.config.horizon},
                )
            )

    def train(self, env: gym.Env, train_data: TrainingData | None) -> None:
        """Train RL policy until threshold."""
        assert train_data is not None
        # Create callback that will stop training at threshold
        callback = RL2MPCCallback(
            reward_threshold=self.config.reward_threshold,
            window_size=self.config.window_size,
            check_freq=train_data.config.get("training_record_interval", 100),
        )

        # Train RL policy
        print("\nTraining RL component until reward threshold...")
        self.rl_policy.train(env, train_data, callback=callback)

        # Extract shortcut information from training data
        assert (
            "shortcut_info" in train_data.config and train_data.config["shortcut_info"]
        ), "Training data missing shortcuts"
        for info in train_data.config["shortcut_info"]:
            source_id = info.get("source_node_id")
            target_id = info.get("target_node_id")
            self._valid_shortcuts.add((source_id, target_id))
        print(
            f"RL2MPC extracted {len(self._valid_shortcuts)} valid shortcuts from training data: {self._valid_shortcuts}"  # pylint: disable=line-too-long
        )

        # Record if threshold was reached during training
        # Note: We don't switch to MPC yet - that happens at execution time
        self._threshold_reached = callback.threshold_reached

    def _initialize_mpc_from_start_state(self) -> None:
        """Initialize MPC nominal trajectory using RL policy from start
        state."""
        if self._env is None or self._start_state is None or self._traj_env is None:
            raise ValueError("Environment or start state not initialized")

        # Use MPC's wrapped environment for trajectory generation
        if self.mpc_policy.env is None:
            raise ValueError("MPC environment not initialized")

        # Extract raw observation (without context features)
        # Add an assert to ensure _start_state is of an expected type
        assert hasattr(self._start_state, "shape") and hasattr(
            self._start_state, "__getitem__"
        ), "_start_state must be an iterable with 'shape' attribute (numpy array or torch tensor)."  # pylint: disable=line-too-long
        raw_obs_size = self._start_state.shape[0]
        if hasattr(self._traj_env, "num_context_features"):
            raw_obs_size -= self._traj_env.num_context_features
        if (
            len(self._start_state.shape) > 0
            and self._start_state.shape[0] > raw_obs_size
        ):
            raw_start_state = self._start_state[:raw_obs_size]
        else:
            raw_start_state = self._start_state

        # Get the base environment (unwrapped from context wrapper)
        base_env = self._traj_env
        context_env = None
        while hasattr(base_env, "env"):
            if hasattr(base_env, "augment_observation"):
                context_env = base_env
                base_env = base_env.env
            else:
                base_env = base_env.env

        # Generate trajectory from start state using RL
        base_env.reset_from_state(raw_start_state)  # type: ignore

        # Set context for trajectory environment
        if context_env is not None and self._current_context is not None:
            context_env.set_context(  # type: ignore
                self._current_context.current_atoms, self._current_context.goal_atoms
            )

        # Generate trajectory
        print(f"Generating MPC trajectory from RL starting at: {raw_start_state}")
        current_raw_obs = raw_start_state
        trajectory = []
        for step in range(self.mpc_policy.config.horizon):
            if context_env is not None:
                aug_obs = context_env.augment_observation(current_raw_obs)
                action = self.rl_policy.get_action(aug_obs)
            else:
                action = self.rl_policy.get_action(current_raw_obs)

            trajectory.append(action)

            # Step environment to get next observation
            next_raw_obs, _, terminated, truncated, _ = base_env.step(action)
            if terminated or truncated:
                print(f"RL trajectory terminated at step {step + 1}")
                # If episode ended early, use zero action for remaining steps
                remaining_steps = self.mpc_policy.config.horizon - (step + 1)
                zero_action = cast(ActType, np.zeros_like(action))
                for _ in range(remaining_steps):
                    trajectory.append(zero_action)
                break
            current_raw_obs = next_raw_obs

        print(f"Generated MPC trajectory from RL: {trajectory}")
        self._cached_trajectory = trajectory
        self.mpc_policy.set_nominal_trajectory(trajectory)

        self._traj_env.close()  # type: ignore[no-untyped-call]

    def get_action(self, obs: ObsType) -> ActType:
        """Get action using either RL or MPC."""
        if self._threshold_reached and (
            not self._using_mpc or self._start_state is None
        ):
            print("\nSwitching to MPC with RL initialization")
            self._start_state = obs
            self._initialize_mpc_from_start_state()
            self._using_mpc = True
            return self.mpc_policy.get_action(obs)

        if self._using_mpc:
            action = self.mpc_policy.get_action(obs)
            return action
        return self.rl_policy.get_action(obs)

    def save(self, path: str) -> None:
        """Save RL policy (MPC doesn't need saving)."""
        self.rl_policy.save(path)

    def load(self, path: str) -> None:
        """Load RL policy."""
        self.rl_policy.load(path)
        self._threshold_reached = True
