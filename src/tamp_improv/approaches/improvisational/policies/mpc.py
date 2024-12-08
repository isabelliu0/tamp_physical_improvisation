"""MPC-based policy implementation."""

from dataclasses import dataclass
from typing import cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from tamp_improv.approaches.improvisational.policies.base import (
    ActType,
    ObsType,
    Policy,
    PolicyContext,
    TrainingData,
)


@dataclass
class MPCConfig:
    """Configuration for MPC policy."""

    num_rollouts: int = 100
    noise_scale: float = 1.0
    num_control_points: int = 5
    horizon: int = 35


class MPCPolicy(Policy[ObsType, ActType]):
    """MPC policy using predictive sampling."""

    def __init__(self, seed: int, config: MPCConfig | None = None) -> None:
        """Initialize policy."""
        super().__init__(seed)
        self.config = config or MPCConfig()
        self._env: gym.Env | None = None
        self._rng = np.random.default_rng(seed)
        self._wrapped_env: gym.Env | None = None
        self._rollout_env: gym.Env | None = None
        self._is_discrete = False
        self._box_space = False

        # Initialize arrays with proper dtypes
        self._control_times = np.zeros(0, dtype=np.float64)
        self._trajectory_times = np.zeros(0, dtype=np.float64)
        self._last_solution = np.zeros(0, dtype=np.float64)

    @property
    def requires_training(self) -> bool:
        """Whether this policy requires training data and training."""
        return False

    def initialize(self, env: gym.Env) -> None:
        """Initialize MPC with environment."""
        self._env = env
        self._is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self._box_space = isinstance(env.action_space, gym.spaces.Box)

        # Initialize trajectory arrays
        self._trajectory_times = np.arange(self.config.horizon, dtype=np.float64)
        self._control_times = np.linspace(
            0, self.config.horizon - 1, self.config.num_control_points, dtype=np.float64
        )

        # Initialize last solution
        if self._is_discrete:
            self._last_solution = np.zeros(self.config.horizon, dtype=np.int32)
        else:
            if not self._box_space:
                raise ValueError("Unsupported action space type")
            box_space = cast(gym.spaces.Box, env.action_space)
            action_shape = box_space.shape if box_space.shape else ()
            shape = (self.config.horizon,) + action_shape
            self._last_solution = np.zeros(shape, dtype=np.float32)

    def configure_context(self, context: PolicyContext[ObsType, ActType]) -> None:
        """Configure MPC with new context/preconditions."""
        if self._env is None:
            raise ValueError("Policy not initialized")

        # Update environment with current preconditions
        if hasattr(self._env, "configure_training"):
            self._env.configure_training(
                TrainingData(
                    states=[],  # MPC doesn't need example states
                    preconditions_to_maintain=[context.preconditions_to_maintain],
                    preconditions_to_achieve=[context.preconditions_to_achieve],
                    config={"max_steps": self.config.horizon},
                )
            )

    def get_action(self, obs: ObsType) -> ActType:
        """Get action using MPC."""
        if self._env is None:
            raise ValueError("Policy not initialized")
        return cast(ActType, self._solve(obs))

    def _solve(self, obs: ObsType) -> NDArray:
        """Run one iteration of predictive sampling."""
        if self._env is None:
            raise ValueError("Policy not initialized")

        # Get candidates
        if np.all(self._last_solution == 0):
            nominal = self._get_initialization()
        else:
            # Shift the solution forward and repeat last action
            if self._is_discrete:
                nominal = np.roll(self._last_solution, -1)
                nominal[-1] = self._last_solution[-1]
            else:
                # For continuous actions, handle potentially multi-dimensional arrays
                nominal = np.zeros_like(self._last_solution)
                if nominal.ndim == 2:
                    nominal[:-1] = self._last_solution[1:]
                    nominal[-1] = self._last_solution[-1]
                else:
                    nominal = np.roll(self._last_solution, -1)
                    nominal[-1] = self._last_solution[-1]

        candidates = [nominal] + self._sample_from_nominal(nominal)
        scores = [self._score_trajectory(traj, obs) for traj in candidates]

        # Pick best trajectory
        best_idx = np.argmax(scores)
        self._last_solution = candidates[best_idx]
        assert isinstance(
            self._last_solution, np.ndarray
        ), "Expected numpy array trajectory"
        return self._last_solution[0]

    def _get_initialization(self) -> NDArray:
        """Initialize trajectory."""
        assert self._env is not None

        if self._is_discrete:
            return self._rng.choice([0, 1], size=self.config.horizon, p=[0.5, 0.5])

        if not self._box_space:
            raise ValueError("Unsupported action space type")

        box_space = cast(gym.spaces.Box, self._env.action_space)
        action_shape = box_space.shape if box_space.shape else ()
        shape = (self.config.num_control_points,) + action_shape
        control_points = self._rng.standard_normal(shape)
        trajectory = np.zeros((self.config.horizon,) + action_shape)

        # Interpolate control points
        for dim in range(
            control_points.shape[-1] if len(control_points.shape) > 1 else 1
        ):
            trajectory[:, dim] = np.interp(
                self._trajectory_times,
                self._control_times,
                (
                    control_points[:, dim]
                    if len(control_points.shape) > 1
                    else control_points
                ),
            )

        return np.clip(trajectory, box_space.low, box_space.high)

    def _sample_from_nominal(self, nominal: NDArray) -> list[NDArray]:
        """Sample trajectories around nominal."""
        if self._env is None:
            raise ValueError("Policy not initialized")

        if self._is_discrete:
            trajectories = []
            for _ in range(self.config.num_rollouts - 1):
                flip_mask = (
                    self._rng.random(size=self.config.horizon) < self.config.noise_scale
                )
                new_traj = nominal.copy()
                new_traj[flip_mask] = 1 - new_traj[flip_mask]
                trajectories.append(new_traj)
            return trajectories

        if not self._box_space:
            raise ValueError("Unsupported action space type")

        box_space = cast(gym.spaces.Box, self._env.action_space)
        points = np.array([nominal[int(t)] for t in self._control_times])

        noise = self._rng.normal(
            loc=0,
            scale=self.config.noise_scale,
            size=(self.config.num_rollouts - 1, self.config.num_control_points)
            + (points.shape[1:] if points.ndim > 1 else ()),
        )
        new_points = points + noise

        trajectories = []
        for pts in new_points:
            trajectory = np.zeros((self.config.horizon,) + box_space.shape)
            for dim in range(pts.shape[-1] if pts.ndim > 1 else 1):
                idx = (..., dim) if pts.ndim > 1 else ...
                trajectory[idx] = np.interp(
                    self._trajectory_times, self._control_times, pts[idx]
                )
            trajectory = np.clip(trajectory, box_space.low, box_space.high)
            trajectories.append(trajectory)
        return trajectories

    def _score_trajectory(self, trajectory: NDArray, init_obs: ObsType) -> float:
        """Score trajectory by simulation."""
        assert self._env is not None

        obs = init_obs
        total_reward = 0.0

        self._env.reset(options={"initial_obs": obs})

        for action in trajectory:
            if self._is_discrete:
                action = int(action)
            obs, reward, terminated, truncated, _ = self._env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return total_reward

    def save(self, path: str) -> None:
        """Save policy parameters."""

    def load(self, path: str) -> None:
        """Load MPC parameters."""
