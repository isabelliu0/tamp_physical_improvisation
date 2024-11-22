"""MPC-based improvisational policy implementation."""

from dataclasses import dataclass
from typing import Generic, cast

import gymnasium as gym
import numpy as np

from tamp_improv.approaches.base import (
    ActType_co,
    ImprovisationalPolicy,
    ObsType_contra,
    PolicyConfig,
)


@dataclass
class MPCPolicyConfig(PolicyConfig):
    """Configuration for MPC policy."""

    num_rollouts: int = 100
    noise_scale: float = 1.0
    num_control_points: int = 5
    horizon: int = 35


class MPCImprovisationalPolicy(
    Generic[ObsType_contra, ActType_co],
    ImprovisationalPolicy[ObsType_contra, ActType_co],
):
    """MPC policy using predictive sampling."""

    def __init__(self, config: MPCPolicyConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._env: gym.Env | None = None
        self._is_discrete: bool = False
        self._box_space: bool = False
        self._control_times = np.zeros(0)
        self._trajectory_times = np.zeros(0)
        self._last_solution = np.zeros(0)

    def train(
        self, env: gym.Env, total_timesteps: int, seed: int | None = None
    ) -> None:
        """Initialize policy for environment."""
        self._env = env
        self._is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self._box_space = isinstance(env.action_space, gym.spaces.Box)

        # Initialize trajectory arrays
        self._trajectory_times = np.arange(self.config.horizon, dtype=np.float32)
        self._control_times = np.linspace(
            0, self.config.horizon - 1, self.config.num_control_points
        )

        # Initialize last solution based on action space
        if self._is_discrete:
            self._last_solution = np.zeros(self.config.horizon, dtype=np.int32)
        else:
            if not self._box_space:
                raise ValueError("Unsupported action space type")
            box_space = cast(gym.spaces.Box, env.action_space)
            action_shape = box_space.shape if box_space.shape else ()
            shape = (self.config.horizon,) + action_shape
            self._last_solution = np.zeros(shape, dtype=np.float32)

    def get_action(self, obs: ObsType_contra) -> ActType_co:
        """Get action using predictive sampling."""
        if self._env is None:
            raise ValueError("Policy not initialized - call train() first")
        return cast(ActType_co, self._solve(obs))

    def _solve(self, obs: ObsType_contra) -> ActType_co:
        """Run one iteration of predictive sampling."""
        assert self._env is not None

        # Get candidate trajectories
        if np.all(self._last_solution == 0):
            nominal = self._get_initialization()
        else:
            nominal = np.vstack((self._last_solution[1:], self._last_solution[-1:]))

        candidates = [nominal] + self._sample_from_nominal(nominal)
        scores = [self._score_trajectory(traj, obs) for traj in candidates]

        # Pick best trajectory
        best_idx = np.argmax(scores)
        self._last_solution = candidates[best_idx]

        if self._is_discrete:
            return cast(ActType_co, int(self._last_solution[0]))
        return cast(ActType_co, self._last_solution[0])

    def _get_initialization(self) -> np.ndarray:
        """Initialize trajectory."""
        assert self._env is not None

        if self._is_discrete:
            # For discrete actions, sample random binary sequence
            return self._rng.choice([0, 1], size=self.config.horizon, p=[0.5, 0.5])

        if not self._box_space:
            raise ValueError("Unsupported action space type")

        # For continuous actions, use spline interpolation
        box_space = cast(gym.spaces.Box, self._env.action_space)
        action_shape = box_space.shape if box_space.shape else ()
        shape = (self.config.num_control_points,) + action_shape
        control_points = self._rng.standard_normal(shape)
        trajectory = np.zeros((self.config.horizon,) + action_shape)

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

    def _sample_from_nominal(self, nominal: np.ndarray) -> list[np.ndarray]:
        """Sample new trajectories around nominal."""
        assert self._env is not None

        if self._is_discrete:
            # For discrete actions, flip with probability noise_scale
            trajectories = []
            for _ in range(self.config.num_rollouts - 1):
                flip_mask = (
                    self._rng.random(size=self.config.horizon) < self.config.noise_scale
                )
                new_traj = nominal.copy()
                new_traj[flip_mask] = 1 - new_traj[flip_mask]  # Flip 0->1 or 1->0
                trajectories.append(new_traj)
            return trajectories

        if not self._box_space:
            raise ValueError("Unsupported action space type")

        # For continuous actions, use spline interpolation with noise
        box_space = cast(gym.spaces.Box, self._env.action_space)
        nominal_control_points = np.array(
            [nominal[int(t)] for t in self._control_times]
        )

        action_shape = box_space.shape if box_space.shape else ()
        noise = self._rng.normal(
            loc=0,
            scale=self.config.noise_scale,
            size=(self.config.num_rollouts - 1, self.config.num_control_points)
            + action_shape,
        )
        new_control_points = nominal_control_points + noise

        trajectories = []
        for points in new_control_points:
            trajectory = np.zeros((self.config.horizon,) + action_shape)
            for dim in range(points.shape[-1] if len(points.shape) > 1 else 1):
                idx = (..., dim) if len(points.shape) > 1 else ...
                trajectory[idx] = np.interp(
                    self._trajectory_times, self._control_times, points[idx]
                )
            trajectory = np.clip(trajectory, box_space.low, box_space.high)
            trajectories.append(trajectory)
        return trajectories

    def _score_trajectory(
        self, trajectory: np.ndarray, init_obs: ObsType_contra
    ) -> float:
        """Score a trajectory by simulating it."""
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
        np.savez(
            path,
            last_solution=self._last_solution,
            config=np.array(
                [
                    self.config.num_rollouts,
                    self.config.noise_scale,
                    self.config.num_control_points,
                    self.config.horizon,
                ]
            ),
        )

    def load(self, path: str) -> None:
        """Load policy parameters."""
        data = np.load(path + ".npz")
        self._last_solution = data["last_solution"]
        config = data["config"]
        self.config = MPCPolicyConfig(
            seed=self.config.seed,
            num_rollouts=int(config[0]),
            noise_scale=float(config[1]),
            num_control_points=int(config[2]),
            horizon=int(config[3]),
        )
