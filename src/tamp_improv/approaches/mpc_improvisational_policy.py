"""MPC-based implementation of improvisational policy using predictive
sampling."""

from dataclasses import dataclass
from typing import TypeVar, cast

import gymnasium as gym
import numpy as np

from tamp_improv.approaches.base_improvisational_tamp_approach import (
    ImprovisationalPolicy,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class PredictiveSamplingConfig:
    """Configuration for predictive sampling."""

    num_rollouts: int = 100
    noise_scale: float = 1.0
    num_control_points: int = 5
    horizon: int = 35


class MPCImprovisationalPolicy(ImprovisationalPolicy[ObsType, ActType]):
    """Predictive sampling policy."""

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        seed: int,
        config: PredictiveSamplingConfig | None = None,
    ) -> None:
        self._config = config or PredictiveSamplingConfig()
        self._rng = np.random.default_rng(seed)
        self._env = env
        self._is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self._box_space = isinstance(env.action_space, gym.spaces.Box)

        # Initialize last solution and times
        if self._is_discrete:
            self._last_solution = np.zeros(self._config.horizon, dtype=np.int32)
        else:
            action_shape = env.action_space.shape if self._box_space else ()
            shape = (self._config.horizon,) + tuple(
                () if action_shape is None else action_shape
            )
            self._last_solution = np.zeros(shape, dtype=np.float32)
            self._control_times = np.linspace(
                0, self._config.horizon - 1, self._config.num_control_points
            )
            self._trajectory_times = np.arange(self._config.horizon)

    def _solve(self, init_obs: ObsType) -> ActType:
        """Run one iteration of predictive sampling."""
        sample_list = []

        # On first call, use initialization
        if np.all(self._last_solution == 0):
            nominal = self._get_initialization()
        else:
            # Warm start by advancing the last solution by one step
            nominal = np.vstack((self._last_solution[1:], self._last_solution[-1:]))
        sample_list.append(nominal)

        # Sample and evaluate trajectories
        samples = self._sample_from_nominal(nominal)
        sample_list.extend(samples)
        scores = [self._score_sample(sample, init_obs) for sample in sample_list]

        # Pick best trajectory
        best_idx = np.argmax(scores)
        self._last_solution = sample_list[best_idx]

        if self._is_discrete:
            return cast(ActType, int(self._last_solution[0]))
        return cast(ActType, self._last_solution[0])

    def _get_initialization(self) -> np.ndarray:
        """Initialize trajectory."""
        if self._is_discrete:
            # For discrete actions, sample random binary sequence
            return self._rng.choice([0, 1], size=self._config.horizon, p=[0.5, 0.5])

        if not self._box_space:
            raise ValueError("Unsupported action space type")

        # For continuous actions, use spline interpolation
        box_space = cast(gym.spaces.Box, self._env.action_space)
        action_shape = box_space.shape if box_space.shape else ()
        shape = (self._config.num_control_points,) + action_shape
        control_points = self._rng.standard_normal(shape)
        trajectory = np.zeros((self._config.horizon,) + action_shape)

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
        """Sample new trajectories around nominal one."""
        if self._is_discrete:
            # For discrete actions, flip with probability noise_scale
            trajectories = []
            for _ in range(self._config.num_rollouts - 1):
                flip_mask = (
                    self._rng.random(size=self._config.horizon)
                    < self._config.noise_scale
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
            scale=self._config.noise_scale,
            size=(self._config.num_rollouts - 1, self._config.num_control_points)
            + action_shape,
        )
        new_control_points = nominal_control_points + noise

        trajectories = []
        for points in new_control_points:
            trajectory = np.zeros((self._config.horizon,) + action_shape)
            for dim in range(points.shape[-1] if len(points.shape) > 1 else 1):
                idx = (..., dim) if len(points.shape) > 1 else ...
                trajectory[idx] = np.interp(
                    self._trajectory_times, self._control_times, points[idx]
                )
            trajectory = np.clip(trajectory, box_space.low, box_space.high)
            trajectories.append(trajectory)
        return trajectories

    def _score_sample(self, trajectory: np.ndarray, init_obs: ObsType) -> float:
        """Evaluate a trajectory by rolling out in the environment."""
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

    def get_action(self, obs: ObsType) -> ActType:
        """Get action using predictive sampling."""
        return self._solve(obs)
