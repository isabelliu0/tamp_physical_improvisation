"""MPC-based implementation of improvisational policy using predictive
sampling."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from tamp_improv.approaches.base_improvisational_tamp_approach import (
    ImprovisationalPolicy,
)
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.benchmarks.blocks2d_env_wrapper import make_pushing_env


@dataclass
class PredictiveSamplingConfig:
    """Configuration for predictive sampling."""

    num_rollouts: int = 100
    noise_scale: float = 1.0
    num_control_points: int = 5
    horizon: int = 35


class MPCImprovisationalPolicy(
    ImprovisationalPolicy[NDArray[np.float32], NDArray[np.float32]]
):
    """Predictive sampling policy."""

    def __init__(
        self,
        seed: int,
        config: Optional[PredictiveSamplingConfig] = None,
    ) -> None:
        self._config = config or PredictiveSamplingConfig()
        self._rng = np.random.default_rng(seed)

        # Create environment for trajectory evaluation
        base_env = Blocks2DEnv()
        self._env = make_pushing_env(base_env, self._config.horizon, seed)

        # Initialize last solution
        self._last_solution: NDArray[np.float32] = np.zeros(
            (self._config.horizon, 3), dtype=np.float32
        )

        # Initialize control and trajectory times
        self._control_times = np.linspace(
            0, self._config.horizon - 1, self._config.num_control_points
        )
        self._trajectory_times = np.arange(self._config.horizon)

    def _solve(self, init_obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run one iteration of predictive sampling."""
        sample_list: List[NDArray[np.float32]] = []

        # On first call, use initialization
        if np.all(self._last_solution == 0):
            nominal = self._get_initialization()
        else:
            # Warm start by advancing the last solution by one step
            nominal = np.vstack((self._last_solution[1:], self._last_solution[-1:]))
        sample_list.append(nominal)

        # Sample new trajectories
        samples = self._sample_from_nominal(nominal)
        sample_list.extend(samples)

        # Evaluate all samples
        scores = [self._score_sample(sample, init_obs) for sample in sample_list]

        # Update solution and return first action
        best_idx = np.argmax(scores)
        self._last_solution = sample_list[best_idx]
        return self._last_solution[0]

    def _get_initialization(self) -> NDArray[np.float32]:
        """Initialize with random control points."""
        control_points = self._rng.standard_normal((self._config.num_control_points, 3))
        trajectory = np.zeros((self._config.horizon, 3))
        for dim in range(3):
            trajectory[:, dim] = np.interp(
                self._trajectory_times, self._control_times, control_points[:, dim]
            )
        return np.clip(trajectory, [-0.1, -0.1, 1.0], [0.1, 0.1, 1.0])

    def _sample_from_nominal(
        self, nominal: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        """Sample new trajectories around nominal one by adding Gaussian
        noise."""
        nominal_control_points = np.array(
            [nominal[int(t)] for t in self._control_times]
        )

        noise = self._rng.normal(
            loc=0,
            scale=self._config.noise_scale,
            size=(self._config.num_rollouts - 1, self._config.num_control_points, 3),
        )
        new_control_points = nominal_control_points + noise

        trajectories = []
        for points in new_control_points:
            trajectory = np.zeros((self._config.horizon, 3), dtype=np.float32)
            for dim in range(3):
                trajectory[:, dim] = np.interp(
                    self._trajectory_times, self._control_times, points[:, dim]
                )
            trajectories.append(np.clip(trajectory, [-0.1, -0.1, 1.0], [0.1, 0.1, 1.0]))
        return trajectories

    def _score_sample(
        self, trajectory: NDArray[np.float32], init_obs: NDArray[np.float32]
    ) -> float:
        """Evaluate a trajectory by rolling out in the environment."""
        obs = init_obs.copy()
        total_reward = 0.0

        self._env.reset(options={"initial_obs": obs})

        for action in trajectory:
            obs, reward, terminated, truncated, _ = self._env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return total_reward

    def get_action(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get action using predictive sampling."""
        return self._solve(obs)
