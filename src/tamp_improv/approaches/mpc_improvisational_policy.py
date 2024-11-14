"""MPC-based implementation of improvisational policy using predictive
sampling."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from tamp_improv.approaches.base_improvisational_tamp_approach import (
    ImprovisationalPolicy,
)
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.benchmarks.pushing_env import make_pushing_env


@dataclass
class PredictiveSamplingConfig:
    """Configuration for predictive sampling."""

    num_rollouts: int = 10
    noise_scale: float = 0.25
    num_control_points: int = 3
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

        # Initialize nominal parameters
        self._nominal_params = np.zeros(
            (self._config.num_control_points, 3), dtype=np.float32
        )

    def _solve(self, init_obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run one iteration of predictive sampling."""
        # Warm start by shifting nominal parameters
        shifted_params = np.roll(self._nominal_params, -1, axis=0)
        shifted_params[-1] = self._nominal_params[-1]  # Repeat last control point
        self._nominal_params = shifted_params

        # Evaluate nominal parameters first
        sample_list = [self._nominal_params.copy()]
        scores = [self._score_sample(self._nominal_params, init_obs)]

        # Sample new candidates around nominal parameters
        num_samples = self._config.num_rollouts - len(sample_list)
        noise = self._rng.normal(
            loc=0,
            scale=self._config.noise_scale,
            size=(num_samples,) + self._nominal_params.shape,
        )

        # Create and evaluate new samples
        for noise_sample in noise:
            new_params = self._nominal_params + noise_sample
            # Clip parameters to respect action bounds
            new_params = np.clip(new_params, [-0.1, -0.1, 1.0], [0.1, 0.1, 1.0])
            sample_list.append(new_params)
            scores.append(self._score_sample(new_params, init_obs))

        # Pick the best parameters
        best_idx = np.argmax(scores)
        self._nominal_params = sample_list[best_idx]

        # Return first action from best parameters
        return self._get_action_from_params(self._nominal_params, 0.0)

    def _score_sample(
        self, params: NDArray[np.float32], init_obs: NDArray[np.float32]
    ) -> float:
        """Evaluate a set of parameters by rolling out in the environment."""
        obs = init_obs.copy()
        total_reward = 0.0

        self._env.reset(options={"initial_obs": obs})

        for t in range(self._config.horizon):
            # Get action for current timestep
            t_normalized = t / (self._config.horizon - 1)
            action = self._get_action_from_params(params, t_normalized)

            # Step environment and accumulate reward
            obs, reward, terminated, truncated, _ = self._env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return total_reward

    def _get_action_from_params(
        self, params: NDArray[np.float32], t: float
    ) -> NDArray[np.float32]:
        """Get action at time t using linear interpolation between control
        points."""
        idx = int(t * (self._config.num_control_points - 1))
        next_idx = min(idx + 1, self._config.num_control_points - 1)
        alpha = t * (self._config.num_control_points - 1) - idx

        action = (1 - alpha) * params[idx] + alpha * params[next_idx]
        return np.clip(action, [-0.1, -0.1, 1.0], [0.1, 0.1, 1.0])

    def get_action(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get action using predictive sampling."""
        return self._solve(obs)
