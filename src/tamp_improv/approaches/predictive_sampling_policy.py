"""Implementation of predictive sampling for improvisational policy."""

from dataclasses import dataclass
from typing import Optional, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from tamp_improv.approaches.base_improvisational_tamp_approach import (
    ImprovisationalPolicy,
)
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.benchmarks.pushing_env import make_pushing_env


@dataclass(frozen=True)
class PredictiveSamplingHyperparameters:
    """Hyperparameters for predictive sampling."""

    num_rollouts: int = 100  # Number of trajectory samples to evaluate
    noise_scale: float = 1.0  # Scale of Gaussian noise for sampling
    num_control_points: int = 10  # Number of spline control points for trajectory
    horizon: int = 20  # Planning horizon
    dt: float = 0.5  # Time step between control points


class PredictiveSamplingImprovisationalPolicy(
    ImprovisationalPolicy[NDArray[np.float32], NDArray[np.float32]]
):
    """MPC-based improvisational policy using predictive sampling.

    Maintains a nominal trajectory represented as a spline and
    iteratively improves it by sampling variations, evaluating them
    throug forward simulation, and selecting the best performing
    trajectory.
    """

    def __init__(
        self,
        env: gym.Env,
        seed: int,
        config: Optional[PredictiveSamplingHyperparameters] = None,
        warm_start: bool = True,
    ) -> None:
        """Initialize the predictive sampling policy.

        Args:
            env: Environment to plan in
            seed: Random seed
            config: Hyperparameters for predictive sampling
            warm_start: Whether to warm-start from previous solution
        """
        self._env = env
        self._config = config or PredictiveSamplingHyperparameters()
        self._warm_start = warm_start
        self._rng = np.random.default_rng(seed)

        # Create new environment instance for simulation
        self._sim_env = make_pushing_env(Blocks2DEnv(), seed=seed)

        # Initialize nominal trajectory
        self._last_solution = self._get_initialization()

    def _get_initialization(self) -> NDArray[np.float32]:
        """Initialize random nominal trajectory.

        Returns:
            Array of shape (num_control_points, action_dim) with random actions.
        """
        action_space = cast(gym.spaces.Box, self._env.action_space)
        shape = (self._config.num_control_points, action_space.shape[0])

        # Sample random actions between bounds
        low = action_space.low
        high = action_space.high
        return self._rng.uniform(low, high, size=shape).astype(np.float32)

    def _solve(
        self,
        initial_obs: NDArray[np.float32],
        steps_taken: int = 0,
    ) -> NDArray[np.float32]:
        """Run one iteration of predictive sampling optimization.

        Args:
            initial_obs: Current observation to plan from
            steps_taken: Number of steps we've already taken in the episode

        Returns:
            Best action to take
        """
        # Calculate remaining horizon
        remaining_horizon = max(1, self._config.horizon - steps_taken)

        if self._warm_start and self._last_solution is not None:
            # Shift control points for remaining horizon
            self._last_solution = np.vstack(
                [
                    self._last_solution[1:],  # Remove first control point
                    self._last_solution[-1:],  # Repeat last control point
                ]
            )

        # Start with current nominal trajectory
        sample_list = [self._last_solution]

        # Generate variations around nominal
        action_space = cast(gym.spaces.Box, self._env.action_space)
        noise_shape = (
            self._config.num_rollouts - 1,
            self._config.num_control_points,
            action_space.shape[0],
        )
        noise = self._rng.normal(
            loc=0,
            scale=self._config.noise_scale,
            size=noise_shape,
        )

        # Add samples with bounds checking
        for i in range(self._config.num_rollouts - 1):
            sample = np.clip(
                self._last_solution + noise[i],
                action_space.low,
                action_space.high,
            ).astype(np.float32)
            sample_list.append(sample)

        # Evaluate all trajectories
        costs = [
            self._evaluate_trajectory(s, initial_obs, horizon=remaining_horizon)
            for s in sample_list
        ]
        best_idx = np.argmin(costs)
        best_sample = sample_list[best_idx]

        if steps_taken % 5 == 0:
            print(f"\nMPC Step {steps_taken} Summary:")
            print(f"Remaining horizon: {remaining_horizon}")
            print(f"Best trajectory cost: {costs[best_idx]}")
            print(f"Mean trajectory cost: {np.mean(costs)}")
            print(f"Cost std dev: {np.std(costs)}")

        # Update nominal and return first action
        self._last_solution = best_sample
        return self._get_action_from_spline(best_sample, 0.0)

    def _evaluate_trajectory(
        self,
        params: NDArray[np.float32],
        initial_obs: NDArray[np.float32],
        horizon: int,
    ) -> float:
        """Evaluate a trajectory by forward simulation.

        Args:
            params: Spline control point parameters
            initial_obs: Initial observation to start from
            horizon: Number of steps to simulate

        Returns:
            Total cost/negative reward of trajectory
        """
        # Reset sim env to get clean state
        self._sim_env.reset()

        # Set simulation state to match current state
        robot_pos = initial_obs[0:2]
        block_1_pos = initial_obs[4:6]
        block_2_pos = initial_obs[6:8]
        gripper_status = initial_obs[10]

        # Set state in base environment
        base_env = cast(Blocks2DEnv, self._sim_env.env)
        base_env.robot_position = robot_pos
        base_env.block_1_position = block_1_pos
        base_env.block_2_position = block_2_pos
        base_env.gripper_status = np.float32(gripper_status)

        # Initialize tracking variables in wrapper
        self._sim_env.steps = 0
        self._sim_env.prev_distance_to_block2 = (
            np.linalg.norm(robot_pos - block_2_pos)
            - (initial_obs[2] + initial_obs[8]) / 2  # robot_width + block_width
        )

        total_cost = 0.0
        # achieved_goal = False

        # Simulate trajectory for remaining horizon
        for t in range(horizon):
            action = self._get_action_from_spline(params, t * self._config.dt)
            obs, reward, terminated, truncated, _ = self._sim_env.step(action)
            total_cost -= reward

            if terminated:
                if not self._sim_env.is_target_area_blocked(obs):
                    # achieved_goal = True
                    total_cost -= (horizon - t) * 1.0  # Bonus for early success
                break

            if truncated:
                break

        # if not achieved_goal:
        #     total_cost += 10.0  # Penalty if didn't achieve goal

        return total_cost

    def _get_action_from_spline(
        self,
        params: NDArray[np.float32],
        t: float,
    ) -> NDArray[np.float32]:
        """Get action from spline at given time using linear interpolation.

        Args:
            params: Spline control point parameters
            t: Time to sample action at

        Returns:
            Interpolated action vector
        """
        action_space = cast(gym.spaces.Box, self._env.action_space)

        # Scale t to find which control points to interpolate between
        idx = int(t / self._config.dt)

        # If past end of trajectory, return last action
        if idx >= len(params) - 1:
            return params[-1]

        # Linear interpolation between control points
        alpha = (t - idx * self._config.dt) / self._config.dt
        action = ((1 - alpha) * params[idx] + alpha * params[idx + 1]).astype(
            np.float32
        )
        return np.clip(action, action_space.low, action_space.high)

    def get_action(
        self, obs: NDArray[np.float32], steps_taken: int = 0
    ) -> NDArray[np.float32]:
        """Get action from policy for current observation.

        Args:
            obs: Current environment observation

        Returns:
            Action to take
        """
        return self._solve(obs, steps_taken)

    def save(self, path: str) -> None:
        """Save policy parameters."""
        np.save(path, self._last_solution)

    def load(self, path: str) -> None:
        """Load policy parameters."""
        self._last_solution = np.load(path)
