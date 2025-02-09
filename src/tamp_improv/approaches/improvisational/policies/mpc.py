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
        self.env: gym.Env | None = None
        self._rng = np.random.default_rng(seed)

        # Space type flags
        self.is_discrete = False
        self.is_multidiscrete = False
        self._box_space = False
        self.action_dims = 1  # Default for Discrete

        # Initialize arrays with proper dtypes
        self._control_times: NDArray[np.float64] = np.zeros(0)
        self._trajectory_times: NDArray[np.float64] = np.zeros(0)
        self.last_solution: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        self._first_solve = False

    @property
    def requires_training(self) -> bool:
        """Whether this policy requires training data and training."""
        return False

    def initialize(self, env: gym.Env) -> None:
        """Initialize MPC with environment."""
        self.env = env
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.is_multidiscrete = isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self._box_space = isinstance(env.action_space, gym.spaces.Box)

        # Get action dimensions
        if self.is_discrete:
            self.action_dims = 1
        elif self.is_multidiscrete:
            multidiscrete_space = cast(gym.spaces.MultiDiscrete, env.action_space)
            self.action_dims = int(len(multidiscrete_space.nvec))
        elif self._box_space:
            box_space = cast(gym.spaces.Box, env.action_space)
            self.action_dims = int(np.prod(box_space.shape))
        else:
            raise ValueError("Unsupported action space type")

        # Initialize trajectory arrays
        self._trajectory_times = np.arange(self.config.horizon, dtype=np.float64)
        self._control_times = np.linspace(
            0, self.config.horizon - 1, self.config.num_control_points, dtype=np.float64
        )

        # Initialize last solution
        self.last_solution = self._create_empty_trajectory()

    def _create_empty_trajectory(self) -> NDArray:
        """Create empty trajectory array with proper shape and type."""
        if self.env is None:
            raise ValueError("Environment not initialized")

        if self.is_discrete:
            return np.zeros(self.config.horizon, dtype=np.int32)
        if self.is_multidiscrete:
            return np.zeros((self.config.horizon, self.action_dims), dtype=np.int32)
        if not self._box_space:
            raise ValueError("Unsupported action space type")
        box_space = cast(gym.spaces.Box, self.env.action_space)
        action_shape = box_space.shape if box_space.shape else ()
        shape = (self.config.horizon,) + action_shape
        return np.zeros(shape, dtype=np.float32)

    def set_nominal_trajectory(self, trajectory: list[ActType]) -> None:
        """Set the nominal trajectory for MPC from a list of actions."""
        if self.env is None:
            raise ValueError("Policy not initialized")

        # Use same shape/type as empty trajectory
        new_solution = self._create_empty_trajectory()
        trajectory_arr = np.array(trajectory)

        # Set values based on action space type
        if self.is_discrete:
            new_solution = trajectory_arr.astype(np.int32)
        elif self.is_multidiscrete:
            new_solution = trajectory_arr.reshape(-1, self.action_dims)
        else:
            box_space = cast(gym.spaces.Box, self.env.action_space)
            if box_space.shape:
                trajectory_arr = trajectory_arr.reshape(-1, *box_space.shape)
            new_solution = np.clip(trajectory_arr, box_space.low, box_space.high)

        self.last_solution = new_solution
        self._first_solve = True

    def configure_context(self, context: PolicyContext[ObsType, ActType]) -> None:
        """Configure MPC with new context/preconditions."""
        if self.env is None:
            raise ValueError("Policy not initialized")

        # Update environment with current preconditions
        if hasattr(self.env, "configure_training"):
            self.env.configure_training(
                TrainingData(
                    states=[],  # MPC doesn't need example states
                    preconditions_to_maintain=[context.preconditions_to_maintain],
                    preconditions_to_achieve=[context.preconditions_to_achieve],
                    config={"max_steps": self.config.horizon},
                )
            )

    def get_action(self, obs: ObsType) -> ActType:
        """Get action using MPC."""
        if self.env is None:
            raise ValueError("Policy not initialized")
        action = self._solve(obs)

        # Convert action to appropriate type
        if self.is_discrete:
            return cast(ActType, int(action))
        if self.is_multidiscrete:
            return cast(ActType, action.astype(np.int32))
        return cast(ActType, action)

    def _solve(self, obs: ObsType) -> NDArray:
        """Run one iteration of predictive sampling."""
        if self.env is None:
            raise ValueError("Policy not initialized")

        # Get candidates
        if np.all(self.last_solution == 0):
            nominal = self._get_initialization()
        else:
            # Shift the solution forward
            if self.is_discrete:
                nominal = np.roll(self.last_solution, -1)
                nominal[-1] = self.last_solution[-1]
            elif self.is_multidiscrete:
                nominal = np.zeros_like(self.last_solution)
                nominal[:-1] = self.last_solution[1:]
                nominal[-1] = self.last_solution[-1]
            else:
                if self._first_solve:
                    nominal = self.last_solution
                    self._first_solve = False
                else:
                    nominal = np.zeros_like(self.last_solution)
                    nominal = np.roll(self.last_solution, -1, axis=0)
                    nominal[-1] = self.last_solution[-1]

        candidates = [nominal] + self._sample_from_nominal(nominal)
        scores = [self._score_trajectory(traj, obs) for traj in candidates]

        # Pick best trajectory
        best_idx = np.argmax(scores)
        self.last_solution = candidates[best_idx]
        assert isinstance(
            self.last_solution, np.ndarray
        ), "Expected numpy array trajectory"
        return self.last_solution[0]

    def _get_initialization(self) -> NDArray:
        """Initialize trajectory."""
        assert self.env is not None

        if self.is_discrete:
            return self._rng.choice([0, 1], size=self.config.horizon, p=[0.5, 0.5])

        if self.is_multidiscrete:
            multidiscrete_space = cast(gym.spaces.MultiDiscrete, self.env.action_space)
            shape = (self.config.horizon, self.action_dims)
            traj = np.zeros(shape, dtype=np.int32)
            for dim in range(self.action_dims):
                nvec = multidiscrete_space.nvec[dim]
                traj[:, dim] = self._rng.integers(
                    0, nvec, size=self.config.horizon, dtype=np.int32
                )
            return traj

        if not self._box_space:
            raise ValueError("Unsupported action space type")

        box_space = cast(gym.spaces.Box, self.env.action_space)
        if box_space.shape:
            control_shape = cast(
                tuple[int, ...], (self.config.num_control_points,) + box_space.shape
            )
            trajectory_shape = cast(
                tuple[int, ...], (self.config.horizon,) + box_space.shape
            )
        else:
            control_shape = (self.config.num_control_points,)
            trajectory_shape = (self.config.horizon,)

        control_points = self._rng.standard_normal(control_shape)
        trajectory: NDArray[np.float32] = np.zeros(trajectory_shape, dtype=np.float32)

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
        if self.env is None:
            raise ValueError("Policy not initialized")

        if self.is_discrete:
            trajectories = []
            for _ in range(self.config.num_rollouts - 1):
                flip_mask = (
                    self._rng.random(size=self.config.horizon) < self.config.noise_scale
                )
                new_traj = nominal.copy()
                new_traj[flip_mask] = 1 - new_traj[flip_mask]
                trajectories.append(new_traj)
            return trajectories

        if self.is_multidiscrete:
            multidiscrete_space = cast(gym.spaces.MultiDiscrete, self.env.action_space)
            trajectories = []
            for _ in range(self.config.num_rollouts - 1):
                new_traj = nominal.copy()
                for dim in range(self.action_dims):
                    flip_mask = (
                        self._rng.random(size=self.config.horizon)
                        < self.config.noise_scale
                    )
                    nvec = multidiscrete_space.nvec[dim]
                    # For each flip, randomly choose a different valid action
                    flips = flip_mask.nonzero()[0]
                    for idx in flips:
                        current_val = new_traj[idx, dim]
                        valid_values = list(range(nvec))
                        valid_values.remove(current_val)
                        new_traj[idx, dim] = self._rng.choice(valid_values)
                trajectories.append(new_traj)
            return trajectories

        if not self._box_space:
            raise ValueError("Unsupported action space type")

        box_space = cast(gym.spaces.Box, self.env.action_space)
        print(
            f"box_space: {box_space}, upper bound: {box_space.high}, lower bound: {box_space.low}"
        )
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
        assert self.env is not None

        _ = self.env.reset_from_state(init_obs)  # type: ignore[attr-defined]
        total_reward = 0.0

        for action in trajectory:
            if self.is_discrete:
                action = int(action)
            elif self.is_multidiscrete:
                action = action.astype(np.int32)
            _, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return total_reward

    def save(self, path: str) -> None:
        """Save policy parameters."""

    def load(self, path: str) -> None:
        """Load MPC parameters."""
