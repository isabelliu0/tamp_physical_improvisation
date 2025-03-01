"""MPC-based policy implementation."""

from dataclasses import dataclass
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
from tamp_improv.utils.gpu_utils import DeviceContext


@dataclass
class MPCConfig:
    """Configuration for MPC policy."""

    num_rollouts: int = 100
    noise_scale: float = 1.0
    num_control_points: int = 5
    horizon: int = 35
    device: str = "cuda"
    batch_size: int = 32


class MPCPolicy(Policy[ObsType, ActType]):
    """MPC policy using predictive sampling."""

    def __init__(self, seed: int, config: MPCConfig | None = None) -> None:
        """Initialize policy."""
        super().__init__(seed)
        self.config = config or MPCConfig()
        self.device_ctx = DeviceContext(self.config.device)
        print(f"MPCPolicy initialized with device: {self.device_ctx.device}")
        self.env: gym.Env | None = None
        self._rng = np.random.default_rng(seed)

        # Space type flags
        self.is_discrete = False
        self.is_multidiscrete = False
        self._box_space = False
        self.action_dims = 1  # Default for Discrete

        # Initialize arrays with proper dtypes
        self._control_times = torch.zeros(0, device=self.device_ctx.device)
        self._trajectory_times = torch.zeros(0, device=self.device_ctx.device)
        self.last_solution = torch.zeros(0, device=self.device_ctx.device)
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
        self._trajectory_times = torch.arange(
            self.config.horizon, dtype=torch.float32, device=self.device_ctx.device
        )
        self._control_times = torch.linspace(
            0,
            self.config.horizon - 1,
            self.config.num_control_points,
            dtype=torch.float32,
            device=self.device_ctx.device,
        )

        # Initialize last solution
        self.last_solution = self._create_empty_trajectory()

    def _create_empty_trajectory(self) -> torch.Tensor:
        """Create empty trajectory array with proper shape and type."""
        if self.env is None:
            raise ValueError("Environment not initialized")

        if self.is_discrete:
            return torch.zeros(
                self.config.horizon, dtype=torch.int32, device=self.device_ctx.device
            )
        if self.is_multidiscrete:
            return torch.zeros(
                (self.config.horizon, self.action_dims),
                dtype=torch.int32,
                device=self.device_ctx.device,
            )
        if not self._box_space:
            raise ValueError("Unsupported action space type")
        box_space = cast(gym.spaces.Box, self.env.action_space)
        action_shape = box_space.shape if box_space.shape else ()
        shape = (self.config.horizon,) + action_shape
        return torch.zeros(shape, dtype=torch.float32, device=self.device_ctx.device)

    def set_nominal_trajectory(self, trajectory: list[ActType]) -> None:
        """Set the nominal trajectory for MPC from a list of actions."""
        if self.env is None:
            raise ValueError("Policy not initialized")

        trajectory_tensor = torch.tensor(trajectory, device=self.device_ctx.device)

        # Set values based on action space type
        if self.is_discrete:
            self.last_solution = trajectory_tensor.to(torch.int32)
        elif self.is_multidiscrete:
            self.last_solution = trajectory_tensor.reshape(-1, self.action_dims).to(
                torch.int32
            )
        else:
            box_space = cast(gym.spaces.Box, self.env.action_space)
            if box_space.shape:
                trajectory_tensor = trajectory_tensor.reshape(-1, *box_space.shape)
            self.last_solution = torch.clamp(
                trajectory_tensor,
                torch.tensor(box_space.low, device=self.device_ctx.device),
                torch.tensor(box_space.high, device=self.device_ctx.device),
            )

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
        action_np = self.device_ctx.numpy(action)

        # Convert action to appropriate type
        if self.is_discrete:
            return cast(ActType, int(action_np))
        if self.is_multidiscrete:
            return cast(ActType, action_np.astype(np.int32))
        return cast(ActType, action_np)

    def _solve(self, obs: ObsType) -> torch.Tensor:
        """Run one iteration of predictive sampling."""
        if self.env is None:
            raise ValueError("Policy not initialized")

        obs_tensor = self.device_ctx(obs)

        # Get candidates
        if torch.all(self.last_solution == 0):
            nominal = self._get_initialization()
        else:
            # Shift the solution forward
            if self.is_discrete:
                nominal = torch.roll(self.last_solution, -1)
                nominal[-1] = self.last_solution[-1]
            elif self.is_multidiscrete:
                nominal = torch.zeros_like(self.last_solution)
                nominal[:-1] = self.last_solution[1:]
                nominal[-1] = self.last_solution[-1]
            else:
                if self._first_solve:
                    nominal = self.last_solution
                    self._first_solve = False
                else:
                    nominal = torch.roll(self.last_solution, -1, dims=0)
                    nominal[-1] = self.last_solution[-1]

        # Process trajectories in batches
        all_candidates = []
        all_scores = []

        # Add nominal trajectory
        all_candidates.append(nominal)
        nominal_score = self._score_trajectory(nominal, obs_tensor)
        all_scores.append(nominal_score)

        # Generate batches of candidates
        num_batches = (
            self.config.num_rollouts - 1 + self.config.batch_size - 1
        ) // self.config.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(
                start_idx + self.config.batch_size, self.config.num_rollouts - 1
            )
            batch_size = end_idx - start_idx

            if batch_size <= 0:
                break

            batch_candidates = self._sample_from_nominal(nominal, batch_size)
            batch_scores = [
                self._score_trajectory(cand, obs_tensor) for cand in batch_candidates
            ]

            all_candidates.extend(batch_candidates)
            all_scores.extend(batch_scores)

        # Find best trajectory
        best_idx = int(
            torch.argmax(torch.tensor(all_scores, device=self.device_ctx.device)).item()
        )
        self.last_solution = all_candidates[best_idx]
        return self.last_solution[0]

    def _get_initialization(self) -> torch.Tensor:
        """Initialize trajectory."""
        assert self.env is not None

        if self.is_discrete:
            probs = torch.tensor([0.5, 0.5], device=self.device_ctx.device)
            return torch.multinomial(probs, self.config.horizon, replacement=True).to(
                torch.int32
            )

        if self.is_multidiscrete:
            multidiscrete_space = cast(gym.spaces.MultiDiscrete, self.env.action_space)
            traj = torch.zeros(
                (self.config.horizon, self.action_dims),
                dtype=torch.int32,
                device=self.device_ctx.device,
            )

            for dim in range(self.action_dims):
                nvec = multidiscrete_space.nvec[dim]
                traj[:, dim] = torch.randint(
                    0, nvec, (self.config.horizon,), device=self.device_ctx.device
                )
            return traj

        if not self._box_space:
            raise ValueError("Unsupported action space type")

        box_space = cast(gym.spaces.Box, self.env.action_space)
        if box_space.shape:
            control_shape = (self.config.num_control_points,) + box_space.shape
            trajectory_shape = (self.config.horizon,) + box_space.shape
        else:
            control_shape = (self.config.num_control_points,)
            trajectory_shape = (self.config.horizon,)

        control_points = torch.randn(control_shape, device=self.device_ctx.device)
        trajectory = torch.zeros(
            trajectory_shape, dtype=torch.float32, device=self.device_ctx.device
        )

        # Interpolate control points
        for dim in range(
            control_points.shape[-1] if len(control_points.shape) > 1 else 1
        ):
            idx = (..., dim) if len(control_points.shape) > 1 else ...
            trajectory[idx] = torch.nn.functional.interpolate(
                control_points[idx].unsqueeze(0).unsqueeze(0),
                size=self.config.horizon,
                mode="linear",
                align_corners=True,
            ).squeeze()

        # Clamp to action space bounds
        low = torch.tensor(box_space.low, device=self.device_ctx.device)
        high = torch.tensor(box_space.high, device=self.device_ctx.device)
        return torch.clamp(trajectory, low, high)

    def _sample_from_nominal(
        self, nominal: torch.Tensor, batch_size: int
    ) -> list[torch.Tensor]:
        """Sample a batch of trajectories around nominal trajectory."""
        if self.env is None:
            raise ValueError("Policy not initialized")

        if self.is_discrete:
            trajectories = []
            for _ in range(batch_size):
                flip_mask = (
                    torch.rand(self.config.horizon, device=self.device_ctx.device)
                    < self.config.noise_scale
                )

                new_traj = nominal.clone()
                new_traj[flip_mask] = 1 - new_traj[flip_mask]
                trajectories.append(new_traj)
            return trajectories

        if self.is_multidiscrete:
            multidiscrete_space = cast(gym.spaces.MultiDiscrete, self.env.action_space)
            trajectories = []
            for _ in range(batch_size):
                new_traj = nominal.clone()
                for dim in range(self.action_dims):
                    nvec = multidiscrete_space.nvec[dim]
                    flip_mask = (
                        torch.rand(self.config.horizon, device=self.device_ctx.device)
                        < self.config.noise_scale
                    )
                    # For each flip, randomly choose a different valid action
                    flips = flip_mask.nonzero(as_tuple=True)[0]
                    if len(flips) > 0:
                        for idx in flips:
                            current_val = int(new_traj[idx, dim].item())
                            valid_values = list(range(nvec))
                            valid_values.remove(current_val)

                            if valid_values:
                                new_val = valid_values[
                                    int(
                                        torch.randint(0, len(valid_values), (1,)).item()
                                    )
                                ]
                                new_traj[idx, dim] = new_val
                trajectories.append(new_traj)
            return trajectories

        box_space = cast(gym.spaces.Box, self.env.action_space)
        points = torch.stack([nominal[int(t)] for t in self._control_times])

        if len(points.shape) > 1:
            noise_shape = (
                batch_size,
                self.config.num_control_points,
                *points.shape[1:],
            )
        else:
            noise_shape = (batch_size, self.config.num_control_points)

        noise = (
            torch.randn(noise_shape, device=self.device_ctx.device)
            * self.config.noise_scale
        )
        noisy_points = points.unsqueeze(0) + noise

        trajectories = []
        for pts in noisy_points:
            trajectory = torch.zeros(
                (self.config.horizon,) + box_space.shape, device=self.device_ctx.device
            )
            for dim in range(pts.shape[-1] if pts.ndim > 1 else 1):
                idx = (..., dim) if pts.ndim > 1 else ...
                trajectory[idx] = torch.nn.functional.interpolate(
                    pts[idx].unsqueeze(0).unsqueeze(0),
                    size=self.config.horizon,
                    mode="linear",
                    align_corners=True,
                ).squeeze()
            low = torch.tensor(box_space.low, device=self.device_ctx.device)
            high = torch.tensor(box_space.high, device=self.device_ctx.device)
            trajectory = torch.clamp(trajectory, low, high)
            trajectories.append(trajectory)
        return trajectories

    def _score_trajectory(
        self, trajectory: torch.Tensor, init_obs: torch.Tensor
    ) -> float:
        """Score trajectory by simulation."""
        assert self.env is not None

        # Convert tensors to numpy for environment interaction
        trajectory_np = self.device_ctx.numpy(trajectory)
        init_obs_np = self.device_ctx.numpy(init_obs)

        _ = self.env.reset_from_state(init_obs_np)  # type: ignore[attr-defined]
        total_reward = 0.0

        for action in trajectory_np:
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
