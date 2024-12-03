"""Wrappers for blocks2d environment."""

from typing import Any, Union

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
)
from task_then_motion_planning.structs import Perceiver

from tamp_improv.approaches.improvisational.policies.base import TrainingData


def is_target_area_blocked(
    block_x: float,
    block_width: float,
    target_x: float,
    target_width: float,
) -> bool:
    """Check if block 2 blocks the target area -- impediment.

    Block 2 is considered blocking if it overlaps with the target area
    enough that another block cannot fit in the remaining space.
    """
    target_left = target_x - target_width / 2
    target_right = target_x + target_width / 2
    block_left = block_x - block_width / 2
    block_right = block_x + block_width / 2

    # If no horizontal overlap, not blocking
    if block_right <= target_left or block_left >= target_right:
        return False

    # Calculate remaining free width
    overlap_width = min(block_right, target_right) - max(block_left, target_left)
    free_width = target_width - overlap_width

    # Block needs at least its width to fit
    return free_width < block_width


class Blocks2DEnvWrapper(gym.Env):
    """Environment wrapper for learning the improvisational pushing policy
    while maintaining operator preconditions in Blocks2D environment."""

    def __init__(
        self,
        base_env: gym.Env,
        perceiver: Perceiver[NDArray[np.float32]],
    ) -> None:
        """Initialize wrapper with environment and perceiver."""
        self.env = base_env
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        self.max_episode_steps = 100
        self.steps = 0
        self.prev_distance_to_block2 = None
        self.perceiver = perceiver

        # Tracking for training data (preconditions and initial states)
        self.training_states: list[NDArray[np.float32]] = []
        self.training_preconditions: list[set[GroundAtom]] = []
        self.current_training_idx: int = 0
        self.total_training_episodes: int = 0
        self.preconditions_to_maintain: set[GroundAtom] = set()

        # Default reset options
        self._custom_reset_options = {
            "robot_pos": np.array([0.5, 1.0], dtype=np.float32),
            "ensure_blocking": True,
        }

        self.render_mode = self.env.render_mode

    def configure_training(
        self,
        training_data: TrainingData,
    ) -> None:
        """Configure environment for training phase."""
        print(f"Configuring environment with {len(training_data)} training scenarios")
        self.training_states = training_data.states
        self.training_preconditions = training_data.preconditions
        self.current_training_idx = 0
        self.total_training_episodes = len(training_data)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment."""
        self.steps = 0

        # If we have training data, use it cyclically
        if self.training_states:
            self.current_training_idx = self.current_training_idx % len(
                self.training_states
            )

            # Get current training scenario
            current_state = self.training_states[self.current_training_idx]
            current_preconditions = self.training_preconditions[
                self.current_training_idx
            ]

            print(f"Training episode {self.current_training_idx + 1}")
            print(f"Using state with robot at: {current_state[0:2]}")
            print(f"Using state with gripper status: {current_state[10]}")
            print(f"Maintaining preconditions: {current_preconditions}")

            # Get positions from current state
            reset_options = {
                "robot_pos": current_state[0:2].copy(),
                "block_1_pos": current_state[4:6].copy(),
                "block_2_pos": current_state[6:8].copy(),
            }

            # Reset base environment with current training state
            obs, info = self.env.reset(seed=seed, options=reset_options)

            # Store current preconditions
            self.preconditions_to_maintain = current_preconditions

            # Initialize distance tracking since we start in training state
            robot_pos = obs[0:2]
            block2_pos = obs[6:8]
            robot_width = obs[2]
            block_width = obs[8]
            self.prev_distance_to_block2 = (
                np.linalg.norm(robot_pos - block2_pos) - (robot_width + block_width) / 2
            )

            # Update index cyclically
            self.current_training_idx += 1

            return obs, info

        # Otherwise use default reset logic for baseline training
        return self._default_reset(seed=seed, options=options)

    def _default_reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Default reset logic for initial training.

        This creates a scenario where block 2 is blocking the target
        area, requiring the agent to learn pushing behavior.
        """
        self.steps = 0

        # Initialize RNG
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Get initial observation to access dimensions
        obs, _ = self.env.reset(seed=seed)

        # Generate random block 2 position that blocks target
        block_2_x = rng.uniform(
            obs[11] - obs[13] / 2,  # target_x - target_width/2
            obs[11] + obs[13] / 2,  # target_x + target_width/2
        )

        reset_options = {
            "robot_pos": self._custom_reset_options["robot_pos"],
            "block_1_pos": self._custom_reset_options[
                "robot_pos"
            ],  # Block 1 starts at robot
            "block_2_pos": np.array([block_2_x, 0.0], dtype=np.float32),
        }

        if options:
            reset_options.update(options)

        obs, info = self.env.reset(options=reset_options)

        # Initialize distance tracking
        robot_pos = obs[0:2]
        block2_pos = obs[6:8]
        robot_width = obs[2]
        block_width = obs[8]
        self.prev_distance_to_block2 = (
            np.linalg.norm(robot_pos - block2_pos) - (robot_width + block_width) / 2
        )

        # Verify block 2 is blocking if required
        if self._custom_reset_options["ensure_blocking"] and not is_target_area_blocked(
            block2_pos[0],
            block_width,
            obs[11],  # target_x
            obs[13],  # target_width
        ):
            # If not blocking, try again
            return self.reset(seed=seed, options=options)

        return obs, info

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        obs, _, _, _, info = self.env.step(action)
        self.steps += 1

        # Check target area blocking and preconditions
        is_blocked = is_target_area_blocked(
            obs[6],  # block2_x
            obs[8],  # block_width
            obs[11],  # target_x
            obs[13],  # target_width
        )
        precondition_penalty = self._calculate_precondition_violation_penalty(obs)

        # Calculate reward components
        distance_reward = self._calculate_distance_reward(obs)
        collision_penalty = -0.1 if self._check_collision(obs) else 0.0

        # Success case: cleared area while maintaining preconditions
        if not is_blocked and np.isclose(precondition_penalty, 0.0, atol=1e-3):
            reward = 30.0
        else:
            reward = (
                -0.1  # Base step penalty
                + distance_reward
                + collision_penalty
                + precondition_penalty
            )

        terminated = not is_blocked  # Terminate when area is cleared
        truncated = self.steps >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def _check_preconditions(self, obs: NDArray[np.float32]) -> bool:
        """Check if all maintained preconditions are still satisfied."""
        if not self.preconditions_to_maintain:
            return True
        current_atoms = self.perceiver.step(obs)
        return self.preconditions_to_maintain.issubset(current_atoms)

    def _calculate_precondition_violation_penalty(
        self, obs: NDArray[np.float32]
    ) -> float:
        """Calculate penalty for violating operator preconditions."""
        if not self._check_preconditions(obs):
            return -5.0
        return 0.0

    def _calculate_distance_reward(self, obs: NDArray[np.float32]) -> float:
        """Calculate reward based on distance to block 2."""
        robot_pos = obs[0:2]
        block2_pos = obs[6:8]
        robot_width = obs[2]
        block_width = obs[8]

        current_distance = (
            np.linalg.norm(robot_pos - block2_pos) - (robot_width + block_width) / 2
        )

        # Reward for moving closer
        distance_delta = self.prev_distance_to_block2 - current_distance
        reward = 0.5 * distance_delta

        # Bonus for being close
        if np.isclose(current_distance, 0.0, atol=2e-2):
            reward += 0.5

        self.prev_distance_to_block2 = current_distance
        return reward

    def _check_collision(self, obs: NDArray[np.float32]) -> bool:
        """Check for collisions between robot and block 2."""
        robot_pos = obs[0:2]
        block2_pos = obs[6:8]
        robot_width, robot_height = obs[2:4]
        block_width, block_height = obs[8:10]

        dx = abs(robot_pos[0] - block2_pos[0])
        dy = abs(robot_pos[1] - block2_pos[1])

        width_sum = (robot_width + block_width) / 2 - 1e-3
        height_sum = (robot_height + block_height) / 2 - 1e-3

        return dx < width_sum and dy < height_sum

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        """Render the environment."""
        rendered: Union[RenderFrame, list[RenderFrame], None] = self.env.render()
        return rendered
