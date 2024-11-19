"""Environment wrapper for learning the pushing policy in Blocks2D
environment."""

from typing import Any, Set, Union, cast

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from numpy.typing import NDArray
from relational_structs import GroundAtom, LiftedOperator

from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.blocks2d_planning import create_blocks2d_planning_models


class PushingEnvWrapper(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Environment wrapper for learning the pushing policy while maintaining
    operator preconditions in Blocks2D environment."""

    def __init__(self, base_env: Blocks2DEnv, seed: int | None = None) -> None:
        """Initialize wrapper without specific preconditions.

        Args:
            base_env: The base Blocks2D environment
            seed: Random seed
        """
        super().__init__()

        if not isinstance(base_env, Blocks2DEnv):
            raise ValueError("Environment must be a Blocks2DEnv instance")

        self.env = base_env
        self.max_episode_steps = 100
        self.steps = 0

        # Initialize planning components for checking atoms
        _, self.predicates, self.perceiver, _, _ = create_blocks2d_planning_models(
            include_pushing_models=True
        )

        # These will be set when preconditions are updated
        self.current_operator: LiftedOperator | None = None
        self.preconditions_to_maintain: Set[GroundAtom] = set()

        # Track previous distance to block 2 for reward shaping
        self.prev_distance_to_block2 = None

        # Same spaces as base env
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

        # Reset options
        self._custom_reset_options = {
            "robot_pos": np.array([0.5, 1.0], dtype=np.float32),
            "ensure_blocking": True,  # Block 2 should block target area
        }

        self.render_mode = self.env.render_mode

        if seed is not None:
            self.reset(seed=seed)

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        """Render the environment.

        Returns:
            Union[RenderFrame, list[RenderFrame], None]: The rendered frame(s)
        """
        rendered = self.env.render()
        return cast(Union[RenderFrame, list[RenderFrame], None], rendered)

    def update_preconditions(
        self, operator: LiftedOperator, preconditions: Set[GroundAtom]
    ) -> None:
        """Update the preconditions that should be maintained.

        Args:
            operator: The operator being executed
            preconditions: The currently satisfied preconditions to maintain
        """
        self.current_operator = operator
        self.preconditions_to_maintain = preconditions

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
            return -1.0
        return 0.0

    def is_target_area_blocked(self, obs: NDArray[np.float32]) -> bool:
        """Check if block 2 blocks the target area.

        A block is considered blocking if it overlaps with the target
        area enough that block 1 cannot fit in the remaining space.
        """
        # Get positions and dimensions
        block_2_x = obs[6]
        block_width = obs[8]
        target_x = obs[11]
        target_width = obs[13]

        # Calculate boundaries
        target_left = target_x - target_width / 2
        target_right = target_x + target_width / 2
        block_left = block_2_x - block_width / 2
        block_right = block_2_x + block_width / 2

        # Check if there's any overlap
        if block_right <= target_left or block_left >= target_right:
            return False  # No overlap

        # Calculate remaining free width in target area
        overlap_width = min(block_right, target_right) - max(block_left, target_left)
        free_width = target_width - overlap_width

        # Block 1 needs at least its width to fit
        return free_width < block_width

    def _get_random_block_2_position(
        self,
        rng: np.random.Generator,
        target_x: float = 0.5,
        ensure_blocking: bool = True,
    ) -> NDArray[np.float32]:
        """Generate valid random positions for blocks."""
        obs, _ = self.env.reset()
        block_width = float(obs[8])
        target_width = float(obs[13])
        margin = block_width / 2

        # Block 2 position depends on whether it should block target area
        if ensure_blocking:
            # Place block 2 somewhere in or near target area
            block_2_x = rng.uniform(
                target_x - target_width / 2, target_x + target_width / 2
            )
        else:
            # Random position away from target
            if rng.random() < 0.5:
                block_2_x = rng.uniform(0.0, target_x - margin)
            else:
                block_2_x = rng.uniform(target_x + margin, 1.0)

        return np.array([block_2_x, 0.0], dtype=np.float32)

    def _check_collision(self, obs: NDArray[np.float32]) -> bool:
        """Check for collisions between robot (holding block 1) and block 2."""
        robot_pos = obs[0:2]
        block2_pos = obs[6:8]
        robot_width = obs[2]
        block_width = obs[8]

        dx = abs(robot_pos[0] - block2_pos[0])
        dy = abs(robot_pos[1] - block2_pos[1])

        width_sum = (robot_width + block_width) / 2 - 1e-3
        height_sum = width_sum  # Since blocks and robot are square

        return dx < width_sum and dy < height_sum

    def _calculate_distance_reward(self, obs: NDArray[np.float32]) -> float:
        """Calculate reward based on distance to block 2 and movement towards
        it."""
        robot_pos = obs[0:2]
        block2_pos = obs[6:8]
        robot_width = obs[2]
        block_width = obs[8]

        # Calculate current distance from robot to block 2
        current_distance = (
            np.linalg.norm(robot_pos - block2_pos) - (robot_width + block_width) / 2
        )

        # Reward for moving closer to block 2
        # Scale the reward to be smaller than the main task reward
        distance_delta = self.prev_distance_to_block2 - current_distance
        distance_reward = 0.5 * distance_delta  # Scale factor can be tuned

        # Add bonus for being very close to block 2
        if np.isclose(current_distance, 0.0, atol=2e-2):  # Threshold can be tuned
            distance_reward += 0.5

        self.prev_distance_to_block2 = current_distance
        return distance_reward

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment with random block positions."""
        self.steps = 0
        self.prev_distance_to_block2 = None

        # Initialize RNG
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Get random block positions
        block_2_pos = self._get_random_block_2_position(
            rng, ensure_blocking=bool(self._custom_reset_options["ensure_blocking"])
        )

        # Update reset options
        robot_pos = self._custom_reset_options["robot_pos"]
        reset_options = options or {}
        reset_options.update(
            {
                # Place block 1 at robot position (since it's being held)
                "block_1_pos": robot_pos,
                "block_2_pos": block_2_pos,
                "robot_pos": robot_pos,
            }
        )

        # Reset the base environment with our options
        obs, info = self.env.reset(seed=seed, options=reset_options)

        # Initialize distance tracking
        robot_pos = obs[0:2]
        block2_pos = obs[6:8]
        robot_width = obs[2]
        block_width = obs[8]
        self.prev_distance_to_block2 = (
            np.linalg.norm(robot_pos - block2_pos) - (robot_width + block_width) / 2
        )

        # Verify block 2 is blocking if required
        if self._custom_reset_options[
            "ensure_blocking"
        ] and not self.is_target_area_blocked(obs):
            return self.reset(seed=seed, options=options)

        return obs, info

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Step the environment."""
        obs, _, _, _, info = self.env.step(action)
        self.steps += 1

        # Check target area and preconditions
        is_blocked = self.is_target_area_blocked(obs)
        precondition_penalty = self._calculate_precondition_violation_penalty(obs)

        # Episode terminates if target area is clear or max steps exceeded
        terminated = not is_blocked
        truncated = self.steps >= self.max_episode_steps

        # Calculate rewards
        if not is_blocked and precondition_penalty == 0.0:
            # Large reward for success (clearing area) while maintaining preconditions
            reward = 10.0
        else:
            # Small step penalty to encourage efficiency
            reward = -0.1
            # Distance-based reward
            reward += self._calculate_distance_reward(obs)
            # Collision penalty
            if self._check_collision(obs):
                reward -= 0.1
            # Penalty for violating preconditions
            reward += precondition_penalty

        return obs, reward, terminated, truncated, info


def make_pushing_env(
    env: Blocks2DEnv, max_episode_steps: int = 100, seed: int | None = None
) -> PushingEnvWrapper:
    """Create a pushing environment."""
    wrapped_env = PushingEnvWrapper(env, seed=seed)
    wrapped_env.max_episode_steps = max_episode_steps
    return wrapped_env
