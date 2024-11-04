"""Environment wrapper for learning the pushing policy in Blocks2D
environment."""

from typing import Any, Dict, Optional, Tuple, Union, cast

import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame
from numpy.typing import NDArray

from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv


class PushingEnvWrapper(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Environment wrapper for learning the pushing policy in Blocks2D
    environment."""

    def __init__(self, base_env: Blocks2DEnv) -> None:
        """Initialize pushing environment wrapper.

        Args:
            base_env: The base Blocks2D environment.
            max_episode_steps: Maximum number of steps per episode.
        """
        super().__init__()

        if not isinstance(base_env, Blocks2DEnv):
            raise ValueError("Environment must be a Blocks2DEnv instance")

        self.env = base_env
        self.max_episode_steps = 100
        self.steps = 0

        # Same spaces as base env
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

        # Override env's reset options with our custom options
        self._custom_reset_options = {
            "block_1_pos": None,  # will be randomly determined
            "block_2_pos": None,  # will be randomly determined
            "robot_pos": np.array([0.5, 1.0], dtype=np.float32),
            "ensure_blocking": True,  # Block 2 should block target area
        }

        self.render_mode = self.env.render_mode

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        """Render the environment.

        Returns:
            Union[RenderFrame, list[RenderFrame], None]: The rendered frame(s)
        """
        rendered = self.env.render()
        return cast(Union[RenderFrame, list[RenderFrame], None], rendered)

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

    def _get_random_block_positions(
        self,
        rng: np.random.Generator,
        target_x: float = 0.5,
        ensure_blocking: bool = True,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Generate valid random positions for blocks."""
        obs, _ = self.env.reset()
        block_width = float(obs[8])
        target_width = float(obs[13])
        margin = block_width / 2

        # Block 1 starts in left or right side, but not in target area
        if rng.random() < 0.5:
            block_1_x = rng.uniform(0.0, target_x - margin)
        else:
            block_1_x = rng.uniform(target_x + margin, 1.0)
        block_1_pos = np.array([block_1_x, 0.0], dtype=np.float32)

        # Block 2 position depends on whether it should block target area
        if ensure_blocking:
            # Place block 2 somewhere in or near target area
            block_2_x = rng.uniform(
                target_x - target_width / 2, target_x + target_width / 2
            )  # determines the difficulty of the task for the RL agent
            block_2_pos = np.array([block_2_x, 0.0], dtype=np.float32)
        else:
            # Random position away from target and block 1
            while True:
                if rng.random() < 0.5:
                    block_2_x = rng.uniform(0.0, target_x - margin)
                else:
                    block_2_x = rng.uniform(target_x + margin, 1.0)

                # Check if block 2 is not too close to block 1
                if abs(block_2_x - block_1_x) > block_width:
                    break
            block_2_pos = np.array([block_2_x, 0.0], dtype=np.float32)

        return block_1_pos, block_2_pos

    def _check_collision(self, obs: NDArray[np.float32]) -> bool:
        """Check for collisions in current step using observations."""

        def check_rectangles_collision(
            pos1: NDArray[np.float32], pos2: NDArray[np.float32]
        ) -> bool:
            dx = abs(pos1[0] - pos2[0])
            dy = abs(pos1[1] - pos2[1])

            # Get widths based on whether robot is involved
            block_width = obs[8]
            block_height = obs[9]
            width_sum = block_width + 1e-3
            height_sum = block_height + 1e-3

            # If robot involved, use robot-block dimensions
            if np.array_equal(pos1, obs[0:2]) or np.array_equal(pos2, obs[0:2]):
                width_sum = (
                    obs[2] + block_width
                ) / 2 + 1e-3  # robot_width + block_width
                height_sum = (
                    obs[3] + block_height
                ) / 2 + 1e-3  # robot_height + block_height

            return dx < width_sum and dy < height_sum

        # Get positions from observation
        robot_pos = obs[0:2]  # robot x,y
        block1_pos = obs[4:6]  # block 1 x,y
        block2_pos = obs[6:8]  # block 2 x,y

        # Check all collision pairs
        return (
            (
                check_rectangles_collision(robot_pos, block1_pos)
                and np.isclose(obs[10], 0.0, atol=1e-3)
            )
            or check_rectangles_collision(robot_pos, block2_pos)
            or check_rectangles_collision(block1_pos, block2_pos)
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        """Reset the environment with random block positions."""
        self.steps = 0

        # Initialize RNG
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Get random block positions
        block_1_pos, block_2_pos = self._get_random_block_positions(
            rng, ensure_blocking=bool(self._custom_reset_options["ensure_blocking"])
        )

        # Update reset options with our positions
        reset_options = options or {}
        reset_options.update(
            {
                "block_1_pos": block_1_pos,
                "block_2_pos": block_2_pos,
                "robot_pos": self._custom_reset_options["robot_pos"],
            }
        )

        # Reset the base environment with our options
        obs, info = self.env.reset(seed=seed, options=reset_options)

        # Verify block 2 is blocking if required
        if self._custom_reset_options[
            "ensure_blocking"
        ] and not self.is_target_area_blocked(obs):
            return self.reset(seed=seed, options=options)

        return obs, info

    def step(
        self,
        action: NDArray[np.float32],
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        obs, _, _, _, info = self.env.step(action)
        self.steps += 1

        # Check if target area is clear
        is_blocked = self.is_target_area_blocked(obs)

        # Episode terminates if target area is clear or max steps exceeded
        terminated = not is_blocked
        truncated = self.steps >= self.max_episode_steps

        # Base reward: 1.0 for success, -0.1 per step
        reward = 1.0 if terminated else -0.1

        # Add small collision penalty if collision occurred in this step
        if self._check_collision(obs):
            reward -= 0.5

        return obs, reward, terminated, truncated, info


def make_pushing_env(
    env: Blocks2DEnv, max_episode_steps: int = 100, seed: Optional[int] = None
) -> PushingEnvWrapper:
    """Create a pushing environment."""
    wrapped_env = PushingEnvWrapper(env)
    wrapped_env.max_episode_steps = max_episode_steps
    if seed is not None:
        wrapped_env.reset(seed=seed)
    return wrapped_env
