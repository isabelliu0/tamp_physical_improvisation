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

        # Initialize tracking variables
        self._prev_block2_target_dist: Optional[float] = None
        self._prev_robot_block2_dist: Optional[float] = None
        self._prev_gripper_status: Optional[float] = None

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
        margin = block_width / 2

        # Block 1 starts in left or right side, but not in target area
        if rng.random() < 0.5:
            block_1_x = rng.uniform(0.0, target_x - margin)
        else:
            block_1_x = rng.uniform(target_x + margin, 1.0)
        block_1_y = 0.0
        block_1_pos = np.array([block_1_x, block_1_y], dtype=np.float32)

        # Block 2 position depends on whether it should block target area
        if ensure_blocking:
            # Place block 2 somewhere in or near target area
            block_2_x = rng.uniform(target_x - margin, target_x + margin)
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

    def _calculate_reward(
        self,
        obs: NDArray[np.float32],
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Calculate reward based on multiple factors.

        1. Success reward: +10.0 for successfully clearing the target area
        2. Step penalty: -0.1 per step to encourage efficiency
        3. Distance-based rewards:
            Positive reward for moving block 2 away when blocking
            Small penalty for moving block 2 away when not blocking
            Reward for robot approaching block 2 efficiently
        4. Penalty for excessive gripper toggling (-0.2)
        """
        robot_x, robot_y = obs[0], obs[1]
        block2_x, block2_y = obs[6], obs[7]
        target_x, target_y = obs[11], obs[12]
        gripper_status = obs[10]

        # Base reward component: -1.0 per step to encourage faster completion
        reward = -1.0

        # Success reward
        if terminated and not truncated:  # Successfully cleared target area
            return 20.0

        # Distance-based rewards
        block2_target_dist = np.sqrt(
            (block2_x - target_x) ** 2 + (block2_y - target_y) ** 2
        )
        robot_block2_dist = np.sqrt(
            (robot_x - block2_x) ** 2 + (robot_y - block2_y) ** 2
        )

        # Reward for moving block 2 away from target
        if self._prev_block2_target_dist is not None:
            dist_improvement = self._prev_block2_target_dist - block2_target_dist
            # If block is blocking, reward moving away; if not blocking, small penalty
            if self.is_target_area_blocked(obs):
                reward += 2.0 * dist_improvement
            else:
                reward -= 0.5 * dist_improvement

        # Reward for robot approaching block 2 efficiently
        if self._prev_robot_block2_dist is not None:
            approach_improvement = self._prev_robot_block2_dist - robot_block2_dist
            if robot_block2_dist > 0.3:  # Only reward approaching if not too close
                reward += 0.5 * approach_improvement

        # Store current distances for next step
        self._prev_block2_target_dist = block2_target_dist
        self._prev_robot_block2_dist = robot_block2_dist

        # Penalize excessive gripper toggling
        if (
            self._prev_gripper_status is not None
            and self._prev_gripper_status != gripper_status
        ):
            reward -= 0.2
        self._prev_gripper_status = gripper_status

        return reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        """Reset the environment with random block positions."""
        self.steps = 0
        self._prev_block2_target_dist = None
        self._prev_robot_block2_dist = None
        self._prev_gripper_status = None

        # Initialize RNG and get random block positions
        rng = np.random.default_rng(seed)
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
        """Step the environment with proper collision handling."""
        obs, _, _, truncated, info = self.env.step(action)
        self.steps += 1

        # If collision occurred, episode is truncated
        if info["collision_occurred"]:
            return obs, -10.0, False, True, info

        # Check if target area is clear
        is_blocked = self.is_target_area_blocked(obs)

        # Determine episode termination
        terminated = not is_blocked
        truncated = truncated or self.steps >= self.max_episode_steps

        # Calculate reward only if no collision occurred
        reward = self._calculate_reward(obs, terminated, truncated)

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
