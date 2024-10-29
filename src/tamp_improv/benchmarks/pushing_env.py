"""Environment wrapper for learning the pushing policy in Blocks2D
environment."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
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

    def render(self) -> None:
        """Render the environment.

        We delegate rendering to the base environment.
        """
        self.env.render()

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

        # Reward: -1.0 per step, 0.0 for success
        reward = 0.0 if terminated else -1.0

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
