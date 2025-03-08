"""Hard-coded pushing policy for testing the framework."""

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from relational_structs import GroundAtom

from tamp_improv.approaches.improvisational.policies.base import (
    Policy,
    PolicyContext,
    TrainingData,
)


class PushingPolicy(Policy[NDArray[np.float32], NDArray[np.float32]]):
    """Hard-coded policy for pushing block 2 out of the way."""

    def __init__(self, seed: int) -> None:
        """Initialize policy."""
        super().__init__(seed)
        self._env: gym.Env | None = None
        self._current_atoms: set[GroundAtom] | None = None
        self._target_preimage: set[GroundAtom] | None = None

    @property
    def requires_training(self) -> bool:
        """Whether this policy requires training data and training."""
        return False

    def initialize(self, env: gym.Env) -> None:
        """Initialize policy with environment."""
        self._env = env

    def configure_context(self, context: PolicyContext) -> None:
        """Configure policy with context information."""
        self._current_atoms = context.current_atoms
        self._target_preimage = context.preimage

        # Print relevant information for debugging
        print("\nConfiguring hard-coded PushingPolicy with context:")
        print(f"Current atoms: {self._current_atoms}")
        print(f"Target preimage: {self._target_preimage}")

    def get_action(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get action for pushing block 2 out of the way."""
        robot_x, robot_y, robot_width = obs[0:3]
        block_1_x = obs[4]
        block_2_x, block_2_y = obs[6:8]
        block_width = obs[8]
        target_x = obs[11]
        target_width = obs[13]

        # Determine push direction
        space_on_left = (
            abs((target_x - target_width / 2) - (block_1_x + block_width / 2))
            if block_1_x < target_x
            else abs(target_x - target_width / 2)
        )
        space_on_right = (
            abs((block_1_x - block_width / 2) - (target_x + target_width / 2))
            if block_1_x > target_x
            else abs(1.0 - (target_x + target_width / 2))
        )

        # Push in direction with more space
        push_direction = -0.1 if space_on_left > space_on_right else 0.1

        # Calculate target position for pushing
        target_x_offset = (robot_width + block_width) / 2  # robot_width + block_width
        target_x_offset *= -np.sign(push_direction)
        target_robot_x = block_2_x + target_x_offset

        # Calculate distance to pushing position
        dist_to_target = np.hypot(target_robot_x - robot_x, block_2_y - robot_y)

        if dist_to_target > 0.1:
            # Move to pushing position
            dx = np.clip(target_robot_x - robot_x, -0.1, 0.1)
            dy = np.clip(block_2_y - robot_y, -0.1, 0.1)
            return np.array([dx, dy, obs[10]])

        # Push
        return np.array([push_direction, 0.0, obs[10]])

    def train(self, env: gym.Env, train_data: TrainingData) -> None:
        """No training needed for hard-coded policy."""

    def save(self, path: str) -> None:
        """Save policy parameters."""

    def load(self, path: str) -> None:
        """Load policy parameters."""
