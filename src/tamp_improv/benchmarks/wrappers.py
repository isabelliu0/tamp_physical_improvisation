"""General wrapper for environments supporting improvisational policies."""

from typing import Any, TypeVar

import gymnasium as gym
from relational_structs import GroundAtom
from task_then_motion_planning.structs import Perceiver

from tamp_improv.approaches.improvisational.policies.base import TrainingData

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class ImprovWrapper(gym.Env):
    """General wrapper for training improvisational policies.

    Handles precondition maintenance and achievement during training.
    """

    def __init__(
        self,
        base_env: gym.Env,
        perceiver: Perceiver[ObsType],
        max_episode_steps: int = 100,
    ) -> None:
        """Initialize wrapper with environment and perceiver."""
        self.env = base_env
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        self.perceiver = perceiver

        # Training state tracking
        self.training_states: list[ObsType] = []
        self.preconditions_to_maintain: list[set[GroundAtom]] = []
        self.preconditions_to_achieve: list[set[GroundAtom]] = []
        self.current_precondition_to_maintain: set[GroundAtom] = set()
        self.current_precondition_to_achieve: set[GroundAtom] = set()
        self.current_training_idx: int = 0

        self.render_mode = base_env.render_mode

    def configure_training(
        self,
        training_data: TrainingData,
    ) -> None:
        """Configure environment for training phase."""
        print(f"Configuring environment with {len(training_data)} training scenarios")
        self.training_states = training_data.states
        self.preconditions_to_maintain = training_data.preconditions_to_maintain
        self.preconditions_to_achieve = training_data.preconditions_to_achieve
        self.current_training_idx = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset environment.

        If training states are configured, cycles through them.
        Otherwise uses default environment reset.
        """
        self.steps = 0

        if self.training_states:
            # Get current training scenario and store current preconditions
            self.current_training_idx = self.current_training_idx % len(
                self.training_states
            )
            current_state = self.training_states[self.current_training_idx]
            self.current_precondition_to_maintain = self.preconditions_to_maintain[
                self.current_training_idx
            ]
            self.current_precondition_to_achieve = self.preconditions_to_achieve[
                self.current_training_idx
            ]

            print(f"Training episode {self.current_training_idx + 1}")
            print(
                f"Maintaining precondition(s): {self.current_precondition_to_maintain}"
            )
            print(f"Precondition(s) to achieve: {self.current_precondition_to_achieve}")

            # Reset with current state
            if hasattr(self.env, "reset_from_state"):
                obs, info = self.env.reset_from_state(current_state, seed=seed)
            else:
                raise AttributeError(
                    "The environment does not have a 'reset_from_state' method."
                )

            # Update index cyclically
            self.current_training_idx += 1

            return obs, info

        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Step environment.

        Rewards:
            Base step penalty (-0.1)
            Precondition violation penalty (-1.0)
            Achievement bonus (1.0)
        """
        obs, _, _, truncated, info = self.env.step(action)
        self.steps += 1

        # Check preconditions
        current_atoms = self.perceiver.step(obs)
        precondition_violation = len(
            self.current_precondition_to_maintain
        ) > 0 and not self.current_precondition_to_maintain.issubset(current_atoms)

        # Check achievement
        achieved = self.current_precondition_to_achieve in current_atoms

        # Calculate reward
        reward = -0.1  # Base step penalty
        if precondition_violation:
            reward += -1.0  # Precondition violation penalty
        if achieved:
            reward += 100.0

        # Termination conditions
        terminated = achieved
        truncated = truncated or self.steps >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        """Render the environment."""
        return self.env.render()