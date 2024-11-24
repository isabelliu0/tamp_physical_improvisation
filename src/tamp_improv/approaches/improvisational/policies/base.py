"""Base policy interface for improvisational approaches."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import gymnasium as gym
from relational_structs import GroundAtom, LiftedOperator

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class TrainingData:
    """Container for policy training data."""

    states: list[Any]  # List of states where intervention needed
    operators: list[LiftedOperator]  # Operators whose preconditions weren't met
    preconditions: list[set[GroundAtom]]  # Preconditions to satisfy
    config: dict[str, Any]  # Additional configuration


class Policy(Generic[ObsType, ActType], ABC):
    """Base class for policies."""

    def __init__(self, seed: int) -> None:
        """Initialize policy."""
        self._seed = seed

    @property
    @abstractmethod
    def requires_training(self) -> bool:
        """Whether this policy requires training data and training."""

    @abstractmethod
    def initialize(self, env: gym.Env) -> None:
        """Initialize policy with environment."""

    @abstractmethod
    def get_action(self, obs: ObsType) -> ActType:
        """Get action from policy."""

    def train(self, env: gym.Env, train_data: TrainingData) -> None:
        """Train the policy if needed.

        Default implementation just initializes the policy and updates
        preconditions. Policies that need training should override this.
        """
        self.initialize(env)
        if hasattr(env, "update_preconditions"):
            env.update_preconditions(
                train_data.operators[0], train_data.preconditions[0]
            )

    @abstractmethod
    def save(self, path: str) -> None:
        """Save policy to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load policy from disk."""
