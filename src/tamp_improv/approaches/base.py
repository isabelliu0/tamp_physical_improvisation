"""Base classes and protocols for approaches."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import gymnasium as gym

from tamp_improv.benchmarks.base import BaseTAMPSystem

# Define type variables with proper variance for protocol
ObsType_contra = TypeVar("ObsType_contra", contravariant=True)
ActType_co = TypeVar("ActType_co", covariant=True)

# Regular type variables for concrete classes
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseApproach(Generic[ObsType, ActType], ABC):
    """Base class for all approaches."""

    def __init__(self, system: BaseTAMPSystem[ObsType, ActType], seed: int) -> None:
        """Initialize approach.

        Args:
            system: The TAMP system to use
            seed: Random seed
        """
        self.system = system
        self._seed = seed

    @abstractmethod
    def reset(self, obs: ObsType, info: dict[str, Any]) -> ActType:
        """Reset approach with initial observation."""

    @abstractmethod
    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ActType:
        """Step approach with new observation."""


class ImprovisationalPolicy(Protocol[ObsType_contra, ActType_co]):
    """Protocol defining the interface for improvisational policies."""

    def train(
        self, env: gym.Env, total_timesteps: int, seed: int | None = None
    ) -> None:
        """Train the policy."""

    def get_action(self, obs: ObsType_contra) -> ActType_co:
        """Get action from policy based on current observation."""

    def save(self, path: str) -> None:
        """Save policy to disk."""

    def load(self, path: str) -> None:
        """Load policy from disk."""


@dataclass
class PolicyConfig:
    """Base configuration for policies."""

    seed: int
