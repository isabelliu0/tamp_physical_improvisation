"""Base class for all approaches."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseApproach(Generic[ObsType, ActType], ABC):
    """Base class for all approaches."""

    def __init__(
        self, system: ImprovisationalTAMPSystem[ObsType, ActType], seed: int
    ) -> None:
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
