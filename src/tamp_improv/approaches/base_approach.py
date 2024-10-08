"""Defines the base class for approaches."""

import abc
from typing import Any, Generic

from gymnasium.core import ActType, ObsType


class BaseApproach(abc.ABC, Generic[ObsType, ActType]):
    """A base class for approaches."""

    @abc.abstractmethod
    def reset(self, obs: ObsType, info: dict[str, Any]) -> ActType:
        """Reset to start a new episode and return an initial action."""

    @abc.abstractmethod
    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ActType:
        """Get an action to take and update any internal state."""
