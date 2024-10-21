"""Defines the base class for approaches."""

import abc
from typing import Any, Generic

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class BaseApproach(abc.ABC, Generic[ObsType, ActType]):
    """A base class for approaches."""

    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, seed: int
    ) -> None:
        self._observation_space = observation_space
        self._action_space = action_space
        self._seed = seed
        self._action_space.seed(seed)

    @abc.abstractmethod
    def reset(self, obs: ObsType) -> ActType:
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
