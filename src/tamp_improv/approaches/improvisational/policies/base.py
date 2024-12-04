"""Base policy interface for improvisational approaches."""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np
from relational_structs import GroundAtom

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class TrainingData:
    """Container for policy training data."""

    states: list[Any]  # List of states where intervention needed
    preconditions_to_maintain: list[set[GroundAtom]]
    preconditions_to_achieve: list[set[GroundAtom]]
    config: dict[str, Any]

    def __post_init__(self):
        """Validate data."""
        assert len(self.states) == len(self.preconditions_to_maintain) and len(
            self.states
        ) == len(
            self.preconditions_to_achieve
        ), "Collected training states and preconditions must be equal length."

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: Path) -> None:
        """Save training data to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save states as numpy array
        states_path = path / "states.npy"
        np.save(states_path, np.array(self.states))

        # Save preconditions as pickle (since they contain custom objects)
        preconditions_paths = {
            "preconditions_to_maintain": self.preconditions_to_maintain,
            "preconditions_to_achieve": self.preconditions_to_achieve,
        }
        for name, obj in preconditions_paths.items():
            file_path = path / f"{name}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)

        # Save config as JSON
        config_path = path / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f)

    @classmethod
    def load(cls, path: Path) -> TrainingData:
        """Load training data from disk."""
        # Load states
        states_path = path / "states.npy"
        states = list(np.load(states_path))

        # Load preconditions
        preconditions = {}
        preconditions_names = ["preconditions_to_maintain", "preconditions_to_achieve"]
        for name in preconditions_names:
            file_path = path / f"{name}.pkl"
            with open(file_path, "rb") as f:
                preconditions[name] = pickle.load(f)

        # Load config
        config_path = path / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return cls(
            states=states,
            preconditions_to_maintain=preconditions["preconditions_to_maintain"],
            preconditions_to_achieve=preconditions["preconditions_to_achieve"],
            config=config,
        )


class Policy(Generic[ObsType, ActType], ABC):
    """Base class for policies."""

    def __init__(self, seed: int) -> None:
        """Initialize policy with environment."""
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
        if hasattr(env, "configure_training"):
            env.configure_training(train_data)

    @abstractmethod
    def save(self, path: str) -> None:
        """Save policy to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load policy from disk."""
