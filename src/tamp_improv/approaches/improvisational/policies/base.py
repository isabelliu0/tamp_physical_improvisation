"""Base policy interface for improvisational approaches."""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    current_atoms: list[set[GroundAtom]]
    preimages: list[set[GroundAtom]]
    config: dict[str, Any]

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: Path) -> None:
        """Save training data to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save states as numpy array
        states_path = path / "states.npy"
        np.save(states_path, np.array(self.states))

        # Save atoms and preimages as pickle
        data_paths = {
            "current_atoms": self.current_atoms,
            "preimages": self.preimages,
        }
        for name, obj in data_paths.items():
            file_path = path / f"{name}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)

        # Save config as JSON
        serializable_config = dict(self.config)

        if "atom_to_index" in serializable_config:
            # Convert keys to strings if they aren't already
            atom_to_index = {
                str(k): v for k, v in serializable_config["atom_to_index"].items()
            }
            serializable_config["atom_to_index"] = atom_to_index

        config_path = path / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(serializable_config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> TrainingData:
        """Load training data from disk."""
        # Load states
        states_path = path / "states.npy"
        states = list(np.load(states_path))

        # Load atoms and preimages
        data = {}
        data_names = [
            "current_atoms",
            "preimages",
        ]
        for name in data_names:
            file_path = path / f"{name}.pkl"
            if file_path.exists():
                with open(file_path, "rb") as f:
                    data[name] = pickle.load(f)
            else:
                print(f"Warning: {file_path} not found, using empty list.")
                data[name] = []

        # Load config
        config_path = path / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "atom_to_index" in config:
            config["atom_to_index"] = {
                k: int(v) if isinstance(v, str) else v
                for k, v in config["atom_to_index"].items()
            }

        return cls(
            states=states,
            current_atoms=data["current_atoms"],
            preimages=data["preimages"],
            config=config,
        )


@dataclass
class PolicyContext(Generic[ObsType, ActType]):
    """Context information passed from approach to policy."""

    preimage: set[GroundAtom]
    current_atoms: set[GroundAtom]
    info: dict[str, Any] = field(default_factory=dict)


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
    def can_initiate(self) -> bool:
        """Check whether the policy can be executed given the current
        context."""

    @abstractmethod
    def get_action(self, obs: ObsType) -> ActType:
        """Get action from policy."""

    def configure_context(self, context: PolicyContext[ObsType, ActType]) -> None:
        """Configure policy with context information."""

    def train(self, env: gym.Env, train_data: TrainingData) -> None:
        """Train the policy if needed.

        Default implementation just initializes the policy and updates
        context. Policies that need training should override this.
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
