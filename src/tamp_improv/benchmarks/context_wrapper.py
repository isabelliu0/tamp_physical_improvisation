"""Context-aware environment wrapper for improvisational TAMP."""

from typing import Any, cast

import gymnasium as gym
import numpy as np
from relational_structs import GroundAtom
from task_then_motion_planning.structs import Perceiver

from tamp_improv.approaches.improvisational.policies.base import TrainingData
from tamp_improv.benchmarks.wrappers import ActType, ObsType


class ContextAwareWrapper(gym.Wrapper):
    """Wrapper that augments observations with context information."""

    def __init__(
        self, env: gym.Env, perceiver: Perceiver[ObsType], max_preimage_size: int = 12
    ) -> None:
        super().__init__(env)
        self.perceiver: Perceiver[ObsType] = perceiver
        self.current_preimage: set[GroundAtom] = set()
        self.current_atoms: set[GroundAtom] = set()

        self.num_context_features = max_preimage_size
        # Add context features to observation space
        if isinstance(env.observation_space, gym.spaces.Box):
            self.observation_space = gym.spaces.Box(
                low=np.append(
                    env.observation_space.low, np.zeros(self.num_context_features)
                ),
                high=np.append(
                    env.observation_space.high, np.ones(self.num_context_features)
                ),
                dtype=np.float32,
            )
            print(
                f"Initialized context wrapper with {self.num_context_features} features"
            )

        # Dictionary mapping atom strings to unique indices
        self._atom_to_index: dict[str, int] = {}
        self._next_index = 0

    def reset(self, **kwargs: Any) -> tuple[ObsType, dict[str, Any]]:
        """Reset environment and augment observation."""
        obs, info = self.env.reset(**kwargs)
        self.current_atoms = self.perceiver.step(obs)
        if hasattr(self.env, "current_preimage"):
            self.current_preimage = self.env.current_preimage
        return self.augment_observation(obs), info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Take a step and augment observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_atoms = self.perceiver.step(obs)
        if hasattr(self.env, "current_preimage"):
            self.current_preimage = self.env.current_preimage
        return (
            self.augment_observation(obs),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def reset_from_state(
        self, state: ObsType, **kwargs: Any
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset from state with context augmentation."""
        if hasattr(self.env, "reset_from_state"):
            obs, info = self.env.reset_from_state(state, **kwargs)
            self.current_atoms = self.perceiver.step(obs)
            return self.augment_observation(obs), info
        raise AttributeError("Wrapped environment doesn't have reset_from_state")

    def augment_observation(self, obs: ObsType) -> ObsType:
        """Augment observation with multi-hot vector for preimage atoms."""
        context = np.zeros(self.num_context_features, dtype=np.float32)
        if not self.current_preimage:
            return cast(ObsType, np.concatenate([obs, context]))
        for atom in self.current_preimage:
            idx = self._get_atom_index(str(atom))
            context[idx] = 1.0
        return cast(ObsType, np.concatenate([obs, context]))

    def set_context(
        self, current_atoms: set[GroundAtom], preimage: set[GroundAtom]
    ) -> None:
        """Set current context for augmentation."""
        self.current_atoms = current_atoms
        self.current_preimage = preimage
        assert (
            len(preimage) <= self.num_context_features
        ), "Preimage bigger than context size"
        for atom in current_atoms.union(preimage):
            self._get_atom_index(str(atom))

    def configure_training(self, train_data: TrainingData) -> None:
        """Configure environment for training with data."""
        if hasattr(self.env, "configure_training"):
            self.env.configure_training(train_data)

        # Load existing atom-to-index mapping if available
        if "atom_to_index" in train_data.config and train_data.config["atom_to_index"]:
            self._atom_to_index = train_data.config["atom_to_index"]
            self._next_index = (
                max(self._atom_to_index.values()) + 1 if self._atom_to_index else 0
            )
            print(f"Loaded {len(self._atom_to_index)} atoms with fixed indices")
        else:
            # Collect all unique atoms from training data
            unique_atoms = set()
            for atoms_set in train_data.current_atoms:
                for atom in atoms_set:
                    unique_atoms.add(str(atom))
            for preimage in train_data.preimages:
                for atom in preimage:
                    unique_atoms.add(str(atom))

            for atom_str in unique_atoms:
                self._get_atom_index(atom_str)

            print(f"New atom-to-index mapping with {len(self._atom_to_index)} entries")

        for atom_str, idx in self._atom_to_index.items():
            print(f"Atom {atom_str} -> index {idx}")

    def _get_atom_index(self, atom_str: str) -> int:
        """Get a unique index for this atom."""
        if atom_str in self._atom_to_index:
            return self._atom_to_index[atom_str]
        assert (
            self._next_index < self.num_context_features
        ), "No more space for new atoms. Increase max_preimage_size"
        idx = self._next_index
        self._atom_to_index[atom_str] = idx
        self._next_index += 1
        return idx

    def get_atom_index_mapping(self) -> dict[str, int]:
        """Get the current atom to index mapping."""
        return self._atom_to_index.copy()
