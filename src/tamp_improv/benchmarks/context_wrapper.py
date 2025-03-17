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
        self, env: gym.Env, perceiver: Perceiver[ObsType], max_preimage_size: int = 10
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

    def reset(self, **kwargs: Any) -> tuple[ObsType, dict[str, Any]]:
        """Reset environment and augment observation."""
        obs, info = self.env.reset(**kwargs)
        self.current_atoms = self.perceiver.step(obs)
        return self.augment_observation(obs), info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Take a step and augment observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_atoms = self.perceiver.step(obs)
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
        """Augment observation with 1-hot vector for preimage atoms."""
        features = np.zeros(self.num_context_features, dtype=np.float32)
        if not self.current_preimage:
            return cast(ObsType, np.concatenate([obs, features]))

        preimage_atoms = list(self.current_preimage)
        current_atoms_str = {str(atom) for atom in self.current_atoms}
        for i, atom in enumerate(preimage_atoms):
            if str(atom) in current_atoms_str:
                features[i] = 1.0

        return cast(ObsType, np.concatenate([obs, features]))

    def set_context(
        self, current_atoms: set[GroundAtom], preimage: set[GroundAtom]
    ) -> None:
        """Set current context for augmentation."""
        self.current_atoms = current_atoms
        self.current_preimage = preimage
        assert (
            len(preimage) <= self.num_context_features
        ), "Preimage bigger than context size"

    def configure_training(self, train_data: TrainingData) -> None:
        """Configure environment for training with data."""
        if hasattr(self.env, "configure_training"):
            self.env.configure_training(train_data)

    def _atoms_intersection(
        self, atoms1: set[GroundAtom], atoms2: set[GroundAtom]
    ) -> set[str]:
        """Find atoms that are present in both sets (approximate)."""
        atoms1_str = {str(atom) for atom in atoms1}
        atoms2_str = {str(atom) for atom in atoms2}
        return atoms1_str.intersection(atoms2_str)

    def _atoms_union(
        self, atoms1: set[GroundAtom], atoms2: set[GroundAtom]
    ) -> set[str]:
        """Union of two atom sets (approximate)."""
        atoms1_str = {str(atom) for atom in atoms1}
        atoms2_str = {str(atom) for atom in atoms2}
        return atoms1_str.union(atoms2_str)

    def _get_predicate_names(self, atoms: set[GroundAtom]) -> set[str]:
        """Get set of predicate names from atoms."""
        return {atom.predicate.name for atom in atoms}

    def _get_object_types(self, atoms: set[GroundAtom]) -> set[str]:
        """Get set of object types from atoms."""
        types = set()
        for atom in atoms:
            for obj in atom.objects:
                types.add(obj.type.name)
        return types
