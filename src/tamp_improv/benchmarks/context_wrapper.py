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

    def __init__(self, env: gym.Env, perceiver: Perceiver[ObsType]) -> None:
        super().__init__(env)
        self.perceiver: Perceiver[ObsType] = perceiver
        self.current_preimage: set[GroundAtom] = set()
        self.current_atoms: set[GroundAtom] = set()

        # Add context features to observation space
        if isinstance(env.observation_space, gym.spaces.Box):
            self.num_context_features = 4
            self.observation_space = gym.spaces.Box(
                low=np.append(
                    env.observation_space.low, np.zeros(self.num_context_features)
                ),
                high=np.append(
                    env.observation_space.high, np.ones(self.num_context_features)
                ),
                dtype=np.float32,
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
        """Augment observation with domain-agnostic context features."""
        features = np.zeros(self.num_context_features, dtype=np.float32)

        # Feature 1: Overlap ratio between current atoms and preimage
        overlap_size = len(
            self._atoms_intersection(self.current_atoms, self.current_preimage)
        )
        union_size = len(self._atoms_union(self.current_atoms, self.current_preimage))
        features[0] = overlap_size / max(union_size, 1)

        # Feature 2: Relative size - how many atoms remain to be achieved
        preimage_size = len(self.current_preimage)
        features[1] = (preimage_size - overlap_size) / max(preimage_size, 1)

        # Feature 3: Predicates in common percentage
        current_preds = self._get_predicate_names(self.current_atoms)
        preimage_preds = self._get_predicate_names(self.current_preimage)
        common_preds = len(current_preds.intersection(preimage_preds))
        all_preds = len(current_preds.union(preimage_preds))
        features[2] = common_preds / max(all_preds, 1)

        # Feature 4: Object types in common percentage
        current_types = self._get_object_types(self.current_atoms)
        preimage_types = self._get_object_types(self.current_preimage)
        common_types = len(current_types.intersection(preimage_types))
        all_types = len(current_types.union(preimage_types))
        features[3] = common_types / max(all_types, 1)

        # Concatenate original observation with features
        return cast(ObsType, np.concatenate([obs, features]))

    def set_context(
        self, current_atoms: set[GroundAtom], preimage: set[GroundAtom]
    ) -> None:
        """Set current context for augmentation."""
        self.current_atoms = current_atoms
        self.current_preimage = preimage

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
