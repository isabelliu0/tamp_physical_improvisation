"""Multi-Policy RL implementation."""

import copy
import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypeVar

import gymnasium as gym
from relational_structs import GroundAtom, Object

from tamp_improv.approaches.improvisational.policies.base import (
    Policy,
    PolicyContext,
    TrainingData,
)
from tamp_improv.approaches.improvisational.policies.rl import (
    RLConfig,
    RLPolicy,
    TrainingProgressCallback,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class MultiRLPolicy(Policy[ObsType, ActType]):
    """Policy that uses multiple specialized RL policies for different
    shortcuts."""

    def __init__(self, seed: int, config: RLConfig | None = None) -> None:
        """Initialize with a seed and optional config."""
        super().__init__(seed)
        self.config = config or RLConfig()
        self._policies: dict[str, RLPolicy] = {}
        self._active_policy_key: str | None = None
        self._current_context: PolicyContext | None = None
        self._policy_patterns: dict[str, dict[str, set[GroundAtom]]] = {}

    @property
    def requires_training(self) -> bool:
        """Whether this policy requires training."""
        return True

    def initialize(self, env: gym.Env) -> None:
        """Initialize the policy."""
        # Base initialization doesn't do much since we'll create
        # specialized policies as needed

    def configure_context(self, context: PolicyContext) -> None:
        """Configure policy with context information."""
        self._current_context = context

        matching_policy = self._find_matching_policy(context)
        if matching_policy:
            self._active_policy_key = matching_policy
            self._policies[matching_policy].configure_context(context)
        else:
            self._active_policy_key = None

    def can_initiate(self) -> bool:
        """Check if we can handle the current context."""
        if not self._current_context:
            return False
        return self._find_matching_policy(self._current_context) is not None

    def get_action(self, obs: ObsType) -> ActType:
        """Get action from the appropriate policy."""
        if not self._active_policy_key or self._active_policy_key not in self._policies:
            raise ValueError("No active policy for current context")
        return self._policies[self._active_policy_key].get_action(obs)

    def train(self, env: gym.Env, train_data: TrainingData | None) -> None:
        """Train multiple specialized policies."""
        assert train_data is not None
        print("\n=== Training Multi-Policy RL ===")
        print(f"Total training examples: {len(train_data.states)}")

        # Group training data by shortcut signature
        grouped_data = self._group_training_data(train_data)

        # Train a policy for each group
        for policy_key, group_data in grouped_data.items():
            print(f"\nTraining policy for shortcut type: {policy_key}")
            print(f"Training examples: {len(group_data.states)}")

            if policy_key not in self._policies:
                self._policies[policy_key] = RLPolicy(self._seed, self.config)

            # Configure the environment with only this group's data
            policy_env = copy.deepcopy(env)
            self._configure_env_recursively(policy_env, group_data)

            # Train the policy with this subset of data with the custom callback
            callback = TrainingProgressCallback(
                check_freq=train_data.config.get("training_record_interval", 100),
                early_stopping=True,
                early_stopping_patience=1,
                early_stopping_threshold=0.8,
                policy_key=policy_key,
            )
            self._policies[policy_key].train(policy_env, group_data, callback=callback)

        print(f"\nTrained {len(self._policies)} specialized policies")

    def _get_policy_key(self, context: PolicyContext) -> str:
        """Create a unique key for a policy based on the context."""
        # Get ground atoms as strings to preserve object information
        source_atoms_str = sorted([str(atom) for atom in context.current_atoms])
        target_atoms_str = sorted([str(atom) for atom in context.preimage])

        # Create hash of the source and target atoms
        source_hash = hashlib.md5("|".join(source_atoms_str).encode()).hexdigest()[:8]
        target_hash = hashlib.md5("|".join(target_atoms_str).encode()).hexdigest()[:8]

        # Include source and target node IDs if available
        source_id = context.info.get("source_node_id", "")
        target_id = context.info.get("target_node_id", "")

        if source_id != "" and target_id != "":
            return f"n{source_id}-to-n{target_id}_{source_hash}_{target_hash}"
        return f"{source_hash}_{target_hash}"

    def _find_matching_policy(self, context: PolicyContext) -> str | None:
        """Find a matching policy based on relevant atoms of the shortcut."""
        source_atoms = context.current_atoms
        target_atoms = context.preimage
        added_atoms = target_atoms - source_atoms
        deleted_atoms = source_atoms - target_atoms

        # Try exact key match first
        key = self._get_policy_key(context)
        if key in self._policies:
            return key

        # Try structural matching
        for policy_key, pattern_info in self._policy_patterns.items():
            train_added = pattern_info["added_atoms"]
            train_deleted = pattern_info["deleted_atoms"]

            # Check predicate subsets
            if not self._check_predicate_subset(added_atoms, train_added):
                continue
            if not self._check_predicate_subset(deleted_atoms, train_deleted):
                continue

            # Pool atoms together for substitution finding
            test_atoms = added_atoms.union(deleted_atoms)
            train_atoms = set(train_added).union(train_deleted)

            # Find substitution
            match_found, _ = find_atom_substitution(train_atoms, test_atoms)
            if match_found:
                return policy_key

        return None

    def _check_predicate_subset(
        self, test_atoms: set[GroundAtom], train_atoms: set[GroundAtom]
    ) -> bool:
        """Check if predicate names in train_atoms are a subset of those in
        test_atoms."""
        test_predicates = {atom.predicate.name for atom in test_atoms}
        train_predicates = {atom.predicate.name for atom in train_atoms}
        return train_predicates.issubset(test_predicates)

    def _group_training_data(self, train_data: TrainingData) -> dict[str, TrainingData]:
        """Group training data by shortcut signature."""
        grouped: dict[str, dict] = {}
        shortcut_info = train_data.config.get("shortcut_info", [])

        for i in range(len(train_data)):
            current_atoms = train_data.current_atoms[i]
            preimage = train_data.preimages[i]
            info = {}
            if i < len(shortcut_info):
                info = shortcut_info[i]

            # Create a context and get the key
            context: PolicyContext[ObsType, ActType] = PolicyContext(
                current_atoms=current_atoms, preimage=preimage, info=info
            )
            policy_key = self._get_policy_key(context)

            if policy_key not in grouped:
                grouped[policy_key] = {
                    "states": [],
                    "current_atoms": [],
                    "preimages": [],
                    "pattern": {
                        "added_atoms": preimage - current_atoms,
                        "deleted_atoms": current_atoms - preimage,
                    },
                }

            grouped[policy_key]["states"].append(train_data.states[i])
            grouped[policy_key]["current_atoms"].append(current_atoms)
            grouped[policy_key]["preimages"].append(preimage)

        # Convert grouped data to TrainingData objects
        result = {}
        for key, group in grouped.items():
            result[key] = TrainingData(
                states=group["states"],
                current_atoms=group["current_atoms"],
                preimages=group["preimages"],
                config=train_data.config,
            )
            self._policy_patterns[key] = group["pattern"]
        return result

    def _configure_env_recursively(
        self, env: gym.Env, training_data: TrainingData
    ) -> None:
        """Recursively unwrap environment to configure the trainable
        wrapper."""
        if hasattr(env, "configure_training"):
            env.configure_training(training_data)
        if hasattr(env, "env"):
            self._configure_env_recursively(env.env, training_data)

    def save(self, path: str) -> None:
        """Save all policies."""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save each policy in its own subdirectory
        for key, policy in self._policies.items():
            safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
            policy_path = os.path.join(path, f"policy_{safe_key}")
            policy.save(policy_path)

        # Save pattern information
        for policy_key, pattern in self._policy_patterns.items():
            pattern_file = os.path.join(path, f"pattern_{policy_key}.pkl")
            with open(pattern_file, "wb") as f:
                pickle.dump(pattern, f)

        # Save a manifest of all policies
        manifest = {
            "policies": list(self._policies.keys()),
            "policy_count": len(self._policies),
        }
        with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)

    def load(self, path: str) -> None:
        """Load all policies."""
        with open(os.path.join(path, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Load pattern information if available
        self._policy_patterns = {}
        for policy_key in manifest["policies"]:
            pattern_file = os.path.join(path, f"pattern_{policy_key}.pkl")
            if os.path.exists(pattern_file):
                with open(pattern_file, "rb") as f:
                    self._policy_patterns[policy_key] = pickle.load(f)

        # Load individual policies
        self._policies = {}
        for key in manifest["policies"]:
            safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
            policy_path = os.path.join(path, f"policy_{safe_key}")

            policy: RLPolicy = RLPolicy(self._seed, self.config)
            policy.load(policy_path)

            self._policies[key] = policy

        print(f"Loaded {len(self._policies)} specialized policies")


def find_atom_substitution(
    train_atoms: set[GroundAtom], test_atoms: set[GroundAtom]
) -> tuple[bool, dict[Object, Object]]:
    """Find if train_atoms can be mapped to a subset of test_atoms."""
    test_atoms_by_pred = defaultdict(list)
    for atom in test_atoms:
        test_atoms_by_pred[atom.predicate.name].append(atom)

    # Quick check - if there are enough atoms of each predicate type in test_atoms
    train_pred_counts = Counter(atom.predicate.name for atom in train_atoms)
    for pred_name, count in train_pred_counts.items():
        if len(test_atoms_by_pred[pred_name]) < count:
            return False, {}

    train_objs_by_type: dict[Any, list[Object]] = defaultdict(list)
    test_objs_by_type: dict[Any, list[Object]] = defaultdict(list)
    for atom in train_atoms:
        for obj in atom.objects:
            if obj not in train_objs_by_type[obj.type]:
                train_objs_by_type[obj.type].append(obj)
    for atom in test_atoms:
        for obj in atom.objects:
            if obj not in test_objs_by_type[obj.type]:
                test_objs_by_type[obj.type].append(obj)

    # Quick check - if there are enough test objects for each type
    for obj_type, objs in train_objs_by_type.items():
        if len(test_objs_by_type[obj_type]) < len(objs):
            return False, {}

    # Sort train objects to ensure deterministic behavior
    train_objects = []
    for obj_type in sorted(train_objs_by_type.keys(), key=lambda t: t.name):
        train_objects.extend(sorted(train_objs_by_type[obj_type], key=lambda o: o.name))

    return find_substitution_helper(
        train_atoms=train_atoms,
        test_atoms_by_pred=test_atoms_by_pred,
        remaining_train_objs=train_objects,
        test_objs_by_type=test_objs_by_type,
        partial_sub={},
    )


def find_substitution_helper(
    train_atoms: set[GroundAtom],
    test_atoms_by_pred: dict[str, list[GroundAtom]],
    remaining_train_objs: list[Object],
    test_objs_by_type: dict[Any, list[Object]],
    partial_sub: dict[Object, Object],
) -> tuple[bool, dict[Object, Object]]:
    """Helper to find_atom_substitution using backtracking search."""
    if not remaining_train_objs:
        return check_substitution_valid(train_atoms, test_atoms_by_pred, partial_sub)

    train_obj = remaining_train_objs[0]
    remaining = remaining_train_objs[1:]
    for test_obj in test_objs_by_type[train_obj.type]:
        if test_obj in partial_sub.values():
            continue
        new_sub = partial_sub.copy()
        new_sub[train_obj] = test_obj
        success, final_sub = find_substitution_helper(
            train_atoms=train_atoms,
            test_atoms_by_pred=test_atoms_by_pred,
            remaining_train_objs=remaining,
            test_objs_by_type=test_objs_by_type,
            partial_sub=new_sub,
        )
        if success:
            return True, final_sub

    return False, {}


def check_substitution_valid(
    train_atoms: set[GroundAtom],
    test_atoms_by_pred: dict[str, list[GroundAtom]],
    substitution: dict[Object, Object],
) -> tuple[bool, dict[Object, Object]]:
    """Check if substitution maps all train_atoms to some subset of
    test_atoms."""
    for train_atom in train_atoms:
        pred_name = train_atom.predicate.name
        if pred_name not in test_atoms_by_pred:
            return False, {}
        subst_objs = tuple(substitution[obj] for obj in train_atom.objects)
        found_match = False
        for test_atom in test_atoms_by_pred[pred_name]:
            if tuple(test_atom.objects) == subst_objs:
                found_match = True
                break
        if not found_match:
            return False, {}
    return True, substitution
