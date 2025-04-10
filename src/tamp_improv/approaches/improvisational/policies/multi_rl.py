"""Multi-Policy RL implementation."""

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import TypeVar

import gymnasium as gym

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

    @property
    def requires_training(self) -> bool:
        """Whether this policy requires training."""
        return True

    def initialize(self, env: gym.Env) -> None:
        """Initialize the policy."""
        # Base initialization doesn't do much since we'll create
        # specialized policies as needed

    def _get_policy_key(self, context: PolicyContext) -> str:
        """Create a unique key for a policy based on the context.

        Encodes both predicates and object information to ensure
        different shortcuts with similar structure but different objects
        get different policies.
        """
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

    def configure_context(self, context: PolicyContext) -> None:
        """Configure policy with context information."""
        self._current_context = context

        policy_key = self._get_policy_key(context)
        self._active_policy_key = policy_key

        if policy_key in self._policies:
            self._policies[policy_key].configure_context(context)

    def can_initiate(self) -> bool:
        """Check if we can handle the current context."""
        if not self._current_context:
            return False

        policy_key = self._get_policy_key(self._current_context)
        return policy_key in self._policies

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

    def _group_training_data(self, train_data: TrainingData) -> dict[str, TrainingData]:
        """Group training data by shortcut signature."""
        grouped: dict[str, dict[str, list]] = {}
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

        self._policies = {}
        for key in manifest["policies"]:
            safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
            policy_path = os.path.join(path, f"policy_{safe_key}")

            policy: RLPolicy = RLPolicy(self._seed, self.config)
            policy.load(policy_path)

            self._policies[key] = policy

        print(f"Loaded {len(self._policies)} specialized policies")
