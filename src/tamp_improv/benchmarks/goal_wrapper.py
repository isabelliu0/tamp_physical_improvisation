"""Goal-conditioned wrapper for learning shortcuts in TAMP."""

from typing import Any, TypeVar

import gymnasium as gym
import numpy as np
from relational_structs import GroundAtom
from task_then_motion_planning.structs import Perceiver

from tamp_improv.approaches.improvisational.policies.base import (
    GoalConditionedTrainingData,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class GoalConditionedWrapper(gym.Wrapper):
    """Wrapper that converts an environment to a goal-conditioned format.

    This wrapper:
    1. Augments observations with goal states from the planning graph
    2. Ensures goal node IDs are higher than source node IDs
    3. Provides appropriate rewards for goal achievement
    """

    def __init__(
        self,
        env: gym.Env,
        node_states: dict[int, ObsType],
        valid_shortcuts: list[tuple[int, int]],
        perceiver: Perceiver | None = None,
        node_preimages: dict[int, set[GroundAtom]] | None = None,
        max_preimage_size: int = 12,
        use_preimages: bool = True,
        success_threshold: float = 0.01,
        success_reward: float = 10.0,
        step_penalty: float = -0.5,
        max_episode_steps: int = 50,
    ):
        """Initialize wrapper with node states."""
        super().__init__(env)
        self.node_states = node_states
        self.valid_shortcuts = valid_shortcuts or []
        self.perceiver = perceiver
        self.node_preimages = node_preimages or {}
        self.use_preimages = use_preimages
        self.max_preimage_size = max_preimage_size
        self.success_threshold = success_threshold
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.max_episode_steps = max_episode_steps
        self.steps = 0

        if self.use_preimages and self.node_preimages is not None:
            assert (
                self.perceiver is not None
            ), "Perceiver must be provided when using preimages"
            self.atom_to_index: dict[str, int] = {}
            self._next_index = 0

            # Create multi-hot vectors for all preimages
            self.preimage_vectors: dict[int, np.ndarray] = {}
            for node_id, preimage in self.node_preimages.items():
                self.preimage_vectors[node_id] = self.create_preimage_vector(preimage)

            # Observation space with preimage vectors
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": env.observation_space,
                    "achieved_goal": gym.spaces.Box(
                        0, 1, shape=(max_preimage_size,), dtype=np.float32
                    ),
                    "desired_goal": gym.spaces.Box(
                        0, 1, shape=(max_preimage_size,), dtype=np.float32
                    ),
                }
            )
        else:
            # Original observation space with raw state goals
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": env.observation_space,
                    "achieved_goal": env.observation_space,
                    "desired_goal": env.observation_space,
                }
            )

        # Current episode information
        self.current_node_id: int | None = None
        self.goal_node_id: int | None = None
        self.goal_state: ObsType | None = None
        self.goal_preimage_vector: np.ndarray | None = None
        self.node_ids = sorted(list(node_states.keys()))

    def configure_training(self, train_data: GoalConditionedTrainingData) -> None:
        """Configure environment for training (for compatibility)."""
        assert hasattr(train_data, "node_states") and train_data.node_states is not None
        assert (
            hasattr(train_data, "valid_shortcuts")
            and train_data.valid_shortcuts is not None
        )
        self.node_states = train_data.node_states
        self.valid_shortcuts = train_data.valid_shortcuts
        self.node_preimages = train_data.node_preimages or {}
        print(
            f"Updated {len(self.node_states)} node states, {len(self.valid_shortcuts)} valid shortcuts, and {len(self.node_preimages)}  node preimages from training data"  # pylint: disable=line-too-long
        )
        self.max_episode_steps = train_data.config.get(
            "max_steps", self.max_episode_steps
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, ObsType], dict[str, Any]]:
        """Reset environment and sample a goal."""
        self.steps = 0
        options = options or {}
        self.current_node_id = options.get("source_node_id")
        self.goal_node_id = options.get("goal_node_id")

        self._sample_valid_nodes()
        assert self.current_node_id in self.node_states, "Invalid source node ID"
        current_state = self.node_states[self.current_node_id]

        # Reset with current state
        if hasattr(self.env, "reset_from_state"):
            obs, info = self.env.reset_from_state(current_state, seed=seed)
        else:
            raise AttributeError(
                "The environment does not have a 'reset_from_state' method."
            )
        assert self.current_node_id is not None and self.goal_node_id is not None
        self.goal_state = self.node_states[self.goal_node_id]

        info.update(
            {
                "source_node_id": self.current_node_id,
                "goal_node_id": self.goal_node_id,
            }
        )

        if self.use_preimages:
            self.goal_preimage_vector = self.preimage_vectors[self.goal_node_id]
            current_preimage_vector = self._get_current_preimage_vector(obs)
            dict_obs = {
                "observation": obs,
                "achieved_goal": current_preimage_vector,
                "desired_goal": self.goal_preimage_vector,
            }
        else:
            dict_obs = {
                "observation": obs,
                "achieved_goal": obs,
                "desired_goal": self.goal_state,
            }

        return dict_obs, info

    def reset_from_state(
        self,
        state: ObsType,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, ObsType], dict[str, Any]]:
        """Reset wrapped environment to a specific state."""
        self.steps = 0
        options = options or {}
        self.current_node_id = options.get("source_node_id")
        self.goal_node_id = options.get("goal_node_id")
        self._sample_valid_nodes()

        if hasattr(self.env, "reset_from_state"):
            obs, info = self.env.reset_from_state(state, seed=seed)
        else:
            raise AttributeError(
                "Base environment doesn't have reset_from_state method"
            )
        assert self.current_node_id is not None and self.goal_node_id is not None
        self.goal_state = self.node_states[self.goal_node_id]

        info.update(
            {
                "source_node_id": self.current_node_id,
                "goal_node_id": self.goal_node_id,
            }
        )

        if self.use_preimages:
            self.goal_preimage_vector = self.preimage_vectors[self.goal_node_id]
            current_preimage_vector = self._get_current_preimage_vector(obs)
            dict_obs = {
                "observation": obs,
                "achieved_goal": current_preimage_vector,
                "desired_goal": self.goal_preimage_vector,
            }
        else:
            dict_obs = {
                "observation": obs,
                "achieved_goal": obs,
                "desired_goal": self.goal_state,
            }

        return dict_obs, info

    def step(
        self, action: ActType
    ) -> tuple[dict[str, ObsType], float, bool, bool, dict[str, Any]]:
        """Step environment and compute goal-conditioned rewards."""
        next_obs, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        if self.use_preimages and self.goal_preimage_vector is not None:
            current_preimage_vector = self._get_current_preimage_vector(next_obs)
            goal_indices = np.where(self.goal_preimage_vector > 0.5)[0]
            goal_achieved = np.all(current_preimage_vector[goal_indices] > 0.5)
            preimage_distance = np.sum(current_preimage_vector[goal_indices] < 0.5)
            info.update(
                {
                    "preimage_distance": preimage_distance,
                    "is_success": goal_achieved,
                    "source_node_id": self.current_node_id,
                    "goal_node_id": self.goal_node_id,
                }
            )
            dict_obs = {
                "observation": next_obs,
                "achieved_goal": current_preimage_vector,
                "desired_goal": self.goal_preimage_vector,
            }
        else:
            goal_distance = np.linalg.norm(next_obs - self.goal_state)
            goal_achieved = goal_distance < self.success_threshold
            info.update(
                {
                    "goal_distance": goal_distance,
                    "is_success": goal_achieved,
                    "source_node_id": self.current_node_id,
                    "goal_node_id": self.goal_node_id,
                }
            )
            dict_obs = {
                "observation": next_obs,
                "achieved_goal": next_obs,
                "desired_goal": self.goal_state,
            }

        goal_reward = self.success_reward if goal_achieved else self.step_penalty
        goal_terminated = bool(goal_achieved)
        truncated = truncated or (self.steps >= self.max_episode_steps)
        return dict_obs, goal_reward, goal_terminated or terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: ObsType,
        desired_goal: ObsType,
        info: list[dict[str, Any]] | dict[str, Any],
        _indices: list[int] | None = None,
    ) -> np.ndarray:
        """Compute the reward for achieving a given goal."""
        if self.use_preimages:
            assert hasattr(achieved_goal, "shape")
            assert hasattr(
                desired_goal, "__getitem__"
            ), "desired_goal must be indexable"
            assert hasattr(
                achieved_goal, "__getitem__"
            ), "achieved_goal must be indexable"
            rewards = np.zeros(achieved_goal.shape[0])
            success = np.zeros(achieved_goal.shape[0], dtype=np.bool_)

            for i in range(achieved_goal.shape[0]):
                goal_indices = np.where(desired_goal[i] > 0.5)[0]
                goal_satisfied = np.all(achieved_goal[i][goal_indices] > 0.5)
                rewards[i] = (
                    self.success_reward if goal_satisfied else self.step_penalty
                )
                success[i] = goal_satisfied

            if isinstance(info, list):
                for i, info_dict in enumerate(info):
                    if i < len(success):
                        info_dict["is_success"] = bool(success[i])
            elif isinstance(info, dict):
                if len(success) > 0:
                    info["is_success"] = bool(success[0])
        else:
            distance = np.linalg.norm(
                np.array(achieved_goal) - np.array(desired_goal), axis=-1
            )
            rewards = np.where(
                distance < self.success_threshold,
                self.success_reward,
                self.step_penalty,
            )
            if isinstance(info, list):
                for i, info_dict in enumerate(info):
                    if i < len(distance):
                        info_dict["is_success"] = bool(
                            distance[i] < self.success_threshold
                        )
            elif isinstance(info, dict):
                if isinstance(distance, np.ndarray) and len(distance) > 0:
                    info["is_success"] = bool(distance[0] < self.success_threshold)
                else:
                    info["is_success"] = bool(distance < self.success_threshold)
        return rewards

    def _sample_valid_nodes(self) -> None:
        """Sample valid source and goal nodes."""
        assert self.node_ids
        if self.current_node_id is None:
            source_nodes = set(source_id for source_id, _ in self.valid_shortcuts)
            self.current_node_id = np.random.choice(list(source_nodes))
        if self.goal_node_id is None and self.current_node_id is not None:
            valid_targets = [
                target_id
                for source_id, target_id in self.valid_shortcuts
                if source_id == self.current_node_id
            ]
            assert (
                valid_targets
            ), "No valid target nodes found for the current source node"
            self.goal_node_id = np.random.choice(valid_targets)

    def _get_atom_index(self, atom_str: str) -> int:
        """Get a unique index for this atom."""
        if atom_str in self.atom_to_index:
            return self.atom_to_index[atom_str]
        assert (
            self._next_index < self.max_preimage_size
        ), "No more space for new atoms. Increase max_preimage_size"
        idx = self._next_index
        self.atom_to_index[atom_str] = idx
        self._next_index += 1
        return idx

    def create_preimage_vector(self, atoms: set[GroundAtom]) -> np.ndarray:
        """Create a multi-hot vector representation of the preimage."""
        vector = np.zeros(self.max_preimage_size, dtype=np.float32)
        for atom in atoms:
            idx = self._get_atom_index(str(atom))
            vector[idx] = 1.0
        return vector

    def _get_current_preimage_vector(self, obs: np.ndarray) -> np.ndarray:
        """Get the multi-hot vector for the current observation's preimage."""
        assert self.perceiver is not None
        atoms = self.perceiver.step(obs)
        return self.create_preimage_vector(atoms)
