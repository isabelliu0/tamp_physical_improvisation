"""HER wrapper for shortcut policy training."""

from typing import Any, TypeVar

import gymnasium as gym
import numpy as np
from relational_structs import GroundAtom

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class ShortcutHERWrapper(gym.Wrapper):
    """Wrapper that converts ImprovWrapper to goal-conditioned format for
    SAC+HER.

    This wrapper must wrap an ImprovWrapper instance, which provides:
    - current_atom_set: set of current atoms
    - goal_atom_set: set of goal atoms
    - observation_space with shape attribute
    """

    def __init__(
        self,
        env: gym.Env,
        max_atom_size: int = 50,
        success_reward: float = 100.0,
        step_penalty: float = -1.0,
    ):
        super().__init__(env)

        assert hasattr(env, "current_atom_set")
        assert hasattr(env, "goal_atom_set")
        assert hasattr(
            env.observation_space, "shape"
        ), "Observation space must have shape"

        self.max_atom_size = max_atom_size
        self.success_reward = success_reward
        self.step_penalty = step_penalty

        self.atom_to_index: dict[str, int] = {}
        self._next_index = 0

        obs_shape = env.observation_space.shape

        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
                ),
                "achieved_goal": gym.spaces.Box(
                    0, 1, shape=(max_atom_size,), dtype=np.float32
                ),
                "desired_goal": gym.spaces.Box(
                    0, 1, shape=(max_atom_size,), dtype=np.float32
                ),
            }
        )

        self.current_goal_vector: np.ndarray

    def reset(
        self, *, seed: int | None = None, **kwargs: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, **kwargs)

        goal_atoms = self.env.goal_atom_set  # type: ignore[attr-defined]
        self.current_goal_vector = self.create_atom_vector(goal_atoms)

        current_atoms = self.env.current_atom_set  # type: ignore[attr-defined]
        achieved_vector = self.create_atom_vector(current_atoms)

        dict_obs = {
            "observation": self._ensure_array(obs),
            "achieved_goal": achieved_vector,
            "desired_goal": self.current_goal_vector,
        }

        return dict_obs, info

    def step(
        self, action: ActType
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = self.env.step(action)

        current_atoms = self.env.current_atom_set  # type: ignore[attr-defined]
        achieved_vector = self.create_atom_vector(current_atoms)

        goal_indices = np.where(self.current_goal_vector > 0.5)[0]
        goal_achieved = np.all(achieved_vector[goal_indices] > 0.5)
        reward = self.success_reward if goal_achieved else self.step_penalty
        info["is_success"] = goal_achieved

        dict_obs = {
            "observation": self._ensure_array(obs),
            "achieved_goal": achieved_vector,
            "desired_goal": self.current_goal_vector,
        }

        return dict_obs, reward, bool(terminated or goal_achieved), truncated, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        _info: dict[str, Any],
        _indices=None,
    ) -> np.ndarray:
        """Compute reward for achieved and desired goals."""
        if achieved_goal.ndim == 1:
            achieved_goal = achieved_goal.reshape(1, -1)
        if desired_goal.ndim == 1:
            desired_goal = desired_goal.reshape(1, -1)

        rewards = np.zeros(achieved_goal.shape[0], dtype=np.float32)

        for i in range(achieved_goal.shape[0]):
            goal_indices = np.where(desired_goal[i] > 0.5)[0]
            goal_satisfied = np.all(achieved_goal[i][goal_indices] > 0.5)
            rewards[i] = self.success_reward if goal_satisfied else self.step_penalty

        return rewards

    def _ensure_array(self, obs: ObsType) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32)
        return np.array(obs, dtype=np.float32)

    def _get_atom_index(self, atom_str: str) -> int:
        if atom_str in self.atom_to_index:
            return self.atom_to_index[atom_str]
        assert self._next_index < self.max_atom_size, "Need to increase max_atom_size."
        idx = self._next_index
        self.atom_to_index[atom_str] = idx
        self._next_index += 1
        return idx

    def create_atom_vector(self, atoms: set[GroundAtom]) -> np.ndarray:
        """Create binary vector representation of a set of atoms."""
        vector = np.zeros(self.max_atom_size, dtype=np.float32)
        for atom in atoms:
            idx = self._get_atom_index(str(atom))
            vector[idx] = 1.0
        return vector
