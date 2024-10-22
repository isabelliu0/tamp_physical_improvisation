"""Task then motion planning approach."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from task_then_motion_planning.planning import (
    TaskThenMotionPlanner,
)

from tamp_improv.approaches.base_approach import BaseApproach
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.blocks2d_planning import (
    Blocks2DPerceiver,
    operators,
    predicates,
    skills,
    types,
)


class TaskThenMotionPlanningApproach(
    BaseApproach[NDArray[np.float32], NDArray[np.float32]]
):
    """Task then motion planning approach."""

    def __init__(self, observation_space, action_space, seed: int) -> None:
        super().__init__(observation_space, action_space, seed)
        self.env_name = "blocks2d"
        self.perceiver = Blocks2DPerceiver(Blocks2DEnv())
        self.planner = TaskThenMotionPlanner(
            types,
            predicates,
            self.perceiver,
            operators,
            set(skills),  # Convert skills to a set of Skill objects
            planner_id="pyperplan",
        )

    def reset(
        self, obs: NDArray[np.float32], info: dict[str, Any]
    ) -> NDArray[np.float32]:
        try:
            self.planner.reset(obs, info)
        except Exception as e:
            print("Error during planner reset:", str(e))
            print("Current problem:")
            print(self.planner._current_problem)  # pylint: disable=protected-access
            print("Current domain:")
            print(self.planner._domain)  # pylint: disable=protected-access
            raise

        init = self.step(
            obs, 0.0, False, False, info
        )  # placeholder values before the first env step
        return init

    def step(
        self,
        obs: NDArray[np.float32],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> NDArray[np.float32]:
        action = self.planner.step(obs)
        return np.array(action, dtype=np.float32)
