"""Complete implementation of MPC-based improvisational TAMP approach."""

from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from gymnasium import spaces
from relational_structs import PDDLProblem
from relational_structs.utils import parse_pddl_plan
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from tomsutils.pddl_planning import run_pddl_planner

from tamp_improv.approaches.base import BaseApproach
from tamp_improv.benchmarks.base import BaseEnvironment

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class MPCConfig:
    """Configuration for MPC approach."""

    num_rollouts: int = 100
    noise_scale: float = 1.0
    num_control_points: int = 5
    horizon: int = 35


class MPCPolicy:
    """MPC policy using predictive sampling."""

    def __init__(self, env: Any, seed: int, config: MPCConfig | None = None) -> None:
        self._env = env
        self._config = config or MPCConfig()
        self._rng = np.random.default_rng(seed)
        self._is_discrete = isinstance(env.action_space, spaces.Discrete)

        # Initialize last solution
        if self._is_discrete:
            self._last_solution = np.zeros(self._config.horizon, dtype=np.int32)
        else:
            action_shape = env.action_space.shape
            shape = (self._config.horizon,) + tuple(
                () if action_shape is None else action_shape
            )
            self._last_solution = np.zeros(shape, dtype=np.float32)
            self._control_times = np.linspace(
                0, self._config.horizon - 1, self._config.num_control_points
            )
            self._trajectory_times = np.arange(self._config.horizon)

    def get_action(self, obs: Any) -> Any:
        """Get action using predictive sampling."""
        return self._solve(obs)

    def _solve(self, init_obs: Any) -> Any:
        """Run one iteration of predictive sampling."""
        sample_list = []

        # Initialize or warm start
        if np.all(self._last_solution == 0):
            nominal = self._get_initialization()
        else:
            nominal = np.vstack((self._last_solution[1:], self._last_solution[-1:]))
        sample_list.append(nominal)

        # Sample and evaluate trajectories
        samples = self._sample_from_nominal(nominal)
        sample_list.extend(samples)
        scores = [self._score_sample(sample, init_obs) for sample in sample_list]

        # Pick best trajectory
        best_idx = np.argmax(scores)
        self._last_solution = sample_list[best_idx]

        if self._is_discrete:
            return int(self._last_solution[0])
        return self._last_solution[0]

    def _get_initialization(self) -> np.ndarray:
        """Initialize trajectory."""
        if self._is_discrete:
            return self._rng.choice([0, 1], size=self._config.horizon, p=[0.5, 0.5])

        # For continuous actions
        shape = (self._config.num_control_points,) + tuple(
            self._env.action_space.shape or ()
        )
        control_points = self._rng.standard_normal(shape)
        trajectory = np.zeros(
            (self._config.horizon,) + tuple(self._env.action_space.shape or ())
        )

        for dim in range(
            control_points.shape[-1] if len(control_points.shape) > 1 else 1
        ):
            trajectory_idx = (..., dim) if len(control_points.shape) > 1 else ...
            control_idx = (..., dim) if len(control_points.shape) > 1 else ...
            trajectory[trajectory_idx] = np.interp(
                self._trajectory_times,
                self._control_times,
                control_points[control_idx],
            )

        return np.clip(
            trajectory, self._env.action_space.low, self._env.action_space.high
        )

    def _sample_from_nominal(self, nominal: np.ndarray) -> list[np.ndarray]:
        """Sample new trajectories around nominal one."""
        if self._is_discrete:
            trajectories = []
            for _ in range(self._config.num_rollouts - 1):
                flip_mask = (
                    self._rng.random(size=self._config.horizon)
                    < self._config.noise_scale
                )
                new_traj = nominal.copy()
                new_traj[flip_mask] = 1 - new_traj[flip_mask]
                trajectories.append(new_traj)
            return trajectories

        # For continuous actions
        noise = self._rng.normal(
            loc=0,
            scale=self._config.noise_scale,
            size=(
                self._config.num_rollouts - 1,
                self._config.num_control_points,
            )
            + tuple(self._env.action_space.shape or ()),
        )

        nominal_control_points = np.array(
            [nominal[int(t)] for t in self._control_times]
        )
        new_control_points = nominal_control_points + noise

        trajectories = []
        for points in new_control_points:
            trajectory = np.zeros(
                (self._config.horizon,) + tuple(self._env.action_space.shape or ())
            )
            for dim in range(points.shape[-1] if len(points.shape) > 1 else 1):
                idx = (..., dim) if len(points.shape) > 1 else ...
                trajectory[idx] = np.interp(
                    self._trajectory_times,
                    self._control_times,
                    points[idx],
                )
            trajectory = np.clip(
                trajectory, self._env.action_space.low, self._env.action_space.high
            )
            trajectories.append(trajectory)
        return trajectories

    def _score_sample(self, trajectory: np.ndarray, init_obs: Any) -> float:
        """Evaluate a trajectory by rolling out in the environment."""
        obs = init_obs
        total_reward = 0.0

        self._env.reset(options={"initial_obs": obs})

        for action in trajectory:
            if self._is_discrete:
                action = int(action)
            obs, reward, terminated, truncated, _ = self._env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return total_reward


class MPCApproach(BaseApproach[ObsType, ActType]):
    """MPC-based improvisational TAMP approach."""

    def __init__(
        self,
        env: BaseEnvironment[ObsType, ActType],
        seed: int,
        config: MPCConfig | None = None,
        planner_id: str = "pyperplan",
    ) -> None:
        self._config = config or MPCConfig()
        super().__init__(env, seed, planner_id)

    def _create_policy(self) -> MPCPolicy:
        """Create MPC policy."""
        return MPCPolicy(self.env.wrapped_env, self._seed, self._config)

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ActType:
        """Reset approach."""
        assert self._perceiver is not None

        # Get initial atoms and goal
        objects, atoms, goal = self._perceiver.reset(obs, info)
        self._goal = goal

        # Create initial problem and plan
        self._current_problem = PDDLProblem(
            self._domain.name, self._domain.name, objects, atoms, goal
        )
        plan_str = run_pddl_planner(
            str(self._domain), str(self._current_problem), planner=self._planner_id
        )
        assert plan_str is not None
        self._current_task_plan = parse_pddl_plan(
            plan_str, self._domain, self._current_problem
        )

        # Reset other variables
        self._current_operator = None
        self._current_skill = None
        self._policy_active = False
        self._target_atoms = set()

        return self.step(obs, 0.0, False, False, info)

    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ActType:
        """Step approach."""
        assert self._perceiver is not None
        atoms = self._perceiver.step(obs)

        # Check if policy achieved target atoms
        if self._policy_active:
            if self._target_atoms.issubset(atoms):
                print("Policy achieved target atoms")
                self._policy_active = False
                self._target_atoms = set()
                self._replan(obs, info)
                return self.step(obs, reward, terminated, truncated, info)
            return self._policy.get_action(obs)

        # Check if current operator completed
        if self._current_operator is None or (
            self._current_operator.add_effects.issubset(atoms)
            and not (self._current_operator.delete_effects & atoms)
        ):
            # Get next operator from plan
            if not self._current_task_plan:
                raise TaskThenMotionPlanningFailure("Empty task plan")

            self._current_operator = self._current_task_plan.pop(0)

            # Get operator preconditions
            full_operator = next(
                op
                for op in self._domain.operators
                if op.name == self._current_operator.name
            )
            full_ground_operator = full_operator.ground(
                self._current_operator.parameters
            )

            # If preconditions not met, activate policy
            if not full_ground_operator.preconditions.issubset(atoms):
                print("Activating policy to achieve preconditions")
                self._policy_active = True
                self._target_atoms = full_ground_operator.preconditions
                return self._policy.get_action(obs)

            # Get skill for operator
            applicable_skills = [
                s for s in self._skills if s.can_execute(self._current_operator)
            ]
            if not applicable_skills:
                raise TaskThenMotionPlanningFailure(
                    f"No skill for operator {self._current_operator}"
                )
            self._current_skill = applicable_skills[0]
            self._current_skill.reset(self._current_operator)

        assert self._current_skill is not None
        return self._current_skill.get_action(obs)

    def _replan(self, obs: ObsType, info: dict[str, Any]) -> None:
        """Replan from current state."""
        assert self._perceiver is not None

        # Get current state
        objects, atoms, _ = self._perceiver.reset(obs, info)

        # Create new problem and plan
        self._current_problem = PDDLProblem(
            self._domain.name, self._domain.name, objects, atoms, self._goal
        )
        plan_str = run_pddl_planner(
            str(self._domain), str(self._current_problem), planner=self._planner_id
        )
        assert plan_str is not None
        self._current_task_plan = parse_pddl_plan(
            plan_str, self._domain, self._current_problem
        )

        # Reset operator and skill
        self._current_operator = None
        self._current_skill = None
