"""Complete implementation of RL-based improvisational TAMP approach."""

from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from relational_structs import PDDLProblem
from relational_structs.utils import parse_pddl_plan
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from tomsutils.pddl_planning import run_pddl_planner

from tamp_improv.approaches.base import BaseApproach
from tamp_improv.benchmarks.base import BaseEnvironment

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class RLConfig:
    """Configuration for RL approach."""

    policy_path: str
    train_online: bool = False
    train_timesteps: int = 100_000


class TrainingProgressCallback(BaseCallback):
    """Callback for tracking training progress."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.success_history: list[bool] = []
        self.episode_lengths: list[int] = []
        self.current_length = 0

    def _on_step(self) -> bool:
        """Called after each training step."""
        self.current_length += 1
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        if dones[0]:  # Episode ended
            success = not infos[0].get("TimeLimit.truncated", False)
            self.success_history.append(success)
            self.episode_lengths.append(self.current_length)
            self.current_length = 0

            if len(self.success_history) % self.check_freq == 0:
                recent_successes = self.success_history[-self.check_freq :]
                recent_lengths = self.episode_lengths[-self.check_freq :]
                success_rate = sum(recent_successes) / len(recent_successes)
                avg_length = sum(recent_lengths) / len(recent_lengths)
                print(f"Episodes: {len(self.success_history)}")
                print(f"Success rate: {success_rate:.2%}")
                print(f"Average episode length: {avg_length:.1f}")

        return True


class RLPolicy:
    """RL policy using PPO."""

    def __init__(self, env: Any) -> None:
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
        )

    def train(self, total_timesteps: int, seed: int | None = None) -> None:
        """Train the policy."""
        callback = TrainingProgressCallback()
        if seed is not None:
            self.model.set_random_seed(seed)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path: str) -> None:
        """Save policy."""
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load policy."""
        self.model = PPO.load(path)

    def get_action(self, obs: Any) -> Any:
        """Get action from policy."""
        if isinstance(obs, (int, float)):
            np_obs = np.array([obs])
        else:
            np_obs = np.array(obs)

        action, _ = self.model.predict(np_obs)

        if isinstance(obs, (int, float)):
            return int(action[0])
        return action


class RLApproach(BaseApproach[ObsType, ActType]):
    """RL-based improvisational TAMP approach."""

    def __init__(
        self,
        env: BaseEnvironment[ObsType, ActType],
        seed: int,
        config: RLConfig,
        planner_id: str = "pyperplan",
    ) -> None:
        self._config = config
        self._policy_trained = not config.train_online
        super().__init__(env, seed, planner_id)

    def _create_policy(self) -> RLPolicy:
        """Create RL policy."""
        policy = RLPolicy(self.env.wrapped_env)
        if not self._config.train_online:
            policy.load(self._config.policy_path)
        return policy

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
                if self._config.train_online and not self._policy_trained:
                    print("Training policy online...")
                    self._policy.train(
                        total_timesteps=self._config.train_timesteps, seed=self._seed
                    )
                    self._policy_trained = True

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
