"""RL-based implementation of improvisational TAMP for Blocks2D environment."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from relational_structs import (
    GroundOperator,
    PDDLDomain,
)
from task_then_motion_planning.planning import TaskThenMotionPlanningFailure
from task_then_motion_planning.structs import Perceiver, Skill, _Action, _Observation

from tamp_improv.approaches.base_improvisational_tamp_approach import (
    ImprovisationalTAMPApproach,
)
from tamp_improv.approaches.rl_improvisational_policy import RLImprovisationalPolicy
from tamp_improv.benchmarks.blocks2d_env_old import Blocks2DEnv
from tamp_improv.benchmarks.blocks2d_env_wrapper_old import make_pushing_env
from tamp_improv.blocks2d_planning import create_blocks2d_planning_models


class RLBlocks2DImprovisationalTAMPApproach(ImprovisationalTAMPApproach):
    """Blocks2D improvisational TAMP approach using learned RL policy.

    This approach uses a trained RL policy to handle situations where
    preconditions aren't met (specifically when the target area is
    blocked), and a TAMP planner for high-level task planning. The RL
    policy is trained separately to learn pushing behaviors.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        seed: int,
        policy_path: str,
        train_online: bool = False,
        train_timesteps: int = 100_000,
        planner_id: str = "pyperplan",
        domain_name: str = "custom-domain",
    ) -> None:
        """Initialize the approach.

        Args:
            observation_space: Observation space from the environment
            action_space: Action space from the environment
            seed: Random seed
            policy_path: Path to saved RL policy
            train_online: Whether to train the policy online
            train_timesteps: Number of timesteps for online training
            planner_id: ID of PDDL planner to use
            domain_name: Name of PDDL domain
        """
        # Create base env and wrapped env for the RL policy
        base_env = Blocks2DEnv()
        pushing_env = make_pushing_env(base_env, seed=seed)

        # Initialize RL policy and load weights
        policy: RLImprovisationalPolicy[NDArray[np.float32], NDArray[np.float32]] = (
            RLImprovisationalPolicy(pushing_env)
        )

        # Store training parameters
        self._train_online = train_online
        self._train_timesteps = train_timesteps
        self._policy_path = policy_path
        self._policy_trained = False

        # If not training online, load existing policy
        if not train_online:
            policy.load(policy_path)
            self._policy_trained = True

        super().__init__(
            observation_space, action_space, seed, policy, planner_id, domain_name
        )

        self.env_name = "blocks2d"

        # Initialize Blocks2D-specific planning components
        types, predicates, _, operators, skills = create_blocks2d_planning_models(
            include_pushing_models=False
        )
        _, _, perceiver, full_precondition_operators, _ = (
            create_blocks2d_planning_models(include_pushing_models=True)
        )

        self._types = types
        self._predicates = predicates
        self._operators = operators
        self._skills = skills
        self._domain_name = domain_name
        self._domain = PDDLDomain(domain_name, operators, predicates, types)
        self._perceiver: Perceiver[NDArray[np.float32]] = perceiver

        # Set up operator mapping
        self._operator_to_full_precondition_operator = {}
        operator_name_to_operator = {o.name: o for o in operators}
        for full_precondition_operator in full_precondition_operators:
            if full_precondition_operator.name not in operator_name_to_operator:
                continue
            operator = operator_name_to_operator[full_precondition_operator.name]
            self._operator_to_full_precondition_operator[operator] = (
                full_precondition_operator
            )

    def _get_skill_for_operator(
        self, operator: GroundOperator
    ) -> Skill[_Observation, _Action]:
        """Get the appropriate skill for executing an operator.

        Args:
            operator: The ground operator to execute

        Returns:
            Skill that can execute the operator

        Raises:
            TaskThenMotionPlanningFailure: If no skill can execute the operator
        """
        applicable_skills = [s for s in self._skills if s.can_execute(operator)]
        if not applicable_skills:
            raise TaskThenMotionPlanningFailure("No skill can execute operator")
        assert len(applicable_skills) == 1, "Multiple operators per skill not supported"
        return applicable_skills[0]

    def _train_policy(self) -> None:
        """Train the policy if using online training."""
        if self._train_online and not self._policy_trained:
            print("\nStarting online policy training...")
            print(f"Training for {self._train_timesteps} timesteps")
            self._policy.train(total_timesteps=self._train_timesteps)
            self._policy_trained = True
            print("Policy training completed")

    def save_policy(self, path: str | None = None) -> None:
        """Save the trained policy.

        Args:
            path: Path to save policy (defaults to self._policy_path)
        """
        save_path = path or self._policy_path
        if self._policy_trained:
            self._policy.save(save_path)

    def step(
        self,
        obs: NDArray[np.float32],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> NDArray[np.float32]:
        """Step function that updates preconditions in the pushing env when
        policy is activated."""
        assert self._perceiver is not None
        atoms = self._perceiver.step(obs)

        # If the policy is already active, check if goal achieved
        if self._policy_active:
            if self._target_atoms.issubset(atoms):
                self._policy_active = False
                self._target_atoms = set()
                self._replan(obs, info)
                return self.step(obs, reward, terminated, truncated, info)
            return self._policy.get_action(obs)

        # If we need a new operator
        if self._current_operator is None or (
            self._current_operator.add_effects.issubset(atoms)
            and not (self._current_operator.delete_effects & atoms)
        ):
            if not self._current_task_plan:
                raise TaskThenMotionPlanningFailure("Empty task plan")

            self._current_operator = self._current_task_plan.pop(0)
            full_lifted_operator = self._operator_to_full_precondition_operator[
                self._current_operator.parent
            ]
            full_ground_operator = full_lifted_operator.ground(
                tuple(self._current_operator.parameters)
            )

            # If preconditions not met, activate policy
            if not full_ground_operator.preconditions.issubset(atoms):
                print("Preconditions not met, activating policy")
                self._train_policy()  # Train policy if online training
                self._policy_active = True

                # Set all preconditions as target (for tracking when to stop policy)
                self._target_atoms = full_ground_operator.preconditions

                # Get currently satisfied preconditions
                currently_satisfied = full_ground_operator.preconditions & atoms

                # Update pushing env to only maintain currently satisfied preconditions
                if hasattr(self._policy.env, "update_preconditions"):
                    self._policy.env.update_preconditions(
                        full_ground_operator.parent,
                        currently_satisfied,  # Only pass already satisfied preconditions
                    )
                return self._policy.get_action(obs)

            # Get skill if preconditions are met
            self._current_skill = self._get_skill_for_operator(self._current_operator)
            self._current_skill.reset(self._current_operator)

        assert self._current_skill is not None
        return self._current_skill.get_action(obs)
