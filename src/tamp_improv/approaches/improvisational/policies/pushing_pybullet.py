"""Hard-coded pushing policy for PyBullet ClearAndPlace environment."""

from typing import Callable

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from pybullet_blocks.envs.clear_and_place_env import (
    ClearAndPlacePyBulletBlocksEnv,
    ClearAndPlacePyBulletBlocksState,
)
from pybullet_blocks.planning_models.perception import (
    Holding,
    NothingOn,
    On,
    object_type,
    robot_type,
)
from pybullet_helpers.geometry import Pose, iter_between_poses, multiply_poses
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from relational_structs import GroundAtom, Object

from tamp_improv.approaches.improvisational.policies.base import (
    Policy,
    PolicyContext,
    TrainingData,
)


class PybulletPushingPolicy(Policy[NDArray[np.float32], NDArray[np.float32]]):
    """Hard-coded policy for pushing obstacle blocks away from target area in
    Pybullet ClearAndPlace environment."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self._env: ClearAndPlacePyBulletBlocksEnv | None = None
        self._wrapped_env: gym.Env | None = None
        self._current_atoms: set[GroundAtom] | None = None
        self._target_preimage: set[GroundAtom] | None = None

        # Push state tracking
        self._state: ClearAndPlacePyBulletBlocksState | None = None
        self._joint_distance_fn: Callable[[list[float], list[float]], float] | None = (
            None
        )
        self._plan: list[NDArray[np.float32]] = []
        self._current_step = 0
        self._push_stage = 0  # 0: init, 1: approach, 2: push, 3: retreat, 4: done
        self._max_motion_planning_time = 1.0  # seconds

    @property
    def requires_training(self) -> bool:
        return False

    def initialize(self, env: gym.Env) -> None:
        self._wrapped_env = env
        base_env = env
        while hasattr(base_env, "env") and not isinstance(
            base_env, ClearAndPlacePyBulletBlocksEnv
        ):
            base_env = base_env.env
        assert isinstance(base_env, ClearAndPlacePyBulletBlocksEnv)
        self._env = base_env
        self._joint_distance_fn = create_joint_distance_fn(self._env.robot)
        self._plan = []
        self._current_step = 0
        self._push_stage = 0

    def can_initiate(self) -> bool:
        """Check if conditions are right for pushing blocks out of the target
        area."""
        assert self._current_atoms is not None and self._target_preimage is not None
        robot = Object("robot", robot_type)
        target_block = Object("T", object_type)
        obstacle_a = Object("A", object_type)
        obstacle_b = Object("B", object_type)
        obstacle_c = Object("C", object_type)
        target_area = Object("target", object_type)
        table = Object("table", object_type)
        current_conditions = {
            GroundAtom(On, [obstacle_a, target_area]),
            GroundAtom(On, [obstacle_b, obstacle_a]),
            GroundAtom(On, [obstacle_c, obstacle_b]),
            GroundAtom(On, [target_block, table]),
        }
        preimage_conditions = {
            GroundAtom(Holding, [robot, target_block]),
            GroundAtom(NothingOn, [target_area]),
        }
        return current_conditions.issubset(
            self._current_atoms
        ) and preimage_conditions.issubset(self._target_preimage)

    def configure_context(self, context: PolicyContext) -> None:
        self._current_atoms = context.current_atoms
        self._target_preimage = context.preimage

    def get_action(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get action for pushing obstacle blocks out of the target area."""
        # Parse observation to state
        self._state = ClearAndPlacePyBulletBlocksState.from_observation(obs)

        # If we're done with the current plan, generate the next one based on push stage
        if not self._plan:
            self._generate_next_plan()

        if self._current_step < len(self._plan):
            action = self._plan[self._current_step]
            self._current_step += 1
            if self._current_step >= len(self._plan):
                self._plan = []
                self._current_step = 0
                self._push_stage += 1
            return action

        # No-op action if we have no plan
        return np.zeros(8, dtype=np.float32)

    def _generate_next_plan(self) -> None:
        """Generate the next plan based on current push stage."""
        if self._env is None or self._state is None or self._joint_distance_fn is None:
            return

        sim = self._env
        state = self._state
        sim.set_state(state)

        # Get initial robot end effector orientation
        init_ee_orn = sim.robot.get_end_effector_pose().orientation

        # Set pushing orientation (tilted down)
        push_ee_orn = multiply_poses(
            Pose((0, 0, 0), init_ee_orn),
            # Tuned hyperparameter angle for pushing the entire stack
            Pose.from_rpy((0, 0, 0), (0.0, -np.pi * (5 / 16), 0.0)),
        ).orientation

        # Find the bottom obstacle block
        obstacle_blocks = list(state.obstacle_block_states)
        bottom_block = obstacle_blocks[0]

        if self._push_stage == 0:  # Move to pushing position
            push_offset = (0.0, 0.075, -0.01)  # Slightly behind and below block
            next_to_block_position = np.add(bottom_block.pose.position, push_offset)
            next_to_block_pose = Pose(tuple(next_to_block_position), push_ee_orn)
            plan = run_smooth_motion_planning_to_pose(
                next_to_block_pose,
                sim.robot,
                collision_ids=sim.get_collision_ids(),
                end_effector_frame_to_plan_frame=Pose.identity(),
                seed=123,
                max_time=self._max_motion_planning_time,
            )
            assert plan is not None
            plan_arrays = [np.array(p, dtype=np.float32) for p in plan]
            self._plan = self._convert_plan_to_actions(plan_arrays, state)

        elif self._push_stage == 1:  # Push block away from target
            push_distance = (0.0, -0.125, -0.01)
            push_target_position = np.add(bottom_block.pose.position, push_distance)
            push_target_pose = Pose(tuple(push_target_position), push_ee_orn)
            end_effector_path = list(
                iter_between_poses(
                    sim.robot.get_end_effector_pose(),
                    push_target_pose,
                    include_start=False,
                    num_interp=100,  # slow movement for stable pushing
                )
            )
            push_plan = smoothly_follow_end_effector_path(
                sim.robot,
                end_effector_path,
                state.robot_state.joint_positions,
                {sim.table_id, sim.target_area_id},
                self._joint_distance_fn,
                max_time=self._max_motion_planning_time,
                include_start=False,
            )
            assert push_plan is not None
            push_plan_arrays = [np.array(p, dtype=np.float32) for p in push_plan]
            self._plan = self._convert_plan_to_actions(push_plan_arrays, state)

        elif self._push_stage == 2:  # Move up after push
            push_target_position = np.add(
                bottom_block.pose.position, (0.0, -0.125, -0.01)
            )  # Same as previous stage
            retreat_offset = (0.0, 0.0, 0.1)
            retreat_position = np.add(push_target_position, retreat_offset)
            retreat_pose = Pose(tuple(retreat_position), init_ee_orn)
            end_effector_path = list(
                iter_between_poses(
                    sim.robot.get_end_effector_pose(),
                    retreat_pose,
                    include_start=False,
                    num_interp=25,
                )
            )
            post_push_plan = smoothly_follow_end_effector_path(
                sim.robot,
                end_effector_path,
                state.robot_state.joint_positions,
                {sim.table_id, sim.target_area_id},
                self._joint_distance_fn,
                max_time=self._max_motion_planning_time,
                include_start=False,
            )
            assert post_push_plan is not None
            post_push_plan_arrays = [
                np.array(p, dtype=np.float32) for p in post_push_plan
            ]
            self._plan = self._convert_plan_to_actions(post_push_plan_arrays, state)

        else:  # Push completed, reset stage
            self._push_stage = 0

    def _convert_plan_to_actions(
        self, plan: list[NDArray[np.float32]], state: ClearAndPlacePyBulletBlocksState
    ) -> list[NDArray[np.float32]]:
        """Convert a joint position plan to a sequence of actions."""
        assert plan is not None and self._env is not None
        plan_as_lists = [p.tolist() for p in plan]
        remapped_plan = remap_joint_position_plan_to_constant_distance(
            plan_as_lists, self._env.robot
        )
        actions = []
        current_joints = state.robot_state.joint_positions
        for joint_state in remapped_plan:
            joint_delta = np.subtract(joint_state, current_joints)
            action = np.hstack([joint_delta[:7], [0.0]]).astype(np.float32)
            assert self._env.action_space.contains(action)
            actions.append(action)
            current_joints = joint_state
        return actions

    def train(self, env: gym.Env, train_data: TrainingData) -> None:
        """No training needed for hard-coded policy."""

    def save(self, path: str) -> None:
        """Save policy parameters."""

    def load(self, path: str) -> None:
        """Load policy parameters."""
