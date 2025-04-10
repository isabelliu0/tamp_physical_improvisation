"""Hard-coded pushing policy for PyBullet ClearAndPlace environment."""

import copy
from typing import Callable

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from pybullet_blocks.envs.clear_and_place_env import (
    ClearAndPlacePyBulletBlocksEnv,
    ClearAndPlacePyBulletBlocksState,
)
from pybullet_blocks.planning_models.perception import (
    GripperEmpty,
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
        self._planning_env: ClearAndPlacePyBulletBlocksEnv | None = None
        self._current_atoms: set[GroundAtom] | None = None
        self._target_preimage: set[GroundAtom] | None = None
        self._context: PolicyContext | None = None

        # Push state tracking
        self._state: ClearAndPlacePyBulletBlocksState | None = None
        self._joint_distance_fn: Callable[[list[float], list[float]], float] | None = (
            None
        )
        self._plan: list[NDArray[np.float32]] = []
        self._current_step = 0
        self._push_stage = 0  # 0: init, 1: approach, 2: push, 3: retreat, 4: done
        self._max_motion_planning_time = 1.0  # seconds

        # Store consistent orientations
        self._init_ee_orn: tuple[float, float, float, float] | None = None
        self._push_ee_orn: tuple[float, float, float, float] | None = None

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
        self._plan = []
        self._current_step = 0
        self._push_stage = 0
        self._init_ee_orn = None
        self._push_ee_orn = None

    def _create_planning_env(self, state: ClearAndPlacePyBulletBlocksState) -> None:
        """Create a separate environment instance for motion planning."""
        assert self._env is not None
        if hasattr(self._env, "clone"):
            self._planning_env = self._env.clone()
        else:
            self._planning_env = copy.deepcopy(self._env)
        self._planning_env.set_state(state)
        self._joint_distance_fn = create_joint_distance_fn(self._planning_env.robot)
        if self._init_ee_orn is None:
            self._init_ee_orn = (
                self._planning_env.robot.get_end_effector_pose().orientation
            )
            self._push_ee_orn = multiply_poses(
                Pose((0, 0, 0), self._init_ee_orn),
                Pose.from_rpy((0, 0, 0), (0.0, -np.pi * (5 / 16), 0.0)),
            ).orientation

    def can_initiate(self) -> bool:
        """Check if conditions are right for pushing blocks out of the target
        area."""
        assert self._current_atoms is not None and self._target_preimage is not None
        source_node_id = None
        target_node_id = None
        if hasattr(self, "_context") and self._context and self._context.info:
            source_node_id = self._context.info.get("source_node_id")
            target_node_id = self._context.info.get("target_node_id")
        specific_transition = source_node_id == 0 and target_node_id == 50

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
            GroundAtom(GripperEmpty, [robot]),
            GroundAtom(NothingOn, [target_area]),
            GroundAtom(NothingOn, [target_block]),
            GroundAtom(On, [target_block, table]),
        }
        atoms_conditions_met = current_conditions.issubset(
            self._current_atoms
        ) and preimage_conditions.issubset(self._target_preimage)
        return specific_transition and atoms_conditions_met

    def configure_context(self, context: PolicyContext) -> None:
        self._current_atoms = context.current_atoms
        self._target_preimage = context.preimage
        self._context = context

    def get_action(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get action for pushing obstacle blocks out of the target area."""
        self._state = ClearAndPlacePyBulletBlocksState.from_observation(obs)

        # If we're done with the current plan, generate the next one based on push stage
        if not self._plan:
            self._create_planning_env(self._state)
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
        assert (
            self._planning_env is not None
            and self._state is not None
            and self._joint_distance_fn is not None
        )
        assert self._init_ee_orn is not None and self._push_ee_orn is not None

        # Use planning environment to generate motion plans
        sim = self._planning_env
        state = self._state

        # Find the bottom obstacle block
        obstacle_blocks = list(state.obstacle_block_states)
        bottom_block = obstacle_blocks[0]

        if self._push_stage == 0:  # Move to pushing position
            push_offset = (0.0, 0.075, -0.01)  # Slightly behind and below block
            next_to_block_position = np.add(bottom_block.pose.position, push_offset)
            next_to_block_pose = Pose(tuple(next_to_block_position), self._push_ee_orn)
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
            push_target_pose = Pose(tuple(push_target_position), self._push_ee_orn)
            end_effector_path = list(
                iter_between_poses(
                    sim.robot.get_end_effector_pose(),
                    push_target_pose,
                    include_start=False,
                    num_interp=150,  # slow movement for stable pushing
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
            retreat_pose = Pose(tuple(retreat_position), self._init_ee_orn)
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
        assert plan is not None and self._planning_env is not None
        plan_as_lists = [p.tolist() for p in plan]
        remapped_plan = remap_joint_position_plan_to_constant_distance(
            plan_as_lists, self._planning_env.robot
        )
        actions = []
        current_joints = state.robot_state.joint_positions
        for joint_state in remapped_plan:
            joint_delta = np.subtract(joint_state, current_joints)
            action = np.hstack([joint_delta[:7], [0.0]]).astype(np.float32)
            assert self._planning_env.action_space.contains(action)
            actions.append(action)
            current_joints = joint_state
        return actions

    def train(self, env: gym.Env, train_data: TrainingData | None) -> None:
        """No training needed for hard-coded policy."""

    def save(self, path: str) -> None:
        """Save policy parameters."""

    def load(self, path: str) -> None:
        """Load policy parameters."""
