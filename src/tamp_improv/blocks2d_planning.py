"""Planning components for the Blocks2D environment."""

from typing import Sequence, Set, Tuple

import numpy as np
from relational_structs import GroundAtom, LiftedOperator, Object, Predicate, Type
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver

from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv

# Create types and predicates.
robot_type = Type("robot")
block_type = Type("block")
types = {robot_type, block_type}

BlockOnTarget = Predicate("BlockOnTarget", [block_type])
BlockNotOnTarget = Predicate("BlockNotOnTarget", [block_type])
Holding = Predicate("Holding", [robot_type, block_type])
GripperEmpty = Predicate("GripperEmpty", [robot_type])
predicates = {BlockOnTarget, BlockNotOnTarget, Holding, GripperEmpty}

# Create operators.
robot = robot_type("?robot")
block = block_type("?block")

PickUpOperator = LiftedOperator(
    "PickUp",
    [robot, block],
    preconditions={GripperEmpty([robot]), BlockNotOnTarget([block])},
    add_effects={Holding([robot, block])},
    delete_effects={GripperEmpty([robot])}
)

PutDownOperator = LiftedOperator(
    "PutDown",
    [robot, block],
    preconditions={Holding([robot, block]), BlockNotOnTarget([block])},
    add_effects={BlockOnTarget([block]), GripperEmpty([robot])},
    delete_effects={Holding([robot, block]), BlockNotOnTarget([block])}
)

operators = {PickUpOperator, PutDownOperator}

# Create perceiver.
class Blocks2DPerceiver(Perceiver[np.ndarray]):
    def __init__(self, env: Blocks2DEnv):
        self.env = env
        self._robot = robot_type("robot")
        self._block = block_type("block")

    def reset(self, obs: np.ndarray) -> Tuple[Set[Object], Set[GroundAtom], Set[GroundAtom]]:
        objects = {self._robot, self._block}
        atoms = self._get_atoms(obs)
        goal: Set[GroundAtom] = {BlockOnTarget([self._block]), GripperEmpty([self._robot])}
        return objects, atoms, goal

    def step(self, obs: np.ndarray) -> set[GroundAtom]:
        return self._get_atoms(obs)

    def _get_atoms(self, obs: np.ndarray) -> set[GroundAtom]:
        _, _, block_x, block_y, gripper_status = obs
        
        atoms = set()
        
        if self._is_on_target(block_x, block_y):
            atoms.add(BlockOnTarget([self._block]))
        else:
            atoms.add(BlockNotOnTarget([self._block]))

        if gripper_status > 0:
            atoms.add(Holding([self._robot, self._block]))
        else:
            atoms.add(GripperEmpty([self._robot]))

        return atoms

    def _is_on_target(self, x: float, y: float) -> bool:
        return 0.45 < x < 0.55 and y < 0.1

# Create skills.
class PickUpSkill(LiftedOperatorSkill[np.ndarray, np.ndarray]):
    def _get_lifted_operator(self) -> LiftedOperator:
        return PickUpOperator

    def _get_action_given_objects(self, objects: Sequence[Object], obs: np.ndarray) -> np.ndarray:
        robot_x, robot_y, block_x, block_y, _ = obs
        distance = np.linalg.norm([robot_x - block_x, robot_y - block_y])
        
        if distance > 0.3:  # Adjusted threshold considering object sizes
            # Move towards the block
            dx = np.clip(block_x - robot_x, -0.1, 0.1)
            dy = np.clip(block_y - robot_y, -0.1, 0.1)
            return np.array([dx, dy, 0.0])
        else:
            # Pick up the block
            return np.array([0.0, 0.0, 1.0])

class PutDownSkill(LiftedOperatorSkill[np.ndarray, np.ndarray]):
    def _get_lifted_operator(self) -> LiftedOperator:
        return PutDownOperator

    def _get_action_given_objects(self, objects: Sequence[Object], obs: np.ndarray) -> np.ndarray:
        robot_x, robot_y, _, _, gripper_status = obs
        target_x, target_y = 0.5, 0.0  # Target location
        distance = np.linalg.norm([robot_x - target_x, robot_y - target_y])
        
        if distance > 0.0 or robot_y > 0.1:  # Ensure robot is close to target and near the bottom
            # Move towards the target
            dx = np.clip(target_x - robot_x, -0.1, 0.1)
            dy = np.clip(target_y - robot_y, -0.1, 0.1)
            return np.array([dx, dy, 1.0])
        elif gripper_status > 0:
            return np.array([0.0, 0.0, -1.0])
        else:
            return np.array([0.0, 0.0, 0.0])

skills = {PickUpSkill(), PutDownSkill()}