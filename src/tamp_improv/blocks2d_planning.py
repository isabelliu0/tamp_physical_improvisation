"""Planning components for the Blocks2D environment."""

from typing import Sequence, Set, Tuple
import numpy as np

from relational_structs import GroundAtom, LiftedAtom, LiftedOperator, Object, Predicate, Type
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver

from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv

# Create types and predicates
robot_type = Type("robot")
block_type = Type("block")
types = {robot_type, block_type}

BlockInTargetArea = Predicate("BlockInTargetArea", [block_type])
BlockNotInTargetArea = Predicate("BlockNotInTargetArea", [block_type])
Holding = Predicate("Holding", [robot_type, block_type])
GripperEmpty = Predicate("GripperEmpty", [robot_type])
TargetAreaClear = Predicate("TargetAreaClear", [])
TargetAreaBlocked = Predicate("TargetAreaBlocked", [])
predicates = {BlockInTargetArea, BlockNotInTargetArea, Holding, GripperEmpty, TargetAreaClear, TargetAreaBlocked}

# Create operators
robot = robot_type("?robot")
block = block_type("?block")

ClearTargetAreaOperator = LiftedOperator(
    "ClearTargetArea",
    [robot],
    preconditions={GripperEmpty([robot]), LiftedAtom(TargetAreaBlocked, [])},
    add_effects={LiftedAtom(TargetAreaClear, [])},
    delete_effects={LiftedAtom(TargetAreaBlocked, [])}
)

PickUpOperator = LiftedOperator(
    "PickUp",
    [robot, block],
    preconditions={GripperEmpty([robot]),
                   LiftedAtom(TargetAreaClear, []),
                   LiftedAtom(BlockNotInTargetArea, [block]),
    },
    add_effects={Holding([robot, block])},
    delete_effects={GripperEmpty([robot])}
)

PutDownOperator = LiftedOperator(
    "PutDown",
    [robot, block],
    preconditions={Holding([robot, block]),
                   LiftedAtom(TargetAreaClear, [])},
    add_effects={LiftedAtom(BlockInTargetArea, [block]), GripperEmpty([robot])},
    delete_effects={Holding([robot, block])}
)

operators = {ClearTargetAreaOperator, PickUpOperator, PutDownOperator}

# Create perceiver
class Blocks2DPerceiver(Perceiver[np.ndarray]):
    def __init__(self, env: 'Blocks2DEnv'):
        self.env = env
        self._robot = robot_type("robot")
        self._block_1 = block_type("block1")
        self._block_2 = block_type("block2")
    
    def reset(self, obs: np.ndarray) -> Tuple[Set[Object], Set[GroundAtom], Set[GroundAtom]]:
        objects = {self._robot, self._block_1, self._block_2}
        atoms = self._get_atoms(obs)
        goal: Set[GroundAtom] = {BlockInTargetArea([self._block_1]), GripperEmpty([self._robot]), GroundAtom(TargetAreaClear, [])}
        return objects, atoms, goal

    def step(self, obs: np.ndarray) -> set[GroundAtom]:
        return self._get_atoms(obs)

    def _get_atoms(self, obs: np.ndarray) -> set[GroundAtom]:
        _, _, _, _, block_1_x, block_1_y, block_2_x, block_2_y, block_width, block_height, gripper_status, target_x, target_y, target_width, target_height = obs
        
        atoms = set()
        
        if self.env.is_block_in_target_area(block_1_x, block_1_y, block_width, block_height, target_x, target_y, target_width, target_height):
            atoms.add(BlockInTargetArea([self._block_1]))
        else:
            atoms.add(BlockNotInTargetArea([self._block_1]))

        if gripper_status > 0:
            atoms.add(Holding([self._robot, self._block_1]))
        else:
            atoms.add(GripperEmpty([self._robot]))
        
        if self._is_target_area_blocked(block_2_x, block_2_y, block_width, block_height, target_x, target_y, target_width, target_height):
            atoms.add(GroundAtom(TargetAreaBlocked, []))
        else:
            atoms.add(GroundAtom(TargetAreaClear, []))

        print("CURRENT ATOMS:", atoms)

        return atoms

    def _is_target_area_blocked(self, block_x: float, block_y: float, block_width: float, block_height: float, target_x: float, target_y: float, target_width: float, target_height: float) -> bool:
        target_left = target_x - target_width / 2
        target_right = target_x + target_width / 2
        target_top = target_y + target_height / 2
        target_bottom = target_y - target_height / 2
        
        block_left = block_x - block_width / 2
        block_right = block_x + block_width / 2
        block_top = block_y + block_height / 2
        block_bottom = block_y - block_height / 2
        
        # Check if there's any overlap
        overlap = (block_left < target_right and block_right > target_left
                   and block_bottom < target_top and block_top > target_bottom)
        
        if overlap:
            # Calculate the remaining free width/height in the target area
            free_width = target_width - max(0, min(block_right, target_right) - max(block_left, target_left))
            free_height = target_height - max(0, min(block_top, target_top) - max(block_bottom, target_bottom))
            # If the free width/height is less than the width/height of block_1, it's blocking
            return (free_width < block_width) or (free_height < block_height)
        
        return False

# Update the skills
class ClearTargetAreaSkill(LiftedOperatorSkill[np.ndarray, np.ndarray]):
    def _get_lifted_operator(self) -> LiftedOperator:
        return ClearTargetAreaOperator

    def _get_action_given_objects(self, objects: Sequence[Object], obs: np.ndarray) -> np.ndarray:
        robot_x, robot_y, robot_width, _, block_1_x, _, block_2_x, block_2_y, block_width, _, _, target_x, _, target_width, _ = obs

        # Determine the best direction to push block 2
        push_direction = self._get_push_direction(block_1_x, block_2_x, target_x, block_width, target_width)
        
        # Calculate distances
        distance = np.linalg.norm([robot_x - block_2_x, robot_y - block_2_y])
        vertical_distance = np.abs(robot_y - block_2_y)
        
        if distance > (robot_width + block_width) / 2:
            dx = np.clip(block_2_x - robot_x, -0.1, 0.1)
            dy = np.clip(block_2_y - robot_y, -0.1, 0.1)
            return np.array([dx, dy, 0.0])
        elif vertical_distance > (robot_width + block_width) / 2: 
            # Move towards the y-level of the block
            dy = np.clip(block_2_y - robot_y, -0.1, 0.1)
            return np.array([0.0, dy, 0.0])
        else:
            # Push the block horizontally
            dx = np.clip(push_direction, -0.1, 0.1)
            return np.array([dx, 0.0, 0.0])
    
    def _get_push_direction(self, block_1_x: float, block_2_x: float, target_x: float, block_width: float, target_width: float) -> float:
        left_margin = (block_2_x - block_width / 2) - (target_x - target_width / 2) + 0.1   # Add a small margin
        right_margin = (target_x + target_width / 2) - (block_2_x + block_width / 2) + 0.1

        if block_1_x < block_2_x:
            left_margin *= 2    # Discourage pushing left if block_1 is on the left
        else:
            right_margin *= 2

        if left_margin < right_margin:
            return -1.0  # Push left
        else:
            return 1.0  # Push right

class PickUpSkill(LiftedOperatorSkill[np.ndarray, np.ndarray]):
    def _get_lifted_operator(self) -> LiftedOperator:
        return PickUpOperator

    def _get_action_given_objects(self, objects: Sequence[Object], obs: np.ndarray) -> np.ndarray:
        robot_x, robot_y, robot_width, _, block_x, block_y, _, _, block_width, _, _, _, _, _, _ = obs
        distance = np.linalg.norm([robot_x - block_x, robot_y - block_y])
        
        if distance > (robot_width + block_width) / 2:
            dx = np.clip(block_x - robot_x, -0.1, 0.1)
            dy = np.clip(block_y - robot_y, -0.1, 0.1)
            return np.array([dx, dy, 0.0])
        else:
            return np.array([0.0, 0.0, 1.0])

class PutDownSkill(LiftedOperatorSkill[np.ndarray, np.ndarray]):
    def _get_lifted_operator(self) -> LiftedOperator:
        return PutDownOperator

    def _get_action_given_objects(self, objects: Sequence[Object], obs: np.ndarray) -> np.ndarray:
        _, _, _, _, block_x, block_y, _, _, _, _, gripper_status, target_x, target_y, _, _ = obs
        distance = np.linalg.norm([target_x - block_x, target_y - block_y])
        
        if distance > 0.1 or block_y > 0.1:  # Ensure robot is close to target and near the bottom
            # Move towards the target
            dx = np.clip(target_x - block_x, -0.1, 0.1)
            dy = np.clip(target_y - block_y, -0.1, 0.1)
            assert dx > 0
            assert np.isclose(dy, 0.0)
            return np.array([dx, dy, 1.0])
        elif gripper_status > 0:
            return np.array([0.0, 0.0, -1.0])
        else:
            return np.array([0.0, 0.0, 0.0])

skills = {ClearTargetAreaSkill(), PickUpSkill(), PutDownSkill()}