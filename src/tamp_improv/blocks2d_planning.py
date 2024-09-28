"""Planning components for the Blocks2D environment."""

from typing import Sequence

import abc
from typing import Sequence

import numpy as np
from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedOperator,
    Object,
    Predicate,
    Type,
)
from task_then_motion_planning.structs import Perceiver, LiftedOperatorSkill

from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv

# Create types.
robot_type = Type("robot")
block_type = Type("block")
location_type = Type("location")
types = {robot_type, block_type, location_type}

# Create predicates.
RobotAt = Predicate("RobotAt", [robot_type, location_type, location_type])
BlockAt = Predicate("BlockAt", [block_type, location_type, location_type])
Holding = Predicate("Holding", [robot_type, block_type])
GripperEmpty = Predicate("GripperEmpty", [robot_type])
GripperActive = Predicate("GripperActive", [robot_type])
predicates = {RobotAt, BlockAt, Holding, GripperEmpty, GripperActive}

# Create operators.
PickUpOperator = LiftedOperator(
    "PickUp",
    [robot_type("?robot"), block_type("?block"), location_type("?location")],
    preconditions={
        RobotAt(robot_type("?robot"), location_type("?location")),
        BlockAt(block_type("?block"), location_type("?location")),
        GripperEmpty(robot_type("?robot"))
    },
    add_effects={
        Holding(robot_type("?robot"), block_type("?block")),
        GripperActive(robot_type("?robot"))
    },
    delete_effects={
        BlockAt(block_type("?block"), location_type("?location")),
        GripperEmpty(robot_type("?robot"))
    }
)

PutDownOperator = LiftedOperator(
    "PutDown",
    [robot_type("?robot"), block_type("?block"), location_type("?location")],
    preconditions={
        RobotAt(robot_type("?robot"), location_type("?location")),
        Holding(robot_type("?robot"), block_type("?block")),
        GripperActive(robot_type("?robot"))
    },
    add_effects={
        BlockAt(block_type("?block"), location_type("?location")),
        GripperEmpty(robot_type("?robot"))
    },
    delete_effects={
        Holding(robot_type("?robot"), block_type("?block")),
        GripperActive(robot_type("?robot"))
    }
)

MoveOperator = LiftedOperator(
    "Move",
    [robot_type("?robot"), location_type("?from"), location_type("?to")],
    preconditions={
        RobotAt(robot_type("?robot"), location_type("?from"))
    },
    add_effects={
        RobotAt(robot_type("?robot"), location_type("?to"))
    },
    delete_effects={
        RobotAt(robot_type("?robot"), location_type("?from"))
    }
)
operators = {MoveOperator, PickUpOperator, PutDownOperator}

# Create perceiver.
class Blocks2DPerceiver(Perceiver[np.ndarray]):
    def __init__(self, env: Blocks2DEnv):
        self.env = env
        self._robot = robot_type("robot")
        self._block = block_type("block")
        self._locations = {
            (0.0, 0.0): location_type("loc-0-0"),
            (0.5, 0.0): location_type("loc-0.5-0"),
            (1.0, 0.0): location_type("loc-1-0"),
            (0.0, 0.5): location_type("loc-0-0/5"),
            (0.5, 0.5): location_type("loc-0.5-0.5"),
            (1.0, 0.5): location_type("loc-1-0.5"),
            (0.0, 1.0): location_type("loc-0-1"),
            (0.5, 1.0): location_type("loc-0.5-1"),
            (1.0, 1.0): location_type("loc-1-1"),
        }

    def reset(self, obs: np.ndarray) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        objects = {self._robot, self._block} | set(self._locations.values())
        atoms = self._get_atoms(obs)
        goal = {BlockAt(self._block, self._locations[("0.5, 0.0")]), GripperEmpty(self._robot)}  # For now the target location is fixed.
        return objects, atoms, goal

    def step(self, obs: np.ndarray) -> set[GroundAtom]:
        return self._get_atoms(obs)

    def _get_atoms(self, obs: np.ndarray) -> set[GroundAtom]:
        # Unpack the observation.
        robot_x, robot_y, block_x, block_y, gripper_status = obs
        
        # Create current atoms.
        atoms = set()
        robot_loc = self._get_nearest_location(robot_x, robot_y)
        atoms.add(RobotAt(self._robot, robot_loc))

        if gripper_status > 0:
            atoms.add(Holding(self._robot, self._block))
        else:
            block_loc = self._get_nearest_location(block_x, block_y)
            atoms.add(BlockAt(self._block, block_loc))
            atoms.add(GripperEmpty(self._robot))

        return atoms
    
    def _get_nearest_location(self, x: float, y: float) -> Object:
        return min(self._locations.items(), key=lambda loc: ((loc[0][0]-x)**2 + (loc[0][1]-y)**2))[1]

# Create skills.
class Blocks2DSkill(LiftedOperatorSkill[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        super().__init__()
        self._action_queue: list[np.ndarray] = []

    def reset(self, ground_operator: GroundOperator) -> None:
        self._action_queue = []
        return super().reset(ground_operator)

    def _get_action_given_objects(self, objects: Sequence[Object], obs: np.ndarray) -> np.ndarray:
        if self._action_queue:
            return self._action_queue.pop(0)

        robot_x, robot_y = obs[:2]
        destination = objects[-1]
        dest_x, dest_y = map(float, destination.name.split("-")[1:])

        dx = np.clip(dest_x - robot_x, -0.1, 0.1)
        dy = np.clip(dest_y - robot_y, -0.1, 0.1)

        self._action_queue = [np.array([dx, dy, 0.0])] + [self._get_final_action()]

        return self._action_queue.pop(0)

    @abc.abstractmethod
    def _get_final_action(self) -> np.ndarray:
        raise NotImplementedError
    
class MoveSkill(Blocks2DSkill):
    def _get_lifted_operator(self) -> LiftedOperator:
        return MoveOperator

    def _get_final_action(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])

class PickUpSkill(Blocks2DSkill):
    def _get_lifted_operator(self) -> LiftedOperator:
        return PickUpOperator

    def _get_final_action(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

class PutDownSkill(Blocks2DSkill):
    def _get_lifted_operator(self) -> LiftedOperator:
        return PutDownOperator

    def _get_final_action(self) -> np.ndarray:
        return np.array([0.0, 0.0, -1.0])

skills = {MoveSkill(), PickUpSkill(), PutDownSkill()}