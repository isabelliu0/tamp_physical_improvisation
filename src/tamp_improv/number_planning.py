"""Planning components for the number environment."""

from typing import Any, Sequence, Set, Tuple, cast

from relational_structs import (
    GroundAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)
from task_then_motion_planning.structs import LiftedOperatorSkill, Perceiver, Skill


def create_number_planning_models(
    switch_off_improvisational_models: bool = False,
) -> tuple[
    set[Type], set[Predicate], Perceiver[Any], set[LiftedOperator], set[Skill[Any, Any]]
]:
    """Create types, predicates, perceiver, operators, and skills for number
    environment."""
    # Create types and predicates
    state_type = Type("state")
    types = {state_type}

    # Predicates for each state
    AtState0 = Predicate("AtState0", [state_type])
    AtState1 = Predicate("AtState1", [state_type])
    AtState2 = Predicate("AtState2", [state_type])
    predicates = {AtState0, AtState1, AtState2}

    if switch_off_improvisational_models:
        CanProgress = Predicate("CanProgress", [state_type])
        predicates.add(CanProgress)

    # Create operators
    state = Variable("?state", state_type)

    ZeroToOneOp = LiftedOperator(
        "ZeroToOne",
        [state],
        preconditions={AtState0([state])},
        add_effects={AtState1([state])},
        delete_effects={AtState0([state])},
    )

    # Only include CanProgress precondition if switch_off_improvisational_models is True.
    one_to_two_preconditions = {AtState1([state])}
    if switch_off_improvisational_models:
        one_to_two_preconditions.add(CanProgress([state]))

    OneToTwoOp = LiftedOperator(
        "OneToTwo",
        [state],
        preconditions=one_to_two_preconditions,
        add_effects={AtState2([state])},
        delete_effects={AtState1([state])},
    )

    operators = {ZeroToOneOp, OneToTwoOp}

    # Create perceiver
    class NumberPerceiver(Perceiver[int]):
        """Perceiver for number environment."""

        def __init__(self) -> None:
            self._state = cast(Object, state_type("state1"))

        def reset(
            self, obs: int, info: dict[str, Any]
        ) -> Tuple[Set[Object], Set[GroundAtom], Set[GroundAtom]]:
            objects: Set[Object] = {self._state}
            atoms = self._get_atoms(obs)
            goal: Set[GroundAtom] = {AtState2([self._state])}
            return objects, atoms, goal

        def step(self, obs: int) -> set[GroundAtom]:
            return self._get_atoms(obs)

        def _get_atoms(self, obs: int) -> set[GroundAtom]:
            atoms = set()
            # Map environment state to corresponding predicate
            state_to_predicate = {0: AtState0, 1: AtState1, 2: AtState2}
            atoms.add(state_to_predicate[obs]([self._state]))

            return atoms

    # Create skills for each operator
    class ZeroToOneSkill(LiftedOperatorSkill[int, int]):
        """Skill for transitioning from state 0 to 1."""

        def _get_lifted_operator(self) -> LiftedOperator:
            return ZeroToOneOp

        def _get_action_given_objects(self, objects: Sequence[Object], obs: int) -> int:
            return 1

    class OneToTwoSkill(LiftedOperatorSkill[int, int]):
        """Skill for transitioning from state 1 to 2."""

        def _get_lifted_operator(self) -> LiftedOperator:
            return OneToTwoOp

        def _get_action_given_objects(self, objects: Sequence[Object], obs: int) -> int:
            return 1

    # Create skill set
    skills: set[Skill[Any, Any]] = {
        cast(Skill[Any, Any], ZeroToOneSkill()),
        cast(Skill[Any, Any], OneToTwoSkill()),
    }

    perceiver = cast(Perceiver[Any], NumberPerceiver())

    return types, predicates, perceiver, operators, skills
