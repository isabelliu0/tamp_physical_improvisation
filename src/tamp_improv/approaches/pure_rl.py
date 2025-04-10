"""Pure RL baseline approach without using TAMP structure."""

from typing import Any, TypeVar

from tamp_improv.approaches.base import ApproachStepResult, BaseApproach
from tamp_improv.approaches.improvisational.policies.base import Policy
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class PureRLApproach(BaseApproach[ObsType, ActType]):
    """Pure RL approach that doesn't use TAMP structure."""

    def __init__(
        self,
        system: ImprovisationalTAMPSystem[ObsType, ActType],
        policy: Policy[ObsType, ActType],
        seed: int,
    ) -> None:
        """Initialize approach."""
        super().__init__(system, seed)
        self.policy = policy

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ApproachStepResult[ActType]:
        """Reset approach with initial observation."""
        return self.step(obs, 0.0, False, False, info)

    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ApproachStepResult[ActType]:
        """Step approach with new observation."""
        action = self.policy.get_action(obs)
        return ApproachStepResult(action=action)
