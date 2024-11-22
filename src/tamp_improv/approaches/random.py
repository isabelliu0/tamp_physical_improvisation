"""Random action approach."""

from typing import Any

from tamp_improv.approaches.base import ActType, BaseApproach, ObsType


class RandomApproach(BaseApproach[ObsType, ActType]):
    """An approach that takes random actions."""

    def reset(self, obs: ObsType, info: dict[str, Any]) -> ActType:
        """Reset approach."""
        return self.system.env.action_space.sample()

    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ActType:
        """Take random action."""
        return self.system.env.action_space.sample()
