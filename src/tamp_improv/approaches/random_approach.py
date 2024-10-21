"""An approach that takes random actions."""

from typing import Any

from gymnasium.core import ActType, ObsType

from tamp_improv.approaches.base_approach import BaseApproach


class RandomApproach(BaseApproach[ObsType, ActType]):
    """An approach that takes random actions."""

    def reset(self, obs: ObsType) -> ActType:
        return self._action_space.sample()

    def step(
        self,
        obs: ObsType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> ActType:
        return self._action_space.sample()
