"""Training utilities for improvisational approaches."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.policies.base import Policy, TrainingData
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # General settings
    seed: int = 42
    num_episodes: int = 50
    max_steps: int = 100
    save_dir: str = "trained_policies"
    render: bool = False

    # Collection settings
    collect_episodes: int = 5

    # Policy-specific settings
    policy_config: dict[str, Any] = None


@dataclass
class Metrics:
    """Training and evaluation metrics."""

    success_rate: float
    avg_episode_length: float
    avg_reward: float
    training_steps: int = 0


def collect_training_data(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: ImprovisationalTAMPApproach[ObsType, ActType],
    config: TrainingConfig,
) -> TrainingData:
    """Collect training data from TAMP execution."""
    training_states = []
    training_operators = []
    training_preconditions = []

    # Run episodes to collect states where preconditions aren't met
    for _ in range(config.collect_episodes):
        obs, info = system.reset()
        approach.reset(obs, info)

        for _ in range(config.max_steps):
            # If we encounter a state where preconditions aren't met
            if approach._policy_active:
                training_states.append(obs)
                training_operators.append(approach._current_operator)
                training_preconditions.append(approach._target_atoms)
                break

            action = approach.step(obs, 0, False, False, info)
            obs, _, term, trunc, info = system.env.step(action)
            if term or trunc:
                break

    return TrainingData(
        states=training_states,
        operators=training_operators,
        preconditions=training_preconditions,
        config=config.policy_config or {},
    )


def run_evaluation_episode(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: ImprovisationalTAMPApproach[ObsType, ActType],
    config: TrainingConfig,
) -> tuple[float, int, bool]:
    """Run single evaluation episode."""
    if config.render:
        from gymnasium.wrappers import RecordVideo

        video_folder = Path(f"videos/{system.name}_eval")
        video_folder.mkdir(parents=True, exist_ok=True)
        system.env = RecordVideo(system.env, str(video_folder))

    obs, info = system.reset()
    approach.reset(obs, info)

    total_reward = 0.0
    for step in range(config.max_steps):
        action = approach.step(obs, total_reward, False, False, info)
        obs, reward, terminated, truncated, info = system.env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            return total_reward, step + 1, terminated

    return total_reward, config.max_steps, False


def train_and_evaluate(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy_cls: type[Policy],
    config: TrainingConfig,
) -> Metrics:
    """Train and evaluate a policy on a system."""
    # Create policy and approach
    policy = policy_cls(seed=config.seed)

    # Create approach (this will initializate policy with wrapped_env)
    approach = ImprovisationalTAMPApproach(system, policy, seed=config.seed)

    # Only collect training data and train if policy requires it
    if policy.requires_training:
        print(f"\nCollecting training data for {system.name}...")
        train_data = collect_training_data(system, approach, config)

        if train_data.states:
            print("\nTraining policy...")
            policy.train(system.wrapped_env, train_data)

            save_path = Path(config.save_dir) / f"{system.name}_{policy_cls.__name__}"
            print(f"Saving policy to {save_path}")
            policy.save(str(save_path))
    else:
        # For non-training policies like MPC, just initialize
        policy.initialize(system.wrapped_env)

    # Run evaluation episodes
    print(f"\nEvaluating on {system.name}...")
    rewards = []
    lengths = []
    successes = []

    for episode in range(config.num_episodes):
        reward, length, success = run_evaluation_episode(system, approach, config)
        rewards.append(reward)
        lengths.append(length)
        successes.append(success)

    return Metrics(
        success_rate=sum(successes) / len(successes),
        avg_episode_length=np.mean(lengths),
        avg_reward=np.mean(rewards),
    )
