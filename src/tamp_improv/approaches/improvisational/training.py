"""Training utilities for improvisational approaches."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, cast

import numpy as np
from gymnasium.wrappers import RecordVideo

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

    # Collection settings
    collect_episodes: int = 100
    episodes_per_scenario: int = 1
    force_collect: bool = False

    # Save/Load settings
    save_dir: str = "trained_policies"
    training_data_dir: str = "training_data"

    # Visualization settings
    render: bool = False
    record_training: bool = False
    training_record_interval: int = 50

    # Policy-specific settings
    policy_config: dict[str, Any] | None = None

    def get_training_data_path(self, system_name: str) -> Path:
        """Get path for training data for specific system."""
        return Path(self.training_data_dir) / system_name


@dataclass
class Metrics:
    """Training and evaluation metrics."""

    success_rate: float
    avg_episode_length: float
    avg_reward: float


def get_or_collect_training_data(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: ImprovisationalTAMPApproach[ObsType, ActType],
    config: TrainingConfig,
) -> TrainingData:
    """Get existing or collect new training data."""
    # Check if saved data exists
    data_path = config.get_training_data_path(system.name)

    if not config.force_collect and data_path.exists():
        print(f"\nLoading existing training data from {data_path}")
        try:
            train_data = TrainingData.load(data_path)

            # Verify config matches
            if (
                train_data.config.get("seed") == config.seed
                and train_data.config.get("collect_episodes") == config.collect_episodes
                and train_data.config.get("max_steps") == config.max_steps
            ):
                print(f"Loaded {len(train_data)} training scenarios")
                train_data.config = config.__dict__
                return train_data
            print("Existing data has different config, collecting new data...")
        except Exception as e:
            print(f"Error loading training data: {e}")
            print("Collecting new data instead...")

    # Collect new data
    train_data = collect_training_data(system, approach, config)

    # Save the collected data
    print(f"\nSaving training data to {data_path}")
    train_data.save(data_path)

    return train_data


def collect_training_data(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: ImprovisationalTAMPApproach[ObsType, ActType],
    config: TrainingConfig,
) -> TrainingData:
    """Collect training data from TAMP execution."""
    training_states: list[Union[int, ObsType]] = []
    preconditions_to_maintain = []
    preconditions_to_achieve = []

    print("\nCollecting training data...")

    # Run episodes to collect states where preconditions aren't met
    for episode in range(config.collect_episodes):
        print(f"\nCollection episode {episode + 1}/{config.collect_episodes}")

        obs, info = system.reset()
        approach.reset(obs, info)

        for _ in range(config.max_steps):
            # If we encounter a state where preconditions aren't met
            if approach.policy_active and approach.prev_obs is not None:
                training_states.append(approach.prev_obs)
                preconditions_to_maintain.append(approach.currently_satisfied)
                preconditions_to_achieve.append(approach.target_atoms)
                break

            action = approach.step(obs, 0, False, False, info)
            obs, _, term, trunc, info = system.env.step(action)
            if term or trunc:
                break

    print(f"\nCollected {len(training_states)} training scenarios")

    return TrainingData(
        states=training_states,
        preconditions_to_maintain=preconditions_to_maintain,
        preconditions_to_achieve=preconditions_to_achieve,
        config=config.__dict__,
    )


def run_evaluation_episode(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: ImprovisationalTAMPApproach[ObsType, ActType],
    policy_cls: type[Policy],
    config: TrainingConfig,
    is_loaded_policy: bool = False,
) -> tuple[float, int, bool]:
    """Run single evaluation episode."""
    # Set up rendering if available
    render_mode = getattr(system.env, "render_mode", None)
    can_render = render_mode is not None
    if config.render and can_render:
        if is_loaded_policy:
            video_folder = Path(
                f"videos/{system.name}_(Loaded){policy_cls.__name__}_eval"
            )
        else:
            video_folder = Path(f"videos/{system.name}_{policy_cls.__name__}_eval")
        video_folder.mkdir(parents=True, exist_ok=True)

        # Record only the base environment, not the planning environment
        recording_env = deepcopy(system.env)
        system.env = RecordVideo(
            recording_env,
            str(video_folder),
            episode_trigger=lambda _: True,
            disable_logger=True,
        )

    obs, info = system.reset()
    approach.reset(obs, info)

    total_reward = 0.0
    for step in range(config.max_steps):
        action = approach.step(obs, total_reward, False, False, info)
        obs, reward, terminated, truncated, info = system.env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            return total_reward, step + 1, terminated

    if config.render and can_render:
        cast(Any, system.env).close()
        system.env = recording_env

    return total_reward, config.max_steps, False


def train_and_evaluate(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy_cls: type[Policy],
    config: TrainingConfig,
    is_loaded_policy: bool = False,
) -> Metrics:
    """Train and evaluate a policy on a system."""
    print(f"\nInitializing training for {system.name}...")

    # Create policy and approach
    policy = policy_cls(seed=config.seed)

    if is_loaded_policy:
        print("Loading saved policy...")
        policy_path = Path(config.save_dir) / f"{system.name}_{policy_cls.__name__}"
        try:
            policy.load(str(policy_path))
            print("Policy loaded successfully")
        except Exception as e:
            print(f"Error loading policy: {e}")
            return Metrics(success_rate=0.0, avg_episode_length=0.0, avg_reward=0.0)

    # Create approach with loaded/new policy
    approach = ImprovisationalTAMPApproach(system, policy, seed=config.seed)

    # Load or collect training data
    if policy.requires_training and not is_loaded_policy:
        train_data = get_or_collect_training_data(system, approach, config)

        if train_data.states:
            print("\nTraining policy...")

            # Set up rendering if available
            render_mode = getattr(system.wrapped_env, "render_mode", None)
            can_render = render_mode is not None
            if config.record_training and can_render:
                video_folder = Path(f"videos/{system.name}_{policy_cls.__name__}_train")
                video_folder.mkdir(parents=True, exist_ok=True)
                system.wrapped_env = RecordVideo(
                    system.wrapped_env,
                    str(video_folder),
                    episode_trigger=lambda x: x % config.training_record_interval == 0,
                )

            policy.train(system.wrapped_env, train_data)

            save_path = Path(config.save_dir) / f"{system.name}_{policy_cls.__name__}"
            print(f"\nSaving policy to {save_path}")
            policy.save(str(save_path))

            if config.record_training and can_render:
                cast(Any, system.wrapped_env).close()

    else:
        if is_loaded_policy:
            print("Using loaded policy - skipping training phase")

        # For non-training policies like MPC, just initialize
        policy.initialize(system.wrapped_env)

    # Run evaluation episodes
    print(f"\nEvaluating policy on {system.name}...")
    rewards = []
    lengths = []
    successes = []

    for episode in range(config.num_episodes):
        print(f"\nEvaluation Episode {episode + 1}/{config.num_episodes}")
        reward, length, success = run_evaluation_episode(
            system,
            approach,
            type(policy),
            config,
            is_loaded_policy=is_loaded_policy,
        )
        rewards.append(reward)
        lengths.append(length)
        successes.append(success)

        print(f"Current Success Rate: {sum(successes)/(episode+1):.2%}")
        print(f"Current Avg Episode Length: {np.mean(lengths):.2f}")
        print(f"Current Avg Reward: {np.mean(rewards):.2f}")

    return Metrics(
        success_rate=float(sum(successes) / len(successes)),
        avg_episode_length=float(np.mean(lengths)),
        avg_reward=float(np.mean(rewards)),
    )
