"""Training utilities for improvisational approaches."""

import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar, Union, cast

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.graph_training import (
    collect_graph_based_training_data,
)
from tamp_improv.approaches.improvisational.policies.base import Policy, TrainingData
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.utils.gpu_utils import set_torch_seed

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

    # Device settings
    device: str = "cuda"
    batch_size: int = 32

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
    training_time: float = 0.0
    total_time: float = 0.0


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
    train_data = collect_graph_based_training_data(system, approach, config.__dict__)

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
    current_atoms_list = []
    preimages_list = []

    # For compatibility (will be removed later)
    preconditions_to_maintain = []
    preconditions_to_achieve = []

    print("\nCollecting training data...")
    approach.training_mode = True

    # Run episodes to collect states for shortcut learning
    for episode in range(config.collect_episodes):
        print(f"\nCollection episode {episode + 1}/{config.collect_episodes}")

        obs, info = system.reset()
        step_result = approach.reset(obs, info)

        # Check first step from reset
        if step_result.terminate and step_result.info:
            training_states.append(step_result.info["training_state"])
            current_atoms_list.append(step_result.info["current_atoms"])
            preimages_list.append(step_result.info["preimage"])

            # For compatibility (will be removed later)
            if "preconditions_to_maintain" in step_result.info:
                preconditions_to_maintain.append(
                    step_result.info["preconditions_to_maintain"]
                )
                preconditions_to_achieve.append(
                    step_result.info["preconditions_to_achieve"]
                )
                continue

        obs, _, term, trunc, info = system.env.step(step_result.action)
        if term or trunc:
            continue

        # Rest of episode
        for _ in range(1, config.max_steps):
            step_result = approach.step(obs, 0, False, False, info)

            # When shortcut needed, collect training data
            if step_result.terminate and step_result.info:
                training_states.append(step_result.info["training_state"])

                current_atoms_list.append(step_result.info["current_atoms"])
                preimages_list.append(step_result.info["preimage"])

                # For compatibility (will be removed later)
                if "preconditions_to_maintain" in step_result.info:
                    preconditions_to_maintain.append(
                        step_result.info["preconditions_to_maintain"]
                    )
                    preconditions_to_achieve.append(
                        step_result.info["preconditions_to_achieve"]
                    )

                break

            obs, _, term, trunc, info = system.env.step(step_result.action)
            if term or trunc:
                break

    approach.training_mode = False

    print(f"\nCollected {len(training_states)} training scenarios")

    return TrainingData(
        states=training_states,
        current_atoms=current_atoms_list,
        preimages=preimages_list,
        preconditions_to_maintain=preconditions_to_maintain,
        preconditions_to_achieve=preconditions_to_achieve,
        config=config.__dict__,
    )


def run_evaluation_episode(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: ImprovisationalTAMPApproach[ObsType, ActType],
    policy_name: str,
    config: TrainingConfig,
    is_loaded_policy: bool = False,
    episode_num: int = 0,
) -> tuple[float, int, bool]:
    """Run single evaluation episode."""
    # Set up rendering if available
    render_mode = getattr(system.env, "render_mode", None)
    can_render = render_mode is not None
    if config.render and can_render:
        if is_loaded_policy:
            video_folder = Path(f"videos/{system.name}_(Loaded){policy_name}_eval")
        else:
            video_folder = Path(f"videos/{system.name}_{policy_name}_eval")
        video_folder.mkdir(parents=True, exist_ok=True)

        # Record only the base environment, not the planning environment
        recording_env = deepcopy(system.env)
        system.env = RecordVideo(
            recording_env,
            str(video_folder),
            episode_trigger=lambda _: True,
            name_prefix=f"episode_{episode_num}",
            disable_logger=True,
        )

    obs, info = system.reset()
    step_result = approach.reset(obs, info)

    total_reward = 0.0
    # First step from reset
    obs, reward, terminated, truncated, info = system.env.step(step_result.action)
    total_reward += float(reward)
    if terminated or truncated:
        if config.render and can_render:
            cast(Any, system.env).close()
            system.env = recording_env
        return total_reward, 1, terminated

    # Rest of steps
    for step in range(1, config.max_steps):
        step_result = approach.step(obs, total_reward, False, False, info)
        obs, reward, terminated, truncated, info = system.env.step(step_result.action)
        total_reward += float(reward)

        if terminated or truncated:
            if config.render and can_render:
                cast(Any, system.env).close()
                system.env = recording_env
            return total_reward, step + 1, terminated

    if config.render and can_render:
        cast(Any, system.env).close()
        system.env = recording_env

    return total_reward, config.max_steps, False


def train_and_evaluate(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy_factory: Callable[[int], Policy[ObsType, ActType]],
    config: TrainingConfig,
    policy_name: str,
) -> Metrics:
    """Train and evaluate a policy on a system."""
    print(f"\nInitializing training for {system.name}...")

    # Set all random seeds at the entry point
    seed = config.seed
    np.random.seed(seed)
    set_torch_seed(seed)

    # Print GPU information
    print("GPU Status:")
    if torch.cuda.is_available():
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  Current CUDA device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name()}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"  CUDA initialized and seeded with: {seed}")
    else:
        print("  CUDA not available, running on CPU")

    training_time = 0.0
    start_time = time.time()

    # Create policy using factory
    policy = policy_factory(config.seed)

    # Create approach with properly initialized policy
    approach = ImprovisationalTAMPApproach(system, policy, seed=config.seed)

    # Load or collect training data for new policy
    if policy.requires_training and "_Loaded" not in policy_name:
        train_data = get_or_collect_training_data(system, approach, config)

        if train_data.states:
            print("\nTraining policy...")

            # Set up rendering if available
            render_mode = getattr(system.wrapped_env, "render_mode", None)
            can_render = render_mode is not None

            if hasattr(system.wrapped_env, "configure_training"):
                system.wrapped_env.configure_training(train_data)

            if config.record_training and can_render:
                video_folder = Path(f"videos/{system.name}_{policy_name}_train")
                video_folder.mkdir(parents=True, exist_ok=True)
                system.wrapped_env = RecordVideo(
                    system.wrapped_env,
                    str(video_folder),
                    episode_trigger=lambda x: x % config.training_record_interval == 0,
                )

            policy.train(system.wrapped_env, train_data)
            training_time = time.time() - start_time

            save_path = Path(config.save_dir) / f"{system.name}_{policy_name}"
            print(f"\nSaving policy to {save_path}")
            policy.save(str(save_path))

            if config.record_training and can_render:
                cast(Any, system.wrapped_env).close()

    elif not policy.requires_training and "_Loaded" not in policy_name:
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
            policy_name,
            config,
            is_loaded_policy="_Loaded" in policy_name,
            episode_num=episode,
        )
        rewards.append(reward)
        lengths.append(length)
        successes.append(success)

        print(f"Current Success Rate: {sum(successes)/(episode+1):.2%}")
        print(f"Current Avg Episode Length: {np.mean(lengths):.2f}")
        print(f"Current Avg Reward: {np.mean(rewards):.2f}")

    total_time = time.time() - start_time
    return Metrics(
        success_rate=float(sum(successes) / len(successes)),
        avg_episode_length=float(np.mean(lengths)),
        avg_reward=float(np.mean(rewards)),
        training_time=training_time,
        total_time=total_time,
    )
