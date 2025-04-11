"""Training utilities for improvisational approaches."""

import inspect
import pickle
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.graph_training import (
    collect_goal_conditioned_training_data,
    collect_graph_based_training_data,
)
from tamp_improv.approaches.improvisational.policies.base import Policy, TrainingData
from tamp_improv.approaches.pure_rl import PureRLApproach
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.goal_wrapper import GoalConditionedWrapper
from tamp_improv.benchmarks.wrappers import PureRLWrapper
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
    max_training_steps_per_shortcut: int = 50

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

    # Shortcut information
    shortcut_info: list[dict[str, Any]] = field(default_factory=list)

    # Context size for augmenting observations
    max_preimage_size: int = 12

    # Goal-conditioned training settings
    success_threshold: float = 0.01
    success_reward: float = 10.0
    step_penalty: float = -0.5

    # Action scaling
    action_scale: float = 1.0

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
    use_random_rollouts: bool = False,
    num_rollouts_per_node: int = 50,
    max_steps_per_rollout: int = 50,
    shortcut_success_threshold: int = 1,
    rng: np.random.Generator | None = None,
) -> TrainingData:
    """Get existing or collect new training data."""
    # Check if saved data exists
    data_path = config.get_training_data_path(system.name)
    signatures_path = data_path / "trained_signatures.pkl"

    if not config.force_collect and data_path.exists():
        print(f"\nLoading existing training data from {data_path}")
        try:
            train_data = TrainingData.load(data_path)
            config.shortcut_info = train_data.config.get("shortcut_info", [])

            if signatures_path.exists():
                with open(signatures_path, "rb") as f:
                    approach.trained_signatures = pickle.load(f)
                print(f"Loaded {len(approach.trained_signatures)} trained signatures")

            # Verify config matches
            if (
                train_data.config.get("seed") == config.seed
                and train_data.config.get("collect_episodes") == config.collect_episodes
                and train_data.config.get("max_steps") == config.max_steps
                and train_data.config.get("using_context_wrapper", False)
                == approach.use_context_wrapper
                and train_data.config.get("use_random_rollouts") == use_random_rollouts
            ):
                if (
                    use_random_rollouts
                    and train_data.config.get("num_rollouts_per_node")
                    == num_rollouts_per_node
                    and train_data.config.get("max_steps_per_rollout")
                    == max_steps_per_rollout
                    and train_data.config.get("shortcut_success_threshold")
                    == shortcut_success_threshold
                ):
                    print(f"Loaded {len(train_data)} training scenarios")
                    train_data.config.update(config.__dict__)
                    return train_data

                print("Existing data has different config, collecting new data...")
        except Exception as e:
            print(f"Error loading training data: {e}")
            print("Collecting new data instead...")

    # Collect new data
    train_data, _ = collect_graph_based_training_data(
        system,
        approach,
        config.__dict__,
        use_random_rollouts=use_random_rollouts,
        num_rollouts_per_node=num_rollouts_per_node,
        max_steps_per_rollout=max_steps_per_rollout,
        shortcut_success_threshold=shortcut_success_threshold,
        action_scale=config.action_scale,
        rng=rng,
    )

    # Save the collected data
    print(f"\nSaving training data to {data_path}")
    train_data.save(data_path)

    # Save trained signatures separately
    if approach.trained_signatures:
        print(f"Saving {len(approach.trained_signatures)} trained signatures")
        with open(signatures_path, "wb") as f:
            pickle.dump(approach.trained_signatures, f)

    return train_data


def run_evaluation_episode(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: (
        ImprovisationalTAMPApproach[ObsType, ActType] | PureRLApproach[ObsType, ActType]
    ),
    policy_name: str,
    config: TrainingConfig,
    is_loaded_policy: bool = False,
    episode_num: int = 0,
    select_random_goal: bool = False,
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
    if (
        hasattr(approach, "reset")
        and "select_random_goal" in inspect.signature(approach.reset).parameters
    ):
        step_result = approach.reset(obs, info, select_random_goal=select_random_goal)  # type: ignore[call-arg]  # pylint: disable=line-too-long
    else:
        step_result = approach.reset(obs, info)

    total_reward = 0.0
    step_count = 0
    success = False

    # Execute first action from the reset
    obs, reward, terminated, truncated, info = system.env.step(step_result.action)
    total_reward += float(reward)
    step_count += 1
    if step_result.terminate or terminated or truncated:
        success = step_result.terminate or terminated
        if config.render and can_render:
            cast(Any, system.env).close()
            system.env = recording_env
        return total_reward, step_count, success

    # Rest of steps
    for _ in range(1, config.max_steps):
        step_result = approach.step(obs, total_reward, False, False, info)
        obs, reward, terminated, truncated, info = system.env.step(step_result.action)
        total_reward += float(reward)
        step_count += 1
        if step_result.terminate or terminated or truncated:
            success = step_result.terminate or terminated
            break

    if config.render and can_render:
        cast(Any, system.env).close()
        system.env = recording_env

    return total_reward, step_count, success


def train_and_evaluate(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy_factory: Callable[[int], Policy[ObsType, ActType]],
    config: TrainingConfig,
    policy_name: str,
    use_context_wrapper: bool = False,
    use_random_rollouts: bool = False,
    num_rollouts_per_node: int = 50,
    max_steps_per_rollout: int = 50,
    shortcut_success_threshold: int = 1,
    select_random_goal: bool = False,
) -> Metrics:
    """Train and evaluate a policy on a system."""
    print(f"\nInitializing training for {system.name}...")

    # Set all random seeds at the entry point
    seed = config.seed
    rng = np.random.default_rng(seed)
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
    approach = ImprovisationalTAMPApproach(
        system,
        policy,
        seed=config.seed,
        max_preimage_size=config.max_preimage_size,
        use_context_wrapper=use_context_wrapper,
    )

    # Load or collect training data for new policy
    if policy.requires_training and "_Loaded" not in policy_name:
        train_data = get_or_collect_training_data(
            system,
            approach,
            config,
            use_random_rollouts=use_random_rollouts,
            num_rollouts_per_node=num_rollouts_per_node,
            max_steps_per_rollout=max_steps_per_rollout,
            shortcut_success_threshold=shortcut_success_threshold,
            rng=rng,
        )

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
            select_random_goal=select_random_goal,
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


def train_and_evaluate_goal_conditioned(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy_factory: Callable[[int], Policy[ObsType, ActType]],
    config: TrainingConfig,
    policy_name: str,
    use_preimages: bool = True,
    use_random_rollouts: bool = False,
    num_rollouts_per_node: int = 50,
    max_steps_per_rollout: int = 50,
    shortcut_success_threshold: int = 1,
) -> Metrics:
    """Train and evaluate a goal-conditioned policy for shortcut learning."""
    print(f"\nInitializing goal-conditioned training for {system.name}...")

    # Set random seeds
    seed = config.seed
    rng = np.random.default_rng(seed)
    set_torch_seed(seed)

    training_time = 0.0
    start_time = time.time()

    # Create policy and approach
    policy = policy_factory(config.seed)
    approach = ImprovisationalTAMPApproach(
        system,
        policy,
        seed=config.seed,
        max_preimage_size=config.max_preimage_size,
    )

    # Collect goal-conditioned training data
    train_data = collect_goal_conditioned_training_data(
        system,
        approach,
        config.__dict__,
        use_random_rollouts=use_random_rollouts,
        num_rollouts_per_node=num_rollouts_per_node,
        max_steps_per_rollout=max_steps_per_rollout,
        shortcut_success_threshold=shortcut_success_threshold,
        rng=rng,
    )

    if policy.requires_training and "_Loaded" not in policy_name:
        if train_data.node_states:
            print("\nPreparing goal-conditioned environment...")

            # Create goal-conditioned environment
            # Here we replace the ImprovWrapper with GoalConditionedWrapper
            goal_env = GoalConditionedWrapper(
                env=system.env,  # access base env, not wrapped env
                node_states=train_data.node_states,
                valid_shortcuts=train_data.valid_shortcuts,
                perceiver=system.perceiver if use_preimages else None,
                node_preimages=train_data.node_preimages if use_preimages else None,
                use_preimages=use_preimages,
                max_preimage_size=config.max_preimage_size,
                success_threshold=config.success_threshold,
                success_reward=config.success_reward,
                step_penalty=config.step_penalty,
                max_episode_steps=config.max_steps,
            )

            # Use this environment for training
            system.wrapped_env = goal_env

            print("\nTraining policy...")

            if hasattr(system.wrapped_env, "configure_training"):
                system.wrapped_env.configure_training(train_data)

            # Record training if requested
            if config.record_training and hasattr(system.wrapped_env, "render_mode"):
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

    # Run evaluation
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


def train_and_evaluate_pure_rl(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy_factory: Callable[[int], Policy[ObsType, ActType]],
    config: TrainingConfig,
    policy_name: str,
) -> Metrics:
    """Train and evaluate a pure RL policy on a system."""
    print(f"\nInitializing pure RL baseline training for {system.name}...")
    seed = config.seed
    set_torch_seed(seed)

    policy = policy_factory(seed)

    obs, info = system.reset()
    _, _, goal_atoms = system.perceiver.reset(obs, info)

    pure_rl_env = PureRLWrapper(
        env=system.env,
        perceiver=system.perceiver,
        goal_atoms=goal_atoms,
        max_episode_steps=config.max_steps,
        step_penalty=config.step_penalty,
        achievement_bonus=config.success_reward,
        action_scale=config.action_scale,
    )

    render_mode = getattr(pure_rl_env, "_render_mode", None)
    can_render = render_mode is not None
    if config.record_training and can_render:
        video_folder = Path(f"videos/{system.name}_{policy_name}_train")
        video_folder.mkdir(parents=True, exist_ok=True)
        pure_rl_env = RecordVideo(
            pure_rl_env,  # type: ignore[assignment]
            str(video_folder),
            episode_trigger=lambda x: x % config.training_record_interval == 0,
            name_prefix="training",
        )

    # Initialize policy
    policy.initialize(pure_rl_env)

    # Train policy if needed
    start_time = time.time()
    if policy.requires_training:
        print("\nTraining pure RL policy...")
        policy.train(pure_rl_env, train_data=None)

        save_path = Path(config.save_dir) / f"{system.name}_{policy_name}"
        policy.save(str(save_path))

    training_time = time.time() - start_time

    approach = PureRLApproach(system, policy, seed)

    # Run evaluation
    print(f"\nEvaluating pure RL policy on {system.name}...")
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
            is_loaded_policy=False,
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
