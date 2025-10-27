"""Training utilities for improvisational approaches."""

import inspect
import pickle
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar, Union, cast

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from tamp_improv.approaches.improvisational.base import ImprovisationalTAMPApproach
from tamp_improv.approaches.improvisational.graph_training import (
    collect_goal_conditioned_training_data,
    collect_graph_based_training_data,
)
from tamp_improv.approaches.improvisational.policies.base import Policy, TrainingData
from tamp_improv.approaches.improvisational.policies.multi_rl import MultiRLPolicy
from tamp_improv.approaches.pure_rl import PureRLApproach, SACHERApproach
from tamp_improv.approaches.hierarchical_rl import HierarchicalRLApproach
from tamp_improv.benchmarks.base import ImprovisationalTAMPSystem
from tamp_improv.benchmarks.context_wrapper import ContextAwareWrapper
from tamp_improv.benchmarks.goal_wrapper import GoalConditionedWrapper
from tamp_improv.benchmarks.hierarchical_wrapper import HierarchicalRLWrapper
from tamp_improv.benchmarks.sac_her_wrapper import SACHERWrapper
from tamp_improv.benchmarks.wrappers import PureRLWrapper
from tamp_improv.utils.gpu_utils import get_gpu_memory_info, set_torch_seed

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class TrainingConfig:
    """Configuration for training."""

    seed: int = 42
    num_episodes: int = 50
    max_steps: int = 100
    max_training_steps_per_shortcut: int = 50

    collect_episodes: int = 100
    episodes_per_scenario: int = 1
    force_collect: bool = False

    save_dir: str = "trained_policies"
    training_data_dir: str = "training_data"

    render: bool = False
    record_training: bool = False
    training_record_interval: int = 50
    fast_eval: bool = False

    device: str = "cuda"
    batch_size: int = 32

    policy_config: Union[dict[str, Any], None] = None
    shortcut_info: list[dict[str, Any]] = field(default_factory=list)
    max_atom_size: int = 12

    success_threshold: float = 0.01
    success_reward: float = 10.0
    step_penalty: float = -0.5
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
    rng: Union[np.random.Generator, None] = None,
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
                train_data.config.get("collect_episodes") == config.collect_episodes
                and train_data.config.get("max_steps") == config.max_steps
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
    approach: Union[
        ImprovisationalTAMPApproach[ObsType, ActType], PureRLApproach[ObsType, ActType], SACHERApproach[ObsType, ActType],
        HierarchicalRLApproach[ObsType, ActType]
    ],
    policy_name: str,
    config: TrainingConfig,
    episode_num: int = 0,
    select_random_goal: bool = False,
) -> tuple[float, int, bool]:
    """Run single evaluation episode."""
    # Set up rendering if available
    render_mode = getattr(system.env, "render_mode", None)
    can_render = render_mode is not None
    if config.render and can_render:
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


def run_evaluation_episode_with_caching(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    approach: Union[
        ImprovisationalTAMPApproach[ObsType, ActType], PureRLApproach[ObsType, ActType], SACHERApproach[ObsType, ActType],
        HierarchicalRLApproach[ObsType, ActType]
    ],
    policy_name: str,
    config: TrainingConfig,
    episode_num: int = 0,
    select_random_goal: bool = False,
) -> tuple[float, int, bool]:
    """Run single evaluation episode."""
    # Set up rendering if available
    render_mode = getattr(system.env, "render_mode", None)
    can_render = render_mode is not None
    if config.render and can_render:
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
    
    if config.fast_eval and not (config.render and can_render):
        step_count = approach.best_eval_total_steps
        success = bool(approach.best_eval_path)
        return success, step_count, success

    best_edges = approach._current_path
    if not best_edges:
        return 0.0, 0, False
    prefix_ids_for_edge: list[tuple[int, ...]] = []
    running_prefix: tuple[int, ...] = (0,)
    for edge in best_edges:
        prefix_ids_for_edge.append(running_prefix)
        running_prefix = running_prefix + (edge.source.id,)

    total_reward = 0.0
    step_count = 0
    done = False
    success = True

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

    key = (0, best_edges[0].source.id, ())
    actions = approach._edge_action_cache.get(key, None)
    if actions is not None:
        for a in actions:
            obs, reward, terminated, truncated, info = system.env.step(a)
            total_reward += float(reward)
            step_count += 1
            done = bool(terminated or truncated)
            if done:
                break
    else:
        for _ in range(approach._max_skill_steps):
            step_result = approach.step(obs, total_reward, False, False, info)
            obs, reward, terminated, truncated, info = system.env.step(step_result.action)
            total_reward += float(reward)
            step_count += 1
            done = bool(step_result.terminate or terminated or truncated)
            if step_result.terminate or terminated or truncated:
                success = step_result.terminate or terminated
                break
    for i, edge in enumerate(best_edges):
        if done:
            break
        key = (edge.source.id, edge.target.id, prefix_ids_for_edge[i])
        actions = approach._edge_action_cache.get(key, None)
        if actions is not None:
            for a in actions:
                obs, reward, terminated, truncated, info = system.env.step(a)
                total_reward += float(reward)
                step_count += 1
                done = bool(terminated or truncated)
                if done:
                    break
            if done:
                break
        else:
            for _ in range(approach._max_skill_steps):
                step_result = approach.step(obs, total_reward, False, False, info)
                obs, reward, terminated, truncated, info = system.env.step(step_result.action)
                total_reward += float(reward)
                step_count += 1
                done = bool(step_result.terminate or terminated or truncated)
                if step_result.terminate or terminated or truncated:
                    success = step_result.terminate or terminated
                    break
            if done:
                break

    if not done:
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
    enable_generalization: bool = False,
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
        memory_info = get_gpu_memory_info()
        assert not isinstance(memory_info, str)
        for gpu in memory_info:
            print(f"  Device {gpu['device_index']} ({gpu['name']}):")
            print(f"    Total memory: {gpu['total_memory']:.2f} GB")
            print(f"    Allocated memory: {gpu['allocated_memory']:.2f} GB")
            print(f"    Free memory: {gpu['free_memory']:.2f} GB")
    else:
        print("  CUDA not available, running on CPU")

    training_time = 0.0
    start_time = time.time()

    # Create policy using factory
    policy = policy_factory(config.seed)

    # Set enable_generalization if policy is MultiRLPolicy
    if isinstance(policy, MultiRLPolicy):
        policy.enable_generalization = enable_generalization

    # Create approach with properly initialized policy
    approach = ImprovisationalTAMPApproach(
        system,
        policy,
        seed=config.seed,
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

            if use_context_wrapper:
                context_env = ContextAwareWrapper(
                    system.wrapped_env,
                    system.perceiver,
                    max_atom_size=config.max_atom_size,
                    max_episode_steps=config.max_steps,
                )
                system.wrapped_env = context_env

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

            save_path = Path(config.save_dir) / f"{system.name}_{policy_name}"

            if isinstance(policy, MultiRLPolicy):
                policy.train(system.wrapped_env, train_data, save_dir=str(save_path))
            else:
                policy.train(system.wrapped_env, train_data)
            training_time = time.time() - start_time

            print(f"\nSaving policy to {save_path}")
            policy.save(str(save_path))

            if config.record_training and can_render:
                cast(Any, system.wrapped_env).close()

    elif not policy.requires_training and "_Loaded" not in policy_name:
        # For non-training policies like MPC, just initialize
        policy.initialize(system.wrapped_env)

    # For MultiRLPolicy, ensure models are properly loaded before evaluation
    if isinstance(policy, MultiRLPolicy) and hasattr(policy, "policies"):
        if not any(
            hasattr(p, "model") and p.model is not None
            for p in policy.policies.values()
        ):
            print(f"No loaded models detected, attempting to load from {save_path}")
            policy.load(str(save_path))

    # Run evaluation episodes
    print(f"\nEvaluating policy on {system.name}...")
    rewards = []
    lengths = []
    successes = []

    for episode in range(config.num_episodes):
        print(f"\nEvaluation Episode {episode + 1}/{config.num_episodes}")
        reward, length, success = run_evaluation_episode_with_caching(
            system,
            approach,
            policy_name,
            config,
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
    use_atom_as_obs: bool = True,
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
                perceiver=system.perceiver if use_atom_as_obs else None,
                node_atoms=train_data.node_atoms if use_atom_as_obs else None,
                use_atom_as_obs=use_atom_as_obs,
                max_atom_size=config.max_atom_size,
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


def train_and_evaluate_sac_her(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy_factory: Callable[[int], Policy[ObsType, ActType]],
    config: TrainingConfig,
    policy_name: str,
    max_atom_size: int = 14,
) -> Metrics:
    """Train and evaluate SAC+HER baseline."""
    print(f"\nInitializing SAC+HER baseline training for {system.name}...")

    seed = config.seed
    set_torch_seed(seed)

    policy = policy_factory(seed)

    obs, info = system.reset()
    _, _, goal_atoms = system.perceiver.reset(obs, info)

    sac_her_env = SACHERWrapper(
        env=system.env,
        perceiver=system.perceiver,
        goal_atoms=goal_atoms,
        max_atom_size=max_atom_size,
        max_episode_steps=config.max_steps,
        step_penalty=config.step_penalty,
        success_reward=config.success_reward,
    )

    start_time = time.time()
    if policy.requires_training:
        print("\nTraining SAC+HER baseline...")
        policy.train(sac_her_env, train_data=None)
        save_path = Path(config.save_dir) / f"{system.name}_{policy_name}"
        policy.save(str(save_path))
    training_time = time.time() - start_time

    system.wrapped_env = sac_her_env

    approach = SACHERApproach(system, policy, seed)

    # Run evaluation
    print(f"\nEvaluating SAC+HER baseline on {system.name}...")
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


def train_and_evaluate_hierarchical_rl(
    system: ImprovisationalTAMPSystem[ObsType, ActType],
    policy_factory: Callable[[int], Policy[ObsType, ActType]],
    config: TrainingConfig,
    policy_name: str,
    single_step_skills: bool = True,
    max_skill_steps: int = 50,
    skill_failure_penalty: float = -1.0,
) -> Metrics:
    """Train and evaluate a hierarchical RL policy on a system."""
    print(f"\nInitializing hierarchical RL baseline training for {system.name}...")
    seed = config.seed
    set_torch_seed(seed)

    policy = policy_factory(seed)

    obs, info = system.reset()
    _, _, _ = system.perceiver.reset(obs, info)

    hierarchical_rl_env = HierarchicalRLWrapper(
        tamp_system=system,
        max_episode_steps=config.max_steps,
        max_skill_steps=max_skill_steps,
        step_penalty=config.step_penalty,
        achievement_bonus=config.success_reward,
        action_scale=config.action_scale,
        skill_failure_penalty=skill_failure_penalty,
        single_step_skills=single_step_skills,
    )

    render_mode = getattr(hierarchical_rl_env, "render_mode", None)
    can_render = render_mode is not None
    if config.record_training and can_render:
        video_folder = Path(f"videos/{system.name}_{policy_name}_train")
        video_folder.mkdir(parents=True, exist_ok=True)
        hierarchical_rl_env = RecordVideo(
            hierarchical_rl_env,  # type: ignore[assignment]
            str(video_folder),
            episode_trigger=lambda x: x % config.training_record_interval == 0,
            name_prefix="training",
        )

    # Train policy
    start_time = time.time()
    if policy.requires_training:
        print("\nTraining hierarchical RL policy...")
        policy.train(hierarchical_rl_env, train_data=None)

        save_path = Path(config.save_dir) / f"{system.name}_{policy_name}"
        policy.save(str(save_path))

    training_time = time.time() - start_time

    approach = HierarchicalRLApproach(system, policy, seed, hierarchical_rl_env)

    # Run evaluation
    print(f"\nEvaluating hierarchical RL policy on {system.name}...")
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
