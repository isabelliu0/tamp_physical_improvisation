"""Tests for RL improvisational policy implementation."""

import numpy as np

from tamp_improv.approaches.rl_improvisational_policy import RLImprovisationalPolicy
from tamp_improv.benchmarks.blocks2d_env import Blocks2DEnv
from tamp_improv.benchmarks.pushing_env import make_pushing_env


def test_policy_basic_functionality():
    """Test basic policy functionality."""
    # Setup
    base_env = Blocks2DEnv()
    env = make_pushing_env(base_env, seed=42)
    policy = RLImprovisationalPolicy(env)

    # Test action generation
    obs, _ = env.reset(seed=42)
    action = policy.get_action(obs)

    # Check action properties
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)  # dx, dy, gripper
    assert env.action_space.contains(action)


def test_training_no_crash():
    """Test that training runs without crashing."""
    base_env = Blocks2DEnv()
    env = make_pushing_env(base_env, seed=42)
    policy = RLImprovisationalPolicy(env)

    # Train for just a few steps to verify it works
    policy.train(total_timesteps=100, seed=42)


def test_save_load(tmp_path):
    """Test saving and loading policy."""
    base_env = Blocks2DEnv()
    env = make_pushing_env(base_env, seed=42)
    policy = RLImprovisationalPolicy(env)

    save_path = str(tmp_path / "test_policy")
    policy.save(save_path)

    new_policy = RLImprovisationalPolicy(env)
    new_policy.load(save_path)

    # Verify both policies give same action
    obs, _ = env.reset(seed=42)
    action1 = policy.get_action(obs)
    action2 = new_policy.get_action(obs)
    np.testing.assert_array_equal(action1, action2)


def test_deterministic_behavior():
    """Test that policy is deterministic with same seed."""
    base_env = Blocks2DEnv()
    env = make_pushing_env(base_env, seed=42)
    policy = RLImprovisationalPolicy(env)

    obs, _ = env.reset(seed=42)
    action1 = policy.get_action(obs)
    action2 = policy.get_action(obs)
    np.testing.assert_array_equal(action1, action2)
