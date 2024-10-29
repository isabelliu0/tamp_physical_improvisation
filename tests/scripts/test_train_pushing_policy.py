"""Test for training script."""

from tamp_improv.scripts.train_pushing_policy import train_pushing_policy


def test_train_pushing_policy(timesteps=1_000):
    """Test training pushing policy."""
    train_pushing_policy(
        total_timesteps=timesteps, seed=42, save_path="trained_policies/pushing_policy_test"
    )
