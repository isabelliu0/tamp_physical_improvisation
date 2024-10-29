"""Tests for policy evaluation."""

from tamp_improv.scripts.evaluate_pushing_policy import evaluate_policy

def test_policy_evaluation():
    """Test trained policy through evaluation."""
    
    results = evaluate_policy(
        policy_path="trained_policies/pushing_policy",
        num_episodes=10,
        seed=42,
        render=False,   # Run evaluation without rendering for faster testing
        debug=True
    )
    
    print(f"\nDetailed Results:")
    print(f"Total Episodes: {results['total_episodes']}")
    print(f"Successful Episodes: {results['successes']}")
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Average episode length: {results['avg_episode_length']:.1f}")
    print(f"Episode lengths: {results['all_lengths']}")
    print(f"Episode rewards: {results['all_rewards']}")

    # # Originally trained results:
    # # Success rate: 99.30%
    # # Average episode length: 36.5
    # assert results['success_rate'] >= 0.9, "Policy should succeed at least 90% of the time"
    # assert results['avg_episode_length'] < 50, "Episodes should complete efficiently"
