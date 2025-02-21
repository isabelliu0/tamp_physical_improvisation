"""Simple script to plot training metrics from log file."""

import re
import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress(log_file, output_file=None):
    """Parse log file and plot training metrics."""
    # Read file
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract metrics with regex
    pattern = r"Training Progress:\s+Episodes: (\d+)\s+Recent Success%: ([\d.]+)%\s+Recent Avg Episode Length: ([\d.]+)\s+Recent Avg Reward: (-?[\d.]+)"
    matches = re.findall(pattern, content)
    
    # Parse matches into lists
    episodes = [int(m[0]) for m in matches]
    success_rates = [float(m[1]) for m in matches]
    episode_lengths = [float(m[2]) for m in matches]
    rewards = [float(m[3]) for m in matches]
    
    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot metrics
    ax1.plot(episodes, success_rates, 'b-o')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Training Progress')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(episodes, rewards, 'g-o')
    ax2.set_ylabel('Average Reward')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(episodes, episode_lengths, 'r-o')
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Avg Episode Length')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    import sys
    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    plot_training_progress(log_file, output_file)