# Shortcut Learning for Abstract Planning

A shortcut learning framework for improving Task and Motion Planning (TAMP) through learned dynamic shortcuts using model-free Reinforcement Learning (RL). This version is for the anonymous code supplement for the NeurIPS 2025 submission.

## Requirements

- Python 3.11+
- Tested on MacOS and Ubuntu 22.04

## Installation

1. **Recommended**: Create and activate a virtual environment.

2. **Install the package**:
   ```bash
   cd pybullet-blocks
   pip install -e ."[develop]"
   cd ../
   pip install -e ".[develop]"
   ```

## Quick Start

### Basic Usage

1. **Run Singular Experiments**:
   All experiments on one singular task of each environment are in tests/approaches/test_graph_training.py. An example command is
   ```python
   pytest -s tests/approaches/test_graph_training.py -k test_multi_rl_obstacle2d_pipeline
   ```
   Some pytests are skipped for the sake of unit tests checking on GitHub. Make sure to uncomment the line before running.

   We provide small datasets for you to start training RL shortcut policies in train_data/. If you wish to use the provided data, make sure to include them in the right paths in order to use them directly. If you wish to collect your own data (recommended for large-scale experiments), you can directly run the corresponding commands.

2. **Reproduce Paper Results**:
   ```python
   python -m src.tamp_improv.run_experiments --system=<env> --episodes=<episodes_per_scenario>
   ```
