# Shortcut Learning for Abstract Planning

A shortcut learning framework for improving Task and Motion Planning (TAMP) through learned dynamic shortcuts using model-free Reinforcement Learning (RL). This version is for anonymous code supplement for NeurIPS 2025 submission.

## Requirements

- Python 3.11+
- PyBullet
- Tested on MacOS

## Installation

1. **Recommended**: Create and activate a virtual environment.

2. **Install the package**:
   ```bash
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

## Clarification

This repository contains the majority of code used for the paper's experiments. However, the two PyBullet environments are maintained in a separate repository (pybullet-blocks) and imported as dependencies for better modularity. For submission purposes and to ensure anonymity, we have included a copy of the relevant PyBullet environment implementations in the pybullet-blocks/ directory alongside this README. Note that when running the code, the system will still import from the external dependency specified in pyproject.toml, not from the local copy.
