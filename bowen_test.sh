source /root/miniconda3/etc/profile.d/conda.sh
conda activate improv

pytest -s tests/approaches/test_graph_training.py -k test_multi_rl_cluttered_drawer_pipeline >> logs/test_graph_training.log