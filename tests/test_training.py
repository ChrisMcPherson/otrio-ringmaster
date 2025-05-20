import os
import subprocess
import sys


def test_training_creates_checkpoint(tmp_path):
    checkpoint = tmp_path / "agent.pkl"
    subprocess.run(
        [sys.executable, "scripts/train_tabular_q.py", "--episodes", "1", "--checkpoint", str(checkpoint)],
        check=True,
    )
    assert checkpoint.exists()

def test_ppo_training(tmp_path):
    checkpoint = tmp_path / "ppo.pkl"
    subprocess.run(
        [sys.executable, "scripts/train_ppo.py", "--episodes", "1", "--checkpoint", str(checkpoint)],
        check=True,
    )
    assert checkpoint.exists()
