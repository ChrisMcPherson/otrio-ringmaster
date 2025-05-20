from pathlib import Path
from scripts.train_tabular_q import train


def test_single_episode_saves(tmp_path: Path):
    out = tmp_path / "agent.json"
    train(num_episodes=1, save_path=str(out))
    assert out.exists()
