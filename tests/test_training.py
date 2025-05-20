from scripts.train_tabular_q import train


def test_single_episode_train(tmp_path):
    save_path = tmp_path / "agent.pkl"
    train(num_episodes=1, save_path=str(save_path))
    assert save_path.exists()
