import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.random_agent import RandomAgent
from agents.ppo import PPOAgent
from envs.otrio_env import OtrioEnv


def play_episode(env: OtrioEnv, ppo: PPOAgent, opponent) -> dict:
    obs, info = env.reset()
    done = False
    last_move_index = None
    player = info["current_player"]
    while not done:
        if player == 0:
            action = ppo.select_action(env.board.clone(), 0)
            last_move_index = len(ppo.rewards)
            obs, reward, done, info = env.step(action)
            ppo.store_outcome(reward, done)
        else:
            action = opponent.select_action(env.board.clone(), 1)
            obs, reward, done, info = env.step(action)
            if done and info.get("winner") == 1 and last_move_index is not None:
                ppo.rewards[last_move_index] = -1.0
        player = env.current_player
    ppo.update()
    return info


def train(episodes: int = 1000, checkpoint: str | None = None):
    env = OtrioEnv(players=2)
    ppo = PPOAgent()
    opponent = RandomAgent()
    for ep in range(episodes):
        info = play_episode(env, ppo, opponent)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}: winner {info.get('winner')}")
        if checkpoint and (ep + 1) % 50 == 0:
            with open(checkpoint, "wb") as f:
                import pickle
                pickle.dump({"policy": ppo.policy, "value": ppo.value}, f)
    if checkpoint:
        with open(checkpoint, "wb") as f:
            import pickle
            pickle.dump({"policy": ppo.policy, "value": ppo.value}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for Otrio")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    train(args.episodes, args.checkpoint)
