import argparse
import sys
from pathlib import Path

# Ensure project root is on the Python path when executed directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.random_agent import RandomAgent
from envs.otrio_env import OtrioEnv


def evaluate(episodes: int = 100):
    """Run random-vs-random episodes to inspect reward balance."""
    env = OtrioEnv(players=2)
    agents = [RandomAgent(), RandomAgent()]
    wins = [0, 0]
    draws = 0

    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        player = info["current_player"]
        while not done:
            agent = agents[player]
            board = env.board.clone()
            action = agent.select_action(board, player)
            obs, reward, done, info = env.step(action)
            player = env.current_player
        winner = info.get("winner")
        if winner is None:
            draws += 1
        else:
            wins[winner] += 1

    total = sum(wins) + draws
    print(f"Played {total} episodes")
    print(f"Player 0 wins: {wins[0]}")
    print(f"Player 1 wins: {wins[1]}")
    print(f"Draws: {draws}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate reward balance with random agents")
    parser.add_argument("--episodes", type=int, default=100, help="number of episodes to run")
    args = parser.parse_args()
    evaluate(args.episodes)
