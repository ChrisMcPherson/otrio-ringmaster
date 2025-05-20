import argparse
import sys
from pathlib import Path

# Ensure project root is on the Python path when the script is executed
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.random_agent import RandomAgent
from agents.tabular_q import TabularQAgent
from envs.otrio_env import OtrioEnv


def play_episode(env: OtrioEnv, agents):
    obs, info = env.reset()
    done = False
    boards = [env.board]
    player = info["current_player"]
    while not done:
        agent = agents[player]
        prev_board_state = env.board.clone()
        action = agent.select_action(prev_board_state, player)
        obs, reward, done, info = env.step(action)
        next_board_state = env.board.clone()
        next_player = env.current_player
        if isinstance(agent, TabularQAgent):
            agent.update(
                prev_board_state,
                player,
                action,
                reward,
                next_board_state,
                next_player,
                done,
            )
        player = next_player
    return info


def train(num_episodes: int = 1000, checkpoint: str | None = None, load: str | None = None):
    env = OtrioEnv(players=2)
    q_agent = TabularQAgent()
    if load:
        q_agent.load(load)
    random_agent = RandomAgent()
    agents = [q_agent, random_agent]
    for episode in range(num_episodes):
        info = play_episode(env, agents)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: winner {info.get('winner')}")
        if checkpoint and (episode + 1) % 100 == 0:
            q_agent.save(checkpoint)
    if checkpoint:
        q_agent.save(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning agent for Otrio")
    parser.add_argument("--episodes", type=int, default=1000, help="number of episodes to train")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to save agent state")
    parser.add_argument("--load", type=str, default=None, help="path to load agent state")
    args = parser.parse_args()
    train(args.episodes, args.checkpoint, args.load)

