
"""Train a tabular Q-learning agent against a random opponent."""

import argparse
from agents.random_agent import RandomAgent
from agents.tabular_q import TabularQAgent
from envs.otrio_env import OtrioEnv


def play_episode(env: OtrioEnv, agents):
    obs, info = env.reset()
    done = False
    player = info["current_player"]
    while not done:
        agent = agents[player]
        prev_board_state = env.board.clone()
        action = agent.select_action(prev_board_state, player)
        obs, reward, done, info = env.step(action)
        next_board_state = env.board.clone()
        if isinstance(agent, TabularQAgent):
            agent.update(prev_board_state, player, action, reward, next_board_state, done)
        player = info.get("current_player", player)
    return info


def train(num_episodes: int = 1000, save_path: str | None = None, load_path: str | None = None):
    env = OtrioEnv(players=2)
    q_agent = TabularQAgent()
    if load_path:
        q_agent.load(load_path)
    random_agent = RandomAgent()
    agents = [q_agent, random_agent]
    win_counts = {0: 0, 1: 0, "draw": 0}
    for episode in range(num_episodes):
        info = play_episode(env, agents)
        if "winner" in info:
            win_counts[info["winner"]] += 1
        else:
            win_counts["draw"] += 1
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: {win_counts}")
    if save_path:
        q_agent.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning agent")
    parser.add_argument("--episodes", type=int, default=1000, help="number of training episodes")
    parser.add_argument("--save", type=str, default=None, help="path to save the trained agent")
    parser.add_argument("--load", type=str, default=None, help="path to a previously saved agent")
    args = parser.parse_args()
    train(num_episodes=args.episodes, save_path=args.save, load_path=args.load)
