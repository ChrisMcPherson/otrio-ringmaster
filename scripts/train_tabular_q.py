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


def play_episode(env: OtrioEnv, agents, learner_index: int = 0):
    """Play one episode and update the learner.

    The function also penalises the learner with a negative reward if the
    opponent wins on its move.
    """

    obs, info = env.reset()
    done = False
    player = info["current_player"]

    last_state = [None for _ in agents]
    last_action = [None for _ in agents]

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

        last_state[player] = prev_board_state
        last_action[player] = action
        player = next_player

    # If the learner lost, give a -1 reward for its previous move
    winner = info.get("winner")
    if winner is not None and winner != learner_index:
        l_state = last_state[learner_index]
        l_action = last_action[learner_index]
        if l_state is not None and l_action is not None:
            learner_agent = agents[learner_index]
            if isinstance(learner_agent, TabularQAgent):
                learner_agent.update(
                    l_state,
                    learner_index,
                    l_action,
                    -1.0,
                    next_board_state,
                    learner_index,
                    True,
                )

    return info


def train(
    num_episodes: int = 1000,
    checkpoint: str | None = None,
    load: str | None = None,
    log_interval: int = 100,
):
    """Train the tabular Q agent against a random opponent."""

    env = OtrioEnv(players=2)
    q_agent = TabularQAgent()
    if load:
        q_agent.load(load)
    random_agent = RandomAgent()
    agents = [q_agent, random_agent]

    win_counts = [0, 0]
    draws = 0

    for episode in range(num_episodes):
        info = play_episode(env, agents, learner_index=0)

        winner = info.get("winner")
        if winner is None:
            draws += 1
        else:
            win_counts[winner] += 1

        if (episode + 1) % log_interval == 0:
            total = sum(win_counts) + draws
            q_wins = win_counts[0]
            print(
                f"Episode {episode + 1}: Q wins {q_wins}/{total}, "
                f"Random wins {win_counts[1]}, draws {draws}"
            )
            if checkpoint:
                q_agent.save(checkpoint)

    if checkpoint:
        q_agent.save(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning agent for Otrio")
    parser.add_argument("--episodes", type=int, default=1000, help="number of episodes to train")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to save agent state")
    parser.add_argument("--load", type=str, default=None, help="path to load agent state")
    parser.add_argument("--log-interval", type=int, default=100, help="episodes between progress logs")
    args = parser.parse_args()
    train(args.episodes, args.checkpoint, args.load, args.log_interval)

