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
        if isinstance(agent, TabularQAgent):
            agent.update(prev_board_state, player, action, reward, next_board_state, done)
        player = info.get("current_player", player)
    return info


def train(num_episodes: int = 1000):
    env = OtrioEnv(players=2)
    q_agent = TabularQAgent()
    random_agent = RandomAgent()
    agents = [q_agent, random_agent]
    for episode in range(num_episodes):
        info = play_episode(env, agents)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: winner {info.get('winner')}")


if __name__ == "__main__":
    train()
