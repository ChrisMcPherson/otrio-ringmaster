from envs.otrio_env import OtrioEnv
from agents.random_agent import RandomAgent


def test_environment_playthrough():
    env = OtrioEnv(players=2)
    agent = RandomAgent()
    obs, info = env.reset()
    done = False
    moves = 0
    while not done and moves < 100:
        player = info["current_player"]
        action = agent.select_action(env.board, player)
        obs, reward, done, info = env.step(action)
        moves += 1
    assert done
