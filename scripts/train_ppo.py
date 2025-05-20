import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.deep.ppo import PPOAgent, Step
from agents.random_agent import RandomAgent
from envs.otrio_env import OtrioEnv


def play_episode(env: OtrioEnv, learner: PPOAgent, opponent: RandomAgent):
    obs, info = env.reset()
    player = info["current_player"]
    steps: list[Step] = []
    last_agent_step: Step | None = None
    done = False
    while not done:
        if player == 0:
            action, step = learner.select_action(env.board.clone(), player)
            well, size = action
            obs, reward, done, info = env.step((well, size))
            step.reward = reward
            steps.append(step)
            last_agent_step = step
            player = env.current_player
        else:
            action = opponent.select_action(env.board, player)
            obs, reward, done, info = env.step(action)
            if done and info.get("winner") == player and last_agent_step is not None:
                last_agent_step.reward = -1.0
            player = env.current_player
    # Compute returns and advantages
    R = 0.0
    for step in reversed(steps):
        R = step.reward + learner.gamma * R
        step.ret = R
        step.adv = step.ret - step.value
    return steps, info


def train(episodes: int = 1000, checkpoint: str | None = None, load: str | None = None):
    env = OtrioEnv(players=2)
    learner = PPOAgent()
    opponent = RandomAgent()
    if load:
        learner.load(load)
    win_count = 0
    batch: list[Step] = []
    for ep in range(1, episodes + 1):
        steps, info = play_episode(env, learner, opponent)
        batch.extend(steps)
        if info.get("winner") == 0:
            win_count += 1
        if ep % 10 == 0:
            learner.update(batch)
            batch.clear()
        if ep % 50 == 0:
            print(f"Episode {ep}: win rate {win_count}/{ep}")
            if checkpoint:
                learner.save(checkpoint)
    if checkpoint:
        learner.save(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent against a random opponent")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()
    train(args.episodes, args.checkpoint, args.load)

