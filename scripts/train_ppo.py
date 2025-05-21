import argparse
import sys
from pathlib import Path
from collections import deque

from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.deep.ppo import PPOAgent, Step
from agents.tabular_q import TabularQAgent
from envs.otrio_env import OtrioEnv


def play_episode(env: OtrioEnv, learner: PPOAgent, opponent: TabularQAgent):
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
            prev_board = env.board.clone()
            action = opponent.select_action(prev_board, player)
            obs, reward, done, info = env.step(action)
            opponent.update(
                prev_board,
                player,
                action,
                reward,
                env.board.clone(),
                env.current_player,
                done,
            )
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
    opponent = TabularQAgent()
    if load:
        learner.load(load)

    writer = SummaryWriter()
    writer.add_text(
        "hparams",
        f"lr={learner.lr}, gamma={learner.gamma}, clip_eps={learner.clip_eps}",
        global_step=0,
    )

    recent_results: deque[int] = deque(maxlen=50)
    batch: list[Step] = []

    for ep in range(1, episodes + 1):
        steps, info = play_episode(env, learner, opponent)
        batch.extend(steps)

        win = 1 if info.get("winner") == 0 else 0
        recent_results.append(win)
        episode_length = len(steps)
        win_rate = sum(recent_results) / len(recent_results)

        writer.add_scalar("win", win, ep)
        writer.add_scalar("episode_length", episode_length, ep)
        writer.add_scalar("win_rate_recent", win_rate, ep)

        if ep % 10 == 0:
            metrics = learner.update(batch)
            writer.add_scalar("loss", metrics["loss"], ep)
            writer.add_scalar("policy_loss", metrics["policy_loss"], ep)
            writer.add_scalar("value_loss", metrics["value_loss"], ep)
            batch.clear()

        if ep % 50 == 0:
            print(f"Episode {ep}: last {len(recent_results)}-episode win rate {win_rate * 100:.1f}%")
            if checkpoint:
                learner.save(checkpoint)

    writer.close()
    if checkpoint:
        learner.save(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent against a tabular Q-learning opponent")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()
    train(args.episodes, args.checkpoint, args.load)

