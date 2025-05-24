import argparse
import sys
from pathlib import Path
from collections import deque

# Extra imports for snapshot pool
import copy
import random
import time
import torch

# ──────────────────────────────────────────────────────────────
# PPO training constants
# ──────────────────────────────────────────────────────────────
TIMESTEPS_PER_UPDATE = 2048      # number of env‑steps to collect before each learner.update

# Snapshot pool parameters (for learner‑vs‑frozen‑snapshot stage)
SNAPSHOT_EP_INTERVAL = 1000    # add learner weights to pool every N episodes
POOL_SIZE            = 10      # keep only the latest N snapshots

from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.deep.ppo import PPOAgent, Step
from agents.tabular_q import TabularQAgent
from envs.otrio_env import OtrioEnv


def play_episode(env: OtrioEnv, learner: PPOAgent, opponent):
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
            opp_action = opponent.select_action(prev_board, player)
            action = opp_action[0] if isinstance(opponent, PPOAgent) else opp_action
            obs, reward, done, info = env.step(action)
            # Only the tabular‑Q opponent learns online.
            if isinstance(opponent, TabularQAgent):
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
    # Compute returns and advantages with GAE
    next_value = 0.0
    gae = 0.0
    for step in reversed(steps):
        delta = step.reward + learner.gamma * next_value - step.value
        gae = delta + learner.gamma * learner.gae_lambda * gae
        step.adv = gae
        step.ret = step.adv + step.value
        next_value = step.value
    return steps, info


def train(
    episodes: int = 1000,
    checkpoint: str | None = None,
    load: str | None = None,
    stage: str = "tabq",           # "tabq" or "pool"
    architecture: str = "mlp",
):
    env = OtrioEnv(players=2)
    learner = PPOAgent(architecture=architecture)

    if stage == "tabq":
        opponent = TabularQAgent()
        opponent_pool = None
    elif stage == "pool":
        opponent = PPOAgent(architecture=architecture)  # frozen opponent (weights replaced each episode)
        opponent_pool: list[tuple[str, dict[str, torch.Tensor]]] = []
        # prime the pool with the learner's initial weights
        init_id = f"init_{int(time.time())}"
        opponent_pool.append((init_id, copy.deepcopy(learner.state_dict())))
        print(f"Added snapshot {init_id} to opponent pool")
    else:
        raise ValueError(f"Unsupported stage: {stage!r}")

    if load:
        learner.load(load)

    writer = SummaryWriter()
    recent_results: deque[int] = deque(maxlen=50)
    batch: list[Step] = []
    timesteps_collected = 0

    for ep in range(1, episodes + 1):
        # --- choose opponent weights if we are in snapshot‑pool mode -------------
        if stage == "pool" and opponent_pool:
            snap_id, snapshot = random.choice(opponent_pool)
            opponent.load_state_dict(snapshot)
            opponent.model.eval()
            print(f"[Episode {ep}] Loaded opponent snapshot {snap_id}")

        steps, info = play_episode(env, learner, opponent)
        batch.extend(steps)
        timesteps_collected += len(steps)

        win = 1 if info.get("winner") == 0 else 0
        recent_results.append(win)
        episode_length = len(steps)
        win_rate = sum(recent_results) / len(recent_results)

        writer.add_scalar("win", win, ep)
        writer.add_scalar("episode_length", episode_length, ep)
        writer.add_scalar("win_rate_recent", win_rate, ep)

        if timesteps_collected >= TIMESTEPS_PER_UPDATE:
            # --- normalise advantages across the collected batch ------------------
            advs = torch.tensor([s.adv for s in batch], dtype=torch.float32)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            for s, a in zip(batch, advs):
                s.adv = a.item()

            # ----------------------------------------------------------------------
            metrics = learner.update(batch)          # returns dict of training stats
            timesteps_collected = 0
            batch.clear()

            # extra TensorBoard logging
            if metrics is not None:
                writer.add_scalar("policy_loss", metrics.get("policy_loss", 0.0), ep)
                writer.add_scalar("value_loss",  metrics.get("value_loss", 0.0), ep)
                writer.add_scalar("entropy",     metrics.get("entropy",     0.0), ep)
                writer.add_scalar("approx_kl",   metrics.get("approx_kl",  0.0), ep)

            # -- periodically snapshot learner weights into the opponent pool -----
            if stage == "pool" and (ep % SNAPSHOT_EP_INTERVAL == 0):
                snap_id = f"ep{ep}_{int(time.time())}"
                opponent_pool.append((snap_id, copy.deepcopy(learner.state_dict())))
                print(f"Added snapshot {snap_id} to opponent pool (size {len(opponent_pool)})")
                if len(opponent_pool) > POOL_SIZE:
                    removed_id, _ = opponent_pool.pop(0)
                    print(f"Removed snapshot {removed_id} from opponent pool")

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
    parser.add_argument(
        "--stage",
        type=str,
        default="tabq",
        choices=["tabq", "pool"],
        help="Training stage: tabq (learner vs tabular-Q) or pool (learner vs frozen snapshot pool)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="mlp",
        choices=["mlp", "mlp2", "conv"],
        help="Neural network architecture: mlp, mlp2, or conv",
    )
    args = parser.parse_args()
    train(args.episodes, args.checkpoint, args.load, args.stage, args.arch)
