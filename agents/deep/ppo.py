import math
import pickle
import random
from dataclasses import dataclass
from typing import List, Tuple

from otrio.board import Board
from otrio.pieces import Size
from otrio.utils import BOARD_SIZE, SIZES


@dataclass
class Step:
    obs: List[float]
    mask: List[int]
    action: int
    logp: float
    value: float
    reward: float
    ret: float = 0.0
    adv: float = 0.0


class PPOAgent:
    """Minimal PPO agent implemented without external dependencies."""

    def __init__(self, lr: float = 0.01, gamma: float = 0.99, clip_eps: float = 0.2, hidden: int = 64):
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.input_dim = len(SIZES) * BOARD_SIZE * BOARD_SIZE * 2 + 1
        self.action_dim = BOARD_SIZE * BOARD_SIZE * len(SIZES)
        rnd = random.Random()
        self.w1 = [[rnd.uniform(-1, 1) / math.sqrt(self.input_dim) for _ in range(hidden)] for _ in range(self.input_dim)]
        self.b1 = [0.0 for _ in range(hidden)]
        self.w2 = [[rnd.uniform(-1, 1) / math.sqrt(hidden) for _ in range(self.action_dim)] for _ in range(hidden)]
        self.b2 = [0.0 for _ in range(self.action_dim)]
        self.wv = [rnd.uniform(-1, 1) / math.sqrt(hidden) for _ in range(hidden)]
        self.bv = 0.0

    # utilities -----------------------------------------------------------
    def _encode(self, board: Board, player: int) -> List[float]:
        obs = board.to_observation()
        vec: List[float] = []
        for p in range(2):
            for s in SIZES:
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_SIZE):
                        vec.append(float(obs[p][s][r][c]))
        vec.append(float(player))
        return vec

    def _legal_mask(self, board: Board, player: int) -> List[int]:
        mask = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for s in SIZES:
                    mask.append(1 if board.is_legal(player, r, c, s) else 0)
        return mask

    def _forward(self, obs: List[float]):
        h = []
        for j in range(len(self.w1[0])):
            v = self.b1[j]
            for i, x in enumerate(obs):
                v += x * self.w1[i][j]
            h.append(math.tanh(v))
        logits = []
        for k in range(self.action_dim):
            v = self.b2[k]
            for j, hv in enumerate(h):
                v += hv * self.w2[j][k]
            logits.append(v)
        value = self.bv + sum(h[j] * self.wv[j] for j in range(len(h)))
        return h, logits, value

    def _log_softmax(self, logits: List[float], mask: List[int]) -> List[float]:
        masked = [l if m else -1e9 for l, m in zip(logits, mask)]
        max_logit = max(masked)
        exps = [math.exp(l - max_logit) for l in masked]
        denom = sum(exps)
        return [l - (max_logit + math.log(denom)) for l in masked]

    def select_action(self, board: Board, player: int) -> Tuple[Tuple[int, int], Step]:
        obs = self._encode(board, player)
        mask = self._legal_mask(board, player)
        h, logits, value = self._forward(obs)
        log_probs = self._log_softmax(logits, mask)
        probs = [math.exp(lp) for lp in log_probs]
        # sample
        r = random.random()
        cum = 0.0
        action_idx = 0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                action_idx = i
                break
        step = Step(obs=obs, mask=mask, action=action_idx, logp=log_probs[action_idx], value=value, reward=0.0)
        well = action_idx // len(SIZES)
        size = action_idx % len(SIZES)
        return (well, size), step

    # gradient helpers ----------------------------------------------------
    def _compute_grad(self, step: Step):
        h, logits, value = self._forward(step.obs)
        log_probs = self._log_softmax(logits, step.mask)
        probs = [math.exp(lp) for lp in log_probs]
        logp = log_probs[step.action]
        ratio = math.exp(logp - step.logp)
        clip_ratio = max(1 - self.clip_eps, min(1 + self.clip_eps, ratio))
        use_clip = (ratio > clip_ratio and step.adv > 0) or (ratio < clip_ratio and step.adv < 0)
        grad_logp = 0.0 if use_clip else -step.adv * ratio
        # grad w2, b2
        dlogits = [grad_logp * ((1 if i == step.action else 0) - probs[i]) for i in range(self.action_dim)]
        gw2 = [[h[j] * dlogits[k] for k in range(self.action_dim)] for j in range(len(h))]
        gb2 = dlogits
        # value grads
        dv = value - step.ret
        gwv = [h[j] * dv for j in range(len(h))]
        gbv = dv
        # grad w1, b1
        dh = [sum(self.w2[j][k] * dlogits[k] for k in range(self.action_dim)) + self.wv[j] * dv for j in range(len(h))]
        dz1 = [dh[j] * (1 - h[j] * h[j]) for j in range(len(h))]
        gw1 = [[step.obs[i] * dz1[j] for j in range(len(h))] for i in range(len(step.obs))]
        gb1 = dz1
        return gw1, gb1, gw2, gb2, gwv, gbv

    def update(self, steps: List[Step], epochs: int = 3, batch_size: int = 8):
        for _ in range(epochs):
            random.shuffle(steps)
            for start in range(0, len(steps), batch_size):
                batch = steps[start:start + batch_size]
                gw1 = [[0.0 for _ in range(len(self.w1[0]))] for _ in range(self.input_dim)]
                gb1 = [0.0 for _ in range(len(self.w1[0]))]
                gw2 = [[0.0 for _ in range(self.action_dim)] for _ in range(len(self.w1[0]))]
                gb2 = [0.0 for _ in range(self.action_dim)]
                gwv = [0.0 for _ in range(len(self.w1[0]))]
                gbv = 0.0
                for step in batch:
                    g1, b1, g2, b2, gv, bv = self._compute_grad(step)
                    for i in range(self.input_dim):
                        for j in range(len(self.w1[0])):
                            gw1[i][j] += g1[i][j]
                    for j in range(len(self.w1[0])):
                        gb1[j] += b1[j]
                    for j in range(len(self.w1[0])):
                        for k in range(self.action_dim):
                            gw2[j][k] += g2[j][k]
                    for k in range(self.action_dim):
                        gb2[k] += b2[k]
                    for j in range(len(self.w1[0])):
                        gwv[j] += gv[j]
                    gbv += bv
                n = float(len(batch))
                for i in range(self.input_dim):
                    for j in range(len(self.w1[0])):
                        self.w1[i][j] -= self.lr * gw1[i][j] / n
                for j in range(len(self.w1[0])):
                    self.b1[j] -= self.lr * gb1[j] / n
                for j in range(len(self.w1[0])):
                    for k in range(self.action_dim):
                        self.w2[j][k] -= self.lr * gw2[j][k] / n
                for k in range(self.action_dim):
                    self.b2[k] -= self.lr * gb2[k] / n
                for j in range(len(self.w1[0])):
                    self.wv[j] -= self.lr * gwv[j] / n
                self.bv -= self.lr * gbv / n

    def save(self, path: str):
        data = {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2,
            'wv': self.wv,
            'bv': self.bv,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']
        self.wv = data['wv']
        self.bv = data['bv']

