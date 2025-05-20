import math
import random
from typing import List, Tuple

from otrio.board import Board
from otrio.pieces import Size
from otrio.utils import BOARD_SIZE


# Utility ---------------------------------------------------------------

def flatten_board(board: Board) -> List[float]:
    """Return a flat vector representation of the board."""
    vec: List[float] = []
    for player in range(board.players):
        for size in Size:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    val = board.grid[r][c][size]
                    vec.append(1.0 if val == player else 0.0)
    return vec


# Simple neural network components -------------------------------------

class LinearLayer:
    def __init__(self, in_dim: int, out_dim: int):
        self.w = [[random.uniform(-0.1, 0.1) for _ in range(in_dim)] for _ in range(out_dim)]
        self.b = [0.0 for _ in range(out_dim)]
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: List[float]) -> List[float]:
        out = []
        for i in range(self.out_dim):
            s = self.b[i]
            for j in range(self.in_dim):
                s += self.w[i][j] * x[j]
            out.append(s)
        return out

    def backward(self, x: List[float], grad_out: List[float], lr: float) -> List[float]:
        grad_in = [0.0 for _ in range(self.in_dim)]
        for i in range(self.out_dim):
            g = grad_out[i]
            for j in range(self.in_dim):
                grad_in[j] += g * self.w[i][j]
                self.w[i][j] -= lr * g * x[j]
            self.b[i] -= lr * g
        return grad_in


def relu(x: List[float]) -> List[float]:
    return [max(0.0, v) for v in x]

def relu_backward(x: List[float], grad: List[float]) -> List[float]:
    return [g if x[i] > 0 else 0.0 for i, g in enumerate(grad)]


def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


class MLP:
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        self.l1 = LinearLayer(in_dim, hidden)
        self.l2 = LinearLayer(hidden, out_dim)

    def forward(self, x: List[float]) -> Tuple[List[float], List[float]]:
        h = relu(self.l1.forward(x))
        o = self.l2.forward(h)
        return o, h

    def backward(self, x: List[float], h: List[float], grad_out: List[float], lr: float):
        grad_h = self.l2.backward(h, grad_out, lr)
        grad_h = relu_backward(h, grad_h)
        self.l1.backward(x, grad_h, lr)


# PPO Agent -------------------------------------------------------------

class PPOAgent:
    def __init__(self, lr: float = 1e-3, gamma: float = 0.99, clip_eps: float = 0.2):
        state_dim = len(flatten_board(Board(2)))
        self.policy = MLP(state_dim, 64, BOARD_SIZE * len(Size))
        self.value = MLP(state_dim, 64, 1)
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        # rollout storage
        self.states: List[List[float]] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def _action_index(self, well: int, size: int) -> int:
        return well * len(Size) + size

    def _index_to_action(self, index: int) -> Tuple[int, int]:
        well = index // len(Size)
        size = index % len(Size)
        return well, size

    def select_action(self, board: Board, player: int) -> Tuple[int, int]:
        state = flatten_board(board)
        logits, h = self.policy.forward(state)
        # mask illegal moves
        probs = softmax(logits)
        legal = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for s in Size:
                    if board.is_legal(player, r, c, s):
                        legal.append(self._action_index(r * BOARD_SIZE + c, int(s)))
        masked_probs = [probs[i] if i in legal else 0.0 for i in range(len(probs))]
        s = sum(masked_probs)
        if s == 0:
            raise RuntimeError("No legal moves")
        masked_probs = [p / s for p in masked_probs]
        # sample action
        r = random.random()
        cumulative = 0.0
        action_index = 0
        for i, p in enumerate(masked_probs):
            cumulative += p
            if r <= cumulative:
                action_index = i
                break
        log_prob = math.log(masked_probs[action_index])
        value, _ = self.value.forward(state)
        self.states.append(state)
        self.actions.append(action_index)
        self.log_probs.append(log_prob)
        self.values.append(value[0])
        return self._index_to_action(action_index)

    def store_outcome(self, reward: float, done: bool):
        self.rewards.append(reward)
        self.dones.append(done)

    def _compute_returns_and_advantages(self) -> Tuple[List[float], List[float]]:
        returns = [0.0 for _ in self.rewards]
        advs = [0.0 for _ in self.rewards]
        R = 0.0
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                R = 0.0
            R = self.rewards[i] + self.gamma * R
            returns[i] = R
        for i in range(len(returns)):
            advs[i] = returns[i] - self.values[i]
        # normalize advantages
        mean_adv = sum(advs) / len(advs)
        var_adv = sum((a - mean_adv) ** 2 for a in advs) / len(advs)
        std_adv = math.sqrt(var_adv) + 1e-8
        advs = [(a - mean_adv) / std_adv for a in advs]
        return returns, advs

    def update(self, epochs: int = 2):
        returns, advs = self._compute_returns_and_advantages()
        for _ in range(epochs):
            for i in range(len(self.states)):
                state = self.states[i]
                action = self.actions[i]
                old_log_prob = self.log_probs[i]
                ret = returns[i]
                adv = advs[i]
                # recompute
                logits, h_policy = self.policy.forward(state)
                probs = softmax(logits)
                log_prob = math.log(probs[action])
                ratio = math.exp(log_prob - old_log_prob)
                clipped_ratio = max(min(ratio, 1 + self.clip_eps), 1 - self.clip_eps)
                weight = -min(ratio * adv, clipped_ratio * adv)
                # policy gradient
                grad_logits = [p for p in probs]
                grad_logits[action] -= 1.0
                grad_logits = [g * weight for g in grad_logits]
                self.policy.backward(state, h_policy, grad_logits, self.lr)
                # value gradient
                val_pred, h_val = self.value.forward(state)
                grad_value = [(val_pred[0] - ret) * 2.0]
                self.value.backward(state, h_val, grad_value, self.lr)
        # clear memory
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
