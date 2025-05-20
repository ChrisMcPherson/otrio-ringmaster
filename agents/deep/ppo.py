import pickle
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

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
    """PPO agent implemented using PyTorch."""

    def __init__(self, lr: float = 1e-3, gamma: float = 0.99, clip_eps: float = 0.2, hidden: int = 64):
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.input_dim = len(SIZES) * BOARD_SIZE * BOARD_SIZE * 2 + 1
        self.action_dim = BOARD_SIZE * BOARD_SIZE * len(SIZES)

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, self.action_dim)
        self.value_head = nn.Linear(hidden, 1)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters()),
            lr=self.lr,
        )

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

    def _forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.model(obs)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    def _log_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = logits.masked_fill(~mask, -1e9)
        return F.log_softmax(logits, dim=-1)

    def select_action(self, board: Board, player: int) -> Tuple[Tuple[int, int], Step]:
        obs = self._encode(board, player)
        mask = self._legal_mask(board, player)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = self._forward(obs_t)
        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
        log_probs = self._log_softmax(logits, mask_t)
        probs = log_probs.exp().squeeze(0)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()
        step = Step(obs=obs, mask=mask, action=action_idx, logp=log_probs[0, action_idx].item(), value=value.item(), reward=0.0)
        well = action_idx // len(SIZES)
        size = action_idx % len(SIZES)
        return (well, size), step

    # gradient helpers ----------------------------------------------------
    def _compute_loss(self, obs, masks, actions, old_logp, returns, advs):
        logits, values = self._forward(obs)
        log_probs = self._log_softmax(logits, masks)
        selected_logp = log_probs[range(len(actions)), actions]
        ratio = torch.exp(selected_logp - old_logp)
        clip_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        policy_loss = -(torch.min(ratio * advs, clip_ratio * advs)).mean()
        value_loss = F.mse_loss(values, returns)
        return policy_loss + 0.5 * value_loss

    def update(self, steps: List[Step], epochs: int = 3, batch_size: int = 8):
        obs = torch.tensor([s.obs for s in steps], dtype=torch.float32)
        masks = torch.tensor([s.mask for s in steps], dtype=torch.bool)
        actions = torch.tensor([s.action for s in steps], dtype=torch.long)
        old_logp = torch.tensor([s.logp for s in steps], dtype=torch.float32)
        returns = torch.tensor([s.ret for s in steps], dtype=torch.float32)
        advs = torch.tensor([s.adv for s in steps], dtype=torch.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for _ in range(epochs):
            perm = torch.randperm(len(steps))
            for start in range(0, len(steps), batch_size):
                idx = perm[start:start + batch_size]
                loss = self._compute_loss(
                    obs[idx],
                    masks[idx],
                    actions[idx],
                    old_logp[idx],
                    returns[idx],
                    advs[idx],
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "policy_head": self.policy_head.state_dict(),
                "value_head": self.value_head.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.model.load_state_dict(data["model"])
        self.policy_head.load_state_dict(data["policy_head"])
        self.value_head.load_state_dict(data["value_head"])

