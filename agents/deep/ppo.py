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

    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.95,
        clip_eps: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.02,
        max_grad_norm: float = 0.5,
        hidden: int = 64,
        architecture: str = "mlp",
    ):
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        # two players × three sizes × 3x3 board plus
        # current-player indicator and win-move counts for each player
        self.input_dim = len(SIZES) * BOARD_SIZE * BOARD_SIZE * 2 + 3
        self.action_dim = BOARD_SIZE * BOARD_SIZE * len(SIZES)

        self.architecture = architecture
        if architecture == "mlp":
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, hidden),
                nn.Tanh(),
            )
        elif architecture == "mlp2":
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
        elif architecture == "conv":
            channels = len(SIZES) * 2
            self.conv = nn.Sequential(
                nn.Conv2d(channels, 16, kernel_size=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=2),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, channels, BOARD_SIZE, BOARD_SIZE)
                conv_out = self.conv(dummy).view(1, -1).size(1)
            self.model = nn.Sequential(
                nn.Linear(conv_out + 3, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.policy_head = nn.Linear(hidden, self.action_dim)
        self.value_head = nn.Linear(hidden, 1)

        params = (
            list(self.model.parameters())
            + list(self.policy_head.parameters())
            + list(self.value_head.parameters())
        )
        if architecture == "conv":
            params += list(self.conv.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    # utilities -----------------------------------------------------------
    def _num_winning_moves(self, board: Board, player: int) -> int:
        """Return how many legal moves result in an immediate win."""
        count = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for s in SIZES:
                    if board.is_legal(player, r, c, s):
                        board.grid[r][c][s] = player
                        if board.check_win(player):
                            count += 1
                        board.grid[r][c][s] = None
        return count

    def _encode(self, board: Board, player: int) -> List[float]:
        obs = board.to_observation()
        vec: List[float] = []
        for p in range(2):
            for s in SIZES:
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_SIZE):
                        vec.append(float(obs[p][s][r][c]))
        # add win-move counts for each player
        vec.append(float(self._num_winning_moves(board, 0)))
        vec.append(float(self._num_winning_moves(board, 1)))
        # indicate whose turn it is
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
        if self.architecture == "conv":
            board_part = obs[:, :-3]
            extra = obs[:, -3:]
            bsz = obs.size(0)
            planes = board_part.view(
                bsz, len(SIZES) * 2, BOARD_SIZE, BOARD_SIZE
            )
            conv_feat = self.conv(planes)
            hidden = self.model(torch.cat([conv_feat, extra], dim=1))
        else:
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

        # PPO surrogate objective
        policy_loss = -(torch.min(ratio * advs, clip_ratio * advs)).mean()
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus to encourage exploration
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()

        # Approximate KL divergence between old and new policy
        approx_kl = (old_logp - selected_logp).mean()

        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        return loss, policy_loss, value_loss, entropy, approx_kl

    def update(self, steps: List[Step], epochs: int = 4, batch_size: int = 512):
        obs = torch.tensor([s.obs for s in steps], dtype=torch.float32)
        masks = torch.tensor([s.mask for s in steps], dtype=torch.bool)
        actions = torch.tensor([s.action for s in steps], dtype=torch.long)
        old_logp = torch.tensor([s.logp for s in steps], dtype=torch.float32)
        returns = torch.tensor([s.ret for s in steps], dtype=torch.float32)
        advs = torch.tensor([s.adv for s in steps], dtype=torch.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        losses = []
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        for _ in range(epochs):
            perm = torch.randperm(len(steps))
            for start in range(0, len(steps), batch_size):
                idx = perm[start:start + batch_size]
                loss, pol, val, ent, kl = self._compute_loss(
                    obs[idx],
                    masks[idx],
                    actions[idx],
                    old_logp[idx],
                    returns[idx],
                    advs[idx],
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters())
                    + ([p for p in self.conv.parameters()] if hasattr(self, "conv") else [])
                    + list(self.policy_head.parameters())
                    + list(self.value_head.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                losses.append(loss.item())
                policy_losses.append(pol.item())
                value_losses.append(val.item())
                entropies.append(ent.item())
                approx_kls.append(kl.item())

        return {
            "loss": float(sum(losses) / len(losses)),
            "policy_loss": float(sum(policy_losses) / len(policy_losses)),
            "value_loss": float(sum(value_losses) / len(value_losses)),
            "entropy": float(sum(entropies) / len(entropies)),
            "approx_kl": float(sum(approx_kls) / len(approx_kls)),
        }

    def save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "conv": self.conv.state_dict() if hasattr(self, "conv") else None,
                "policy_head": self.policy_head.state_dict(),
                "value_head": self.value_head.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.model.load_state_dict(data["model"])
        if data.get("conv") is not None and hasattr(self, "conv"):
            self.conv.load_state_dict(data["conv"])
        self.policy_head.load_state_dict(data["policy_head"])
        self.value_head.load_state_dict(data["value_head"])

    def state_dict(self) -> dict:
        """Return a state dict containing the model and head weights."""
        return {
            "model": self.model.state_dict(),
            "conv": self.conv.state_dict() if hasattr(self, "conv") else None,
            "policy_head": self.policy_head.state_dict(),
            "value_head": self.value_head.state_dict(),
        }

    def load_state_dict(self, state: dict):
        """Load weights from a state dict produced by :meth:`state_dict`."""
        self.model.load_state_dict(state["model"])
        if state.get("conv") is not None and hasattr(self, "conv"):
            self.conv.load_state_dict(state["conv"])
        self.policy_head.load_state_dict(state["policy_head"])
        self.value_head.load_state_dict(state["value_head"])

