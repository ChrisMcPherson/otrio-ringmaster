import random
import pickle
from collections import defaultdict
from typing import Tuple, Dict

from otrio.board import Board
from otrio.pieces import Size
from otrio.utils import BOARD_SIZE


class TabularQAgent:
    def __init__(self, alpha: float = 0.5, gamma: float = 0.9, epsilon: float = 0.1):
        self.q: Dict[str, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the Q-table to disk using pickle."""
        with open(path, "wb") as f:
            pickle.dump(dict(self.q), f)

    def load(self, path: str) -> None:
        """Load Q-table from ``path`` overwriting current values."""
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        self.q = defaultdict(lambda: defaultdict(float), {
            k: defaultdict(float, v) for k, v in loaded.items()
        })

    def _state_key(self, board: Board, player: int) -> str:
        # simple string representation
        marks = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for size in Size:
                    owner = board.grid[r][c][size]
                    marks.append('.' if owner is None else str(owner))
        marks.append(str(player))
        return ''.join(marks)

    def select_action(self, board: Board, player: int) -> Tuple[int, int]:
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for size in Size:
                    if board.is_legal(player, r, c, size):
                        well = r * BOARD_SIZE + c
                        moves.append((well, int(size)))
        if not moves:
            raise RuntimeError("No legal moves")
        if random.random() < self.epsilon:
            return random.choice(moves)
        key = self._state_key(board, player)
        q_values = self.q[key]
        best_move = max(moves, key=lambda m: q_values[m])
        return best_move

    def update(self, prev_board: Board, player: int, action: Tuple[int, int], reward: float, next_board: Board, done: bool):
        key = self._state_key(prev_board, player)
        next_key = self._state_key(next_board, (player + 1) % next_board.players)
        q_values = self.q[key]
        next_q_values = self.q[next_key]
        best_next = max(next_q_values.values(), default=0.0)
        q_old = q_values[action]
        target = reward + (0 if done else self.gamma * best_next)
        q_values[action] = q_old + self.alpha * (target - q_old)
