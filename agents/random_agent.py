import random
from typing import Tuple

from otrio.board import Board
from otrio.pieces import Size
from otrio.utils import BOARD_SIZE


class RandomAgent:
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
        return random.choice(moves)
