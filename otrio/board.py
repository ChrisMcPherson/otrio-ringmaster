from typing import List, Optional, Tuple

from .pieces import Size
from .utils import BOARD_SIZE, SIZES, empty_board


class Board:
    def __init__(self, players: int = 2):
        self.players = players
        self.grid: List[List[List[Optional[int]]]] = empty_board(players)

    def reset(self):
        self.grid = empty_board(self.players)

    def clone(self) -> "Board":
        new_b = Board(self.players)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for s in SIZES:
                    new_b.grid[r][c][s] = self.grid[r][c][s]
        return new_b

    def is_legal(self, player: int, row: int, col: int, size: Size) -> bool:
        return self.grid[row][col][size] is None

    def apply_move(self, player: int, row: int, col: int, size: Size):
        if not self.is_legal(player, row, col, size):
            raise ValueError("Illegal move")
        self.grid[row][col][size] = player

    def is_full(self) -> bool:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                for size in SIZES:
                    if self.grid[row][col][size] is None:
                        return False
        return True

    def check_win(self, player: int) -> bool:
        lines = []
        # rows and columns
        for i in range(BOARD_SIZE):
            lines.append([(i, j) for j in range(BOARD_SIZE)])
            lines.append([(j, i) for j in range(BOARD_SIZE)])
        # diagonals
        lines.append([(i, i) for i in range(BOARD_SIZE)])
        lines.append([(i, BOARD_SIZE - 1 - i) for i in range(BOARD_SIZE)])

        # check same size line
        for size in SIZES:
            for line in lines:
                if all(self.grid[r][c][size] == player for r, c in line):
                    return True

        # check ascending/descending size lines
        asc = [Size.SMALL, Size.MEDIUM, Size.LARGE]
        desc = list(reversed(asc))
        for line in lines:
            if all(
                self.grid[r][c][asc[i]] == player for i, (r, c) in enumerate(line)
            ):
                return True
            if all(
                self.grid[r][c][desc[i]] == player for i, (r, c) in enumerate(line)
            ):
                return True

        # stacked three sizes in a well
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if all(self.grid[r][c][s] == player for s in SIZES):
                    return True

        return False

    def legal_moves(self, player: int) -> List[Tuple[int, int, Size]]:
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for s in SIZES:
                    if self.is_legal(player, r, c, s):
                        moves.append((r, c, s))
        return moves

    def to_observation(self) -> List[List[List[List[int]]]]:
        """Return observation planes: players x sizes x 3 x 3."""
        obs = [
            [
                [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
                for _ in SIZES
            ]
            for _ in range(self.players)
        ]
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                for s in SIZES:
                    owner = self.grid[r][c][s]
                    if owner is not None:
                        obs[owner][s][r][c] = 1
        return obs
