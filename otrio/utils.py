from typing import List, Optional

from .pieces import Size

BOARD_SIZE = 3  # 3x3 grid
SIZES = [Size.SMALL, Size.MEDIUM, Size.LARGE]


def empty_board(players: int) -> List[List[List[Optional[int]]]]:
    """Create an empty board: wells x sizes with None for empty."""
    return [[ [None for _ in SIZES] for _ in range(BOARD_SIZE) ] for _ in range(BOARD_SIZE)]
