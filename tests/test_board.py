from otrio.board import Board
from otrio.pieces import Size


def test_win_same_size_line():
    b = Board(players=2)
    # player 0 places small pieces in first row
    b.apply_move(0, 0, 0, Size.SMALL)
    b.apply_move(0, 0, 1, Size.SMALL)
    b.apply_move(0, 0, 2, Size.SMALL)
    assert b.check_win(0)
