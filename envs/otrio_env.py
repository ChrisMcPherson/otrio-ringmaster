from typing import Tuple, Dict, Any
import random

from otrio.board import Board
from otrio.pieces import Size
from otrio.utils import BOARD_SIZE


class OtrioEnv:
    """Simple turn-based environment for the Otrio board game."""

    def __init__(self, players: int = 2):
        self.players = players
        self.board = Board(players)
        self.current_player = random.randrange(players)
        self.done = False

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        self.board.reset()
        self.current_player = random.randrange(self.players)
        self.done = False
        return self.board.to_observation(), {"current_player": self.current_player}

    def step(self, action: Tuple[int, int]) -> Tuple[Any, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done")
        well, size_id = action
        row, col = divmod(well, BOARD_SIZE)
        size = Size(size_id)
        info = {"current_player": self.current_player}
        reward = 0.0
        illegal = not self.board.is_legal(self.current_player, row, col, size)
        if illegal:
            reward = -0.05
            info["illegal_move"] = True
            obs = self.board.to_observation()
            return obs, reward, False, info
        opponent = (self.current_player + 1) % self.players
        before = self.board.num_winning_moves(opponent)
        self.board.apply_move(self.current_player, row, col, size)
        after = self.board.num_winning_moves(opponent)
        if self.board.check_win(self.current_player):
            reward = 1.0
            if after < before:
                reward += 0.1 * (before - after)
            self.done = True
            info["winner"] = self.current_player
        elif self.board.is_full():
            self.done = True
        else:
            if after < before:
                reward += 0.1 * (before - after)
            self.current_player = (self.current_player + 1) % self.players
        obs = self.board.to_observation()
        return obs, reward, self.done, info
