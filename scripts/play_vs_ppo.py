import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.deep.ppo import PPOAgent
from envs.otrio_env import OtrioEnv
from otrio.pieces import Size
from otrio.utils import BOARD_SIZE, SIZES


def render(board):
    for r in range(BOARD_SIZE):
        cells = []
        for c in range(BOARD_SIZE):
            cell = board.grid[r][c]
            cells.append(''.join(str(cell[s]) if cell[s] is not None else '.' for s in SIZES))
        print(' '.join(cells))
    print()


def prompt_move(board, player):
    while True:
        try:
            well = int(input("Select well (0-8): "))
            size = int(input("Select size (0:small 1:medium 2:large): "))
            row, col = divmod(well, BOARD_SIZE)
            s = Size(size)
            if board.is_legal(player, row, col, s):
                return well, size
            print("Illegal move, try again.")
        except Exception:
            print("Invalid input, try again.")


def play(model_path: str, human_player: int = 1):
    env = OtrioEnv(players=2)
    agent = PPOAgent()
    agent.load(model_path)

    obs, info = env.reset()
    player = info["current_player"]
    done = False

    while not done:
        render(env.board)
        if player == human_player:
            well, size = prompt_move(env.board, player)
        else:
            (well, size), _ = agent.select_action(env.board.clone(), player)
        obs, reward, done, info = env.step((well, size))
        player = env.current_player

    render(env.board)
    winner = info.get("winner")
    if winner is None:
        print("Game ended in a draw.")
    elif winner == human_player:
        print("You win!")
    else:
        print("Agent wins!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against a trained PPO agent")
    parser.add_argument("--model", type=str, required=True, help="path to the saved model")
    parser.add_argument("--human-player", type=int, default=1, choices=[0, 1], help="0 to play first, 1 to play second")
    args = parser.parse_args()
    play(args.model, args.human_player)
