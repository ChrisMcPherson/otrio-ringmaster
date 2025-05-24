import argparse
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.deep.ppo import PPOAgent
from envs.otrio_env import OtrioEnv
from otrio.pieces import Size
from otrio.utils import BOARD_SIZE, SIZES


def parse_args():
    parser = argparse.ArgumentParser(description="Play PPO agent with Streamlit UI")
    parser.add_argument("--model", type=str, required=True, help="path to saved model")
    parser.add_argument("--human-player", type=int, default=1, choices=[0, 1], help="0 to play first, 1 to play second")
    parser.add_argument(
        "--arch",
        type=str,
        default="mlp",
        choices=["mlp", "mlp2", "conv"],
        help="Architecture used when training the model",
    )
    return parser.parse_args()


args = parse_args()


def init_game():
    env = OtrioEnv(players=2)
    agent = PPOAgent(architecture=args.arch)
    agent.load(args.model)
    obs, info = env.reset()
    st.session_state.env = env
    st.session_state.agent = agent
    st.session_state.player = info["current_player"]
    st.session_state.done = False
    st.session_state.info = info
    st.session_state.human_player = args.human_player
    st.session_state.selected_size = 0


def agent_turn():
    env = st.session_state.env
    agent = st.session_state.agent
    player = st.session_state.player
    (well, size), _ = agent.select_action(env.board.clone(), player)
    _, _, done, info = env.step((well, size))
    st.session_state.done = done
    st.session_state.player = env.current_player
    st.session_state.info = info


if "env" not in st.session_state:
    init_game()

# automated agent move if it's not the human's turn
if not st.session_state.done and st.session_state.player != st.session_state.human_player:
    agent_turn()
    st.experimental_rerun()

board = st.session_state.env.board
st.title("Play Otrio vs PPO Agent")

st.sidebar.radio(
    "Select piece size",
    options=[0, 1, 2],
    format_func=lambda x: Size(x).name.title(),
    key="selected_size",
)
if st.sidebar.button("Reset Game"):
    init_game()
    st.experimental_rerun()

for r in range(BOARD_SIZE):
    cols = st.columns(BOARD_SIZE)
    for c in range(BOARD_SIZE):
        cell = board.grid[r][c]
        label = "".join(str(cell[s]) if cell[s] is not None else "." for s in SIZES)
        disabled = (
            st.session_state.done
            or st.session_state.player != st.session_state.human_player
            or not board.is_legal(st.session_state.player, r, c, Size(st.session_state.selected_size))
        )
        if cols[c].button(label or " ", key=f"{r}-{c}", disabled=disabled):
            _, _, done, info = st.session_state.env.step((r * BOARD_SIZE + c, st.session_state.selected_size))
            st.session_state.done = done
            st.session_state.player = st.session_state.env.current_player
            st.session_state.info = info
            if not done and st.session_state.player != st.session_state.human_player:
                agent_turn()
            st.experimental_rerun()

st.write("")

if st.session_state.done:
    winner = st.session_state.info.get("winner")
    if winner is None:
        st.success("Game ended in a draw.")
    elif winner == st.session_state.human_player:
        st.success("You win!")
    else:
        st.error("Agent wins!")
