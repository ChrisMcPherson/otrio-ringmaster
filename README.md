# otrio‑rl

A reference **reinforcement‑learning (RL) codebase** for training agents to master the board‑game **Otrio**.  
The project supplies:

* a pure‑Python implementation of Otrio’s rules  
* a Gymnasium‑compatible environment with legal‑move masking  
* baseline agents (random, heuristic, tabular Q, PPO)
* scripts for large‑scale self‑play, evaluation and curriculum learning

---

## 1  Game overview

**Otrio** is played on a 3 × 3 grid of “wells”. Each well can hold up to **three concentric rings**:

* **S** – small  
* **M** – medium  
* **L** – large  

Each player has **nine pieces** (3 × S,M,L) in their color.

### Official rules encoded here

1. **Move** – Choose an unused piece of your color and place it in any well **not already containing that size & color**. Items never move once placed.  
2. **Win** – A player wins immediately upon satisfying *any* of:

   * three identical size & color in a straight line (row/column/diagonal)  
   * a straight line of your color in ascending _or_ descending size (S→M→L or L→M→S)  
   * all three sizes of your color stacked in a single well  

3. **Episode end** – On the first win (winner gets +1, others −1) or a full board (draw = 0).

The base environment targets two‑player play, but N ≤ 4 players are supported.

---

## 2  Environment API (Gymnasium)

```python
import gymnasium as gym
from otrio.envs import OtrioEnv

env = OtrioEnv(players=2)          # Discrete(9) × Discrete(3) actions
obs, info = env.reset()
obs, reward, terminated, _, info = env.step((well_id, size_id))
```

### Observation space  
`Box(0,1, shape=(players, sizes, 3, 3), dtype=uint8)` – binary planes indicating ownership of each ring.

### Action space  
`Tuple(Discrete(9), Discrete(3))` → *(well index 0–8, size 0–2)*.  
A boolean `legal_moves` mask in `info` supports action‑masking layers.

### Reward scheme  

| case | winner | losers | draw |
|------|--------|--------|------|
| reward | `+1` | `−1` | `0` |

Illegal actions return `(obs, −0.05, False, False, info)`.

---

## 3  Repository layout

```
otrio-rl/
├── otrio/                 # core game logic (no ML deps)
│   ├── board.py           # Board, move legality, win detection
│   ├── pieces.py          # enums & helpers
│   └── utils.py
├── envs/                  # Gym wrappers
│   └── otrio_env.py
├── agents/
│   ├── random_agent.py
│   ├── rule_based.py
│   ├── tabular_q.py
│   └── deep/
│       ├── networks.py
│       └── ppo.py
├── training/
│   ├── self_play.py
│   ├── loop.py
│   └── curricula.py
├── evaluation/
│   ├── tournament.py
│   └── metrics.py
├── scripts/
│   ├── train_ppo.py
│   ├── train_alpha_zero.py
│   └── benchmark.sh
├── tests/
├── configs/
└── README.md   ← you are here
```

Design principles:

* **pure‑Python core** for fast unit‑testing  
* **vectorised self‑play** via `gym.vector` or Ray to saturate GPUs  
* **action masking** (`MaskedCategorical`) to respect legal‑move sets  
* **curriculum** – random opponent → self‑play → multi‑player variants  
* **reproducibility** – seed everything, checkpoint nets & replay buffers

---

## 4  Getting started

```bash
# clone & install
git clone https://github.com/your‑org/otrio‑rl.git
cd otrio‑rl
pip install -e ".[dev]"

# quick sanity check
pytest

# train a PPO agent
python scripts/train_ppo.py --episodes 1000 --checkpoint ppo_agent.pkl

# or quickly try the tabular Q-learning baseline
python scripts/train_tabular_q.py --episodes 1000 --checkpoint q_agent.pkl
```

Logs (TensorBoard & CSV) land in `./runs/`, checkpoints in `./checkpoints/`.

---

## 5  Research extensions

* **Otrio‑Extreme** – forbid ascending‑size wins  
* **Piece‑draft variant** – players draft color‑agnostic rings → imperfect‑info RL  
* **MuZero port** – swap PPO for a learned model & MCTS planner

---

## License

Apache 2.0. See `LICENSE` for details.

Enjoy experimenting — pull requests welcome!
