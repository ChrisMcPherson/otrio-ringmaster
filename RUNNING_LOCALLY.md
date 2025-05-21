# Running Otrio-RL Locally

## Setup

```bash
# clone and enter the repo
git clone https://github.com/your-org/otrio-rl.git
cd otrio-rl

# install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The project has no heavy dependencies so installation is quick. Tests can be run with `pytest`.

## Training the baseline agents

To train the tabular Q-learning agent for a small sanity check:

```bash
python scripts/train_tabular_q.py --episodes 200 --checkpoint q_agent.pkl
```

To try the lightweight PPO implementation:

```bash
python scripts/train_ppo.py --episodes 200 --checkpoint ppo_agent.pkl --stage tabq

# continue training from a checkpoint against a snapshot pool of yourself
python scripts/train_ppo.py --episodes 200 --stage pool \
    --load checkpoints/after_tabq.pt --checkpoint selfplay_ppo.pt
```

Both scripts print periodic win statistics and write checkpoints when the
`--checkpoint` argument is supplied.
