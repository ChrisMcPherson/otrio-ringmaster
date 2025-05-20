# Running Otrio-RL Locally

This short guide explains how to set up the environment and train the example Tabular Q-learning agent. The project currently includes a simple `RandomAgent` and a `TabularQAgent`.

## 1. Clone the repository

```bash
git clone https://github.com/your-org/otrio-rl.git
cd otrio-rl
```

## 2. Install dependencies

Only a minimal set of Python packages is required. Install them with:

```bash
python -m pip install -r requirements.txt
```

## 3. Run the unit tests (optional)

Verify the environment works:

```bash
pytest -q
```

## 4. Train the Tabular Q-learning agent

Execute the training script directly from the repository root:

```bash
python scripts/train_tabular_q.py --episodes 1000 --checkpoint q_agent.pkl
```

The script pits the `TabularQAgent` against a `RandomAgent` for the specified number of episodes. A checkpoint file is written if `--checkpoint` is provided.

## Notes

Advanced agents described in the README (such as PPO or AlphaZero-style models) are not yet implemented. Only the tabular Q-learning baseline and a random policy are available at the moment.

