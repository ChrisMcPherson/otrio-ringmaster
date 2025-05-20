# Running Otrio-RL Locally

This short guide shows how to set up the project, run the tests and train the example tabular Q‑learning agent. The deeper RL agents mentioned in the README (PPO, AlphaZero-style) are not yet implemented, so only the baseline components are available.

## Setup

```bash
# clone the repository
git clone <repo-url>
cd otrio-ringmaster

# install dependencies (only pytest is required)
pip install -r requirements.txt
```

## Running the tests

Execute the unit tests to verify the environment works:

```bash
pytest -q
```

## Training the tabular Q agent

You can train the provided baseline agent against a random opponent:

```bash
python scripts/train_tabular_q.py --episodes 1000 --checkpoint q_agent.pkl
```

Progress is printed every 100 episodes by default. The checkpoint path is optional, but allows you to resume training later via `--load q_agent.pkl`.

## Evaluating the reward scheme

A helper script `scripts/evaluate_reward_balance.py` runs two random agents and reports the distribution of wins and draws. This can be useful for verifying that the reward settings are symmetrical and working as expected:

```bash
python scripts/evaluate_reward_balance.py --episodes 100
```
