# SOVEREIGN — Strategic Simulation for Deep Reinforcement Learning

SOVEREIGN is a simplified Gymnasium-compatible RL environment that models a
three-nation geopolitical conflict. A militarily superior **Invader** (the RL
agent) must decide how to pursue its objectives through joint
**political–military** actions, while a rule-based **Defender** resists and a
stochastic **Neutral** nation shifts its alignment based on Invader behaviour.

> **Core research question:** Can a militarily superior agent learn, through
> experience alone, that invasion is a strategically dominated strategy?

---

## File Structure

```
sovereign_project/
├── main.py              # Random-action demo
├── train.py             # DQN training + evaluation
├── dqn_agent.py         # DQN agent (Q-net, replay buffer, ε-greedy) from scratch
├── sovereign_env.py     # Gymnasium environment (SovereignEnv)
├── game_logic.py        # Turn structure, action decoding, state updates
├── reward.py            # Reward calculation (positive & negative terms)
├── map.py               # Map / territory setup (9 territories)
├── config.py            # All constants, weights, and thresholds
├── defender.py          # Rule-based Defender behaviour
├── neutral.py           # Neutral posture drift-diffusion model
├── requirements.txt     # Minimal dependencies
└── README.md            # This file
```

---

## Installation

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Random-action demo (environment sanity check)

```bash
python main.py
```

Runs a few episodes with random actions and prints per-step info — mainly
useful for verifying the environment mechanics.

### 2. Train the DQN agent

```bash
python train.py --episodes 300
```

Logs per-episode reward, runs a greedy evaluation every 25 episodes, and
saves the trained weights to `dqn_sovereign.pt`.

### 3. Evaluate a saved agent

```bash
python train.py --eval --demo
```

## RL Algorithm: DQN (from scratch)

`dqn_agent.py` implements a standard **Deep Q-Network**:

- **Q-network:** 2-layer MLP (`16 → 128 → 128 → 20`) mapping observations
  to Q-values over the 20 joint actions.
- **Target network:** separate frozen copy, synced every 500 gradient steps.
- **Replay buffer:** 50 000 transitions, batch size 64.
- **ε-greedy exploration:** linear decay from 1.0 → 0.05 over 20 000 steps.
- **Optimiser:** Adam, lr = 5e-4, gradient clipping at norm 1.0.
- **Loss:** MSE on the Bellman residual
  `r + γ · max_a' Q_target(s', a') · (1 − done)`.

All DQN hyperparameters live in `config.py` (`DQN_*` constants).

### Observed result

Within ~50 training episodes the agent reliably converges on the
**peace-dominant policy** predicted by the rulebook (§10): it repeatedly
chooses `NEGOTIATE`, triggering the negotiated-settlement terminal
(+40 reward). This demonstrates the PDF's central research claim — a
militarily superior agent *learns* that invasion is dominated.

---

## Action Space

The agent selects a single integer action in `Discrete(20)`, which encodes a
**joint political–military** decision:

| Political (5)     | Military (4) |
|--------------------|--------------|
| SEEK_ALLIANCE      | ADVANCE      |
| IMPOSE_SANCTION    | HOLD         |
| ISSUE_THREAT       | WITHDRAW     |
| NEGOTIATE          | STRIKE       |
| DO_NOTHING         |              |

Decoding: `action = pol_index × 4 + mil_index`

---

## State / Observation

The observation is a flat `float32` vector of length **16**:

| Index     | Variable              | Range      |
|-----------|-----------------------|------------|
| 0–8       | Territory control     | {0,1,2,3}  |
| 9         | Invader units (norm.) | [0, 1]     |
| 10        | Defender units (norm.)| [0, 1]     |
| 11        | Legitimacy L          | [0, 1]     |
| 12        | Economy E             | [0, 1]     |
| 13        | Neutral posture θ     | [-1, +1]   |
| 14        | Occupation duration   | [0, 1]     |
| 15        | Step progress         | [0, 1]     |

---

## Terminal Conditions

An episode ends when:

- **Legitimacy ≤ 0** — political collapse (reward: −50)
- **All Invader units destroyed** — military defeat (−30)
- **Negotiated settlement** — diplomatic resolution (+40)
- **Total conquest** — all territories captured (+10, intentionally modest)
- **Time limit** — 200 steps reached (0)

---

## Training API Example

```python
from sovereign_env import SovereignEnv
from dqn_agent   import DQNAgent

env   = SovereignEnv(seed=0)
obs, _ = env.reset()
agent = DQNAgent(obs_dim=obs.shape[0], n_actions=env.action_space.n, seed=0)

for episode in range(300):
    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs)
        next_obs, r, term, trunc, _ = env.step(action)
        agent.remember(obs, action, r, next_obs, term)
        agent.learn()
        obs, done = next_obs, term or trunc

agent.save("dqn_sovereign.pt")
```

---

## References

- Meta FAIR, *Human-level play in the game of Diplomacy* (Science, 2022)
- CS-272 Reinforcement Learning course project
