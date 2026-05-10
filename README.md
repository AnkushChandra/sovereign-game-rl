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
├── section10_experiments.py # Rulebook Section 10 ablation protocol
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

You can also train a single Section 10 condition directly with `train.py`:

```bash
python train.py --episodes 300 --experiment full
python train.py --episodes 300 --experiment no-legitimacy
python train.py --episodes 300 --experiment no-occupation
python train.py --episodes 300 --experiment no-neutral
python train.py --episodes 300 --experiment baseline
python train.py --episodes 300 --experiment earlier-sanctions
python train.py --episodes 300 --experiment later-sanctions
```

Manual toggles are also available:

```bash
python train.py --episodes 300 --no-legitimacy
python train.py --episodes 300 --no-occupation
python train.py --episodes 300 --no-neutral
python train.py --episodes 300 --sanction-threshold 0.45
```

### 3. Evaluate a saved agent

```bash
python train.py --eval --demo
```

To evaluate a model under a Section 10 condition, pass the same experiment flag:

```bash
python train.py --eval --experiment no-neutral
```

### 4. Run the PDF Section 10 experiments

```bash
python section10_experiments.py --episodes 300 --eval-episodes 10
```

For a very fast smoke test:

```bash
python section10_experiments.py --quick
```

This trains a fresh DQN agent under each ablation condition from the rulebook:

| Experiment | L active | t_occ active | θ active | Expected policy |
|------------|----------|--------------|----------|-----------------|
| Full model | yes | yes | yes | Negotiate or deter |
| No legitimacy | no | yes | yes | Slower invasion |
| No occupation cost | yes | no | yes | Partial invasion |
| No neutral posture | yes | yes | no | Invasion |
| Baseline all off | no | no | no | Always invade |
| Earlier sanctions | yes | yes | yes, θ threshold=0.45 | More negotiation |
| Later sanctions | yes | yes | yes, θ threshold=0.75 | More aggression tolerated |

The runner prints a comparison table with average reward, peace rate,
aggression rate, collapse rate, and final neutral posture.

## RL Algorithm: DQN (from scratch)

`dqn_agent.py` implements a standard **Deep Q-Network**:

- **Q-network:** 2-layer MLP (`49 → 128 → 128 → 20`) mapping observations
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

The observation is a flat `float32` vector of length **49**, matching the
state variables in Section 4.1:

| Slice     | Variable              | Type / Range |
|-----------|-----------------------|--------------|
| 0–26      | Territory control `M` | one-hot `|V| × 3` for Invader, Defender, Neutral |
| 27–35     | Invader units `U_I`   | per-territory unit vector, normalized |
| 36–44     | Defender units `U_D`  | per-territory unit vector, normalized |
| 45        | Legitimacy `L`        | `[0, 1]` |
| 46        | Economy `E`           | `[0, 1]` |
| 47        | Neutral posture `θ`   | `[-1, +1]` |
| 48        | Occupation `t_occ`    | integer state, normalized in observation |

---

## Supply Lines

The environment implements the rulebook's Section 3.2 connectivity rule.
Each turn, the Invader's connected supply network is recomputed with a graph
search starting from `Invader Home`.

- **Connected territories:** Invader-controlled territories reachable from
  `Invader Home` through other Invader-controlled territories.
- **Disconnected occupied territories:** Invader-controlled non-home
  territories not reachable from `Invader Home`.
- **Economy `E`:** computed from connected resource value divided by total
  Invader-controlled resource value, then adjusted by sanctions and political
  economy modifiers.
- **Defender economy:** tracked internally for the `θ < -0.85` threshold event,
  where Neutral formally allies with the Invader and Defender `E` is reduced.
- **Reward accounting:** Section 8 territory reward follows the rulebook formula
  and sums resources for territories controlled by the Invader.
- **Occupation cost:** each disconnected occupied territory adds extra
  occupation burden.
- **Insurgency:** probability increases with `t_occ`; each insurgency event
  destroys one Invader unit and applies the insurgency reward penalty.

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
