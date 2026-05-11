## Setup

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
venv\Scripts\activate
```

## How to Run

Run a random-action environment smoke test:

```bash
python main.py
```

Train the DQN agent with the default settings:

```bash
python train.py --episodes 300
```

Evaluate a saved model:

```bash
python train.py --eval
```

Evaluate and print one greedy step-by-step demo episode:

```bash
python train.py --eval --demo
```

Choose an experiment preset:

```bash
python train.py --experiment full --episodes 300
python train.py --experiment no_legitimacy --episodes 300
python train.py --experiment no_occupation --episodes 300
python train.py --experiment no_neutral --episodes 300
python train.py --experiment baseline --episodes 300
```

Use the original reward and terminal profile instead of the tuned profile:

```bash
python train.py --config original --episodes 300
```

By default, trained weights are saved to:

```text
dqn_sovereign.pt
```

## Repository Organization

```text
sovereign-game-rl/
├── README.md            # Setup, execution, and repository guide
├── requirements.txt     # Python dependencies
├── main.py              # Random-action smoke test
├── train.py             # Training, evaluation, demos, and experiment presets
├── dqn_agent.py         # DQN network, replay buffer, and epsilon-greedy policy
├── sovereign_env.py     # Gymnasium Env wrapper
├── game_logic.py        # Turn execution, action decoding, state transitions
├── reward.py            # Reward components and insurgency sampling
├── defender.py          # Rule-based Defender response policy
├── neutral.py           # Neutral alignment update and threshold events
├── map.py               # Default territory graph and territory metadata
├── config.py            # Tuned constants, rewards, thresholds, DQN settings
└── config_original.py   # Original reward and terminal override profile
```

## Code Structure

`sovereign_env.py` is the public Gymnasium interface. It exposes
`SovereignEnv`, defines the observation and action spaces, and delegates game
updates to `game_logic.py`.

`game_logic.py` coordinates one full environment step. It decodes the joint
action, applies political and military effects, invokes the Defender policy,
updates occupation and economy state, applies Neutral events, computes rewards,
and checks terminal conditions.

`reward.py`, `neutral.py`, `defender.py`, and `map.py` keep major mechanics in
separate modules so the environment wrapper stays small and readable.

`config.py` is the main place to change experiment parameters. It contains
starting units, action effects, reward weights, terminal rewards, threshold
values, DQN hyperparameters, and experiment preset definitions.

`train.py` is the main experiment runner. It supports training, evaluation,
greedy demo episodes, behavior summaries, config profiles, and mechanism
ablations.

## Environment Summary

The action space is `Discrete(20)`. Each integer action represents one
political action and one military action:

```text
action = political_index * 4 + military_index
```

Political actions:

```text
SEEK_ALLIANCE
IMPOSE_SANCTION
ISSUE_THREAT
NEGOTIATE
DO_NOTHING
```

Military actions:

```text
ADVANCE
HOLD
WITHDRAW
STRIKE
```

The observation is a `float32` vector of length 16:

```text
0-8   territory controller encoding
9     normalized Invader units
10    normalized Defender units
11    Invader legitimacy
12    Invader economy
13    Neutral alignment
14    normalized occupation duration
15    normalized step count
```

The default map has 9 territories:

```text
    I_HOME -- C1 -- C2 -- D_HOME
               |     |
              C3 -- C4
               |     |
    N_HOME -- C5 -- C6
```

## Experiment Presets

The `--experiment` flag selects which environment mechanisms are active:

```text
full           legitimacy on,  occupation on,  neutral alignment on
no_legitimacy  legitimacy off, occupation on,  neutral alignment on
no_occupation  legitimacy on,  occupation off, neutral alignment on
no_neutral     legitimacy on,  occupation on,  neutral alignment off
baseline       legitimacy off, occupation off, neutral alignment off
```

These presets are defined in `config.py` and passed into `SovereignEnv` through
`train.py`.

## Example API Usage

```python
from sovereign_env import SovereignEnv

env = SovereignEnv(seed=0, experiment="full")
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## DQN Implementation

`dqn_agent.py` implements DQN directly with PyTorch:

- A two-hidden-layer MLP maps observations to Q-values.
- A replay buffer stores transitions for off-policy learning.
- A target network is periodically synchronized for stable bootstrapping.
- Epsilon-greedy exploration decays over training steps.
- Gradients are clipped to reduce unstable updates.

The relevant hyperparameters are named with the `DQN_` prefix in `config.py`.

