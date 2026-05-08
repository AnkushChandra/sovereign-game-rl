"""
config.py — Constants and configuration for the SOVEREIGN environment.

All tunable parameters live here so experiments can adjust them in one place.
"""

# ─────────────────────────────────────────────
# Starting military units
# ─────────────────────────────────────────────
INVADER_GROUND_UNITS = 12
INVADER_STRIKE_UNITS = 3
DEFENDER_GROUND_UNITS = 6
DEFENDER_STRIKE_UNITS = 1
NEUTRAL_GROUND_UNITS = 4

# Defender home-turf effectiveness bonus (20%)
DEFENDER_HOME_BONUS = 0.20

# ─────────────────────────────────────────────
# Initial state values
# ─────────────────────────────────────────────
INITIAL_LEGITIMACY = 1.0        # L starts at 1.0
INITIAL_ECONOMY = 1.0           # E (supply index) starts at 1.0
INITIAL_THETA = 0.0             # Neutral posture starts at true-neutral
INITIAL_T_OCC = 0               # Occupation counter starts at 0

# ─────────────────────────────────────────────
# Episode limits
# ─────────────────────────────────────────────
MAX_STEPS = 200                 # T_max

# ─────────────────────────────────────────────
# Political actions  (indices)
# ─────────────────────────────────────────────
POLITICAL_ACTIONS = [
    "SEEK_ALLIANCE",
    "IMPOSE_SANCTION",
    "ISSUE_THREAT",
    "NEGOTIATE",
    "DO_NOTHING",
]
NUM_POLITICAL = len(POLITICAL_ACTIONS)

# ─────────────────────────────────────────────
# Military actions  (indices)
# ─────────────────────────────────────────────
MILITARY_ACTIONS = [
    "ADVANCE",
    "HOLD",
    "WITHDRAW",
    "STRIKE",
]
NUM_MILITARY = len(MILITARY_ACTIONS)

# Joint action space size = 5 × 4 = 20
NUM_ACTIONS = NUM_POLITICAL * NUM_MILITARY

# ─────────────────────────────────────────────
# Political-action effects on (L, θ, E)
# Each entry: (delta_L, delta_theta, delta_E_target)
# ─────────────────────────────────────────────
POLITICAL_EFFECTS = {
    "SEEK_ALLIANCE":   (+0.01, -0.05,  0.00),
    "IMPOSE_SANCTION": (-0.02, +0.04, -0.03),
    "ISSUE_THREAT":    (-0.03, +0.03,  0.00),
    "NEGOTIATE":       (+0.03, -0.04,  0.00),
    "DO_NOTHING":      ( 0.00,  0.00,  0.00),   # handled with special rules
}

# ─────────────────────────────────────────────
# Military-action effects on (L, t_occ)
# delta_L per action; t_occ logic handled in game_logic
# ─────────────────────────────────────────────
MILITARY_EFFECTS = {
    "ADVANCE":  -0.05,   # L cost of advancing
    "HOLD":      0.00,
    "WITHDRAW": +0.02,   # small L recovery
    "STRIKE":   -0.08,   # heavy L cost
}

# ─────────────────────────────────────────────
# Neutral posture drift coefficients  (Section 7.2)
# ─────────────────────────────────────────────
DRIFT_ALPHA = 0.04    # legitimacy coupling
DRIFT_BETA  = 0.05    # advance shock
DRIFT_GAMMA = 0.10    # strike shock
DRIFT_DELTA = 0.04    # negotiate pull (toward center)
DRIFT_EPS   = 0.03    # alliance-seeking pull
DRIFT_ZETA  = 0.03    # occupation drift
DRIFT_NOISE_STD = 0.02  # σ for Gaussian noise

# ─────────────────────────────────────────────
# Neutral posture threshold events  (Section 7.3)
# ─────────────────────────────────────────────
SANCTION_THRESHOLD       = 0.60   # θ > 0.60 → sanctions
COALITION_THRESHOLD      = 0.85   # θ > 0.85 → Neutral joins Defender
SUPPLY_ROUTE_THRESHOLD   = -0.60  # θ < -0.60 → supply routes open
ALLY_INVADER_THRESHOLD   = -0.85  # θ < -0.85 → Neutral allies Invader

SANCTION_ECON_PENALTY    = 0.01   # E reduced per step while sanctions active
COALITION_DEFENDER_BONUS_UNITS = 2
COALITION_L_PENALTY      = 0.10
SUPPLY_ROUTE_OCC_REDUCTION = 0.30  # 30% occupation-cost reduction
ALLY_INVADER_DEF_E_PENALTY = 0.02
ALLY_INVADER_L_PENALTY     = 0.05

# Hysteresis: sanctions lifted only if θ < 0.50 for this many consecutive steps
SANCTION_LIFT_THRESHOLD  = 0.50
SANCTION_LIFT_STEPS      = 5

# ─────────────────────────────────────────────
# Reward weights  (Section 8)
# ─────────────────────────────────────────────
W_TERRITORY   = 0.30   # territory control
W_RESOURCE    = 0.20   # newly captured resource bonus
W_OCCUPATION  = 0.25   # occupation cost
W_LEGITIMACY  = 0.15   # legitimacy deficit
W_SANCTION    = 0.20   # sanction penalty
W_INSURGENCY  = 0.10   # insurgency event

# ─────────────────────────────────────────────
# Insurgency  (Section 8.3)
# ─────────────────────────────────────────────
INSURGENCY_LAMBDA = 0.05   # hazard-rate parameter

# ─────────────────────────────────────────────
# Terminal rewards  (Section 9)
# ─────────────────────────────────────────────
TERMINAL_POLITICAL_COLLAPSE = -50.0
TERMINAL_MILITARY_DEFEAT    = -30.0
TERMINAL_NEGOTIATED_PEACE   = +40.0
TERMINAL_TIME_LIMIT         =   0.0
TERMINAL_TOTAL_CONQUEST     = +10.0

# ─────────────────────────────────────────────
# Negotiated-settlement conditions
# ─────────────────────────────────────────────
NEGOTIATE_CONSECUTIVE_NEEDED = 5    # steps of consecutive NEGOTIATE
NEGOTIATE_NO_AGGRESSION_STEPS = 5   # no ADVANCE/STRIKE for this many steps
NEGOTIATE_MIN_LEGITIMACY = 0.40     # L must be above this

# ─────────────────────────────────────────────
# Slow-decay / slow-drift for DO_NOTHING
# ─────────────────────────────────────────────
DO_NOTHING_L_DECAY = -0.005   # if L < 0.5
DO_NOTHING_THETA_DRIFT = 0.01  # if t_occ > 0, drifts toward Defender

# Number of territories
NUM_TERRITORIES = 9

# ─────────────────────────────────────────────
# DQN hyperparameters (for dqn_agent.py / train.py)
# ─────────────────────────────────────────────
DQN_HIDDEN_SIZES   = (128, 128)   # MLP hidden layer widths
DQN_LEARNING_RATE  = 5e-4
DQN_GAMMA          = 0.99         # discount factor
DQN_BATCH_SIZE     = 64
DQN_BUFFER_SIZE    = 50_000
DQN_MIN_BUFFER     = 1_000        # no learning until buffer has this many transitions
DQN_TARGET_UPDATE  = 500          # sync target net every N gradient steps
DQN_GRAD_CLIP      = 1.0          # max gradient norm

# ε-greedy exploration schedule
DQN_EPS_START      = 1.0
DQN_EPS_END        = 0.05
DQN_EPS_DECAY_STEPS = 20_000      # linear decay steps

# Training loop defaults
DQN_TOTAL_EPISODES = 300
DQN_EVAL_EVERY     = 25           # run a greedy eval every N episodes
DQN_EVAL_EPISODES  = 5
