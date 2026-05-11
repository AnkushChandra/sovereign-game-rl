"""
config_original.py — Original SOVEREIGN tuning profile.

This file contains the earlier/default values so experiments can compare the
original reward/terminal settings against the currently tuned settings in
config.py.
"""

OVERRIDES = {
    # Episode horizon
    "MAX_STEPS": 50,

    # Reward weights from the original rulebook-style implementation.
    "W_TERRITORY": 0.30,
    "W_RESOURCE": 0.20,
    "W_OCCUPATION": 0.25,
    "W_LEGITIMACY": 0.15,
    "W_SANCTION": 0.20,
    "W_INSURGENCY": 0.10,
    "W_STEP": 0.00,

    # Original terminal rewards.
    "TERMINAL_POLITICAL_COLLAPSE": -50.0,
    "TERMINAL_MILITARY_DEFEAT": -30.0,
    "TERMINAL_NEGOTIATED_PEACE": 40.0,
    "TERMINAL_TIME_LIMIT": 0.0,
    "TERMINAL_TOTAL_CONQUEST": 10.0,
}
