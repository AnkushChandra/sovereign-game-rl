"""
neutral.py — Neutral posture update logic (drift-diffusion model).

The Neutral nation's political alignment θ ∈ [-1, +1] evolves each step:

    θ_{t+1} = clip(θ_t + drift(s_t, a_t) + noise, -1, +1)

Threshold events fire when θ crosses key values (with hysteresis).
"""

import numpy as np

from config import (
    DRIFT_ALPHA, DRIFT_BETA, DRIFT_GAMMA, DRIFT_DELTA,
    DRIFT_EPS, DRIFT_ZETA, DRIFT_NOISE_STD,
    MAX_STEPS,
    SANCTION_THRESHOLD, COALITION_THRESHOLD,
    SUPPLY_ROUTE_THRESHOLD, ALLY_INVADER_THRESHOLD,
    SANCTION_ECON_PENALTY, COALITION_DEFENDER_BONUS_UNITS,
    COALITION_L_PENALTY, SUPPLY_ROUTE_OCC_REDUCTION,
    ALLY_INVADER_DEF_E_PENALTY, ALLY_INVADER_L_PENALTY,
    SANCTION_LIFT_THRESHOLD, SANCTION_LIFT_STEPS,
)


def compute_drift(legitimacy, mil_action, pol_action, t_occ):
    """
    Compute the deterministic drift μ(s_t, a_t) for the neutral posture.

    Positive drift → posture moves toward Defender (+1).
    Negative drift → posture moves toward Invader (-1).

    Args:
        legitimacy : float  — current Invader legitimacy L
        mil_action : str    — e.g. "ADVANCE", "STRIKE", etc.
        pol_action : str    — e.g. "NEGOTIATE", "SEEK_ALLIANCE", etc.
        t_occ      : int    — current occupation duration

    Returns:
        mu : float — deterministic drift component
    """
    mu = 0.0

    # Low legitimacy alienates the Neutral (pushes θ toward Defender)
    mu += DRIFT_ALPHA * (1.0 - legitimacy)

    # Military actions shock posture toward Defender
    if mil_action == "ADVANCE":
        mu += DRIFT_BETA
    elif mil_action == "STRIKE":
        mu += DRIFT_GAMMA

    # Diplomatic actions pull posture toward center / Invader
    if pol_action == "NEGOTIATE":
        mu -= DRIFT_DELTA
    elif pol_action == "SEEK_ALLIANCE":
        mu -= DRIFT_EPS

    # Prolonged occupation steadily alienates
    mu += DRIFT_ZETA * (t_occ / MAX_STEPS)

    return mu


def update_theta(theta, legitimacy, mil_action, pol_action, t_occ, rng=None):
    """
    Update the neutral posture θ by one step.

    Args:
        theta      : float  — current posture
        legitimacy : float  — Invader legitimacy L
        mil_action : str
        pol_action : str
        t_occ      : int
        rng        : numpy Generator (optional, for reproducibility)

    Returns:
        theta_new : float — updated posture, clipped to [-1, +1]
    """
    if rng is None:
        rng = np.random.default_rng()

    drift = compute_drift(legitimacy, mil_action, pol_action, t_occ)
    noise = rng.normal(0.0, DRIFT_NOISE_STD)
    theta_new = np.clip(theta + drift + noise, -1.0, 1.0)
    return float(theta_new)


def check_threshold_events(theta, state):
    """
    Fire threshold events based on the current θ and update state accordingly.

    Modifies `state` dict in-place and returns a list of event strings that fired.

    Expected keys in state:
        sanctions_active        : bool
        coalition_fired         : bool
        supply_routes_open      : bool
        ally_invader_fired      : bool
        economy                 : float   (Invader E)
        legitimacy              : float   (Invader L)
        defender_units          : int
        sanctions_below_counter : int     (hysteresis counter)

    Args:
        theta : float
        state : dict  (modified in-place)

    Returns:
        events : list[str]
    """
    events = []

    # ── Sanctions (θ > 0.60) ──────────────────────
    if theta > SANCTION_THRESHOLD and not state["sanctions_active"]:
        state["sanctions_active"] = True
        events.append("SANCTIONS_IMPOSED")

    # Hysteresis: lift sanctions only if θ < 0.50 for several consecutive steps
    if state["sanctions_active"]:
        if theta < SANCTION_LIFT_THRESHOLD:
            state["sanctions_below_counter"] += 1
            if state["sanctions_below_counter"] >= SANCTION_LIFT_STEPS:
                state["sanctions_active"] = False
                state["sanctions_below_counter"] = 0
                events.append("SANCTIONS_LIFTED")
        else:
            state["sanctions_below_counter"] = 0

    # Apply ongoing sanction cost
    if state["sanctions_active"]:
        state["economy"] = max(0.0, state["economy"] - SANCTION_ECON_PENALTY)

    # ── Coalition (θ > 0.85) — one-time event ────
    if theta > COALITION_THRESHOLD and not state["coalition_fired"]:
        state["coalition_fired"] = True
        state["defender_units"] += COALITION_DEFENDER_BONUS_UNITS
        state["legitimacy"] = max(0.0, state["legitimacy"] - COALITION_L_PENALTY)
        events.append("NEUTRAL_JOINS_DEFENDER")

    # ── Supply routes open (θ < -0.60) ───────────
    if theta < SUPPLY_ROUTE_THRESHOLD and not state["supply_routes_open"]:
        state["supply_routes_open"] = True
        events.append("SUPPLY_ROUTES_OPEN")

    # ── Neutral allies Invader (θ < -0.85) — one-time ──
    # PDF: "Defender E reduced; L −0.05".
    # Since we do not track a separate Defender supply index, we substitute
    # an equivalent military hit by destroying one Defender unit.
    if theta < ALLY_INVADER_THRESHOLD and not state["ally_invader_fired"]:
        state["ally_invader_fired"] = True
        state["legitimacy"] = max(0.0, state["legitimacy"] - ALLY_INVADER_L_PENALTY)
        state["defender_units"] = max(0, state["defender_units"] - 1)
        events.append("NEUTRAL_ALLIES_INVADER")

    return events
