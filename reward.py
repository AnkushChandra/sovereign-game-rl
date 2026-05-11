"""
reward.py — Reward calculation for the SOVEREIGN environment.

    r_t = r_pos(s_t, a_t) − r_neg(s_t, a_t)

Positive terms reward territory control and resource capture.
Negative terms penalise occupation cost, legitimacy loss, sanctions, and insurgency.
"""

import math
import numpy as np

from config import (
    W_TERRITORY, W_RESOURCE, W_OCCUPATION, W_LEGITIMACY,
    W_SANCTION, W_INSURGENCY, W_STEP,
    INSURGENCY_LAMBDA, MAX_STEPS,
)


def compute_territory_reward(territories):
    """
    Sum of resource_value for all territories controlled by the Invader.

    Args:
        territories : list[dict]

    Returns:
        float
    """
    return sum(
        t["resource_value"]
        for t in territories
        if t["controller"] == "I"
    )


def compute_resource_capture_bonus(newly_captured):
    """
    Small bonus for territories captured *this step*.

    Args:
        newly_captured : list[dict]  — territories newly taken this step

    Returns:
        float
    """
    return sum(t["resource_value"] for t in newly_captured)


def compute_occupation_cost(t_occ):
    """
    Linear occupation cost proportional to t_occ / T_max.

    Args:
        t_occ : int

    Returns:
        float
    """
    return t_occ / MAX_STEPS


def compute_legitimacy_penalty(legitimacy):
    """
    Penalty proportional to the legitimacy *deficit* (1 - L).

    Args:
        legitimacy : float [0, 1]

    Returns:
        float
    """
    return 1.0 - legitimacy


def mechanism_enabled(state, name):
    """Return whether a Section 10 mechanism is active for this episode."""
    return state.get("mechanisms", {}).get(name, True)


def compute_sanction_penalty(sanctions_active, economy):
    """
    Active only when sanctions threshold is crossed.

    Args:
        sanctions_active : bool
        economy : float

    Returns:
        float
    """
    if sanctions_active:
        return 1.0 - economy
    return 0.0


def compute_insurgency(t_occ, rng=None):
    """
    Stochastic insurgency event based on occupation duration.

    p(insurgency | t_occ) = 1 − exp(−λ · t_occ)

    Returns (occurred: bool, penalty: float).
    """
    if rng is None:
        rng = np.random.default_rng()

    if t_occ == 0:
        return False, 0.0

    prob = 1.0 - math.exp(-INSURGENCY_LAMBDA * t_occ)
    occurred = rng.random() < prob
    penalty = 1.0 if occurred else 0.0
    return occurred, penalty


def compute_reward(state, newly_captured, rng=None):
    """
    Compute the total step reward for the Invader.

    Args:
        state          : dict with keys territories, legitimacy, economy,
                         theta, t_occ, sanctions_active
        newly_captured : list[dict]
        rng            : numpy Generator

    Returns:
        reward          : float
        reward_info     : dict  — breakdown for debugging
    """
    territories = state["territories"]
    legitimacy  = state["legitimacy"]
    economy     = state["economy"]
    t_occ       = state["t_occ"]
    sanctions_active = state.get("sanctions_active", False)

    # ── Positive terms ──
    r_territory = W_TERRITORY * compute_territory_reward(territories)
    r_capture   = W_RESOURCE  * compute_resource_capture_bonus(newly_captured)

    # ── Negative terms ──
    r_occ = 0.0
    if mechanism_enabled(state, "occupation"):
        r_occ = W_OCCUPATION * compute_occupation_cost(t_occ)

    r_legit = 0.0
    if mechanism_enabled(state, "legitimacy"):
        r_legit = W_LEGITIMACY * compute_legitimacy_penalty(legitimacy)

    r_sanction = 0.0
    if mechanism_enabled(state, "neutral_posture"):
        r_sanction = W_SANCTION * compute_sanction_penalty(sanctions_active, economy)

    insurgency_occurred, ins_pen = False, 0.0
    if mechanism_enabled(state, "occupation"):
        insurgency_occurred, ins_pen = compute_insurgency(t_occ, rng)
    r_insurgency = W_INSURGENCY * ins_pen
    r_step = W_STEP

    # ── Net reward ──
    r_pos = r_territory + r_capture
    r_neg = r_occ + r_legit + r_sanction + r_insurgency + r_step
    reward = r_pos - r_neg

    reward_info = {
        "r_territory": r_territory,
        "r_capture": r_capture,
        "r_occupation": -r_occ,
        "r_legitimacy": -r_legit,
        "r_sanction": -r_sanction,
        "r_insurgency": -r_insurgency,
        "r_step": -r_step,
        "insurgency_occurred": insurgency_occurred,
        "r_pos": r_pos,
        "r_neg": r_neg,
        "reward": reward,
    }
    return reward, reward_info
