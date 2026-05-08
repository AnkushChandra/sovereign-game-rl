"""
game_logic.py — Turn structure, action decoding, and state updates.

Coordinates the 12-step turn sequence described in Section 5 of the rulebook.
"""

import numpy as np

from config import (
    POLITICAL_ACTIONS, MILITARY_ACTIONS,
    NUM_POLITICAL, NUM_MILITARY,
    POLITICAL_EFFECTS, MILITARY_EFFECTS,
    INVADER_GROUND_UNITS, INVADER_STRIKE_UNITS,
    DEFENDER_GROUND_UNITS, DEFENDER_STRIKE_UNITS,
    INITIAL_LEGITIMACY, INITIAL_ECONOMY, INITIAL_THETA, INITIAL_T_OCC,
    DO_NOTHING_L_DECAY, DO_NOTHING_THETA_DRIFT,
    MAX_STEPS,
    NEGOTIATE_CONSECUTIVE_NEEDED, NEGOTIATE_NO_AGGRESSION_STEPS,
    NEGOTIATE_MIN_LEGITIMACY,
    TERMINAL_POLITICAL_COLLAPSE, TERMINAL_MILITARY_DEFEAT,
    TERMINAL_NEGOTIATED_PEACE, TERMINAL_TIME_LIMIT,
    TERMINAL_TOTAL_CONQUEST,
    SUPPLY_ROUTE_OCC_REDUCTION,
    NUM_TERRITORIES,
)
from map import build_default_map, T_INVADER_HOME, T_DEFENDER_HOME
from defender import defender_respond
from neutral import update_theta, check_threshold_events
from reward import compute_reward


# ─────────────────────────────────────────────
# Action encoding / decoding
# ─────────────────────────────────────────────

def decode_action(action_int):
    """
    Decode a single integer action into (political, military) pair.

    Encoding:  action_int = pol_idx * NUM_MILITARY + mil_idx

    Args:
        action_int : int in [0, NUM_POLITICAL * NUM_MILITARY)

    Returns:
        (pol_action: str, mil_action: str)
    """
    pol_idx = action_int // NUM_MILITARY
    mil_idx = action_int % NUM_MILITARY
    return POLITICAL_ACTIONS[pol_idx], MILITARY_ACTIONS[mil_idx]


def encode_action(pol_action, mil_action):
    """Encode a (political, military) pair into a single integer."""
    pol_idx = POLITICAL_ACTIONS.index(pol_action)
    mil_idx = MILITARY_ACTIONS.index(mil_action)
    return pol_idx * NUM_MILITARY + mil_idx


# ─────────────────────────────────────────────
# State initialisation
# ─────────────────────────────────────────────

def init_state(rng=None):
    """
    Create and return the initial game state dict.

    Args:
        rng : numpy Generator (optional)

    Returns:
        state : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    territories, adjacency = build_default_map()

    state = {
        # Map
        "territories": territories,
        "adjacency": adjacency,

        # Military
        "invader_units": INVADER_GROUND_UNITS + INVADER_STRIKE_UNITS,
        "defender_units": DEFENDER_GROUND_UNITS + DEFENDER_STRIKE_UNITS,

        # Core state variables
        "legitimacy": INITIAL_LEGITIMACY,
        "economy": INITIAL_ECONOMY,
        "theta": INITIAL_THETA,
        "t_occ": INITIAL_T_OCC,

        # Bookkeeping
        "step": 0,
        "done": False,
        "rng": rng,

        # Threshold event flags
        "sanctions_active": False,
        "coalition_fired": False,
        "supply_routes_open": False,
        "ally_invader_fired": False,
        "sanctions_below_counter": 0,

        # Negotiation tracking
        "consecutive_negotiate": 0,
        "steps_since_aggression": 0,

        # Defender-home flag for combat bonus
        "invader_on_defender_home": False,
    }
    return state


# ─────────────────────────────────────────────
# Political effects
# ─────────────────────────────────────────────

def apply_political_action(pol_action, state):
    """
    Apply the political action's effects on legitimacy, theta (direct), and economy.

    Also updates negotiation tracking counters.

    Per PDF Section 6.1:
      - DO_NOTHING: slow L decay if L < 0.5; slow θ drift if t_occ > 0.
    """
    delta_L, _direct_theta, delta_E = POLITICAL_EFFECTS[pol_action]

    # DO_NOTHING special rules
    if pol_action == "DO_NOTHING":
        if state["legitimacy"] < 0.5:
            delta_L = DO_NOTHING_L_DECAY
        if state["t_occ"] > 0:
            # Slow θ drift toward Defender when occupying and doing nothing
            state["theta"] = float(
                np.clip(state["theta"] + DO_NOTHING_THETA_DRIFT, -1.0, 1.0)
            )

    state["legitimacy"] = float(np.clip(state["legitimacy"] + delta_L, 0.0, 1.0))
    state["economy"] = float(np.clip(state["economy"] + delta_E, 0.0, 1.0))

    # Negotiation counter
    if pol_action == "NEGOTIATE":
        state["consecutive_negotiate"] += 1
    else:
        state["consecutive_negotiate"] = 0


# ─────────────────────────────────────────────
# Military effects
# ─────────────────────────────────────────────

def apply_military_action(mil_action, state):
    """
    Apply the military action: update legitimacy, territory, units, t_occ.

    Returns:
        newly_captured : list[dict]  — territories captured this step
    """
    delta_L = MILITARY_EFFECTS[mil_action]
    state["legitimacy"] = float(np.clip(state["legitimacy"] + delta_L, 0.0, 1.0))

    newly_captured = []
    territories = state["territories"]
    adjacency = state["adjacency"]

    # Aggression tracking
    if mil_action in ("ADVANCE", "STRIKE"):
        state["steps_since_aggression"] = 0
    else:
        state["steps_since_aggression"] += 1

    # ── ADVANCE: claim one adjacent contested / enemy territory ──
    if mil_action == "ADVANCE" and state["invader_units"] > 0:
        # Find an adjacent territory to claim
        invader_territories = [t["id"] for t in territories if t["controller"] == "I"]
        for tid in invader_territories:
            for neighbor in adjacency[tid]:
                t = territories[neighbor]
                if t["controller"] in ("Contested", "D"):
                    t["controller"] = "I"
                    newly_captured.append(t)
                    break  # only one per step
            if newly_captured:
                break

    # ── WITHDRAW: cede one contested territory back ──
    elif mil_action == "WITHDRAW":
        for t in territories:
            if t["controller"] == "I" and t["is_home"] != "I":
                t["controller"] = "Contested"
                break

    # ── STRIKE: destroy one Defender unit (costs heavy legitimacy — already applied) ──
    elif mil_action == "STRIKE" and state["invader_units"] > 0:
        if state["defender_units"] > 0:
            state["defender_units"] -= 1

    # ── HOLD: no territory change ──
    # (no action needed)

    return newly_captured


# ─────────────────────────────────────────────
# Occupation duration
# ─────────────────────────────────────────────

def update_occupation(state):
    """Increment t_occ if Invader holds non-home territory, else reset."""
    territories = state["territories"]
    invader_non_home = any(
        t["controller"] == "I" and t["is_home"] != "I"
        for t in territories
    )
    if invader_non_home:
        state["t_occ"] += 1
    else:
        state["t_occ"] = 0


# ─────────────────────────────────────────────
# Economy maintenance (supply-line simplification)
# ─────────────────────────────────────────────

def update_economy(state):
    """
    Maintain Invader economy E in [0, 1].

    E accumulates changes from:
      - political actions (e.g. IMPOSE_SANCTION −0.03)
      - sanctions threshold event (−0.01/step while active)
      - supply-route threshold event (small positive drift when open)

    This function only clamps the current value and applies the
    supply-route bonus if active. It does NOT overwrite the running
    state set by political actions and threshold events, which would
    discard their effects.
    """
    # Small positive drift when supply routes are open (θ < -0.60)
    if state["supply_routes_open"]:
        state["economy"] += 0.005

    state["economy"] = float(np.clip(state["economy"], 0.0, 1.0))


# ─────────────────────────────────────────────
# Defender-home detection
# ─────────────────────────────────────────────

def update_defender_home_flag(state):
    """Check if Invader controls the Defender's home territory."""
    state["invader_on_defender_home"] = (
        state["territories"][T_DEFENDER_HOME]["controller"] == "I"
    )


# ─────────────────────────────────────────────
# Terminal conditions
# ─────────────────────────────────────────────

def check_terminal(state):
    """
    Check all terminal conditions.

    Returns:
        done   : bool
        reason : str or None
        reward : float  — terminal bonus/penalty (added on top of step reward)
    """
    # Political collapse
    if state["legitimacy"] <= 0.0:
        return True, "political_collapse", TERMINAL_POLITICAL_COLLAPSE

    # Military defeat
    if state["invader_units"] <= 0:
        return True, "military_defeat", TERMINAL_MILITARY_DEFEAT

    # Total conquest
    if all(t["controller"] == "I" for t in state["territories"]):
        return True, "total_conquest", TERMINAL_TOTAL_CONQUEST

    # Negotiated settlement
    if (
        state["consecutive_negotiate"] >= NEGOTIATE_CONSECUTIVE_NEEDED
        and state["steps_since_aggression"] >= NEGOTIATE_NO_AGGRESSION_STEPS
        and state["legitimacy"] >= NEGOTIATE_MIN_LEGITIMACY
    ):
        return True, "negotiated_peace", TERMINAL_NEGOTIATED_PEACE

    # Time limit
    if state["step"] >= MAX_STEPS:
        return True, "time_limit", TERMINAL_TIME_LIMIT

    return False, None, 0.0


# ─────────────────────────────────────────────
# Full turn execution
# ─────────────────────────────────────────────

def execute_turn(action_int, state):
    """
    Execute one full turn (steps 1-12 from the rulebook).

    Args:
        action_int : int  — joint action chosen by the Invader agent
        state      : dict — current game state (modified in-place)

    Returns:
        reward     : float
        done       : bool
        info       : dict  — debug / logging information
    """
    rng = state["rng"]
    info = {}

    # Step 1: Observe (implicit — state already available to agent)

    # Step 2-3: Decode Invader action
    pol_action, mil_action = decode_action(action_int)
    info["pol_action"] = pol_action
    info["mil_action"] = mil_action

    # Step 2: Apply political action
    apply_political_action(pol_action, state)

    # Step 3: Apply military action
    newly_captured = apply_military_action(mil_action, state)
    info["newly_captured"] = [t["name"] for t in newly_captured]

    # Step 4: Defender responds
    defender_result = defender_respond(mil_action, pol_action, state)
    state["invader_units"] = max(
        0, state["invader_units"] - defender_result["units_destroyed"]
    )
    info["defender"] = defender_result

    # Step 5-6: Resolve outcomes & update territory map (already done above)
    update_defender_home_flag(state)

    # Step 7: Update L, E, t_occ
    update_occupation(state)
    update_economy(state)

    # Occupation cost reduction if supply routes open
    effective_t_occ = state["t_occ"]
    if state["supply_routes_open"]:
        effective_t_occ = int(effective_t_occ * (1.0 - SUPPLY_ROUTE_OCC_REDUCTION))
    state_for_reward = {**state, "t_occ": effective_t_occ}

    # Step 8: Neutral posture shift
    state["theta"] = update_theta(
        state["theta"], state["legitimacy"],
        mil_action, pol_action, state["t_occ"], rng,
    )
    info["theta"] = state["theta"]

    # Step 9: Threshold events
    events = check_threshold_events(state["theta"], state)
    info["threshold_events"] = events

    # Step 10: Check terminal conditions
    done, reason, terminal_reward = check_terminal(state)
    info["terminal_reason"] = reason

    # Step 11: Compute reward
    step_reward, reward_info = compute_reward(state_for_reward, newly_captured, rng)
    total_reward = step_reward + (terminal_reward if done else 0.0)
    info["reward_breakdown"] = reward_info

    # Step 12: Advance step counter and set done flag
    state["step"] += 1
    state["done"] = done

    return total_reward, done, info
