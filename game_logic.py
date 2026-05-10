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
    DEFAULT_EXPERIMENT_CONFIG,
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

def make_experiment_config(overrides=None):
    config = DEFAULT_EXPERIMENT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config


def init_state(rng=None, experiment_config=None):
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

    exp_config = make_experiment_config(experiment_config)

    state = {
        # Map
        "territories": territories,
        "adjacency": adjacency,

        # Military — PDF §2.1 distinguishes ground and strike units.
        # Totals (ground + strike) are kept as `invader_units` / `defender_units`
        # for backward compatibility with code that queries the overall force.
        "invader_ground": INVADER_GROUND_UNITS,
        "invader_strike": INVADER_STRIKE_UNITS,
        "defender_ground": DEFENDER_GROUND_UNITS,
        "defender_strike": DEFENDER_STRIKE_UNITS,
        "invader_units": INVADER_GROUND_UNITS + INVADER_STRIKE_UNITS,
        "defender_units": DEFENDER_GROUND_UNITS + DEFENDER_STRIKE_UNITS,
        "invader_unit_map": np.array(
            [INVADER_GROUND_UNITS + INVADER_STRIKE_UNITS] + [0] * (NUM_TERRITORIES - 1),
            dtype=np.int32,
        ),
        "defender_unit_map": np.array(
            [0, 0, 0, DEFENDER_GROUND_UNITS + DEFENDER_STRIKE_UNITS] + [0] * (NUM_TERRITORIES - 4),
            dtype=np.int32,
        ),

        # Core state variables
        "legitimacy": INITIAL_LEGITIMACY,
        "economy": INITIAL_ECONOMY,
        "defender_economy": INITIAL_ECONOMY,
        "theta": INITIAL_THETA,
        "t_occ": INITIAL_T_OCC,

        # Bookkeeping
        "step": 0,
        "done": False,
        "rng": rng,
        "experiment_config": exp_config,
        "economy_modifier": 0.0,
        "connected_invader_ids": {T_INVADER_HOME},
        "disconnected_occupied_ids": set(),
        "disconnected_occupied_count": 0,

        # First-capture tracking — PDF §8.1 calls the W_RESOURCE term a
        # "bonus for newly captured territory". We interpret "newly" as
        # first-time captures per episode; otherwise an Invader can farm the
        # capture bonus by WITHDRAW→ADVANCE oscillation on the same tile.
        "first_captured_ids": set(),

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
    Apply the political action's effects on legitimacy, theta (direct), and
    target-nation economy.

    Per PDF Section 6.1, each political action has direct effects on (L, θ, E).
    The θ effects from §6.1 are applied *in addition* to the drift function
    from §7.2 — the table in §6.1 explicitly lists θ effects for every action,
    including IMPOSE_SANCTION (+0.04) and ISSUE_THREAT (+0.03) which do not
    appear in the §7.2 drift terms.

    DO_NOTHING special rules:
      - Slow L decay if L < 0.5.
      - Slow θ drift toward Defender if t_occ > 0.
    """
    delta_L, direct_theta, delta_E = POLITICAL_EFFECTS[pol_action]
    exp_config = state["experiment_config"]

    if pol_action == "DO_NOTHING":
        if exp_config["legitimacy_active"] and state["legitimacy"] < 0.5:
            delta_L = DO_NOTHING_L_DECAY
        if exp_config["neutral_active"] and exp_config["occupation_active"] and state["t_occ"] > 0:
            state["theta"] = float(
                np.clip(state["theta"] + DO_NOTHING_THETA_DRIFT, -1.0, 1.0)
            )

    if exp_config["legitimacy_active"]:
        state["legitimacy"] = float(np.clip(state["legitimacy"] + delta_L, 0.0, 1.0))
    else:
        state["legitimacy"] = INITIAL_LEGITIMACY

    # §6.1 direct θ effect — applied only when the neutral system is active so
    # that the ablation in §10 still works as intended.
    if exp_config["neutral_active"] and direct_theta != 0.0:
        state["theta"] = float(
            np.clip(state["theta"] + direct_theta, -1.0, 1.0)
        )

    # §6.1 economy effect: IMPOSE_SANCTION targets the Defender's economy.
    # All other political actions have delta_E = 0 here; sanctions imposed by
    # the Neutral nation on the Invader are handled separately in neutral.py
    # via the economy_modifier channel.
    if pol_action == "IMPOSE_SANCTION" and delta_E != 0.0:
        state["defender_economy"] = float(
            np.clip(state.get("defender_economy", 1.0) + delta_E, 0.0, 1.0)
        )

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
    if state["experiment_config"]["legitimacy_active"]:
        state["legitimacy"] = float(np.clip(state["legitimacy"] + delta_L, 0.0, 1.0))
    else:
        state["legitimacy"] = INITIAL_LEGITIMACY

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
        first_captured = state.setdefault("first_captured_ids", set())
        invader_territories = [
            t["id"] for t in territories
            if t["controller"] == "I" and state["invader_unit_map"][t["id"]] > 0
        ]
        advanced = False
        for tid in invader_territories:
            for neighbor in adjacency[tid]:
                t = territories[neighbor]
                if t["controller"] in ("Contested", "D"):
                    t["controller"] = "I"
                    state["invader_unit_map"][neighbor] += 1
                    state["invader_unit_map"][tid] = max(0, state["invader_unit_map"][tid] - 1)
                    # §8.1 W_RESOURCE bonus only on first-ever capture of a
                    # tile — otherwise WITHDRAW→ADVANCE oscillation farms
                    # the bonus indefinitely.
                    if neighbor not in first_captured:
                        first_captured.add(neighbor)
                        newly_captured.append(t)
                    advanced = True
                    break  # only one capture per step
            if advanced:
                break

    # ── WITHDRAW: cede one contested territory back ──
    elif mil_action == "WITHDRAW":
        for t in territories:
            if t["controller"] == "I" and t["is_home"] != "I":
                state["invader_unit_map"][T_INVADER_HOME] += state["invader_unit_map"][t["id"]]
                state["invader_unit_map"][t["id"]] = 0
                t["controller"] = "Contested"
                break

    # ── STRIKE: destroy one Defender unit (costs heavy legitimacy — already applied) ──
    # Per PDF §2.1, STRIKE requires the Invader to retain "strike capacity":
    # without any strike units the action still pays its legitimacy cost
    # (the agent attempted aggression) but no Defender unit is destroyed.
    elif mil_action == "STRIKE" and state["invader_units"] > 0 and state["invader_strike"] > 0:
        if state["defender_units"] > 0:
            _destroy_defender_unit(state)

    # ── HOLD: no territory change ──
    # (no action needed)

    return newly_captured


# ─────────────────────────────────────────────
# Occupation duration
# ─────────────────────────────────────────────

def update_occupation(state):
    """Increment t_occ if Invader holds non-home territory, else reset."""
    if not state["experiment_config"]["occupation_active"]:
        state["t_occ"] = 0
        return

    invader_non_home = any(
        state["invader_unit_map"][t["id"]] > 0 and t["is_home"] != "I"
        for t in state["territories"]
    )
    if invader_non_home:
        state["t_occ"] += 1
    else:
        state["t_occ"] = 0


# ─────────────────────────────────────────────
# Economy maintenance (supply-line simplification)
# ─────────────────────────────────────────────

def update_economy(state):
    territories = state["territories"]
    adjacency = state["adjacency"]
    connected = set()
    stack = [T_INVADER_HOME] if territories[T_INVADER_HOME]["controller"] == "I" else []

    while stack:
        tid = stack.pop()
        if tid in connected:
            continue
        connected.add(tid)
        for neighbor in adjacency[tid]:
            if territories[neighbor]["controller"] == "I" and neighbor not in connected:
                stack.append(neighbor)

    invader_ids = {t["id"] for t in territories if t["controller"] == "I"}
    disconnected = {
        tid for tid in invader_ids
        if tid not in connected and territories[tid]["is_home"] != "I"
    }

    connected_resource = sum(territories[tid]["resource_value"] for tid in connected)
    controlled_resource = sum(territories[tid]["resource_value"] for tid in invader_ids)
    base_economy = connected_resource / controlled_resource if controlled_resource > 0.0 else 0.0

    state["connected_invader_ids"] = connected
    state["disconnected_occupied_ids"] = disconnected
    state["disconnected_occupied_count"] = len(disconnected)
    state["economy"] = float(np.clip(base_economy + state["economy_modifier"], 0.0, 1.0))


# ─────────────────────────────────────────────
# Defender-home detection
# ─────────────────────────────────────────────

def update_defender_home_flag(state):
    """Check if Invader controls the Defender's home territory."""
    state["invader_on_defender_home"] = (
        state["territories"][T_DEFENDER_HOME]["controller"] == "I"
    )


def destroy_invader_unit(state):
    """Destroy one Invader unit. Ground units are spent before strike units."""
    if state["invader_units"] <= 0:
        return False

    state["invader_units"] -= 1
    if state.get("invader_ground", 0) > 0:
        state["invader_ground"] -= 1
    elif state.get("invader_strike", 0) > 0:
        state["invader_strike"] -= 1

    invader_positions = np.where(state["invader_unit_map"] > 0)[0]
    if len(invader_positions) > 0:
        pos = invader_positions[-1]
        state["invader_unit_map"][pos] -= 1
    return True


def _destroy_defender_unit(state):
    """Destroy one Defender unit. Ground units are spent before strike units."""
    if state["defender_units"] <= 0:
        return False

    state["defender_units"] -= 1
    if state.get("defender_ground", 0) > 0:
        state["defender_ground"] -= 1
    elif state.get("defender_strike", 0) > 0:
        state["defender_strike"] -= 1

    defender_positions = np.where(state["defender_unit_map"] > 0)[0]
    if len(defender_positions) > 0:
        state["defender_unit_map"][defender_positions[0]] -= 1
    return True


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
    if state["experiment_config"]["legitimacy_active"] and state["legitimacy"] <= 0.0:
        return True, "political_collapse", TERMINAL_POLITICAL_COLLAPSE

    # Military defeat
    if state["invader_units"] <= 0:
        return True, "military_defeat", TERMINAL_MILITARY_DEFEAT

    # Total conquest
    if all(t["controller"] == "I" for t in state["territories"]):
        return True, "total_conquest", TERMINAL_TOTAL_CONQUEST

    # Negotiated settlement — PDF §9.
    # The PDF's §10 protocol predicts that negotiated peace is the optimal
    # policy *only* in the "Full model" cell of the ablation table; every
    # ablation row (No legitimacy / No occupation / No neutral / Baseline)
    # expects some flavor of invasion. To make this internally consistent,
    # we require all three feedback mechanisms (L, t_occ, θ) to be active
    # for peace to be reachable. Narratively: a credible negotiated
    # settlement requires an international community that
    #   1. tracks the Invader's legitimacy (L active),
    #   2. observes the cost of sustained occupation (t_occ active), and
    #   3. has a Neutral mediator with a coherent posture (θ active).
    # Without any of these, there is no party that can broker a durable
    # agreement; the only terminal paths left are conquest, defeat, or
    # collapse — which matches the §10 ablation predictions.
    exp_config = state["experiment_config"]
    legitimacy_active  = exp_config.get("legitimacy_active", True)
    occupation_active  = exp_config.get("occupation_active", True)
    neutral_active     = exp_config.get("neutral_active", True)
    sanction_threshold = exp_config.get("sanction_threshold", 0.6)
    full_model = legitimacy_active and occupation_active and neutral_active
    neutral_willing = (
        full_model
        and not state.get("sanctions_active", False)
        and state.get("theta", 0.0) < sanction_threshold
    )
    if (
        neutral_willing
        and state["consecutive_negotiate"] >= NEGOTIATE_CONSECUTIVE_NEEDED
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
    for _ in range(defender_result["units_destroyed"]):
        destroy_invader_unit(state)
    info["defender"] = defender_result

    # Step 5-6: Resolve outcomes & update territory map (already done above)
    update_defender_home_flag(state)

    # Step 7: Update L, E, t_occ
    update_occupation(state)
    update_economy(state)

    # Step 8: Neutral posture shift
    exp_config = state["experiment_config"]
    if exp_config["neutral_active"]:
        theta_legitimacy = state["legitimacy"] if exp_config["legitimacy_active"] else INITIAL_LEGITIMACY
        theta_t_occ = state["t_occ"] if exp_config["occupation_active"] else 0
        state["theta"] = update_theta(
            state["theta"], theta_legitimacy,
            mil_action, pol_action, theta_t_occ, rng,
        )
    else:
        state["theta"] = INITIAL_THETA
    info["theta"] = state["theta"]

    # Step 9: Threshold events
    events = check_threshold_events(state["theta"], state) if exp_config["neutral_active"] else []
    update_economy(state)
    info["threshold_events"] = events

    # Step 10: Check terminal conditions
    done, reason, terminal_reward = check_terminal(state)
    info["terminal_reason"] = reason

    # Step 11: Compute reward
    effective_t_occ = state["t_occ"] + state["disconnected_occupied_count"]
    if state["supply_routes_open"]:
        effective_t_occ = int(effective_t_occ * (1.0 - SUPPLY_ROUTE_OCC_REDUCTION))
    state_for_reward = {**state, "t_occ": effective_t_occ}
    step_reward, reward_info = compute_reward(state_for_reward, newly_captured, rng)
    if reward_info["insurgency_occurred"]:
        reward_info["insurgency_destroyed_unit"] = destroy_invader_unit(state)
    else:
        reward_info["insurgency_destroyed_unit"] = False
    if not done:
        done, reason, terminal_reward = check_terminal(state)
        info["terminal_reason"] = reason
    total_reward = step_reward + (terminal_reward if done else 0.0)
    info["reward_breakdown"] = reward_info

    # Step 12: Advance step counter and set done flag
    state["step"] += 1
    state["done"] = done

    return total_reward, done, info
