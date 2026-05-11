"""
defender.py — Rule-based Defender behavior.

The Defender responds to Invader actions each step.
Its policy prioritises defense when threatened and holds otherwise.
"""

from config import DEFENDER_HOME_BONUS


def defender_respond(invader_mil_action, invader_pol_action, state):
    """
    Determine the Defender's response for this step.

    Simple rule-based policy:
        1. If Invader ADVANCEs or STRIKEs → Defender DEFEND (destroys 1 Invader unit
           with probability boosted on home territory).
        2. If Defender units are low (≤ 3) → Defender HOLDs (conserves forces).
        3. Otherwise → Defender PATROLs (no effect, passive stance).

    Args:
        invader_mil_action : str   — Invader's military action this step
        invader_pol_action : str   — Invader's political action (unused for now)
        state              : dict  — current game state

    Returns:
        response : dict with keys:
            action          : str   — "DEFEND", "HOLD", or "PATROL"
            units_destroyed : int   — Invader units destroyed this step
            description     : str   — human-readable summary
    """
    defender_units = state["defender_units"]
    invader_units = state["invader_units"]

    # ── Aggressive Invader → Defender fights back ──
    if invader_mil_action in ("ADVANCE", "STRIKE"):
        if defender_units <= 0:
            return {
                "action": "HOLD",
                "units_destroyed": 0,
                "description": "Defender has no units left; cannot resist.",
            }

        rng = state["rng"]

        # Base resistance destroys one Invader unit.
        units_destroyed = 1

        # Home-turf advantage: 20% chance to destroy one additional unit.
        invader_on_home = state.get("invader_on_defender_home", False)
        home_bonus_fired = False
        if invader_on_home and defender_units >= 2 and rng.random() < DEFENDER_HOME_BONUS:
            units_destroyed += 1
            home_bonus_fired = True

        # Cap: can't destroy more Invader units than exist
        units_destroyed = min(units_destroyed, invader_units)

        return {
            "action": "DEFEND",
            "units_destroyed": units_destroyed,
            "description": (
                f"Defender resists {invader_mil_action}! "
                f"Destroys {units_destroyed} Invader unit(s)."
                + (" Home-turf bonus triggered." if home_bonus_fired else "")
            ),
        }

    # ── Defender is weak → HOLD ──
    if defender_units <= 3:
        return {
            "action": "HOLD",
            "units_destroyed": 0,
            "description": "Defender forces are low; holding positions.",
        }

    # ── Default: passive patrol ──
    return {
        "action": "PATROL",
        "units_destroyed": 0,
        "description": "Defender patrols; no engagement.",
    }
