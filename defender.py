"""
defender.py — Rule-based Defender behavior.

The Defender responds deterministically to Invader actions each step.
Its policy prioritises defense when threatened and holds otherwise.
"""

import numpy as np

from config import DEFENDER_HOME_BONUS


def defender_respond(invader_mil_action, invader_pol_action, state):
    """
    Determine the Defender's response for this step.

    Simple rule-based policy:
        1. If Invader ADVANCEs or STRIKEs → Defender DEFEND (destroys 1 Invader unit
           with a +DEFENDER_HOME_BONUS probability of an additional kill when
           the fight is on the Defender's home territory, per PDF §2.1).
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
    rng = state.get("rng")
    if rng is None:
        rng = np.random.default_rng()

    if invader_mil_action in ("ADVANCE", "STRIKE"):
        if defender_units <= 0:
            return {
                "action": "HOLD",
                "units_destroyed": 0,
                "description": "Defender has no units left; cannot resist.",
            }

        units_destroyed = 1

        # PDF §2.1: Defender has +20% unit effectiveness on its own territory.
        # We model this as a DEFENDER_HOME_BONUS (= 0.20) probability of an
        # additional Invader-unit destruction when combat occurs on the
        # Defender's home territory.
        invader_on_home = state.get("invader_on_defender_home", False)
        if invader_on_home and defender_units >= 2 and rng.random() < DEFENDER_HOME_BONUS:
            units_destroyed += 1

        units_destroyed = min(units_destroyed, invader_units)

        return {
            "action": "DEFEND",
            "units_destroyed": units_destroyed,
            "description": (
                f"Defender resists {invader_mil_action}! "
                f"Destroys {units_destroyed} Invader unit(s)."
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
