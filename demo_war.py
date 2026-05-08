"""
demo_war.py — Narrated, step-by-step playthrough of one SOVEREIGN episode.

Runs a *fixed aggressive* Invader policy so you can watch how the game
evolves: territory capture, defender resistance, θ drift, threshold events,
legitimacy collapse, and the terminal outcome.

Usage:
    python demo_war.py
"""

import numpy as np

from sovereign_env import SovereignEnv
from game_logic   import decode_action, encode_action


# ─────────────────────────────────────────────
# Helpers for pretty printing
# ─────────────────────────────────────────────

BAR = "─" * 78

def render_bar(value, v_min, v_max, width=20):
    """Return a small ASCII bar [####----] for a scalar."""
    norm = (value - v_min) / (v_max - v_min)
    norm = max(0.0, min(1.0, norm))
    filled = int(round(norm * width))
    return "█" * filled + "·" * (width - filled)


def print_state_panel(env):
    """Print a compact state summary."""
    s = env._state
    terr = "  ".join(
        f"{t['name'][:3]}:{t['controller']}"
        for t in s["territories"]
    )
    print(f"    map   : {terr}")
    print(f"    L  {s['legitimacy']:.3f}  [{render_bar(s['legitimacy'], 0, 1)}]")
    print(f"    E  {s['economy']:.3f}  [{render_bar(s['economy'], 0, 1)}]")
    print(f"    θ  {s['theta']:+.3f} [{render_bar(s['theta'], -1, 1)}]")
    print(f"    t_occ={s['t_occ']}   I_units={s['invader_units']}   D_units={s['defender_units']}")
    flags = []
    if s['sanctions_active']:   flags.append("SANCTIONS")
    if s['coalition_fired']:    flags.append("NEUTRAL-IN-COALITION")
    if s['supply_routes_open']: flags.append("SUPPLY-ROUTES-OPEN")
    if s['ally_invader_fired']: flags.append("NEUTRAL-ALLIES-I")
    if flags:
        print(f"    flags : {', '.join(flags)}")


# ─────────────────────────────────────────────
# Fixed aggressive policy
# ─────────────────────────────────────────────

def aggressive_policy(obs, step):
    """
    A scripted aggressive policy:
      - Opening (steps 1-3):      ISSUE_THREAT + ADVANCE
      - Mid-game (steps 4-10):    ISSUE_THREAT + STRIKE (on odd), ADVANCE (on even)
      - Late-game (step 11+):     IMPOSE_SANCTION + ADVANCE
    Returns a joint-action integer.
    """
    if step <= 3:
        return encode_action("ISSUE_THREAT", "ADVANCE")
    if step <= 10:
        if step % 2 == 1:
            return encode_action("ISSUE_THREAT", "STRIKE")
        else:
            return encode_action("ISSUE_THREAT", "ADVANCE")
    return encode_action("IMPOSE_SANCTION", "ADVANCE")


# ─────────────────────────────────────────────
# Narrated episode
# ─────────────────────────────────────────────

def run():
    env = SovereignEnv(seed=7)
    obs, _ = env.reset(seed=7)

    print("=" * 78)
    print("  THE SOVEREIGN WAR — Narrated Playthrough")
    print("=" * 78)
    print("\n  Nations at t=0:")
    print("    • Invader  — 12 ground + 3 strike units, home = Invader Home")
    print("    • Defender —  6 ground + 1 strike, home = Defender Home")
    print("    • Neutral  —  4 ground, θ = 0  (true neutral)")
    print("\n  Policy under test: FIXED AGGRESSIVE  (threat-and-advance)\n")
    print("  Initial state:")
    print_state_panel(env)
    print()

    total_reward = 0.0
    step = 0
    done = False

    while not done:
        step += 1
        action = aggressive_policy(obs, step)
        pol, mil = decode_action(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # ── Step header ──
        print(BAR)
        print(f" STEP {step}")
        print(BAR)
        print(f"  Invader plays : {pol}  +  {mil}")

        # Defender reaction
        d = info["defender"]
        print(f"  Defender      : {d['action']:<7s} — {d['description']}")

        # Territory change
        if info["newly_captured"]:
            print(f"  CAPTURED      : {', '.join(info['newly_captured'])}  ")

        # Threshold events
        if info["threshold_events"]:
            for ev in info["threshold_events"]:
                print(f"  ★★★ EVENT    : {ev}")

        # Reward breakdown
        rb = info["reward_breakdown"]
        print(
            f"  reward        : {reward:+.3f}   "
            f"(T={rb['r_territory']:+.2f}  C={rb['r_capture']:+.2f}  "
            f"O={rb['r_occupation']:+.2f}  L={rb['r_legitimacy']:+.2f}  "
            f"S={rb['r_sanction']:+.2f}  I={rb['r_insurgency']:+.2f})"
        )
        if rb["insurgency_occurred"]:
            print(f"                  ** insurgency struck this step **")

        # State panel
        print_state_panel(env)

        if done:
            reason = info.get("terminal_reason", "?")
            print("\n" + "=" * 78)
            print(f"  EPISODE ENDED — reason: {reason.upper()}")
            print(f"  total reward : {total_reward:+.3f}")
            print(f"  steps played : {step}")
            print("=" * 78)
            break


if __name__ == "__main__":
    run()
