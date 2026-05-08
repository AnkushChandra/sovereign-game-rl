"""
main.py — Run and test the SOVEREIGN environment.

This script:
  1. Creates the SovereignEnv
  2. Runs several episodes with random actions
  3. Prints decoded actions, rewards, and state info each step

To use with an RL library later, see the commented example at the bottom.
"""

import numpy as np
from sovereign_env import SovereignEnv
from game_logic import decode_action


def run_random_episode(env, max_steps=50, verbose=True):
    """
    Play one episode with random actions, printing step-by-step info.

    Args:
        env       : SovereignEnv instance
        max_steps : cap for this demo (env also has its own MAX_STEPS)
        verbose   : whether to print each step

    Returns:
        total_reward : float — cumulative reward over the episode
    """
    obs, info = env.reset()
    total_reward = 0.0

    if verbose:
        print("=" * 70)
        print("EPISODE START")
        print("=" * 70)

    for step in range(max_steps):
        action = env.action_space.sample()
        pol, mil = decode_action(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if verbose:
            print(f"\n--- Step {step + 1} ---")
            print(f"  Action {action:>2d}  →  Political: {pol:<18s} Military: {mil}")
            print(f"  Reward: {reward:+.4f}   Cumulative: {total_reward:+.4f}")
            print(f"  L={obs[-5]:.3f}  E={obs[-4]:.3f}  θ={obs[-3]:+.3f}  "
                  f"t_occ={obs[-2]*200:.0f}  step={obs[-1]*200:.0f}")

            # Show defender response
            d = info.get("defender", {})
            if d:
                print(f"  Defender: {d.get('description', '')}")

            # Show threshold events
            events = info.get("threshold_events", [])
            if events:
                print(f"  Events: {', '.join(events)}")

            # Show newly captured territories
            captured = info.get("newly_captured", [])
            if captured:
                print(f"  Captured: {', '.join(captured)}")

            if done:
                reason = info.get("terminal_reason", "unknown")
                print(f"\n  >>> EPISODE ENDED: {reason}  "
                      f"(terminated={terminated}, truncated={truncated})")
                break

    if verbose:
        print(f"\nTotal reward: {total_reward:+.4f}")
        print("=" * 70)

    return total_reward


def main():
    """Run a few demo episodes with random actions."""
    env = SovereignEnv(render_mode=None)

    num_episodes = 3
    rewards = []

    for ep in range(num_episodes):
        print(f"\n{'#' * 70}")
        print(f"# EPISODE {ep + 1} / {num_episodes}")
        print(f"{'#' * 70}")
        r = run_random_episode(env, max_steps=60, verbose=True)
        rewards.append(r)

    print("\n\n===== SUMMARY =====")
    for i, r in enumerate(rewards):
        print(f"  Episode {i + 1}: total reward = {r:+.4f}")
    print(f"  Mean reward: {np.mean(rewards):+.4f}")

    env.close()


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Want to train an RL agent?  Run the DQN trainer:
#
#     python train.py --episodes 300
#
# See dqn_agent.py for the from-scratch DQN implementation.
# ─────────────────────────────────────────────────────────────────────────────
