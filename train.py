"""
train.py — Train a DQN agent on the SOVEREIGN environment.

Usage:
    python train.py                  # train with defaults from config.py
    python train.py --episodes 500
    python train.py --eval           # only run greedy evaluation from saved model

The trained model is saved to  sovereign_project/dqn_sovereign.pt  by default.
"""

import argparse
import os
import time
from collections import deque

import numpy as np

from sovereign_env import SovereignEnv
from dqn_agent import DQNAgent
from game_logic import decode_action
from config import (
    DQN_TOTAL_EPISODES, DQN_EVAL_EVERY, DQN_EVAL_EPISODES,
    MAX_STEPS, DEFAULT_EXPERIMENT_CONFIG, SECTION10_EXPERIMENTS,
)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "dqn_sovereign.pt")

EXPERIMENT_PRESETS = {
    "full": DEFAULT_EXPERIMENT_CONFIG,
    "no-legitimacy": SECTION10_EXPERIMENTS[1]["config"],
    "no-occupation": SECTION10_EXPERIMENTS[2]["config"],
    "no-neutral": SECTION10_EXPERIMENTS[3]["config"],
    "baseline": SECTION10_EXPERIMENTS[4]["config"],
    "earlier-sanctions": SECTION10_EXPERIMENTS[5]["config"],
    "later-sanctions": SECTION10_EXPERIMENTS[6]["config"],
}


def build_experiment_config(args):
    config = DEFAULT_EXPERIMENT_CONFIG.copy()
    config.update(EXPERIMENT_PRESETS[args.experiment])
    if args.no_legitimacy:
        config["legitimacy_active"] = False
    if args.no_occupation:
        config["occupation_active"] = False
    if args.no_neutral:
        config["neutral_active"] = False
    if args.sanction_threshold is not None:
        config["sanction_threshold"] = args.sanction_threshold
    return config


# ─────────────────────────────────────────────
# Evaluation (greedy, no ε-exploration)
# ─────────────────────────────────────────────

def evaluate(agent, env, n_episodes=DQN_EVAL_EPISODES, verbose=False):
    """
    Run n_episodes with greedy actions and return mean reward + mean length.
    """
    rewards, lengths, reasons = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward, ep_len = 0.0, 0
        done = False
        reason = "unknown"
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            ep_len += 1
            done = terminated or truncated
            if done:
                reason = info.get("terminal_reason", "unknown")
        rewards.append(ep_reward)
        lengths.append(ep_len)
        reasons.append(reason)
        if verbose:
            print(f"    eval ep {ep+1}: reward={ep_reward:+.2f}  len={ep_len}  reason={reason}")
    return np.mean(rewards), np.mean(lengths), reasons


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(total_episodes=DQN_TOTAL_EPISODES, seed=0, save_path=MODEL_PATH,
          experiment_config=None, experiment_name="full"):
    """
    Run the DQN training loop.

    Args:
        total_episodes : number of training episodes
        seed           : RNG seed for reproducibility
        save_path      : where to write the trained model

    Returns:
        history : dict with lists of per-episode metrics
    """
    env = SovereignEnv(seed=seed, experiment_config=experiment_config)
    eval_env = SovereignEnv(seed=seed + 1000, experiment_config=experiment_config)

    obs, _ = env.reset(seed=seed)
    agent = DQNAgent(
        obs_dim=obs.shape[0],
        n_actions=env.action_space.n,
        seed=seed,
    )

    print("=" * 72)
    print(f"Training DQN on SOVEREIGN for {total_episodes} episodes")
    print(f"  experiment={experiment_name}")
    print(f"  config={env._state['experiment_config'] if env._state else experiment_config}")
    print(f"  obs_dim={obs.shape[0]}   n_actions={env.action_space.n}")
    print(f"  device={agent.device}")
    print("=" * 72)

    history = {
        "episode_rewards": [],
        "episode_lengths": [],
        "terminal_reasons": [],
        "eval_rewards": [],
        "losses": [],
    }
    recent_rewards = deque(maxlen=25)
    start_time = time.time()

    for ep in range(1, total_episodes + 1):
        obs, _ = env.reset()
        ep_reward, ep_len, ep_loss = 0.0, 0, []
        done = False
        reason = "unknown"

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.remember(obs, action, reward, next_obs, terminated)  # use terminated (not time-limit)
            loss = agent.learn()
            if loss is not None:
                ep_loss.append(loss)

            obs = next_obs
            ep_reward += reward
            ep_len += 1
            if done:
                reason = info.get("terminal_reason", "unknown")

        history["episode_rewards"].append(ep_reward)
        history["episode_lengths"].append(ep_len)
        history["terminal_reasons"].append(reason)
        history["losses"].append(np.mean(ep_loss) if ep_loss else 0.0)
        recent_rewards.append(ep_reward)

        # Progress print
        if ep % 10 == 0 or ep == 1:
            avg25 = np.mean(recent_rewards)
            elapsed = time.time() - start_time
            print(
                f"ep {ep:>4d}/{total_episodes} | "
                f"R={ep_reward:+7.2f}  avg25={avg25:+7.2f}  "
                f"len={ep_len:3d}  end={reason:<18s}  "
                f"ε={agent.epsilon():.3f}  "
                f"t={elapsed:.1f}s"
            )

        # Periodic greedy evaluation
        if ep % DQN_EVAL_EVERY == 0:
            eval_reward, eval_len, eval_reasons = evaluate(agent, eval_env)
            history["eval_rewards"].append((ep, eval_reward))
            print(
                f"  >> eval @ ep {ep}: mean_reward={eval_reward:+.2f}  "
                f"mean_len={eval_len:.1f}  reasons={eval_reasons}"
            )

    agent.save(save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Total training time: {time.time() - start_time:.1f}s")
    return history, agent


# ─────────────────────────────────────────────
# Demo the trained policy
# ─────────────────────────────────────────────

def demo_greedy_episode(agent, env=None, render=True):
    """Run one greedy episode and print step-by-step info."""
    if env is None:
        env = SovereignEnv()
    obs, _ = env.reset()
    total_reward = 0.0
    print("\n" + "=" * 72)
    print("GREEDY DEMO EPISODE")
    print("=" * 72)

    for step in range(MAX_STEPS):
        action = agent.select_action(obs, greedy=True)
        pol, mil = decode_action(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if render:
            print(
                f"step {step+1:>3d} | {pol:<15s} {mil:<9s} "
                f"r={reward:+.3f}  L={obs[-4]:.2f}  E={obs[-3]:.2f}  "
                f"θ={obs[-2]:+.2f}  t_occ={int(obs[-1]*MAX_STEPS)}"
            )
        if terminated or truncated:
            print(f"\nTerminal: {info.get('terminal_reason')}  "
                  f"total_reward={total_reward:+.2f}")
            break
    return total_reward


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DQN on SOVEREIGN.")
    parser.add_argument("--episodes", type=int, default=DQN_TOTAL_EPISODES,
                        help="Number of training episodes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true",
                        help="Only run evaluation on the saved model.")
    parser.add_argument("--demo", action="store_true",
                        help="Run one greedy demo episode after training/loading.")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to save / load the model.")
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_PRESETS.keys()),
        default="full",
        help="Section 10 preset to train/evaluate.",
    )
    parser.add_argument("--no-legitimacy", action="store_true",
                        help="Disable legitimacy L effects.")
    parser.add_argument("--no-occupation", action="store_true",
                        help="Disable occupation duration t_occ effects.")
    parser.add_argument("--no-neutral", action="store_true",
                        help="Disable neutral posture θ effects.")
    parser.add_argument("--sanction-threshold", type=float, default=None,
                        help="Override the neutral sanctions threshold.")
    args = parser.parse_args()
    experiment_config = build_experiment_config(args)

    if args.eval:
        env = SovereignEnv(seed=args.seed, experiment_config=experiment_config)
        obs, _ = env.reset()
        agent = DQNAgent(obs.shape[0], env.action_space.n, seed=args.seed)
        agent.load(args.model)
        print(f"Evaluating experiment={args.experiment}")
        print(f"config={env._state['experiment_config']}")
        mean_r, mean_len, reasons = evaluate(agent, env, n_episodes=10, verbose=True)
        print(f"\nEval over 10 episodes: mean_reward={mean_r:+.2f}  mean_len={mean_len:.1f}")
        if args.demo:
            demo_greedy_episode(agent, env)
        return

    history, agent = train(
        total_episodes=args.episodes,
        seed=args.seed,
        save_path=args.model,
        experiment_config=experiment_config,
        experiment_name=args.experiment,
    )

    # Final evaluation
    eval_env = SovereignEnv(seed=args.seed + 9999, experiment_config=experiment_config)
    mean_r, mean_len, reasons = evaluate(agent, eval_env, n_episodes=10, verbose=True)
    print(f"\nFinal eval: mean_reward={mean_r:+.2f}  mean_len={mean_len:.1f}")

    if args.demo:
        demo_greedy_episode(agent, eval_env)


if __name__ == "__main__":
    main()
