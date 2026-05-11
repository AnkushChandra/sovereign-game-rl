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
from collections import Counter, deque

import numpy as np

import config as config_module
import game_logic as game_logic_module
import neutral as neutral_module
import reward as reward_module
import sovereign_env as sovereign_env_module
from sovereign_env import SovereignEnv
from dqn_agent import DQNAgent
from game_logic import decode_action
from config import (
    DQN_TOTAL_EPISODES, DQN_EVAL_EVERY, DQN_EVAL_EPISODES,
    MAX_STEPS, EXPERIMENT_PRESETS,
)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "dqn_sovereign.pt")
CONFIG_PROFILES = ("tuned", "original")


def apply_config_profile(profile):
    """Apply a named config profile before environments are created."""
    if profile == "tuned":
        config_module.ACTIVE_CONFIG_PROFILE = "tuned"
        return

    if profile != "original":
        raise ValueError(f"Unknown config profile: {profile}")

    from config_original import OVERRIDES

    modules = [
        config_module,
        game_logic_module,
        neutral_module,
        reward_module,
        sovereign_env_module,
    ]

    for name, value in OVERRIDES.items():
        for module in modules:
            if hasattr(module, name):
                setattr(module, name, value)
        if name in globals():
            globals()[name] = value

    config_module.ACTIVE_CONFIG_PROFILE = "original"


# ─────────────────────────────────────────────
# Evaluation (greedy, no ε-exploration)
# ─────────────────────────────────────────────

def count_invader_territories(env):
    """Count territories currently controlled by the Invader."""
    return sum(
        t["controller"] == "I"
        for t in env._state["territories"]
    )


def classify_invasion(metric):
    """Classify invasion behavior for one evaluated episode."""
    if metric["max_invader_territories"] <= metric["start_invader_territories"]:
        return "no_invasion"
    if metric["terminal_reason"] == "total_conquest":
        return "fast_conquest" if metric["length"] <= 15 else "slow_conquest"
    return "partial_invasion"


def summarize_metrics(metrics):
    """Aggregate per-episode behavior metrics into compact eval statistics."""
    if not metrics:
        return {}

    first_advances = [
        m["first_advance_step"]
        for m in metrics
        if m["first_advance_step"] is not None
    ]
    return {
        "classes": Counter(m["invasion_class"] for m in metrics),
        "avg_advances": float(np.mean([m["advance_count"] for m in metrics])),
        "avg_strikes": float(np.mean([m["strike_count"] for m in metrics])),
        "avg_negotiates": float(np.mean([m["negotiate_count"] for m in metrics])),
        "avg_max_invader_territories": float(np.mean([
            m["max_invader_territories"] for m in metrics
        ])),
        "avg_first_advance": (
            float(np.mean(first_advances)) if first_advances else None
        ),
    }


def format_metric_summary(summary):
    """Return one readable line for eval behavior metrics."""
    if not summary:
        return "metrics unavailable"

    classes = ", ".join(
        f"{name}={count}"
        for name, count in sorted(summary["classes"].items())
    )
    first_adv = summary["avg_first_advance"]
    first_adv_text = f"{first_adv:.1f}" if first_adv is not None else "none"
    return (
        f"class[{classes}]  "
        f"adv={summary['avg_advances']:.1f}  "
        f"strike={summary['avg_strikes']:.1f}  "
        f"neg={summary['avg_negotiates']:.1f}  "
        f"max_I={summary['avg_max_invader_territories']:.1f}  "
        f"first_adv={first_adv_text}"
    )


def evaluate(agent, env, n_episodes=DQN_EVAL_EPISODES, verbose=False):
    """
    Run n_episodes with greedy actions and return reward, length, and behavior metrics.
    """
    rewards, lengths, reasons, metrics = [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward, ep_len = 0.0, 0
        done = False
        reason = "unknown"
        first_advance_step = None
        advance_count = 0
        strike_count = 0
        negotiate_count = 0
        start_invader_territories = count_invader_territories(env)
        max_invader_territories = start_invader_territories
        while not done:
            action = agent.select_action(obs, greedy=True)
            pol, mil = decode_action(action)
            if mil == "ADVANCE":
                advance_count += 1
                if first_advance_step is None:
                    first_advance_step = ep_len + 1
            elif mil == "STRIKE":
                strike_count += 1
            if pol == "NEGOTIATE":
                negotiate_count += 1

            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            ep_len += 1
            max_invader_territories = max(
                max_invader_territories,
                count_invader_territories(env),
            )
            done = terminated or truncated
            if done:
                reason = info.get("terminal_reason", "unknown")
        rewards.append(ep_reward)
        lengths.append(ep_len)
        reasons.append(reason)
        metric = {
            "length": ep_len,
            "terminal_reason": reason,
            "first_advance_step": first_advance_step,
            "advance_count": advance_count,
            "strike_count": strike_count,
            "negotiate_count": negotiate_count,
            "start_invader_territories": start_invader_territories,
            "max_invader_territories": max_invader_territories,
            "final_invader_territories": count_invader_territories(env),
        }
        metric["invasion_class"] = classify_invasion(metric)
        metrics.append(metric)
        if verbose:
            print(
                f"    eval ep {ep+1}: reward={ep_reward:+.2f}  len={ep_len}  "
                f"reason={reason}  class={metric['invasion_class']}  "
                f"adv={advance_count}  strike={strike_count}  "
                f"neg={negotiate_count}  max_I={max_invader_territories}  "
                f"first_adv={first_advance_step}"
            )
    return np.mean(rewards), np.mean(lengths), reasons, summarize_metrics(metrics)


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(total_episodes=DQN_TOTAL_EPISODES, seed=0, save_path=MODEL_PATH,
          experiment="full"):
    """
    Run the DQN training loop.

    Args:
        total_episodes : number of training episodes
        seed           : RNG seed for reproducibility
        save_path      : where to write the trained model

    Returns:
        history : dict with lists of per-episode metrics
    """
    env = SovereignEnv(seed=seed, experiment=experiment)
    eval_env = SovereignEnv(seed=seed + 1000, experiment=experiment)

    obs, _ = env.reset(seed=seed)
    agent = DQNAgent(
        obs_dim=obs.shape[0],
        n_actions=env.action_space.n,
        seed=seed,
    )

    print("=" * 72)
    print(f"Training DQN on SOVEREIGN for {total_episodes} episodes")
    print(f"  config={getattr(config_module, 'ACTIVE_CONFIG_PROFILE', 'tuned')}")
    print(f"  experiment={experiment}   mechanisms={env.mechanisms}")
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
            eval_reward, eval_len, eval_reasons, eval_metrics = evaluate(agent, eval_env)
            history["eval_rewards"].append((ep, eval_reward))
            print(
                f"  >> eval @ ep {ep}: mean_reward={eval_reward:+.2f}  "
                f"mean_len={eval_len:.1f}  reasons={eval_reasons}"
            )
            print(f"     behavior: {format_metric_summary(eval_metrics)}")

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
                f"r={reward:+.3f}  L={obs[-5]:.2f}  E={obs[-4]:.2f}  "
                f"θ={obs[-3]:+.2f}  t_occ={int(obs[-2]*MAX_STEPS)}"
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
    parser.add_argument("--experiment", type=str, default="full",
                        choices=sorted(EXPERIMENT_PRESETS),
                        help="Section 10 experiment preset.")
    parser.add_argument("--config", type=str, default="tuned",
                        choices=CONFIG_PROFILES,
                        help="Config profile: tuned=current config.py, original=config_original.py.")
    parser.add_argument("--eval", action="store_true",
                        help="Only run evaluation on the saved model.")
    parser.add_argument("--demo", action="store_true",
                        help="Run one greedy demo episode after training/loading.")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to save / load the model.")
    args = parser.parse_args()
    apply_config_profile(args.config)

    if args.eval:
        env = SovereignEnv(seed=args.seed, experiment=args.experiment)
        obs, _ = env.reset()
        agent = DQNAgent(obs.shape[0], env.action_space.n, seed=args.seed)
        agent.load(args.model)
        mean_r, mean_len, reasons, metrics = evaluate(
            agent, env, n_episodes=10, verbose=True,
        )
        print(f"\nEval over 10 episodes: mean_reward={mean_r:+.2f}  mean_len={mean_len:.1f}")
        print(f"Behavior: {format_metric_summary(metrics)}")
        if args.demo:
            demo_greedy_episode(agent, env)
        return

    history, agent = train(
        total_episodes=args.episodes,
        seed=args.seed,
        save_path=args.model,
        experiment=args.experiment,
    )

    # Final evaluation
    eval_env = SovereignEnv(seed=args.seed + 9999, experiment=args.experiment)
    mean_r, mean_len, reasons, metrics = evaluate(
        agent, eval_env, n_episodes=10, verbose=True,
    )
    print(f"\nFinal eval: mean_reward={mean_r:+.2f}  mean_len={mean_len:.1f}")
    print(f"Behavior: {format_metric_summary(metrics)}")

    if args.demo:
        demo_greedy_episode(agent, eval_env)


if __name__ == "__main__":
    main()
