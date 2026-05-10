"""
sovereign_env.py — Gymnasium-compatible environment for the SOVEREIGN game.

Usage:
    import gymnasium as gym
    from sovereign_env import SovereignEnv

    env = SovereignEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import (
    NUM_ACTIONS, NUM_TERRITORIES, MAX_STEPS,
)
from game_logic import init_state, execute_turn, decode_action


class SovereignEnv(gym.Env):
    """
    SOVEREIGN: A 3-nation geopolitical strategy environment for DRL research.

    Observation (flat float32 vector, length = NUM_TERRITORIES * 5 + 4):
        territory control M       one-hot |V| x 3 for Invader/Defender/Neutral
        invader unit map U_I      |V| integer vector, normalised
        defender unit map U_D     |V| integer vector, normalised
        legitimacy L              ∈ [0, 1]
        economy E                 ∈ [0, 1]
        neutral posture θ         ∈ [-1, 1]
        occupation duration t_occ normalised by MAX_STEPS

    Action:  Discrete(20) — 5 political × 4 military
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, seed=None, experiment_config=None):
        """Initialise the SOVEREIGN environment."""
        super().__init__()

        self.render_mode = render_mode
        self.experiment_config = experiment_config

        # Action space: 5 political × 4 military = 20
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Observation space: PDF Section 4.1 state variables, flattened
        obs_size = (NUM_TERRITORIES * 3) + (NUM_TERRITORIES * 2) + 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32,
        )

        # Internal state (set on reset)
        self._state = None
        self._rng = np.random.default_rng(seed)

    # ─────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Returns:
            observation : np.ndarray
            info        : dict
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._state = init_state(rng=self._rng, experiment_config=self.experiment_config)

        obs = self._build_obs()
        info = {"step": 0, "message": "Episode started."}
        return obs, info

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action : int — joint political-military action index

        Returns:
            observation : np.ndarray
            reward      : float
            terminated  : bool
            truncated   : bool
            info        : dict
        """
        assert self._state is not None, "Call reset() before step()."
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward, done, info = execute_turn(int(action), self._state)

        obs = self._build_obs()

        # Gymnasium convention: terminated vs truncated
        terminated = done and info.get("terminal_reason") != "time_limit"
        truncated  = done and info.get("terminal_reason") == "time_limit"

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Print a human-readable summary of the current state."""
        if self._state is None:
            print("[SovereignEnv] No state — call reset() first.")
            return

        s = self._state
        terr_summary = "  ".join(
            f"{t['name']}:{t['controller']}" for t in s["territories"]
        )
        print(
            f"Step {s['step']:>3d} | "
            f"L={s['legitimacy']:.2f}  E={s['economy']:.2f}  "
            f"D_E={s['defender_economy']:.2f}  "
            f"θ={s['theta']:+.3f}  t_occ={s['t_occ']}  "
            f"I_units={s['invader_units']}  D_units={s['defender_units']}"
        )
        print(f"  Map: {terr_summary}")
        if s["sanctions_active"]:
            print("  *** SANCTIONS ACTIVE ***")
        if s["coalition_fired"]:
            print("  *** NEUTRAL JOINED DEFENDER ***")

    def close(self):
        """Clean up (nothing to do for this env)."""
        pass

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _build_obs(self):
        """
        Convert internal state dict to a flat numpy observation vector.
        """
        s = self._state
        ctrl_map = {"I": 0, "D": 1, "N": 2}
        territory_ctrl = np.zeros((NUM_TERRITORIES, 3), dtype=np.float32)
        for territory in s["territories"]:
            controller = territory["controller"]
            if controller in ctrl_map:
                territory_ctrl[territory["id"], ctrl_map[controller]] = 1.0

        invader_units = s["invader_unit_map"].astype(np.float32) / 15.0
        defender_units = s["defender_unit_map"].astype(np.float32) / 15.0

        scalars = np.array([
            s["legitimacy"],
            s["economy"],
            s["theta"],
            s["t_occ"] / MAX_STEPS,
        ], dtype=np.float32)

        return np.concatenate([territory_ctrl.flatten(), invader_units, defender_units, scalars])
