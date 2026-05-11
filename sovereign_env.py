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
    get_experiment_mechanisms,
)
from game_logic import init_state, execute_turn, decode_action


class SovereignEnv(gym.Env):
    """
    SOVEREIGN: A 3-nation geopolitical strategy environment for DRL research.

    Observation (flat float32 vector, length = NUM_TERRITORIES + 7):
        [0 .. NUM_TERRITORIES-1]  territory control  (0=Contested, 1=I, 2=D, 3=N)
        [NUM_TERRITORIES]         invader_units  (normalised)
        [NUM_TERRITORIES+1]       defender_units (normalised)
        [NUM_TERRITORIES+2]       legitimacy     L  ∈ [0, 1]
        [NUM_TERRITORIES+3]       economy        E  ∈ [0, 1]
        [NUM_TERRITORIES+4]       theta          θ  ∈ [-1, 1]
        [NUM_TERRITORIES+5]       t_occ          (normalised by MAX_STEPS)
        [NUM_TERRITORIES+6]       step           (normalised by MAX_STEPS)

    Action:  Discrete(20) — 5 political × 4 military
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, seed=None, experiment="full", mechanisms=None):
        """Initialise the SOVEREIGN environment."""
        super().__init__()

        self.render_mode = render_mode
        self.experiment = experiment
        self.mechanisms = get_experiment_mechanisms(experiment, mechanisms)

        # Action space: 5 political × 4 military = 20
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Observation space: compact float vector
        obs_size = NUM_TERRITORIES + 7
        self.observation_space = spaces.Box(
            low=-1.0, high=3.0, shape=(obs_size,), dtype=np.float32,
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

        self._state = init_state(
            rng=self._rng,
            experiment=self.experiment,
            mechanisms=self.mechanisms,
        )

        obs = self._build_obs()
        info = {
            "step": 0,
            "message": "Episode started.",
            "experiment": self.experiment,
            "mechanisms": self.mechanisms.copy(),
        }
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

        Controller encoding: Contested=0, I=1, D=2, N=3
        """
        s = self._state
        ctrl_map = {"Contested": 0.0, "I": 1.0, "D": 2.0, "N": 3.0}

        territory_ctrl = np.array(
            [ctrl_map[t["controller"]] for t in s["territories"]],
            dtype=np.float32,
        )

        scalars = np.array([
            s["invader_units"] / 15.0,          # normalised
            s["defender_units"] / 15.0,          # normalised
            s["legitimacy"],
            s["economy"],
            s["theta"],
            s["t_occ"] / MAX_STEPS,
            s["step"] / MAX_STEPS,
        ], dtype=np.float32)

        return np.concatenate([territory_ctrl, scalars])
