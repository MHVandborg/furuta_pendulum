"""
Gymnasium environment for the Furuta (rotary inverted) pendulum.

Two training modes
------------------
  'balance'  — keep the pendulum upright.
               Starts near vertical; difficulty widens the starting angle.
  'swingup'  — pump energy from hanging and reach vertical.
               Always starts near hanging (θ₂ ≈ 0) with a small random kick.

Observation
-----------
  balance (4 floats, all in [-1, 1]):
    [0]  arm angle   θ₁, wrapped to [-π,π], normalised by π
    [1]  pendulum error  (θ₂ − π) wrapped to [-π,π], normalised by π
           → 0 when upright, ±1 when hanging straight down
    [2]  arm angular velocity  ω₁, clipped & normalised by OMEGA1_MAX
    [3]  pendulum angular velocity  ω₂, clipped & normalised by OMEGA2_MAX

  swingup (5 floats, all in [-1, 1]):
    [0–3]  same as balance
    [4]  normalised energy error  E_err / E_MAX
           E_err = E_pendulum − E_upright
           0 when the pendulum has exactly the energy to reach vertical.
           Negative when under-energised (needs more pumping).
           Positive when over-energised (has passed through vertical).

  The energy input is only given to the swing-up network because near the
  balance point E_err ≈ 0 by definition — it carries no additional
  information the balance network doesn't already have from θ and ω.

Action (1 float, [-1, 1])
--------------------------
  Normalised torque command — scaled by params.tau_max inside step().

Reward
------
  balance:  r = cos(θ_err) − 0.01·ω₁²
              +1 when perfectly upright and still.
              Smooth gradient everywhere — cos gives useful signal even at 90°+.
              The 0.01 factor keeps the arm-velocity penalty small relative to
              the angle reward until ω₁ exceeds ~10 rad/s (≈1.6 rev/s).

  swingup:  r = cos(θ₂ − π)
              +1 when upright, −1 when hanging.
              No arm-velocity penalty — the agent must spin freely.

Handoff condition (firmware, not training)
------------------------------------------
  Switch from swing-up to balance when BOTH:
    |θ₂ − π| < HANDOFF_ANGLE_RAD   (geometrically near vertical)
    E_err     > −HANDOFF_ENERGY_TOL (has enough energy to stay there)

  Using both conditions prevents handing off when the pendulum is "near"
  vertical but already falling back down with insufficient energy.

Curriculum (balance only)
--------------------------
  The `difficulty` attribute (0.0–1.0) controls the starting angle range.

    difficulty = 0.0  → reset within ±5° of upright
    difficulty = 1.0  → fully random reset (any angle)

  Updated by the CurriculumCallback in train.py as reward improves.

Domain randomisation
--------------------
  When `domain_randomisation=True`, physical parameters are re-sampled
  within ±20% of nominal on every reset().  This forces the policy to
  generalise across small modelling errors and reduces the sim-to-real gap.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .physics import FurutaParams, rk4_step, G


def _angle_normalize(x: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-π, π]."""
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def _pendulum_energy(th2: float, w2: float, p: FurutaParams) -> float:
    """Total mechanical energy of the pendulum about its pivot (J).

    E = ½·Jp·ω₂²  −  mp·g·Lp·cos(θ₂)

    Reference zero is at the pivot.  At θ₂=π (upright) with ω₂=0:
      E_upright = mp·g·Lp  (positive, pendulum CoM above pivot)
    At θ₂=0 (hanging) with ω₂=0:
      E_hanging = −mp·g·Lp (negative, CoM below pivot)
    """
    kinetic  = 0.5 * p.Jp * w2 ** 2
    potential = -p.m2 * G * p.Lp * np.cos(th2)
    return kinetic + potential


def _energy_upright(p: FurutaParams) -> float:
    """Energy the pendulum must have to just reach vertical (ω₂=0 at top)."""
    return p.m2 * G * p.Lp


# Handoff thresholds exported for use in firmware comments / future evaluate.py
HANDOFF_ANGLE_RAD: float = np.radians(20.0)   # |θ₂ − π| must be below this
HANDOFF_ENERGY_TOL: float = 0.05              # E_err must be above −this (J)


class FurutaEnv(gym.Env):
    """Single Furuta pendulum Gymnasium environment.

    Parameters
    ----------
    mode            'balance' or 'swingup' — selects reward function and
                    reset distribution (see module docstring).
    dt              Physics integration timestep (s). Default 1 ms.
    control_steps   Number of physics steps per gym step.
                    Default 2 → 500 Hz effective control loop.
    max_steps       Episode length in gym steps (2000 × 2ms = 4 s).
    difficulty      Curriculum parameter in [0, 1]. Only used in 'balance'
                    mode: 0 = easy (near upright), 1 = full random.
    domain_randomisation
                    Re-sample physical params on each reset().
    render_mode     "human" opens a live matplotlib window.
                    "rgb_array" returns pixel arrays from render().
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Velocity normalisation limits
    OMEGA1_MAX: float = 4.0 * np.pi   # arm:      ≈ 2 rev/s
    OMEGA2_MAX: float = 10.0 * np.pi  # pendulum: ≈ 5 rev/s

    def __init__(
        self,
        mode: str = "balance",
        dt: float = 1e-3,
        control_steps: int = 2,
        max_steps: int = 2000,
        difficulty: float = 0.0,
        domain_randomisation: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        assert mode in ("balance", "swingup"), f"mode must be 'balance' or 'swingup', got '{mode}'"
        self.mode = mode
        self.dt = dt
        self.control_steps = control_steps
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.domain_randomisation = domain_randomisation
        self.render_mode = render_mode

        self._rng = np.random.default_rng()
        self._nominal_params = FurutaParams.nominal()
        self._params = self._nominal_params  # replaced on reset()

        # ------------------------------------------------------------------ #
        # Gymnasium spaces                                                     #
        # ------------------------------------------------------------------ #
        # Swing-up gets one extra input: normalised energy error E_err.
        obs_size = 5 if mode == "swingup" else 4
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Internal state
        self._state: np.ndarray = np.zeros(4)
        self._step_count: int = 0

        # Renderer (lazy initialised)
        self._fig = None
        self._ax = None

    # ------------------------------------------------------------------ #
    # Core API                                                             #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Optionally re-sample physical parameters
        if self.domain_randomisation:
            self._params = self._nominal_params.randomised(self._rng)
        else:
            self._params = self._nominal_params

        if self.mode == "balance":
            # Start within ±30° of upright with physically realistic velocities.
            # The handoff from swing-up always happens near vertical (±20°) with
            # the pendulum still moving upward — so velocity is small and bounded.
            # We train slightly wider (±30°) than the handoff angle for robustness.
            max_angle = np.radians(30.0)
            theta2_err = self._rng.uniform(-max_angle, max_angle)
            theta2 = np.pi + theta2_err
            # Arm: moderate velocity from swing-up spinning
            w1 = self._rng.uniform(-3.0, 3.0)
            # Pendulum: small velocity — near top it has low speed
            w2 = self._rng.uniform(-1.5, 1.5)
        else:
            # Swing-up: always start near hanging (θ₂ ≈ 0) with a small random kick.
            # The agent's job is to reach vertical from the natural resting position.
            theta2 = self._rng.uniform(-np.radians(15.0), np.radians(15.0))
            w1 = 0.0
            w2 = 0.0

        self._state = np.array([
            self._rng.uniform(-np.pi, np.pi),   # θ₁: random arm angle
            theta2,                              # θ₂: mode-dependent start
            w1 if self.mode == "balance" else 0.0,
            w2 if self.mode == "balance" else 0.0,
        ])
        self._step_count = 0

        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Clip and scale torque
        torque = float(np.clip(action[0], -1.0, 1.0)) * self._params.tau_max

        # Advance physics for control_steps substeps at dt each
        for _ in range(self.control_steps):
            self._state = rk4_step(self._state, torque, self.dt, self._params)

        self._step_count += 1

        obs = self._obs()
        reward = self._reward()
        terminated = False  # no hard failure state — let the episode run
        truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _obs(self) -> np.ndarray:
        """Return the normalised observation vector.

        balance: 4 floats  [θ₁_norm, θ₂_err_norm, ω₁_norm, ω₂_norm]
        swingup: 5 floats  [θ₁_norm, θ₂_err_norm, ω₁_norm, ω₂_norm, E_err_norm]
        """
        th1, th2, w1, w2 = self._state

        theta2_err = _angle_normalize(th2 - np.pi)  # 0 when upright

        obs = [
            _angle_normalize(th1) / np.pi,
            theta2_err / np.pi,
            np.clip(w1 / self.OMEGA1_MAX, -1.0, 1.0),
            np.clip(w2 / self.OMEGA2_MAX, -1.0, 1.0),
        ]

        if self.mode == "swingup":
            E_err = _pendulum_energy(th2, w2, self._params) - _energy_upright(self._params)
            # Normalise: ±E_MAX covers the full range from hanging to upright (×2 for over-swing)
            E_MAX = 2.0 * _energy_upright(self._params)
            obs.append(float(np.clip(E_err / E_MAX, -1.0, 1.0)))

        return np.array(obs, dtype=np.float32)

    def _reward(self) -> float:
        """Compute per-step reward based on the training mode."""
        _, th2, w1, _ = self._state
        theta2_err = _angle_normalize(th2 - np.pi)  # 0 = upright

        if self.mode == "balance":
            # Cosine reward: +1 upright, 0 at 90°, -1 hanging.
            # Smooth gradient everywhere — works from any starting angle.
            # Small arm-velocity penalty discourages unnecessary spinning.
            return float(np.cos(theta2_err)) - 0.01 * w1 ** 2
        else:
            # Cosine reward: +1 when upright, -1 when hanging.
            # No arm-velocity penalty — the agent must spin freely to pump energy.
            return float(np.cos(theta2_err))

    # ------------------------------------------------------------------ #
    # Rendering                                                            #
    # ------------------------------------------------------------------ #

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(5, 5))

        ax = self._ax
        ax.clear()
        ax.set_xlim(-0.45, 0.45)
        ax.set_ylim(-0.45, 0.45)
        ax.set_aspect("equal")
        ax.set_title(f"Furuta Pendulum   step {self._step_count}")
        ax.axhline(0, color="lightgray", linewidth=0.8)
        ax.axvline(0, color="lightgray", linewidth=0.8)

        th1, th2 = self._state[0], self._state[1]
        p = self._params

        # Arm endpoint (in horizontal plane — we project to 2D top-down)
        arm_tip = np.array([p.L1 * np.cos(th1), p.L1 * np.sin(th1)])
        ax.plot([0, arm_tip[0]], [0, arm_tip[1]], "b-", linewidth=3, label="arm")
        ax.plot(*arm_tip, "bo", markersize=6)

        # Pendulum: shown as a line in the arm's tangential plane, projected
        # onto the XY plane for simplicity
        pend_dir = np.array([
            np.cos(th1) * np.sin(th2),
            np.sin(th1) * np.sin(th2),
        ])
        pend_tip = arm_tip + p.L2 * pend_dir
        ax.plot(
            [arm_tip[0], pend_tip[0]],
            [arm_tip[1], pend_tip[1]],
            "r-", linewidth=2, label="pendulum",
        )
        ax.plot(*pend_tip, "ro", markersize=5)

        # Motor
        motor = mpatches.Circle((0, 0), 0.015, color="gray", zorder=5)
        ax.add_patch(motor)

        ax.legend(loc="upper right", fontsize=8)

        if self.render_mode == "human":
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.001)
            return None

        # rgb_array
        self._fig.canvas.draw()
        buf = self._fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[..., :3]
        return frame

    # ------------------------------------------------------------------ #
    # Curriculum setters (called via VecEnv.env_method() from callbacks)  #
    # ------------------------------------------------------------------ #

    def set_difficulty(self, difficulty: float) -> None:
        self.difficulty = difficulty

    def set_max_steps(self, max_steps: int) -> None:
        self.max_steps = max_steps

    def close(self) -> None:
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None
