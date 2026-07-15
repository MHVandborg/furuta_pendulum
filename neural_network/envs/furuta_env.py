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
  balance (5 floats, all in [-1, 1]):
    [0]  sin(θ₁)     arm angle sine component    — naturally in [-1, 1]
    [1]  cos(θ₁)     arm angle cosine component  — naturally in [-1, 1]
    [2]  pendulum error  (θ₂ − π) wrapped to [-π,π], normalised by π
           → 0 when upright, ±1 when hanging straight down
    [3]  arm angular velocity  ω₁, clipped & normalised by OMEGA1_MAX
    [4]  pendulum angular velocity  ω₂, clipped & normalised by OMEGA2_MAX

  Using (sin, cos) rather than θ₁/π avoids the discontinuity at the ±π
  wrap boundary.  The representation is periodic and smooth everywhere.

  swingup (6 floats, all in [-1, 1]):
    [0–4]  same as balance
    [5]  normalised energy error  E_err / E_MAX
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
  balance:  r = cos(θ_err)
              − arm_coeff · w1_norm² · settle_gate
              − effort_coeff · u²

              settle_gate ramps 0→1 as a leaky counter accumulates: +1 each
              step |ω₂| < W2_SETTLE_RAD_S, −1 otherwise (clamped to
              [0, SETTLE_GRACE_STEPS]), gate = counter / SETTLE_GRACE_STEPS.
              This is a *duration* gate, not an instantaneous one —
              deliberately, because instantaneous state can't tell "genuinely
              at rest" from "about to fall": both look identical for the
              first instant, since ω₂ hasn't had time to build up yet under
              gravity. An instantaneous-ω₂ version of this gate taxed the
              corrective torque needed right after any small disturbance
              (including right after a near-zero-error reset, which also
              starts with near-zero ω₂ by construction) — so PPO learned to
              let *every* episode fall through a full swing before ever
              correcting, regardless of starting error.

              The counter used to hard-reset to 0 on any violation instead of
              decaying by 1. That's exploitable: PPO learned to deliberately
              flick ω₂ just over the threshold every ~40-50 steps, wiping the
              counter before it ever neared SETTLE_GRACE_STEPS (confirmed via
              rollout: counter peaked at 49/100 across an entire 2000-step
              episode) — collecting the "settled" cos(θ_err) reward while
              essentially never paying the arm penalty, via a permanent
              idle/correct/reset limit cycle (the "catch and release"
              pattern). Decaying by 1 instead of resetting to 0 means a
              genuinely mostly-settled trajectory still nets forward over
              multiple cycles and eventually crosses the threshold, while a
              real corrective transient (unsettled for a long stretch) still
              drains it properly — closing the loophole without re-taxing
              legitimate corrections.

              An earlier version of this gate also required |θ_err| to be
              within a few degrees of upright before counting as settled.
              That's exploitable too: for any constant arm speed W, the
              Furuta pendulum has a genuine relative equilibrium at tilt
              ε ≈ L1·W²/G (centrifugal effect balancing gravity), with ω₂ = 0
              there. The agent picked a W whose ε sat just outside the angle
              band, collecting cos(θ_err) ≈ 0.99 forever with the gate
              permanently off despite the arm spinning continuously
              (confirmed via rollout — every seed converged to a ~7° offset
              at a constant ~170°/s spin, matching the formula almost
              exactly). Gating on ω₂ alone closes this: sustained near-zero
              ω₂ under gravity is only physically possible at θ_err ≈ 0° or
              180° (hanging), so this equilibrium now also counts as settled.

              arm_coeff         — env._arm_penalty_coeff   (default 2.0)
              W2_SETTLE_RAD_S   — class constant, default 0.5 rad/s ≈ 29°/s.
              SETTLE_GRACE_STEPS— class constant, default 100 steps (0.2s @ 500Hz).
              effort_coeff      — env._effort_penalty_coeff (default 0.001)
                                   u is the normalised torque command in [−1, 1].
                                   At full torque the cost is 0.001/step — small enough
                                   not to inhibit corrections, large enough to eliminate
                                   gratuitous high-frequency torque chattering on hardware.

  swingup:  r = cos(θ₂ − π)
              +1 when upright, −1 when hanging.
              No arm-velocity penalty — the agent must spin freely.

Episode termination (balance only)
-----------------------------------
  balance resets always start within ±45° of upright with energy already
  directed toward vertical (see reset()), so ever approaching hanging isn't
  a normal recoverable event — it means control has failed outright. The
  episode terminates immediately (terminated=True, distinct from the
  max_steps truncation) once |θ_err| exceeds FAILURE_ANGLE_RAD (90°).

  This replaces relying on per-step reward magnitude to discourage large
  excursions. A continuous penalty, however large, competes against however
  many steps of near-max reward remain in the episode after recovering —
  over a 2000-step episode a ~150-step detour through hanging still nets a
  strongly positive total once the agent recovers, so no penalty scale ever
  reliably eliminates the incentive to "pay the tuition" for a wide swing.
  Terminating removes the entire remaining reward instead of a bounded
  amount, which is what actually makes a large excursion a bad trade.

  swingup has no such termination — starting near hanging and passing
  through every angle, including hanging, is the entire point there.

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

    # Velocity normalisation limits — used for both observations and penalties
    OMEGA1_MAX: float = 4.0 * np.pi   # arm:      ≈ 2 rev/s
    OMEGA2_MAX: float = 10.0 * np.pi  # pendulum: ≈ 5 rev/s

    # Pendulum speed below which it's considered "settled" for the arm-penalty
    # gate — see module docstring under Reward. Absolute (not OMEGA2_MAX-relative)
    # because it needs to reflect genuinely-at-rest, not a fraction of the huge
    # swing-up velocity range.
    W2_SETTLE_RAD_S: float = 0.5   # ≈ 29 deg/s
    SETTLE_GRACE_STEPS: int = 100  # consecutive settled steps (0.2s @ 500Hz) before the arm penalty ramps in

    # Balance mode only: |θ_err| beyond this is treated as a catastrophic
    # failure, ending the episode immediately (see module docstring under
    # Reward / Handoff). Generous enough to allow legitimate overshoot while
    # correcting a worst-case ±45° start, but far short of ever approaching
    # hanging (180°).
    FAILURE_ANGLE_RAD: float = np.radians(90.0)

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
        # Arm angle is encoded as (sin, cos) — 2 inputs — plus pendulum error,
        # arm velocity, pendulum velocity, and (swingup only) energy error.
        obs_size = 6 if mode == "swingup" else 5
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
        self._last_u: float = 0.0  # last normalised torque command, for effort penalty
        self._settled_steps: int = 0  # consecutive steps spent near-upright & near-still

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
            # difficulty=0.0 → ±5°  (easy, just above handoff threshold)
            # difficulty=1.0 → ±180° (fully random)
            # Linear interpolation gives ±45° at difficulty=0.23, matching the
            # swing-up handoff window (±20°) plus a robustness margin.
            max_angle = np.radians(5.0) + self.difficulty * np.radians(175.0)
            theta2_err = self._rng.uniform(-max_angle, max_angle)
            theta2 = np.pi + theta2_err

            # Arm velocity scales with curriculum difficulty too, not just the
            # pendulum angle range above. It used to be sampled at a fixed
            # ±360°/s regardless of difficulty, which meant "stage 0" (±5°
            # pendulum) still threw the full arm-spin disturbance at the agent
            # — not actually easy on both axes, just on one. That's a
            # plausible reason training plateaued at stage 1: the difficulty
            # jump from stage 0→1 tripled the pendulum angle range while the
            # already-large arm disturbance stayed exactly as hard as it'd
            # ever be, so the curriculum's reward thresholds (calibrated
            # assuming a gentle step) didn't match the actual jump in
            # difficulty.
            #
            # Range: OMEGA1_MAX·(0.05 + difficulty·0.45), so difficulty=1.0
            # still reaches the original fixed cap of OMEGA1_MAX/2 (360°/s —
            # the reward-floor-preserving cap described below), while
            # difficulty=0.0 starts gentle (~36°/s) instead of full-strength.
            #
            # The original cap is not arbitrary: with arm_penalty coeff=2.0,
            # any arm speed above OMEGA1_MAX/sqrt(2) ≈ 509 deg/s gives NEGATIVE
            # reward even at perfect pendulum balance, so the agent can never
            # discover what "good" looks like from those starts.
            max_w1 = self.OMEGA1_MAX * (0.05 + self.difficulty * 0.45)
            w1 = self._rng.uniform(-max_w1, max_w1)

            # Pendulum velocity: directed toward upright with enough energy to
            # reach vertical (as in a real swing-up handoff).
            #
            # At angle theta2_err from upright the minimum speed to reach vertical:
            #   ω₂_min = sqrt(2·m₂·g·Lp·(1 − cos(θ_err)) / Jp)
            # A random factor in [0.9, 1.4] covers:
            #   0.9  — just short of upright (hardest, must apply torque immediately)
            #   1.0  — barely makes it (nominal handoff)
            #   1.4  — arrives with excess energy (overshoots slightly)
            # This matches the real handoff physics and prevents the "static drop"
            # failure mode where zero-momentum steep starts are unrecoverable.
            p = self._params
            energy_needed = p.m2 * G * p.Lp * (1.0 - np.cos(theta2_err))
            w2_min = float(np.sqrt(max(0.0, 2.0 * energy_needed / p.Jp)))
            w2_scale = self._rng.uniform(0.9, 1.4)
            # Sign: toward upright — negative w2 if theta2_err > 0, positive if < 0
            w2 = -float(np.sign(theta2_err)) * w2_min * w2_scale if abs(theta2_err) > 1e-6 else 0.0
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
        self._settled_steps = 0

        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Clip and scale torque
        u = float(np.clip(action[0], -1.0, 1.0))
        torque = u * self._params.tau_max
        self._last_u = u  # stored for _reward()

        # Advance physics for control_steps substeps at dt each
        for _ in range(self.control_steps):
            self._state = rk4_step(self._state, torque, self.dt, self._params)

        self._step_count += 1

        obs = self._obs()
        reward = self._reward()

        terminated = False
        if self.mode == "balance":
            theta2_err = _angle_normalize(self._state[1] - np.pi)
            terminated = bool(abs(theta2_err) > self.FAILURE_ANGLE_RAD)
        truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _obs(self) -> np.ndarray:
        """Return the normalised observation vector.

        balance: 5 floats  [sin(θ₁), cos(θ₁), θ₂_err_norm, ω₁_norm, ω₂_norm]
        swingup: 6 floats  [sin(θ₁), cos(θ₁), θ₂_err_norm, ω₁_norm, ω₂_norm, E_err_norm]

        Arm angle is encoded as (sin, cos) rather than θ₁/π to avoid the
        discontinuity at the ±π wrap boundary.  The pair is naturally in
        [-1, 1] and is periodic and smooth everywhere.
        """
        th1, th2, w1, w2 = self._state

        theta2_err = _angle_normalize(th2 - np.pi)  # 0 when upright

        obs = [
            float(np.sin(th1)),
            float(np.cos(th1)),
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
        _, th2, w1, w2 = self._state
        theta2_err = _angle_normalize(th2 - np.pi)  # 0 = upright

        if self.mode == "balance":
            # Gated arm penalty: only active once the PENDULUM has been
            # near-still for a sustained stretch — no angle condition. This
            # has to be a *duration* condition, not an instantaneous one: a
            # state that's about to fall looks identical — for an instant —
            # to a state that's genuinely at rest, because ω₂ hasn't had time
            # to build up yet. Gating on instantaneous ω₂ alone taxed the
            # corrective torque needed right after a small disturbance, so
            # the agent learned to let every episode fall through a full
            # swing before ever correcting.
            #
            # The counter leaks (-1/step) rather than hard-resetting to 0 on
            # a violation — a hard reset let PPO deliberately flick ω₂ over
            # the threshold every ~40-50 steps to wipe the counter for free,
            # settling into a permanent idle/correct/reset limit cycle that
            # never paid the intended penalty (confirmed via rollout: counter
            # peaked at 49/100 across a full episode). Leaking still lets a
            # genuine corrective transient drain it, just not instantly.
            #
            # An earlier version also required |θ_err| < 5° to count as
            # settled. That created an exploit: for any constant arm speed W,
            # the Furuta pendulum has a genuine relative equilibrium at a
            # fixed tilt ε ≈ L1·W²/G (centrifugal effect balancing gravity).
            # The agent picked W ≈ 170°/s → ε ≈ 7°, just outside the 5° band,
            # producing a state with ω₂ = 0 (so cos(θ_err) ≈ 0.99, almost full
            # reward) that permanently sat outside the gate, so the arm penalty
            # never activated despite the arm spinning forever — confirmed via
            # rollout: every seed converged to ~7° offset with a constant ~170°/s
            # spin, matching the equilibrium formula almost exactly. Dropping
            # the angle condition closes this: sustained near-zero ω₂ under
            # gravity alone is only possible at θ_err≈0° or θ_err≈180°
            # (hanging), so this equilibrium now also counts as "settled" and
            # gets taxed like any other residual spin.
            coeff = getattr(self, "_arm_penalty_coeff", 2.0)
            upright = float(np.cos(theta2_err))         # +1 upright, -1 hanging

            is_settled = abs(w2) < self.W2_SETTLE_RAD_S
            if is_settled:
                self._settled_steps = min(self.SETTLE_GRACE_STEPS, self._settled_steps + 1)
            else:
                self._settled_steps = max(0, self._settled_steps - 1)
            gate = min(1.0, self._settled_steps / self.SETTLE_GRACE_STEPS)
            w1_norm = float(np.clip(w1 / self.OMEGA1_MAX, -1.0, 1.0))
            if getattr(self, "_arm_penalty_linear", False):
                arm_penalty = coeff * abs(w1_norm) * gate
            else:
                arm_penalty = coeff * w1_norm ** 2 * gate

            # Control effort penalty: discourages high-frequency torque chattering.
            # u is in [-1, 1]; at full torque the cost is effort_coeff/step —
            # small relative to the cosine reward but enough to smooth real hardware.
            effort_coeff  = getattr(self, "_effort_penalty_coeff", 0.001)
            effort_penalty = effort_coeff * self._last_u ** 2

            return upright - arm_penalty - effort_penalty
        else:
            # No penalties during swing-up — the agent must spin freely.
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
