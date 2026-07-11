"""
Real-time 3D animation of the Furuta pendulum balance controller.

Run from the neural_network/ directory:

    python visualize.py            # default seed 1 (~30° start)
    python visualize.py --seed 7   # easy start (~15°)
    python visualize.py --seed 8   # hard start (~45°)

The animation loops continuously. Close the window to quit.
"""

import argparse
import sys

sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from stable_baselines3 import PPO

from envs import FurutaEnv


# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #

def pendulum_geometry(th1, th2, p):
    """Return (origin, arm_tip, pend_tip) as (3,) float arrays.

    Coordinate frame:
      - Motor shaft at origin, pointing up (Z).
      - Arm rotates in the XY horizontal plane.
      - Pendulum swings in the vertical plane that contains the arm direction.

    θ₂ = 0 → hanging straight down   (pend_tip directly below arm_tip)
    θ₂ = π → balanced upright        (pend_tip directly above arm_tip)
    """
    origin = np.zeros(3)
    arm_tip = np.array([p.L1 * np.cos(th1),
                        p.L1 * np.sin(th1),
                        0.0])
    pend_tip = arm_tip + p.L2 * np.array([
        np.sin(th2) * np.cos(th1),   # X: extends along arm direction
        np.sin(th2) * np.sin(th1),   # Y: extends along arm direction
        -np.cos(th2),                 # Z: +1 when upright, -1 when hanging
    ])
    return origin, arm_tip, pend_tip


# --------------------------------------------------------------------------- #
# Episode runner                                                                #
# --------------------------------------------------------------------------- #

def run_episode(seed: int, difficulty: float = 0.23,
                model_path: str = "models/balance_best/best_model.zip"):
    model = PPO.load(model_path)
    env = FurutaEnv(mode="balance", difficulty=difficulty, domain_randomisation=False)
    obs, _ = env.reset(seed=seed)
    p = env._params

    states, rewards, torques = [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        states.append(env._state.copy())
        rewards.append(r)
        torques.append(float(action[0]))
        done = term or trunc

    return np.array(states), np.array(rewards), np.array(torques), p


# --------------------------------------------------------------------------- #
# Animation                                                                    #
# --------------------------------------------------------------------------- #

# Show every Nth control step — keeps animation close to real-time at ~30 fps.
# Control dt = 2 ms, so FRAME_SKIP=16 → 32 ms/frame ≈ 31 fps real-time.
FRAME_SKIP = 16
DT_CTRL    = 2e-3   # seconds per control step


def build_animation(seed: int, model_path: str = "models/balance_best/best_model.zip"):
    print(f"Running episode (seed={seed}, model={model_path}) …")
    states, rewards, torques, p = run_episode(seed=seed, model_path=model_path)
    cum_rewards = np.cumsum(rewards)
    n_frames = len(states) // FRAME_SKIP

    start_err = np.degrees(((states[0, 1] - np.pi + np.pi) % (2 * np.pi)) - np.pi)
    print(f"  start angle: {start_err:+.1f}°   total reward: {rewards.sum():.0f} / {len(rewards)}")

    # ------------------------------------------------------------------ #
    # Figure layout                                                        #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor("#1a1a2e")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1a2e")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333355")
    ax.yaxis.pane.set_edgecolor("#333355")
    ax.zaxis.pane.set_edgecolor("#333355")
    ax.tick_params(colors="gray", labelsize=7)
    ax.xaxis.label.set_color("gray")
    ax.yaxis.label.set_color("gray")
    ax.zaxis.label.set_color("gray")

    R = p.L1 + p.L2 + 0.02
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(-p.L2 - 0.02, p.L2 + 0.05)
    ax.set_xlabel("X (m)", labelpad=4)
    ax.set_ylabel("Y (m)", labelpad=4)
    ax.set_zlabel("Z (m)", labelpad=4)
    ax.set_title("Furuta Pendulum — Balance Controller", color="white", pad=10)
    ax.view_init(elev=22, azim=45)

    # Ground ring
    gc = np.linspace(0, 2 * np.pi, 120)
    ax.plot(R * np.cos(gc), R * np.sin(gc), -p.L2 - 0.01,
            color="#334", linewidth=0.8, zorder=0)

    # Motor at origin
    ax.scatter([0], [0], [0], color="#aaaaaa", s=120, zorder=5, depthshade=False)

    # ------------------------------------------------------------------ #
    # Dynamic artists                                                      #
    # ------------------------------------------------------------------ #
    arm_line,  = ax.plot([], [], [], color="#4488ff", linewidth=5,
                         solid_capstyle="round", zorder=4)
    pend_line, = ax.plot([], [], [], color="#ff4444", linewidth=3.5,
                         solid_capstyle="round", zorder=4)
    arm_joint, = ax.plot([], [], [], "o", color="#4488ff",
                         markersize=9, zorder=6)
    pend_bob,  = ax.plot([], [], [], "o", color="#ff4444",
                         markersize=12, zorder=6)

    # Upright reference: dashed green line above the arm pivot
    ref_line,  = ax.plot([], [], [], color="#44ff88", linewidth=1.2,
                         linestyle="--", alpha=0.55, zorder=3)

    # Ground shadows (projected onto the floor plane)
    arm_shadow,  = ax.plot([], [], [], color="#336",  linewidth=2, alpha=0.3, zorder=1)
    pend_shadow, = ax.plot([], [], [], color="#633",  linewidth=1.5, alpha=0.3, zorder=1)

    # ------------------------------------------------------------------ #
    # Info overlay (2D text on the axes)                                  #
    # ------------------------------------------------------------------ #
    info_text = ax.text2D(
        0.02, 0.97, "",
        transform=ax.transAxes,
        color="white", fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#111122", alpha=0.7),
    )

    # Torque indicator bar (small inset axis, bottom-right)
    ax_tau = fig.add_axes([0.80, 0.18, 0.035, 0.32])
    ax_tau.set_facecolor("#111122")
    ax_tau.set_ylim(-1.05, 1.05)
    ax_tau.set_xlim(0, 1)
    ax_tau.axhline(0, color="#555588", linewidth=0.8)
    ax_tau.set_xticks([])
    ax_tau.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_tau.tick_params(colors="gray", labelsize=7)
    ax_tau.set_title("τ (norm)", color="gray", fontsize=8, pad=4)
    for spine in ax_tau.spines.values():
        spine.set_edgecolor("#333355")
    tau_bar = ax_tau.bar([0.5], [0.0], width=0.7, color="#ff8800",
                         align="center", bottom=0.0)

    # ------------------------------------------------------------------ #
    # Update function                                                      #
    # ------------------------------------------------------------------ #
    def update(frame):
        idx = min(frame * FRAME_SKIP, len(states) - 1)
        th1, th2, w1, w2 = states[idx]
        origin, a_tip, p_tip = pendulum_geometry(th1, th2, p)
        floor_z = -p.L2 - 0.01

        # Arm
        arm_line.set_data([origin[0], a_tip[0]], [origin[1], a_tip[1]])
        arm_line.set_3d_properties([origin[2], a_tip[2]])

        # Pendulum
        pend_line.set_data([a_tip[0], p_tip[0]], [a_tip[1], p_tip[1]])
        pend_line.set_3d_properties([a_tip[2], p_tip[2]])

        # Joints / bob
        arm_joint.set_data([a_tip[0]], [a_tip[1]])
        arm_joint.set_3d_properties([a_tip[2]])
        pend_bob.set_data([p_tip[0]], [p_tip[1]])
        pend_bob.set_3d_properties([p_tip[2]])

        # Upright reference above pivot (dashed vertical target line)
        ref_line.set_data([a_tip[0], a_tip[0]], [a_tip[1], a_tip[1]])
        ref_line.set_3d_properties([0.0, p.L2])

        # Shadows on ground plane
        arm_shadow.set_data([origin[0], a_tip[0]], [origin[1], a_tip[1]])
        arm_shadow.set_3d_properties([floor_z, floor_z])
        pend_shadow.set_data([a_tip[0], p_tip[0]], [a_tip[1], p_tip[1]])
        pend_shadow.set_3d_properties([floor_z, floor_z])

        # Info text
        err_deg = np.degrees(((th2 - np.pi + np.pi) % (2 * np.pi)) - np.pi)
        t_s = idx * DT_CTRL
        info_text.set_text(
            f"t      = {t_s:5.2f} s\n"
            f"pend   = {err_deg:+6.1f} °\n"
            f"arm ω  = {np.degrees(w1):+6.0f} °/s\n"
            f"reward = {cum_rewards[idx]:6.0f} / {idx + 1:4d}"
        )

        # Torque bar
        tau = torques[idx] if idx < len(torques) else 0.0
        tau_bar[0].set_height(abs(tau))
        tau_bar[0].set_y(min(tau, 0.0))
        tau_bar[0].set_facecolor("#ff8800" if tau >= 0 else "#4488ff")

        return (arm_line, pend_line, arm_joint, pend_bob,
                ref_line, arm_shadow, pend_shadow, info_text)

    interval_ms = FRAME_SKIP * DT_CTRL * 1000   # real-time cadence
    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    return fig, ani


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Visualise Furuta balance controller")
    p.add_argument("--seed", type=int, default=1,
                   help="Episode seed (default 1 — ~30° start)")
    p.add_argument("--model", type=str, default="models/balance_best/best_model.zip",
                   help="Path to a saved SB3 model zip "
                        "(default: models/balance_best/best_model.zip)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fig, ani = build_animation(seed=args.seed, model_path=args.model)
    plt.show()
