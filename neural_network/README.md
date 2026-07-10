# Neural Network — Training & Deployment

This directory contains the reinforcement learning training pipeline for the Furuta pendulum
controller. Two separate networks are trained in simulation and deployed to the MCU:

| Network | Role | Inputs |
|---------|------|--------|
| **Balance** | Keeps the pendulum upright once near vertical | 4 floats |
| **Swing-up** | Pumps energy into the pendulum from the resting position | 5 floats |

The firmware switches from swing-up to balance when the pendulum is both geometrically
close to vertical **and** has sufficient energy to stay there (see Handoff Condition below).

---

## Quick Start (new machine)

```powershell
# 1. Clone the repo and enter this directory
cd neural_network

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
.\venv\Scripts\Activate.ps1          # Windows PowerShell
# source venv/bin/activate           # Linux / macOS

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Train the balance controller (~30–60 min on CPU, faster on GPU)
python train.py --mode balance

# 6. Train the swing-up controller
python train.py --mode swingup
```

> **Note:** The `venv/` folder is git-ignored. Every machine needs its own copy.
> Trained model files (`.zip`, `.tflite`) are also git-ignored — transfer them
> manually between machines (USB, network share, etc.).

---

## Requirements

- Python 3.11+
- No GPU required — CPU training works fine, just takes longer
- ~4 GB RAM minimum during training (PyTorch + 8 parallel envs)

Tested on Python 3.11 on Windows. Dependencies are pinned in `requirements.txt`.

---

## Directory Structure

```
neural_network/
├── envs/
│   ├── __init__.py
│   ├── physics.py           # Lagrangian equations of motion + RK4 integrator
│   └── furuta_env.py        # Gymnasium environment (reward, reset, observations)
├── train.py                 # PPO training script (balance + swing-up)
├── models/                  # Saved model checkpoints — git-ignored
│   ├── balance_best/        # Best balance checkpoint (saved during training)
│   ├── balance_final.zip    # Final balance model
│   ├── swingup_best/        # Best swing-up checkpoint
│   └── swingup_final.zip    # Final swing-up model
├── exported/                # TFLite files + C arrays for firmware — git-ignored
├── logs/                    # TensorBoard training logs — git-ignored
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Training

### Commands

```powershell
# Balance controller — starts near vertical, curriculum widens starting angle
python train.py --mode balance

# Swing-up controller — always starts near hanging position
python train.py --mode swingup

# All options
python train.py --help
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `balance` | Which controller to train |
| `--timesteps` | `2,000,000` | Total environment steps |
| `--n-envs` | `8` | Parallel training environments (scale to your CPU core count) |
| `--no-save` | off | Skip saving — use for quick smoke tests |

### How long does it take?

On a modern CPU (8 cores), expect ~1,000–1,500 steps/second with 8 parallel envs.

| Steps | Wall time | What you see |
|-------|-----------|--------------|
| 60,000 | ~1 min | Random policy, no learning visible yet |
| 500,000 | ~8 min | Policy starts committing, reward improves |
| 2,000,000 | ~30 min | Full training — balance should hold reliably |

On a GPU the training is significantly faster. No code changes needed —
PyTorch detects CUDA automatically.

### Monitoring with TensorBoard

```powershell
# In a separate terminal, from the neural_network directory
.\venv\Scripts\tensorboard --logdir logs
# Then open http://localhost:6006 in a browser
```

Key metrics to watch:
- `rollout/ep_rew_mean` — mean episode reward, should climb toward 0
- `train/explained_variance` — how well the critic understands the environment, should climb toward 1.0
- `train/entropy_loss` — policy randomness, should decrease as policy commits to learned behaviour

### Training order

Train **balance first** — it is easier (starts near vertical, clean reward signal) and
the curriculum is well-defined. Once balance works, train swing-up.

---

## Architecture

### Two controllers, same observation format

Both networks receive the same base observation:

```
[0]  θ₁ / π            arm angle, normalised to [-1, 1]
[1]  (θ₂ − π) / π      pendulum error: 0 = upright, ±1 = hanging
[2]  ω₁ / ω₁_max       arm angular velocity, clipped to [-1, 1]
[3]  ω₂ / ω₂_max       pendulum angular velocity, clipped to [-1, 1]
```

The swing-up network gets one extra input:

```
[4]  E_err / E_max      energy error: 0 = exactly enough to reach vertical
                        negative = needs more energy, positive = over-energised
```

Energy is a more informative input for swing-up than angle alone — it tells the
network exactly how much more pumping is needed, regardless of the current angle.

### Network

Both networks: MLP, 2 hidden layers × 64 neurons, tanh activation.

| | Balance | Swing-up |
|-|---------|---------|
| Inputs | 4 | 5 |
| Hidden layers | 2 × 64, tanh | 2 × 64, tanh |
| Output | 1 (normalised torque) | 1 (normalised torque) |

Small networks are intentional — they must run at 500 Hz on the SAMD51 M4F MCU.

### Reward functions

**Balance:**
```
r = -(θ_err² + 0.01·ω_arm²)     range: (-π², 0]
```
Quadratic penalty on deviation from vertical and arm spinning. Maximum 0 (perfect balance).

**Swing-up:**
```
r = cos(θ₂ − π)                  range: [-1, +1]
```
+1 when upright, −1 when hanging. No arm velocity penalty — the agent must spin freely
to pump energy into the pendulum.

### Curriculum learning (balance only)

The `difficulty` parameter (0–1) controls how far from vertical episodes start:

| `difficulty` | Starting range |
|---|---|
| 0.0 | ±5° (easy — almost already balanced) |
| 0.15 | ±30° |
| 0.50 | ±90° (horizontal) |
| 1.0 | ±180° (fully random) |

The `CurriculumCallback` in `train.py` automatically advances difficulty as
`ep_rew_mean` crosses the thresholds `-2.0 → -0.5 → -0.1`.

### Domain randomisation

On every `reset()`, all physical parameters are re-sampled within ±20% of their
nominal values. This forces the policy to generalise across manufacturing tolerances
and modelling errors, reducing the sim-to-real gap.

---

## Physics Simulation

Implemented directly in `envs/physics.py` — no external physics engine.

**Lagrangian equations of motion:**

```
Arm:       M₁₁(θ₂)·α₁ = τ − b₁·ω₁ − 2·m₂·(L₁ + Lp·sinθ₂)·Lp·cosθ₂·ω₁·ω₂
Pendulum:  Jp·α₂      = −b₂·ω₂ + m₂·(L₁ + Lp·sinθ₂)·Lp·cosθ₂·ω₁² − m₂·g·Lp·sinθ₂
                                   └── centrifugal (swing-up) ───────┘   └── gravity ─┘
```

Integrated with **RK4** at 1 ms (1 kHz). Two integration steps per control step
→ 500 Hz effective control rate, matching the firmware loop.

**Physical parameters** (update from Fusion 360 CAD after mechanical design):

| Symbol | Nominal | Description |
|--------|---------|-------------|
| L₁ | 0.15 m | Arm length: motor shaft → pendulum pivot |
| m₁ | 0.05 kg | Arm mass |
| L₂ | 0.20 m | Pendulum length: pivot → tip |
| m₂ | 0.03 kg | Pendulum mass |
| b₁ | 1.0×10⁻³ N·m·s/rad | Arm viscous friction |
| b₂ | 5.0×10⁻⁴ N·m·s/rad | Pendulum viscous friction |

> These are placeholders. Update `FurutaParams.nominal()` in `envs/physics.py`
> once real measurements are available from the CAD model or system identification.

---

## Handoff Condition

The firmware switches from swing-up to balance when **both** are true:

```
|θ₂ − π| < 20°          (geometrically near vertical)
E_pendulum > E_upright − 0.05 J   (has enough energy to stay there)
```

Using both conditions prevents handing off when the pendulum is geometrically
"near" vertical but already falling back with insufficient energy.

---

## Export to MCU

After training, convert the models to a format the MCU can run:

```powershell
# Step 1: Export both models to TFLite
python export_model.py
# Output: exported/balance_policy.tflite
#         exported/swingup_policy.tflite

# Step 2: Convert to C byte arrays for embedding in firmware
# (run from repo root, requires xxd — available in Git Bash on Windows)
xxd -i neural_network/exported/balance_policy.tflite > firmware/control/balance_model_data.cc
xxd -i neural_network/exported/swingup_policy.tflite > firmware/control/swingup_model_data.cc
```

The `.cc` files are compiled directly into the firmware image. Both models live
in flash and are interpreted by the TFLM library — no file system or OS needed.

---

## System Identification

For a tighter sim-to-real fit, measure real motor inertia and friction from hardware:

1. Flash a test firmware that commands known torque steps
2. Record angular acceleration α from the AS5600 encoder
3. Fit `τ = J·α + b·ω` (linear regression) → extract real J and b
4. Update `FurutaParams.nominal()` with measured values
5. Retrain with tighter domain randomisation (±10% instead of ±20%)

See the ROADMAP for the full procedure.

---

## Double Furuta (Future — rev2)

- Extend observation to 6 inputs: `[θ_arm, θ_p1, θ_p2, ω_arm, ω_p1, ω_p2]`
- Extend physics to the double-pendulum Lagrangian
- Switch algorithm from PPO to **SAC** (better sample efficiency for 6D control)
- Consider LSTM policy for the coupled dynamics (see ROADMAP rev2 notes)
