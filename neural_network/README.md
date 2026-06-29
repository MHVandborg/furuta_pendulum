# Neural Network — Training & Deployment

This directory contains the reinforcement learning training pipeline for the Furuta pendulum
controller. The trained policy is exported as a `.tflite` model and deployed to the MCU via
TensorFlow Lite Micro (TFLM).

---

## Approach

The controller is trained entirely in **simulation** using reinforcement learning, then
transferred to real hardware (sim-to-real). A single neural network handles both swing-up
and balancing — there is no mode switch between the two.

| Property | Value |
|---|---|
| Algorithm | PPO (single pendulum) → SAC (double pendulum) |
| Framework | TensorFlow / Keras + Gymnasium |
| Observation space | `[θ_arm, θ_p, ω_arm, ω_p]` — 4 floats, normalised |
| Action space | `[torque]` — 1 float, continuous, in [−1, +1] |
| Network | MLP, 2 hidden layers × 64 neurons, tanh activation |
| Reward | Dense: `-(θ_p² + 0.01·ω_arm²)` per timestep |
| Sim-to-real strategy | Domain randomisation + curriculum learning |

---

## Directory Structure

```
neural_network/
├── envs/
│   └── furuta_env.py        # Gymnasium environment (equations of motion + reward)
├── train.py                 # PPO training script
├── evaluate.py              # Evaluate a saved policy in simulation
├── export_model.py          # Convert trained Keras model → .tflite → C array
├── models/                  # Saved .h5 checkpoints (git-ignored if large)
└── exported/                # furuta_policy.tflite + nn_model_data.cc
```

---

## Physics Simulation

The environment implements the Furuta pendulum **Lagrangian equations of motion** directly
in Python — no external physics engine is required.

**State vector:** `[θ_arm, θ_pendulum, ω_arm, ω_pendulum]`

**Physical parameters (populated from Fusion 360 CAD after mechanical design):**

| Parameter | Symbol | Unit | Description |
|---|---|---|---|
| Arm length | L₁ | m | Distance from motor shaft to pendulum pivot |
| Arm mass | m₁ | kg | Mass of horizontal arm |
| Pendulum length | L₂ | m | Distance from pivot to pendulum tip |
| Pendulum mass | m₂ | kg | Mass of pendulum rod |
| Motor inertia | J | kg·m² | Rotor + arm moment of inertia |
| Friction | b | N·m·s | Viscous friction at motor shaft |

**Domain randomisation:** On every `reset()`, each parameter is sampled uniformly within
±20% of its nominal value. This forces the NN to learn a robust policy rather than
overfitting to perfect simulation physics, which reduces the sim-to-real gap.

---

## Reward Function

```
r = -(θ_p² + 0.01·ω_arm²)   per timestep
```

- Heavily penalises deviation of the pendulum from vertical (θ_p = 0 is upright)
- Small penalty on arm angular velocity to discourage wild spinning
- Dense reward at every timestep — the agent gets a continuous learning signal

---

## Curriculum Learning

Training starts easy and gradually becomes harder:

1. **Early training:** `reset()` places the pendulum within ±5° of vertical
2. **As policy improves:** the starting angle range widens (±30°, ±90°, ±180°)
3. **Final training:** `reset()` from any angle — full swing-up required

This prevents the agent from getting stuck when the reward signal is too sparse at the start.

---

## Training

```bash
# Install dependencies
pip install tensorflow gymnasium numpy scipy stable-baselines3

# Train (PPO, single Furuta)
python neural_network/train.py

# Evaluate trained policy in sim
python neural_network/evaluate.py --model models/furuta_policy.h5
```

---

## Export to MCU

After training, convert the Keras model to a format the MCU can run:

```bash
# Step 1: Convert Keras → TFLite
python neural_network/export_model.py
# Output: neural_network/exported/furuta_policy.tflite

# Step 2: Convert TFLite → C byte array for embedding in firmware
xxd -i neural_network/exported/furuta_policy.tflite \
    > firmware/control/nn_model_data.cc
```

The resulting `nn_model_data.cc` is compiled directly into the firmware image. The model
lives in flash and is interpreted at runtime by the TFLM library — no file system or OS needed.

---

## System Identification (Optional)

For a tighter sim-to-real fit, real motor inertia and friction can be identified from
hardware measurements:

1. Flash a test firmware that commands a series of known torque steps
2. Record angular acceleration α from the AS5600 encoder at each step
3. Fit `τ = J·α + b·ω` (linear regression) to extract J and b
4. Update nominal sim parameters, retrain with tighter domain randomisation bounds (±10%)

This is a one-day task once the firmware can command torques and read encoders reliably.

---

## Double Furuta (Future)

When extending to the double pendulum configuration:

- Extend observation space to 6 inputs: `[θ_arm, θ_p1, θ_p2, ω_arm, ω_p1, ω_p2]`
- Extend equations of motion to the double-pendulum Lagrangian
- Switch training algorithm from PPO to **SAC** (Soft Actor-Critic) — better sample
  efficiency for higher-dimensional continuous control problems
- Retrain from scratch using the same curriculum learning approach
