# Furuta Pendulum — Roadmap

This document tracks the development phases from an empty repo to a working double Furuta
pendulum balanced by a neural network.

---

## Phase 0 — Toolchain & Firmware Scaffold

Get a compilable project on the MCU and confirm the flash pipeline works.

- [ ] ARM GCC + CMake build system configured, VS Code tasks defined
- [ ] Folder structure: `firmware/hal/`, `firmware/drivers/`, `firmware/control/`, `firmware/main.c`
- [ ] Flash UF2 bootloader via Atmel-ICE + OpenOCD (one-time, SWD)
- [ ] Blink onboard LED (smoke-test that code runs on hardware)
- [ ] USB CDC debug output — `printf` over USB, no UART needed

---

## Phase 1 — Hardware Validation *(no NN yet)*

Validate every sensor and actuator in isolation before integrating the controller.

- [ ] I2C0 driver → read AS5600 arm encoder angle (SERCOM0, PA00/PA01)
- [ ] I2C1 driver → read AS5600 pendulum encoder angle (SERCOM4, PB08/PB09)
- [ ] TCC0 3-phase PWM + DRV8313 startup sequence (EN, SLEEP, RESET, FAULT)
- [ ] Command a known torque and verify motor moves
- [ ] Log encoder angles + motor response continuously over USB CDC
- [ ] Confirm both sensors read correct angles through a full revolution

---

## Phase 2 — Mechanical Design

Design and print the physical structure. Physical dimensions extracted from CAD are used
directly as simulation parameters in Phase 3.

- [ ] Fusion 360: horizontal rotating arm (PLA)
- [ ] Fusion 360: pendulum rod and pivot bearing mount (PLA or metal rod for consistent mass distribution)
- [ ] 3D print and assemble on motor shaft
- [ ] Measure / extract from CAD:
  - Arm length L₁ and mass m₁ (moment of inertia about pivot)
  - Pendulum length L₂ and mass m₂ (moment of inertia about pivot)
- [ ] These values are the nominal parameters for the physics simulation

---

## Phase 3 — Physics Simulation + RL Environment

Build the Gymnasium environment the RL agent will train in.

- [ ] Set up Python environment (TensorFlow, Gymnasium, numpy, scipy)
- [ ] Implement Furuta pendulum equations of motion (Lagrangian formulation)
  - States: θ_arm, θ_pendulum, ω_arm, ω_pendulum
  - Input: torque τ
  - Output: θ̈_arm, θ̈_pendulum → integrate with fixed timestep
- [ ] Wrap in a Gymnasium `Env` subclass:
  - `observation_space`: Box(4,) — [θ_arm, θ_p, ω_arm, ω_p] normalised
  - `action_space`: Box(1,) — torque in [−1, +1]
  - `reset()`: randomise starting state (curriculum-controlled)
  - `step()`: advance physics, compute reward, check termination
- [ ] Dense reward: `r = -(θ_p² + 0.01·ω_arm²)` per timestep
  - Heavily penalises deviation from vertical
  - Small penalty on arm velocity to discourage wild spinning
- [ ] Domain randomisation: sample L₁, L₂, m₁, m₂, friction within ±20% of nominal on each `reset()`
- [ ] Curriculum learning: start `reset()` near vertical (±5°), gradually widen to full range as training improves
- [ ] Smoke-test: verify a hand-coded near-optimal action produces positive reward signal

---

## Phase 4 — RL Training (single Furuta, PPO)

Train the neural network policy entirely in simulation.

- [ ] Train a PPO agent on the Gymnasium environment
  - Policy network: MLP, 2 hidden layers × 64 neurons, tanh activation
  - Inputs: [θ_arm, θ_p, ω_arm, ω_p] (4 values)
  - Output: torque command (1 continuous value)
- [ ] Validate policy in simulation — pendulum holds balance for > 30 seconds from arbitrary start
- [ ] *(Optional but recommended)* Motor system identification:
  - Command torque steps via firmware, record α from encoder
  - Fit `τ = J·α + b·ω` to get real J, b values
  - Update nominal sim parameters, retrain with tighter domain randomisation bounds
- [ ] Export trained model:
  1. Save Keras model: `model.save('furuta_policy.h5')`
  2. Convert to TFLite: `TFLiteConverter.from_keras_model(model)` → `furuta_policy.tflite`
  3. Convert to C array: `xxd -i furuta_policy.tflite > firmware/control/nn_model_data.cc`

---

## Phase 5 — TFLM Firmware Integration

Bring the trained policy onto the MCU and close the real hardware loop.

- [ ] Add TensorFlow Lite Micro source tree to CMake build
- [ ] `firmware/control/nn_inference.c`:
  - Load model from C byte array in flash
  - Allocate tensor arena (static, no malloc)
  - Expose `nn_inference(float state[4], float *torque_out)`
- [ ] Full 500Hz control loop in `main.c`:
  1. Read both AS5600 encoders
  2. Compute angular velocities (differentiate + low-pass filter)
  3. Normalise state vector
  4. Run NN inference → torque command
  5. Send torque command to DRV8313 via SimpleFOC
- [ ] Test on real hardware
- [ ] Identify and close sim-to-real gaps:
  - Log real state trajectories over USB CDC
  - Compare to sim predictions
  - Update sim parameters if needed, retrain if gap is large

---

## Phase 6 — Double Furuta *(requires PCB rev2)*

Extend the hardware, simulation, and controller to the double pendulum configuration.

### Hardware
- [ ] PCB rev2 design: add AS5600 #3, route to a spare SERCOM as third I2C bus
- [ ] Order and assemble PCB rev2
- [ ] Design pendulum 2 arm and pivot mount in Fusion 360
- [ ] Extract L₃, m₃ from CAD

### Simulation
- [ ] Extend equations of motion to double-pendulum Lagrangian (6 state variables)
  - Additional states: θ_p2, ω_p2
- [ ] Extend `observation_space` to Box(6,)
- [ ] Retune reward and curriculum for the harder problem

### Training
- [ ] Switch from PPO to SAC (better sample efficiency for higher-dimensional continuous control)
- [ ] Extend policy network inputs from 4 → 6
- [ ] Train, validate, export as before

### Firmware
- [ ] Add I2C2 driver for AS5600 #3
- [ ] Extend state vector from 4 → 6 inputs
- [ ] Deploy new model, test on real hardware

---

## Hardware at a Glance

| Item | PCB Rev | Status |
|---|---|---|
| 2× AS5600 encoder | rev1 | On PCB |
| GBM2804H-100T motor | rev1 | — |
| SimpleFOC Mini (DRV8313) | rev1 | — |
| ST7735S 1.8" LCD | rev1 | — |
| Atmel-ICE + Tag-Connect | — | — |
| UF2 bootloader binary | — | ✅ In repo |
| AS5600 encoder #3 | rev2 | Planned |
