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
- [ ] Pendulum encoder zero calibration on boot: let pendulum hang still, record raw
  count as `zero_offset`, compute `θ₂ = (raw − zero_offset) / 4096 × 2π` from
  then on. The network convention is θ₂=0 at hanging, θ₂=π at upright — if this
  offset is wrong the controller will drive the motor the wrong way.
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
- [ ] PCB rev2 design: switch all 3 encoders to AS5048A (SPI) — see Rev2 notes below
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
- [ ] Replace I2C encoder drivers with SPI AS5048A driver (shared bus, 3 CS pins)
- [ ] Extend state vector from 4 → 6 inputs
- [ ] Deploy new model, test on real hardware

---

## Rev2 PCB Improvement Notes

Ideas identified during rev1 development. Implement together with the Phase 6 hardware redesign.

### Encoder upgrade: AS5600 → AS5048A (SPI)

- **Why:** AS5048A is 14-bit (0.022°) vs AS5600 12-bit (0.088°). 4× better angle resolution
  gives 4× better velocity estimates near the balance point, which tightens balance quality
  for both the single and double pendulum.
- **Double pendulum motivation:** 6-state control compounds velocity noise across two
  pendulums. Noisy ω_p1 and ω_p2 simultaneously makes the harder balancing problem
  significantly worse. The upgrade is clearly worthwhile at rev2.
- **Bus simplification:** All three AS5048A encoders share one SPI bus — just different CS
  pins (e.g. PA15, PA16, PA17). Eliminates the two-I2C-bus workaround needed because both
  AS5600s share the hardwired address 0x36.
- **Speed:** SPI at 10 MHz → ~3 μs/read vs ~90 μs/read over I2C. Parallel CS transactions
  possible; all three reads complete in under 10 μs total.

### I2C Fast Mode+ on rev1 (no hardware change needed)

- AS5600 supports up to 1 MHz (Fast Mode+). Rev1 firmware uses 400 kHz.
- Changing the SERCOM baud register halves read latency to ~36 μs/encoder at no cost.
- Worth doing in firmware before committing to the encoder swap.

### Control loop frequency

- 500 Hz is already ~15× the pendulum's natural frequency (~1.6 Hz) — well beyond what
  the single pendulum dynamics require.
- Higher frequency (1 kHz) improves velocity estimate quality (finer finite-difference
  resolution) but provides no additional control-theoretic benefit for the single pendulum.
- For the double pendulum, the second pendulum's higher-frequency modes may justify 1 kHz.
  Re-evaluate after rev1 balance is working.

### Velocity estimation: EMA filter in firmware

- Velocities are computed by finite-differencing successive encoder readings, which produces
  a noisy staircase signal near the balance point (smallest detectable ω ≈ 44°/s with AS5600).
- An Exponential Moving Average (EMA) filter in firmware handles this at zero cost:
    `ω_filtered = α·ω_raw + (1−α)·ω_filtered_prev`
- This is one multiply + one add per loop iteration. No network change needed.
- α ≈ 0.3–0.5 is a good starting range; tune on real hardware by watching the USB CDC log.
- Do this before considering any network architecture changes for noise.

### Motor dynamics modelling

- Currently the sim assumes torque is delivered instantaneously. Real motors have inductance
  lag and back-EMF that cause the delivered torque to lag the commanded torque.
- **System identification procedure (after firmware can command torques):**
  1. Command a series of known torque steps via firmware
  2. Record angular acceleration α from the encoder at each step
  3. Fit `τ = J·α + b·ω` (linear regression) to extract real J and b values
  4. Add a first-order lag model: `τ_actual(s) = τ_cmd(s) / (1 + T_motor·s)`
  5. Update `FurutaParams` nominal values and retrain with tighter domain randomisation (±10%)
- This is a one-day task once hardware is working and is the single biggest lever for
  closing the sim-to-real gap.

### Swing-up / balance handoff condition

- Current firmware plan uses a pure angle threshold (|θ₂ − π| < 20°) to switch networks.
- Better condition uses both angle AND energy:
    switch when |θ₂ − π| < 20°  AND  E_pendulum > E_upright − 0.05 J
- The energy condition prevents handing off when the pendulum is geometrically "near"
  vertical but already falling back down with insufficient energy to stay there.
- E_upright = mp·g·Lp — computed from FurutaParams; update after mechanical design.

### Neural network architecture improvements (post rev1)

- **Frame stacking for balance (fallback):** If balance is noisy after EMA tuning, feed
  the last 2–3 observations as input [s_t, s_{t-1}, s_{t-2}] = 12 floats. The network
  learns its own implicit filter. A circular buffer in firmware RAM handles this trivially.
  Try EMA first — this is only needed if the firmware filter is insufficient.
- **Previous action as input:** Feeding the last torque command helps the network account
  for motor lag not captured in the physics model. Low cost: 1 extra input to both networks.
- **Recurrent networks (LSTM/GRU) for rev2 / double pendulum:** LSTM maintains hidden state
  between loop iterations, learning arbitrary temporal patterns. Harder to train than MLP
  but genuinely useful for the 6-state double pendulum where coupled dynamics are harder
  to observe cleanly. CMSIS-NN supports it. Revisit when extending to double pendulum.

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
| 3× AS5048A encoder (SPI) | rev2 | Planned |
