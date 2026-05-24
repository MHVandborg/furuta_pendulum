# Furuta Pendulum

A Furuta pendulum (rotary inverted pendulum) controlled by a neural network running on a custom PCB.

The system consists of a horizontal rotating arm driven by a motor, with a freely swinging pendulum attached at the end. The goal is to swing up and balance the pendulum in the inverted (upright) position using a trained neural network controller.

---

## Hardware

### MCU — ATSAMD51J20A
ARM Cortex-M4F running at 120MHz with a hardware FPU (Floating Point Unit).
- Hardware FPU is essential for efficient neural network inference and FOC math
- 1MB flash provides ample space for NN weights, FOC code, USB and LCD drivers
- 192KB RAM — sufficient for real-time inference at 500–1000Hz
- Integrated USB Full Speed — clean PCB design, no external USB-UART chip needed
- 64-pin QFN package — enough I/O without excess
- Bare-metal friendly — clean register-level programming without heavy frameworks

### Motor — GBM2804H-100T (Gimbal BLDC)
A brushless gimbal motor designed for low-speed, high-torque applications.
- Very low cogging torque — smooth, predictable motion critical for NN-based control
- Gimbal motors are specifically wound to minimize magnetic detents
- Low inertia — fast dynamic response needed for balancing
- High cogging in cheaper motors would introduce non-linearities that degrade NN performance

### Motor Driver — SimpleFOC Mini (DRV8313)
A compact 3-phase gate driver board implementing Field Oriented Control (FOC).
- FOC provides smooth, continuous torque control at any speed
- DRV8313 handles all low-level phase commutation internally
- SimpleFOC library natively supported — accepts high-level torque commands
- 2.5A continuous — well matched to the GBM2804

### Encoders — 2x AS5600 (Magnetic, 12-bit)
Magnetic absolute rotary encoders for angle measurement.
- One AS5600 measures the **arm angle** (motor shaft)
- One AS5600 measures the **pendulum angle**
- 12-bit resolution (4096 steps/revolution) — sufficient for this application
- I2C interface — simple wiring, supported natively on ATSAMD51
- Note: both have the same I2C address (0x36), so they require separate I2C buses

### Display — ST7735S 1.8" TFT LCD (128x160, SPI)
A small color display for showing system state, angles, and NN output.
- SPI interface — fast enough for real-time updates
- 3.3V compatible — direct connection to MCU

---

## Control Architecture

```
[AS5600 arm]  [AS5600 pendulum]
      |               |
      +-------+-------+
              |
        [Neural Network]
        4 inputs: θ_arm, θ_pendulum, ω_arm, ω_pendulum
        Output: torque_command
              |
        [SimpleFOC / DRV8313]
              |
        [GBM2804H-100T]
```

The balancing controller consists of two phases:
- **Swing-up**: energy-based controller (or separate NN) to bring the pendulum near the upright position
- **Balance**: NN controller to maintain the inverted position

---

## Repository Structure

```
furuta_pendulum/
├── firmware/          # MCU source code (VS Code, ARM GCC, CMake)
├── hardware/
│   ├── pcb/           # KiCad schematic and PCB layout files
│   └── mechanical/    # Fusion 360 3D design files
├── neural_network/    # NN training scripts and exported model weights
└── docs/              # Datasheets, notes, references
```

---

## Toolchain
- ARM GCC
- CMake
- VS Code
- OpenOCD + CMSIS-DAP debugger
- CMSIS-NN for optimized neural network inference on M4F

---

## Contact

If you use or build on this project, I'd love to hear about it! Feel free to open an issue or reach out directly.
