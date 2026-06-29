# Furuta Pendulum

A Furuta (rotary inverted) pendulum system controlled by a neural network running on a custom PCB.

The system consists of a horizontal rotating arm driven by a BLDC motor, with a freely swinging pendulum attached at the end. The neural network controller learns to swing up and balance the pendulum using reinforcement learning trained in a physics simulation, then deployed directly onto the MCU.

## End Goal — Double Furuta Pendulum

The ultimate goal is a **double Furuta pendulum**: motorised arm → first free pendulum → second free pendulum, all balanced simultaneously in the inverted position by a single neural network. The current hardware (PCB rev1) targets the single pendulum first as a validated stepping stone. Double Furuta requires a PCB rev2 with a third encoder.

---

## Control Architecture

### Single Furuta (PCB rev1 — current)

```
[AS5600 arm]  [AS5600 pendulum 1]
      |               |
      +-------+-------+
              |
        [Neural Network]  MLP 2×64, trained with PPO
        inputs:  θ_arm, θ_p1, ω_arm, ω_p1
        output:  torque_command (−1…+1)
              |
        [SimpleFOC / DRV8313]
              |
        [GBM2804H-100T BLDC]
```

### Double Furuta (PCB rev2 — future)

```
[AS5600 arm]  [AS5600 pendulum 1]  [AS5600 pendulum 2]
      |               |                    |
      +-----------+---+--------------------+
                  |
            [Neural Network]  MLP 2×64 (extended inputs), trained with SAC
            inputs:  θ_arm, θ_p1, θ_p2, ω_arm, ω_p1, ω_p2
            output:  torque_command (−1…+1)
                  |
            [SimpleFOC / DRV8313]
                  |
            [GBM2804H-100T BLDC]
```

A single NN handles both swing-up and balancing via curriculum learning — it trains near vertical first, then faces progressively larger starting angles.

---

## Repository Structure

```
furuta_pendulum/
├── firmware/          # Bare-metal C (ARM GCC, CMake) → firmware/README.md
├── hardware/
│   ├── pcb/           # KiCad schematic + layout → hardware/pcb/COMPONENTS.md
│   └── mechanical/    # Fusion 360 3D design files
├── neural_network/    # RL training + Gymnasium env + exported weights → neural_network/README.md
├── bootloader/        # UF2 bootloader binary for ATSAMD51J20A
└── docs/              # Datasheets and reference documents
```

---

## Toolchain

| Area | Tools |
|---|---|
| Firmware | ARM GCC, CMake, VS Code, Atmel-ICE + Tag-Connect TC2030 (SWD), UF2 bootloader |
| ML / Training | Python, TensorFlow / Keras, Gymnasium, PPO → SAC |
| Inference on MCU | TensorFlow Lite Micro (TFLM) |
| Mechanical | Fusion 360, PLA (FDM) |
| PCB | KiCad |

---

## PCB Revision History

| Rev | Tag | Date | Notes |
|-----|-----|------|-------|
| 1 | [`hw-rev1`](../../releases/tag/hw-rev1) | 2026-06-07 | First production run — JLCPCB. Single Furuta (2 encoders) |
| 2 | *(planned)* | TBD | Add AS5600 #3 + third I2C bus for double Furuta |

---

## Further Reading

| Document | Contents |
|---|---|
| [ROADMAP.md](ROADMAP.md) | Full phased development plan with checklists |
| [hardware/pcb/COMPONENTS.md](hardware/pcb/COMPONENTS.md) | Component choices, rationale, and PCB design decisions |
| [firmware/README.md](firmware/README.md) | Firmware architecture, build instructions, peripheral map |
| [neural_network/README.md](neural_network/README.md) | RL training approach, simulation, reward design, export pipeline |

---

## Contact

If you use or build on this project, I'd love to hear about it! Feel free to open an issue or reach out directly.

