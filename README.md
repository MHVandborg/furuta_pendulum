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
- 64-pin TQFP package — enough I/O without excess
- Bare-metal friendly — clean register-level programming without heavy frameworks
- No external crystal — uses internal DFLL48M locked to USB SOF for accurate 48MHz

### Motor — GBM2804H-100T (Gimbal BLDC)
A brushless gimbal motor designed for low-speed, high-torque applications.
- Very low cogging torque — smooth, predictable motion critical for NN-based control
- Gimbal motors are specifically wound to minimize magnetic detents
- Low inertia — fast dynamic response needed for balancing
- High cogging in cheaper motors would introduce non-linearities that degrade NN performance
- Driven at 12V from external PSU (3S LiPo or bench supply)

### Motor Driver — SimpleFOC Mini (DRV8313)
An external module connected via header/jumper wires. Implements Field Oriented Control (FOC).
- FOC provides smooth, continuous torque control at any speed
- DRV8313 handles all low-level phase commutation internally
- Motor supply: 8–60V (12V recommended) — separate from logic supply
- Control signals: PWM_A/B/C (3-phase), EN (enable), nFT (fault), nSP (sleep), nRT (reset)
- Connected via Molex Micro-Fit 3-pin connector for motor phases

### Encoders — 2x AS5600 (Magnetic, 12-bit)
External modules connected via JST-SH 4-pin cables.
- One AS5600 measures the **arm angle** (motor shaft) — I2C bus 0 (SERCOM0)
- One AS5600 measures the **pendulum angle** — I2C bus 1 (SERCOM4)
- 12-bit resolution (4096 steps/revolution) — sufficient for this application
- Both share I2C address 0x36 — separated onto independent buses to avoid conflict
- 4.7kΩ I2C pull-ups on each bus placed on the PCB near the MCU

### Display — ST7735S 1.8" TFT LCD (128x160, SPI)
A small color display for showing system state, angles, and NN output.
- SPI interface via SERCOM3 — fast enough for real-time updates
- 3.3V compatible — direct connection to MCU
- 7-pin header: VCC, GND, SCK, MOSI, CS, DC, RST

### Power
- **Logic supply**: USB-C 5V → AMS1117-3.3 LDO → 3.3V (USB cable or bench supply)
- **Motor supply**: 12V PSU or 3S LiPo via J1 VM connector
- Pi-filter on +VM rail suppresses motor switching noise
- TVS diodes on both rails: SMBJ15A (+VM) and SMBJ5.0A (+5V) for ESD/spike protection
- USBLC6-2SC6 on USB D+/D- signal lines for ESD protection
- Separate ground planes: PGND (motor side) and GND (logic side), joined via ferrite bead at star point near LDO

### User Interface
- 4× LEDs: 2× power indicators (3.3V and 12V rails), 2× user-controllable (MCU GPIO)
- 3× tactile buttons (10kΩ pull-up, EIC interrupt capable)

### Debug
- SWD header (4-pin, 1.27mm) — compatible with Atmel-ICE for bootloader flashing and debugging

---

## Pin Assignment

| Pins | Function | Peripheral |
|------|----------|------------|
| PA00 / PA01 | I2C0_SDA / I2C0_SCL (arm encoder) | SERCOM0 |
| PA08 / PA09 / PA10 | PWM_A / PWM_B / PWM_C | TCC0 |
| PA11 | DRV_EN | GPIO out |
| PA12 | DRV_FAULT (~nFT) | GPIO in, interrupt |
| PA13 | DRV_SLEEP (~nSP) | GPIO out |
| PA14 | DRV_RESET (~nRT) | GPIO out |
| PA15 / PA16 | BTN1 / BTN2 | GPIO in, EIC |
| PA17 / PA18 | LED1 / LED2 | GPIO out |
| PA19 | LCD_RST | GPIO out |
| PA20 | LCD_CS | GPIO out |
| PA21 | LCD_DC | GPIO out |
| PA22 / PA23 | LCD_MOSI / LCD_SCK | SERCOM3 SPI |
| PA24 / PA25 | USB_DM / USB_DP | USB |
| PA30 / PA31 | SWDIO / SWDCLK | SWD |
| PB00 | BTN3 | GPIO in, EIC |
| PB08 / PB09 | I2C1_SDA / I2C1_SCL (pendulum encoder) | SERCOM4 |

---

## Control Architecture

```
[AS5600 arm]  [AS5600 pendulum]
      |               |
   I2C0            I2C1
      |               |
      +-------+-------+
              |
        [ATSAMD51J20A]
        Neural Network Inference (CMSIS-NN)
        4 inputs: θ_arm, θ_pendulum, ω_arm, ω_pendulum
        Output: torque_command
              |
           TCC0 PWM
              |
     [SimpleFOC Mini / DRV8313]
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
- Atmel-ICE + OpenOCD for SWD programming and debugging
- CMSIS-NN for optimized neural network inference on M4F

---

## Open To-Dos

- [ ] **FB3 ferrite bead**: Find a part matching the ATSAMD51 VSW pin spec (check ATSAMD51 datasheet section on VREG/VSW for recommended part or impedance)
- [ ] **Shielded cables**: Check stock at work for suitable shielded cables for encoder wiring
- [ ] **D2 TVS diode (5V rail)**: Find a smaller replacement for SMBJ5.0A — look for a 5V TVS in SOD-323 or smaller from work stock or JLCPCB basic library; update schematic with part number and footprint

---

## Contact

If you use or build on this project, I'd love to hear about it! Feel free to open an issue or reach out directly.
