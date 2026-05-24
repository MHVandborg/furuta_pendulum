# Furuta Pendulum — Project Guidelines

## Project Overview
This is a Furuta (rotary inverted) pendulum project. The goal is to swing up and balance an inverted pendulum using a neural network controller running on a custom PCB.

## Hardware
- **MCU**: ATSAMD51J20A (ARM Cortex-M4F, 120MHz, FPU, 1MB flash, 192KB RAM)
- **Motor**: GBM2804H-100T (gimbal BLDC)
- **Motor Driver**: SimpleFOC Mini (DRV8313) — FOC via SimpleFOC library
- **Encoders**: 2x AS5600 (12-bit magnetic, I2C) — one for arm, one for pendulum
- **Display**: ST7735S 1.8" TFT (128x160, SPI)

## Architecture
```
[AS5600 arm] [AS5600 pendulum]
       |              |
       +------+-------+
              |
        [Neural Network]  (θ_arm, θ_pendulum, ω_arm, ω_pendulum) → torque_command
              |
       [SimpleFOC / DRV8313]
              |
        [GBM2804H-100T]
```

## Firmware (firmware/)
- Bare-metal C — no HAL, no ASF, no Arduino frameworks
- ARM GCC + CMake + VS Code
- CMSIS headers for register access
- CMSIS-NN for neural network inference
- NN runs at 500–1000Hz control loop
- Two AS5600s are on separate I2C buses (same I2C address 0x36)

## PCB (hardware/pcb/)
- Designed in KiCad
- ATSAMD51J20A in QFN64
- USB-C with 27Ω series resistors on D+/D- and ESD protection
- Separate I2C buses for each AS5600

## Mechanical (hardware/mechanical/)
- Designed in Fusion 360

## Neural Network (neural_network/)
- Training scripts and exported model weights
- Weights are exported for CMSIS-NN inference on M4F
