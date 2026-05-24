---
description: "Use when writing, reviewing, or debugging firmware code for the ATSAMD51J20A. Covers bare-metal C conventions, register access, peripheral setup, and CMSIS-NN inference."
applyTo: "firmware/**"
---

# Firmware Guidelines

## Style
- Pure C (C11) — no C++ unless absolutely necessary
- No HAL, no ASF, no Arduino — direct register access via CMSIS headers only
- All peripheral configuration done via register writes, never via library calls
- Prefer `volatile` for all hardware register accesses

## Naming
- Peripherals: `SERCOM0`, `TC0`, `TCC1` etc. — use exact CMSIS names
- Functions: `snake_case`, prefixed by module — e.g. `i2c_init()`, `encoder_read_angle()`
- Constants: `ALL_CAPS` with module prefix — e.g. `AS5600_ADDR`, `PWM_FREQ_HZ`

## Peripherals
- I2C (AS5600 arm): SERCOM on dedicated bus
- I2C (AS5600 pendulum): separate SERCOM — both encoders share address 0x36
- SPI (ST7735S LCD): dedicated SERCOM
- TCC (3-phase PWM for DRV8313): center-aligned, complementary outputs
- USB: CDC class for debug output — no printf to UART

## Neural Network
- Inference via CMSIS-NN — use `arm_fully_connected_f32` and related functions
- Weights stored in flash as `const float` arrays
- Control loop target: 500–1000Hz
- Inputs: θ_arm, θ_pendulum, ω_arm, ω_pendulum (all in radians / radians per second)
- Output: torque command (normalized -1.0 to 1.0)

## Build
- CMake — `cmake -B build && cmake --build build`
- Toolchain: `arm-none-eabi-gcc`
- Debug: OpenOCD + CMSIS-DAP
