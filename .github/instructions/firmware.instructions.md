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
- Peripherals: `SERCOM0`, `TC0`, `TCC0` etc. — use exact CMSIS names
- Functions: `snake_case`, prefixed by module — e.g. `i2c_init()`, `encoder_read_angle()`
- Constants: `ALL_CAPS` with module prefix — e.g. `AS5600_ADDR`, `PWM_FREQ_HZ`

## Pin Assignment
| Pins | Function | Peripheral |
|------|----------|------------|
| PA00 / PA01 | I2C0_SDA / I2C0_SCL (arm encoder) | SERCOM0 |
| PA08 / PA09 / PA10 | PWM_A / PWM_B / PWM_C | TCC0 WO[0/1/2] |
| PA11 | DRV_EN (active high) | GPIO out |
| PA12 | DRV_FAULT (~nFT, active low) | GPIO in, EIC interrupt |
| PA13 | DRV_SLEEP (~nSP, active low) | GPIO out |
| PA14 | DRV_RESET (~nRT, active low) | GPIO out |
| PA15 / PA16 | BTN1 / BTN2 | GPIO in, EIC interrupt |
| PA17 / PA18 | LED1 / LED2 | GPIO out |
| PA19 | LCD_RST | GPIO out |
| PA20 | LCD_CS | GPIO out |
| PA21 | LCD_DC | GPIO out |
| PA22 / PA23 | LCD_MOSI / LCD_SCK | SERCOM3 SPI |
| PA24 / PA25 | USB_DM / USB_DP | USB |
| PA30 / PA31 | SWDIO / SWDCLK | SWD |
| PB00 | BTN3 | GPIO in, EIC interrupt |
| PB08 / PB09 | I2C1_SDA / I2C1_SCL (pendulum encoder) | SERCOM4 |

## Peripherals
- **I2C0** (AS5600 arm encoder): SERCOM0, PA00/PA01, address 0x36
- **I2C1** (AS5600 pendulum encoder): SERCOM4, PB08/PB09, address 0x36
- **SPI** (ST7735S LCD): SERCOM3, PA22/PA23, write-only (no MISO)
- **TCC0** (3-phase PWM for DRV8313): center-aligned, 3 complementary outputs on PA08–PA10
- **USB**: CDC class for debug output — no printf to UART
- **EIC**: PA12 (DRV_FAULT falling edge), PA15/PA16/PB00 (button press)
- **Clock**: Internal DFLL48M locked to USB SOF — no external crystal

## DRV8313 Control Sequence
1. Boot with DRV_EN low, DRV_SLEEP high (driver disabled, not sleeping)
2. Run encoder init and self-checks
3. Assert DRV_EN high to enable driver only when ready
4. Monitor DRV_FAULT via EIC interrupt — on fault: disable DRV_EN immediately, log fault, attempt reset via DRV_RESET pulse

## Neural Network
- Inference via CMSIS-NN — use `arm_fully_connected_f32` and related functions
- Weights stored in flash as `const float` arrays
- Control loop target: 500–1000Hz
- Inputs: θ_arm, θ_pendulum, ω_arm, ω_pendulum (all in radians / radians per second)
- Output: torque command (normalized -1.0 to 1.0)

## Build
- CMake — `cmake -B build && cmake --build build`
- Toolchain: `arm-none-eabi-gcc`
- Debug: Atmel-ICE + OpenOCD via SWD header (J6)
