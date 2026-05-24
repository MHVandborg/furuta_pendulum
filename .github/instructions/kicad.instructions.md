---
description: "Use when designing, reviewing, or modifying the PCB schematic or layout in KiCad. Covers component choices, net naming, and layout guidelines for the Furuta pendulum PCB."
applyTo: "hardware/pcb/**"
---

# KiCad PCB Guidelines

## MCU — ATSAMD51J20A (QFN64)
- Decouple every VDD pin with 100nF close to the pin + bulk 10µF per power domain
- SWD header: SWDIO, SWDCLK, GND, 3V3 (4-pin)
- USB-C: 27Ω series on D+ and D-, ESD protection IC (e.g. USBLC6-2SC6)
- Crystal not required — use internal DFLL48M oscillator

## Net Naming
- Power: `+3V3`, `+5V`, `GND`
- I2C bus 1 (arm encoder): `I2C0_SDA`, `I2C0_SCL`
- I2C bus 2 (pendulum encoder): `I2C1_SDA`, `I2C1_SCL`
- Motor phases: `PHASE_A`, `PHASE_B`, `PHASE_C`
- PWM to DRV8313: `PWM_A`, `PWM_B`, `PWM_C`
- DRV8313 enable/fault: `DRV_EN`, `DRV_FAULT`

## Layout Priorities
1. Decouple caps as close to MCU pins as possible
2. Keep motor phase traces short and wide (high current)
3. Separate analog (encoder) and digital/power domains
4. Keep USB D+/D- as a matched-length differential pair

## Connectors
- Motor phases: 3-pin JST-XH or screw terminal
- AS5600 #1 and #2: 4-pin JST-SH (I2C + 3V3 + GND)
- SWD: 4-pin 1.27mm pitch header
- USB: USB-C
- LCD: 7-pin header (VCC, GND, SCK, MOSI, CS, DC, RST)
