---
description: "Use when designing, reviewing, or modifying the PCB schematic or layout in KiCad. Covers component choices, net naming, and layout guidelines for the Furuta pendulum PCB."
applyTo: "hardware/pcb/**"
---

# KiCad PCB Guidelines

## MCU — ATSAMD51J20A (TQFP-64)
- Decouple every VDD pin with 100nF X7R 0402 close to pin + bulk 10µF X5R 0805 per power domain
- SWD header: SWDIO, SWDCLK, GND, +3V3 (4-pin, 1.27mm pitch) — compatible with Atmel-ICE
- USB-C: 27Ω series on D+ and D-, ESD protection IC (USBLC6-2SC6)
- No external crystal — use internal DFLL48M locked to USB SOF

## Pin Assignment
| Pins | Function | Peripheral |
|------|----------|------------|
| PA00 / PA01 | I2C0_SDA / I2C0_SCL (arm encoder) | SERCOM0 |
| PA08 / PA09 / PA10 | PWM_A / PWM_B / PWM_C | TCC0 |
| PA11 | DRV_EN | GPIO out |
| PA12 | DRV_FAULT | GPIO in, interrupt |
| PA13 | DRV_SLEEP | GPIO out |
| PA14 | DRV_RESET | GPIO out |
| PA15 / PA16 | BTN1 / BTN2 | GPIO in, EIC |
| PA17 / PA18 | LED1 / LED2 | GPIO out |
| PA19 | LCD_RST | GPIO out |
| PA20 | LCD_CS | GPIO out |
| PA21 | LCD_DC | GPIO out |
| PA22 / PA23 | LCD_MOSI / LCD_SCK | SERCOM3 SPI |
| PA24 / PA25 | USB_DM / USB_DP | USB |
| PA30 / PA31 | SWDIO / SWDCLK | SWD |
| PB08 / PB09 | I2C1_SDA / I2C1_SCL (pendulum encoder) | SERCOM4 |

## Net Naming
- Net names are generally left as KiCad assigns them — do not rename nets just to match a convention
- One explicit exception: the motor-side ground plane is named **`PGND`** (not auto-generated)
- The logic-side ground is `GND` (KiCad default)
- Power rails use whatever name KiCad assigns from the placed power symbol (e.g. `+3.3V`, `+5V`, `+12V`)

## Power Architecture
- **Logic**: USB-C VBUS (5V) → AMS1117-3.3 LDO → 3.3V
- **Motor**: J1 VM connector → SMBJ15A TVS → pi-filter → +VM → DRV8313
- TVS diodes: SMBJ15A on +VM (shunt to PGND), SMBJ5.0A on +5V (shunt to GND)
- PGND and GND joined via ferrite bead (FB4, BLM21PG121SN1L, 0805) at star point near LDO
- FB3 (BLM18PG121SN1D, 0603) on MCU VSW pin for internal switching regulator filter

## Capacitor Types
- 100nF bypass: X7R 0402 MLCC
- 10µF bulk: X5R 0805 MLCC
- Do NOT use Y5V or Z5U — poor stability over temperature and voltage

## Layout Priorities
1. Decouple caps as close to MCU VDD pins as possible
2. Keep motor phase traces short and wide (high current, 1A+ per phase)
3. PGND and GND are separate pours — connect only at star point near LDO/power input
4. Keep USB D+/D- as matched-length differential pair
5. Place TVS diodes as close as possible to their respective input connectors
6. I2C pull-up resistors (4.7kΩ) placed on PCB near MCU, one pair per bus
7. DRV8313 and encoder signals grouped near their respective connectors

## Connectors
- Motor phases: 3-pin Molex Micro-Fit (43045-0300), locking — use shielded cable, shield to PGND at PCB end only
- AS5600 encoders: 4-pin headers for I2C + 3.3V + GND
- VM input (J1): 2-pin 2.54mm pin header for 12V PSU
- SWD (J4): 4-pin 1.27mm pitch header — compatible with Atmel-ICE
- USB: USB-C receptacle
- LCD (J2): 7-pin 2.54mm header (VCC, GND, SCK, MOSI, CS, DC, RST)
- Buttons: 10kΩ pull-up to 3.3V on PCB; SW3 connected to PB00 via direct wire (no net label)
- LEDs: 4 total — 2 power indicators (3.3V and 12V rails), 2 user GPIO (PA17/PA18)
