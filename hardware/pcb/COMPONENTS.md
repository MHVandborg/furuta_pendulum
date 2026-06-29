# Hardware — Component Choices & PCB Design

This document explains the component selection rationale and PCB design decisions for the
Furuta pendulum controller. For the high-level system overview see the [root README](../../README.md).

---

## Component Choices

### MCU — ATSAMD51J20A

ARM Cortex-M4F, 120MHz, hardware FPU, 1MB flash, 192KB RAM.

| Requirement | Why this MCU |
|---|---|
| Neural network inference at 500–1000Hz | Hardware FPU handles float32 math natively — no soft-float penalty |
| FOC math (sin/cos, Clarke/Park transforms) | Same FPU advantage |
| Two separate I2C buses | Multiple SERCOMs configurable as I2C; rev1 uses SERCOM0 + SERCOM4 |
| USB debug output without extra chip | Integrated USB Full Speed (no UART-to-USB converter needed) |
| Enough flash for NN weights + code | 1MB — comfortably fits TFLM runtime + model weights + all drivers |
| Bare-metal friendly | Clean register map, excellent CMSIS headers, no mandatory framework |

The 64-pin QFN package gives enough I/O without going to a larger (harder to solder) package.

---

### Motor — GBM2804H-100T (Gimbal BLDC)

A brushless gimbal motor designed for camera stabilisation.

| Requirement | Why a gimbal motor |
|---|---|
| Smooth, low-cogging torque | Gimbal motors are wound with many poles specifically to minimise magnetic detents — cogging in a standard BLDC would introduce non-linearities that degrade NN performance |
| Low speed, high torque | Balancing a pendulum requires sustained torque at near-zero speed, not high RPM |
| Low inertia | Fast dynamic response is critical — the pendulum falls quickly |
| Predictable dynamics | The NN policy trained in simulation must transfer to real hardware; unpredictable motor behaviour widens the sim-to-real gap |

---

### Motor Driver — SimpleFOC Mini (DRV8313)

A compact 3-phase gate driver module.

| Requirement | Why FOC |
|---|---|
| Continuous torque at any speed | Field Oriented Control decouples torque from speed — you can command full torque at 0 RPM |
| High-level torque command interface | The NN outputs a normalised torque value (−1…+1); FOC translates this to phase currents |
| Compact form factor | SimpleFOC Mini fits on a small PCB, 2.5A continuous — matched to the GBM2804 |
| Complementary PWM | DRV8313 accepts three half-bridge inputs; TCC0 on the SAMD51 generates centre-aligned complementary PWM |

**DRV8313 control signals on PCB rev1:**

| Signal | Pin | Description |
|---|---|---|
| DRV_EN | PA11 | Active high — enables gate drive |
| DRV_FAULT (nFAULT) | PA12 | Active low, EIC interrupt — latches on overcurrent/overtemp |
| DRV_SLEEP (nSLEEP) | PA13 | Active low — puts driver into low-power sleep |
| DRV_RESET (nRESET) | PA14 | Active low pulse — clears fault latch |

**Startup sequence:** boot with DRV_EN low, DRV_SLEEP high → run encoder init → assert DRV_EN → monitor DRV_FAULT via EIC.

---

### Encoders — AS5600 (12-bit Magnetic, I2C)

Contactless magnetic rotary encoders. One on the motor shaft (arm), one on the pendulum pivot.

| Requirement | Why AS5600 |
|---|---|
| Absolute position | AS5600 gives absolute angle at power-on — no homing sequence needed |
| 12-bit resolution (4096 steps/rev) | ~0.088° per step — more than sufficient for control at 500Hz |
| Simple interface | I2C, single address (0x36), 3.3V compatible |
| Contactless | No wear, works through plastic covers |

**Address conflict:** Both AS5600s are fixed at I2C address 0x36. They are placed on
**separate I2C buses** (SERCOM0 for arm, SERCOM4 for pendulum) — this is hardwired into
PCB rev1 and cannot be changed in software.

**Double Furuta (PCB rev2):** A third AS5600 for the second pendulum will require a third
I2C bus on a spare SERCOM.

---

### Display — ST7735S 1.8" TFT LCD (128×160, SPI)

A small colour display for showing system state, encoder angles, and NN output during development and demo.

- SPI interface via SERCOM3 (PA22/PA23) — fast enough for real-time angle display
- Write-only (no MISO line needed) — saves a pin
- 3.3V logic compatible — direct MCU connection, no level shifter

---

## PCB Design — rev1

### Power Architecture

```
USB-C VBUS (5V) ──→ Pi-filter ──→ AMS1117-3.3 LDO ──→ 3.3V (logic)
                       |
                   SMBJ5.0A TVS

Motor connector (12V) ──→ SMBJ12A TVS ──→ Pi-filter ──→ 12VF (motor rail)
```

- **Logic (3.3V):** AMS1117-3.3 LDO from VBUS. Pi-filter on VBUS: 10µF → ferrite bead → 100nF
- **Motor rail (12V):** Enters on J3 pins 1–2, TVS clamps transients, pi-filter reduces switching noise, exits on J3 pins 3–4 to the DRV8313 module
- **PGND / GND isolation:** Motor ground (PGND) and logic ground (GND) are separate copper pours, joined at a single star point near the LDO via ferrite bead (BLM21PG121SN1L, 0805)

### Decoupling Strategy

- 100nF X7R 0402 bypass caps as close as possible to each MCU VDD pin
- 10µF X5R 0805 bulk caps on 3.3V and motor rails
- No Y5V or Z5U dielectrics (capacitance collapses under DC bias)

### Debug Interface

Tag-Connect TC2030 (with legs) footprint — no connector body on the PCB, the cable clips on during programming only.

| Signal | Notes |
|---|---|
| SWDIO / SWDCLK | PA30 / PA31 — standard ARM SWD |
| nRESET | 100nF cap to GND for noise immunity |
| SWO | PA27 — optional trace output |
| VTref | Connected to 3.3V rail |

Programmed via Atmel-ICE using SWD protocol (not JTAG).  
After the UF2 bootloader is flashed once via SWD, subsequent firmware updates can be done via USB (double-tap RESET → FEATHERBOOT drive appears).

### USB

- D+ / D− on PA24 / PA25 with 27Ω series resistors
- ESD protection diode on VBUS and data lines
- USB-C connector

### I2C Pull-ups

4.7kΩ pull-up resistors on SDA and SCL, one pair per bus, placed near the MCU rather than near the sensors.

---

## PCB Revision History

| Rev | Tag | Date | Notes |
|---|---|---|---|
| 1 | [`hw-rev1`](../../../releases/tag/hw-rev1) | 2026-06-07 | First production run — JLCPCB. Single Furuta (2 encoders, 2 I2C buses) |
| 2 | *(planned)* | TBD | Add AS5600 #3 on a third I2C bus (SERCOM) for double Furuta pendulum |
