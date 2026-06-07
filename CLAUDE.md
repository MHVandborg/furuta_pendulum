# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Furuta (rotary inverted) pendulum controlled by a neural network running on a custom PCB. The goal is swing-up and balance using a trained NN controller on an ARM Cortex-M4F MCU.

**Control flow:** AS5600 arm encoder + AS5600 pendulum encoder → Neural Network (θ_arm, θ_pendulum, ω_arm, ω_pendulum → torque_command) → SimpleFOC / DRV8313 → GBM2804H-100T BLDC

## Repository Structure

| Directory | Contents |
|-----------|----------|
| `firmware/` | Bare-metal C, ARM GCC + CMake for ATSAMD51J20A |
| `hardware/pcb/` | KiCad schematic + PCB layout |
| `hardware/mechanical/` | Fusion 360 3D design files |
| `neural_network/` | Training scripts and exported model weights |
| `docs/` | Datasheets and reference docs |

## Firmware

### Build & Flash

```bash
cmake -B build && cmake --build build
```

Flash via Atmel-ICE + OpenOCD over SWD (Tag-Connect TC2030, J6 header):

```bash
openocd -f interface/atmel-ice.cfg -f target/atsame5x.cfg \
  -c "program build/furuta_pendulum.elf verify reset exit"
```

Debug output goes to USB CDC (no UART printf).

### Rules — non-negotiable

- **Pure C (C11)** — no C++, no HAL, no ASF, no Arduino
- **No dynamic allocation** — no `malloc`, no `new`
- Direct register access via CMSIS headers only — never via library calls
- `volatile` on all hardware register accesses
- Control loop target: **500–1000Hz**

### Naming Conventions

- Functions: `snake_case` with module prefix — `i2c_init()`, `encoder_read_angle()`
- Constants: `ALL_CAPS` with module prefix — `AS5600_ADDR`, `PWM_FREQ_HZ`
- Peripherals: exact CMSIS names — `SERCOM0`, `TC0`, `TCC0`

### Pin Assignment

Pins are grouped by peripheral block to keep related signals physically adjacent on the package:
- **PA00–PA01** — encoder I2C bus 0 (arm)
- **PA08–PA14** — motor driver block: 3-phase PWM + all four DRV8313 control signals
- **PA19–PA23** — LCD block: control GPIOs + SERCOM3 SPI
- **PA24–PA25** — USB
- **PA30–PA31** — SWD debug
- **PB01–PB03** — UI block: button + status LEDs
- **PB08–PB09** — encoder I2C bus 1 (pendulum)

| Pins | Function | Peripheral |
|------|----------|------------|
| PA00 / PA01 | I2C0_SDA / I2C0_SCL (arm encoder) | SERCOM0 |
| PA08 / PA09 / PA10 | PWM_A / PWM_B / PWM_C | TCC0 WO[0/1/2] |
| PA11 | DRV_EN (active high) | GPIO out |
| PA12 | DRV_FAULT (~nFT, active low) | GPIO in, EIC interrupt |
| PA13 | DRV_SLEEP (~nSP, active low) | GPIO out |
| PA14 | DRV_RESET (~nRT, active low) | GPIO out |
| PB01 | BTN1 | GPIO in, EIC interrupt |
| PB02 / PB03 | LED1 / LED2 | GPIO out |
| PA19 | LCD_RST | GPIO out |
| PA20 | LCD_CS | GPIO out |
| PA21 | LCD_DC | GPIO out |
| PA22 / PA23 | LCD_MOSI / LCD_SCK | SERCOM3 SPI |
| PA24 / PA25 | USB_DM / USB_DP | USB |
| PA30 / PA31 | SWDIO / SWDCLK | SWD |
| PB08 / PB09 | I2C1_SDA / I2C1_SCL (pendulum encoder) | SERCOM4 |

### Peripheral Configuration

- **I2C0** (arm encoder): SERCOM0, PA00/PA01, address 0x36
- **I2C1** (pendulum encoder): SERCOM4, PB08/PB09, address 0x36 — **separate bus because both AS5600s share address**
- **SPI** (LCD ST7735S): SERCOM3, PA22/PA23, write-only (no MISO)
- **TCC0** (3-phase PWM): center-aligned, complementary outputs PA08–PA10
- **USB**: CDC class — debug output only, no UART
- **EIC**: PA12 DRV_FAULT falling edge; PB01 BTN1 press
- **Clock**: internal DFLL48M locked to USB SOF — no external crystal

### DRV8313 Startup Sequence

1. Boot with DRV_EN low, DRV_SLEEP high
2. Run encoder init and self-checks
3. Assert DRV_EN high only when ready
4. Monitor DRV_FAULT via EIC — on fault: disable DRV_EN, log, pulse DRV_RESET

### Neural Network Inference

- CMSIS-NN — `arm_fully_connected_f32` and related functions
- Weights in flash as `const float` arrays
- Inputs in radians / rad·s⁻¹; output torque command normalized to −1.0…+1.0

---

## PCB (KiCad)

### Power Architecture

- **Logic**: USB-C VBUS (5V) → AMS1117-3.3 LDO → 3.3V
- **Motor**: J3 pin 2 (12V in) → SMBJ12A TVS (D2) → FB2 pi-filter → J3 pin 4 (+12VF out to DRV8313 module). J3 pins 1 & 3 are PGND in/out. J3 is a single 2×7 pass-through connector — motor supply enters on pins 1–2, filtered supply exits on pins 3–4, control signals on pins 5–14.
- TVS packages: SMB (DO-214AA), footprint `Diode_SMD:D_SMB`
- Pi-filter on VBUS: 10µF → ferrite bead (~600Ω@100MHz) → 100nF
- PGND and GND joined via ferrite bead (FB4, BLM21PG121SN1L, 0805) at star point near LDO

### Net Naming

- KiCad-assigned names are fine — do not rename nets just for convention
- Exception: motor-side ground = **`PGND`**; logic ground = `GND`

### Capacitors

- Bypass 100nF: X7R 0402 MLCC
- Bulk 10µF: X5R 0805 MLCC
- Never Y5V or Z5U

### Layout Priorities

1. Decouple caps as close to MCU VDD pins as possible
2. Motor phase traces: short and wide (1A+ per phase)
3. PGND and GND are separate pours — connect only at star point
4. USB D+/D−: matched-length differential pair
5. TVS diodes close to their input connectors
6. I2C pull-ups (4.7kΩ) near MCU, one pair per bus

### Reading the Schematic

KiCad `.kicad_sch` files are S-expression text. The file at `hardware/pcb/furuta_pendulum.kicad_sch` is ~566KB — use targeted grep, not full reads. Key patterns:

```
(symbol (lib_id "...") ... (property "Reference" "R1") (property "Value" "10k") (property "Footprint" "..."))
(label "NET_NAME" ...)
(wire (pts (xy x1 y1) (xy x2 y2)))
```

To extract all placed components (ref / value / footprint), the `lib_symbols` block ends around line 13107 — skip past it:

```powershell
$sch = Get-Content "hardware/pcb/furuta_pendulum.kicad_sch"
$i = 0; $capturing = $false; $depth = 0; $results = @()
foreach ($line in $sch) {
    $i++
    if ($i -lt 13108) { continue }
    if ($line -eq "`t(symbol") { $capturing = $true; $depth = 1; $ref=""; $val=""; $fp=""; continue }
    if ($capturing) {
        if ($line -match 'lib_id "([^"]+)"') { $libid = $Matches[1] }
        if ($line -match 'property "Reference" "([^"]+)"') { $ref = $Matches[1] }
        if ($line -match 'property "Value" "([^"]+)"') { $val = $Matches[1] }
        if ($line -match 'property "Footprint" "([^"]+)"') { $fp = $Matches[1] }
        $depth += ([regex]::Matches($line,'\(')).Count - ([regex]::Matches($line,'\)')).Count
        if ($depth -le 0) { $capturing = $false; if ($ref -notmatch '^#') { $results += "$ref`t$val`t$fp" } }
    }
}
$results | Sort-Object
```
