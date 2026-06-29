# Bootloader

Adafruit UF2 bootloader for the ATSAMD51J20A on the Furuta Pendulum Controller PCB.

**File:** `bootloader-feather_m4-v4.0.0.bin`  
**Source:** https://github.com/adafruit/uf2-samdx1/releases/tag/v4.0.0

## Flash via Atmel-ICE (first time or recovery)

Requires: Atmel-ICE connected via SWD, OpenOCD installed.

```powershell
openocd -f interface/cmsis-dap.cfg -f target/atsame5x.cfg -c "program bootloader/bootloader-feather_m4-v4.0.0.bin verify reset exit 0x00000000"
```

Run this from the project root: `C:\Electronics\furuta_pendulum`

## Enter bootloader mode (USB)

Double-tap RESET quickly (<500ms between taps).  
Board appears as USB drive **FEATHERBOOT** — drag and drop a `.uf2` firmware file onto it.

## Normal firmware update (after bootloader is installed)

1. Double-tap RESET → FEATHERBOOT drive appears
2. Copy your `.uf2` file onto the drive
3. Board reboots automatically into new firmware
