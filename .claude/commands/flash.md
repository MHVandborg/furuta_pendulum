---
description: Flash the firmware binary to the ATSAMD51J20A via Atmel-ICE and OpenOCD over SWD (Tag-Connect TC2030).
---

# Flash Firmware

Build (if not already built) and flash to the board via Atmel-ICE + OpenOCD.

## Steps

1. Build the firmware:
```powershell
cmake -B build && cmake --build build
```

2. Flash via OpenOCD:
```powershell
openocd -f interface/atmel-ice.cfg -f target/atsame5x.cfg `
  -c "program build/furuta_pendulum.elf verify reset exit"
```

If the build directory doesn't exist yet, run the cmake configure step first:
```powershell
cmake -B build -DCMAKE_TOOLCHAIN_FILE=<path-to-toolchain> && cmake --build build
```

## Troubleshooting

- **"Error: init mode failed"** — check that the Atmel-ICE USB is connected and the Tag-Connect TC2030 is seated firmly on J6.
- **"verify FAILED"** — the binary may be too large for flash. Check `arm-none-eabi-size build/furuta_pendulum.elf`.
- **No OpenOCD target found** — ensure you're using the `atsame5x.cfg` target (SAMD51 is in the SAM E5x family).
