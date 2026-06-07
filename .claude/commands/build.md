---
description: Build the firmware with CMake and arm-none-eabi-gcc.
---

# Build Firmware

Build the bare-metal firmware for the ATSAMD51J20A.

## Steps

Configure (first time only):
```powershell
cmake -B build
```

Build:
```powershell
cmake --build build
```

Clean rebuild:
```powershell
cmake --build build --clean-first
```

Check binary size after building:
```powershell
arm-none-eabi-size build/furuta_pendulum.elf
```

## Notes

- Toolchain: `arm-none-eabi-gcc`
- The `firmware/` directory contains the CMakeLists.txt — run cmake from the `firmware/` directory if it is not at the root.
- Flash target is 1MB; RAM is 192KB. Watch `.text` and `.data`/`.bss` sections respectively.
