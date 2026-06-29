# Firmware

Bare-metal C firmware for the ATSAMD51J20A. Implements sensor reading, motor control, and
real-time neural network inference using TensorFlow Lite Micro.

---

## Architecture

```
firmware/
├── hal/                   # Low-level peripheral drivers (direct register access)
│   ├── i2c.c/h            # SERCOM I2C (two independent buses)
│   ├── spi.c/h            # SERCOM SPI (LCD)
│   ├── pwm.c/h            # TCC0 3-phase centre-aligned PWM
│   ├── usb_cdc.c/h        # USB CDC — debug printf output
│   └── gpio.c/h           # GPIO init helpers
├── drivers/               # Device-level drivers (built on hal/)
│   ├── as5600.c/h         # AS5600 encoder: read raw angle, compute velocity
│   └── st7735s.c/h        # ST7735S LCD: init, draw text/values
├── control/               # Controller logic
│   ├── foc.c/h            # DRV8313 interface: torque command → PWM duty cycles
│   ├── nn_inference.c/h   # TFLM inference: state[4] → torque_out
│   └── nn_model_data.cc   # Generated: xxd of the .tflite model (flash-stored weights)
└── main.c                 # Entry point, system init, 500Hz control loop
```

---

## Control Loop (500Hz)

```
1. Read AS5600 arm encoder      (I2C0 — SERCOM0, PA00/PA01)
2. Read AS5600 pendulum encoder (I2C1 — SERCOM4, PB08/PB09)
3. Differentiate angles → angular velocities (+ low-pass filter)
4. Normalise state: [θ_arm, θ_p, ω_arm, ω_p]
5. nn_inference(state, &torque)
6. foc_set_torque(torque)        → TCC0 PWM → DRV8313
```

---

## Build

### Prerequisites

- ARM GCC (`arm-none-eabi-gcc`)
- CMake ≥ 3.20
- VS Code with the Cortex-Debug extension (optional but recommended)

### Compile

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Output: `build/furuta_pendulum.elf` and `build/furuta_pendulum.uf2`

### Flash (first time — SWD via Atmel-ICE)

```bash
openocd -f interface/cmsis-dap.cfg -f target/atsame5x.cfg \
  -c "program build/furuta_pendulum.elf verify reset exit"
```

### Flash (subsequent updates — USB UF2)

1. Double-tap the RESET button quickly (< 500ms between taps)
2. A USB drive called **FEATHERBOOT** appears
3. Copy `build/furuta_pendulum.uf2` onto the drive
4. Board reboots automatically into new firmware

---

## Coding Rules (non-negotiable)

- **Pure C11** — no C++, no HAL, no ASF, no Arduino frameworks
- **No dynamic allocation** — no `malloc`, no `new`; all buffers are statically sized
- **Direct register access** via CMSIS headers only (`SERCOM0->I2CM.CTRLA.reg = ...`)
- `volatile` on all hardware register accesses
- Control loop target: **500–1000Hz**

---

## Peripheral Map

| Peripheral | Function | Pins |
|---|---|---|
| SERCOM0 (I2C) | AS5600 arm encoder | PA00 SDA, PA01 SCL |
| SERCOM3 (SPI) | ST7735S LCD | PA22 MOSI, PA23 SCK |
| SERCOM4 (I2C) | AS5600 pendulum encoder | PB08 SDA, PB09 SCL |
| TCC0 | 3-phase PWM (FOC) | PA08 WO0, PA09 WO1, PA10 WO2 |
| USB | CDC debug output | PA24 DM, PA25 DP |
| EIC | DRV_FAULT interrupt | PA12 |
| EIC | Button interrupt | PB01 |
| GPIO out | DRV_EN, DRV_SLEEP, DRV_RESET | PA11, PA13, PA14 |
| GPIO out | LED1, LED2 | PB02, PB03 |
| GPIO out | LCD_RST, LCD_CS, LCD_DC | PA19, PA20, PA21 |
| SWD | Debug / programming | PA30 SWDIO, PA31 SWDCLK |

---

## Neural Network Inference (TFLM)

The trained Keras model is converted to a `.tflite` file and then embedded as a C byte array
in flash. TensorFlow Lite Micro (TFLM) interprets the model at runtime — no dynamic
allocation, no operating system required.

**Generating `nn_model_data.cc` from a trained model:**

```bash
# On the training machine (Python)
python neural_network/export_model.py          # outputs furuta_policy.tflite

# Convert to C array
xxd -i furuta_policy.tflite > firmware/control/nn_model_data.cc
```

**Inference in firmware (`nn_inference.c`):**
- Static tensor arena (no heap)
- Input: `float state[4]` — normalised [θ_arm, θ_p, ω_arm, ω_p]
- Output: `float torque` — clamped to [−1.0, +1.0]
- Typical latency: < 100µs at 120MHz on the M4F

---

## Debug Output

All debug output goes over **USB CDC** (`printf` → USB serial, no UART).  
Connect with any serial terminal at any baud rate (USB CDC is baud-agnostic):

```bash
# Linux / macOS
screen /dev/ttyACM0

# Windows — use PuTTY or VS Code serial monitor
```
