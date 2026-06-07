---
description: Review the Furuta pendulum schematic for ERC violations, missing footprints, power connectivity issues, and design-rule compliance. Reports a structured punch list of findings.
---

# Schematic Review

**PREREQUISITE:** The KiCad MCP server must be running and connected. The server takes up to **2 minutes** to initialise (pcbnew/wxApp startup). If the server appears as "still connecting" in the system context, use ToolSearch to wait for it — do not give up immediately.

Before proceeding, attempt a simple query:

```
list_schematic_components(schematicPath: "hardware/pcb/furuta_pendulum.kicad_sch")
```

If the server is still starting up and the tool is not yet available, call ToolSearch repeatedly (up to 3 times, 30 seconds apart) before concluding the server is unavailable. If after ~2 minutes the tool still cannot be called, **stop and report the error to the user.** Do not attempt workarounds, file parsing, or token-expensive manual reviews. The KiCad MCP must be available to proceed.

---

If the above query succeeds, proceed with the full systematic review of `hardware/pcb/furuta_pendulum.kicad_sch` using the KiCad MCP server. Work through each phase in order, collecting findings, then present a single consolidated report at the end.

## Phase 1 — ERC

Run the Electrical Rules Check and collect all violations:

```
run_erc(schematicPath: "hardware/pcb/furuta_pendulum.kicad_sch")
```

Note every error and warning with its severity and location.

## Phase 2 — Component audit

List all placed components and check for:
- Missing footprint (empty Footprint field)
- Power symbols used as components (reference starts with `#PWR`) — these are fine, just skip them
- Unexpected duplicate references

```
list_schematic_components(schematicPath: "hardware/pcb/furuta_pendulum.kicad_sch")
```

## Phase 3 — Net connectivity

Generate the full netlist and verify the following critical nets exist and have the expected connections:

```
generate_netlist(schematicPath: "hardware/pcb/furuta_pendulum.kicad_sch")
```

Check:
- `+3.3V` reaches MCU VDD pins, AS5600 VDD, LCD VCC, and pull-up resistors
- `+5V` (VBUS) reaches the AMS1117-3.3 LDO input
- `GND` is connected to MCU GND pins, LDO GND, USB shield
- `PGND` is isolated from GND except at the star-point ferrite bead (FB4)
- `+VM` (+12V motor rail) goes only to DRV8313 and TVS
- I2C buses are separate: `I2C0_SDA`/`I2C0_SCL` on SERCOM0 (PA00/PA01), `I2C1_SDA`/`I2C1_SCL` on SERCOM4 (PB08/PB09)
- USB: `USB_DM`/`USB_DP` on PA24/PA25 with 27Ω series resistors before ESD IC

## Phase 4 — Wiring quality

Run all three wiring-quality checks:

```
find_floating_labels(schematicPath: "hardware/pcb/furuta_pendulum.kicad_sch")
find_orphaned_wires(schematicPath: "hardware/pcb/furuta_pendulum.kicad_sch")
find_wires_crossing_symbols(schematicPath: "hardware/pcb/furuta_pendulum.kicad_sch")
find_overlapping_elements(schematicPath: "hardware/pcb/furuta_pendulum.kicad_sch")
```

## Phase 5 — Design rule spot-checks

Cross-reference component values against known requirements from CLAUDE.md:

- Decoupling caps: check that 100nF X7R 0402 caps exist for each MCU VDD pin; 10µF X5R 0805 bulk caps per power domain
- No Y5V or Z5U dielectric in any capacitor value fields
- TVS diodes: SMBJ12A on +VM, SMBJ5.0A on VBUS — verify both are present
- Pi-filter on VBUS: 10µF cap → ferrite bead → 100nF cap
- I2C pull-ups: 4.7kΩ resistors on each I2C bus (two pairs total)
- SWD: Tag-Connect TC2030 with 100nF cap on nRESET to GND
- USB: 27Ω on D+ and D-, ESD IC (USBLC6-2SC6)

## Phase 6 — Visual check

Render a PNG of the schematic for a visual sanity check:

```
get_schematic_view(schematicPath: "hardware/pcb/furuta_pendulum.kicad_sch", format: "png", width: 1600, height: 1200)
```

## Report format

Present findings as a single punch list grouped by severity:

```
## Schematic Review — furuta_pendulum

### Errors (must fix)
- [ERC] ...
- [Net] ...

### Warnings (should fix)
- [ERC] ...
- [Wiring] ...

### Info / Observations
- [BOM] ...
- [Visual] ...

### Passed checks
- ERC: N violations
- Floating labels: none
- Orphaned wires: none
- Wires crossing symbols: none
- Power net connectivity: OK
```

Flag any finding that contradicts CLAUDE.md rules as an **Error**.
