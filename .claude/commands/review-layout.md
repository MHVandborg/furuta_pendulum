---
description: Review the Furuta pendulum PCB layout for DRC violations, net class compliance, and layout guidelines. Reports a structured punch list of findings.
---

# PCB Layout Review

**PREREQUISITE:** The KiCad MCP server must be running and connected. The server takes up to **2 minutes** to initialise (pcbnew/wxApp startup). If the server appears as "still connecting" in the system context, use ToolSearch to wait for it — do not give up immediately.

Before proceeding, attempt a simple query:

```
get_board_info()
```

If the server is still starting up and the tool is not yet available, call ToolSearch repeatedly (up to 3 times, 30 seconds apart) before concluding the server is unavailable. If after ~2 minutes the tool still cannot be called, **stop and report the error to the user.** Do not attempt workarounds, file parsing, or token-expensive manual reviews. The KiCad MCP must be available to proceed.

---

If the above query succeeds, proceed with the full systematic review of `hardware/pcb/furuta_pendulum.kicad_pcb` using the KiCad MCP server. Work through each phase in order, collecting findings, then present a single consolidated report at the end.

## Phase 1 — Board info

Get the board summary:

```
get_board_info()
```

Note: board dimensions, layer count, copper layers, DRC status.

## Phase 2 — DRC

Run the full Design Rule Check:

```
run_drc()
```

Collect all violations. Then get the filtered error list:

```
get_drc_violations(severity: "error")
get_drc_violations(severity: "warning")
```

## Phase 3 — Design rule verification

Get the current design rules and net classes:

```
get_design_rules()
```

Cross-check against required rules from CLAUDE.md:
- Motor phase traces (PWM_A / PWM_B / PWM_C, DRV8313 outputs): must be short and wide — minimum 0.5mm, ideally 1mm for 1A+ continuous
- USB D+/D−: should be in the `USB` net class with matched-length differential pair rules
- I2C nets: 4.7kΩ pull-ups placed near MCU
- PGND and GND are separate copper pours — only joined at star-point (FB4)

## Phase 4 — Layout visual inspection

Render a 2D view of the full board:

```
get_board_2d_view(width: 1600, height: 1200, format: "png", responseMode: "inline")
```

Visually check:
- Decoupling caps are placed close to MCU VDD pins (not on the other side of the board)
- TVS diodes (SMBJ12A on +VM, SMBJ5.0A on VBUS) are close to their respective input connectors
- DRV8313 / motor driver section is grouped near J1 (VM connector)
- Tag-Connect TC2030 SWD pads (J6) are accessible from the top
- USB-C connector and ESD IC are close together

If the full board image is too large to read details, use the board extents to pick a region and re-render:

```
get_board_extents()
```

Then zoom into critical areas (power input, MCU decoupling, motor driver) using a region render if the MCP supports it.

## Phase 5 — Copper pour / zone audit

Check the zone/pour configuration for PGND and GND:
- GND pour should cover logic area (F.Cu and B.Cu)
- PGND pour should cover motor driver area only
- Both should be filled and up to date (no fill errors in DRC)

## Report format

Present findings as a single punch list grouped by severity:

```
## PCB Layout Review — furuta_pendulum

### Board summary
- Size: W × H mm, N copper layers
- DRC status: N errors, N warnings

### Errors (must fix)
- [DRC] ...
- [Layout] ...

### Warnings (should fix)
- [DRC] ...
- [Trace width] ...

### Info / Observations
- [Visual] ...
- [Placement] ...

### Passed checks
- DRC errors: N
- Net class rules: OK / issues listed
- Power separation (PGND vs GND): OK / issues listed
- TVS placement: OK / issues listed
- Decoupling cap placement: OK / issues listed
```

Flag any finding that contradicts CLAUDE.md layout priorities as an **Error**.
