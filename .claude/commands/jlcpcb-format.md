---
description: Reformat both production files for JLCPCB after a KiCad plugin export — fixes bom.csv (Value→Comment, LCSC Part #→LCSC Part Number) and positions.csv (mm suffix on coordinates, top→T/bottom→B, strips non-assembly parts). Run once after each plugin export before uploading to JLCPCB.
---

# Format BOM + CPL for JLCPCB

Reformats both files the KiCad fabrication plugin generates in `hardware/pcb/production/`.
The logic lives in `hardware/pcb/production/jlcpcb-format.ps1` — edit that file to change behaviour.

## Run

```powershell
& "hardware/pcb/production/jlcpcb-format.ps1"
```

## Output

Both files are overwritten in place. Upload them together on the JLCPCB assembly order page.
