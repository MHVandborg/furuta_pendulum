---
description: Extract a BOM (Bill of Materials) from the KiCad schematic — lists all placed components with reference, value, and footprint.
---

# Extract BOM from Schematic

The schematic at `hardware/pcb/furuta_pendulum.kicad_sch` is ~566KB. The `lib_symbols` block ends around line 13107, so the extraction script skips past it before parsing placed instances.

## Run

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

## Output

Tab-separated lines: `Reference  Value  Footprint`

To export to a file:
```powershell
# (append | Out-File bom.tsv to the script above)
$results | Sort-Object | Out-File "hardware/pcb/bom.tsv"
```
