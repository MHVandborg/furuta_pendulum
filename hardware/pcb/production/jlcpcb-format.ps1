# --- BOM ---
$bom = Import-Csv "$PSScriptRoot\bom.csv"
$bom | ForEach-Object {
    [PSCustomObject]@{
        Comment            = if ($_.Value) { $_.Value } else { $_.Comment }
        Designator         = ($_.Designator -replace ',\s+', ',')
        Footprint          = $_.Footprint
        "LCSC Part Number" = $(
            $p = if ($_."LCSC Part #") { $_."LCSC Part #" } elseif ($_."LCSC Part Number") { $_."LCSC Part Number" } else { $_."JLCPCB Part#" }
            if ($p -eq "0") { "" } else { $p }
        )
    }
} | Export-Csv "$PSScriptRoot\bom.csv" -NoTypeInformation -UseQuotes AsNeeded

# --- CPL ---
# Filter to only designators present in the BOM (removes fiducials, mounting holes, Tag-Connect etc.)
$bomDesignators = @{}
Import-Csv "$PSScriptRoot\bom.csv" | ForEach-Object {
    ($_.Designator -split ',') | ForEach-Object { $bomDesignators[$_.Trim()] = $true }
}

$cpl = Import-Csv "$PSScriptRoot\positions.csv"
$cpl | Where-Object { $bomDesignators.ContainsKey($_.Designator) } | ForEach-Object {
    $x = $_.("Mid X") -replace "mm$", ""
    $y = $_.("Mid Y") -replace "mm$", ""
    [PSCustomObject]@{
        Designator = $_.Designator
        "Mid X"    = $x + "mm"
        "Mid Y"    = $y + "mm"
        Layer      = if ($_.Layer -eq "top") { "T" } elseif ($_.Layer -eq "bottom") { "B" } else { $_.Layer }
        Rotation   = $_.Rotation
    }
} | Export-Csv "$PSScriptRoot\positions.csv" -NoTypeInformation -UseQuotes AsNeeded

Write-Host "Done. bom.csv and positions.csv ready for JLCPCB."
