param(
    [string]$TectonicPath = 'C:\Users\admin\tools\tectonic\tectonic.exe'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $TectonicPath -PathType Leaf)) {
    throw "Tectonic executable not found: $TectonicPath"
}
Push-Location $PSScriptRoot
try {
    & $TectonicPath -X compile --keep-logs build_tsg.tex
    if ($LASTEXITCODE -ne 0) {
        throw "Tectonic compilation failed with exit code $LASTEXITCODE"
    }
    Copy-Item -LiteralPath build_tsg.pdf -Destination paper.pdf -Force
}
finally {
    Pop-Location
}
