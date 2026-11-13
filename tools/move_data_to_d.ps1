# Move large data folders to D: and leave directory junctions at the original paths (2026-11-01).
# Code keeps reading C:\Users\admin\Desktop\EVMALOCALUPDATE\data\... through the junction.
# To undo: remove the junction (rmdir <path>, not Remove-Item -Recurse) and move the folder back.
$ErrorActionPreference = 'Stop'
$proj = 'C:\Users\admin\Desktop\EVMALOCALUPDATE'
$dst = 'D:\EVMALOCALUPDATE_data'
$log = Join-Path $dst 'move_log.txt'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$items = @('data', 'ercot_local_downloader\ercot_sced_rtc_output', 'ercot_local_downloader\ercot_cop_snapshot_output',
           'ercot_local_downloader\ercot_plan_probe_output', 'ercot_local_downloader\ercot_ader_sced_output')
function Stat($p) { $f = Get-ChildItem -LiteralPath $p -Recurse -File -Force -ErrorAction SilentlyContinue; return @($f.Count, [int64](($f | Measure-Object Length -Sum).Sum)) }
foreach ($rel in $items) {
    $src = Join-Path $proj $rel
    $to = Join-Path $dst $rel
    $it = Get-Item -LiteralPath $src -Force -ErrorAction SilentlyContinue
    if (-not $it) { Add-Content $log "$(Get-Date -f s) $rel missing, skipped"; continue }
    if ($it.Attributes -band [IO.FileAttributes]::ReparsePoint) { Add-Content $log "$(Get-Date -f s) $rel already a link, skipped"; continue }
    $before = Stat $src
    Add-Content $log "$(Get-Date -f s) $rel start: $($before[0]) files, $($before[1]) bytes"
    New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null
    robocopy $src $to /E /MOVE /COPY:DAT /DCOPY:DAT /R:2 /W:2 /NP /NFL /NDL /NJH /NJS | Out-Null
    $rc = $LASTEXITCODE
    $after = Stat $to
    $left = if (Test-Path -LiteralPath $src) { (Stat $src)[0] } else { 0 }
    Add-Content $log "$(Get-Date -f s) $rel robocopy exit $rc; at D: $($after[0]) files, $($after[1]) bytes; left at C: $left files"
    if ($rc -ge 8 -or $after[0] -ne $before[0] -or $after[1] -ne $before[1] -or $left -ne 0) {
        Add-Content $log "$(Get-Date -f s) $rel MISMATCH, no junction made; stopping"
        exit 1
    }
    if (Test-Path -LiteralPath $src) { Remove-Item -LiteralPath $src -Force -Recurse }
    cmd /c mklink /J "$src" "$to" | Out-Null
    $ok = (Get-Item -LiteralPath $src -Force).Attributes -band [IO.FileAttributes]::ReparsePoint
    Add-Content $log "$(Get-Date -f s) $rel junction made: $([bool]$ok)"
}
Add-Content $log "$(Get-Date -f s) ALL_DONE"
