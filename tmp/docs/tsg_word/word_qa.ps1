param(
    [Parameter(Mandatory=$true)][string]$DocxPath,
    [Parameter(Mandatory=$true)][string]$PdfPath,
    [switch]$SaveCurrent
)
$ErrorActionPreference = 'Stop'
$ownedApp = $false
try { $word = [Runtime.InteropServices.Marshal]::GetActiveObject('Word.Application') }
catch { $word = New-Object -ComObject Word.Application; $word.Visible = $false; $ownedApp = $true }
$oldAlerts = $word.DisplayAlerts
$oldSecurity = $word.AutomationSecurity
$word.DisplayAlerts = 0
$word.AutomationSecurity = 3
$opened = $false
$doc = $null
try {
    foreach ($candidate in $word.Documents) {
        if ($candidate.FullName -eq $DocxPath) { $doc = $candidate; break }
    }
    if ($null -eq $doc) {
        $doc = $word.Documents.Open($DocxPath, $false, $true, $false)
        $opened = $true
    }
    $wasSaved = $doc.Saved
    if ($SaveCurrent -and -not $doc.Saved) {
        if ($doc.ReadOnly) { throw 'The requested draft is read-only.' }
        $doc.Save()
    }
    $doc.Repaginate()
    $doc.ExportAsFixedFormat($PdfPath, 17)
    [pscustomobject]@{
        Name=$doc.Name; FullName=$doc.FullName; SavedBefore=$wasSaved;
        Saved=$doc.Saved; ReadOnly=$doc.ReadOnly; Pages=$doc.ComputeStatistics(2);
        Paragraphs=$doc.Paragraphs.Count; Tables=$doc.Tables.Count;
        InlineShapes=$doc.InlineShapes.Count
    } | ConvertTo-Json -Compress
}
finally {
    try { if ($opened -and $null -ne $doc) { $doc.Close() } } catch { }
    try { $word.DisplayAlerts = $oldAlerts; $word.AutomationSecurity = $oldSecurity } catch { }
    try { if ($ownedApp) { $word.Quit() } } catch { }
}
