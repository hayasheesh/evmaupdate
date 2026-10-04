param(
    [Parameter(Mandatory = $true, Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $CommandArgs
)

$ErrorActionPreference = 'Stop'

function Get-ActivePowerSchemeGuid {
    $activeScheme = powercfg /getactivescheme
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to read the active power scheme.'
    }
    $match = [regex]::Match(
        ($activeScheme -join ' '),
        '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
    )
    if (-not $match.Success) {
        throw "Could not parse the active power-scheme GUID: $activeScheme"
    }
    return $match.Value
}

function Invoke-PowerCfg {
    param([Parameter(Mandatory = $true)][string[]] $Arguments)
    & powercfg @Arguments | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "powercfg failed: $($Arguments -join ' ')"
    }
}

function Set-ExperimentPowerLimit {
    $schemeGuid = Get-ActivePowerSchemeGuid
    Invoke-PowerCfg -Arguments @('/setacvalueindex', $schemeGuid, 'SUB_PROCESSOR', 'PROCTHROTTLEMAX', '50')
    Invoke-PowerCfg -Arguments @('/setacvalueindex', $schemeGuid, 'SUB_PROCESSOR', 'PROCTHROTTLEMIN', '5')
    Invoke-PowerCfg -Arguments @('/setacvalueindex', $schemeGuid, 'SUB_PROCESSOR', 'PERFBOOSTMODE', '0')
    Invoke-PowerCfg -Arguments @('/setactive', $schemeGuid)
    Write-Host "[power] AC processor state set to max=50%, min=5%, boost=0 ($schemeGuid)."
    Write-Host '[power] IMPORTANT: the wrapper will restore max=100%, min=100%, boost=2 on exit.'
}

function Restore-ExperimentPowerPlan {
    try {
        $schemeGuid = Get-ActivePowerSchemeGuid
        Invoke-PowerCfg -Arguments @('/setacvalueindex', $schemeGuid, 'SUB_PROCESSOR', 'PROCTHROTTLEMAX', '100')
        Invoke-PowerCfg -Arguments @('/setacvalueindex', $schemeGuid, 'SUB_PROCESSOR', 'PROCTHROTTLEMIN', '100')
        Invoke-PowerCfg -Arguments @('/setacvalueindex', $schemeGuid, 'SUB_PROCESSOR', 'PERFBOOSTMODE', '2')
        Invoke-PowerCfg -Arguments @('/setactive', $schemeGuid)
    }
    catch {
        Write-Warning 'CPU power-plan restore failed. Restore max=100%, min=100%, boost=2 manually.'
        Write-Warning $_
        return
    }
    Write-Host "[power] Restored AC processor state to max=100%, min=100%, boost=2 ($schemeGuid)."
}

if (-not $CommandArgs -or $CommandArgs.Count -eq 0) {
    throw 'Pass the executable and its arguments after the wrapper path.'
}

$exitCode = 1
try {
    Set-ExperimentPowerLimit
    $executable = $CommandArgs[0]
    $arguments = if ($CommandArgs.Count -gt 1) {
        $CommandArgs[1..($CommandArgs.Count - 1)]
    } else {
        @()
    }
    & $executable @arguments
    $exitCode = $LASTEXITCODE
}
finally {
    Restore-ExperimentPowerPlan
}

exit $exitCode
