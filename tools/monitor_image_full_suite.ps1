param(
    [string]$RunRoot = "results/python-paper-90-10/2026-07-22-full",
    [int]$SuitePid = 25760,
    [int]$IntervalSeconds = 10
)

$ErrorActionPreference = "SilentlyContinue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $RunRoot))
$StatusPath = Join-Path $RunRoot "run-status.json"
$ProgressJson = Join-Path $RunRoot "progress.json"
$ProgressHtml = Join-Path $RunRoot "progress.html"

$phases = @(
    [ordered]@{ id = "cangjie/caltech"; label = "Cangjie - Caltech"; minutes = 45 },
    [ordered]@{ id = "cangjie/cub"; label = "Cangjie - CUB"; minutes = 70 },
    [ordered]@{ id = "cangjie/coco"; label = "Cangjie - COCO"; minutes = 60 },
    [ordered]@{ id = "milvus/caltech"; label = "Milvus - Caltech"; minutes = 8 },
    [ordered]@{ id = "qdrant/caltech"; label = "Qdrant - Caltech"; minutes = 8 },
    [ordered]@{ id = "chroma/caltech"; label = "Chroma - Caltech"; minutes = 8 },
    [ordered]@{ id = "milvus/cub"; label = "Milvus - CUB"; minutes = 10 },
    [ordered]@{ id = "qdrant/cub"; label = "Qdrant - CUB"; minutes = 10 },
    [ordered]@{ id = "chroma/cub"; label = "Chroma - CUB"; minutes = 10 },
    [ordered]@{ id = "milvus/coco"; label = "Milvus - COCO"; minutes = 10 },
    [ordered]@{ id = "qdrant/coco"; label = "Qdrant - COCO"; minutes = 10 },
    [ordered]@{ id = "chroma/coco"; label = "Chroma - COCO"; minutes = 10 }
)
$totalWeight = 0.0
foreach ($phase in $phases) { $totalWeight += [double]$phase.minutes }
$lastCurrent = $null
$phaseStarted = [datetimeoffset]::Now

while ($true) {
    if (!(Test-Path -LiteralPath $StatusPath)) {
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }
    $status = Get-Content -LiteralPath $StatusPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $now = [datetimeoffset]::Now
    if ($status.current -ne $lastCurrent) {
        $lastCurrent = $status.current
        if ($status.completed.Count -eq 0 -and $status.current -eq "cangjie/caltech") {
            $phaseStarted = [datetimeoffset]::Parse($status.startedAt)
        } else {
            $phaseStarted = $now
        }
    }
    $completed = @($status.completed)
    $completedWeight = 0.0
    $phaseRows = @()
    foreach ($phase in $phases) {
        $phaseState = "pending"
        $phaseProgress = 0.0
        if ($completed -contains $phase.id) {
            $phaseState = "complete"
            $phaseProgress = 1.0
            $completedWeight += $phase.minutes
        } elseif ($phase.id -eq $status.current -and $status.state -eq "running") {
            $phaseState = "running"
            $elapsed = ($now - $phaseStarted).TotalMinutes
            $phaseProgress = [math]::Min(0.95, $elapsed / [double]$phase.minutes)
            $completedWeight += $phase.minutes * $phaseProgress
        } elseif ($phase.id -eq $status.current -and $status.state -eq "failed") {
            $phaseState = "failed"
        }
        $phaseRows += [ordered]@{
            id = $phase.id
            label = $phase.label
            state = $phaseState
            estimatedPercent = [math]::Round($phaseProgress * 100, 1)
        }
    }
    $suiteProcess = Get-Process -Id $SuitePid -ErrorAction SilentlyContinue
    $worker = Get-Process -Name main -ErrorAction SilentlyContinue |
        Sort-Object StartTime -Descending | Select-Object -First 1
    $progress = [ordered]@{
        updatedAt = $now.ToString("o")
        state = $status.state
        current = $status.current
        overallEstimatedPercent = [math]::Round(100.0 * $completedWeight / $totalWeight, 1)
        elapsedMinutes = [math]::Round(($now - [datetimeoffset]::Parse($status.startedAt)).TotalMinutes, 1)
        suiteProcessRunning = ($null -ne $suiteProcess)
        workerCpuSeconds = if ($worker) { [math]::Round($worker.CPU, 1) } else { $null }
        workerMemoryMB = if ($worker) { [math]::Round($worker.WorkingSet64 / 1MB, 1) } else { $null }
        error = $status.error
        precision = "phase-exact, within-phase-estimated"
        phases = $phaseRows
    }
    $progress | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ProgressJson -Encoding UTF8

    $items = foreach ($row in $phaseRows) {
        $symbol = if ($row.state -eq "complete") { "OK" } elseif ($row.state -eq "running") { ">" } elseif ($row.state -eq "failed") { "!" } else { "." }
        $pct = if ($row.state -eq "running") { " / est. $($row.estimatedPercent)%" } else { "" }
        "<li class='$($row.state)'><span>$symbol</span><b>$($row.label)</b><em>$($row.state)$pct</em></li>"
    }
    $errorHtml = if ($status.error) { "<p class='error'>$($status.error)</p>" } else { "" }
    $html = @"
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="refresh" content="$IntervalSeconds"><title>Violas full benchmark progress</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}body{max-width:980px;margin:32px auto;padding:0 20px;background:Canvas;color:CanvasText}h1{font-size:22px}.meta{display:flex;gap:24px;flex-wrap:wrap;margin:16px 0}.bar{height:18px;background:color-mix(in srgb,CanvasText 12%,Canvas);border-radius:9px;overflow:hidden}.bar i{display:block;height:100%;width:$($progress.overallEstimatedPercent)%;background:#3b82f6}.value{font-size:32px;font-weight:600;margin:8px 0}ul{list-style:none;padding:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px}li{display:grid;grid-template-columns:24px 1fr auto;gap:8px;padding:10px;border-bottom:1px solid color-mix(in srgb,CanvasText 18%,Canvas)}li em{font-style:normal;opacity:.7}.complete span{color:#16a34a}.running{background:color-mix(in srgb,#3b82f6 12%,Canvas)}.running span{color:#3b82f6}.failed,.error{color:#dc2626}.note{opacity:.7;font-size:13px;margin-top:18px}
</style></head><body><h1>Violas full benchmark</h1><div class="value">$($progress.overallEstimatedPercent)%</div><div class="bar"><i></i></div><div class="meta"><span>State: $($status.state)</span><span>Current: $($status.current)</span><span>Elapsed: $($progress.elapsedMinutes) min</span><span>Updated: $($now.ToString('HH:mm:ss'))</span></div>$errorHtml<ul>$($items -join '')</ul><p class="note">Completed phases are exact. Progress inside the current phase is estimated from measured runtime. Auto-refresh: $IntervalSeconds seconds.</p></body></html>
"@
    Set-Content -LiteralPath $ProgressHtml -Value $html -Encoding UTF8
    if ($status.state -in @("complete", "failed") -or !$suiteProcess) { break }
    Start-Sleep -Seconds $IntervalSeconds
}
