param(
    [string[]]$Datasets = @("caltech", "cub", "coco"),
    [string[]]$Backends = @("milvus", "qdrant", "chroma"),
    [string]$RunRoot = "results/python-paper-90-10/2026-07-22-full",
    [int]$MaxQueries = 200,
    [switch]$LiveOutput,
    [switch]$Resume,
    [switch]$SkipCangjie,
    [switch]$SkipDatabases
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $RunRoot))
$LogRoot = Join-Path $RunRoot "logs"
$CangjieRoot = Join-Path $RunRoot "cangjie"
$ExternalRoot = Join-Path $RunRoot "external"
$StatusPath = Join-Path $RunRoot "run-status.json"
$BetaList = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

New-Item -ItemType Directory -Force -Path $LogRoot, $CangjieRoot, $ExternalRoot | Out-Null

$status = [ordered]@{
    schemaVersion = 1
    startedAt = (Get-Date).ToString("o")
    finishedAt = $null
    state = "running"
    datasets = $Datasets
    backends = $Backends
    maxQueries = $MaxQueries
    betas = $BetaList.Split(",")
    completed = @()
    current = $null
    error = $null
}

function Save-Status {
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-Persisted {
    param([scriptblock]$Action, [string]$LogPath, [string]$Label)
    $status.current = $Label
    Save-Status
    $exitCode = 0
    if ($LiveOutput) {
        # A transcript keeps native stdout attached to the terminal, avoiding the
        # buffering caused by Tee-Object while still persisting every result.
        Start-Transcript -LiteralPath $LogPath -Force | Out-Null
        try {
            & $Action 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            Stop-Transcript | Out-Null
        }
    } else {
        & $Action 2>&1 | Tee-Object -FilePath $LogPath
        $exitCode = $LASTEXITCODE
    }
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode"
    }
    $status.completed += $Label
    Save-Status
}

function Test-CangjieComplete {
    param([string]$LogPath)
    if (!(Test-Path -LiteralPath $LogPath)) { return $false }
    $rows = @((Select-String -LiteralPath $LogPath -Pattern "PAPER_SUMMARY|" -SimpleMatch))
    $optimized = @((Select-String -LiteralPath $LogPath -Pattern "implementation=normalized-f64-o2" -SimpleMatch))
    return ($rows.Count -ge 11 -and $optimized.Count -ge 11)
}

function Test-ExternalComplete {
    param([string]$JsonPath)
    if (!(Test-Path -LiteralPath $JsonPath)) { return $false }
    try {
        $payload = Get-Content -LiteralPath $JsonPath -Raw -Encoding UTF8 | ConvertFrom-Json
        return (@($payload.runs).Count -ge 11 -and
                $payload.config.evaluationProtocol -eq "python-image-benchmark-mixed-gt-v2")
    } catch {
        return $false
    }
}

function Mark-Completed {
    param([string]$Label)
    if ($status.completed -notcontains $Label) { $status.completed += $Label }
    Save-Status
}

try {
    docker version | Set-Content -LiteralPath (Join-Path $RunRoot "docker-version.txt") -Encoding UTF8
    docker ps --format "table {{.Names}}`t{{.Image}}`t{{.Status}}`t{{.Ports}}" |
        Set-Content -LiteralPath (Join-Path $RunRoot "docker-containers.txt") -Encoding UTF8
    python --version 2>&1 | Set-Content -LiteralPath (Join-Path $RunRoot "python-version.txt") -Encoding UTF8
    git -C $RepoRoot rev-parse HEAD | Set-Content -LiteralPath (Join-Path $RunRoot "git-commit.txt") -Encoding UTF8

    foreach ($dataset in $Datasets) {
        $artifact = Join-Path $RepoRoot "artifacts/python-paper-90-10/$dataset-full"
        $inputPath = Join-Path $artifact "cangjie_input.txt"
        if (!(Test-Path -LiteralPath $inputPath)) {
            throw "missing Cangjie input: $inputPath"
        }

        if (!$SkipCangjie) {
            $log = Join-Path $CangjieRoot "$dataset.log"
            $summaryLog = Join-Path $CangjieRoot "$dataset.summary.log"
            $cjRoot = Join-Path $RepoRoot "cj_core"
            if ($Resume -and (Test-CangjieComplete -LogPath $summaryLog)) {
                Write-Host "RESUME: cangjie/$dataset already complete; skipping"
                Mark-Completed -Label "cangjie/$dataset"
            } else {
                Invoke-Persisted -Label "cangjie/$dataset" -LogPath $log -Action {
                    Push-Location $cjRoot
                    try {
                        "paper $inputPath $MaxQueries all $summaryLog" | cjpm run
                    } finally {
                        Pop-Location
                    }
                }
            }
        }

        if (!$SkipDatabases) {
            foreach ($backend in $Backends) {
                $json = Join-Path $ExternalRoot "$backend-$dataset-full.json"
                $log = Join-Path $LogRoot "$backend-$dataset-full.log"
                if ($Resume -and (Test-ExternalComplete -JsonPath $json)) {
                    Write-Host "RESUME: $backend/$dataset already complete; skipping"
                    Mark-Completed -Label "$backend/$dataset"
                } else {
                    Invoke-Persisted -Label "$backend/$dataset" -LogPath $log -Action {
                        python (Join-Path $RepoRoot "tools/external_db_benchmark.py") `
                            --backend $backend `
                            --artifact $artifact `
                            --scale full `
                            --max-queries $MaxQueries `
                            --betas $BetaList `
                            --candidate-multiplier 10 `
                            --output $json
                    }
                }
            }
        }
    }

    python (Join-Path $RepoRoot "tools/summarize_image_full_results.py") --run-root $RunRoot
    if ($LASTEXITCODE -ne 0) { throw "result summarization failed with exit code $LASTEXITCODE" }
    $status.state = "complete"
    $status.current = $null
    $status.finishedAt = (Get-Date).ToString("o")
    Save-Status
} catch {
    $status.state = "failed"
    $status.error = $_.Exception.Message
    $status.finishedAt = (Get-Date).ToString("o")
    Save-Status
    throw
}
