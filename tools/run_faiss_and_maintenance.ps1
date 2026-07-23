param(
    [string[]]$Datasets = @("caltech", "cub", "coco"),
    [string]$RunRoot = "results/faiss-and-maintenance",
    [int]$MaxQueries = 0,
    [int]$MutationCount = 200,
    [int]$MaintenanceRepeats = 3,
    [int]$MaintenanceWarmupRuns = 1,
    [string]$MaintenanceBackends = "cangjie,faiss",
    [ValidateSet("paper-local", "service")]
    [string]$DatabaseExecutionMode = "service",
    [string]$FaissSourceRoot = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $RunRoot))
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

foreach ($dataset in $Datasets) {
    $artifact = Join-Path $RepoRoot "artifacts/python-paper-90-10/$dataset-full"
    if (!(Test-Path -LiteralPath (Join-Path $artifact "manifest.json"))) {
        throw "missing frozen artifact: $artifact"
    }
    $faissJson = Join-Path $RunRoot "faiss-$dataset.json"
    python (Join-Path $RepoRoot "tools/run_faiss_baseline.py") `
        --artifact $artifact `
        --max-queries $MaxQueries `
        --warmup-queries 20 `
        --repeats 3 `
        --output $faissJson
    if ($LASTEXITCODE -ne 0) { throw "Faiss baseline failed: $dataset" }

    $maintenanceJson = Join-Path $RunRoot "maintenance-$dataset.json"
    python (Join-Path $RepoRoot "tools/run_maintenance_benchmark.py") `
        --artifact $artifact `
        --mutation-count $MutationCount `
        --repeats $MaintenanceRepeats `
        --warmup-runs $MaintenanceWarmupRuns `
        --backends $MaintenanceBackends `
        --execution-mode $DatabaseExecutionMode `
        --local-state-dir (Join-Path $RunRoot "database-state") `
        --output $maintenanceJson
    if ($LASTEXITCODE -ne 0) { throw "maintenance benchmark failed: $dataset" }
}

$locArgs = @(
    (Join-Path $RepoRoot "tools/count_source_lines.py"),
    "--output",
    (Join-Path $RunRoot "source-lines.json")
)
if ($FaissSourceRoot) {
    $locArgs += @("--faiss-root", $FaissSourceRoot)
}
python @locArgs
if ($LASTEXITCODE -ne 0) { throw "source-line count failed" }

Write-Host "All outputs persisted under $RunRoot"
