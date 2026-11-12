# E4-B (repeat-visit specialist) then 224 seed 43, each followed by a dev evaluation.
# Usage (from D:\challenge\AI):
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_e4b_seed43.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_e4b_seed43.ps1 -Python D:\Program\Anaconda\envs\pytorch\python.exe

param(
    [string]$Python = ''
)

$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = 'src'
$env:PYTHONUNBUFFERED = '1'

# Pick the first interpreter whose torch can actually see the GPU (a fresh
# powershell.exe may auto-activate the CPU-only "base" env).
$candidates = @()
if ($Python) { $candidates += $Python }
if ($env:CONDA_PREFIX) { $candidates += (Join-Path $env:CONDA_PREFIX 'python.exe') }
$candidates += 'D:\Program\Anaconda\envs\pytorch\python.exe'
$onPath = Get-Command python -ErrorAction SilentlyContinue
if ($onPath) { $candidates += $onPath.Source }
$candidates = $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique

$py = $null
foreach ($candidate in $candidates) {
    & $candidate -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) { $py = $candidate; break }
    Write-Host "skip $candidate (no CUDA)" -ForegroundColor DarkGray
}
if (-not $py) {
    Write-Host "ERROR: no interpreter with CUDA found." -ForegroundColor Red
    exit 1
}
Write-Host "python: $py" -ForegroundColor Cyan
& $py -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.get_device_name(0))"

$manifest = '..\docs\splits\longitudinal_200.json'
$jobs = @(
    'configs\p8\p8_spec_224_s42.json',
    'configs\p8\p8_b0_224_12ep_s43.json'
)

$started = Get-Date
Write-Host "=== start $($started.ToString('s')) ===" -ForegroundColor Cyan

foreach ($config in $jobs) {
    $name = [IO.Path]::GetFileNameWithoutExtension($config)
    $runDir = Join-Path 'runs\p8' $name
    $logPath = Join-Path 'runs' "$name.log"

    if (Test-Path (Join-Path $runDir 'best.pt')) {
        Write-Host "=== $name already trained, skipping ===" -ForegroundColor Yellow
    }
    else {
        if (Test-Path $runDir) {
            Write-Host "=== ${name}: removing incomplete run dir ===" -ForegroundColor Yellow
            Remove-Item $runDir -Recurse -Force
        }
        Write-Host "=== $name train start $((Get-Date).ToString('s')) ===" -ForegroundColor Green
        & $py -m cxr train --config $config --holdout-manifest $manifest 2>&1 |
            Tee-Object -FilePath $logPath
        Write-Host "=== $name train exit $LASTEXITCODE $((Get-Date).ToString('s')) ===" -ForegroundColor Green
    }

    if (Test-Path (Join-Path $runDir 'best.pt')) {
        Write-Host "=== $name dev evaluation ===" -ForegroundColor Green
        & $py -m cxr evaluate --config $config `
            --checkpoint "runs/p8/$name/best.pt" `
            --split train --holdout-manifest $manifest 2>&1 |
            Tee-Object -FilePath (Join-Path 'runs' "$name.eval.log")
    }
}

$elapsed = (Get-Date) - $started
Write-Host "=== done, elapsed $([math]::Round($elapsed.TotalMinutes, 1)) min ===" -ForegroundColor Cyan

Write-Host "`n=== summary ===" -ForegroundColor Cyan
Get-ChildItem 'runs\p8' -Directory | ForEach-Object {
    $dev = Join-Path $_.FullName 'evaluation_dev.json'
    $best = Join-Path $_.FullName 'best_metrics.json'
    $devAuc = if (Test-Path $dev) { (Get-Content $dev -Raw | ConvertFrom-Json).macro_auc } else { $null }
    $valAuc = if (Test-Path $best) { (Get-Content $best -Raw | ConvertFrom-Json).macro_auc } else { $null }
    [pscustomobject]@{
        run   = $_.Name
        dev   = if ($null -ne $devAuc) { [math]::Round($devAuc, 4) } else { '-' }
        valid = if ($null -ne $valAuc) { [math]::Round($valAuc, 4) } else { '-' }
    }
} | Format-Table -AutoSize
