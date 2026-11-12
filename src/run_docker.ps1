param(
    [ValidateSet('check', 'audit', 'test', 'lint', 'typecheck', 'format', 'freeze', 'weights', 'train', 'evaluate', 'predict', 'multiview', 'benchmark', 'profile', 'latency', 'server-check')]
    [string]$Action = 'check',
    [string]$Config = 'configs/local_image_mean.json',
    [string]$Checkpoint = '',
    [string]$Split = '',
    [string]$Output = '',
    [switch]$IncludeSubject,
    [switch]$Holdout,
    [int]$Batches = 12,
    [int]$Warmup = 2,
    [string]$Workers = '0,2,4',
    [int]$Images = 64
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $projectRoot
$competitionImage = 'cxr-phase1:official'
$dockerArguments = @(
    'run', '--rm', '--gpus', 'all', '--shm-size=2g',
    '--mount', "type=bind,source=$projectRoot,target=/workspace",
    '--mount', "type=bind,source=$repoRoot,target=/repo,readonly",
    '--mount', "type=bind,source=$projectRoot\cxr_first,target=/workspace/cxr_first,readonly",
    '--workdir', '/workspace',
    '--env', 'PYTHONPATH=/workspace/src',
    '--env', 'TORCH_HOME=/workspace/.cache/torch',
    '--env', 'PYTHONUNBUFFERED=1',
    '--env', 'OMP_NUM_THREADS=4',
    $competitionImage
)

if ($Action -eq 'check') {
    $pythonArguments = @('python', '-c', @'
import json, platform, torch, torchvision
from pathlib import Path
assert platform.python_version_tuple()[:2] == ('3', '12'), 'Expected Python 3.12'
assert torch.__version__.split('+')[0] == '2.10.0', 'Expected PyTorch 2.10.0'
assert torchvision.__version__.split('+')[0] == '0.25.0', 'Expected torchvision 0.25.0'
assert torch.version.cuda == '12.8', 'Expected CUDA runtime 12.8'
assert torch.backends.cudnn.version() // 10000 == 9, 'Expected cuDNN 9'
assert torch.cuda.is_available(), 'CUDA unavailable inside container'
x = torch.randn(32, 32, device='cuda')
assert torch.isfinite(x @ x).all()
metadata = {'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu': torch.cuda.get_device_name(0)}
Path('.work/phase1').mkdir(parents=True, exist_ok=True)
Path('.work/phase1/container-environment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print(json.dumps(metadata, indent=2))
'@)
} elseif ($Action -eq 'test') {
    $pythonArguments = @('python', '-m', 'pytest', '-q', '--basetemp', '/workspace/.work/pytest-container')
} elseif ($Action -eq 'lint') {
    $pythonArguments = @('python', '-m', 'ruff', 'check', 'src')
} elseif ($Action -eq 'typecheck') {
    $pythonArguments = @('python', '-m', 'mypy', 'src/cxr')
} elseif ($Action -eq 'format') {
    $pythonArguments = @('python', '-m', 'ruff', 'format', 'src')
} elseif ($Action -eq 'weights') {
    $pythonArguments = @('python', 'src/download_weights.py')
} elseif ($Action -eq 'profile' -or $Action -eq 'latency') {
    if (!$Checkpoint -or !$Output) { throw 'Provide -Checkpoint and -Output for diagnostics.' }
    if ($Action -eq 'profile') {
        $pythonArguments = @('python', '-m', 'cxr.profiling', '--config', $Config,
            '--checkpoint', $Checkpoint, '--batches', "$Batches", '--warmup', "$Warmup",
            '--workers', $Workers, '--output-dir', $Output)
    } else {
        $pythonArguments = @('python', '-m', 'cxr.latency', '--config', $Config,
            '--checkpoint', $Checkpoint, '--images', "$Images", '--warmup', "$Warmup",
            '--output-json', $Output)
    }
} elseif ($Action -eq 'server-check') {
    $pythonArguments = @('bash', 'src/server/run.sh', '--mode', 'native', 'check')
    $dockerArguments = $dockerArguments[0..($dockerArguments.Length - 2)] + @(
        '--env', 'CXR_PYTHON=/opt/cxr-venv/bin/python', $competitionImage)
} elseif ($Action -eq 'freeze') {
    $pythonArguments = @('python', '-c', @'
from importlib.metadata import distributions, version
from pathlib import Path
names = sorted({item.metadata['Name'] for item in distributions() if item.metadata['Name']}, key=str.lower)
snapshot = ''.join(f'{name}=={version(name)}\n' for name in names)
Path('docs/container-packages.txt').write_text(snapshot, encoding='utf-8')
print(snapshot)
'@)
} else {
    $pythonArguments = @('python', '-m', 'cxr', $Action, '--config', $Config.Replace('\', '/'))
    if ($Checkpoint) {
        $checkpointOption = if ($Action -eq 'train') { '--resume' } else { '--checkpoint' }
        $pythonArguments += @($checkpointOption, $Checkpoint.Replace('\', '/'))
    }
    if ($Split) { $pythonArguments += @('--split', $Split) }
    if ($Output) { $pythonArguments += @('--output', $Output.Replace('\', '/')) }
    if ($Holdout) {
        $pythonArguments += @('--holdout-manifest', '/repo/docs/splits/longitudinal_200.json')
    }
    if ($IncludeSubject) { $pythonArguments += '--include-subject' }
}

& docker @dockerArguments @pythonArguments
exit $LASTEXITCODE
