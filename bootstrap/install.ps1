# TrainKit installer for Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/Miracle197/model-toybox/main/bootstrap/install.ps1 | iex"
# Install from a local checkout instead:  $env:TRAINKIT_SOURCE = "C:\path\to\trainkit"; .\bootstrap\install.ps1
$ErrorActionPreference = "Stop"

$Source = if ($env:TRAINKIT_SOURCE) { $env:TRAINKIT_SOURCE } else { "git+https://github.com/Miracle197/model-toybox" }

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "Installing uv (fast Python installer, also provides Python itself)..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# Git is needed to install from a git URL. winget ships with Windows 10/11.
if ($Source -like "git+*" -and -not (Get-Command git -ErrorAction SilentlyContinue)) {
    Say "Installing Git..."
    winget install --id Git.Git -e --silent --accept-package-agreements --accept-source-agreements
    $env:Path = "$env:ProgramFiles\Git\cmd;$env:Path"
}

Say "Installing TrainKit from $Source ..."
uv tool install --python 3.12 --force $Source
uv tool update-shell | Out-Null
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

Say "Checking this computer..."
trainkit doctor

Write-Host ""
Write-Host "TrainKit is installed. Open a NEW terminal, then:"
Write-Host "  trainkit setup     # pick a model; installs PyTorch for your GPU, weights and a test dataset"
Write-Host "  trainkit train ... # train with a live view and web dashboard"
Write-Host "  trainkit play      # try models on your camera, videos or images"
