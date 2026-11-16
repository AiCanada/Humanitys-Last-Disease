<#
.SYNOPSIS
  Install the Human program on Windows: WSL, Ubuntu, the environment, and a desktop icon.

.DESCRIPTION
  One command for a new computer. It does four things and stops at the first
  one it cannot do, saying what to do about it:

    1. Makes sure WSL and an Ubuntu distribution are there. Installing WSL needs
       administrator rights and a restart; this detects that case and tells you
       rather than half-doing it.
    2. Runs scripts/install_human.sh inside that distribution, which adds the
       few system packages the build needs and then hands over to setup.sh.
       Everything that builds the environment lives in setup.sh; none of it is
       repeated here.
    3. Puts a "Human" icon on your desktop.
    4. Prints how to start it.

  Safe to run again. Each step checks before acting, so a second run after a
  restart picks up where the first stopped.

  The download is large -- uv, Python 3.12 and the CUDA build of torch, plus the
  model weights unless you skip them -- so expect several GB and a long wait.

.PARAMETER Distro
  The WSL distribution to use or install. Default Ubuntu.

.PARAMETER Repository
  Clone the program from here instead of installing the checkout this script
  sits in. Use this on a computer that has no checkout yet.

.PARAMETER Target
  Where to clone, as Linux names it. Default ~/human/RFD4-Proteina.

.PARAMETER SkipWeights
  Install the environment without downloading the model weights. The panel runs;
  generation needs the weights later.

.PARAMETER SkipSearchEnvs
  Do not ask about the optional AlphaFold 2 / AlphaFold 3 search environments.

.PARAMETER SkipShortcut
  Do not create the desktop icon.

.PARAMETER WhatIf
  Print what each step would do and change nothing.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\Install-Human.ps1
  .\scripts\Install-Human.ps1 -SkipWeights -SkipSearchEnvs
  .\scripts\Install-Human.ps1 -Repository https://github.com/AiCanada/RFD4-Proteina.git
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$Distro = "Ubuntu",
    [string]$Repository = "",
    [string]$Target = "",
    [switch]$SkipWeights,
    [switch]$SkipSearchEnvs,
    [switch]$SkipShortcut
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$repoRoot = Split-Path -Parent $here
$step = 0

function Say([string]$Text, [string]$Colour = "Cyan") {
    Write-Host ""
    Write-Host $Text -ForegroundColor $Colour
}
function Step([string]$Text) {
    $script:step++
    Write-Host ""
    Write-Host ("[{0}] {1}" -f $script:step, $Text) -ForegroundColor Cyan
}
function Get-WslPath([string]$WindowsPath) {
    # Matches Cell\cell_relay.py default_repo, so every part of this project
    # names the checkout the same way inside WSL.
    $full = [System.IO.Path]::GetFullPath($WindowsPath)
    if ([System.IO.Path]::GetPathRoot($full) -match '^([A-Za-z]):\\$') {
        return "/mnt/" + $Matches[1].ToLower() + "/" + ($full.Substring(3) -replace '\\', '/')
    }
    return $full -replace '\\', '/'
}
function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    return (New-Object Security.Principal.WindowsPrincipal $identity).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

Write-Host "Installing the Human program." -ForegroundColor Green
Write-Host "Checkout: $repoRoot"
Write-Host "WSL distribution: $Distro"

# ------------------------------------------------------------------ 1. WSL
Step "Checking WSL and $Distro"
$wsl = Join-Path $env:SystemRoot "System32\wsl.exe"
if (-not (Test-Path $wsl)) {
    throw @"
WSL is not on this computer.

Open PowerShell as Administrator and run:
  wsl --install -d $Distro

Restart when it asks, finish the Ubuntu user setup it opens, then run this
script again. It will carry on from here.
"@
}

# `wsl -l -q` prints UTF-16 with a BOM, which is why the names are cleaned
# rather than compared as they arrive.
$installed = @()
$raw = & $wsl -l -q 2>$null
if ($LASTEXITCODE -eq 0 -and $raw) {
    $installed = @($raw | ForEach-Object { $_ -replace "`0", "" } | ForEach-Object { $_.Trim([char]0xFEFF, ' ', "`r") } |
        Where-Object { $_ })
}

if ($installed -contains $Distro) {
    Write-Host "  $Distro is installed." -ForegroundColor Green
} else {
    if ($installed.Count -gt 0) {
        Write-Host ("  Installed distributions: {0}" -f ($installed -join ", "))
    }
    if (-not (Test-Admin)) {
        throw @"
$Distro is not installed, and installing a distribution needs administrator rights.

Open PowerShell as Administrator and run:
  wsl --install -d $Distro

Finish the Ubuntu user setup it opens (it asks for a username and password of
your choosing -- those are Ubuntu's, not your Windows ones), then run this
script again.

If you already have a different distribution, use it instead:
  .\scripts\Install-Human.ps1 -Distro <name>
"@
    }
    if ($PSCmdlet.ShouldProcess($Distro, "wsl --install -d")) {
        Say "Installing $Distro. Windows may ask to restart; if it does, restart and run this again."
        & $wsl --install -d $Distro
        if ($LASTEXITCODE -ne 0) { throw "wsl --install -d $Distro failed with code $LASTEXITCODE." }
        throw @"
$Distro was installed. Finish its first-run user setup in the window that opened,
then run this script again to build the environment.
"@
    }
}

# A distribution can be installed and still have no user set up, in which case
# every command fails in a way that does not mention the real cause.
if ($PSCmdlet.ShouldProcess($Distro, "check that it runs")) {
    $probe = & $wsl -d $Distro -- bash -lc "echo ready" 2>$null
    if ($LASTEXITCODE -ne 0 -or "$probe".Trim() -ne "ready") {
        throw @"
$Distro is installed but does not run a command yet. Open it once from the Start
menu and finish the user setup it asks for, then run this script again.
"@
    }
    Write-Host "  $Distro runs commands." -ForegroundColor Green
}

# ------------------------------------------------- 2. environment, in Linux
Step "Building the environment inside $Distro"
$installer = "$(Get-WslPath $repoRoot)/scripts/install_human.sh"
$inner = @()
if ($Repository) { $inner += @("--repository", $Repository) }
if ($Target) { $inner += @("--target", $Target) }
$setupFlags = @()
if ($SkipWeights) { $setupFlags += "--skip-weights" }
if ($SkipSearchEnvs) { $setupFlags += "--skip-search-envs" }
if ($setupFlags.Count -gt 0) { $inner += @("--") + $setupFlags }

$command = "bash '$installer' " + (($inner | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }) -join " ")
Write-Host "  $command" -ForegroundColor DarkGray
Write-Host "  This downloads several GB and takes a while. It is safe to run again if interrupted."
if ($PSCmdlet.ShouldProcess($Distro, "run install_human.sh")) {
    & $wsl -d $Distro -- bash -lc $command
    if ($LASTEXITCODE -ne 0) {
        throw "The Linux installer exited with code $LASTEXITCODE. Scroll up for the reason; nothing after this ran."
    }
}

# ------------------------------------------------------------- 3. the icon
if ($SkipShortcut) {
    Step "Skipping the desktop icon (-SkipShortcut)"
} else {
    Step "Putting a Human icon on your desktop"
    if ($PSCmdlet.ShouldProcess("Desktop", "create the Human shortcut")) {
        & (Join-Path $here "Install-HumanShortcut.ps1") -Distro $Distro -StartMenu
    }
}

# ------------------------------------------------------------------ 4. done
Say "Installed." "Green"
Write-Host "Start it by double-clicking Human on your desktop, or with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$here\Start-Human.ps1`""
if ($SkipWeights) {
    Write-Host ""
    Write-Host "The model weights were skipped, so the panel runs but generation cannot." -ForegroundColor Yellow
    Write-Host "Add them later with: wsl -d $Distro -- bash -lc `"cd '$(Get-WslPath $repoRoot)' && bash setup.sh --native`""
}
