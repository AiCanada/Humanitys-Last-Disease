<#
.SYNOPSIS
  Start the `human` program and let it open its panel in this computer's browser.

.DESCRIPTION
  The program runs inside WSL, where the environment is, and it opens the panel
  itself: `ui._open_browser` notices it is in WSL and asks Windows to open its
  own window, so nothing here needs to find the address or launch a browser.
  That matters -- the panel binds to port 0, so its port is different on every
  start, and a second opinion about which port to open would be wrong half the
  time.

  This is the thing the desktop shortcut runs. Closing the window stops the
  program, which is also how the panel's own Close page ends it.

  For reaching the panel from a phone instead, use Cell\Start-Cell.ps1. That
  publishes it over Tailscale and is a different job from this one.

.PARAMETER Distro
  The WSL distribution holding the checkout's environment. Default: the default one.

.PARAMETER Arguments
  Extra arguments for the program, after --ui. For example -Arguments "--disease asthma".

.PARAMETER NoUi
  Run the program without --ui, as a terminal session in this window.

.EXAMPLE
  .\Start-Human.ps1
  .\Start-Human.ps1 -Distro Ubuntu
  .\Start-Human.ps1 -Arguments "--catalog my_catalog.json"
#>
[CmdletBinding()]
param(
    [string]$Distro = "",
    [string]$Arguments = "",
    [switch]$NoUi
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$repoRoot = Split-Path -Parent $here

function Get-WslPath([string]$WindowsPath) {
    # The same conversion Cell\cell_relay.py's default_repo does, so both agree
    # about how this checkout is named inside WSL.
    $full = [System.IO.Path]::GetFullPath($WindowsPath)
    $drive = [System.IO.Path]::GetPathRoot($full)
    if ($drive -match '^([A-Za-z]):\\$') {
        $rest = $full.Substring(3) -replace '\\', '/'
        return "/mnt/" + $Matches[1].ToLower() + "/" + $rest
    }
    return $full -replace '\\', '/'
}

$wsl = Join-Path $env:SystemRoot "System32\wsl.exe"
if (-not (Test-Path $wsl)) {
    throw "WSL is not installed on this computer. Run scripts\Install-Human.ps1 first."
}

$distroArgs = @()
if ($Distro) { $distroArgs = @("-d", $Distro) }

$repoInWsl = Get-WslPath $repoRoot
$program = ".venv/bin/human"

# Checked before launching, because a missing environment otherwise shows up as
# a shell error in a window that closes immediately, and the person is left with
# a shortcut that appears to do nothing at all.
$probe = & $wsl @distroArgs -- bash -lc "test -x '$repoInWsl/$program' && echo present" 2>$null
if ($LASTEXITCODE -ne 0 -or "$probe".Trim() -ne "present") {
    $where = if ($Distro) { "the $Distro distribution" } else { "the default WSL distribution" }
    throw @"
The program is not installed in $where.

Looked for: $repoInWsl/$program

Install it with:
  powershell -ExecutionPolicy Bypass -File "$here\Install-Human.ps1"
"@
}

$flags = @()
if (-not $NoUi) { $flags += "--ui" }
$tail = ($flags -join " ")
if ($Arguments) { $tail = ($tail + " " + $Arguments).Trim() }

Write-Host "Starting the human program in WSL." -ForegroundColor Cyan
Write-Host "  checkout: $repoInWsl"
if (-not $NoUi) {
    Write-Host "  The panel opens in your browser by itself. Its address is printed below."
}
Write-Host "  Close this window to stop the program." -ForegroundColor DarkGray
Write-Host ""

# Run in the foreground so this window is the program's window: its output is
# the progress, and closing it ends the program. No output redirection, so the
# panel's printed address and every log line land here as they happen.
& $wsl @distroArgs -- bash -lc "cd '$repoInWsl' && exec $program $tail"
$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "The program finished." -ForegroundColor Green
} else {
    Write-Host "The program exited with code $code." -ForegroundColor Yellow
    Write-Host "Scroll up for the reason. The log is at ~/.config/Humanitys-Last-Disease/ and the run's own directory."
}
# Held open so a failure is readable when this was started from a shortcut,
# where the window would otherwise close on the error.
if ($Host.Name -eq "ConsoleHost" -and -not $env:HUMAN_NO_PAUSE) {
    Write-Host ""
    Write-Host "Press Enter to close this window." -ForegroundColor DarkGray
    [void](Read-Host)
}
exit $code
