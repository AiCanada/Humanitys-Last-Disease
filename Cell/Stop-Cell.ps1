<#
.SYNOPSIS
  Stop mirroring: close the tailnet port and stop the Cell relay.

.DESCRIPTION
  A program the phone started runs under the relay, so stopping the relay interrupts it.
  When such a program is running this script stops and says so. -Force goes ahead: the
  program is interrupted the way Ctrl+C would, then the relay ends.
  Tailscale itself stays as it is, unless -TailscaleDown is given.
#>
[CmdletBinding()]
param(
    [int]$Port = 8765,
    [int]$RelayPort = 8766,
    [int]$HttpsPort = 443,
    [switch]$Force,
    [switch]$TailscaleDown
)

$here = $PSScriptRoot
. (Join-Path $here "Cell.Common.ps1")
$pidFile = Join-Path $here "cell.pid"

$ping = Invoke-CellRelay $RelayPort "/cell/ping"
if ($ping -and (@("starting", "running", "stopping") -contains $ping.program) -and -not $Force) {
    Write-Warning "The program is running under Cell. Stopping Cell now would interrupt it."
    Write-Host "Stop the program from the phone first, or run:  .\Stop-Cell.ps1 -Force"
    exit 1
}

$tailscale = Get-CellTailscale
if ($tailscale) {
    $target = "127.0.0.1:$RelayPort"
    $serve = Get-CellServe $tailscale
    $entry = Get-CellServeEntry $serve $Port
    if ($entry -and $entry.TCPForward -eq $target) {
        $null = Invoke-CellNative { & $tailscale serve --tcp=$Port off }
        Write-Host "Tailnet port $Port closed."
    }
    $secure = Get-CellServeEntry $serve $HttpsPort
    if ($secure -and $secure.TCPForward -eq $target -and $secure.TerminateTLS) {
        $null = Invoke-CellNative { & $tailscale serve --tls-terminated-tcp=$HttpsPort off }
        Write-Host "Tailnet https port $HttpsPort closed."
    }
}

if ((Get-CellRelayProcess $pidFile) -or $ping) {
    Stop-CellRelay $RelayPort $pidFile
    Write-Host "Cell relay stopped."
} else {
    if (Test-Path $pidFile) { Remove-Item $pidFile -Force -Confirm:$false }
    Write-Host "Cell relay was not running."
}

if ($TailscaleDown -and $tailscale) {
    & $tailscale down
    Write-Host "Tailscale disconnected."
}
