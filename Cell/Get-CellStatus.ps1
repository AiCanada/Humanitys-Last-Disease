<#
.SYNOPSIS
  Show whether the phone mirror is working: Tailscale, the relay, the tailnet port,
  the program, the panel, and the phones.
#>
[CmdletBinding()]
param(
    [int]$Port = 8765,
    [int]$RelayPort = 8766,
    [int]$HttpsPort = 443
)

$here = $PSScriptRoot
. (Join-Path $here "Cell.Common.ps1")

function Show([string]$Name, [bool]$Ok, [string]$Detail, [switch]$Neutral) {
    $mark = "FAIL"; $color = "Red"
    if ($Ok) { $mark = " OK "; $color = "Green" }
    if ($Neutral -and -not $Ok) { $mark = " -- "; $color = "DarkGray" }
    Write-Host "[$mark] " -ForegroundColor $color -NoNewline
    Write-Host ("{0,-14} {1}" -f $Name, $Detail)
}

$tailscale = Get-CellTailscale
if (-not $tailscale) { Show "Tailscale" $false "not installed"; exit 1 }
$state = Get-CellTailscaleState $tailscale
$up = $state -and $state.BackendState -eq "Running"
$address = ""
if ($state) { $address = @($state.Self.TailscaleIPs | Where-Object { $_ -notmatch ":" })[0] }
$detail = "not answering"
if ($state) { $detail = "$($state.BackendState)  $address" }
Show "Tailscale" $up $detail

# Whether the relay is up is asked with the lightest request it has. The fuller
# status is asked separately and given longer, so a busy computer is not reported
# as a relay that is not running.
$ping = Invoke-CellRelay $RelayPort "/cell/ping"
$relayOk = [bool]($ping -and $ping.cell)
$detail = "not running - run .\Start-Cell.ps1"
if ($relayOk) { $detail = "version $($ping.cell) on 127.0.0.1:$RelayPort" }
Show "Relay" $relayOk $detail
$status = $null
if ($relayOk) { $status = Invoke-CellRelay $RelayPort "/cell/status" "GET" 15 }

# Who is admitted is what the relay enforces, not what the published port looks like.
if ($relayOk) {
    $gate = "$($ping.gate)"
    if ($gate -eq "owner") { Show "Access" $true "only your own Tailscale devices" }
    elseif ($gate -eq "tailnet") { Show "Access" $true "every device in the tailnet (-AnyTailnetDevice)" }
    else { Show "Access" $false "the relay is not checking devices - run .\Start-Cell.ps1 -Force" }
}

$target = "127.0.0.1:$RelayPort"
$serve = Get-CellServe $tailscale
$entry = Get-CellServeEntry $serve $Port
$served = [bool]($entry -and $entry.TCPForward -eq $target)
$detail = "port $Port is not published - run .\Start-Cell.ps1"
if ($served) {
    $detail = "port $Port, tailnet only"
    if (-not $entry.ProxyProtocol) {
        # Without the device line the relay cannot tell a phone from this computer.
        $served = $false
        $detail = "port $Port is published without the device line - run .\Start-Cell.ps1"
    }
}
Show "Tailnet port" $served $detail
$secure = Get-CellServeEntry $serve $HttpsPort
$httpsUrl = ""
if ($secure -and $secure.TCPForward -eq $target -and $secure.TerminateTLS) {
    $httpsUrl = "https://$($secure.TerminateTLS)"
    if ($HttpsPort -ne 443) { $httpsUrl += ":$HttpsPort" }
    $httpsUrl += "/"
    Show "Tailnet https" $true "port $HttpsPort"
}

if ($relayOk -and -not $status) {
    Show "Program" $false "the relay did not report in time - try again" -Neutral
} elseif ($relayOk) {
    $program = $status.program
    $running = @("starting", "running", "stopping") -contains $program.state
    $detail = "not running - the phone can start it"
    if ($running) { $detail = "$($program.state), started from the phone" }
    elseif ($status.panel.open) { $running = $true; $detail = "running, started on this computer" }
    elseif ($program.state -eq "exited") { $detail = "finished with exit code $($program.code)" }
    Show "Program" $running $detail -Neutral
    $detail = "not open"
    if ($status.panel.open) { $detail = "open, on step $($status.panel.step)" }
    Show "Panel" ([bool]$status.panel.open) $detail -Neutral
}

$phones = Get-CellPhones $state
if ($phones.Count -eq 0) {
    Show "Phone" $false "none in this tailnet - install Tailscale on the phone, same account"
} else {
    foreach ($phone in $phones) { Show "Phone" ([bool]$phone.Online) (Get-CellPhoneLine $phone $state) -Neutral }
}

$autostart = Test-Path (Get-CellAutostartPath)
$detail = "off - Cell must be started by hand after a restart (.\Install-CellAutostart.ps1)"
if ($autostart) { $detail = "on - Cell starts when you sign in to Windows" }
Show "Autostart" $autostart $detail -Neutral

if ($up -and $relayOk -and $served) {
    Write-Host ""
    Write-Host "Phone address: http://${address}:$Port/" -ForegroundColor Cyan
    if ($httpsUrl) { Write-Host "               $httpsUrl" -ForegroundColor Cyan }
}
$startLog = Join-Path $here "cell.start.log"
if (Test-Path $startLog) {
    # Written when Start-Cell.ps1 failed, which at sign-in happens with no window to read.
    Write-Host ""
    Write-Host "The last start failed:" -ForegroundColor Red
    Get-Content $startLog -Tail 3 | ForEach-Object { Write-Host "  $_" }
}
$log = Join-Path $here "cell.log"
if (Test-Path $log) {
    Write-Host ""
    Write-Host "Recent relay log:"
    Get-Content $log -Tail 6 | ForEach-Object { Write-Host "  $_" }
}
if (-not ($up -and $relayOk -and $served)) { exit 1 }
