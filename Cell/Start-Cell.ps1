<#
.SYNOPSIS
  Mirror the `human` program to a phone (Android or iPhone) over Tailscale, with full control.

.DESCRIPTION
  1. Brings Tailscale up when it is stopped.
  2. Starts the Cell relay on 127.0.0.1:<RelayPort>.
  3. Publishes the relay to the tailnet only, with `tailscale serve`, on <Port>.
  The phone then opens http://<this computer's Tailscale address>:<Port>/
  From that page the phone can start the program, use every panel page, read the
  output, and stop the program.

  Running it again is safe. A relay that is running a program is left alone, because
  stopping the relay would interrupt that program. Use -Force to restart it anyway.

.PARAMETER Https
  Also publish https://<this computer>.<tailnet>.ts.net/ . The tailnet must have HTTPS
  certificates enabled (admin console > DNS > Enable HTTPS). Stays on for later runs.

.PARAMETER AnyTailnetDevice
  Admit every device in the tailnet, not only the ones signed in as this computer's user.

.PARAMETER WireGuardAddress
  This computer's address inside a plain WireGuard tunnel (WireGuard app, not Tailscale).
  The relay also listens there. Needs -WireGuardPeer. See README.md for the firewall rule.

.PARAMETER WireGuardPeer
  The phone's address inside that tunnel. Only it is admitted there.

.PARAMETER Program
  The command that starts the program inside WSL, run in the checkout. Default .venv/bin/human

.PARAMETER Distro
  The WSL distribution that holds the checkout's environment. Default: the default one.

.PARAMETER PanelPort
  Use this panel port and skip the search, for a program started on the computer.

.PARAMETER Quiet
  Print nothing but problems, and write a failure to cell.start.log. Used at sign-in
  by Install-CellAutostart.ps1, where there is no window to read.
#>
[CmdletBinding()]
param(
    [int]$Port = 8765,
    [int]$RelayPort = 8766,
    [switch]$Https,
    [int]$HttpsPort = 443,
    [switch]$AnyTailnetDevice,
    [string]$WireGuardAddress = "",
    [string[]]$WireGuardPeer = @(),
    [string]$Program = "",
    [string]$Distro = "",
    [int]$PanelPort = 0,
    [switch]$Force,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
. (Join-Path $here "Cell.Common.ps1")
$pidFile = Join-Path $here "cell.pid"
$logFile = Join-Path $here "cell.log"
$startLog = Join-Path $here "cell.start.log"

function Say([string]$Text, [string]$Color = "") {
    if ($Quiet) { return }
    if ($Color) { Write-Host $Text -ForegroundColor $Color } else { Write-Host $Text }
}

function Quote([string]$Value) {
    # One argument for Start-Process, which joins its list with spaces and adds no quotes.
    if ($Value -match '[\s"]') { return '"' + ($Value -replace '"', '\"') + '"' }
    return $Value
}

function Start-CellMirror {
    $WireGuardPeer = @($WireGuardPeer | Where-Object { $_ })
    if ($WireGuardAddress -and $WireGuardPeer.Count -eq 0) {
        throw "-WireGuardAddress needs -WireGuardPeer: the phone's address inside the tunnel. Without it anyone who can reach that address would get the panel."
    }
    $tailscale = Get-CellTailscale
    if (-not $tailscale) { throw "Tailscale is not installed. Install it from https://tailscale.com/download/windows" }
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $python) { throw "Windows Python was not found on PATH. The relay needs it (standard library only)." }

    # 1. Tailscale up. At sign-in the service can take a while to answer.
    $patience = 60
    if ($Quiet) { $patience = 240 }
    $state = $null
    $deadline = (Get-Date).AddSeconds($patience)
    while (-not $state -and (Get-Date) -lt $deadline) {
        $state = Get-CellTailscaleState $tailscale
        if (-not $state) { Start-Sleep -Seconds 2 }
    }
    if (-not $state) { throw "Tailscale is not answering. Open the Tailscale app, then run this again." }
    if ($state.BackendState -ne "Running") {
        Say "Tailscale is $($state.BackendState). Bringing it up..."
        # The timeout matters at sign-in: without it this waits forever when
        # Tailscale needs a browser login that nobody is there to give.
        $null = Invoke-CellNative { & $tailscale up --timeout 30s }
        $deadline = (Get-Date).AddSeconds(30)
        do {
            $state = Get-CellTailscaleState $tailscale
            if ($state -and $state.BackendState -eq "Running") { break }
            Start-Sleep -Seconds 1
        } while ((Get-Date) -lt $deadline)
        if (-not $state -or $state.BackendState -ne "Running") {
            throw "Tailscale did not start. Open the Tailscale app and sign in, then run this again."
        }
    }
    $addresses = @($state.Self.TailscaleIPs)
    $address = @($addresses | Where-Object { $_ -notmatch ":" })[0]
    $dnsName = "$($state.Self.DNSName)".TrimEnd(".")
    $login = ""
    if ($state.User) { $login = $state.User."$($state.Self.UserID)".LoginName }
    $wanted = "owner"
    if ($AnyTailnetDevice) { $wanted = "tailnet" }

    # 2. Relay
    $keep = $false
    $ping = Invoke-CellRelay $RelayPort "/cell/ping"
    if ($ping -and $ping.cell -and (@("starting", "running", "stopping") -contains $ping.program) -and -not $Force) {
        $keep = $true
        Say "The Cell relay is running the program, so it is left as it is. (-Force restarts it and interrupts the program.)" "Yellow"
    }
    if (-not $keep) {
        Stop-CellRelay $RelayPort $pidFile
        $relay = Join-Path $here "cell_relay.py"
        $arguments = @("-u", (Quote $relay), "--listen-port", "$RelayPort", "--log-file", (Quote $logFile))
        foreach ($name in @($addresses) + @($dnsName, ($dnsName -split '\.')[0])) {
            if ($name) { $arguments += @("--allow-host", (Quote $name)) }
        }
        if ($AnyTailnetDevice) {
            $arguments += @("--tailscale", (Quote $tailscale), "--any-tailnet-device")
        } else {
            $arguments += @("--owner-only", (Quote $tailscale))
        }
        if ($WireGuardAddress) {
            $arguments += @("--bind", $WireGuardAddress)
            foreach ($peer in $WireGuardPeer) { $arguments += @("--allow-peer", $peer) }
        }
        if ($PanelPort -gt 0) { $arguments += @("--panel-port", "$PanelPort") }
        if ($Program) { $arguments += @("--program", (Quote $Program)) }
        if ($Distro) { $arguments += @("--distro", (Quote $Distro)) }
        # No output redirection here on purpose: a redirected child inherits this
        # script's own output handles, and whoever waits on those would wait forever.
        $process = Start-Process -FilePath $python -ArgumentList $arguments -WindowStyle Hidden -PassThru
        Set-Content -Path $pidFile -Value $process.Id -Encoding ascii
        $ping = $null
        $deadline = (Get-Date).AddSeconds(20)
        while (-not $ping -and (Get-Date) -lt $deadline) {
            if ($process.HasExited) { break }
            Start-Sleep -Milliseconds 300
            $ping = Invoke-CellRelay $RelayPort "/cell/ping" "GET" 2
        }
        if (-not $ping) {
            $tail = ""
            if (Test-Path $logFile) { $tail = (Get-Content $logFile -Tail 5) -join "`n" }
            throw "The Cell relay did not start. Is port $RelayPort in use? Last lines of cell.log:`n$tail"
        }
        Say "Cell relay $($ping.cell) started (process $($process.Id))."
    }
    # What the relay that is actually running admits. A kept relay keeps its own rule.
    $gate = "$($ping.gate)"
    if ($keep -and $gate -and $gate -ne $wanted) {
        Write-Warning "The running relay admits '$gate' devices and was not restarted, so the access rule asked for now is not in effect. Run this again with -Force to change it (the program is interrupted)."
    }

    # 3. Publish to the tailnet only (never Funnel). Tailscale Serve always names the
    # connecting device to the relay in a PROXY protocol line: that line is how the
    # relay tells a phone from a program on this computer, in both access modes.
    $target = "127.0.0.1:$RelayPort"
    $serve = Get-CellServe $tailscale
    if ($null -eq $serve) {
        throw "Tailscale Serve did not answer, so nothing was published. Run: tailscale serve status"
    }
    $entry = Get-CellServeEntry $serve $Port
    if ($entry -and $entry.TCPForward -ne $target) {
        throw "Tailscale Serve already uses port $Port for something else ($($entry.TCPForward)). Choose another with -Port."
    }
    if (-not ($entry -and $entry.ProxyProtocol -eq 1 -and -not $entry.TerminateTLS)) {
        $null = Invoke-CellNative { & $tailscale serve --tcp=$Port off }
        & $tailscale serve --bg --proxy-protocol 1 --tcp $Port "tcp://$target" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "tailscale serve failed. Run: tailscale serve status" }
    }

    $httpsUrl = ""
    $secure = Get-CellServeEntry $serve $HttpsPort
    $ours = $secure -and $secure.TCPForward -eq $target -and $secure.TerminateTLS
    if ($Https -or $ours) {
        $domains = @($state.CertDomains | Where-Object { $_ })
        if ($domains.Count -eq 0) {
            Write-Warning "HTTPS is not enabled for this tailnet, so only the http address is published. Enable it at https://login.tailscale.com/admin/dns (Enable HTTPS), then run this again with -Https. Enabling it publishes this computer's tailnet name in the public certificate log."
        } elseif ($secure -and -not $ours) {
            Write-Warning "Tailscale Serve already uses port $HttpsPort for something else, so the https address is not published. Choose another with -HttpsPort."
        } else {
            $published = $true
            if (-not ($ours -and $secure.ProxyProtocol -eq 1)) {
                $null = Invoke-CellNative { & $tailscale serve --tls-terminated-tcp=$HttpsPort off }
                & $tailscale serve --bg --proxy-protocol 1 --tls-terminated-tcp $HttpsPort "tcp://$target" | Out-Null
                $published = $LASTEXITCODE -eq 0
            }
            if ($published) {
                $httpsUrl = "https://$($domains[0])"
                if ($HttpsPort -ne 443) { $httpsUrl += ":$HttpsPort" }
                $httpsUrl += "/"
            } else {
                Write-Warning "The https address could not be published. Run: tailscale serve status"
            }
        }
    }

    if ($Quiet) { return }
    Write-Host ""
    Write-Host "Cell is ready. On the phone, with the Tailscale app connected, open:" -ForegroundColor Green
    Write-Host "    http://${address}:$Port/" -ForegroundColor Cyan
    if ($httpsUrl) { Write-Host "    $httpsUrl   (same page, over https)" -ForegroundColor Cyan }
    if ($WireGuardAddress) {
        Write-Host "    http://${WireGuardAddress}:$RelayPort/   (WireGuard app tunnel, admitted: $($WireGuardPeer -join ', '))" -ForegroundColor Cyan
    }
    Write-Host "  Android: Chrome, then menu > Add to Home screen."
    Write-Host "  iPhone:  Safari, then Share > Add to Home Screen."
    Write-Host ""
    $phones = Get-CellPhones $state
    if ($phones.Count -eq 0) {
        Write-Host "No phone is in this tailnet yet. Install Tailscale on it and sign in as $login." -ForegroundColor Yellow
    } else {
        foreach ($phone in $phones) { Write-Host ("  " + (Get-CellPhoneLine $phone $state)) }
        if (-not ($phones | Where-Object { $_.OS -match '^ios$' })) {
            Write-Host "  No iPhone yet: install Tailscale from the App Store and sign in as $login."
        }
    }
    if ($gate -eq "tailnet") {
        Write-Host "Access: every device in the tailnet." -ForegroundColor Yellow
    } elseif ($gate -eq "owner") {
        Write-Host "Access: only devices signed in to Tailscale as $login."
    } else {
        Write-Host "Access: the relay is not checking devices." -ForegroundColor Red
    }
    if (-not (Test-Path (Get-CellAutostartPath))) {
        Write-Host "Cell does not start by itself after a restart. To make it: .\Install-CellAutostart.ps1"
    }
    Write-Host "The phone can start the program itself. Check with .\Get-CellStatus.ps1, stop with .\Stop-Cell.ps1"
}

try {
    Start-CellMirror
    if (Test-Path $startLog) { Remove-Item -LiteralPath $startLog -Force -Confirm:$false -ErrorAction SilentlyContinue }
} catch {
    # At sign-in this runs hidden. The reason is kept where Get-CellStatus.ps1 shows it.
    $line = "{0} Start-Cell failed: {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $_.Exception.Message
    try { Set-Content -Path $startLog -Value $line -Encoding utf8 } catch { }
    if ($Quiet) { exit 1 }
    throw
}
