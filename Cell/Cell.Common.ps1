# Shared by Start-Cell.ps1, Stop-Cell.ps1, Get-CellStatus.ps1 and Install-CellAutostart.ps1.
# Written for Windows PowerShell 5.1.

function Get-CellTailscale {
    $command = Get-Command tailscale -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $default = Join-Path $env:ProgramFiles "Tailscale\tailscale.exe"
    if (Test-Path $default) { return $default }
    return $null
}

function Invoke-CellNative([scriptblock]$Block) {
    # Run a native command with its error text discarded, and return its exit code.
    # In PowerShell 5.1 a native command's stderr becomes a terminating error under
    # ErrorActionPreference Stop, so the preference is relaxed for the call.
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Block 2>$null | Out-Null } finally { $ErrorActionPreference = $old }
    return $LASTEXITCODE
}

function Get-CellNativeJson([scriptblock]$Block) {
    # The JSON a native command prints, as an object, or $null when it prints none.
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $text = (& $Block 2>$null | Out-String) } finally { $ErrorActionPreference = $old }
    if (-not $text -or -not $text.Trim()) { return $null }
    try { return $text | ConvertFrom-Json } catch { return $null }
}

function Get-CellTailscaleState([string]$Exe) {
    return Get-CellNativeJson { & $Exe status --json }
}

function Get-CellServe([string]$Exe) {
    return Get-CellNativeJson { & $Exe serve status --json }
}

function Get-CellServeEntry($Serve, [int]$Port) {
    if ($Serve -and $Serve.TCP) { return $Serve.TCP."$Port" }
    return $null
}

function Get-CellPhones($State) {
    if (-not $State -or -not $State.Peer) { return @() }
    return @($State.Peer.PSObject.Properties.Value | Where-Object { $_.OS -match '^(android|ios)$' })
}

function Get-CellPhoneLine($Phone, $State) {
    $kind = "Android"
    if ($Phone.OS -match '^ios$') { $kind = "iPhone/iPad" }
    $line = "{0} ({1}) - " -f $Phone.HostName, $kind
    if ($Phone.Online) { $line += "online" } else { $line += "offline" }
    if ($State -and $Phone.UserID -ne $State.Self.UserID) { $line += " - another user's device" }
    return $line
}

function Invoke-CellRelay([int]$RelayPort, [string]$Path, [string]$Method = "GET", [int]$TimeoutSec = 4) {
    # The relay's answer as an object, or $null when it does not answer.
    # The Origin header is what the relay requires of a POST.
    try {
        $headers = @{ Origin = "http://127.0.0.1:$RelayPort" }
        return Invoke-RestMethod -Uri "http://127.0.0.1:$RelayPort$Path" -Method $Method -Headers $headers `
            -TimeoutSec $TimeoutSec -UseBasicParsing
    } catch {
        return $null
    }
}

function Get-CellRelayProcess([string]$PidFile) {
    # The process the pid file names, only when it really is the relay. A pid can be
    # reused by another program after a restart, and that program must not be stopped.
    if (-not (Test-Path $PidFile)) { return $null }
    $id = Get-Content $PidFile | Select-Object -First 1
    if (-not ($id -match '^\d+$')) { return $null }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$id" -ErrorAction SilentlyContinue
    if ($process -and $process.CommandLine -and $process.CommandLine -match 'cell_relay\.py') { return $process }
    return $null
}

function Stop-CellRelay([int]$RelayPort, [string]$PidFile) {
    # Ask the relay to stop the program it runs, then itself. Whatever is left is ended.
    $asked = Invoke-CellRelay $RelayPort "/cell/shutdown" "POST" 10
    if ($asked) {
        $deadline = (Get-Date).AddSeconds(45)
        while ((Get-Date) -lt $deadline -and (Get-CellRelayProcess $PidFile)) { Start-Sleep -Milliseconds 250 }
    }
    $left = Get-CellRelayProcess $PidFile
    if ($left) { Stop-Process -Id $left.ProcessId -Force -Confirm:$false -ErrorAction SilentlyContinue }
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force -Confirm:$false }
}

function Get-CellAutostartPath {
    return Join-Path ([Environment]::GetFolderPath("Startup")) "Cell mirror.lnk"
}
