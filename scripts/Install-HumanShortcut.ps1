<#
.SYNOPSIS
  Put a "Human" icon on your desktop that starts the program.

.DESCRIPTION
  One shortcut, pointing at Start-Human.ps1 in this folder. Nothing else is
  changed, nothing is installed, and no administrator rights are needed.
  -Remove takes it out again.

  Same shape as Cell\Install-CellAutostart.ps1, which does this for the phone
  mirror's logon entry: a WScript.Shell shortcut written with the COM object
  Windows itself uses. Kept to Windows PowerShell 5.1 for the same reason.

.PARAMETER StartMenu
  Also add it to the Start menu, so it is reachable by typing "Human".

.PARAMETER Distro
  Pin the shortcut to one WSL distribution. Default: the default one.

.PARAMETER Name
  The shortcut's name. Default "Human".

.PARAMETER Remove
  Delete the shortcuts this put there.

.EXAMPLE
  .\Install-HumanShortcut.ps1
  .\Install-HumanShortcut.ps1 -StartMenu -Distro Ubuntu
  .\Install-HumanShortcut.ps1 -Remove
#>
[CmdletBinding()]
param(
    [switch]$StartMenu,
    [string]$Distro = "",
    [string]$Name = "Human",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$repoRoot = Split-Path -Parent $here

$targets = @([IO.Path]::Combine([Environment]::GetFolderPath("Desktop"), "$Name.lnk"))
$startMenuPath = [IO.Path]::Combine(
    [Environment]::GetFolderPath("ApplicationData"), "Microsoft\Windows\Start Menu\Programs", "$Name.lnk")
if ($StartMenu -or $Remove) { $targets += $startMenuPath }

if ($Remove) {
    $gone = 0
    foreach ($path in $targets) {
        if (Test-Path $path) { Remove-Item $path -Force -Confirm:$false; $gone++; Write-Host "Removed $path" }
    }
    if ($gone -eq 0) { Write-Host "No $Name shortcut was installed." }
    return
}

$start = Join-Path $here "Start-Human.ps1"
if (-not (Test-Path $start)) { throw "Start-Human.ps1 is not beside this script: $start" }

$powershell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$icon = Join-Path $repoRoot "assets\human.ico"

$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$start`""
if ($Distro) { $arguments += " -Distro `"$Distro`"" }

$shell = New-Object -ComObject WScript.Shell
foreach ($path in $targets) {
    $parent = Split-Path -Parent $path
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $shortcut = $shell.CreateShortcut($path)
    $shortcut.TargetPath = $powershell
    $shortcut.Arguments = $arguments
    # The checkout, so a relative path the program is given means what the
    # person typing it in that folder would mean.
    $shortcut.WorkingDirectory = $repoRoot
    $shortcut.Description = "Start the Human disease design program and open its panel"
    # Shown normally, not hidden: this window is the program's own output and
    # closing it is how the program is stopped.
    $shortcut.WindowStyle = 1
    if (Test-Path $icon) { $shortcut.IconLocation = "$icon,0" }
    $shortcut.Save()
    Write-Host "Installed: $path"
}

Write-Host ""
Write-Host "Double-click $Name to start the program. The panel opens in your browser."
Write-Host "Remove it with: .\Install-HumanShortcut.ps1 -Remove"
