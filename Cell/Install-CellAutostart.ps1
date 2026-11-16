<#
.SYNOPSIS
  Start Cell by itself when you sign in to Windows, so the phone still connects after a restart.

.DESCRIPTION
  Puts one shortcut, "Cell mirror", in your Startup folder. It runs Start-Cell.ps1 hidden.
  Nothing else is changed and no administrator rights are needed. -Remove takes it out.

  The computer must be signed in for the phone to reach the program: a locked screen is
  fine, a signed-out one is not.

.PARAMETER StartArguments
  Options Start-Cell.ps1 should get at logon, for example "-Https" or "-AnyTailnetDevice".

.EXAMPLE
  .\Install-CellAutostart.ps1
  .\Install-CellAutostart.ps1 -StartArguments "-Https"
  .\Install-CellAutostart.ps1 -Remove
#>
[CmdletBinding()]
param(
    [string]$StartArguments = "",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
. (Join-Path $here "Cell.Common.ps1")
$shortcutPath = Get-CellAutostartPath

if ($Remove) {
    if (Test-Path $shortcutPath) {
        Remove-Item $shortcutPath -Force -Confirm:$false
        Write-Host "Autostart removed."
    } else {
        Write-Host "Autostart was not installed."
    }
    return
}

$start = Join-Path $here "Start-Cell.ps1"
$powershell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $powershell
$shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$start`" -Quiet $StartArguments".Trim()
$shortcut.WorkingDirectory = $here
$shortcut.WindowStyle = 7
$shortcut.Description = "Start the Cell phone mirror for the human program"
$shortcut.Save()
Write-Host "Autostart installed: $shortcutPath"
Write-Host "Cell now starts when you sign in to Windows. Remove with: .\Install-CellAutostart.ps1 -Remove"
