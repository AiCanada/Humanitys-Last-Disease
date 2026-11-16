# Paste a Grok Bot swarm task into the Grok Bot composer and send it.
# grokBotDesktop.js spawns this with the task on stdin (never argv).
# Exit 0 only after a pixel on Grok Bot is topmost, Ctrl+V, and Enter.
# Stay HWND_TOPMOST until after Enter so the click is not eaten by Chrome.
# Click with SetCursorPos (same DPI space as GetWindowRect). Do not mix
# SendInput ABSOLUTE with GetSystemMetrics: that misses at 150% DPI.
# The process has to be per-monitor DPI aware before the clipboard assembly
# loads. Otherwise SetCursorPos clamps a physical composer point to the
# virtual screen and the click lands on the taskbar.
# Spawn without CREATE_NO_WINDOW: a hidden console cannot inject keys.

$ErrorActionPreference = 'Stop'

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class GrokBotDpi {
  [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr value);
}
"@
[void][GrokBotDpi]::SetProcessDpiAwarenessContext([IntPtr](-4))

$MaxChars = 16384
$WaitMs = 12000
$PollMs = 250
$AfterTopMostMs = 450
$AfterClickMs = 350
$AfterPasteMs = 800

$in = New-Object System.IO.StreamReader(
  [Console]::OpenStandardInput(),
  (New-Object System.Text.UTF8Encoding $false),
  $false,
  4096
)
try {
  $text = $in.ReadToEnd()
} finally {
  $in.Dispose()
}
if ([string]::IsNullOrEmpty($text)) {
  [Console]::Error.WriteLine('no task')
  exit 1
}
if ($text.Length -gt $MaxChars) {
  [Console]::Error.WriteLine('task too long')
  exit 1
}

try {
  Add-Type -AssemblyName System.Windows.Forms
} catch {
  [Console]::Error.WriteLine('clipboard')
  exit 4
}

# The task is NOT put on the clipboard here. The clipboard is global: writing
# it before knowing there is a Grok Bot window to paste into destroyed the
# user's own clipboard even on the runs that then exited 2 having pasted
# nothing. It is written below, after every check has passed, and the previous
# contents are put back in the finally.

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class GrokBotPasteWin {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int X, int Y);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
  [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
  [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint idAttach, uint idAttachTo, bool fAttach);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern void SwitchToThisWindow(IntPtr hWnd, bool fAltTab);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
  [DllImport("user32.dll")] public static extern IntPtr WindowFromPoint(POINT p);
  [DllImport("user32.dll")] public static extern IntPtr GetAncestor(IntPtr hWnd, uint gaFlags);
  [DllImport("user32.dll")] public static extern bool AllowSetForegroundWindow(int dwProcessId);
  [DllImport("user32.dll")] public static extern bool LockSetForegroundWindow(int uLockCode);
  [DllImport("user32.dll")] public static extern bool SystemParametersInfo(uint uiAction, uint uiParam, IntPtr pvParam, uint fWinIni);
  [DllImport("user32.dll", EntryPoint = "SystemParametersInfo")]
  public static extern bool SystemParametersInfoBool(uint uiAction, uint uiParam, ref bool pvParam, uint fWinIni);
  [DllImport("user32.dll", SetLastError = true)]
  public static extern uint SendInput(uint nInputs, INPUT[] pInputs, int cbSize);
  [StructLayout(LayoutKind.Sequential)]
  public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }
  [StructLayout(LayoutKind.Sequential)]
  public struct POINT { public int X; public int Y; }
  [StructLayout(LayoutKind.Sequential)]
  public struct INPUT { public uint type; public InputUnion U; }
  [StructLayout(LayoutKind.Explicit)]
  public struct InputUnion {
    [FieldOffset(0)] public MOUSEINPUT mi;
    [FieldOffset(0)] public KEYBDINPUT ki;
  }
  [StructLayout(LayoutKind.Sequential)]
  public struct MOUSEINPUT {
    public int dx; public int dy; public uint mouseData; public uint dwFlags; public uint time; public UIntPtr dwExtraInfo;
  }
  [StructLayout(LayoutKind.Sequential)]
  public struct KEYBDINPUT {
    public ushort wVk; public ushort wScan; public uint dwFlags; public uint time; public UIntPtr dwExtraInfo;
  }
  public static readonly IntPtr HWND_TOPMOST = new IntPtr(-1);
  public static readonly IntPtr HWND_NOTOPMOST = new IntPtr(-2);
  public const int SW_RESTORE = 9;
  public const int SW_SHOW = 5;
  public const int SW_SHOWNORMAL = 1;
  public const uint SWP_NOSIZE = 0x0001;
  public const uint SWP_NOMOVE = 0x0002;
  public const uint SWP_SHOWWINDOW = 0x0040;
  public const uint GA_ROOT = 2;
  public const uint INPUT_MOUSE = 0;
  public const uint INPUT_KEYBOARD = 1;
  public const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
  public const uint MOUSEEVENTF_LEFTUP = 0x0004;
  public const uint KEYEVENTF_KEYUP = 0x0002;
  public const uint SPI_SETFOREGROUNDLOCKTIMEOUT = 0x2001;
  public const int LSFW_UNLOCK = 2;
  public const int ASFW_ANY = -1;
  public const ushort VK_MENU = 0x12;
  public const ushort VK_CONTROL = 0x11;
  public const ushort VK_V = 0x56;
  public const ushort VK_RETURN = 0x0D;
  static INPUT Key(ushort vk, bool up) {
    INPUT i = new INPUT();
    i.type = INPUT_KEYBOARD;
    i.U.ki.wVk = vk;
    i.U.ki.wScan = 0;
    i.U.ki.dwFlags = up ? KEYEVENTF_KEYUP : 0;
    return i;
  }
  public static void Send(INPUT[] inputs) {
    uint n = SendInput((uint)inputs.Length, inputs, Marshal.SizeOf(typeof(INPUT)));
    if (n != (uint)inputs.Length) throw new InvalidOperationException("SendInput");
  }
  public static void Tap(ushort vk) {
    Send(new INPUT[] { Key(vk, false), Key(vk, true) });
  }
  public static void Chord(ushort hold, ushort tap) {
    Send(new INPUT[] {
      Key(hold, false), Key(tap, false), Key(tap, true), Key(hold, true)
    });
  }
  public static void ClickAt(int x, int y) {
    SetCursorPos(x, y);
    INPUT down = new INPUT();
    down.type = INPUT_MOUSE;
    down.U.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    INPUT up = new INPUT();
    up.type = INPUT_MOUSE;
    up.U.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    Send(new INPUT[] { down, up });
  }
  public static bool PointBelongsTo(IntPtr hWnd, int x, int y) {
    POINT p;
    p.X = x;
    p.Y = y;
    IntPtr hit = WindowFromPoint(p);
    if (hit == IntPtr.Zero || hWnd == IntPtr.Zero) return false;
    if (hit == hWnd) return true;
    IntPtr root = GetAncestor(hit, GA_ROOT);
    return root == hWnd;
  }
  public static void SetTopMost(IntPtr hWnd, bool topMost) {
    SetWindowPos(
      hWnd,
      topMost ? HWND_TOPMOST : HWND_NOTOPMOST,
      0, 0, 0, 0,
      SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
    );
  }
  public static void ShowRestored(IntPtr hWnd) {
    ShowWindow(hWnd, SW_RESTORE);
    ShowWindow(hWnd, SW_SHOWNORMAL);
    ShowWindow(hWnd, SW_SHOW);
    BringWindowToTop(hWnd);
    SwitchToThisWindow(hWnd, true);
  }
  public static long ForegroundId() {
    return GetForegroundWindow().ToInt64();
  }
}
"@

function Get-GrokBotWindow {
  Get-Process -Name 'Grok Bot' -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowHandle -ne [IntPtr]::Zero } |
    Sort-Object {
      $rect = New-Object GrokBotPasteWin+RECT
      [void][GrokBotPasteWin]::GetWindowRect($_.MainWindowHandle, [ref]$rect)
      [Math]::Abs($rect.Right - $rect.Left) * [Math]::Abs($rect.Bottom - $rect.Top)
    } -Descending |
    Select-Object -First 1
}

function Test-Foreground([IntPtr]$hwnd) {
  return [GrokBotPasteWin]::ForegroundId() -eq $hwnd.ToInt64()
}

# The composer is sand-kit-message-input-frame. Grok only publishes that tree
# while a screen reader is announced, and the tree stays after the flag is put
# back, so the flag is not left on. A miss falls through to the coordinate clicks.
function Get-ComposerPoint([IntPtr]$hwnd) {
  Add-Type -AssemblyName UIAutomationClient -ErrorAction SilentlyContinue
  Add-Type -AssemblyName UIAutomationTypes -ErrorAction SilentlyContinue
  $root = [System.Windows.Automation.AutomationElement]::FromHandle($hwnd)
  $all = $root.FindAll(
    [System.Windows.Automation.TreeScope]::Descendants,
    [System.Windows.Automation.Condition]::TrueCondition
  )
  $chosen = $null
  $chosenRank = -1
  foreach ($el in $all) {
    $cls = [string]$el.Current.ClassName
    if (-not $cls) { continue }
    $token = ($cls -split ' ', 2)[0]
    $rank = -1
    if ($token -eq 'tiptap' -and [string]$el.Current.Name -eq 'Prompt') { $rank = 3 }
    elseif ($token -eq 'tiptap') { $rank = 2 }
    elseif ($token -eq 'sand-kit-message-input-frame') { $rank = 1 }
    if ($rank -lt 0 -or $rank -lt $chosenRank) { continue }
    $bounds = $el.Current.BoundingRectangle
    if ($bounds.Width -lt 40 -or $bounds.Height -lt 16) { continue }
    $chosenRank = $rank
    $chosen = @{
      x = [int]($bounds.X + ($bounds.Width / 2))
      y = [int]($bounds.Y + ($bounds.Height / 2))
    }
    if ($rank -eq 3) { break }
  }
  return $chosen
}

$deadline = (Get-Date).AddMilliseconds($WaitMs)
$proc = Get-GrokBotWindow
while (-not $proc -and (Get-Date) -lt $deadline) {
  Start-Sleep -Milliseconds $PollMs
  $proc = Get-GrokBotWindow
}
if (-not $proc) {
  [Console]::Error.WriteLine('Grok Bot window not found')
  exit 2
}

$hwnd = $proc.MainWindowHandle
$rect = New-Object GrokBotPasteWin+RECT
if ([GrokBotPasteWin]::IsIconic($hwnd)) {
  [void][GrokBotPasteWin]::ShowWindow($hwnd, [GrokBotPasteWin]::SW_RESTORE)
  Start-Sleep -Milliseconds 700
}
[void][GrokBotPasteWin]::GetWindowRect($hwnd, [ref]$rect)
$width = [Math]::Abs($rect.Right - $rect.Left)
$height = [Math]::Abs($rect.Bottom - $rect.Top)
if ($width -lt 200 -or $height -lt 200) {
  [Console]::Error.WriteLine('Grok Bot window too small')
  exit 3
}

[void][GrokBotPasteWin]::SystemParametersInfo(
  [GrokBotPasteWin]::SPI_SETFOREGROUNDLOCKTIMEOUT,
  0,
  [IntPtr]::Zero,
  0
)
[void][GrokBotPasteWin]::LockSetForegroundWindow([GrokBotPasteWin]::LSFW_UNLOCK)
[void][GrokBotPasteWin]::AllowSetForegroundWindow([GrokBotPasteWin]::ASFW_ANY)

$fore = [GrokBotPasteWin]::GetForegroundWindow()
$foreProcOut = 0
$targetProcOut = 0
$ourTid = [GrokBotPasteWin]::GetCurrentThreadId()
$foreTid = [GrokBotPasteWin]::GetWindowThreadProcessId($fore, [ref]$foreProcOut)
$targetTid = [GrokBotPasteWin]::GetWindowThreadProcessId($hwnd, [ref]$targetProcOut)
$attachedFore = $false
$attachedTarget = $false
$attachedPair = $false
if ($foreTid -ne 0 -and $foreTid -ne $ourTid) {
  $attachedFore = [GrokBotPasteWin]::AttachThreadInput($ourTid, $foreTid, $true)
}
if ($targetTid -ne 0 -and $targetTid -ne $ourTid) {
  $attachedTarget = [GrokBotPasteWin]::AttachThreadInput($ourTid, $targetTid, $true)
}
if (
  $foreTid -ne 0 -and $targetTid -ne 0 -and $foreTid -ne $targetTid
) {
  $attachedPair = [GrokBotPasteWin]::AttachThreadInput($foreTid, $targetTid, $true)
}

$topMost = $false
$clipboardWritten = $false
$priorClipboard = ''
# The `finally` below is the only thing that unpins this window, and a process
# terminated from outside runs no cleanup at all -- not `finally`, and not a
# PowerShell.Exiting handler either, because a timeout kill is a
# TerminateProcess. So there is no second chance to add here: what keeps the
# window from being left HWND_TOPMOST over everything until the application
# restarts is the caller's timeout staying above this script's own budget. See
# grok_bot.PASTE_BUDGET_S, which that timeout is derived from.
try {
  [GrokBotPasteWin]::ShowRestored($hwnd)
  [GrokBotPasteWin]::SetTopMost($hwnd, $true)
  $topMost = $true
  Start-Sleep -Milliseconds $AfterTopMostMs
  [void][GrokBotPasteWin]::GetWindowRect($hwnd, [ref]$rect)
  $width = [Math]::Abs($rect.Right - $rect.Left)
  $height = [Math]::Abs($rect.Bottom - $rect.Top)
  $titleX = [int](($rect.Left + $rect.Right) / 2)
  $titleY = $rect.Top + 12
  $composeX = $rect.Left + [int]($width * 0.30)
  $composeYs = @(
    [int]($rect.Bottom - 88),
    [int]($rect.Bottom - 120),
    [int]($rect.Bottom - 64),
    [int]($rect.Bottom - 150)
  )

  $wshell = New-Object -ComObject WScript.Shell
  $frontDeadline = (Get-Date).AddMilliseconds(5000)
  $onTop = $false
  while ((Get-Date) -lt $frontDeadline) {
    [GrokBotPasteWin]::SetTopMost($hwnd, $true)
    [GrokBotPasteWin]::ShowRestored($hwnd)
    [void]$wshell.AppActivate($proc.Id)
    [void][GrokBotPasteWin]::SetForegroundWindow($hwnd)
    Start-Sleep -Milliseconds 180
    # Raise with the API, then ask who owns the pixel, and only then click.
    # Clicking first and checking afterwards put a real mouse-down and
    # mouse-up into whatever window owned that point -- and this loop only
    # repeats when the answer is "not Grok Bot", so every one of those clicks
    # landed in another application. The click stays because a real click
    # takes focus where SetForegroundWindow can be refused; it is now only
    # ever delivered to a pixel already shown to belong to the target.
    if (-not [GrokBotPasteWin]::PointBelongsTo($hwnd, $titleX, $titleY)) {
      Start-Sleep -Milliseconds 220
      continue
    }
    [GrokBotPasteWin]::ClickAt($titleX, $titleY)
    Start-Sleep -Milliseconds 120
    if ([GrokBotPasteWin]::PointBelongsTo($hwnd, $titleX, $titleY)) {
      $onTop = $true
      break
    }
    Start-Sleep -Milliseconds 220
  }
  if (-not $onTop) {
    [Console]::Error.WriteLine('Grok Bot not topmost')
    exit 5
  }

  [GrokBotPasteWin]::Tap([GrokBotPasteWin]::VK_MENU)
  [void][GrokBotPasteWin]::SetForegroundWindow($hwnd)
  Start-Sleep -Milliseconds 120

  $composer = $null
  $readerWas = $false
  $readerOn = $false
  try {
    [void][GrokBotPasteWin]::SystemParametersInfoBool(0x46, 0, [ref]$readerWas, 0)
    if (-not $readerWas) {
      [void][GrokBotPasteWin]::SystemParametersInfo(0x47, 1, [IntPtr]::Zero, 0)
      $readerOn = $true
      Start-Sleep -Milliseconds 800
    }
    $composer = Get-ComposerPoint $hwnd
  } catch {
    $composer = $null
  } finally {
    if ($readerOn) {
      [void][GrokBotPasteWin]::SystemParametersInfo(0x47, 0, [IntPtr]::Zero, 0)
    }
  }

  # The composer control is the click. The guessed points stay for a Grok
  # build that does not publish that control. Each one is checked before the
  # click, so a miss is not delivered into another window.
  $clickedComposer = $false
  if ($composer -and [GrokBotPasteWin]::PointBelongsTo($hwnd, $composer.x, $composer.y)) {
    [GrokBotPasteWin]::ClickAt($composer.x, $composer.y)
    Start-Sleep -Milliseconds $AfterClickMs
    $clickedComposer = $true
  } else {
    foreach ($composeY in $composeYs) {
      if ($composeY -le ($rect.Top + 40)) { continue }
      if (-not [GrokBotPasteWin]::PointBelongsTo($hwnd, $composeX, $composeY)) { continue }
      [GrokBotPasteWin]::ClickAt($composeX, $composeY)
      Start-Sleep -Milliseconds 90
    }
    Start-Sleep -Milliseconds $AfterClickMs
    $lastY = $composeYs[0]
    if (-not [GrokBotPasteWin]::PointBelongsTo($hwnd, $composeX, $lastY)) {
      [Console]::Error.WriteLine('Grok Bot not topmost')
      exit 5
    }
  }
  if (-not (Test-Foreground $hwnd)) {
    [void][GrokBotPasteWin]::SetForegroundWindow($hwnd)
    Start-Sleep -Milliseconds 80
  }
  if (-not (Test-Foreground $hwnd)) {
    [Console]::Error.WriteLine('Grok Bot not foreground')
    exit 5
  }

  # Every check has passed, so this is the first moment the task is allowed on
  # the clipboard. Text only is read back: an image or a file list cannot be
  # recovered through GetText, so for those the finally clears the clipboard
  # rather than claiming to restore it.
  try { $priorClipboard = [System.Windows.Forms.Clipboard]::GetText() } catch { $priorClipboard = '' }
  $copied = $false
  for ($i = 0; $i -lt 8 -and -not $copied; $i++) {
    try {
      [System.Windows.Forms.Clipboard]::SetText($text)
      $clipboardWritten = $true
      if ([System.Windows.Forms.Clipboard]::GetText() -eq $text) {
        $copied = $true
      }
    } catch {
      Start-Sleep -Milliseconds 80
    }
  }
  if (-not $copied) {
    [Console]::Error.WriteLine('clipboard')
    exit 4
  }
  [GrokBotPasteWin]::Chord([GrokBotPasteWin]::VK_CONTROL, [GrokBotPasteWin]::VK_V)
  Start-Sleep -Milliseconds $AfterPasteMs
  [GrokBotPasteWin]::Tap([GrokBotPasteWin]::VK_RETURN)
} finally {
  if ($clipboardWritten) {
    try {
      if ([string]::IsNullOrEmpty($priorClipboard)) {
        [System.Windows.Forms.Clipboard]::Clear()
      } else {
        [System.Windows.Forms.Clipboard]::SetText($priorClipboard)
      }
    } catch {
      # A clipboard that cannot be written back is not worth failing the paste
      # that already succeeded.
    }
  }
  if ($topMost) {
    [GrokBotPasteWin]::SetTopMost($hwnd, $false)
  }
  if ($attachedPair) {
    [void][GrokBotPasteWin]::AttachThreadInput($foreTid, $targetTid, $false)
  }
  if ($attachedTarget) {
    [void][GrokBotPasteWin]::AttachThreadInput($ourTid, $targetTid, $false)
  }
  if ($attachedFore) {
    [void][GrokBotPasteWin]::AttachThreadInput($ourTid, $foreTid, $false)
  }
}
exit 0
