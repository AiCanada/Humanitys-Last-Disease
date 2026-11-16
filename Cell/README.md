# Cell

Run the `human` program from a phone. Cell gives an Android phone or an iPhone the same
option panel the desktop window shows, plus the things only the computer could do before:
start the program, read what it prints, and stop it. The connection is private, over
Tailscale, which is a WireGuard tunnel.

## How it works

```
phone browser ──Tailscale (WireGuard)──> this computer :8765    tailscale serve, tailnet only
                                              └─> 127.0.0.1:8766    cell_relay.py
                                                      ├─> /cell/…    Cell's own page: start, output, stop
                                                      └─> 127.0.0.1:<panel port>    the human panel, in WSL
```

The panel listens on `127.0.0.1` at a port that changes on every start, and it was written
for one browser on the same computer. `cell_relay.py` holds a fixed port and stands between
the phone and the panel. The program itself is not changed, and its panel is still bound
to localhost.

## On this computer

```powershell
cd <checkout>\Cell
.\Start-Cell.ps1              # Tailscale up, relay on, tailnet port open; prints the phone address
.\Get-CellStatus.ps1          # one line per link in the chain, and the recent relay log
.\Install-CellAutostart.ps1   # start Cell by itself when you sign in to Windows
.\Stop-Cell.ps1               # tailnet port closed, relay off
```

If PowerShell refuses to run scripts:
`powershell -ExecutionPolicy Bypass -File .\Start-Cell.ps1`

You do not need to start `human` yourself. The phone starts it. A program you start in a
terminal on the computer is found and mirrored as well.

The computer has to be on, awake, and signed in to Windows. A locked screen is fine.
Without the autostart, Cell is off after every restart until `Start-Cell.ps1` is run again.

## On an Android phone

1. Install **Tailscale** from Google Play and sign in with the same account as this computer.
2. Turn the connection on in the app.
3. In Chrome, open the address `Start-Cell.ps1` printed: `http://100.x.y.z:8765/`.
   Use the numbers, not the `ts.net` name: Chrome treats that name as a public site and
   may insist on https.
4. Chrome menu → **Add to Home screen** gives it an icon.

## On an iPhone

1. Install **Tailscale** from the App Store and sign in with the same account as this computer.
2. Turn the connection on. In the Tailscale app, **Settings → VPN On Demand** keeps it on
   by itself.
3. In Safari, open the same address: `http://100.x.y.z:8765/`.
4. **Share → Add to Home Screen.** It then opens full screen, as its own app, with the
   H.U.M.A.N icon. There is no address bar there, so every Cell page has a **Reload** button,
   and every panel page has a round **Cell** button that leads back to the Cell page.

An iPhone or Android phone runs one VPN at a time, so Tailscale and the WireGuard app
cannot both be connected on the same phone.

## Using it from the phone

**The Cell page** (`/cell/`) is where the phone lands when the program is not running.

- **Start program** starts `human` on the computer. The panel opens on the phone by itself
  a few seconds later. The panel window also opens on the computer's desktop, as always.
- **Program output** shows what the terminal would show. **Run log** shows the end of the
  program's own log (`~/.config/Humanitys-Last-Disease/Humanitys-Last-Disease.log` inside WSL).
- **Stop program** interrupts it the way Ctrl+C does.
- **Answer a question from the program** types one line into the program, should it ever
  ask something outside the panel. The panel itself does not need it.

**The panel pages** work as on the desktop. Three things are different on purpose:

- **Long steps show a Working page.** A phone gives up on a page that takes too long to
  arrive (an iPhone after 60 seconds), and a download or a generation run takes far longer.
  When the program does not answer within a few seconds, the phone is shown a page with
  the progress bar and the output, and moves on to the next panel page by itself when it
  is ready. Locking the phone or losing signal meanwhile does not disturb the run. Open
  the address again and you are taken to wherever the program is now.
- **The browser's Back does not rewind the run.** On the desktop, going back in the
  browser sends the program back a step. A phone reloads old tabs on its own and a swipe
  from the screen edge is Back, so through Cell an old address shows the current page.
  An iPhone can also bring an old page back from memory without asking for it again; the
  page notices and moves to the current one. Use the **Back** button on the panel page
  to go back a step.
- **The file picker shows every file.** Phones grey out file types they do not know, such
  as `.cif` and `.pdb`. The program still checks the file after the upload.

The desktop window and the phone show the same session. When one of them moves the panel
on, the other follows within a second. If both send the same page at once, the program
uses the first and both get its next page.

## HTTPS (optional)

Inside the tailnet the connection is already encrypted by WireGuard, but the address is
`http://`, so browsers label it "Not secure". To get `https://<computer>.<tailnet>.ts.net/`:

1. In the Tailscale admin console, **DNS → Enable HTTPS**. This publishes the computer's
   tailnet name in the public certificate log. The name is the one shown on the Machines
   page, and it can be changed there first.
2. `.\Start-Cell.ps1 -Https`

The setting stays on for later runs. With https, Chrome on Android offers a full install
of the page as an app.

## Using the WireGuard app instead of Tailscale

Tailscale already carries the traffic in WireGuard and needs no router setup. A plain
WireGuard tunnel between the WireGuard apps works too, but you set it up yourself, and no
tunnel is configured on this computer at present.

1. On the computer, make a tunnel that listens, for example `cell.conf`:

   ```ini
   [Interface]
   PrivateKey = <computer private key, from: wg genkey>
   Address = 10.66.0.1/24
   ListenPort = 51820

   [Peer]
   PublicKey = <phone public key>
   AllowedIPs = 10.66.0.2/32
   ```

   Install it from an administrator prompt: `wireguard /installtunnelservice C:\path\to\cell.conf`

2. On the phone, in the WireGuard app:

   ```ini
   [Interface]
   PrivateKey = <phone private key>
   Address = 10.66.0.2/32

   [Peer]
   PublicKey = <computer public key>
   Endpoint = <your home's public address>:51820
   AllowedIPs = 10.66.0.1/32
   PersistentKeepalive = 25
   ```

3. On the router, forward UDP port 51820 to this computer. This is not possible when the
   internet provider puts the home behind carrier-grade NAT.
4. Once, from an administrator PowerShell, let the phone reach the relay on that address:

   ```powershell
   New-NetFirewallRule -DisplayName "Cell relay (WireGuard)" -Direction Inbound -Protocol TCP `
       -LocalPort 8766 -LocalAddress 10.66.0.1 -RemoteAddress 10.66.0.2 -Action Allow
   ```

5. `.\Start-Cell.ps1 -WireGuardAddress 10.66.0.1 -WireGuardPeer 10.66.0.2`

The phone then opens `http://10.66.0.1:8766/`.

## Safety

The panel has no sign-in of its own. Whoever reaches it can start runs, name machines to
run on, and use the saved API keys. Cell therefore checks every connection:

- **Tailnet only.** The port is published with `tailscale serve`, never `tailscale funnel`.
- **Your devices only.** Tailscale tells the relay which device opened each connection,
  and the relay admits only devices signed in to Tailscale as this computer's own user.
  Devices of other users, devices shared in from another tailnet, and tagged devices get a
  "Not allowed" page. When Tailscale itself does not answer, the device is asked to wait
  and nothing is let in. `cell.log` records each device as `allowed` or `REFUSED`.
  `-AnyTailnetDevice` admits every device Tailscale can name instead. `Get-CellStatus.ps1`
  shows which of the two rules the running relay applies.
- **Only its own addresses.** A request that names any other host is refused, so a web
  page elsewhere cannot point its own name at the relay.
- **Only its own pages.** A form or a Start/Stop request is accepted only when it was sent
  by a page of this relay, at the address the phone uses. Another site in the browser, or
  another program's page on the same computer, cannot submit one, and cannot make the
  relay do work with requests it cannot read the answers to.
- **Stopping the relay is for the computer alone.** No phone can do it.
- **The phone is the key.** Anyone holding the unlocked phone can do what you can. If it is
  lost, remove it on the Machines page of the Tailscale admin console.
- The owner check does not cover a plain WireGuard tunnel. There only the address given
  with `-WireGuardPeer` is admitted, and `-WireGuardAddress` is refused without it.

## Limits

- The program's **Grok Bot** question drives the Grok Bot desktop app on the computer. It
  needs the computer's desktop and is unlikely to work while the screen is locked.
- A program the phone started runs under the relay. `Stop-Cell.ps1` refuses to stop the
  relay while it runs; `-Force` interrupts the program first. Restarting the computer ends
  it, as it would end one started in a terminal.
- A program started in a terminal on the computer can be driven and stopped from the phone,
  but its output is on that terminal. The phone shows its run log only.
- Uploads are limited to 200 MB (`--max-upload-mb` on the relay).

## What has been checked

Checked on this computer: the 36 tests below, including the real panel pages served from
WSL; starting the real `human` through the Tailscale address, reading its first page and
stopping it; the Cell page, the Working page and the Cell button in desktop Chrome. Three
reviewers then attacked and exercised private copies of the relay, and what they found
was fixed and is covered by those tests.

Not checked: a real phone browser, either kind. The iPhone path has never been run, since
no iPhone is in the tailnet. `-Https` has not been run because HTTPS is not enabled for
the tailnet. The WireGuard app path has not been run.

## Files

| File | Purpose |
| --- | --- |
| `Start-Cell.ps1` | Start mirroring and print the phone address. `-Https`, `-HttpsPort`, `-AnyTailnetDevice`, `-Program`, `-Distro`, `-Port`, `-RelayPort`, `-PanelPort`, `-WireGuardAddress`, `-WireGuardPeer`, `-Force`, `-Quiet`. |
| `Get-CellStatus.ps1` | Tailscale, relay, access rule, tailnet port, program, panel, phones, autostart, and why the last start failed. Exit code 1 when the mirror is down. |
| `Stop-Cell.ps1` | Stop mirroring. `-Force`, `-TailscaleDown`. |
| `Install-CellAutostart.ps1` | A shortcut in your Startup folder. `-Remove` takes it out. `-StartArguments "-Https"` passes options. |
| `Cell.Common.ps1` | Functions the scripts share. |
| `cell_relay.py` | The relay: owner check, proxy, Working page, routing. |
| `cell_program.py` | Starts, watches and stops the program in WSL. |
| `cell_pages.py` | The Cell page, the phone tags added to panel pages, the icon. |
| `tests/` | `test_cell.py`, a stand-in panel, and a harness that serves the real panel pages. |
| `cell.log`, `cell.pid`, `cell.start.log` | Written while running, or when a start failed; ignored by git. |

## Tests

```powershell
cd <checkout>\Cell
python -m unittest discover -s tests -v
```

Standard library only. One test serves the real panel pages from WSL and is skipped when
WSL or the checkout's `.venv` is missing. It writes nothing outside a temporary folder.
While that one test runs, a phone looking at the Cell page can briefly see its panel.

## Troubleshooting

Run `.\Get-CellStatus.ps1` first. It names the link that is down.

- **The phone cannot connect:** the Tailscale app on the phone must be on and must list
  this computer. After a restart of the computer, sign in to Windows; the autostart then
  starts Cell. If it did not, `Get-CellStatus.ps1` shows why the last start failed.
- **"Not allowed":** the phone is signed in to Tailscale as a different user. Sign in with
  the same account, or start with `-AnyTailnetDevice`.
- **"One moment":** the computer could not ask Tailscale about the phone yet, or the panel
  was busy. The page tries again by itself.
- **Start program does nothing useful:** read the Program output on the Cell page. If the
  environment is in another WSL distribution, start with `-Distro <name>`; if the command
  differs, with `-Program "<command>"`.
- **A program started on the computer is not found:** give its port from the
  `Option panel: http://127.0.0.1:<port>/` line: `.\Start-Cell.ps1 -PanelPort <port>`.
