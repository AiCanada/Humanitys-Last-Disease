"""Start, watch and stop the ``human`` program for the Cell relay.

The program runs inside WSL. The relay is its parent: the program's output comes
back over a pipe, so the Cell page can show what a terminal would show, and the
panel's port is read from the ``Option panel:`` line instead of being searched for.

Because of that pipe, the program ends up with nowhere to write if the relay is
stopped while it runs. ``Stop-Cell.ps1`` therefore stops the program first.
"""

from __future__ import annotations

import codecs
import collections
import logging
import queue
import re
import subprocess
import threading
import time

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
#: The program prints each of these once, as a whole line, when it starts. They are
#: matched as whole lines and taken once per run: the program also prints text it
#: fetched or was told, and that text must not be able to move the panel address or
#: the log the Cell page reads.
PANEL_LINE = re.compile(r"^Option panel: http://127\.0\.0\.1:(\d{1,5})/$")
PID_LINE = re.compile(r"^CELL-PID (\d+)$")
LOG_LINE = re.compile(r"^Run log: (.+)$")
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
#: Output lines kept for the Cell page. Older lines are still in the run log.
MAX_LINES = 4000
MAX_LINE_CHARS = 4000
#: Lines handed to the page in one answer.
MAX_BATCH = 600
#: How long each signal is given before the next, harsher one.
STOP_STEPS = (("INT", 15.0), ("TERM", 6.0), ("KILL", 4.0))
DEFAULT_LOG = "/.config/Humanitys-Last-Disease/Humanitys-Last-Disease.log"
#: Lines waiting to be typed into the program. A program that does not read its
#: input must not be able to block the relay, so the write happens on its own thread.
STDIN_QUEUE = 64
#: Seconds one reading of the run log is reused. Each reading starts a process in WSL.
LOG_REUSE = 2.0


class Program:
    """One ``human`` process at a time, started from the phone."""

    def __init__(
        self,
        *,
        distro: str,
        repo: str,
        command: str,
        local_argv: list[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._distro = distro
        self._repo = repo
        self._command = command
        #: Set by the tests: run this command on Windows instead of ``command`` in WSL.
        self._local = list(local_argv) if local_argv else None
        self._log = logger or logging.getLogger("cell")
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._linux_pid: int | None = None
        self._panel_port: int | None = None
        self._run_log: str | None = None
        self._started: float | None = None
        self._ended: float | None = None
        self._code: int | None = None
        self._stopping = False
        self._lines: collections.deque[tuple[int, str]] = collections.deque(maxlen=MAX_LINES)
        self._seq = 0
        self._partial = ""
        self._home: str | None = None
        self._typed: queue.Queue[str] | None = None
        self._log_read: tuple[float, list[str], str] = (0.0, [], "")
        self._log_lock = threading.Lock()

    # ------------------------------------------------------------------ WSL
    def _wsl(self) -> list[str]:
        return ["wsl.exe", *(["-d", self._distro] if self._distro else [])]

    def _wsl_run(self, argv: list[str], timeout: float = 15.0) -> tuple[int, str]:
        """Run one command in the distribution, with no shell in between."""
        try:
            done = subprocess.run(
                [*self._wsl(), "-e", *argv],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
                creationflags=NO_WINDOW,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return 1, str(error)
        text = (done.stdout + done.stderr).decode("utf-8", errors="replace").replace("\x00", "")
        return done.returncode, text

    def _home_dir(self) -> str | None:
        if self._home is None:
            code, text = self._wsl_run(["sh", "-c", "echo $HOME"])
            home = text.strip().splitlines()[-1].strip() if code == 0 and text.strip() else ""
            if home.startswith("/"):
                self._home = home
        return self._home

    # ---------------------------------------------------------------- state
    def alive(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def panel_port(self) -> int | None:
        """The panel port of the program started here, while that program runs."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return None
            return self._panel_port

    def state(self) -> dict[str, object]:
        with self._lock:
            alive = self._proc is not None and self._proc.poll() is None
            if alive and self._stopping:
                name = "stopping"
            elif alive and self._panel_port is None and time.time() - (self._started or 0) < 180:
                name = "starting"
            elif alive:
                name = "running"
            elif self._ended is not None:
                name = "exited"
            else:
                name = "idle"
            return {
                "state": name,
                "managed": alive,
                "code": self._code,
                "started": self._started,
                "ended": self._ended,
            }

    # ---------------------------------------------------------------- start
    def start(self) -> tuple[bool, str]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return False, "The program is already running."
            if self._local is not None:
                argv = list(self._local)
            else:
                # $$ is the shell's id, and exec keeps it, so the first line names the
                # program's own Linux process. A login shell gives it the same PATH a
                # terminal would. The script carries no quotes: it crosses wsl.exe.
                script = "echo CELL-PID $$; export PYTHONUNBUFFERED=1; exec " + self._command
                argv = [*self._wsl(), "--cd", self._repo, "-e", "bash", "-lc", script]
            try:
                proc = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=NO_WINDOW,
                    bufsize=0,
                )
            except OSError as error:
                self._log.error("program did not start: %s", error)
                return False, f"The program could not be started: {error}"
            self._proc = proc
            self._linux_pid = None
            self._panel_port = None
            self._run_log = None
            self._started = time.time()
            self._ended = None
            self._code = None
            self._stopping = False
            self._partial = ""
            typed: queue.Queue[str] = queue.Queue(maxsize=STDIN_QUEUE)
            self._typed = typed
            self._add("[Cell] Starting " + self._command)
        self._log.info("program started (windows pid %s)", proc.pid)
        threading.Thread(target=self._read, args=(proc,), name="cell-program-output", daemon=True).start()
        threading.Thread(target=self._type, args=(proc, typed), name="cell-program-input", daemon=True).start()
        return True, "Starting."

    def _type(self, proc: subprocess.Popen[bytes], typed: queue.Queue[str]) -> None:
        """Write queued lines to the program's input. A full pipe blocks here, and nowhere else."""
        while proc.poll() is None:
            try:
                text = typed.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                if proc.stdin is None:
                    return
                proc.stdin.write((text + "\n").encode("utf-8"))
                proc.stdin.flush()
            except (OSError, ValueError):
                return

    # --------------------------------------------------------------- output
    def _add(self, text: str) -> None:
        """Keep one finished output line. The caller holds the lock."""
        self._seq += 1
        self._lines.append((self._seq, text[:MAX_LINE_CHARS]))

    def _line(self, text: str) -> None:
        text = ANSI.sub("", text).rstrip()
        found = PID_LINE.match(text)
        if found and self._linux_pid is None:
            self._linux_pid = int(found.group(1))
            return
        if self._panel_port is None:
            panel = PANEL_LINE.match(text)
            if panel and 0 < int(panel.group(1)) < 65536:
                self._panel_port = int(panel.group(1))
                self._log.info("program opened its panel on 127.0.0.1:%s", self._panel_port)
        if self._run_log is None:
            log_file = LOG_LINE.match(text)
            if log_file:
                self._run_log = log_file.group(1).strip()
        self._add(text)

    @staticmethod
    def _overwritten(text: str) -> str:
        """What a terminal would show for text holding carriage returns.

        A carriage return restarts the line only once something is written after it.
        One at the very end is kept, so the text before it stays visible until then.
        """
        head, found, tail = text.rstrip("\r").rpartition("\r")
        return tail if found else head or tail

    def _feed(self, text: str) -> None:
        """Add output text. A carriage return restarts the line, as a terminal does."""
        with self._lock:
            # Joined before the line ends are normalised: a read can end between
            # the carriage return and the line feed of one line ending.
            pieces = (self._partial + text).replace("\r\n", "\n").split("\n")
            self._partial = pieces.pop()
            for piece in pieces:
                self._line(self._overwritten(piece))
            head, found, tail = self._partial.rpartition("\r")
            if found and tail:
                self._partial = tail
            self._partial = self._partial[-MAX_LINE_CHARS:]

    def _read(self, proc: subprocess.Popen[bytes]) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        stream = proc.stdout
        try:
            while stream is not None:
                chunk = stream.read(65536)
                if not chunk:
                    break
                # wsl.exe reports its own failures in UTF-16; dropping the zero
                # bytes leaves those messages readable beside the UTF-8 output.
                self._feed(decoder.decode(chunk).replace("\x00", ""))
        except (OSError, ValueError):
            pass
        code = proc.wait()
        with self._lock:
            if self._partial:
                self._line(self._overwritten(self._partial))
                self._partial = ""
            if self._proc is proc:
                self._code = code
                self._ended = time.time()
                self._panel_port = None
                self._stopping = False
            self._add(f"[Cell] The program ended with exit code {code}.")
        self._log.info("program ended with exit code %s", code)

    def output(self, since: int) -> dict[str, object]:
        """Lines after ``since``. ``reset`` means the page's counter is from an older relay."""
        with self._lock:
            reset = since > self._seq
            if reset:
                since = 0
            lines = [text for seq, text in self._lines if seq > since][-MAX_BATCH:]
            partial = ANSI.sub("", self._partial.rstrip("\r"))
            return {"lines": lines, "next": self._seq, "partial": partial, "reset": reset}

    def send_line(self, text: str) -> tuple[bool, str]:
        """Type one line into the program's standard input."""
        text = text.replace("\r", "").replace("\n", " ")[:2000]
        with self._lock:
            proc = self._proc
            typed = self._typed
            if proc is None or proc.poll() is not None or typed is None:
                return False, "The program is not running under Cell."
            try:
                typed.put_nowait(text)
            except queue.Full:
                return False, "The program is not reading its input."
            self._add("> " + text)
        return True, "Sent."

    # ----------------------------------------------------------------- stop
    def _signal(self, pid: int, name: str) -> tuple[int, str]:
        """Signal the program's process group, as Ctrl+C in a terminal does, or the process alone."""
        code, text = self._wsl_run(["kill", "-s", name, "--", f"-{pid}"])
        if code != 0:
            code, text = self._wsl_run(["kill", "-s", name, str(pid)])
        return code, text.strip()

    def _stop_managed(self, proc: subprocess.Popen[bytes], pid: int | None) -> None:
        def gone(seconds: float) -> bool:
            try:
                proc.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                return False
            return True

        if self._local is not None or pid is None:
            proc.terminate()
            gone(5.0)
        else:
            for name, seconds in STOP_STEPS:
                code, text = self._signal(pid, name)
                self._log.info("sent SIG%s to linux pid %s (exit %s %s)", name, pid, code, text)
                if gone(seconds):
                    break
        if proc.poll() is None:
            proc.kill()

    def _pid_listening(self, port: int) -> int | None:
        """The Linux process listening on ``port``, for a program started on the computer."""
        code, text = self._wsl_run(["ss", "-ltnpH"])
        if code != 0:
            return None
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[3].rpartition(":")[2] == str(port):
                found = re.search(r"pid=(\d+)", line)
                if found:
                    return int(found.group(1))
        return None

    def stop(self, external_port: int | None = None) -> tuple[bool, str]:
        """Interrupt the program. ``external_port`` names a panel that Cell did not start."""
        with self._lock:
            proc = self._proc
            alive = proc is not None and proc.poll() is None
            if alive and self._stopping:
                return True, "The program is already stopping."
            if alive:
                self._stopping = True
                pid = self._linux_pid
                self._add("[Cell] Stop requested.")
        if alive and proc is not None:
            threading.Thread(
                target=self._stop_managed, args=(proc, pid), name="cell-program-stop", daemon=True
            ).start()
            return True, "Stopping the program."
        if external_port is not None and self._local is None:
            pid = self._pid_listening(external_port)
            if pid is None:
                return False, "This program was started on the computer and Cell could not find it. Stop it there."
            code, text = self._signal(pid, "INT")
            if code != 0:
                return False, f"The program could not be interrupted: {text}"
            self._log.info("interrupted external program, linux pid %s", pid)
            return True, "Asked the program to stop."
        return False, "The program is not running."

    # ------------------------------------------------------------------ log
    def tail_log(self, count: int = 200) -> tuple[list[str], str]:
        """The end of the program's run log, and a note when there is nothing to show.

        One reading is reused for a moment, and readings do not overlap: each one
        starts a process in WSL, and the page asks every few seconds.
        """
        with self._log_lock:
            at, lines, note = self._log_read
            if time.monotonic() - at < LOG_REUSE:
                return lines, note
            lines, note = self._read_log(count)
            self._log_read = (time.monotonic(), lines, note)
            return lines, note

    def _read_log(self, count: int) -> tuple[list[str], str]:
        with self._lock:
            path = self._run_log
        if self._local is not None:
            if not path:
                return [], "No run log."
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    return handle.read().splitlines()[-count:], ""
            except OSError:
                return [], "No run log yet."
        if not path:
            home = self._home_dir()
            if not home:
                return [], "The run log could not be read."
            path = home + DEFAULT_LOG
        code, text = self._wsl_run(["tail", "-n", str(count), "--", path], timeout=10)
        if code != 0:
            return [], "No run log yet."
        return text.splitlines()[-count:], ""
