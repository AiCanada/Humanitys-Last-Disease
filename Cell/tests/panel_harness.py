#!/usr/bin/env python3
"""Serve the REAL option-panel pages from a real BrowserSession: no job, no window.

Run inside WSL with the repository's virtual environment, from the repository root:

    cd <checkout>
    .venv/bin/python Cell/tests/panel_harness.py --delay 5

What it does
    * Builds ``ui.BrowserSession(opener=...)`` with an opener that only records the
      address. No Edge window is opened, and because the opener is not the module's
      own ``_open_browser`` the session never runs PowerShell on ``close()``.
    * Shows real pages through the real request handler, one ``session.exchange``
      per page: the landing page (``render_options``), the confirm page, the
      structures page, and so on. Each page gets the same Back button the real
      ``OptionPanel.ask`` gives it (``with_nav``).
    * After every POST it holds the answer for ``--delay`` seconds, the way a real
      step blocks while it works, and calls ``session.note_progress`` every
      ``--tick`` seconds so ``/progress`` moves.
    * After ``--steps`` answered pages it calls ``session.close()``. That answers the
      POST still being held with the short "Closing the panel." page, reports
      ``"closed": true`` on ``/progress`` for a moment, and stops the server.

Nothing is written outside ``--scratch``. The key store, the ssh host files, the
run log and the upload directory are pointed there before the package is imported,
and byte-code writing is switched off so the checkout gets no new ``.pyc`` file.

Every event is one line on stdout: ``[seconds] EVENT key=value ...``. The first
useful one is ``PORT port=NNNNN``. ``--port-file`` writes the same number to a file.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

START = time.monotonic()
PAGE_NAMES = ("options", "confirm", "choice", "focus", "structures", "targets", "inference", "done")


def say(event: str, **detail: object) -> None:
    """One flushed line per event, stamped with seconds since the harness started."""
    tail = " ".join(f"{key}={json.dumps(value)}" for key, value in detail.items())
    print(f"[{time.monotonic() - START:8.3f}] {event} {tail}".rstrip(), flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--delay", type=float, default=5.0, help="seconds each POST is held before the next page (default 5)"
    )
    parser.add_argument("--tick", type=float, default=0.5, help="seconds between note_progress calls (default 0.5)")
    parser.add_argument("--steps", type=int, default=3, help="answered pages before session.close() (default 3)")
    parser.add_argument(
        "--pages",
        default="options,confirm,structures",
        help=f"comma-separated pages, shown in order and repeated when steps run past them. Known: {', '.join(PAGE_NAMES)}",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory the landing page lists. Default is OptionForm()'s own 'disease_runs' under the cwd",
    )
    parser.add_argument("--rows", type=int, default=6, help="structure rows on the structures page (default 6)")
    parser.add_argument("--scratch", default=None, help="directory for every side effect (default: a new temp dir)")
    parser.add_argument("--port-file", default=None, help="write the panel port here once the server is listening")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=900.0,
        help="close and exit after this long even if nobody answered (default 900, 0 waits forever)",
    )
    return parser.parse_args()


def prepare_environment(scratch: Path) -> None:
    """Point every file the panel could read or write at the scratch directory."""
    for name in ("keys", "ssh", "tmp"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
    os.environ["RFD4_KEY_DIR"] = str(scratch / "keys")  # key_store.key_directory()
    os.environ["RFD4_SSH_DIR"] = str(scratch / "ssh")  # machines.ssh_directory()
    os.environ["HUMANITYS_LAST_DISEASE_LOG"] = str(scratch / "Humanitys-Last-Disease.log")  # logs.log_path()
    os.environ["TMPDIR"] = str(scratch / "tmp")
    # BrowserSession.upload_dir() calls tempfile.mkdtemp(prefix="Humanitys-Last-Disease-upload-").
    tempfile.tempdir = str(scratch / "tmp")
    sys.dont_write_bytecode = True


def main() -> int:
    arguments = parse_arguments()
    if hasattr(sys.stdout, "reconfigure"):
        # ui.py prints "Option panel: <url>" without a flush. Behind a pipe that line
        # would wait for the buffer to fill.
        sys.stdout.reconfigure(line_buffering=True)
    scratch = Path(arguments.scratch) if arguments.scratch else Path(tempfile.mkdtemp(prefix="panel-harness-"))
    scratch.mkdir(parents=True, exist_ok=True)
    prepare_environment(scratch)
    say("START", pid=os.getpid(), cwd=os.getcwd(), scratch=str(scratch), python=sys.version.split()[0])

    began = time.monotonic()
    from rfproteina.disease import ui
    from rfproteina.disease.logs import redact_fields
    from rfproteina.disease.lookup import Disease
    from rfproteina.disease.machines import discover_remote_machines
    from rfproteina.disease.structures import StructureOption

    say("IMPORTED", seconds=round(time.monotonic() - began, 3), torch_loaded="torch" in sys.modules)

    names = [name.strip() for name in arguments.pages.split(",") if name.strip()]
    unknown = [name for name in names if name not in PAGE_NAMES]
    if unknown or not names:
        print(f"Unknown page(s): {', '.join(unknown) or '(none given)'}", file=sys.stderr)
        return 2

    disease = Disease("Asthma", "MONDO_0004979")
    others = [Disease("Allergic asthma", "MONDO_0004784"), Disease("Childhood onset asthma", "MONDO_0005405")]
    rows = [
        StructureOption(
            group="protein",
            accession=f"P{number:05d}",
            entry_id=f"{number}HRN",
            chain="A",
            title=f"harness structure {number}",
            predicted=number % 3 == 0,
            source="harness",
        )
        for number in range(1, max(1, arguments.rows) + 1)
    ]
    ranked = [
        {
            "identifier": row.accession,
            "label": row.title,
            "combined": 0.5,
            "program_score": 3,
            "sources_asked": 7,
            "model_rating": None,
            "structures": 1,
            "structures_total": 1,
            "experimental_structures": 0 if row.predicted else 1,
        }
        for row in rows
    ]
    landing = ui.OptionForm() if arguments.out_dir is None else ui.OptionForm(out_dir=arguments.out_dir)

    def opener(url: str) -> None:
        port = int(url.rstrip("/").rsplit(":", 1)[-1])
        say("PORT", port=port, url=url)
        if arguments.port_file:
            Path(arguments.port_file).write_text(f"{port}\n", encoding="ascii")

    session = ui.BrowserSession(opener=opener)

    def render(name: str) -> str:
        if name == "options":
            # The same call OptionPanel._until_named makes for the first page.
            return ui.render_options(landing, machines=tuple(discover_remote_machines("")))
        if name == "confirm":
            return ui.render_confirm(disease, others)
        if name == "choice":
            return ui.render_disease_choice([disease, *others])
        if name == "focus":
            return ui.render_focus(disease, [("R-HSA-177929", "Signaling by EGFR")], [("EGFR", "P00533")])
        if name == "structures":
            return ui.render_structures(rows)
        if name == "targets":
            return ui.render_targets(ranked)
        if name == "inference":
            return ui.render_inference(len(rows))
        # The finished page, as OptionPanel.finish_choice shows it.
        session.note_closing(["Harness run: no job was started and nothing was generated."])
        return session.done_page()

    finished = threading.Event()
    failure: list[BaseException] = []

    def run_pages() -> None:
        try:
            index = 0
            for answered in range(1, arguments.steps + 1):
                name = names[index % len(names)]
                began_render = time.monotonic()
                html = ui.with_nav(render(name), back=index > 0, forward=False)
                say(
                    "PAGE",
                    step=index,
                    page=name,
                    bytes=len(html.encode("utf-8")),
                    render_seconds=round(time.monotonic() - began_render, 3),
                )
                fields = session.exchange(html, step=index)
                say("POST", step=index, answered=answered, fields=redact_fields(fields))
                ticks = int(round(arguments.delay / arguments.tick)) if arguments.delay > 0 and arguments.tick > 0 else 0
                action = ui.RECORDS_ACTION if answered % 2 else ui.DOWNLOAD_ACTION
                for done in range(1, ticks + 1):
                    session.note_progress(done, ticks, f"harness step {index}", action)
                    time.sleep(arguments.tick)
                if ticks:
                    say("WORKED", step=index, seconds=arguments.delay, progress_calls=ticks)
                move = (fields.get("__nav") or [""])[0].strip()
                if move == "back" and index > 0:
                    index = ui.back_target(fields, index)
                elif move in {"start", "fresh"}:
                    index = 0
                else:
                    index += 1
        except BaseException as error:  # noqa: BLE001 - reported by the main thread
            failure.append(error)
        finally:
            finished.set()

    def stop(_number: int, _frame: object) -> None:
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)

    worker = threading.Thread(target=run_pages, name="panel-harness-pages", daemon=True)
    worker.start()
    code = 0
    try:
        if not finished.wait(timeout=arguments.max_seconds or None):
            say("TIMEOUT", seconds=arguments.max_seconds)
            code = 3
    except KeyboardInterrupt:
        say("INTERRUPTED")
        code = 130
    finally:
        began_close = time.monotonic()
        say("CLOSE", step=session.step())
        session.close()
        say("CLOSED", seconds=round(time.monotonic() - began_close, 3))
    if failure:
        say("FAILED", error=repr(failure[0]))
        return 1
    say("EXIT", code=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
