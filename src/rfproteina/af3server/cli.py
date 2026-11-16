"""``rfd4-af3server`` -- the two ends of an AlphaFold Server round trip.

The middle of the round trip is a person: the server publishes no submission
API, so the upload is a click and the download is a browser's. What this
automates is everything on either side of that click, which is where the
mistakes actually live -- a request the server rejects after the upload, a
result archive filed against the wrong design, a batch that quietly ran the
first thirty of forty designs.

::

    rfd4-af3server login    --downloads ~/Downloads --accept-terms
    rfd4-af3server prepare  --run disease_runs/my_run --to af3server_jobs
    rfd4-af3server submit   --run disease_runs/my_run --to af3server_jobs
    #  ... upload the printed file, wait, download the results ...
    rfd4-af3server retrieve --run disease_runs/my_run --watch

Every subcommand takes ``--json`` and prints a machine-readable record instead
of prose, so this is usable from a script without parsing sentences.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rfproteina.af3server import request as request_module
from rfproteina.af3server import session as session_module
from rfproteina.disease.publish import repository_relative
from rfproteina.metrics.oracles.imported_refolds import PathOracle

# ``bundle`` is imported inside ``_retrieve`` rather than here. It reaches the
# oracle metric tables, which pull in torch and a model stack, and ``login``,
# ``status``, ``logout``, ``check`` and ``prepare`` have no use for either.

#: Default directory, under the run, that upload files are written to.
DEFAULT_BATCH_DIR = "af3server_requests"
#: How long ``--watch`` waits for an archive before giving up.
WATCH_SECONDS = 1800
#: How often ``--watch`` looks.
WATCH_INTERVAL = 5.0
#: Archives this will pick up on its own.
_ARCHIVE_GLOB = "*.zip"


@dataclass
class Printer:
    """Prose for a person, or one JSON record for a program."""

    as_json: bool
    stream: object = None

    def say(self, line: str) -> None:
        if not self.as_json:
            print(line, file=self.stream or sys.stdout)

    def record(self, payload: dict[str, object]) -> None:
        if self.as_json:
            print(json.dumps(payload, indent=2, sort_keys=True), file=self.stream or sys.stdout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfd4-af3server",
        description="Prepare AlphaFold Server requests and file the results it returns.",
    )
    parser.add_argument("--json", action="store_true", help="Print one JSON record instead of prose.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    login = subparsers.add_parser("login", help="Record that this machine has a signed-in browser session.")
    login.add_argument("--account", help="A label for the account. Never a password or a token.")
    login.add_argument("--downloads", type=Path, help="Where the browser puts downloads.")
    login.add_argument("--accept-terms", action="store_true", help="Record that you accepted the terms on the site.")
    login.add_argument(
        "--daily-limit",
        type=int,
        help=f"This account's own daily job allowance. Defaults to the published "
        f"{session_module.DAILY_JOB_LIMIT}; set it if yours differs, so a batch is not reported as "
        f"overflowing a limit that does not apply to it.",
    )

    subparsers.add_parser("status", help="Show the recorded session and today's remaining job estimate.")
    subparsers.add_parser("logout", help="Forget the recorded session and ledger.")

    prepare = subparsers.add_parser("prepare", help="Write upload files for the designs in a run.")
    _add_run_arguments(prepare)

    submit = subparsers.add_parser("submit", help="Write the files, open the upload page, record the submission.")
    _add_run_arguments(submit)
    submit.add_argument("--no-open", action="store_true", help="Do not open a browser; just print the page.")
    submit.add_argument("--no-record", action="store_true", help="Do not count these jobs against the quota ledger.")

    retrieve = subparsers.add_parser("retrieve", help="File downloaded result archives against the run's designs.")
    retrieve.add_argument("--run", type=Path, required=True, help="The run directory the designs are under.")
    retrieve.add_argument("--archive", type=Path, action="append", default=[], help="A result archive. Repeatable.")
    retrieve.add_argument("--watch", action="store_true", help="Wait for new archives in the downloads directory.")
    retrieve.add_argument("--watch-seconds", type=int, default=WATCH_SECONDS, help="How long --watch waits.")
    retrieve.add_argument("--designs", help="Comma-separated design ids. Defaults to every design in the run.")
    retrieve.add_argument(
        "--chain-of-interest",
        help="Chain the interface metrics are taken on, named as the DESIGN spells it. The server "
        "keeps no chain names and labels them by position, so this is translated per design; "
        "naming the server's own label works too.",
    )
    retrieve.add_argument("--save-id", help="File under this save id rather than the design's default slot.")
    retrieve.add_argument("--every-model", action="store_true", help="File all five models, not just the best.")
    retrieve.add_argument("--oracle-name", default="af3", help="Oracle whose paths the folds are filed at.")

    check = subparsers.add_parser("check", help="Validate an upload file against the dialect, without sending it.")
    check.add_argument("file", type=Path, nargs="+", help="Upload JSON files to validate.")

    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=Path, required=True, help="A run directory holding generated designs.")
    parser.add_argument("--to", type=Path, help=f"Where to write upload files. Default: <run>/{DEFAULT_BATCH_DIR}")
    parser.add_argument("--designs", help="Comma-separated design ids. Defaults to every design in the run.")
    parser.add_argument(
        "--per-file",
        type=int,
        default=request_module.JOBS_PER_FILE,
        help=f"Jobs per upload file. The server imports at most {request_module.JOBS_PER_FILE}.",
    )
    # On both prepare and submit, because submit builds the same files: a flag
    # that changed the request on one verb and not the other would mean the file
    # you validated is not the file you uploaded.
    parser.add_argument(
        "--design-templates",
        action="store_true",
        help="Let the server find PDB templates for the designed chain. Off by default: a template for the "
        "design under test can supply the answer the refold is meant to produce from sequence alone.",
    )
    parser.add_argument(
        "--no-context-templates",
        action="store_true",
        help="Also turn templates off for the non-designed chains. They are left on by default, because the "
        "target's experimental structure is a real input to the complex question.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    printer = Printer(arguments.json)
    try:
        handler = {
            "login": _login,
            "status": _status,
            "logout": _logout,
            "prepare": _prepare,
            "submit": _submit,
            "retrieve": _retrieve,
            "check": _check,
        }[arguments.command]
        return handler(arguments, printer)
    except ValueError as exc:
        # Every refusal in this package is a ValueError carrying a sentence
        # meant for the person who ran the command.
        printer.say(f"Refused: {exc}")
        printer.record({"error": str(exc)})
        return 2


def _login(arguments: argparse.Namespace, printer: Printer) -> int:
    ledger = session_module.sign_in(
        account=arguments.account,
        downloads=arguments.downloads,
        terms_acknowledged=True if arguments.accept_terms else None,
        daily_limit=arguments.daily_limit,
    )
    printer.say("Recorded. Nothing was sent anywhere, and no credential was stored.")
    printer.say(f"  Sign in yourself at {session_module.submission_url()} with your Google account.")
    if not ledger.terms_acknowledged:
        printer.say("  You have not recorded accepting the non-commercial terms (--accept-terms).")
    if ledger.downloads is None:
        printer.say("  No downloads directory recorded, so `retrieve --watch` has nowhere to look.")
    printer.record(ledger.as_dict())
    return 0


def _status(arguments: argparse.Namespace, printer: Printer) -> int:
    ledger = session_module.load_ledger()
    printer.say(f"AlphaFold Server  {session_module.submission_url()}")
    printer.say(f"  Session recorded here: {'yes' if ledger.account else 'no label stored'}")
    printer.say(f"  Terms acknowledged:    {'yes' if ledger.terms_acknowledged else 'no'}")
    printer.say(f"  Downloads watched:     {ledger.downloads or 'not set'}")
    published = "" if ledger.daily_limit is None else " (set here, not the published 30)"
    printer.say(f"  Submitted today:       {ledger.today()} of {ledger.limit}{published}")
    printer.say(f"  Remaining (estimate):  {ledger.remaining()}")
    printer.say("  The estimate is this machine's own count. The server's own tally is the one that decides.")
    printer.record(ledger.as_dict())
    return 0


def _logout(arguments: argparse.Namespace, printer: Printer) -> int:
    existed = session_module.sign_out()
    printer.say("Forgot the recorded session." if existed else "There was nothing recorded.")
    printer.say("Your browser session is untouched: sign out there if that is what you meant.")
    printer.record({"forgot": existed})
    return 0


def _build(arguments: argparse.Namespace, printer: Printer) -> tuple[request_module.RequestReport, Path]:
    """The shared half of prepare and submit."""
    destination = arguments.to or (arguments.run / DEFAULT_BATCH_DIR)
    report = request_module.jobs_from_run(
        arguments.run,
        example_ids=_split(arguments.designs),
        design_templates=arguments.design_templates,
        # None leaves the server's own default, which is on. Only False turns it
        # off, and it has to be emitted explicitly to do so.
        context_templates=False if arguments.no_context_templates else None,
    )
    if report.jobs:
        report.files = request_module.write_batches(report.jobs, destination, per_file=arguments.per_file)
    return report, destination


def _prepare(arguments: argparse.Namespace, printer: Printer) -> int:
    report, destination = _build(arguments, printer)
    _say_build(report, destination, printer)
    printer.record(report.as_dict())
    return 0 if report.jobs else 1


def _submit(arguments: argparse.Namespace, printer: Printer) -> int:
    report, destination = _build(arguments, printer)
    _say_build(report, destination, printer)
    if not report.jobs:
        printer.record(report.as_dict())
        return 1

    ledger = session_module.load_ledger()
    plan = session_module.plan_submission(len(report.jobs), ledger)
    if not ledger.terms_acknowledged:
        printer.say("")
        printer.say("You have not recorded accepting the AlphaFold Server terms. It is free for")
        printer.say("non-commercial use only, and you accept that on the site, not here.")
    printer.say("")
    printer.say(f"Daily budget: {plan.remaining_estimate} of {ledger.limit} left, by this machine's own count.")
    if plan.deferred:
        printer.say(f"  {plan.today} of {len(report.jobs)} jobs fit today; {plan.deferred} will not ({plan.blocker}).")
        printer.say("  The server lets a job be saved and run when the quota refreshes.")
    printer.say("")
    printer.say("Now do the upload yourself -- there is no submission API:")
    for path in report.files:
        printer.say(f"  1. Upload {repository_relative(path)} with the page's Upload JSON button.")
    printer.say("  2. The jobs arrive as saved drafts. Run them.")
    printer.say("  3. Download each result archive, then: rfd4-af3server retrieve --run <run> --watch")

    if not arguments.no_open:
        _open_page(session_module.submission_url(), printer)
    else:
        printer.say(f"  Page: {session_module.submission_url()}")

    recorded = 0
    if not arguments.no_record:
        # Every job that was written, not only the ones the plan says fit today.
        # The files are handed over whole and which drafts get run is the
        # person's choice, so the count this can honestly keep is what it gave
        # them. The ledger is an estimate either way -- a draft is not a run,
        # and the server's own tally is the one that decides.
        recorded = len(report.jobs)
        session_module.record_submission(recorded)
        printer.say("")
        printer.say(f"Counted {recorded} job(s) against today's ledger. Use --no-record if you did not submit them.")

    printer.record({**report.as_dict(), "plan": plan.as_dict(), "recorded": recorded})
    return 0


def _say_build(report: request_module.RequestReport, destination: Path, printer: Printer) -> None:
    printer.say(f"{len(report.jobs)} job(s) built from the run.")
    for path in report.files:
        printer.say(f"  wrote {repository_relative(path)}")
    for job in report.jobs:
        sized = job.tokens()
        floor = "at least " if not sized.exact else ""
        # Both names, because the results come back under the server's and the
        # screen's criteria are written against the design's.
        chains = ", ".join(f"{strand}->{label}" for strand, label in job.chain_labels().items())
        marked = f" (design chain {job.chain_of_interest})" if job.chain_of_interest else ""
        printer.say(f"  {job.name}: chains {chains}{marked}, {floor}{sized.tokens} tokens")
        if not sized.exact:
            printer.say(f"    uncounted ligand atoms: {', '.join(sized.uncounted)} -- the total is a floor")
    if report.refused:
        printer.say(f"{len(report.refused)} design(s) could not be sent:")
        for name, blocker in report.refused:
            printer.say(f"  {name}: {blocker}")
        printer.say("  A design is refused rather than sent without an entity it holds: a request that")
        printer.say("  drops a chain does not fail, it answers a different question.")
    if not report.jobs:
        printer.say(f"Nothing to upload, so nothing was written to {destination}.")


def _retrieve(arguments: argparse.Namespace, printer: Printer) -> int:
    from rfproteina.af3server.bundle import import_archive, read_archive

    oracle = _oracle(arguments.oracle_name)
    archives = list(arguments.archive)
    if arguments.watch:
        found = _watch(printer, arguments.watch_seconds, seen=set(archives))
        archives.extend(found)
    if not archives:
        printer.say("No archives to file. Pass --archive, or --watch with a downloads directory recorded.")
        printer.record({"archives": [], "imported": 0})
        return 1

    placed = 0
    records: list[dict[str, object]] = []
    for archive in archives:
        read = read_archive(archive)
        if isinstance(read, str):
            printer.say(f"{archive.name}: {read}")
            records.append({"archive": archive.name, "blocker": read})
            continue
        report = import_archive(
            oracle,
            arguments.run,
            archive,
            example_ids=_split(arguments.designs),
            chain_of_interest=arguments.chain_of_interest,
            save_id=arguments.save_id,
            every_model=arguments.every_model,
        )
        placed += len(report.placed)
        printer.say(f"{archive.name}: filed {len(report.placed)} fold(s).")
        for item in report.placed:
            label = f" [{item.save_id}]" if item.save_id else ""
            printer.say(f"  {item.example_id}{label} -> {repository_relative(item.confidence)}")
        for item in report.refused:
            printer.say(f"  refused {item.example_id}: {item.blocker}")
        for name in report.unmatched:
            printer.say(f"  matched no design: {name}")
        if report.unmatched:
            printer.say("  The server lowercases and sanitises a job name, so check the design ids.")
        records.append({"archive": archive.name, **report.as_dict(), **read.as_dict()})

    printer.record({"imported": placed, "archives": records})
    return 0 if placed else 1


def _check(arguments: argparse.Namespace, printer: Printer) -> int:
    worst = 0
    records: list[dict[str, object]] = []
    for path in arguments.file:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            printer.say(f"{path.name}: could not be read as JSON ({exc})")
            records.append({"file": path.name, "problems": ["unreadable"]})
            worst = 2
            continue
        problems = request_module.validate_payload(payload)
        if problems:
            worst = max(worst, 1)
            printer.say(f"{path.name}: {len(problems)} problem(s) the server would reject.")
            for problem in problems:
                printer.say(f"  {problem}")
        else:
            jobs = len(payload) if isinstance(payload, list) else 0
            printer.say(f"{path.name}: {jobs} job(s), accepted by the dialect.")
        records.append({"file": path.name, "problems": problems})
    printer.record({"files": records})
    return worst


def _watch(printer: Printer, seconds: int, seen: set[Path]) -> list[Path]:
    """Archives that appear in the recorded downloads directory while waiting.

    Only files that appear *after* this starts are taken. A directory already
    holding yesterday's archives would otherwise be re-imported on every call,
    and re-filing an old fold over a new one is a silent wrong answer.
    """
    ledger = session_module.load_ledger()
    if ledger.downloads is None:
        printer.say("No downloads directory recorded. Run: rfd4-af3server login --downloads <dir>")
        return []
    directory = ledger.downloads
    if not directory.is_dir():
        printer.say(f"The recorded downloads directory does not exist: {directory}")
        return []
    before = {path.resolve() for path in directory.glob(_ARCHIVE_GLOB)} | {path.resolve() for path in seen}
    printer.say(f"Watching {directory} for new {_ARCHIVE_GLOB} for up to {seconds}s. Ctrl-C to stop.")
    deadline = time.monotonic() + seconds
    found: list[Path] = []
    try:
        while time.monotonic() < deadline:
            for path in sorted(directory.glob(_ARCHIVE_GLOB)):
                resolved = path.resolve()
                if resolved in before or not _settled(path):
                    continue
                before.add(resolved)
                found.append(path)
                printer.say(f"  picked up {path.name}")
            if found:
                return found
            time.sleep(WATCH_INTERVAL)
    except KeyboardInterrupt:
        printer.say("  stopped waiting.")
    return found


def _settled(path: Path) -> bool:
    """Whether a file has stopped growing, so a partial download is not read."""
    try:
        first = path.stat().st_size
        time.sleep(0.5)
        return first > 0 and path.stat().st_size == first
    except OSError:
        return False


def _oracle(name: str) -> PathOracle:
    """The canonical paths folds are filed at, and nothing else.

    Filing a fold and asking whether one has arrived are both questions about
    where files go, and answering them needs no model, no weights and no GPU.
    This used to build a real ``AF3Oracle``, which answers them correctly and
    brings torch and a ~10 GB model stack along to do it -- for a command whose
    whole purpose is that the fold happened somewhere else.

    ``PathOracle`` restates the four path members and is kept honest by a test
    that calls ``OracleMetric``'s own unbound methods against one, so the layout
    cannot drift apart from the real oracle's quietly.
    """
    return PathOracle(oracle_name=name)


def _open_page(url: str, printer: Printer) -> None:
    """Open the submission page for the person. Never fetches it."""
    printer.say(f"  Page: {url}")
    try:
        opened = webbrowser.open(url)
    except (OSError, webbrowser.Error):
        opened = False
    if not opened:
        printer.say("  (could not open a browser here -- open that address yourself)")


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
