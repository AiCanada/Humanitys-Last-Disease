"""The ``rfd4-af3server`` verbs, exercised without a network or an oracle.

The commands tested here are the ones that run before and after the upload a
person does by hand: recording a session, building a request, validating a file,
and reporting the quota plan. ``retrieve`` is covered by
``tests/test_af3server_bundle.py`` against synthetic archives, because it needs
a real oracle's paths.

The property worth guarding at this level is that a refusal is **said**, not
swallowed. A `prepare` that silently wrote four jobs for five designs, or a
`submit` that quietly ran the first thirty of forty, would read afterwards as a
complete campaign.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfproteina.af3server.cli import main
from rfproteina.af3server.session import DAILY_JOB_LIMIT, load_ledger, record_submission

SEQUENCE = "SILDSLLEKYQFPEDLTLYIKSVFPATDSFSRFKDMVLYYVESKYSSS"


@pytest.fixture
def store(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("RFD4_KEY_DIR", str(tmp_path / "keys"))
    return tmp_path / "keys"


def design_cif(sequence: str = SEQUENCE, strand: str = "A") -> str:
    """A single-entity design, written flat the way the repository's writer does."""
    return (
        "data_design\n"
        "#\n"
        "_entity.id                 0\n"
        "_entity.type               polymer\n"
        "#\n"
        "_entity_poly.entity_id                      0\n"
        "_entity_poly.type                           polypeptide(L)\n"
        f"_entity_poly.pdbx_seq_one_letter_code_can   {sequence}\n"
        f"_entity_poly.pdbx_strand_id                 {strand}\n"
        "#\n"
    )


def run_dir(tmp_path: Path, designs: dict[str, str] | None = None) -> Path:
    """A run directory holding generated designs where the screen looks for them."""
    out = tmp_path / "run"
    for name, text in (designs or {"binder_0": design_cif()}).items():
        directory = out / "inference_samples" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}_generated.cif").write_text(text, encoding="utf-8")
    return out


def record(capsys) -> dict:
    """The JSON record a ``--json`` invocation printed."""
    return json.loads(capsys.readouterr().out)


class TestSession:
    def test_login_records_without_sending_anything(self, store: Path, tmp_path: Path, capsys) -> None:
        code = main(["login", "--downloads", str(tmp_path / "dl"), "--accept-terms"])
        assert code == 0
        assert "no credential was stored" in capsys.readouterr().out
        ledger = load_ledger()
        assert ledger.terms_acknowledged is True
        assert ledger.downloads == tmp_path / "dl"

    def test_login_refuses_something_that_looks_like_a_credential(self, store: Path, capsys) -> None:
        """The one field in the whole tool somebody might paste a token into."""
        code = main(["login", "--account", "ya29.a0AfB_byC3nK9xQ4821zzTokenLookingValue"])
        assert code == 2
        assert "never needs a password" in capsys.readouterr().out
        assert load_ledger().account is None

    def test_status_says_the_estimate_is_this_machines_own(self, store: Path, capsys) -> None:
        record_submission(4)
        assert main(["status"]) == 0
        printed = capsys.readouterr().out
        assert f"Submitted today:       4 of {DAILY_JOB_LIMIT}" in printed
        assert "The server's own tally is the one that decides." in printed

    def test_status_as_json_leaves_out_the_account_label(self, store: Path, capsys) -> None:
        main(["login", "--account", "someone@example.org"])
        capsys.readouterr()
        main(["--json", "status"])
        body = record(capsys)
        assert body["signed_in_recorded"] is True
        assert "someone@example.org" not in json.dumps(body)

    def test_logout_says_whether_there_was_anything_to_forget(self, store: Path, capsys) -> None:
        assert main(["logout"]) == 0
        assert "nothing recorded" in capsys.readouterr().out
        main(["login", "--account", "lab"])
        capsys.readouterr()
        main(["logout"])
        printed = capsys.readouterr().out
        assert "Forgot the recorded session." in printed
        assert "Your browser session is untouched" in printed


class TestPrepare:
    def test_a_run_of_designs_becomes_an_upload_file(self, store: Path, tmp_path: Path, capsys) -> None:
        out = run_dir(tmp_path)
        code = main(["prepare", "--run", str(out), "--to", str(tmp_path / "jobs")])
        assert code == 0
        written = sorted((tmp_path / "jobs").glob("*.json"))
        assert [path.name for path in written] == ["af3server_jobs.json"]
        payload = json.loads(written[0].read_text(encoding="utf-8"))
        assert isinstance(payload, list) and payload[0]["name"] == "binder_0"
        assert "1 job(s) built" in capsys.readouterr().out

    def test_a_design_that_cannot_be_sent_is_named(self, store: Path, tmp_path: Path, capsys) -> None:
        """Named, with the reason. A silently shorter batch is the real failure."""
        out = run_dir(
            tmp_path,
            {"binder_0": design_cif(), "binder_1": design_cif(SEQUENCE[:-1] + "X")},
        )
        code = main(["prepare", "--run", str(out), "--to", str(tmp_path / "jobs")])
        assert code == 0
        printed = capsys.readouterr().out
        assert "1 design(s) could not be sent" in printed
        assert "binder_1: sequence_not_standard_amino_acids" in printed
        assert "answers a different question" in printed

    def test_a_run_with_nothing_sendable_exits_nonzero_and_writes_nothing(
        self, store: Path, tmp_path: Path, capsys
    ) -> None:
        out = run_dir(tmp_path, {"binder_0": design_cif(SEQUENCE[:-1] + "X")})
        destination = tmp_path / "jobs"
        assert main(["prepare", "--run", str(out), "--to", str(destination)]) == 1
        assert not destination.exists()
        assert "Nothing to upload" in capsys.readouterr().out

    def test_the_designed_chain_gets_no_template_by_default(self, store: Path, tmp_path: Path, capsys) -> None:
        """A template for the design under test can answer for it."""
        out = run_dir(tmp_path)
        main(["prepare", "--run", str(out), "--to", str(tmp_path / "jobs")])
        payload = json.loads((tmp_path / "jobs" / "af3server_jobs.json").read_text(encoding="utf-8"))
        chain = payload[0]["sequences"][0]["proteinChain"]
        # No chain of interest is recorded in this bare run, so every chain is
        # context and keeps the server default.
        assert "useStructureTemplate" not in chain

    def test_the_template_flags_exist_on_both_prepare_and_submit(self, store: Path, tmp_path: Path) -> None:
        """The file you validate has to be the file you upload.

        A flag that shaped the request on one verb and not the other would make
        ``prepare`` and ``submit`` write different bytes for the same run.
        """
        from rfproteina.af3server.cli import build_parser

        parser = build_parser()
        actions = {
            command: {option for action in sub._actions for option in action.option_strings}
            for command, sub in parser._subparsers._group_actions[0].choices.items()
        }
        for flag in ("--design-templates", "--no-context-templates", "--per-file", "--designs", "--to"):
            assert flag in actions["prepare"], flag
            assert flag in actions["submit"], flag

    def test_only_the_named_designs_are_built(self, store: Path, tmp_path: Path, capsys) -> None:
        out = run_dir(tmp_path, {"binder_0": design_cif(), "binder_1": design_cif()})
        main(["--json", "prepare", "--run", str(out), "--to", str(tmp_path / "jobs"), "--designs", "binder_1"])
        body = record(capsys)
        assert body["jobs"] == 1

    def test_a_named_design_that_is_not_in_the_run_is_named(self, store: Path, tmp_path: Path, capsys) -> None:
        """Asking for a design that is not there must not pass silently."""
        out = run_dir(tmp_path)
        main(["--json", "prepare", "--run", str(out), "--to", str(tmp_path / "jobs"), "--designs", "binder_9"])
        body = record(capsys)
        assert body["refused"] == [{"example_id": "binder_9", "blocker": "design_not_in_run"}]


class TestSubmit:
    def test_submit_prints_the_plan_and_records_the_jobs(self, store: Path, tmp_path: Path, capsys) -> None:
        out = run_dir(tmp_path)
        code = main(["submit", "--run", str(out), "--to", str(tmp_path / "jobs"), "--no-open"])
        assert code == 0
        printed = capsys.readouterr().out
        assert "Now do the upload yourself -- there is no submission API" in printed
        assert "Upload JSON button" in printed
        assert load_ledger().today() == 1

    def test_no_record_leaves_the_ledger_alone(self, store: Path, tmp_path: Path, capsys) -> None:
        out = run_dir(tmp_path)
        main(["submit", "--run", str(out), "--to", str(tmp_path / "jobs"), "--no-open", "--no-record"])
        assert load_ledger().today() == 0

    def test_a_batch_over_the_daily_quota_is_split_and_the_overflow_named(
        self, store: Path, tmp_path: Path, capsys
    ) -> None:
        """The quota is spent at submission, and the deferred jobs are said out loud."""
        record_submission(DAILY_JOB_LIMIT - 1)
        out = run_dir(tmp_path, {f"binder_{index}": design_cif() for index in range(3)})
        main(["--json", "submit", "--run", str(out), "--to", str(tmp_path / "jobs"), "--no-open"])
        body = record(capsys)
        assert body["plan"] == {
            "today": 1,
            "deferred": 2,
            "remaining_estimate": 1,
            "blocker": "over_daily_quota",
        }
        # Every job written is counted, not only the one the plan says fits: the
        # files are handed over whole, and which drafts get run is not ours to
        # know. The plan is the warning; the ledger is what was given out.
        assert body["recorded"] == 3

    def test_unacknowledged_terms_are_mentioned_before_the_upload(self, store: Path, tmp_path: Path, capsys) -> None:
        out = run_dir(tmp_path)
        main(["submit", "--run", str(out), "--to", str(tmp_path / "jobs"), "--no-open"])
        assert "non-commercial use only" in capsys.readouterr().out

    def test_acknowledged_terms_are_not_nagged_about(self, store: Path, tmp_path: Path, capsys) -> None:
        main(["login", "--accept-terms"])
        capsys.readouterr()
        out = run_dir(tmp_path)
        main(["submit", "--run", str(out), "--to", str(tmp_path / "jobs"), "--no-open"])
        assert "non-commercial use only" not in capsys.readouterr().out


class TestCheck:
    def test_a_good_file_is_accepted(self, store: Path, tmp_path: Path, capsys) -> None:
        out = run_dir(tmp_path)
        main(["prepare", "--run", str(out), "--to", str(tmp_path / "jobs")])
        capsys.readouterr()
        code = main(["check", str(tmp_path / "jobs" / "af3server_jobs.json")])
        assert code == 0
        assert "accepted by the dialect" in capsys.readouterr().out

    def test_every_problem_is_listed_not_only_the_first(self, store: Path, tmp_path: Path, capsys) -> None:
        """One fix should be one round trip, not one per problem."""
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps([{"name": "has space", "modelSeeds": "nope", "sequences": [], "dialect": "wrong"}]),
            encoding="utf-8",
        )
        assert main(["check", str(path)]) == 1
        printed = capsys.readouterr().out
        assert "problem(s) the server would reject" in printed
        assert printed.count("\n  ") >= 4

    def test_a_bare_object_root_is_refused_as_the_wrong_dialect(self, store: Path, tmp_path: Path, capsys) -> None:
        """A list root is how the upstream loader tells this dialect from the native one."""
        path = tmp_path / "native.json"
        path.write_text(json.dumps({"name": "x", "sequences": []}), encoding="utf-8")
        assert main(["check", str(path)]) == 1
        assert "the root must be a list of jobs" in capsys.readouterr().out

    def test_an_unreadable_file_exits_two_rather_than_claiming_it_passed(
        self, store: Path, tmp_path: Path, capsys
    ) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        assert main(["check", str(path)]) == 2
        assert "could not be read as JSON" in capsys.readouterr().out
