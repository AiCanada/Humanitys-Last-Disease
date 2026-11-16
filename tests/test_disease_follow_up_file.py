"""The follow-up plan written beside a run has to be runnable as written.

``follow_up.json`` exists so the next commands survive the terminal scrollback.
It was written through ``scrubbed_json``, whose scrubber is built for free text
and replaces a personal prefix -- the checkout root among them -- wherever it
appears in a string. A command line is not free text: the replacement landed in
the middle of ``out_dir=...`` and left ``<home>/disease_runs/gene_8TLD_2``,
which is not a path. The same commands printed to the terminal were never
scrubbed and were correct, so the two copies disagreed and the stored one -- the
only copy that outlives the session -- was the one that could not be run.

Both halves are pinned here: the stored command still carries no user name and
no checkout path, and it is still a command.
"""

import json
import shlex
import tempfile
from pathlib import Path

from rfproteina.disease.cli import _portable_command, _portable_token
from rfproteina.disease.publish import scrubbed_json

_ROOT = Path(__file__).resolve().parent.parent


def _plan(*paths: Path) -> list[str]:
    """One planned command, shaped as ``follow_up._command`` shapes it."""
    spec, fresh = paths
    return [
        "rfd4-generate",
        "inference_experiment=ppi_binder",
        f"inputs={spec.as_posix()}",
        f"out_dir={fresh.as_posix()}",
        "+disease.host=gpu-01",
        "recycle_strategy=medium/front_loaded",
        "num_replicates=8",
        "batch_size=1",
    ]


def test_a_stored_command_under_the_checkout_is_still_a_command() -> None:
    """The reported defect: the one copy that outlives the session was broken."""
    command = _plan(_ROOT / "disease_runs" / "gene.yaml", _ROOT / "disease_runs" / "gene_follow_up")
    stored = _portable_command(command)

    # Written down, then read back the way a user would.
    text = scrubbed_json({"version": 1, "commands": [stored]})
    read_back = json.loads(text)["commands"][0]
    tokens = shlex.split(read_back)

    assert tokens[0] == "rfd4-generate"
    inputs = next(token for token in tokens if token.startswith("inputs="))
    out_dir = next(token for token in tokens if token.startswith("out_dir="))
    # Relative to the repository root, which is where the model is run from, and
    # not the ``<home>`` placeholder that cannot be resolved to anything.
    assert inputs == "inputs=disease_runs/gene.yaml"
    assert out_dir == "out_dir=disease_runs/gene_follow_up"
    assert "<home>" not in read_back
    assert _ROOT / inputs.split("=", 1)[1] == _ROOT / "disease_runs" / "gene.yaml"


def test_a_stored_command_carries_no_user_name_or_checkout_path() -> None:
    """The scrubbing this replaced was there for a reason, and it still holds."""
    command = _plan(_ROOT / "disease_runs" / "gene.yaml", _ROOT / "disease_runs" / "out")
    text = scrubbed_json({"commands": [_portable_command(command)]})
    assert str(_ROOT) not in text
    assert _ROOT.as_posix() not in text
    assert _ROOT.name not in text


def test_a_path_outside_the_checkout_keeps_everything_after_its_prefix() -> None:
    """Two runs on a shared mount still read as two runs, with no name on them."""
    outside = _portable_token("out_dir=/home/someone/runs/gene_8TLD_2")
    assert outside.startswith("out_dir=")
    assert "someone" not in outside
    # The components after the prefix are what tell one run from another.
    assert outside.endswith("/runs/gene_8TLD_2")


def test_a_token_that_is_not_a_path_is_left_exactly_as_it_was() -> None:
    """Nothing that is not a path is reshaped into one."""
    for token in (
        "rfd4-generate",
        "num_replicates=8",
        "batch_size=1",
        "+disease.host=gpu-01",
        "generation=inverse_folding",
        "inference_experiment=ppi_binder",
        # Holds a separator and is not a path. Resolved against the working
        # directory this came back as an absolute path to somewhere that does
        # not exist, which is why only absolute values are rewritten.
        "recycle_strategy=medium/front_loaded",
        "recycle_strategy=medium/flat",
    ):
        assert _portable_token(token) == token


def test_a_foreign_spelled_absolute_path_is_left_exactly_as_it_was() -> None:
    r"""A value this platform cannot resolve must not be joined onto the cwd.

    ``_absolute_path`` used to accept both platforms' spellings, on the thought
    that a plan written under WSL may be read on Windows. But the token is
    rewritten when the plan is written, so only this machine's spelling can
    occur -- and ``_portable_path`` resolves what it is handed, so a
    ``C:\Users\name\runs`` value under WSL was joined onto the working
    directory. The stored token then differed from every directory the program
    happened to be run in, and was not a path in any of them.
    """
    windows = r"inputs=C:\Users\someone\runs\gene.yaml"
    posix = "inputs=/mnt/c/Users/someone/runs/gene.yaml"
    here = _portable_token(windows)
    # Whichever platform this runs on, one of the two spellings is foreign and
    # is returned untouched rather than resolved into something invented.
    foreign = windows if here == windows else posix
    assert _portable_token(foreign) == foreign

    # And the scrubber still takes the account name out of what is written.
    text = scrubbed_json({"commands": [_portable_command(["rfd4-generate", foreign])]})
    assert "someone" not in text


def test_the_stored_command_does_not_depend_on_the_working_directory() -> None:
    """The plan is read later from anywhere, so it cannot encode where it was written."""
    import os

    command = _plan(_ROOT / "disease_runs" / "gene.yaml", _ROOT / "disease_runs" / "out")
    start = Path.cwd()
    seen = set()
    try:
        for where in (_ROOT, _ROOT / "tests", Path(tempfile.gettempdir())):
            os.chdir(where)
            seen.add(_portable_command(command))
    finally:
        os.chdir(start)
    assert len(seen) == 1, f"the stored plan changed with the working directory: {seen}"
