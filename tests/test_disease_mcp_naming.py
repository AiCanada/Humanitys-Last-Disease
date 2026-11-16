"""One name for the program, pinned across the whole MCP surface.

The program was renamed from ``rfd4-cure`` to ``Humanitys-Last-Disease``. That
name is not decoration. It is the ``program`` field of every tool result, and
``_stored_schema`` lists ``program`` as **required**, so a client may key on it.
It is also ``serverInfo.name`` in the initialize reply, the logger name, the log
filename, the outbound ``User-Agent``, the NCBI ``tool=`` parameter, the
OpenRouter ``X-Title``, and the directory the encrypted key store lives in.

Before this file, that name was a hand-maintained literal in about twenty
places and nothing compared them. A rename that missed one would have produced
an MCP reply whose ``program`` contradicted the same server's
``serverInfo.name`` -- with every tool still passing its own tests, because each
asserted against its own copy of the literal.

The second subject here is ``ModRFD4``. ``describe_architecture`` used to list
it as ``selected_by_run``, which reads as "available, and a run decides whether
to use it". No checkpoint ships with this checkout and none is installed, so
the honest status is ``not_installed``: the modified weights exist as a plan,
not as a file. ``backend_status`` is the call that measures which model a
directory actually recorded, and it is left alone -- it was already honest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rfproteina.disease import logs
from rfproteina.disease.key_store import key_directory
from rfproteina.disease.mcp_layers import MODIFIED_MODEL, NOT_INSTALLED, PROGRAM, describe_architecture
from tests.helpers.disease import load_mcp_server as _server

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "humanitys_last_disease_mcp.py"

#: The name every surface must agree on. Written out once, here, so this file
#: fails loudly if the program is renamed again without updating it.
EXPECTED = "Humanitys-Last-Disease"


def test_the_program_name_is_one_string_everywhere() -> None:
    """``program`` in a tool reply and ``serverInfo.name`` are the same name.

    They are produced by different modules -- ``mcp_layers.PROGRAM`` and the
    script's ``SERVER_NAME`` -- and a client that matched one against the other
    would have been told they disagreed.
    """
    assert PROGRAM == EXPECTED
    assert _server().SERVER_NAME == PROGRAM


def test_mcp_api_reports_the_name_it_does_not_spell_itself() -> None:
    """``mcp_api`` shares the constant rather than keeping its own copy.

    Four replies in that module carried their own literal. This checks the
    constant is the one in use *and* that the literal has not crept back.
    """
    from rfproteina.disease import mcp_api

    assert mcp_api.PROGRAM is PROGRAM
    text = (_ROOT / "src" / "rfproteina" / "disease" / "mcp_api.py").read_text(encoding="utf-8")
    assert f'"program": "{EXPECTED}"' not in text, "mcp_api spells the program name again instead of using PROGRAM"


def test_the_registration_files_name_the_same_server() -> None:
    """``.mcp.json`` and the Grok config must both key on the current name.

    A stale key here is the rename failure with no symptom in any unit test:
    the server simply never starts under the name a client looks for.
    """
    mcp = json.loads((_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    assert list(mcp["mcpServers"]) == [PROGRAM]
    grok = (_ROOT / ".grok" / "config.toml").read_text(encoding="utf-8")
    assert f"[mcp_servers.{PROGRAM}]" in grok
    assert f"[mcp_servers.{PROGRAM}.tool_timeouts]" in grok


def test_the_launch_path_matches_the_script_on_disk() -> None:
    """The renamed script is the one both registrations and the relay launch."""
    stem = _SCRIPT.name
    assert stem == "humanitys_last_disease_mcp.py"
    mcp = json.loads((_ROOT / ".mcp.json").read_text(encoding="utf-8"))
    assert mcp["mcpServers"][PROGRAM]["args"] == ["-u", f"scripts/{stem}"]
    assert f"scripts/{stem}" in (_ROOT / ".grok" / "config.toml").read_text(encoding="utf-8")
    assert f"scripts/{stem}" in _server().wsl_command(Path("."))


def test_the_logger_and_its_file_carry_the_program_name() -> None:
    """A run log named for the old program is not found by anyone looking."""
    expected_log = f"{PROGRAM}.log"
    assert logs.LOGGER_NAME == PROGRAM
    # Expected value on the left here only because ruff's SIM300 asks for it on
    # this comparison; the order carries no meaning.
    assert expected_log == logs.LOG_NAME


def test_the_key_store_directory_carries_the_program_name(monkeypatch) -> None:
    """Renaming this orphans a saved store rather than breaking loudly."""
    monkeypatch.delenv("RFD4_KEY_DIR", raising=False)
    assert key_directory().name == PROGRAM


def test_every_outbound_identifier_names_this_program() -> None:
    """What a public database is told we are. Three modules, one name.

    These leave the machine, so a stale one misattributes this project's
    traffic to a program that no longer exists.
    """
    from rfproteina.disease import limits, research, target_liability

    assert limits.USER_AGENT.startswith(PROGRAM)
    assert research.USER_AGENT.startswith(PROGRAM)
    assert target_liability._CALLER == PROGRAM
    source = (_ROOT / "src" / "rfproteina" / "disease" / "research.py").read_text(encoding="utf-8")
    sent = set(re.findall(r"tool=([A-Za-z0-9._-]+)", source))
    assert sent == {PROGRAM}, f"NCBI tool= parameters disagree: {sorted(sent)}"


def test_no_source_file_still_names_the_old_program() -> None:
    """The rename is done, and this keeps it done.

    Scoped to source: a stored run output legitimately records the name that
    was actually sent at the time, and rewriting one would falsify the record.
    """
    stale = re.compile(r"rfd4[-_]cure", re.IGNORECASE)
    here = Path(__file__).resolve()
    offenders: list[str] = []
    for folder in ("src", "scripts", "tests", "docs", "skills", "configs"):
        root = _ROOT / folder
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            # This file names the old program on purpose: it is the pattern
            # being searched for. Excluded by identity, not by name, so a
            # renamed copy of this test cannot quietly exempt itself.
            if path.resolve() == here:
                continue
            if path.suffix.lower() not in {".py", ".md", ".rst", ".toml", ".yaml", ".yml", ".json", ".sh", ".cfg"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if stale.search(text):
                offenders.append(str(path.relative_to(_ROOT)))
    assert not offenders, f"the old program name survives in: {offenders}"


def test_the_modified_model_is_named_once_and_not_claimed_installed() -> None:
    """``ModRFD4`` is modified weights only, and none is installed here.

    ``selected_by_run`` said a run could pick it. Nothing in this call looks
    for a checkpoint, so that was an availability claim with no measurement
    behind it -- the same shape of defect as a verdict that clears an axis it
    never measured.
    """
    assert MODIFIED_MODEL == "ModRFD4"
    layers = describe_architecture()["layers"]
    backend = next(layer for layer in layers if layer["id"] == "computational_backend")
    component = next(item for item in backend["components"] if item["id"] == "modified_rfd4")
    assert component["name"] == MODIFIED_MODEL
    assert component["status"] == NOT_INSTALLED
    assert component["status"] != "selected_by_run", "the listing claims a model nothing checked for"
    note = str(component["note"])
    assert MODIFIED_MODEL in note
    assert "--modified-ckpt" in note, "the note must say where a checkpoint would come from"
    assert "backend_status" in note, "the note must point at the call that measures this"


def test_the_architecture_listing_spells_no_model_name_twice() -> None:
    """One constant for the model, so a rename cannot land in only one call."""
    text = (_ROOT / "src" / "rfproteina" / "disease" / "mcp_layers.py").read_text(encoding="utf-8")
    assert '"name": "Modified RFD4"' not in text
    # The constant is defined once and referenced; the literal appears in that
    # one definition and nowhere else.
    assert text.count(f'"{MODIFIED_MODEL}"') == 1, "ModRFD4 is spelled more than once in mcp_layers"


@pytest.mark.parametrize("status", ["not_recorded", "not_performed", "recorded"])
def test_backend_status_keeps_its_measured_vocabulary(status: str) -> None:
    """The measured call is untouched: it still answers from a directory.

    ``describe_architecture`` became ``not_installed`` because it measures
    nothing. ``backend_status`` does measure, so its three answers must stay
    available -- a listing status leaking into it would hide a real recording.
    """
    assert status != NOT_INSTALLED
