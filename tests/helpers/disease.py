"""Fixtures the disease tests had each been writing for themselves.

The disease tests grew one file per episode -- 66 files, 1,133 tests -- and each
new file brought its own copy of the same scaffolding. The duplication was
measurable: the ubiquitin sequence below was pasted as a module constant in
nineteen files under six different names, the MCP script loader in seven, the
output-schema check in seven, the finished-run builder in seventeen.

The tests themselves are not duplicated: 1,133 tests share only five names and
no two assert the same thing. So nothing here removes a test. What it removes
is the scaffolding each file had to carry before it could write one.

Imported directly (``from tests.helpers.disease import UBIQUITIN``) rather than
re-exported from ``tests.helpers``, so this module can grow without every test
that touches the helpers package paying for it, and so it is free to import
from the package later without a cycle.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:  # the program is imported inside the bodies that need it, never here
    from collections.abc import Callable

    from rfproteina.disease.structures import StructureOption

PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: The MCP server as a client starts it. Named here because seven test files
#: each spelled this path out.
MCP_SCRIPT = PROJECT_ROOT / "scripts" / "humanitys_last_disease_mcp.py"

#: Human ubiquitin, P0CG47 residues 1-76. The disease tests' standard designed
#: chain: long enough to hold a 9-mer window, a 19-residue hydropathy window
#: and a tryptic peptide, short enough to paste into a CIF, and a real sequence
#: rather than a run of one letter, so a measurement taken on it is a plausible
#: number rather than a degenerate one.
#:
#: Nineteen files defined this same string, under the names ``_UBIQUITIN``,
#: ``_ONE``, ``_BINDER``, ``_PROTEIN``, ``_SEQUENCE`` and ``UBIQUITIN``. A file
#: keeping its own name for it is welcome to -- ``_DECOY = UBIQUITIN`` says
#: something the bare constant does not -- but the residues live here.
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

#: One of each of the twenty amino acids, in alphabetical order by code. Used
#: where a test needs every residue to appear exactly once -- a composition
#: count, a letter table, a redaction rule that reads a run of residues as a
#: sequence.
ONE_OF_EACH = "ACDEFGHIKLMNPQRSTVWY"


def load_mcp_server(name: str = "humanitys_last_disease_mcp") -> ModuleType:
    """The MCP server script, loaded as a module without installing it.

    The script is not importable as part of the package: a client starts it by
    path, so a test reads it the same way. ``module_from_spec`` never registers
    it in ``sys.modules``, which is why ``name`` is inert and why seven copies
    of this function could disagree about it without any test noticing.
    """
    spec = importlib.util.spec_from_file_location(name, MCP_SCRIPT)
    assert spec is not None and spec.loader is not None, f"{MCP_SCRIPT} is not loadable as a module"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tool_schemas() -> dict[str, object]:
    """Each tool's declared output schema, by tool name."""
    return {str(tool["name"]): tool.get("outputSchema") for tool in load_mcp_server()._TOOLS}


def validated(tool: str, result: object) -> object:
    """A tool's result as a client receives it, checked against the tool's own schema.

    Round-tripped through ``json.dumps`` with ``allow_nan=False`` first, because
    a non-finite float is not JSON and a client would never see this result at
    all; and every schema error is reported rather than the first, because one
    undeclared field makes a validating client discard the whole answer and the
    useful question is which fields those are.

    Seven files checked a schema; six did it with an open-coded
    ``jsonschema.validate`` that stopped at the first error and let a NaN
    through. This is the seventh, which did it properly.
    """
    import jsonschema

    declared = tool_schemas()[tool]
    wire = json.loads(json.dumps(result, allow_nan=False))
    errors = [
        f"{list(error.absolute_path)}: {error.message}"
        for error in jsonschema.Draft202012Validator(declared).iter_errors(wire)
    ]
    assert not errors, errors
    return wire


# --- what a designed chain is made of ----------------------------------------

#: A second, unrelated 76-residue chain: the target in a two-chain run, where
#: ``UBIQUITIN`` is the binder. Three files pasted this same string, twice as
#: ``_TARGET`` and once as ``_TWO``.
TARGET_CHAIN = "SEEDWPWQRGQQQPTDDWGSHYRQWPDGRTYLEHHMTKLFCNAVVRGEYYMKNPQLLWSTAGGGKVVNTTPDCFLE"

#: One deposited structure as the structures page lists it.
ROW_P00533 = {
    "group": "protein",
    "accession": "P00533",
    "entry_id": "1EGFR",
    "chain": "A",
    "title": "egfr",
    "predicted": False,
    "source": "fixture",
}


# --- keeping the developer's own machine out of a test -----------------------
#
# Autouse and imported rather than defined per file. A test that reads the
# operator's configured proteome passes here and fails for everyone else, which
# is the one failure a test must not have. Import with ``# noqa: F401``: an
# autouse fixture applies to the module that imports it.


@pytest.fixture(autouse=True)
def no_developer_files(monkeypatch) -> None:
    """Neither table a developer may have configured."""
    monkeypatch.delenv("HUMAN_PROTEOME", raising=False)
    monkeypatch.delenv("HUMAN_TRANSCRIPTOME", raising=False)


@pytest.fixture(autouse=True)
def no_developer_proteome(monkeypatch) -> None:
    """A proteome on this machine must not decide what a screen measured.

    ``tests/conftest.py`` loads ``.env.local`` with ``override=True`` before
    collection, so a ``HUMAN_PROTEOME`` line there would reach every
    ``screen_generation`` call in the importing file and turn an unconfigured
    axis into a measured one, making the result depend on the machine rather
    than the fixture.

    Six files guarded themselves this way, under four names. Three of the six
    were invisible to a duplicate scan that hashes bodies: one carried this
    docstring, one annotated ``monkeypatch``, one wrote its two ``delenv``
    calls in the other order.
    """
    monkeypatch.delenv("HUMAN_PROTEOME", raising=False)


# --- the files a run leaves behind -------------------------------------------


def binder_cif(sequence: str) -> str:
    """One polymer chain under a ``data_binder`` block, as the screen reads it."""
    return (
        "data_binder\n#\nloop_\n_entity_poly.entity_id\n"
        "_entity_poly.pdbx_strand_id\n_entity_poly.pdbx_seq_one_letter_code_can\n"
        f"1 A {sequence}\n#\n"
    )


def design_cif(sequence: str, bonds: str = "") -> str:
    """The same chain under ``data_design``, optionally followed by bond rows.

    Kept separate from ``binder_cif`` rather than parameterised: the loop is
    byte-identical and only the data block name differs, which a test is
    entitled to read.
    """
    text = (
        "data_design\n#\nloop_\n"
        "_entity_poly.entity_id\n"
        "_entity_poly.pdbx_strand_id\n"
        "_entity_poly.pdbx_seq_one_letter_code_can\n"
        f"1 A {sequence}\n#\n"
    )
    if bonds:
        text += bonds
    return text


def new_folder(tmp_path: Path, name: str) -> Path:
    """A fresh directory under the test's own path."""
    folder = tmp_path / name
    folder.mkdir()
    return folder


def finished_run(root: Path, name: str = "gene_1ABC_1", *, sequence: str = UBIQUITIN) -> Path:
    """A finished run: one design and a metrics row naming its chain."""
    run = root / name
    sample = run / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    (sample / "binder_0_generated.cif").write_text(design_cif(sequence), encoding="utf-8")
    metrics = run / "inference_metrics"
    metrics.mkdir(parents=True)
    (metrics / "per_example_metrics.csv").write_text(
        "example_id,chain_of_interest\nbinder_0,A\n",
        encoding="utf-8",
    )
    return run


def job_file(run: Path, *, kind: str = "protein", target: str = "P05113", inside: bool = True) -> Path:
    """The job record a run carries, written inside the run or beside it."""
    from rfproteina.disease.screen import base_name

    payload = {"binder_kind": kind, "target_id": target, "generation_started": True}
    path = (run / f"{base_name(run)}.job.json") if inside else run.parent / f"{run.name}.job.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- screening one, and reading the page it prints ---------------------------
#
# The program is imported inside each body. This module is imported by a third
# of the disease tests, several of them only for ``UBIQUITIN``, and importing
# ``rfproteina.disease`` at module scope here costs each of them 1.5s.


def screen_protein_run(output: Path) -> dict:
    """Screen a finished run as a protein binder against the standard target."""
    from rfproteina.disease.screen import screen_generation

    return screen_generation(output, binder_kind="protein", target_id="P05113")


def screened_design_run(
    root: Path,
    sequence: str,
    *,
    kind: str = "protein",
    chain: str = "A",
    bonds: str = "",
) -> dict[str, object]:
    """Build a one-design run and screen it, returning the verdict record."""
    from rfproteina.disease.screen import screen_generation

    run = root / "gene_1ABC_1" if root.name != "gene_1ABC_1" else root
    sample = run / "inference_samples" / "binder_0"
    sample.mkdir(parents=True)
    (sample / "binder_0_generated.cif").write_text(design_cif(sequence, bonds), encoding="utf-8")
    metrics = run / "inference_metrics"
    metrics.mkdir(parents=True)
    (metrics / "per_example_metrics.csv").write_text(
        f"example_id,chain_of_interest\nbinder_0,{chain}\n",
        encoding="utf-8",
    )
    return screen_generation(run, binder_kind=kind, target_id="P05113")


def closing_page(record: dict) -> str:
    """The closing page for one screened entry, with no spoken lines."""
    from rfproteina.disease.screen_board import render_closing

    return render_closing([], [{"entry_id": "1ABC", "chain": "1", "screen": record}])


def screen_page(record: dict[str, object]) -> str:
    """The closing page for one screened entry, with its spoken lines above it."""
    from rfproteina.disease.screen import format_screen_lines
    from rfproteina.disease.screen_board import render_closing

    return render_closing(
        format_screen_lines("1ABC", "A", record),
        [{"entry_id": "1ABC", "chain": "A", "screen": record}],
    )


def structure_option(**overrides: object) -> StructureOption:
    """One listed structure, overriding whichever fields a test cares about."""
    from rfproteina.disease.structures import StructureOption

    values: dict[str, object] = {
        "group": "protein",
        "accession": "P35225",
        "entry_id": "1ABC",
        "chain": "1",
        "title": "interleukin",
        "predicted": False,
        "source": "rcsb",
    }
    values.update(overrides)
    return StructureOption(**values)  # type: ignore[arg-type]


# --- standing in for what is outside the test --------------------------------


def block_live_calls(monkeypatch) -> None:
    """Make any network call or subprocess a failure rather than a slow test."""

    def fail_open(*_args, **_kwargs):
        raise AssertionError("a live call was made")

    monkeypatch.setattr("rfproteina.disease.sources.urlopen", fail_open)
    monkeypatch.setattr("rfproteina.disease.research.urlopen", fail_open)
    monkeypatch.setattr("subprocess.run", fail_open)
    monkeypatch.setattr("subprocess.Popen", fail_open)


def url_reader(answers: dict[str, object]) -> Callable[[str], object]:
    """Answer by URL fragment. Anything unplanned is a failure, not a blank."""

    def read(url: str) -> object:
        for fragment, answer in answers.items():
            if fragment in url:
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise AssertionError(f"unplanned lookup: {url}")

    return read


def refuse_prompt(_prompt: str) -> str:
    """A prompt reader for a run that must never reach one."""
    raise AssertionError("prompt")


def asthma_catalog(directory: Path) -> Path:
    """The one-disease catalog the research tests start from."""
    path = directory / "catalog.json"
    path.write_text('[{"name": "Asthma", "disease_id": "EFO_0000270"}]', encoding="utf-8")
    return path


#: A deposit naming three accessions across three chains, for the tests that ask
#: which chain a target is on.
DEPOSIT_CIF = """\
data_deposit
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 9
C 2
E 3
#
loop_
_struct_ref.id
_struct_ref.db_name
_struct_ref.db_code
_struct_ref.pdbx_db_accession
_struct_ref.pdbx_db_isoform
_struct_ref.entity_id
_struct_ref.pdbx_seq_one_letter_code
_struct_ref.pdbx_align_begin
1 UNP RECEPTOR_HUMAN Q01344 ? 3
;QQQQ
;
21
2 UNP TARGET_HUMAN P05113 ? 2
;PPPP
;
20
3 UNP DECOY_HUMAN P99999 ? 9
;AAAA
;
1
#
loop_
_struct_ref_seq.align_id
_struct_ref_seq.ref_id
_struct_ref_seq.pdbx_strand_id
_struct_ref_seq.pdbx_db_accession
1 2 C P05113
2 1 F Q01344
3 3 A P99999
#
"""


def condition_spec(input_name: str, *chains: str, backbone: tuple[str, ...] = ()) -> str:
    """A binder spec conditioning on the named chains, and optionally a backbone."""
    lines = ["binder:", f'  input: "{input_name}"', "  conditions:"]
    if backbone:
        lines.extend(["    C_CRDGRP:", "      1:"])
        lines.extend(f"        - select: \"sel('{name}/*/*/[N,CA,C,O]') and not has_nan_coord()\"" for name in backbone)
    lines.extend(["    C_SEQ:", "      true:"])
    lines.extend(f"        - select: \"sel('{chain}')\"" for chain in chains)
    lines.append("")
    return "\n".join(lines)


# --- driving the MCP server's own read loop ----------------------------------


class Gated:
    """Hands frames to the read loop in order, each held until its gate opens.

    A BytesIO gives the loop everything at once, so it races ahead of the worker
    and a cancellation always arrives before the call it names has started --
    which is a real case, but not the one that matters for a long run. Gating a
    frame on an event the handler sets puts the cancellation where a client's
    would actually land: inside the call.
    """

    def __init__(self, frames):
        self._frames = list(frames)

    def readline(self):
        if not self._frames:
            return b""
        payload, gate = self._frames.pop(0)
        if gate is not None:
            assert gate.wait(10), "the handler never started"
        return payload

    def read(self, _size=-1):
        return b""


def exchange_with(server, messages, handler):
    """Drive one loaded server module, for a test that also reaches into it."""
    raw = b"".join(json.dumps(message).encode("utf-8") + b"\n" for message in messages)
    outgoing = io.BytesIO()
    assert server.serve(stdin=io.BytesIO(raw), stdout=outgoing, handler=handler) == 0
    lines = [json.loads(line) for line in outgoing.getvalue().splitlines() if line.strip()]
    return outgoing.getvalue(), lines


def exchange_gated(server, frames, handler):
    """Drive a server with each frame held until the handler opens its gate."""
    outgoing = io.BytesIO()
    incoming = Gated([(json.dumps(message).encode("utf-8") + b"\n", gate) for message, gate in frames])
    assert server.serve(stdin=incoming, stdout=outgoing, handler=handler) == 0
    return [json.loads(line) for line in outgoing.getvalue().splitlines() if line.strip()]


def ok_handler(_name, _arguments):
    """A tool that answers, for a test whose subject is the loop around it."""
    return {"program": "Humanitys-Last-Disease", "answered": True}
